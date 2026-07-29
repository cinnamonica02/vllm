# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TD-vs-plain microbenchmark for fused_moe_kernel's weights-only TD path
(USE_TD_B, block-quantized branch).

use_td_b has no A-gather (unlike the Blackwell/XPU USE_TD path), so unlike
Kernel A's USE_TD it isn't gated on M/K thresholds - it engages for any
quantized block-scaled MoE call on Hopper+. This benchmark exists to check
whether that's actually safe across small M (Kernel A needed an M>=64 gate
to fix a regression there) before this is PR-worthy.

Toggle via forcing current_platform.has_device_capability, matching how
test_fused_moe_block_fp8_td_b_matches_plain isolates the two code paths on
the same real GPU (there's no VLLM_TRITON_USE_TD-style env var for USE_TD_B).
"""

import torch

from benchmarks.kernels.benchmark_moe import get_model_params
from tests.kernels.moe.utils import fused_moe, make_test_quant_config
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.platforms import current_platform
from vllm.transformers_utils.config import get_config
from vllm.triton_utils import triton as vllm_triton

assert current_platform.is_cuda(), "Only support benchmarking on CUDA device."

DEFAULT_MODEL = "gaunernst/DeepSeek-V2-Lite-Chat-FP8"
BATCH_SIZES = [1, 16, 64, 128, 256, 512, 1024, 2048, 4096]
BLOCK_SHAPE = [128, 128]


def build_runner(M, N, K, E, topk, device):
    torch.manual_seed(0)
    a = torch.randn((M, K), dtype=torch.bfloat16, device=device) / 10
    w1, w2, quant_config = make_test_quant_config(
        E,
        N,
        K,
        torch.bfloat16,
        torch.float8_e4m3fn,
        block_shape=BLOCK_SHAPE,
    )
    score = torch.randn((M, E), dtype=torch.bfloat16, device=device)
    vllm_config = VllmConfig()
    original_has_device_capability = current_platform.has_device_capability

    def run(use_td_b: bool):
        # Only intercept the exact capability (90) the USE_TD_B gate queries;
        # fall through for anything else so unrelated capability-gated
        # dispatch elsewhere in the call stack isn't silently perturbed,
        # which would confound a timing comparison in a way a correctness
        # assertion wouldn't necessarily catch.
        def patched(capability, *args, **kwargs):
            if capability == 90:
                return use_td_b
            return original_has_device_capability(capability, *args, **kwargs)

        with set_current_vllm_config(vllm_config):
            current_platform.has_device_capability = patched
            return fused_moe(
                a, w1, w2, score, topk, renormalize=False, quant_config=quant_config
            )

    return run


@vllm_triton.testing.perf_report(
    vllm_triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=BATCH_SIZES,
        x_log=False,
        line_arg="provider",
        line_vals=["td_off", "td_on"],
        line_names=["TD off (plain)", "TD on (USE_TD_B)"],
        ylabel="ms (lower is better)",
        plot_name="fused_moe_kernel weights-only TD (block-fp8)",
        args={},
    )
)
def benchmark_latency(batch_size, provider, N, K, E, topk):
    M = batch_size
    run = build_runner(M, N, K, E, topk, device="cuda")
    use_td_b = provider == "td_on"
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = vllm_triton.testing.do_bench(
        lambda: run(use_td_b), quantiles=quantiles
    )
    return ms, min_ms, max_ms


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    config = get_config(model=args.model, trust_remote_code=args.trust_remote_code)
    E, topk, N, K = get_model_params(config)

    original_has_device_capability = current_platform.has_device_capability
    try:
        print(f"\nBenchmarking {args.model}: E={E} topk={topk} N={N} K={K}")
        print(f"Latency comparison (block_shape={BLOCK_SHAPE}):")
        benchmark_latency.run(print_data=True, N=N, K=K, E=E, topk=topk)
    finally:
        current_platform.has_device_capability = original_has_device_capability

    print("\nBenchmark finished!")
