# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct TD-vs-raw-pointer microbenchmark for w8a8_triton_block_scaled_mm.

benchmark_block_fp8_gemm.py drives the kernel through
init_fp8_linear_kernel/apply_weights, which has drifted from the current
kernel API (missing constructor args, apply() no longer exists, kernel
auto-selection ignores use_cutlass). Rather than patch that dispatch layer
further, call the kernel wrapper directly. Toggle via VLLM_TRITON_USE_TD,
matching how validate_fp8_td.sh runs a script twice to compare TD on/off.
"""

import torch

from benchmarks.kernels.benchmark_block_fp8_gemm import DEEPSEEK_V3_SHAPES
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
    w8a8_triton_block_scaled_mm,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton as vllm_triton

assert current_platform.is_cuda(), (
    "Only support benchmarking w8a8 block fp8 kernel on CUDA device."
)

BATCH_SIZES = [1, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def build_runner(M, N, K, block_size, device):
    fp8_info = torch.finfo(current_platform.fp8_dtype())
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    block_n, block_k = block_size

    A_ref = (torch.rand(M, K, dtype=torch.bfloat16, device=device) - 0.5) * 2 * fp8_max
    A, As = per_token_group_quant_fp8(A_ref, block_k)

    B_ref = (torch.rand(N, K, dtype=torch.bfloat16, device=device) - 0.5) * 2 * fp8_max
    B = B_ref.clamp(min=fp8_min, max=fp8_max).to(current_platform.fp8_dtype())

    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k
    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32, device=device) * 1e-2

    def run():
        return w8a8_triton_block_scaled_mm(A, B, As, Bs, block_size, torch.bfloat16)

    return run


@vllm_triton.testing.perf_report(
    vllm_triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=BATCH_SIZES,
        x_log=False,
        line_arg="provider",
        line_vals=["w8a8-block-fp8-triton"],
        line_names=["w8a8-block-fp8-triton"],
        ylabel="TFLOP/s (larger is better)",
        plot_name="W8A8 Block FP8 GEMM (direct kernel call)",
        args={},
    )
)
def benchmark_tflops(batch_size, provider, N, K, block_size=(128, 128)):
    M = batch_size
    run = build_runner(M, N, K, block_size, device="cuda")
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = vllm_triton.testing.do_bench_cudagraph(
        run, quantiles=quantiles
    )
    to_tflops = lambda t_ms: (2 * M * N * K) * 1e-12 / (t_ms * 1e-3)
    return to_tflops(ms), to_tflops(max_ms), to_tflops(min_ms)


if __name__ == "__main__":
    block_size = (128, 128)

    for N, K in DEEPSEEK_V3_SHAPES:
        print(f"\nBenchmarking DeepSeek-V3, N={N} K={K}")
        print(f"TFLOP/s comparison (block_size={block_size}):")
        benchmark_tflops.run(print_data=True, N=N, K=K, block_size=block_size)

    print("\nBenchmark finished!")
