# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch Modal runner: raw pointer vs hoisted TD on H200 (sm_90), stock Triton,
at block sizes 16 and 32, on the current head of the PR branch.

Replaces the original H100/H200 throughput table in PR #51506 (whose logs
were lost) and gives the Hopper counterpart of oonyshch's GB200 block-size-32
numbers. Follows her method: warmup prompts discarded, fresh process (and
cache) per cell, repetitions interleaved across arms, median [min, max].

Same prebuilt-image approach as modal_validate_hoisted_vs_loop_local_*.py:
never touches vLLM's build, just patches the PR head's kernel (and helper)
file into the installed package. Stock Triton only - no source build - since
that is what users get and the hoisted kernel already passes on it.

Usage:
    modal run modal_validate_h200_block_sizes.py

Set BLOCK_SIZES = [16] below to halve the cost.
"""

import os
import re
import shlex
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

import modal

REPO_URL = "https://github.com/cinnamonica02/vllm.git"
BRANCH = "cinnamonica02/attn-td-hoist-fix"
KERNEL_REL_PATH = "vllm/v1/attention/ops/triton_unified_attention.py"
HELPER_REL_PATH = "vllm/v1/attention/ops/triton_attention_helpers.py"

MODEL = "Qwen/Qwen3-4B"
BASE_BENCH_ARGS = [
    "--model", MODEL,
    "--dtype", "bfloat16",
    "--attention-config", '{"backend":"TRITON_ATTN"}',
    "--max-model-len", "4096",
    "--gpu-memory-utilization", "0.85",
    "--seed", "42",
    "--dataset-name", "random",
    "--num-prompts", "200",
]
WORKLOADS = [
    ("prefill_heavy", "2048", "128"),
    ("decode_heavy", "128", "1024"),
    ("balanced", "1024", "256"),
]
BLOCK_SIZES = [16, 32]
ARMS = ["raw", "hoisted"]
REPS = 3
NUM_WARMUPS = 20

image = (
    modal.Image.from_registry("vllm/vllm-openai:latest")
    .dockerfile_commands(["ENTRYPOINT []"])
    .apt_install("git", "curl", "ca-certificates", "python-is-python3")
    .run_commands("python3 -m pip install uv")
)

app = modal.App("vllm-h200-block-sizes", image=image)


def step(msg: str, t0: float) -> None:
    elapsed = time.monotonic() - t0
    print(f"\n[+{elapsed:6.0f}s] === {msg} ===", flush=True)


def extract_throughput(log_path: str) -> float:
    text = Path(log_path).read_text()
    m = re.search(r"Throughput:\s*([\d.]+)\s*requests?/s", text)
    if not m:
        raise RuntimeError(f"Could not find throughput line in {log_path}")
    return float(m.group(1))


def supports_num_warmups() -> bool:
    """The image's vLLM may predate `--num-warmups`; probe instead of assuming."""
    out = subprocess.run(
        ["vllm", "bench", "throughput", "--help=all"],
        capture_output=True, text=True,
    )
    return "--num-warmups" in out.stdout + out.stderr


def run_bench(
    arm: str,
    block_size: int,
    workload: tuple[str, str, str],
    rep: int,
    warmups: bool,
    t0: float,
    logs: dict[str, str],
) -> float:
    name, in_len, out_len = workload
    key = f"bench_{name}_bs{block_size}_{arm}_rep{rep}"
    log_path = f"/tmp/{key}.log"
    bench_args = BASE_BENCH_ARGS + [
        "--random-input-len", in_len,
        "--random-output-len", out_len,
        "--block-size", str(block_size),
    ]
    if warmups:
        bench_args += ["--num-warmups", str(NUM_WARMUPS)]
    bench_cmd = (
        "vllm bench throughput "
        + " ".join(shlex.quote(a) for a in bench_args)
        + f" 2>&1 | tee {log_path}"
    )
    step(f"Throughput {name} block={block_size} {arm} rep {rep}/{REPS}", t0)
    subprocess.run(
        ["bash", "-c", f"set -o pipefail && {bench_cmd}"],
        env={**os.environ, "VLLM_TRITON_USE_TD": "1" if arm == "hoisted" else "0"},
        check=True,
    )
    logs[key] = Path(log_path).read_text()
    return extract_throughput(log_path)


def run_correctness(t0: float) -> tuple[bool, str]:
    log_path = "/tmp/correctness_use_td.log"
    step("Correctness (-k use_td, covers both block sizes in the PR's matrix)", t0)
    # Not check=True: a failure is a result to record, and the benches should
    # still be skipped by the caller, not crash before logs are collected.
    result = subprocess.run(
        ["bash", "-c",
         "set -o pipefail && python3 -m pytest "
         "tests/kernels/attention/test_triton_unified_attention.py "
         f"-k use_td -q 2>&1 | tee {log_path}"],
        cwd="repo",
    )
    return result.returncode == 0, Path(log_path).read_text()


def capture_environment(t0: float) -> str:
    log_path = "/tmp/environment.log"
    step("Capturing environment metadata", t0)
    subprocess.run(
        ["bash", "-c",
         "set -o pipefail && ("
         "echo '--- nvidia-smi ---'; nvidia-smi --query-gpu=name,driver_version,"
         "memory.total --format=csv; "
         "echo '--- python versions ---'; python3 -c \""
         "import torch, triton, vllm; "
         "print('vllm:', vllm.__version__); "
         "print('torch:', torch.__version__); "
         "print('torch.version.cuda:', torch.version.cuda); "
         "print('triton:', triton.__version__); "
         "print('gpu name (torch):', torch.cuda.get_device_name(0))\""
         f") 2>&1 | tee {log_path}"],
        check=True,
    )
    return Path(log_path).read_text()


def fmt(values: list[float]) -> str:
    return f"{statistics.median(values):.2f} [{min(values):.2f}, {max(values):.2f}]"


@app.function(gpu="H200", timeout=10800)
def run_validation() -> dict[str, str]:
    t0 = time.monotonic()
    results: dict[str, str] = {}

    step("Locating the image's already-installed vLLM", t0)
    installed_vllm_dir = subprocess.run(
        ["python3", "-c", "import vllm, os; print(os.path.dirname(vllm.__file__))"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"Installed vLLM package at: {installed_vllm_dir}")

    step("Cloning the PR branch", t0)
    subprocess.run(
        ["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, "repo"],
        check=True,
    )
    pr_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd="repo", capture_output=True, text=True, check=True,
    ).stdout.strip()
    results["commit"] = f"PR head ({BRANCH}): {pr_head}\n"
    print(results["commit"])

    step("Patching the PR head's kernel + helper into the installed package", t0)
    dst_dir = Path(installed_vllm_dir) / "v1/attention/ops"
    (dst_dir / "triton_unified_attention.py").write_text(
        (Path("repo") / KERNEL_REL_PATH).read_text()
    )
    (dst_dir / "triton_attention_helpers.py").write_text(
        (Path("repo") / HELPER_REL_PATH).read_text()
    )

    step("Installing test requirements", t0)
    subprocess.run(
        ["uv", "pip", "install", "--system", "-r", "repo/requirements/test/cuda.in"],
        check=True,
    )
    # Delete the clone's vllm/ so it can't shadow-import the patched package.
    subprocess.run(["rm", "-rf", "repo/vllm"], check=True)

    results["environment"] = capture_environment(t0)

    passed, results["correctness_use_td"] = run_correctness(t0)
    if not passed:
        results["summary"] = "Correctness FAILED - throughput skipped. See log.\n"
        print(results["summary"])
        return results

    warmups = supports_num_warmups()
    results["warmups"] = (
        f"--num-warmups {NUM_WARMUPS}: {'used' if warmups else 'NOT supported by image'}\n"
    )
    print(results["warmups"])

    samples: dict[tuple[int, str, str], list[float]] = {}
    for block_size in BLOCK_SIZES:
        for workload in WORKLOADS:
            for rep in range(1, REPS + 1):
                for arm in ARMS:
                    samples.setdefault((block_size, workload[0], arm), []).append(
                        run_bench(arm, block_size, workload, rep, warmups, t0, results)
                    )

    lines = [f"H200 (sm_90), PR head {pr_head[:12]}, req/s median [min, max]\n"]
    for block_size in BLOCK_SIZES:
        lines += [
            f"\nblock size {block_size}\n",
            "| Workload | in/out | raw ptr | TD hoisted | hoisted/raw |",
            "|---|---|---|---|---|",
        ]
        for name, in_len, out_len in WORKLOADS:
            raw = samples[(block_size, name, "raw")]
            hoisted = samples[(block_size, name, "hoisted")]
            ratio = statistics.median(hoisted) / statistics.median(raw)
            lines.append(
                f"| {name} | {in_len}/{out_len} | {fmt(raw)} | {fmt(hoisted)} "
                f"| {ratio:.3f}x |"
            )
    results["summary"] = "\n".join(lines) + "\n"
    print(results["summary"])

    step("Done", t0)
    return results


@app.local_entrypoint()
def main():
    results = run_validation.remote()

    out_dir = Path("run_logs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, content in results.items():
        (out_dir / f"{name}.log").write_text(content)
    (out_dir / "results.md").write_text(
        results.get("commit", "") + results.get("environment", "")
        + results.get("warmups", "") + "\n" + results.get("summary", "")
    )

    print("\n" + "=" * 60)
    print(results.get("summary", "(missing - run may have failed)"))
    print(f"Logs + results.md saved to {out_dir}/ - copy this folder somewhere safe.")
