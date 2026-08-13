# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch (non-interactive) Modal runner comparing hoisted TD (this PR's head)
against loop-local TD (the pre-hoist implementation) on H200 (sm_90), using
stock Triton - the thing BabyDrangoner measured on SM120 that we hadn't
measured anywhere else.

Deliberately sm_90-only, not B200. On sm_90 the pre-hoist code was never
broken - it already compiles and runs with stock Triton, just at
num_stages=1 (a PR code comment: "Hopper's num_stages=1 fallback masked
this same [allocator] requirement instead of erroring"). So this needs no
Triton-from-source build and no hand-reconstructed allocator-only variant -
just two real commits (pre-hoist, PR head), same stock Triton the image
already ships. On sm_100/sm_120 the pre-hoist code doesn't compile with
stock Triton at all (the original bug), so isolating just the hoist effect
there would still need songdejun's Triton plus a hand-patched
allocator-only control - the reconstruction risk that ruled this out for
B200 stays ruled out here.

Same vllm-openai-prebuilt-image approach as the other validate scripts:
never touches vLLM's own build system, just patches the one file that
differs between pre-hoist and PR head directly into the installed package.
Pre-hoist commit is found dynamically (git merge-base HEAD upstream/main,
fetched from the real vllm-project/vllm remote, not the fork's own
possibly-stale main) rather than hardcoded, since the exact SHA has
already shifted once this session after a rebase.

Usage:
    modal run modal_validate_hoisted_vs_loop_local_sm90.py
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
UPSTREAM_URL = "https://github.com/vllm-project/vllm.git"
BRANCH = "cinnamonica02/unified-attn-td-hoist-descriptor"
KERNEL_REL_PATH = "vllm/v1/attention/ops/triton_unified_attention.py"

MODEL = "Qwen/Qwen3-4B"
COMMON_BENCH_ARGS = [
    "--model", MODEL,
    "--dtype", "bfloat16",
    "--attention-config", '{"backend":"TRITON_ATTN"}',
    "--max-model-len", "4096",
    "--gpu-memory-utilization", "0.85",
    "--seed", "42",
    "--dataset-name", "random",
    "--random-input-len", "1024",
    "--random-output-len", "256",
    "--num-prompts", "200",
]

image = (
    modal.Image.from_registry("vllm/vllm-openai:latest")
    .dockerfile_commands(["ENTRYPOINT []"])
    .apt_install(
        "git", "curl", "ca-certificates", "python-is-python3",
    )
    .run_commands("python3 -m pip install uv")
)

app = modal.App("vllm-hoisted-vs-loop-local-sm90", image=image)


def step(msg: str, t0: float) -> None:
    elapsed = time.monotonic() - t0
    print(f"\n[+{elapsed:6.0f}s] === {msg} ===", flush=True)


def extract_throughput(log_path: str) -> float:
    text = Path(log_path).read_text()
    m = re.search(r"Throughput:\s*([\d.]+)\s*requests?/s", text)
    if not m:
        raise RuntimeError(f"Could not find throughput line in {log_path}")
    return float(m.group(1))


def run_bench(variant: str, rep: int, t0: float) -> float:
    log_path = f"/tmp/bench_{variant}_rep{rep}.log"
    bench_cmd = (
        "vllm bench throughput "
        + " ".join(shlex.quote(a) for a in COMMON_BENCH_ARGS)
        + f" 2>&1 | tee {log_path}"
    )
    step(f"Throughput ({variant}) rep {rep}/3", t0)
    subprocess.run(
        ["bash", "-c", f"set -o pipefail && {bench_cmd}"],
        env={**os.environ, "VLLM_TRITON_USE_TD": "1"},
        check=True,
    )
    return extract_throughput(log_path)


def run_correctness(variant: str, t0: float) -> tuple[bool, str]:
    log_path = f"/tmp/correctness_{variant}.log"
    step(f"Correctness ({variant})", t0)
    # Not check=True: loop-local genuinely can fail to compile at all (a
    # real MLIR pipeliner failure, not a script bug - confirmed once
    # already). That's a valid result to record, not something that
    # should crash the whole run before we get to test the other variant.
    result = subprocess.run(
        ["bash", "-c",
         "set -o pipefail && python3 -m pytest "
         "tests/kernels/attention/test_triton_unified_attention.py "
         f"-k use_td -x -v 2>&1 | tee {log_path}"],
        cwd="repo",
    )
    return result.returncode == 0, Path(log_path).read_text()


@app.function(gpu="H200", timeout=3600)
def run_validation() -> dict[str, str]:
    t0 = time.monotonic()

    step("Locating the image's already-installed vLLM", t0)
    find_vllm = subprocess.run(
        ["python3", "-c", "import vllm, os; print(os.path.dirname(vllm.__file__))"],
        capture_output=True, text=True, check=True,
    )
    installed_vllm_dir = find_vllm.stdout.strip()
    print(f"Installed vLLM package at: {installed_vllm_dir}")

    step("Cloning vLLM PR branch (full history, need merge-base)", t0)
    subprocess.run(
        ["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, "repo"],
        check=True,
    )

    step("Finding the pre-hoist commit (merge-base against real upstream, "
         "not the fork's own main)", t0)
    subprocess.run(
        ["git", "remote", "add", "upstream", UPSTREAM_URL], cwd="repo", check=True
    )
    subprocess.run(["git", "fetch", "upstream", "main"], cwd="repo", check=True)
    mb = subprocess.run(
        ["git", "merge-base", "HEAD", "upstream/main"],
        cwd="repo", capture_output=True, text=True, check=True,
    )
    pre_hoist_commit = mb.stdout.strip()
    print(f"Pre-hoist commit: {pre_hoist_commit}")

    step("Extracting both kernel file versions", t0)
    hoisted_src = (Path("repo") / KERNEL_REL_PATH).read_text()
    loop_local_src = subprocess.run(
        ["git", "show", f"{pre_hoist_commit}:{KERNEL_REL_PATH}"],
        cwd="repo", capture_output=True, text=True, check=True,
    ).stdout

    dst = Path(installed_vllm_dir) / "v1/attention/ops/triton_unified_attention.py"

    step("Installing test requirements", t0)
    subprocess.run(
        ["uv", "pip", "install", "--system", "-r", "repo/requirements/test/cuda.in"],
        check=True,
    )

    # Delete the clone's local vllm/ dir now that we've pulled both file
    # versions out of it - nothing left that could shadow-import the
    # installed package instead of what we're patching in below. See the
    # other validate scripts for why this matters.
    subprocess.run(["rm", "-rf", "repo/vllm"], check=True)

    results: dict[str, str] = {}

    for variant, src in [("loop_local", loop_local_src), ("hoisted", hoisted_src)]:
        step(f"Patching in {variant} kernel", t0)
        dst.write_text(src)

        passed, correctness_log = run_correctness(variant, t0)
        results[f"correctness_{variant}"] = correctness_log

        if not passed:
            msg = (f"{variant}: correctness FAILED (compile or runtime "
                   f"error) - skipping throughput, nothing to benchmark "
                   f"if it doesn't run. See correctness_{variant}.log.\n")
            results[f"summary_{variant}"] = msg
            print(msg)
            continue

        throughputs = [run_bench(variant, rep, t0) for rep in range(1, 4)]
        summary = (
            f"{variant}: median={statistics.median(throughputs):.2f} "
            f"[{min(throughputs):.2f}, {max(throughputs):.2f}] req/s\n"
            f"All reps: {throughputs}\n"
        )
        results[f"summary_{variant}"] = summary
        print(summary)

    step("Done", t0)
    return results


@app.local_entrypoint()
def main():
    results = run_validation.remote()

    print("\n" + "=" * 60)
    print("SUMMARY (median [min, max] over 3 interleaved reps)")
    print("=" * 60)
    for variant in ("loop_local", "hoisted"):
        print(f"\n--- {variant} ---")
        print(results.get(f"summary_{variant}", "(missing - run may have failed)"))

    out_dir = Path("run_logs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, content in results.items():
        (out_dir / f"{name}.log").write_text(content)
    print(f"\nLogs saved to {out_dir}/")
