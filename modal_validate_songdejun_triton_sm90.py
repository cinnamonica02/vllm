# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch (non-interactive) Modal runner testing whether triton-lang/triton#10281
(songdejun's tensormap predication fix, still unmerged) changes the
hoisted-TD throughput picture on H200 (sm_90/Hopper) and B200 (sm_100/
Blackwell datacenter), run in parallel.

Coverage picture before this script:
  - sm_120 (RTX PRO 6000 Blackwell): BabyDrangoner already tested #10281
    thoroughly there (see PR comments) - not repeated here.
  - sm_90 (H100/H200): only tested against stock Triton 3.7.1 so far, not
    against #10281 itself.
  - sm_100 (B200): only tested against stock Triton (the PR's own existing
    numbers) - never against a newer Triton or #10281 at all.

This fills both remaining gaps in one run, in parallel (two separate Modal
containers, each on its own GPU type - no reason to pay for two ~30-60min
Triton-from-source builds sequentially when they don't depend on each
other).

Deliberately does NOT attempt the "loop-local TD + allocator-only control"
comparison BabyDrangoner ran on sm_120 - that requires hand-reconstructing
a variant that reverts the hoist but keeps the allocator registration (not
a clean revert of each other, the allocator call was added in the same
commit as the hoist), and BabyDrangoner already settled that qualitative
question (loop-local stays slower than hoisted); very unlikely to flip on
sm_90 or sm_100 specifically enough to justify the extra build risk.

Uses the current PR head (cinnamonica02/unified-attn-td-hoist-descriptor)
throughout on both GPUs - the code path being tested is exactly what's up
for review, not the pre-hoist commit.

songdejun's branch is ~516 commits behind triton-lang/triton main, so
this merges it onto current main before building, same as the earlier
(failed) B200 attempt. That run failed on a missing git identity (the
merge creates a real commit, which git refuses without one); fixed here.

Each GPU's run: correctness sanity check, then throughput for raw-pointer
(TD off) and hoisted TD (TD on), balanced workload (1024/256), matching
the model/config already used elsewhere in this PR's testing for direct
comparability against the existing numbers.

Usage:
    modal run modal_validate_songdejun_triton_sm90.py
"""

import os
import re
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

import modal

REPO_URL = "https://github.com/cinnamonica02/vllm.git"
BRANCH = "cinnamonica02/unified-attn-td-hoist-descriptor"

TRITON_REPO_URL = "https://github.com/triton-lang/triton.git"
SONGDEJUN_FORK_URL = "https://github.com/songdejun/triton.git"
SONGDEJUN_BRANCH = "fix-tensormap-predication"

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
    modal.Image.from_registry(
        "nvidia/cuda:13.0.3-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install(
        "git", "curl", "ca-certificates", "build-essential", "cmake", "ninja-build"
    )
    .pip_install("uv")
)

app = modal.App("vllm-songdejun-triton-multi-gpu-validate", image=image)


def step(msg: str, t0: float, tag: str) -> None:
    elapsed = time.monotonic() - t0
    print(f"\n[{tag} +{elapsed:6.0f}s] === {msg} ===", flush=True)


def extract_throughput(log_path: str) -> float:
    text = Path(log_path).read_text()
    m = re.search(r"Throughput:\s*([\d.]+)\s*requests?/s", text)
    if not m:
        raise RuntimeError(f"Could not find throughput line in {log_path}")
    return float(m.group(1))


def _run_validation(tag: str) -> tuple[bool, dict[str, str]]:
    t0 = time.monotonic()

    # Fresh container has no git identity; the merge below creates a real
    # commit (not a fast-forward), which git refuses without one. Throwaway
    # identity, never pushed anywhere.
    subprocess.run(
        ["git", "config", "--global", "user.email", "modal-ci@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "config", "--global", "user.name", "Modal CI"], check=True
    )

    step("Cloning vLLM PR head", t0, tag)
    subprocess.run(
        ["git", "clone", "--branch", BRANCH, "--single-branch", "--depth", "1",
         REPO_URL, "repo"],
        check=True,
    )

    step("Creating venv", t0, tag)
    subprocess.run(["uv", "venv", "--python", "3.12"], cwd="repo", check=True)
    # Absolute paths throughout - "triton" is a sibling of "repo", not a
    # subdirectory, so uv's cwd-relative venv auto-discovery would silently
    # miss repo/.venv once invoked from cwd="triton".
    venv_abs = str((Path.cwd() / "repo" / ".venv").resolve())
    venv_python = str(Path(venv_abs) / "bin" / "python")
    venv_vllm = str(Path(venv_abs) / "bin" / "vllm")
    uv_env = {**os.environ, "VIRTUAL_ENV": venv_abs}

    step("Installing vLLM (precompiled)", t0, tag)
    subprocess.run(
        ["uv", "pip", "install", "-e", ".", "--torch-backend=auto"],
        cwd="repo",
        env={**uv_env, "VLLM_USE_PRECOMPILED": "1"},
        check=True,
    )

    step("Installing test requirements", t0, tag)
    subprocess.run(
        ["uv", "pip", "install", "-r", "requirements/test/cuda.in"],
        cwd="repo",
        env=uv_env,
        check=True,
    )

    step("Cloning Triton + fetching songdejun's branch", t0, tag)
    subprocess.run(["git", "clone", TRITON_REPO_URL, "triton"], check=True)
    subprocess.run(
        ["git", "remote", "add", "songdejun", SONGDEJUN_FORK_URL],
        cwd="triton",
        check=True,
    )
    subprocess.run(
        ["git", "fetch", "songdejun", SONGDEJUN_BRANCH], cwd="triton", check=True
    )
    subprocess.run(
        ["git", "checkout", "-b", "test-build", f"songdejun/{SONGDEJUN_BRANCH}"],
        cwd="triton",
        check=True,
    )

    step("Merging songdejun's branch onto current Triton main", t0, tag)
    subprocess.run(
        ["git", "merge", "origin/main", "--no-edit"], cwd="triton", check=True
    )

    step("Building Triton from source (LLVM build - this is the long part)",
         t0, tag)
    subprocess.run(
        ["uv", "pip", "install", "-e", ".", "-v"],
        cwd="triton",
        env=uv_env,
        check=True,
    )

    step("Verifying which Triton actually got installed", t0, tag)
    subprocess.run(
        [venv_python, "-c",
         "import triton, subprocess; "
         "print('triton module path:', triton.__file__); "
         "print('triton version:', triton.__version__); "
         "print('HEAD in that checkout:', subprocess.run("
         "['git', 'rev-parse', 'HEAD'], cwd='triton', "
         "capture_output=True, text=True).stdout.strip())"],
        check=True,
    )

    step("Correctness sanity check", t0, tag)
    subprocess.run(
        f"{venv_python} -m pytest "
        f"tests/kernels/attention/test_triton_unified_attention.py "
        f"-k use_td -x -v 2>&1 | tee /tmp/{tag}_correctness_use_td.log",
        shell=True,
        cwd="repo",
        check=True,
    )

    # 3 interleaved reps (off/on/off/on/off/on, not off,off,off then
    # on,on,on) so a systematic drift over the run (thermal, noisy
    # neighbor) doesn't bias one arm more than the other - matching
    # BabyDrangoner's methodology on this axis. Each rep is a fresh
    # `vllm bench throughput` subprocess, so KV cache/engine state is
    # naturally fresh per cell without needing anything extra for that.
    off_throughputs: list[float] = []
    on_throughputs: list[float] = []
    for rep in range(1, 4):
        step(f"Throughput rep {rep}/3: balanced, TD off (raw pointer)",
             t0, tag)
        log_path = f"/tmp/{tag}_bench_balanced_td_off_rep{rep}.log"
        subprocess.run(
            f"{venv_vllm} bench throughput "
            + " ".join(COMMON_BENCH_ARGS)
            + f" 2>&1 | tee {log_path}",
            shell=True,
            cwd="repo",
            env={**uv_env, "VLLM_TRITON_USE_TD": "0"},
            check=True,
        )
        off_throughputs.append(extract_throughput(log_path))

        step(f"Throughput rep {rep}/3: balanced, TD on (hoisted, "
             f"songdejun's Triton)", t0, tag)
        log_path = f"/tmp/{tag}_bench_balanced_td_on_rep{rep}.log"
        subprocess.run(
            f"{venv_vllm} bench throughput "
            + " ".join(COMMON_BENCH_ARGS)
            + f" 2>&1 | tee {log_path}",
            shell=True,
            cwd="repo",
            env={**uv_env, "VLLM_TRITON_USE_TD": "1"},
            check=True,
        )
        on_throughputs.append(extract_throughput(log_path))

    step("Done", t0, tag)

    summary = (
        f"TD off (raw ptr): median={statistics.median(off_throughputs):.2f} "
        f"[{min(off_throughputs):.2f}, {max(off_throughputs):.2f}] req/s\n"
        f"TD on (hoisted):  median={statistics.median(on_throughputs):.2f} "
        f"[{min(on_throughputs):.2f}, {max(on_throughputs):.2f}] req/s\n"
        f"All reps off: {off_throughputs}\n"
        f"All reps on:  {on_throughputs}\n"
    )

    logs = {"summary": summary}
    path = Path(f"/tmp/{tag}_correctness_use_td.log")
    if path.exists():
        logs["correctness_use_td"] = path.read_text()
    for rep in range(1, 4):
        for state in ("off", "on"):
            path = Path(f"/tmp/{tag}_bench_balanced_td_{state}_rep{rep}.log")
            if path.exists():
                logs[f"bench_balanced_td_{state}_rep{rep}"] = path.read_text()

    return True, logs


@app.function(gpu="H200", timeout=7200)
def run_h200() -> tuple[bool, dict[str, str]]:
    return _run_validation("h200")


@app.function(gpu="B200", timeout=7200)
def run_b200() -> tuple[bool, dict[str, str]]:
    return _run_validation("b200")


@app.local_entrypoint()
def main():
    print("Kicking off H200 and B200 runs in parallel...")
    h200_call = run_h200.spawn()
    b200_call = run_b200.spawn()

    results = {"h200": h200_call.get(), "b200": b200_call.get()}

    print("\n" + "=" * 60)
    print("SUMMARY (median [min, max] over 3 interleaved reps)")
    print("=" * 60)
    for gpu_tag, (success, logs) in results.items():
        print(f"\n--- {gpu_tag} ---")
        print(logs.get("summary", "(no summary - run may have failed)"))

    out_dir = Path("run_logs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    for gpu_tag, (success, logs) in results.items():
        for name, content in logs.items():
            print(f"\n=== {gpu_tag}/{name}.log ===")
            print(content)
            (out_dir / f"{gpu_tag}_{name}.log").write_text(content)

    print(f"\nLogs saved to {out_dir}/")
