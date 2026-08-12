# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch (non-interactive) Modal runner testing whether triton-lang/triton#10281
(songdejun's tensormap predication fix, still unmerged) resolves the
pipeliner crash on B200 for the *pre-hoist* USE_TD path.

Targets vLLM commit 7e85d3a42c (the parent of this branch's own hoist fix,
74e1633150) deliberately - that's the last commit where the bug actually
reproduces. Testing #10281 against the current (already-hoisted) branch
would show nothing, since the hoist already sidesteps the issue on stock
Triton regardless of this fix landing.

songdejun's branch (fix-tensormap-predication) is ~516 commits behind
triton-lang/triton main, so this merges it onto current main before
building rather than building the stale branch directly. Confirmed via a
local dry-run merge first: the actual patch is small and self-contained
(27 lines in PipeliningUtility.cpp's predicateOp, matching the scf.if
fallback approach from the PR's review thread) and merges cleanly with
zero conflicts against current main.

Builds Triton from source (not on PyPI - #10281 is still open/unmerged as
of this writing), so this takes considerably longer than the other
validate scripts (LLVM build included). Expect ~30-60min. Every phase
prints a timestamped marker and streams its own subprocess output live
(no `-q`/silent-redirect steps) so a stall or failure is visible as it
happens, not just after the fact.

Usage:
    modal run modal_validate_songdejun_triton_fix.py
"""

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import modal

REPO_URL = "https://github.com/cinnamonica02/vllm.git"
PRE_HOIST_COMMIT = "7e85d3a42c"

TRITON_REPO_URL = "https://github.com/triton-lang/triton.git"
SONGDEJUN_FORK_URL = "https://github.com/songdejun/triton.git"
SONGDEJUN_BRANCH = "fix-tensormap-predication"

LOG_NAMES = ["correctness_use_td_songdejun_fix"]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.3-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install(
        "git", "curl", "ca-certificates", "build-essential", "cmake", "ninja-build"
    )
    .pip_install("uv")
)

app = modal.App("vllm-songdejun-triton-fix-validate", image=image)


def step(msg: str, t0: float) -> None:
    elapsed = time.monotonic() - t0
    print(f"\n[+{elapsed:6.0f}s] === {msg} ===", flush=True)


@app.function(gpu="B200", timeout=7200)
def run_validation() -> tuple[bool, dict[str, str]]:
    t0 = time.monotonic()

    step("Cloning vLLM (pre-hoist commit)", t0)
    subprocess.run(["git", "clone", REPO_URL, "repo"], check=True)
    subprocess.run(["git", "checkout", PRE_HOIST_COMMIT], cwd="repo", check=True)

    step("Creating venv", t0)
    subprocess.run(["uv", "venv", "--python", "3.12"], cwd="repo", check=True)
    venv_python = "repo/.venv/bin/python"

    step("Installing vLLM (precompiled)", t0)
    subprocess.run(
        ["uv", "pip", "install", "-e", ".", "--torch-backend=auto"],
        cwd="repo",
        env={**os.environ, "VLLM_USE_PRECOMPILED": "1"},
        check=True,
    )

    step("Installing test requirements", t0)
    subprocess.run(
        ["uv", "pip", "install", "-r", "requirements/test/cuda.in"],
        cwd="repo",
        check=True,
    )

    step("Cloning Triton + fetching songdejun's branch", t0)
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

    step("Merging songdejun's branch onto current Triton main", t0)
    # Confirmed via local dry-run: merges cleanly, no conflicts.
    subprocess.run(
        ["git", "merge", "origin/main", "--no-edit"], cwd="triton", check=True
    )

    step("Building Triton from source (LLVM build - this is the long part)", t0)
    subprocess.run(
        ["uv", "pip", "install", "-e", ".", "-v"],
        cwd="triton",
        check=True,
    )

    step("Running correctness test (streaming + logging to file)", t0)
    # shell + tee: streams live to Modal's log AND saves to the file we
    # read back below, instead of going silent for the whole test run.
    result = subprocess.run(
        f"{venv_python} -m pytest "
        f"tests/kernels/attention/test_triton_unified_attention.py "
        f"-k use_td -x -v 2>&1 | tee /tmp/correctness_use_td_songdejun_fix.log",
        shell=True,
        cwd="repo",
    )

    step(f"Done, exit code {result.returncode}", t0)

    logs = {}
    for name in LOG_NAMES:
        path = Path(f"/tmp/{name}.log")
        if path.exists():
            logs[name] = path.read_text()

    return result.returncode == 0, logs


@app.local_entrypoint()
def main():
    success, logs = run_validation.remote()

    out_dir = Path("run_logs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, content in logs.items():
        print(f"\n=== {name}.log ===")
        print(content)
        (out_dir / f"{name}.log").write_text(content)

    print(f"\nLogs saved to {out_dir}/")

    if not success:
        print("Test failed/crashed - see logs. This may itself be informative "
              "(confirms bug still present) or may be an unrelated build issue.")
