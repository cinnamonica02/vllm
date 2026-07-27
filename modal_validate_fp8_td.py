# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch (non-interactive) Modal runner for validate_fp8_td.sh.

Avoids `modal shell`'s interactive-attach path, which stalls waiting for an
H100 scheduling slot on free-tier accounts. Batch functions (`modal run`)
get scheduled independently of an interactive TTY and free the GPU the
moment the script exits, instead of idling on a held-open shell.

Usage:
    modal run modal_validate_fp8_td.py
"""

import subprocess
from datetime import datetime
from pathlib import Path

import modal

REPO_URL = "https://github.com/cinnamonica02/vllm.git"
BRANCH = "cinnamonica02/w8a8-fp8-block-scaled-td"

LOG_NAMES = [
    "bench_td_off",
    "bench_td_on",
    "e2e_td_off",
    "e2e_td_on",
    "server_td_off",
    "server_td_on",
]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.3-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "curl", "ca-certificates", "build-essential")
    .pip_install("uv")
)

app = modal.App("vllm-fp8-td-validate", image=image)


@app.function(gpu="H100", timeout=3600)
def run_validation() -> tuple[bool, dict[str, str]]:
    subprocess.run(
        ["git", "clone", "--branch", BRANCH, "--single-branch", "--depth", "1",
         REPO_URL, "repo"],
        check=True,
    )
    # No check=True: a failure must still fall through to log collection
    # below, not lose everything to an exception before logs are read.
    result = subprocess.run(["bash", "validate_fp8_td.sh"], cwd="repo")

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
        raise RuntimeError("validate_fp8_td.sh failed - see logs above")
