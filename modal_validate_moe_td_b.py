# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch (non-interactive) Modal runner for validate_moe_td_b.sh.

Correctness + kernel-benchmark + e2e check for the fused_moe_kernel
weights-only TD (USE_TD_B) prototype. Modeled on modal_validate_fp8_td.py
(the Kernel A / dense-GEMM equivalent); see that file for why `modal run`
(batch) is used instead of `modal shell` (interactive scheduling stalls on
H100 free-tier capacity).

Usage:
    modal run modal_validate_moe_td_b.py
"""

import subprocess
from datetime import datetime
from pathlib import Path

import modal

REPO_URL = "https://github.com/cinnamonica02/vllm.git"
BRANCH = "cinnamonica02/moe-w8a8-block-scaled-td"

LOG_NAMES = [
    "moe_td_b_correctness",
    "moe_td_b_benchmark",
    "e2e_td_off_sharegpt",
    "e2e_td_off_prefill_heavy",
    "e2e_td_off_decode_heavy",
    "e2e_td_on_sharegpt",
    "e2e_td_on_prefill_heavy",
    "e2e_td_on_decode_heavy",
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

app = modal.App("vllm-moe-td-b-validate", image=image)


@app.function(gpu="H100", timeout=5400)
def run_validation() -> tuple[bool, dict[str, str]]:
    subprocess.run(
        ["git", "clone", "--branch", BRANCH, "--single-branch", "--depth", "1",
         REPO_URL, "repo"],
        check=True,
    )
    # No check=True: a failure must still fall through to log collection
    # below, not lose everything to an exception before logs are read.
    result = subprocess.run(["bash", "validate_moe_td_b.sh"], cwd="repo")

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
        raise RuntimeError("validate_moe_td_b.sh failed - see logs above")
