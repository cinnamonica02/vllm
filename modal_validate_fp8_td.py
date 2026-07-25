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

import modal

REPO_URL = "https://github.com/cinnamonica02/vllm.git"
BRANCH = "cinnamonica02/w8a8-fp8-block-scaled-td"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.3-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "curl", "ca-certificates", "build-essential")
    .pip_install("uv")
)

app = modal.App("vllm-fp8-td-validate", image=image)


@app.function(gpu="H100", timeout=3600)
def run_validation():
    subprocess.run(
        ["git", "clone", "--branch", BRANCH, "--single-branch", "--depth", "1",
         REPO_URL, "repo"],
        check=True,
    )
    subprocess.run(["bash", "validate_fp8_td.sh"], check=True, cwd="repo")

    print("\n=== bench_td_off.log ===")
    subprocess.run(["cat", "/tmp/bench_td_off.log"], check=True)
    print("\n=== bench_td_on.log ===")
    subprocess.run(["cat", "/tmp/bench_td_on.log"], check=True)


@app.local_entrypoint()
def main():
    run_validation.remote()
