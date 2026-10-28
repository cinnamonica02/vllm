# SPDX-License-Identifier: Apache-2.0
"""Run the unified-attention TD validation on a Modal B300."""

import subprocess

import modal

REPO_URL = "https://github.com/cinnamonica02/vllm.git"
BRANCH = "cinnamonica02/unified-attn-td-hoist-descriptor"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.3-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "curl", "ca-certificates", "build-essential")
    .pip_install("uv")
)

app = modal.App("vllm-unified-attn-td-validate-b300", image=image)


@app.function(gpu="B300", timeout=3600)
def run_validation() -> int:
    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            BRANCH,
            "--single-branch",
            "--depth",
            "1",
            REPO_URL,
            "repo",
        ],
        check=True,
    )
    return subprocess.run(
        ["bash", "validate_unified_attn_td.sh"], cwd="repo"
    ).returncode


@app.local_entrypoint()
def main() -> None:
    return_code = run_validation.remote()
    if return_code:
        raise RuntimeError(
            f"validate_unified_attn_td.sh failed with exit code {return_code}"
        )
