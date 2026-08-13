# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch (non-interactive) Modal runner testing whether triton-lang/triton#10281
(songdejun's tensormap predication fix, still unmerged) changes the
hoisted-TD throughput picture on H200 (sm_90/Hopper) and B200 (sm_100/
Blackwell datacenter), run in parallel.

Coverage picture before this script:
  - sm_120 (RTX PRO 6000 Blackwell): BabyDrangoner already tested #10281
    thoroughly there (see PR comments) - not repeated here.
  - sm_90 / sm_100: only tested against stock Triton so far, not #10281.

REFACTORED to start from vllm/vllm-openai's prebuilt image instead of
building vLLM from source. Building vLLM from source repeatedly hit an
unrelated wall: vLLM's requirements/cuda.txt pins torch==2.13.0, but the
vendored vllm-flash-attn submodule (pinned to a fixed commit in
cmake/external_projects/vllm_flash_attn.cmake) has a stale hardcoded
"PyTorch 2.4.0 expected" check, and its C++ extensions fail to compile
against 2.13.0. That's a real inconsistency in vLLM's current from-source
build, not something in our control, and not related to Triton/#10281 at
all - confirmed deterministic (baked into a fixed GIT_TAG, not a moving
target), so retrying wasn't going to help.

BabyDrangoner's own numbers (torch 2.11.0, despite his base commit already
including the 2.13.0 bump) only make sense if he used a prebuilt image
too, rather than rebuilding vLLM's C++ extensions locally - matching the
`docker run ... vllm/vllm-openai:latest` pattern the original Triton
issue itself used to repro.

So: base image ships vLLM already compiled and working against a tested
torch. We never touch vLLM's build system at all. Only two things get
swapped on top of that known-good base:
  1. The one Python file this PR actually changes
     (vllm/v1/attention/ops/triton_unified_attention.py) - a plain file
     copy over the installed package, no recompile needed since it's
     pure Python.
  2. Triton itself, built from source (songdejun's branch merged onto
     current triton-lang/triton main - no wheel exists for that specific
     combination anywhere, source build is the only option there).

One risk this specifically guards against: after cloning the repo to get
that one file (and the test suite), the local checkout's own vllm/
directory must not be importable when running tests/benchmarks - if it
were, Python could shadow-import the uncompiled, unpatched local source
tree instead of the properly-installed (and now-patched) package in
site-packages, silently defeating the entire point. Fixed by locating the
real installed vllm path *before* ever cd-ing into the clone, then
deleting the clone's local vllm/ directory entirely right after copying
out the one file we need from it.

Usage:
    modal run modal_validate_songdejun_triton_sm90.py
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
BRANCH = "cinnamonica02/unified-attn-td-hoist-descriptor"
KERNEL_REL_PATH = "vllm/v1/attention/ops/triton_unified_attention.py"

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
    modal.Image.from_registry("vllm/vllm-openai:latest")
    # vllm-openai's image bakes in a Docker ENTRYPOINT that treats every
    # arg as a `vllm` CLI argument (it's designed to run as a turnkey
    # server container). That swallows Modal's own container bootstrap
    # command (`python -u -R ... -m modal._container_entrypoint ...`),
    # handing it to `vllm` as arguments instead of actually running
    # Python - hence "vllm: error: unrecognized arguments: -u -R ...".
    # Clearing it restores normal exec behavior for Modal's own startup.
    .dockerfile_commands(["ENTRYPOINT []"])
    .apt_install(
        "git", "curl", "ca-certificates", "build-essential", "cmake", "ninja-build",
        # Same zlib/libxml2 gap as before - Triton's prebuilt LLVM package
        # references ZLIB::ZLIB in its CMake export files.
        "zlib1g-dev", "libxml2-dev",
        # Adds the conventional /usr/bin/python -> python3 symlink. This
        # image only ships python3, which breaks more than just our own
        # `python -m pip` calls below - Modal's own SDK probes for a
        # `python` binary to determine the image's interpreter version
        # when building a Function from a custom registry image, and
        # fails with a platform-level ConflictError if it can't find one.
        # A real symlink to the existing interpreter, not a second Python
        # install - doesn't risk losing access to the image's
        # pre-installed vLLM/torch.
        "python-is-python3",
    )
    # Not .pip_install("uv"): that helper shells out to plain `python`,
    # which doesn't exist on this base image (only `python3` is on PATH),
    # and fails with "python: not found". Explicit python3 invocation
    # instead.
    .run_commands("python3 -m pip install uv")
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

    # Fresh container has no git identity; the Triton merge below creates
    # a real commit (not a fast-forward), which git refuses without one.
    # Throwaway identity, never pushed anywhere.
    subprocess.run(
        ["git", "config", "--global", "user.email", "modal-ci@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "config", "--global", "user.name", "Modal CI"], check=True
    )

    step("Locating the image's already-installed vLLM", t0, tag)
    # Deliberately run this *before* cloning our repo into the cwd, so
    # there's no local vllm/ directory anywhere nearby that could shadow
    # the real installed package for this lookup itself.
    find_vllm = subprocess.run(
        ["python3", "-c", "import vllm, os; print(os.path.dirname(vllm.__file__))"],
        capture_output=True, text=True, check=True,
    )
    installed_vllm_dir = find_vllm.stdout.strip()
    print(f"Installed vLLM package at: {installed_vllm_dir}")

    step("Cloning vLLM PR branch (just for the kernel file + tests)", t0, tag)
    subprocess.run(
        ["git", "clone", "--branch", BRANCH, "--single-branch", "--depth", "1",
         REPO_URL, "repo"],
        check=True,
    )

    step("Patching the one changed file into the installed package", t0, tag)
    src = Path("repo") / KERNEL_REL_PATH
    dst = Path(installed_vllm_dir) / "v1/attention/ops/triton_unified_attention.py"
    dst.write_text(src.read_text())
    print(f"Copied {src} -> {dst}")

    # Delete the clone's local vllm/ source tree entirely now that we've
    # taken the one file we needed from it - guarantees nothing below can
    # ever shadow-import it instead of the patched, installed package.
    subprocess.run(["rm", "-rf", "repo/vllm"], check=True)

    step("Installing minimal test deps (pytest only - avoid disturbing "
         "the base image's already-working torch/vllm-flash-attn combo "
         "by reinstalling the full test requirements file)", t0, tag)
    subprocess.run(
        ["uv", "pip", "install", "--system", "pytest", "pytest-asyncio"],
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
    # Not -e (editable): setuptools' modern PEP 660 editable-install
    # mechanism (a "meta path finder") doesn't correctly expose all of
    # Triton's nested subpackages - confirmed by a prior run where
    # `import triton` succeeded but `triton.language` raised
    # AttributeError, breaking torch/vllm's own import chain (torch._
    # dynamo touches triton.language.dtype at import time). We don't need
    # editable mode here anyway - not iterating on Triton's source live,
    # just building it once. A regular install actually materializes the
    # discovered files into site-packages instead of relying on that
    # finder.
    subprocess.run(
        ["uv", "pip", "install", "--system", ".", "-v"],
        cwd="triton",
        check=True,
    )

    step("Verifying which Triton actually got installed, and that vLLM "
         "still imports the patched kernel file", t0, tag)
    subprocess.run(
        ["python3", "-c",
         "import triton, vllm, subprocess; "
         "print('triton module path:', triton.__file__); "
         "print('triton version:', triton.__version__); "
         "print('HEAD in that checkout:', subprocess.run("
         "['git', 'rev-parse', 'HEAD'], cwd='triton', "
         "capture_output=True, text=True).stdout.strip()); "
         "print('vllm module path:', vllm.__file__)"],
        check=True,
    )

    step("Correctness sanity check", t0, tag)
    subprocess.run(
        f"python3 -m pytest "
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
            f"vllm bench throughput "
            # shlex.quote, not a plain join: shell=True means /bin/sh
            # re-parses this whole string, and would strip the double
            # quotes out of --attention-config's JSON value as its own
            # quoting syntax before vllm ever sees it (confirmed - that's
            # exactly what broke last run, '{"backend":"TRITON_ATTN"}'
            # arrived as the mangled {backend:TRITON_ATTN}).
            + " ".join(shlex.quote(a) for a in COMMON_BENCH_ARGS)
            + f" 2>&1 | tee {log_path}",
            shell=True,
            env={**os.environ, "VLLM_TRITON_USE_TD": "0"},
            check=True,
        )
        off_throughputs.append(extract_throughput(log_path))

        step(f"Throughput rep {rep}/3: balanced, TD on (hoisted, "
             f"songdejun's Triton)", t0, tag)
        log_path = f"/tmp/{tag}_bench_balanced_td_on_rep{rep}.log"
        subprocess.run(
            f"vllm bench throughput "
            # shlex.quote, not a plain join: shell=True means /bin/sh
            # re-parses this whole string, and would strip the double
            # quotes out of --attention-config's JSON value as its own
            # quoting syntax before vllm ever sees it (confirmed - that's
            # exactly what broke last run, '{"backend":"TRITON_ATTN"}'
            # arrived as the mangled {backend:TRITON_ATTN}).
            + " ".join(shlex.quote(a) for a in COMMON_BENCH_ARGS)
            + f" 2>&1 | tee {log_path}",
            shell=True,
            env={**os.environ, "VLLM_TRITON_USE_TD": "1"},
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


@app.function(gpu="H200", timeout=14400)
def run_h200() -> tuple[bool, dict[str, str]]:
    return _run_validation("h200")


@app.function(gpu="B200", timeout=14400)
def run_b200() -> tuple[bool, dict[str, str]]:
    return _run_validation("b200")


@app.local_entrypoint()
def main():
    print("Kicking off H200 and B200 runs in parallel...")
    h200_call = run_h200.spawn()
    b200_call = run_b200.spawn()

    # Each .get() in its own try/except: a plain dict literal would abort
    # entirely on the first failed .get(), losing the other GPU's result
    # even if it succeeded - independent containers, so one failing
    # shouldn't hide the other's outcome.
    results: dict[str, tuple[bool, dict[str, str]]] = {}
    for gpu_tag, call in [("h200", h200_call), ("b200", b200_call)]:
        try:
            results[gpu_tag] = call.get()
        except Exception as e:
            print(f"\n[{gpu_tag}] FAILED: {e}")
            results[gpu_tag] = (False, {})

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
