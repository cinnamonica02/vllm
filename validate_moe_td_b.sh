#!/usr/bin/env bash
# Correctness + kernel-benchmark + e2e validation for the fused_moe_kernel
# weights-only TD (USE_TD_B) prototype. Run this on a RunPod H100/H200/B200
# pod (see RUNPOD_CUDA_DEVEL_TEMPLATE.md), from the vllm repo root. Aborts
# on first failure (set -e). Same order validate_fp8_td.sh followed for
# Kernel A (correctness, then benchmark, then e2e last).
#
# Self-contained log collection: regardless of where the script fails (or
# whether it completes), every log that was written before that point gets
# copied into a timestamped run_logs/ dir and summarized on exit. Ported
# from modal_validate_moe_td_b.py's local_entrypoint, which did this on the
# Modal-client side - moved here because RunPod has no equivalent client-side
# step, everything happens on the pod itself.
set -uo pipefail

LOG_NAMES=(
    moe_td_b_correctness
    moe_td_b_benchmark
    e2e_td_off_sharegpt
    e2e_td_off_prefill_heavy
    e2e_td_off_decode_heavy
    e2e_td_on_sharegpt
    e2e_td_on_prefill_heavy
    e2e_td_on_decode_heavy
    server_td_off
    server_td_on
)

collect_logs() {
    rc=$?
    trap - EXIT
    set +e

    out_dir="run_logs/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$out_dir"

    echo ""
    echo "=== Log summary ==="
    for name in "${LOG_NAMES[@]}"; do
        src="/tmp/${name}.log"
        if [ -f "$src" ]; then
            cp "$src" "$out_dir/${name}.log"
            echo "  [present] ${name}.log"
        else
            echo "  [missing] ${name}.log"
        fi
    done
    echo ""
    echo "Logs saved to ${out_dir}/"

    if [ "$rc" -eq 0 ]; then
        echo "RESULT: PASS"
    else
        echo "RESULT: FAIL (exit $rc)"
    fi
    exit "$rc"
}
trap collect_logs EXIT
set -e

echo "=== GPU check ==="
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
major=${cap%%.*}
if [ "$major" -lt 9 ]; then
    echo "FAIL: compute capability $cap < 9.0 - need Hopper+ (H100/H200/B200)." \
         "tl.make_tensor_descriptor needs TMA, wrong GPU tier was allocated."
    exit 1
fi

echo ""
echo "=== Environment setup (precompiled install, pinned commit) ==="
if [ ! -d .venv ]; then
    uv venv --python 3.12
fi
source .venv/bin/activate
# vLLM's release-v2 Buildkite pipeline stopped publishing x86_64 nightly
# wheels for cu130 for ~4 days (2026-07-29 to 2026-07-31, confirmed via
# direct Buildkite build-status checks - a genuine upstream CI regression,
# not a local/gating issue). Fixed upstream as of 2026-07-31. Pinning to
# this specific known-good commit rather than auto-detecting HEAD: it's
# only 43 commits ahead of this branch's own base, none touching
# fused_moe.py or the correctness test, so drift risk is minimal, and it
# avoids depending on the very latest commit's wheel having finished
# building yet (release-v2 builds take ~35 min each).
# VLLM_VERSION_OVERRIDE skips setuptools_scm's git-based version detection
# entirely (it short-circuits to SETUPTOOLS_SCM_PRETEND_VERSION instead of
# shelling out to git). Needed on top of the shallow (--depth 1) clone: on a
# shallow clone, `git status --porcelain --untracked-files=no` has been seen
# to time out after 40s on Modal's container filesystem, failing the whole
# build before it even reaches the actual install. We pin the wheel commit
# below anyway, so the exact version string here is arbitrary.
VLLM_USE_PRECOMPILED=1 VLLM_USE_PRECOMPILED_RUST=1 \
    VLLM_PRECOMPILED_WHEEL_COMMIT=10e6b400150c8d2cbedad54260def4871d464667 \
    VLLM_VERSION_OVERRIDE=0.0.0+tdbvalidate \
    uv pip install -e . --torch-backend=auto 2>&1 | tee /tmp/vllm_build.log
uv pip install -r requirements/test/cuda.in -q

echo ""
echo "=== Correctness: fused_moe_kernel USE_TD_B (weights-only TD) vs plain ==="
python -m pytest tests/kernels/moe/test_triton_moe_ptpc_fp8.py \
    -k "td_b_matches_plain" -x -v 2>&1 | tee /tmp/moe_td_b_correctness.log

echo ""
echo "=== Benchmark: TD off (plain) vs TD on (USE_TD_B), real model shapes ==="
python -m benchmarks.kernels.benchmark_fused_moe_td_b \
    2>&1 | tee /tmp/moe_td_b_benchmark.log

echo ""
echo "=== E2E: TD off vs TD auto-gated (M>=1024), mirroring afierka-intel's #42436 methodology ==="
# Same three-dataset design and reporting convention as #42436's own
# multi-platform validation (sharegpt / prefill_heavy / decode_heavy,
# N=3 reps, seed=42, --no-enable-prefix-caching). prefill_heavy/decode_heavy
# aren't real --dataset-name values - afierka's own labels for --dataset-name
# random with different input/output length ratios. Prompt counts for those
# two weren't stated in #42436; using 3x each dataset's concurrency for a
# stable measurement (afierka only gave sharegpt's count explicitly, 500).
E2E_MODEL="gaunernst/DeepSeek-V2-Lite-Chat-FP8"
SHAREGPT_PATH="/tmp/ShareGPT_V3_unfiltered_cleaned_split.json"
if [ ! -f "$SHAREGPT_PATH" ]; then
    curl -fsSL -o "$SHAREGPT_PATH" \
        "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
fi

wait_for_server() {
    for i in $(seq 1 400); do
        if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health \
                2>/dev/null | grep -q 200; then
            echo "Server ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "FAIL: server did not become ready within 400s"
    return 1
}

run_e2e_sweep() {
    label="$1"
    echo ""
    echo "--- E2E serve: $label ---"
    vllm serve "$E2E_MODEL" \
        --linear-backend triton --port 8000 --no-enable-prefix-caching \
        > "/tmp/server_${label}.log" 2>&1 &
    server_pid=$!
    wait_for_server

    for rep in 1 2 3; do
        vllm bench serve --model "$E2E_MODEL" --host 127.0.0.1 --port 8000 \
            --dataset-name sharegpt --dataset-path "$SHAREGPT_PATH" \
            --num-prompts 500 --request-rate inf \
            --max-concurrency 64 --seed 42 \
            2>&1 | tee -a "/tmp/e2e_${label}_sharegpt.log"
        vllm bench serve --model "$E2E_MODEL" --host 127.0.0.1 --port 8000 \
            --dataset-name random --random-input-len 2048 --random-output-len 512 \
            --num-prompts 96 --request-rate inf --max-concurrency 32 --seed 42 \
            2>&1 | tee -a "/tmp/e2e_${label}_prefill_heavy.log"
        vllm bench serve --model "$E2E_MODEL" --host 127.0.0.1 --port 8000 \
            --dataset-name random --random-input-len 512 --random-output-len 2048 \
            --num-prompts 24 --request-rate inf --max-concurrency 8 --seed 42 \
            2>&1 | tee -a "/tmp/e2e_${label}_decode_heavy.log"
    done

    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    sleep 2
}

VLLM_TRITON_USE_TD_B=0 run_e2e_sweep td_off
# Deliberately left unset here, not =1: unset falls through to
# resolve_moe_use_td_b's own M>=1024 auto-gate, which is what actually ships.
# Forcing =1 would turn TD on for every forward pass regardless of batch
# size, including the small-M decode steps where the kernel benchmark shows
# a regression - conflating that into the e2e number instead of measuring
# the real, gated behavior.
run_e2e_sweep td_on

echo ""
echo "=== Done ==="
echo "Logs: /tmp/moe_td_b_correctness.log  /tmp/moe_td_b_benchmark.log"
echo "E2E logs: /tmp/e2e_{td_off,td_on}_{sharegpt,prefill_heavy,decode_heavy}.log"
