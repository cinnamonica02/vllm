#!/usr/bin/env bash
# One-shot validation for the unified-attention TD (TMA) hoisted-descriptor
# fix (oonyshch's root-cause + fix, ported into
# vllm/v1/attention/ops/triton_unified_attention.py).
# Run this inside the Modal H100/H200/B200 shell, from the vllm repo root.
# Aborts on first failure (set -e) so we don't burn GPU time benchmarking
# a kernel that failed correctness.
set -euo pipefail

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
echo "=== Environment setup (precompiled install, no C++/CUDA rebuild) ==="
if [ ! -d .venv ]; then
    uv venv --python 3.12
fi
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto -q
uv pip install -r requirements/test/cuda.in -q

echo ""
echo "=== Correctness: USE_TD path (numerical equivalence vs raw pointer) ==="
echo "Reuses the existing pilot-kernel test suite (#40327) unchanged - this"
echo "PR only changes *how* K/V tiles are loaded (descriptor hoisted out of"
echo "the loop), not what they load, so no new test was added."
python -m pytest tests/kernels/attention/test_triton_unified_attention.py \
    -k "use_td" -x -v 2>&1 | tee /tmp/correctness_use_td.log

echo ""
echo "=== Compile check: default num_stages (no workaround) ==="
echo "Pre-fix, USE_TD failed to compile at default num_stages (pipeliner"
echo "can't predicate tensormap_create) and only ran forced to num_stages=1."
echo "The correctness run above used unified_attention()'s normal launch"
echo "path with no num_stages override - if it got this far, the hoist"
echo "fixed the compile failure. Confirming explicitly:"
if grep -q "FAILED\|Error" /tmp/correctness_use_td.log; then
    echo "FAIL: correctness/compile step above did not pass cleanly"
    exit 1
fi
echo "PASS: USE_TD path compiled and ran at default num_stages"

MODEL="Qwen/Qwen3-4B"
COMMON=(--model "$MODEL" --dtype bfloat16
        --attention-config '{"backend":"TRITON_ATTN"}'
        --max-model-len 4096 --gpu-memory-utilization 0.85 --seed 42)

run_workload() {
    local name=$1 in_len=$2 out_len=$3
    echo ""
    echo "=== Throughput: $name (in=$in_len out=$out_len) - TD OFF (raw ptr, default on CUDA) ==="
    VLLM_TRITON_USE_TD=0 vllm bench throughput "${COMMON[@]}" \
        --dataset-name random --input-len "$in_len" --output-len "$out_len" \
        --num-prompts 200 2>&1 | tee "/tmp/bench_${name}_td_off.log"

    echo ""
    echo "=== Throughput: $name (in=$in_len out=$out_len) - TD ON (hoisted, default num_stages) ==="
    VLLM_TRITON_USE_TD=1 vllm bench throughput "${COMMON[@]}" \
        --dataset-name random --input-len "$in_len" --output-len "$out_len" \
        --num-prompts 200 2>&1 | tee "/tmp/bench_${name}_td_on.log"
}

# Same three workload shapes as oonyshch's H200 table, for a direct comparison.
run_workload prefill_heavy 2048 128
run_workload decode_heavy 128 1024
run_workload balanced 1024 256

echo ""
echo "=== Done ==="
echo "Logs: /tmp/correctness_use_td.log"
echo "      /tmp/bench_{prefill_heavy,decode_heavy,balanced}_td_{off,on}.log"
echo "Copy all out (e.g. via modal volume, or paste back) before exiting the shell."
