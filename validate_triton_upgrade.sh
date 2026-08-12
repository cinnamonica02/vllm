#!/usr/bin/env bash
# oonyshch asked us to test against a newer Triton than the 3.6.0 most
# testing in this thread used, given triton-lang/triton#10281 (predication
# fix for TensormapCreateOp, still unmerged/contested) is relevant context.
# Scoped down: correctness + one throughput workload (not the full sweep),
# since this is a smoke test for "does a newer Triton change the picture,"
# not a full re-validation - re-run the full suite only if something here
# looks different from what's already confirmed on 3.6.0.
set -euo pipefail

echo "=== GPU check ==="
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
major=${cap%%.*}
if [ "$major" -lt 9 ]; then
    echo "FAIL: compute capability $cap < 9.0 - need Hopper+ (H100/H200/B200)."
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
echo "=== Triton version before upgrade ==="
python -c "import triton; print(triton.__version__)"

echo ""
echo "=== Upgrading Triton to latest available ==="
uv pip install -U triton -q

echo ""
echo "=== Triton version after upgrade ==="
python -c "import triton; print(triton.__version__)" | tee /tmp/triton_version.log

echo ""
echo "=== Correctness: USE_TD path on the newer Triton ==="
python -m pytest tests/kernels/attention/test_triton_unified_attention.py \
    -k "use_td" -x -v 2>&1 | tee /tmp/correctness_use_td_newer_triton.log

if grep -q "FAILED\|Error" /tmp/correctness_use_td_newer_triton.log; then
    echo "FAIL: correctness did not pass cleanly on the newer Triton"
    exit 1
fi
echo "PASS: USE_TD path compiled and ran on the newer Triton"

MODEL="Qwen/Qwen3-4B"
COMMON=(--model "$MODEL" --dtype bfloat16
        --attention-config '{"backend":"TRITON_ATTN"}'
        --max-model-len 4096 --gpu-memory-utilization 0.85 --seed 42)

echo ""
echo "=== Throughput: balanced (in=1024 out=256) - TD OFF, newer Triton ==="
VLLM_TRITON_USE_TD=0 vllm bench throughput "${COMMON[@]}" \
    --dataset-name random --random-input-len 1024 --random-output-len 256 \
    --num-prompts 200 2>&1 | tee /tmp/bench_balanced_td_off_newer_triton.log

echo ""
echo "=== Throughput: balanced (in=1024 out=256) - TD ON, newer Triton ==="
VLLM_TRITON_USE_TD=1 vllm bench throughput "${COMMON[@]}" \
    --dataset-name random --random-input-len 1024 --random-output-len 256 \
    --num-prompts 200 2>&1 | tee /tmp/bench_balanced_td_on_newer_triton.log

echo ""
echo "=== Done ==="
echo "Logs: /tmp/triton_version.log /tmp/correctness_use_td_newer_triton.log"
echo "      /tmp/bench_balanced_td_{off,on}_newer_triton.log"
