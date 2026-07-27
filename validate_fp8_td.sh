#!/usr/bin/env bash
# One-shot validation for the W8A8 block-FP8 TD kernel path.
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
echo "=== Correctness: TD-forced path (boundary shapes) ==="
python -m pytest tests/kernels/quantization/test_block_fp8.py -k "matmul_td" -x -v

echo ""
echo "=== Correctness: raw-pointer path smoke check (should be unaffected) ==="
python -m pytest tests/kernels/quantization/test_block_fp8.py \
    -k "test_w8a8_block_fp8_matmul and 4096 and 13824 and 16384" -x -v

echo ""
echo "=== Correctness passed. Benchmark: TD OFF (baseline) ==="
VLLM_TRITON_USE_TD=0 python -m benchmarks.kernels.benchmark_block_fp8_td 2>&1 | tee /tmp/bench_td_off.log

echo ""
echo "=== Benchmark: TD auto-detect (K-gated on CUDA, no override set) ==="
python -m benchmarks.kernels.benchmark_block_fp8_td 2>&1 | tee /tmp/bench_td_on.log

E2E_MODEL="gaunernst/DeepSeek-V2-Lite-Chat-FP8"

wait_for_server() {
    for i in $(seq 1 400); do
        if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health 2>/dev/null | grep -q 200; then
            echo "Server ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "FAIL: server did not become ready within 400s"
    return 1
}

echo ""
echo "=== E2E serve: TD OFF (baseline) ==="
VLLM_TRITON_USE_TD=0 vllm serve "$E2E_MODEL" \
    --linear-backend triton --port 8000 \
    > /tmp/server_td_off.log 2>&1 &
server_pid=$!
wait_for_server
vllm bench serve \
    --model "$E2E_MODEL" \
    --host 127.0.0.1 --port 8000 \
    --dataset-name random --random-input-len 128 --random-output-len 128 \
    --num-prompts 300 --request-rate inf \
    2>&1 | tee /tmp/e2e_td_off.log
kill "$server_pid" 2>/dev/null || true
wait "$server_pid" 2>/dev/null || true
sleep 2

echo ""
echo "=== E2E serve: TD auto-detect (K-gated on CUDA, no override set) ==="
vllm serve "$E2E_MODEL" \
    --linear-backend triton --port 8000 \
    > /tmp/server_td_on.log 2>&1 &
server_pid=$!
wait_for_server
vllm bench serve \
    --model "$E2E_MODEL" \
    --host 127.0.0.1 --port 8000 \
    --dataset-name random --random-input-len 128 --random-output-len 128 \
    --num-prompts 300 --request-rate inf \
    2>&1 | tee /tmp/e2e_td_on.log
kill "$server_pid" 2>/dev/null || true
wait "$server_pid" 2>/dev/null || true

echo ""
echo "=== Done ==="
echo "Logs: /tmp/bench_td_off.log  /tmp/bench_td_on.log  /tmp/e2e_td_off.log  /tmp/e2e_td_on.log"
echo "Server logs: /tmp/server_td_off.log  /tmp/server_td_on.log"
echo "Copy all out (e.g. via modal volume, or paste back) before exiting the shell."
