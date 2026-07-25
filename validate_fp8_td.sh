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

echo ""
echo "=== E2E: preparing trimmed DeepSeek-V3 dummy model (config+tokenizer only) ==="
python benchmarks/kernels/prepare_e2e_fp8_td_model.py /tmp/deepseek_v3_mini

echo ""
echo "=== E2E latency: TD OFF (baseline) ==="
VLLM_TRITON_USE_TD=0 vllm bench latency \
    --model /tmp/deepseek_v3_mini \
    --load-format dummy \
    --linear-backend triton \
    --input-len 128 --output-len 128 --batch-size 8 \
    2>&1 | tee /tmp/e2e_td_off.log

echo ""
echo "=== E2E latency: TD auto-detect (K-gated on CUDA, no override set) ==="
vllm bench latency \
    --model /tmp/deepseek_v3_mini \
    --load-format dummy \
    --linear-backend triton \
    --input-len 128 --output-len 128 --batch-size 8 \
    2>&1 | tee /tmp/e2e_td_on.log

echo ""
echo "=== Done ==="
echo "Logs: /tmp/bench_td_off.log  /tmp/bench_td_on.log  /tmp/e2e_td_off.log  /tmp/e2e_td_on.log"
echo "Copy all out (e.g. via modal volume, or paste back) before exiting the shell."
