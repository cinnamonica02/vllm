# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a minimal DeepSeek-V3 model dir for e2e --load-format dummy benchmarking.

Downloads only config.json + tokenizer files (no weights) from the real
DeepSeek-V3 checkpoint, then trims num_hidden_layers to stay within
first_k_dense_replace so every remaining layer is a plain dense MLP with
the real intermediate_size=18432 - the same K we already validated as a
win in the kernel microbenchmark, not a synthetic shape and not routed
MoE (moe_intermediate_size=2048, which wouldn't exercise the TD gate).
"""

import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "deepseek-ai/DeepSeek-V3"
NUM_LAYERS = 2


def main(out_dir: str) -> None:
    local_dir = snapshot_download(
        MODEL_ID,
        allow_patterns=["config.json", "tokenizer*", "*.jinja"],
        local_dir=out_dir,
    )
    config_path = Path(local_dir) / "config.json"
    config = json.loads(config_path.read_text())

    assert NUM_LAYERS <= config["first_k_dense_replace"], (
        "trimmed layer count must stay within first_k_dense_replace so "
        "every layer is dense (real K=intermediate_size), not routed MoE"
    )
    config["num_hidden_layers"] = NUM_LAYERS
    config["num_nextn_predict_layers"] = 0
    config_path.write_text(json.dumps(config, indent=2))
    print(f"Prepared trimmed DeepSeek-V3 config at {local_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/deepseek_v3_mini")
