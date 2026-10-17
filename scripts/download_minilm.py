"""Download the MiniLM embedding model files and verify they load.

Who: Person 2. Where: a JupyterLab terminal inside IBM's container on the VM
(huggingface_hub, transformers, and torch are preinstalled there), from the repo root:

    python scripts/download_minilm.py

Also works on a laptop with requirements-dev.txt installed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surgedesk import config  # noqa: E402

REPO = "sentence-transformers/all-MiniLM-L6-v2"
FILES = [
    "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt",
    "special_tokens_map.json", "model.safetensors", "modules.json", "sentence_bert_config.json",
]


def main() -> None:
    from huggingface_hub import snapshot_download

    config.MINILM_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(REPO, local_dir=str(config.MINILM_DIR), allow_patterns=FILES)
    print("Downloaded to", config.MINILM_DIR)
    for f in sorted(config.MINILM_DIR.iterdir()):
        print(f"  {f.name:<28} {f.stat().st_size / 1e6:8.2f} MB")

    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(config.MINILM_DIR))
    model = AutoModel.from_pretrained(str(config.MINILM_DIR)).eval()
    with torch.inference_mode():
        enc = tok(["test sentence"], return_tensors="pt", truncation=True, max_length=32)
        out = model(**enc).last_hidden_state
    print("Load test OK, hidden size:", out.shape[-1])
    assert out.shape[-1] == config.EMBED_DIM


if __name__ == "__main__":
    main()
