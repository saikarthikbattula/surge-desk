"""Local MiniLM embeddings; no model downloads at runtime (Agent 1)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import config
from .contracts import ContractError, Report
from .fileio import load_embeddings, read_jsonl, save_embeddings, write_json
from .textutil import mask_privacy
from . import artifacts
from .provenance import file_hash, report_identity, tree_hash


class Embedder:
    def __init__(self, model_dir: Path = config.MINILM_DIR, threads: int = config.N_THREADS, device: str = "cpu"):
        if device != "cpu":
            raise NotImplementedError("Only CPU embeddings are supported; NNPA requires a measured device comparison")
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.threads, self.device = threads, device
        self.model_dir = Path(model_dir)
        self._torch = torch
        torch.set_num_threads(threads)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(model_dir), local_files_only=True).to(device)
        self.model.eval()

    def encode(self, texts: list[str], batch_size: int = config.EMBED_BATCH) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not texts:
            return np.empty((0, config.EMBED_DIM), dtype=np.float32)
        torch = self._torch
        batches = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                tokens = self.tokenizer([mask_privacy(t) for t in texts[start:start + batch_size]],
                                        truncation=True, max_length=config.EMBED_MAX_TOKENS,
                                        padding=True, return_tensors="pt")
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                hidden = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                batches.append(pooled.cpu().numpy().astype(np.float32))
        result = np.concatenate(batches, axis=0)
        if result.shape != (len(texts), config.EMBED_DIM):
            raise ContractError(f"Expected {config.EMBED_DIM}-dimensional embeddings, got {result.shape}")
        return result


def embedding_identity(reports, embedder=None):
    return {"reports": report_identity(reports), "model": tree_hash(getattr(embedder, "model_dir", config.MINILM_DIR)),
                "encoder": type(embedder).__qualname__ if embedder is not None else "Embedder",
                "configuration": [config.EMBED_DIM, config.EMBED_MAX_TOKENS, config.EMBED_BATCH,
                                  getattr(embedder, "threads", config.N_THREADS), getattr(embedder, "device", "cpu")],
                "implementation": [file_hash(Path(__file__)), file_hash(Path(__file__).with_name("textutil.py"))]}


def embed_event(event: str, force: bool = False, embedder: Embedder | None = None) -> np.ndarray:
    reports = read_jsonl(config.event_file(event, config.REPORTS_FILE), Report)
    target = config.event_file(event, config.EMBEDDINGS_FILE)
    manifest = config.event_file(event, config.EMBED_MANIFEST_FILE)
    identity = embedding_identity(reports, embedder)
    if not force and artifacts.valid(manifest, identity, [target]):
        try:
            cached = load_embeddings(target)
            if cached.shape == (len(reports), config.EMBED_DIM) and np.isfinite(cached).all():
                return cached
        except (OSError, ValueError, ContractError):
            pass
    embedder = embedder if embedder is not None else Embedder()
    start = time.perf_counter()
    embeddings = embedder.encode([r.text for r in reports])
    seconds = time.perf_counter() - start
    if embeddings.shape != (len(reports), config.EMBED_DIM) or not np.isfinite(embeddings).all():
        raise ContractError(f"Embedding/report shape mismatch: {embeddings.shape}, {len(reports)} reports")
    save_embeddings(target, embeddings)
    metrics = dict(n=len(reports), seconds=seconds, per_sec=len(reports) / seconds if seconds else 0.0,
                   threads=getattr(embedder, "threads", config.N_THREADS), device=getattr(embedder, "device", "cpu"))
    write_json(config.event_file(event, "embed_timing.json"), metrics)
    artifacts.complete(manifest, identity, [target])
    print(f"{event}: {metrics['per_sec']:.2f} reports/sec")
    return embeddings
