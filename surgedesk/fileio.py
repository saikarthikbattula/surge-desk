"""File I/O for handoff files. Always use these so contracts are validated.

Owner: foundation (frozen).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, TypeVar

import numpy as np

from .contracts import RECORD_TYPES, ContractError, _Record

T = TypeVar("T", bound=_Record)


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _to_dict(row) -> dict:
    if isinstance(row, _Record):
        row.validate()
        return row.to_dict()
    if isinstance(row, dict):
        return row
    raise TypeError(f"Cannot serialize {type(row).__name__}; pass a contract record or dict")


def write_jsonl(path: Path | str, rows: Iterable) -> int:
    """Atomically write records (validated) or dicts as JSON Lines. Returns the row count."""
    lines = [json.dumps(_to_dict(r), ensure_ascii=False) for r in rows]
    _atomic_write_text(Path(path), "\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def append_jsonl(path: Path | str, row) -> None:
    """Append one record (validated) or dict. Used for decisions.jsonl and llm_calls.jsonl."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_to_dict(row), ensure_ascii=False) + "\n")


def read_jsonl(path: Path | str, cls: type[T] | None = None) -> list:
    """Read JSON Lines. With cls, every line is parsed and validated into that record type."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run the stage that produces it first.")
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if cls is None:
                out.append(d)
            else:
                try:
                    out.append(cls.from_dict(d))
                except ContractError as e:
                    raise ContractError(f"{path}:{i}: {e}") from e
    return out


def validate_file(path: Path | str) -> int:
    """Validate a handoff file by its name (reports/tagged/incidents/briefings/decisions.jsonl)."""
    path = Path(path)
    cls = RECORD_TYPES.get(path.name)
    if cls is None:
        raise ValueError(f"No contract registered for {path.name}")
    return len(read_jsonl(path, cls))


def write_json(path: Path | str, obj) -> None:
    _atomic_write_text(Path(path), json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")


def read_json(path: Path | str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_embeddings(path: Path | str, emb: np.ndarray) -> None:
    """Save float32 [n, 384] L2-normalized embeddings atomically."""
    path = Path(path)
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim != 2:
        raise ContractError(f"embeddings must be 2-D, got shape {emb.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, emb)
    os.replace(tmp, path)


def load_embeddings(path: Path | str) -> np.ndarray:
    emb = np.load(Path(path))
    if emb.dtype != np.float32 or emb.ndim != 2:
        raise ContractError(f"{path}: expected float32 2-D array, got {emb.dtype} {emb.shape}")
    return emb
