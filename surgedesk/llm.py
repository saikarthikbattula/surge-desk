"""Local Granite adapter for Stage 4 briefings.

The model is loaded lazily and shared by callers in this process.
"""
from __future__ import annotations

import gc
import importlib
import json
import threading
import time
from pathlib import Path

from . import config, fileio, textutil


class GraniteLLM:
    def __init__(
        self,
        model_path: Path = config.GRANITE_PATH,
        n_threads: int = config.N_THREADS,
        n_ctx: int = config.GRANITE_CTX,
        verbose: bool = False,
        log_path: Path | None = None,
    ):
        path = Path(model_path).expanduser()
        if not path.is_absolute():
            path = config.ROOT / path
        if not path.is_file():
            raise FileNotFoundError(
                f"Granite model file not found: {path}. "
                "See handoffs/person3_granite.md for the model setup."
            )

        self.log_path = Path(log_path) if log_path is not None else None
        self.model_file = path.name
        llama_cpp = importlib.import_module("llama_cpp")
        start = time.perf_counter()
        self.model = llama_cpp.Llama(
            model_path=str(path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_threads_batch=n_threads,
            seed=config.SEED,
            verbose=verbose,
        )
        self.load_s = time.perf_counter() - start

    @staticmethod
    def _timing() -> dict:
        return {
            "seconds": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "ok": False,
            "attempts": 0,
        }

    @staticmethod
    def _add_usage(timing: dict, response: dict) -> None:
        usage = response.get("usage") or {}
        timing["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        timing["completion_tokens"] += int(usage.get("completion_tokens") or 0)

    @staticmethod
    def _content(response: dict) -> str:
        content = response["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Granite response content is not text")
        return content

    def _record(self, kind: str, timing: dict, error: str | None) -> None:
        if self.log_path is not None:
            fileio.append_jsonl(
                self.log_path, {"kind": kind, "timing": timing, "error": error}
            )

    @staticmethod
    def _plain_json_fields(value: dict, schema: dict) -> dict:
        """Adapt scalar or per-report answers; grounding still checks every quote."""
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return value
        reports = [item for item in value.values() if isinstance(item, dict)]
        if not any(name in item for item in [value, *reports] for name in properties):
            raise ValueError("JSON response had no extraction fields")
        fields = {}
        for name in properties:
            candidates = [value.get(name)] + [item.get(name) for item in reports]
            field = next((item for item in candidates if item is not None), None)
            if isinstance(field, dict):
                if field.get("evidence") is None and isinstance(field.get("value"), str):
                    field = {**field, "evidence": field["value"]}
                fields[name] = field
            elif isinstance(field, str):
                fields[name] = {"value": field, "evidence": field, "confidence": 0.0}
            else:
                fields[name] = None
        return fields

    def chat_json(
        self, system: str, user: str, schema: dict, max_tokens: int
    ) -> tuple[dict | None, dict]:
        """Parse plain chat output without native grammar, which aborts on LinuxONE.

        The caller's ground_fields check still rejects quotes absent from masked
        reports. Malformed output is retried once and otherwise returns None.
        """
        system = textutil.mask_privacy(system)
        user = textutil.mask_privacy(user)
        timing = self._timing()
        start = time.perf_counter()
        parsed: dict | None = None
        error: str | None = None

        try:
            for retry in range(2):
                prompt = user if retry == 0 else user + "\nReturn only valid JSON."
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ]
                kwargs = {
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                }
                timing["attempts"] += 1
                response = self.model.create_chat_completion(**kwargs)

                self._add_usage(timing, response)
                try:
                    value = json.loads(self._content(response))
                    if not isinstance(value, dict):
                        raise ValueError("JSON response was not an object")
                    parsed = self._plain_json_fields(value, schema)
                except (ValueError, TypeError) as exc:
                    error = type(exc).__name__  # Backend exceptions may echo private prompt/output text.
                    continue

                timing["ok"] = True
                error = None
                break
        except Exception as exc:
            error = type(exc).__name__  # Backend exceptions may echo private prompt/output text.
        finally:
            timing["seconds"] = time.perf_counter() - start
            self._record("json", timing, error)

        return parsed, timing

    def chat_text(self, system: str, user: str, max_tokens: int) -> tuple[str, dict]:
        system = textutil.mask_privacy(system)
        user = textutil.mask_privacy(user)
        timing = self._timing()
        start = time.perf_counter()
        content = ""
        error: str | None = None

        try:
            timing["attempts"] = 1
            response = self.model.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            self._add_usage(timing, response)
            content = self._content(response)
            timing["ok"] = bool(content.strip())
            if not timing["ok"]:
                error = "Granite returned empty text"
        except Exception as exc:
            error = type(exc).__name__  # Backend exceptions may echo private prompt/output text.
        finally:
            timing["seconds"] = time.perf_counter() - start
            self._record("text", timing, error)

        return content, timing


_singleton: GraniteLLM | None = None
_singleton_lock = threading.Lock()


def get_llm(**kwargs) -> GraniteLLM:
    """Return the process-wide Granite instance, loading it on first use."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            kwargs.setdefault("model_path", config.GRANITE_PATH)
            _singleton = GraniteLLM(**kwargs)
        elif "log_path" in kwargs:
            _singleton.log_path = Path(kwargs["log_path"]) if kwargs["log_path"] is not None else None
        return _singleton


def unload() -> None:
    """Release the singleton and collect the model's native allocations."""
    global _singleton
    with _singleton_lock:
        instance = _singleton
        _singleton = None
        if instance is not None:
            instance.model = None
    del instance
    gc.collect()
