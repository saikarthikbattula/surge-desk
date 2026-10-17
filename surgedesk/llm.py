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
        self._llama_cpp = llama_cpp
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

    def chat_json(
        self, system: str, user: str, schema: dict, max_tokens: int
    ) -> tuple[dict | None, dict]:
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
                try:
                    timing["attempts"] += 1
                    response = self.model.create_chat_completion(
                        **kwargs,
                        response_format={"type": "json_object", "schema": schema},
                    )
                except (TypeError, ValueError):
                    grammar = self._llama_cpp.LlamaGrammar.from_json_schema(
                        json.dumps(schema)
                    )
                    timing["attempts"] += 1
                    response = self.model.create_chat_completion(
                        **kwargs, grammar=grammar
                    )

                self._add_usage(timing, response)
                try:
                    value = json.loads(self._content(response))
                    if not isinstance(value, dict):
                        raise ValueError("JSON response was not an object")
                except (ValueError, TypeError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    continue

                parsed = value
                timing["ok"] = True
                error = None
                break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
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
            error = f"{type(exc).__name__}: {exc}"
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
