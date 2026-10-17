"""Tiny test helpers (pytest isn't installed on the VM).

In tests/test_*.py:
    from surgedesk.testing import skip, assert_close, load_fixture
"""
from __future__ import annotations

import math

from . import config
from .fileio import load_embeddings, read_jsonl


class SkipTest(Exception):
    """Raised by skip(); the self-test counts the test as skipped."""


def skip(reason: str) -> None:
    raise SkipTest(reason)


def assert_close(a: float, b: float, tol: float = 1e-6, msg: str = "") -> None:
    if not (math.isclose(float(a), float(b), abs_tol=tol)):
        raise AssertionError(f"{msg} {a} != {b} (tol {tol})")


def fixture_dir(event: str = config.FIXTURE_EVENT):
    return config.FIXTURES / "outputs" / event


def load_fixture(name: str, cls=None, event: str = config.FIXTURE_EVENT):
    """load_fixture('reports.jsonl', Report) or load_fixture('embeddings.npy')."""
    path = fixture_dir(event) / name
    if name.endswith(".npy"):
        return load_embeddings(path)
    return read_jsonl(path, cls)
