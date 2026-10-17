"""Self-test: `python -m surgedesk.selftest` (exit code 0 = pass).

1. Validates every fixture file against its contract.
2. Runs the stub pipeline on the fixture event and validates its outputs.
3. Discovers tests/test_*.py and runs every function named test_* (no pytest needed,
   since pytest is not installed on the VM). A test may call `skip("reason")`.

Each agent adds its own tests in tests/test_agent<N>_*.py. Tests must be fast
(seconds), offline, and must not load Granite unless the function name ends in
`_granite` (those only run with --granite).

Owner: foundation (frozen).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import traceback
from pathlib import Path

from . import config
from .contracts import Report
from .fileio import read_jsonl, validate_file
from .testing import SkipTest


def _check_fixtures() -> list[str]:
    errors = []
    d = config.FIXTURES / "outputs" / config.FIXTURE_EVENT
    for name in ["reports.jsonl", "tagged.jsonl", "incidents.jsonl", "briefings.jsonl", "decisions.jsonl"]:
        try:
            n = validate_file(d / name)
            print(f"  ok   fixture {name:<16} {n} rows")
        except Exception as e:  # noqa: BLE001
            errors.append(f"fixture {name}: {e}")
            print(f"  FAIL fixture {name}: {e}")
    return errors


def _check_stub_pipeline() -> list[str]:
    from .stubs import run_stub_pipeline

    errors = []
    try:
        reports = read_jsonl(config.FIXTURES / "outputs" / config.FIXTURE_EVENT / "reports.jsonl", Report)
        res = run_stub_pipeline(config.FIXTURE_EVENT, reports)
        for name in ["reports.jsonl", "tagged.jsonl", "incidents.jsonl", "briefings.jsonl"]:
            validate_file(Path(res["dir"]) / name)
        print(f"  ok   stub pipeline: {res['reports']} reports -> {res['incidents']} incidents -> {res['briefings']} briefings")
    except Exception as e:  # noqa: BLE001
        errors.append(f"stub pipeline: {e}")
        print(f"  FAIL stub pipeline: {e}")
        traceback.print_exc()
    return errors


def _run_tests(pattern: str, include_granite: bool) -> tuple[int, int, int, list[str]]:
    tests_dir = config.ROOT / "tests"
    passed = failed = skipped = 0
    errors: list[str] = []
    files = sorted(tests_dir.glob("test_*.py"))
    if pattern:
        files = [f for f in files if pattern in f.name]
    for path in files:
        spec = importlib.util.spec_from_file_location(f"_sdtest_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        try:
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
        except SkipTest as e:
            print(f"  skip {path.name}: {e}")
            skipped += 1
            continue
        except Exception as e:  # noqa: BLE001
            errors.append(f"{path.name}: import failed: {e}")
            print(f"  FAIL {path.name}: import failed: {e}")
            failed += 1
            continue
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            if name.endswith("_granite") and not include_granite:
                continue
            fn = getattr(mod, name)
            t0 = time.time()
            try:
                fn()
                passed += 1
                print(f"  ok   {path.name}::{name} ({time.time() - t0:.2f}s)")
            except SkipTest as e:
                skipped += 1
                print(f"  skip {path.name}::{name}: {e}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                errors.append(f"{path.name}::{name}: {e}")
                print(f"  FAIL {path.name}::{name}: {e}")
                traceback.print_exc()
    return passed, failed, skipped, errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SurgeDesk self-test")
    ap.add_argument("-k", default="", help="only run test files whose name contains this text")
    ap.add_argument("--granite", action="store_true", help="also run tests ending in _granite (loads Granite)")
    args = ap.parse_args(argv)

    print("SurgeDesk self-test")
    errors = _check_fixtures() + _check_stub_pipeline()
    p, f, s, terr = _run_tests(args.k, args.granite)
    errors += terr
    print("-" * 64)
    print(f"tests: {p} passed, {f} failed, {s} skipped; foundation errors: {len(errors) - len(terr)}")
    if errors:
        print("SELFTEST FAILED")
        for e in errors:
            print("  -", e)
        return 1
    print("SELFTEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
