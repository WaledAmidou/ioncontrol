#!/usr/bin/env python
"""
Minimal test runner for environments without pytest.

    python tests/run_tests.py

Identical assertions to `pytest -q`; use pytest when it is available.
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_smoke as T  # noqa: E402


def main():
    fns = [(n, f) for n, f in vars(T).items()
           if n.startswith("test_") and callable(f)]
    npass = nskip = nfail = 0
    for name, fn in fns:
        if not T.HAVE_DATA and getattr(fn, "_needs_data", False):
            print(f"SKIP  {name} (no .xpt files)")
            nskip += 1
            continue
        try:
            fn()
            print(f"PASS  {name}")
            npass += 1
        except Exception:
            print(f"FAIL  {name}")
            traceback.print_exc()
            nfail += 1
    print(f"\n{npass} passed, {nskip} skipped, {nfail} failed")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
