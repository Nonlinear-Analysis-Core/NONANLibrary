#!/usr/bin/env python3
"""Headless entry point for the NONAN Python test suite.

    python3 tests/python/run_tests.py            # everything
    python3 tests/python/run_tests.py emd        # name filter

Exits 0 if every test passed, 1 otherwise.

Deliberately stdlib unittest, not pytest: the suite must run on a bare
interpreter with numpy and nothing else, so that "the tests do not run here"
is never an excuse. numpy and scipy are the only third-party imports, and both
are already required by the library itself (requirements.txt).
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def main(argv):
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(REPO, "python"))

    pattern = "test_*.py"
    if len(argv) > 1:
        pattern = f"test*{argv[1]}*.py"

    suite = unittest.defaultTestLoader.discover(HERE, pattern=pattern)
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    print("\n=============== NONAN python test summary ===============")
    print(f"  ran      {result.testsRun}")
    print(f"  failed   {len(result.failures)}")
    print(f"  errored  {len(result.errors)}")
    print(f"  skipped  {len(result.skipped)}")
    print("=========================================================")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
