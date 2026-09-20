from __future__ import annotations

import os

import pytest

# Tests should not emit analytics requests. In addition to avoiding external
# network access, this prevents Daft's daemon telemetry threads from racing
# with interpreter shutdown after a test suite creates many runners.
#
# CI and `make test` set DAFT_ANALYTICS_ENABLED=0 in the environment, which is
# what actually guarantees this: Daft reads the opt-out while `daft` is being
# imported, so setting it here only works as long as nothing imports daft
# before this conftest. This line is the fallback for a bare `pytest` run.
os.environ.setdefault("DO_NOT_TRACK", "1")

# Try to import lance; if it fails, all tests in this directory will be skipped.
lance = pytest.importorskip("lance")
