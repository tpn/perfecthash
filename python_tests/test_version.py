from __future__ import annotations

import perfecthash
from perfecthash._version import resolve_runtime_version


def test_package_version() -> None:
    assert perfecthash.__version__ == resolve_runtime_version()
