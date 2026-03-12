from __future__ import annotations

import perfecthash


def test_package_version() -> None:
    assert perfecthash.__version__ == "0.1.0a0"
