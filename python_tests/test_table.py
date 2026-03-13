from __future__ import annotations

import pytest

from perfecthash import BuildTableOptions, TableBackend, build_table
from perfecthash.hash_functions import GoodHashFunction
from perfecthash.online.rawdog_jit import find_default_rawdog_jit_library_path

pytestmark = pytest.mark.skipif(
    find_default_rawdog_jit_library_path() is None,
    reason="PerfectHash shared library not found for Table API tests.",
)


def test_build_table_auto_selects_rawdog_jit() -> None:
    keys = [1, 3, 5, 7, 11, 13, 17, 19]

    with build_table(
        keys,
        hash_function="MultiplyShiftR",
    ) as table:
        indexes = table.index_many(keys)

    assert table.backend == TableBackend.RawdogJit
    assert table.hash_function == GoodHashFunction.MultiplyShiftR
    assert table.key_count == len(keys)
    assert len(indexes) == len(set(indexes))


def test_build_table_accepts_explicit_options_model() -> None:
    keys = [37, 13, 1, 53, 7, 19, 41, 3]
    options = BuildTableOptions(
        hash_function=GoodHashFunction.MultiplyShiftR,
        backend=TableBackend.RawdogJit,
    )

    with build_table(
        keys,
        hash_function=options.hash_function,
        backend=options.backend,
        library_path=options.library_path,
    ) as table:
        assert len(table.index_many(keys)) == len(keys)
