from __future__ import annotations

from pathlib import Path

import pytest

from perfecthash.hash_functions import GoodHashFunction
from perfecthash.online import OnlineLibraryNotFoundError, build_rawdog_table
from perfecthash.online.rawdog import find_default_library_path
from perfecthash.online.rawdog_jit import _candidate_roots

pytestmark = pytest.mark.skipif(
    find_default_library_path() is None,
    reason="PerfectHash shared library not found for online RawDog tests.",
)


def test_candidate_roots_include_repo_root() -> None:
    roots = _candidate_roots()

    expected_root = Path(__file__).resolve().parents[2]
    assert roots[0] == expected_root
    assert (roots[0] / "pyproject.toml").is_file()
    assert (roots[0] / "python" / "perfecthash").is_dir()


def test_build_rawdog_table_indexes_small_sorted_key_set() -> None:
    keys = [1, 3, 5, 7, 11, 13, 17, 19]

    with build_rawdog_table(
        keys,
        GoodHashFunction.MultiplyShiftR,
    ) as table:
        indexes = [table.index(key) for key in keys]

    assert len(indexes) == len(set(indexes))
    assert sorted(indexes) == list(range(len(keys)))


def test_build_rawdog_table_indexes_small_unsorted_key_set() -> None:
    keys = [37, 13, 1, 53, 7, 19, 41, 3]

    with build_rawdog_table(
        keys,
        GoodHashFunction.MultiplyShiftR,
    ) as table:
        indexes = table.indexes(keys)

    assert len(indexes) == len(set(indexes))


def test_build_rawdog_table_rejects_out_of_range_keys() -> None:
    with pytest.raises(ValueError, match="uint32 range"):
        build_rawdog_table([0x1_0000_0000], GoodHashFunction.MultiplyShiftR)


def test_build_rawdog_table_raises_if_library_missing(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "libPerfectHash.so"

    with pytest.raises(OnlineLibraryNotFoundError):
        build_rawdog_table(
            [1, 3, 5],
            GoodHashFunction.MultiplyShiftR,
            library_path=missing_path,
        )


def test_find_default_rawdog_jit_library_path_prefix_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    library_path = tmp_path / "lib" / "libPerfectHash.so"
    library_path.parent.mkdir(parents=True)
    library_path.write_text("", encoding="utf-8")

    monkeypatch.delenv("PERFECTHASH_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("PERFECTHASH_LIB_PATH", raising=False)
    monkeypatch.setenv("PERFECTHASH_PREFIX", str(tmp_path))

    assert find_default_library_path() == library_path.resolve()
