from __future__ import annotations

from perfecthash import GOOD_HASH_FUNCTIONS
from perfecthash.hash_functions import GOOD_HASH_FUNCTION_NAMES


def test_good_hash_functions_match_curated_set() -> None:
    assert [hash_function.value for hash_function in GOOD_HASH_FUNCTIONS] == [
        "MultiplyShiftR",
        "MultiplyShiftRX",
        "Mulshrolate1RX",
        "Mulshrolate2RX",
        "Mulshrolate3RX",
        "Mulshrolate4RX",
    ]


def test_good_hash_function_names_match_curated_set() -> None:
    assert GOOD_HASH_FUNCTION_NAMES == (
        "MultiplyShiftR",
        "MultiplyShiftRX",
        "Mulshrolate1RX",
        "Mulshrolate2RX",
        "Mulshrolate3RX",
        "Mulshrolate4RX",
    )
