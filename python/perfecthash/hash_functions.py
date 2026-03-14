from __future__ import annotations

from enum import StrEnum


class GoodHashFunction(StrEnum):
    MultiplyShiftR = "MultiplyShiftR"
    MultiplyShiftRX = "MultiplyShiftRX"
    Mulshrolate1RX = "Mulshrolate1RX"
    Mulshrolate2RX = "Mulshrolate2RX"
    Mulshrolate3RX = "Mulshrolate3RX"
    Mulshrolate4RX = "Mulshrolate4RX"


GOOD_HASH_FUNCTIONS: tuple[GoodHashFunction, ...] = tuple(GoodHashFunction)
GOOD_HASH_FUNCTION_NAMES: tuple[str, ...] = tuple(
    hash_function.value for hash_function in GOOD_HASH_FUNCTIONS
)
