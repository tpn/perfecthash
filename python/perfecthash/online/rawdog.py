from __future__ import annotations

from .rawdog_jit import (
    OnlineLibraryNotFoundError,
    RawdogJitTable,
    build_rawdog_jit_table,
    find_default_rawdog_jit_library_path,
)

# Backward-compatible aliases for the initial bootstrap slice.
RawdogTable = RawdogJitTable
build_rawdog_table = build_rawdog_jit_table
find_default_library_path = find_default_rawdog_jit_library_path

__all__ = [
    "OnlineLibraryNotFoundError",
    "RawdogJitTable",
    "RawdogTable",
    "build_rawdog_jit_table",
    "build_rawdog_table",
    "find_default_library_path",
    "find_default_rawdog_jit_library_path",
]
