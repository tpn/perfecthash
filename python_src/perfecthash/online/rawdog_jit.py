from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from perfecthash.discovery import installed_library_dirs, source_checkout_root
from perfecthash.hash_functions import GoodHashFunction

_ContextPointer = ctypes.c_void_p
_TablePointer = ctypes.c_void_p

_RAWDOG_JIT_HASH_FUNCTIONS: dict[GoodHashFunction, int] = {
    GoodHashFunction.MultiplyShiftR: 0,
    GoodHashFunction.MultiplyShiftRX: 4,
    GoodHashFunction.Mulshrolate1RX: 5,
    GoodHashFunction.Mulshrolate2RX: 6,
    GoodHashFunction.Mulshrolate3RX: 7,
    GoodHashFunction.Mulshrolate4RX: 8,
}


class OnlineLibraryNotFoundError(RuntimeError):
    pass


def _candidate_library_names() -> tuple[str, ...]:
    if sys.platform == "win32":
        return ("PerfectHash.dll", "PerfectHashOnlineCore.dll")

    if sys.platform == "darwin":
        return ("libPerfectHash.dylib", "libPerfectHashOnlineCore.dylib")

    return ("libPerfectHash.so", "libPerfectHashOnlineCore.so")


def _candidate_roots() -> tuple[Path, ...]:
    package_root = source_checkout_root(__file__)
    if package_root is None:
        return ()

    sibling_root = package_root.parent / "perfecthash"
    if sibling_root == package_root:
        return (package_root,)

    return (package_root, sibling_root)


def find_default_rawdog_jit_library_path() -> Path | None:
    for env_name in ("PERFECTHASH_LIBRARY_PATH", "PERFECTHASH_LIB_PATH"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidate = Path(env_value).expanduser()
            if candidate.is_file():
                return candidate.resolve()

    library_names = _candidate_library_names()

    for directory in installed_library_dirs():
        for library_name in library_names:
            candidate = directory / library_name
            if candidate.is_file():
                return candidate.resolve()

    for root in _candidate_roots():
        for library_name in library_names:
            for candidate in sorted(root.glob(f"build*/lib/{library_name}")):
                if candidate.is_file():
                    return candidate.resolve()

            direct_candidate = root / "build" / "lib" / library_name
            if direct_candidate.is_file():
                return direct_candidate.resolve()

    return None


def _load_library(library_path: Path | None = None) -> ctypes.CDLL:
    candidate = library_path or find_default_rawdog_jit_library_path()
    if candidate is None:
        raise OnlineLibraryNotFoundError(
            "Unable to locate a PerfectHash shared library. "
            "Set PERFECTHASH_LIBRARY_PATH or build the C library first."
        )
    if not candidate.is_file():
        raise OnlineLibraryNotFoundError(
            f"PerfectHash shared library not found: {candidate}"
        )

    library = ctypes.CDLL(str(candidate))
    library.PhOnlineRawdogOpen.argtypes = [ctypes.POINTER(_ContextPointer)]
    library.PhOnlineRawdogOpen.restype = ctypes.c_int32

    library.PhOnlineRawdogClose.argtypes = [_ContextPointer]
    library.PhOnlineRawdogClose.restype = None

    library.PhOnlineRawdogCreateTable32.argtypes = [
        _ContextPointer,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint64,
        ctypes.POINTER(_TablePointer),
    ]
    library.PhOnlineRawdogCreateTable32.restype = ctypes.c_int32

    library.PhOnlineRawdogIndex32.argtypes = [
        _TablePointer,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    library.PhOnlineRawdogIndex32.restype = ctypes.c_int32

    library.PhOnlineRawdogReleaseTable.argtypes = [_TablePointer]
    library.PhOnlineRawdogReleaseTable.restype = None

    return library


def _check_hresult(name: str, result: int) -> None:
    if result < 0:
        raise RuntimeError(f"{name} failed with HRESULT 0x{result & 0xFFFFFFFF:08X}")


def _coerce_keys(keys: Sequence[int] | Iterable[int]) -> list[int]:
    key_list = list(keys)
    if not key_list:
        raise ValueError("keys must not be empty")

    for key in key_list:
        if key < 0 or key > 0xFFFFFFFF:
            raise ValueError(f"key out of uint32 range: {key}")

    return key_list


class RawdogJitTable:
    def __init__(
        self,
        *,
        library: ctypes.CDLL,
        context: _ContextPointer,
        table: _TablePointer,
        hash_function: GoodHashFunction,
        keys: Sequence[int],
        library_path: Path,
    ) -> None:
        self._library = library
        self._context = context
        self._table = table
        self.hash_function = hash_function
        self.keys = tuple(keys)
        self.library_path = library_path

    def close(self) -> None:
        if self._table:
            self._library.PhOnlineRawdogReleaseTable(self._table)
            self._table = _TablePointer()

        if self._context:
            self._library.PhOnlineRawdogClose(self._context)
            self._context = _ContextPointer()

    def index(self, key: int) -> int:
        if key < 0 or key > 0xFFFFFFFF:
            raise ValueError(f"key out of uint32 range: {key}")

        result_index = ctypes.c_uint32()
        result = self._library.PhOnlineRawdogIndex32(
            self._table,
            ctypes.c_uint32(key),
            ctypes.byref(result_index),
        )
        _check_hresult("PhOnlineRawdogIndex32", result)
        return int(result_index.value)

    def indexes(self, keys: Sequence[int] | Iterable[int]) -> list[int]:
        return [self.index(key) for key in keys]

    def __enter__(self) -> RawdogJitTable:
        return self

    def __exit__(self, *args: object) -> None:
        del args
        self.close()


def build_rawdog_jit_table(
    keys: Sequence[int] | Iterable[int],
    hash_function: GoodHashFunction,
    *,
    library_path: str | Path | None = None,
) -> RawdogJitTable:
    key_list = _coerce_keys(keys)
    resolved_library_path: Path
    if library_path is not None:
        resolved_library_path = Path(library_path).expanduser().resolve()
    else:
        discovered_library_path = find_default_rawdog_jit_library_path()
        if discovered_library_path is None:
            raise OnlineLibraryNotFoundError(
                "Unable to locate a PerfectHash shared library. "
                "Set PERFECTHASH_LIBRARY_PATH or build the C library first."
            )
        resolved_library_path = discovered_library_path

    library = _load_library(resolved_library_path)

    context = _ContextPointer()
    result = library.PhOnlineRawdogOpen(ctypes.byref(context))
    _check_hresult("PhOnlineRawdogOpen", result)

    array_type = ctypes.c_uint32 * len(key_list)
    key_array = array_type(*key_list)
    table = _TablePointer()

    try:
        result = library.PhOnlineRawdogCreateTable32(
            context,
            _RAWDOG_JIT_HASH_FUNCTIONS[hash_function],
            key_array,
            ctypes.c_uint64(len(key_list)),
            ctypes.byref(table),
        )
        _check_hresult("PhOnlineRawdogCreateTable32", result)
    except Exception:
        if context:
            library.PhOnlineRawdogClose(context)
        raise

    return RawdogJitTable(
        library=library,
        context=context,
        table=table,
        hash_function=hash_function,
        keys=key_list,
        library_path=resolved_library_path,
    )
