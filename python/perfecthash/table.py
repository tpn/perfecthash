from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from .hash_functions import GoodHashFunction
from .online.rawdog_jit import RawdogJitTable, build_rawdog_jit_table


class TableBackend(StrEnum):
    Auto = "auto"
    RawdogJit = "rawdog_jit"


class BuildTableOptions(BaseModel):
    model_config = ConfigDict(frozen=True)

    hash_function: GoodHashFunction
    backend: TableBackend = TableBackend.Auto
    library_path: Path | None = None


class _TableImpl(Protocol):
    @property
    def hash_function(self) -> GoodHashFunction: ...

    @property
    def keys(self) -> Sequence[int]: ...

    @property
    def library_path(self) -> Path: ...

    def close(self) -> None: ...

    def index(self, key: int) -> int: ...

    def indexes(self, keys: Sequence[int] | Iterable[int]) -> list[int]: ...


class Table:
    def __init__(self, impl: _TableImpl, *, backend: TableBackend) -> None:
        self._impl = impl
        self.backend = backend
        self.hash_function = impl.hash_function
        self.library_path = impl.library_path

    @property
    def key_count(self) -> int:
        return len(self._impl.keys)

    def close(self) -> None:
        self._impl.close()

    def index(self, key: int) -> int:
        return self._impl.index(key)

    def index_many(self, keys: Sequence[int] | Iterable[int]) -> list[int]:
        return self._impl.indexes(keys)

    def __enter__(self) -> Table:
        return self

    def __exit__(self, *args: object) -> None:
        del args
        self.close()


def build_table(
    keys: Sequence[int] | Iterable[int],
    *,
    hash_function: GoodHashFunction | str,
    backend: TableBackend | str = TableBackend.Auto,
    library_path: str | Path | None = None,
) -> Table:
    normalized_hash_function = (
        hash_function
        if isinstance(hash_function, GoodHashFunction)
        else GoodHashFunction(hash_function)
    )
    normalized_backend = (
        backend if isinstance(backend, TableBackend) else TableBackend(backend)
    )
    normalized_library_path = (
        library_path
        if library_path is None or isinstance(library_path, Path)
        else Path(library_path)
    )

    options = BuildTableOptions(
        hash_function=normalized_hash_function,
        backend=normalized_backend,
        library_path=normalized_library_path,
    )

    if options.backend in (TableBackend.Auto, TableBackend.RawdogJit):
        impl: RawdogJitTable = build_rawdog_jit_table(
            keys,
            options.hash_function,
            library_path=options.library_path,
        )
        return Table(impl, backend=TableBackend.RawdogJit)

    raise NotImplementedError(f"Unsupported backend: {options.backend}")
