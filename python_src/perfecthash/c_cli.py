from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from .c_argv import render_bulk_create_c_argv, render_create_c_argv
from .discovery import installed_binary_dirs, source_checkout_root
from .models import BulkCreateRequest, CreateRequest


class CliBinaryNotFoundError(RuntimeError):
    pass


def _create_binary_names() -> tuple[str, ...]:
    if sys.platform == "win32":
        return ("PerfectHashCreate.exe",)

    return ("PerfectHashCreate", "PerfectHashCreateExe")


def _bulk_create_binary_names() -> tuple[str, ...]:
    if sys.platform == "win32":
        return ("PerfectHashBulkCreate.exe",)

    return ("PerfectHashBulkCreate", "PerfectHashBulkCreateExe")


def _candidate_roots() -> tuple[Path, ...]:
    package_root = source_checkout_root(__file__)
    if package_root is None:
        return ()

    sibling_root = package_root.parent / "perfecthash"
    if sibling_root == package_root:
        return (package_root,)

    return (package_root, sibling_root)


def _find_binary_path(
    *,
    env_names: tuple[str, ...],
    binary_names: tuple[str, ...],
) -> Path | None:
    for env_name in env_names:
        env_value = os.environ.get(env_name)
        if env_value:
            candidate = Path(env_value).expanduser()
            if candidate.is_file():
                return candidate.resolve()

    for directory in installed_binary_dirs():
        for binary_name in binary_names:
            candidate = directory / binary_name
            if candidate.is_file():
                return candidate.resolve()

    for root in _candidate_roots():
        for binary_name in binary_names:
            for candidate in sorted(root.glob(f"build*/bin/**/{binary_name}")):
                if candidate.is_file():
                    return candidate.resolve()

            for candidate in sorted(root.glob(f"build*/bin/{binary_name}")):
                if candidate.is_file():
                    return candidate.resolve()

            direct_candidates = (
                root / "build" / "bin" / binary_name,
                root / "src" / "x64" / "Release" / binary_name,
            )
            for candidate in direct_candidates:
                if candidate.is_file():
                    return candidate.resolve()

    return None


def find_default_create_binary_path() -> Path | None:
    return _find_binary_path(
        env_names=("PERFECTHASH_CREATE_BINARY", "PERFECTHASH_CREATE_EXE"),
        binary_names=_create_binary_names(),
    )


def find_default_bulk_create_binary_path() -> Path | None:
    return _find_binary_path(
        env_names=(
            "PERFECTHASH_BULK_CREATE_BINARY",
            "PERFECTHASH_BULK_CREATE_EXE",
        ),
        binary_names=_bulk_create_binary_names(),
    )


def render_create_command(
    request: CreateRequest,
    *,
    create_binary: str | Path | None = None,
) -> list[str]:
    resolved_binary = (
        Path(create_binary).expanduser().resolve()
        if create_binary is not None
        else find_default_create_binary_path()
    )
    if resolved_binary is None or not resolved_binary.is_file():
        raise CliBinaryNotFoundError(
            "Unable to locate the PerfectHash create binary. "
            "Set PERFECTHASH_CREATE_BINARY or pass --create-binary."
        )

    argv = render_create_c_argv(request)
    argv[0] = str(resolved_binary)
    return argv


def format_command(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def run_create_command(
    request: CreateRequest,
    *,
    create_binary: str | Path | None = None,
) -> subprocess.CompletedProcess[bytes]:
    argv = render_create_command(request, create_binary=create_binary)
    return subprocess.run(argv, check=False)


def render_bulk_create_command(
    request: BulkCreateRequest,
    *,
    bulk_create_binary: str | Path | None = None,
) -> list[str]:
    resolved_binary = (
        Path(bulk_create_binary).expanduser().resolve()
        if bulk_create_binary is not None
        else find_default_bulk_create_binary_path()
    )
    if resolved_binary is None or not resolved_binary.is_file():
        raise CliBinaryNotFoundError(
            "Unable to locate the PerfectHash bulk-create binary. "
            "Set PERFECTHASH_BULK_CREATE_BINARY or pass --bulk-create-binary."
        )

    argv = render_bulk_create_c_argv(request)
    argv[0] = str(resolved_binary)
    return argv


def run_bulk_create_command(
    request: BulkCreateRequest,
    *,
    bulk_create_binary: str | Path | None = None,
) -> subprocess.CompletedProcess[bytes]:
    argv = render_bulk_create_command(
        request,
        bulk_create_binary=bulk_create_binary,
    )
    return subprocess.run(argv, check=False)
