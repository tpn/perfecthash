from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from pathlib import Path


def _dedupe_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    unique_paths: list[Path] = []
    seen: set[Path] = set()

    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)

    return tuple(unique_paths)


def source_checkout_root(module_file: str | Path) -> Path | None:
    module_path = Path(module_file).resolve()

    for candidate in module_path.parents:
        pyproject = candidate / "pyproject.toml"
        package_dir = candidate / "python_src" / "perfecthash"
        if pyproject.is_file() and package_dir.is_dir():
            return candidate

    return None


def package_native_root() -> Path:
    return Path(__file__).resolve().parent / "_native"


def install_prefixes() -> tuple[Path, ...]:
    candidates: list[Path] = []

    for env_name in (
        "PERFECTHASH_PREFIX",
        "PERFECTHASH_INSTALL_PREFIX",
        "CONDA_PREFIX",
    ):
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(Path(env_value).expanduser())

    candidates.append(Path(sys.prefix))
    candidates.append(Path(sys.executable).resolve().parent.parent)

    if getattr(sys, "base_prefix", sys.prefix) != sys.prefix:
        candidates.append(Path(sys.base_prefix))

    return _dedupe_paths(candidates)


def installed_binary_dirs() -> tuple[Path, ...]:
    candidates = [package_native_root() / "bin"]

    for prefix in install_prefixes():
        if sys.platform == "win32":
            candidates.extend(
                [
                    prefix / "Scripts",
                    prefix / "bin",
                    prefix / "Library" / "bin",
                ]
            )
        else:
            candidates.append(prefix / "bin")

    return _dedupe_paths(candidates)


def installed_library_dirs() -> tuple[Path, ...]:
    candidates = [package_native_root() / "lib"]
    if sys.platform == "win32":
        candidates.append(package_native_root() / "bin")

    for prefix in install_prefixes():
        if sys.platform == "win32":
            candidates.extend(
                [
                    prefix / "Library" / "bin",
                    prefix / "bin",
                    prefix / "Lib",
                ]
            )
        else:
            candidates.append(prefix / "lib")

    return _dedupe_paths(candidates)
