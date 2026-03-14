from __future__ import annotations

import os
import re
import subprocess
from importlib.metadata import PackageNotFoundError, version as package_version
from functools import lru_cache
from pathlib import Path

FALLBACK_VERSION = "0.63.0"
PACKAGE_DISTRIBUTION_NAME = "tpn-perfecthash"
_VERSION_PATTERN = re.compile(r"^[0-9]+(\.[0-9]+){1,3}$")


def _normalize_version(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip()
    if normalized.startswith("v"):
        normalized = normalized[1:]

    if _VERSION_PATTERN.fullmatch(normalized):
        return normalized

    return None


def _source_checkout_root() -> Path | None:
    module_path = Path(__file__).resolve()

    for candidate in module_path.parents:
        pyproject = candidate / "pyproject.toml"
        package_dir = candidate / "python" / "perfecthash"
        if pyproject.is_file() and package_dir.is_dir():
            return candidate

    return None


def _git_output(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None

    if result.returncode != 0:
        return None

    output = result.stdout.strip()
    return output or None


def _resolve_version_from_git(root: Path) -> str | None:
    exact_tag = _git_output(root, "describe", "--tags", "--exact-match", "--match", "v[0-9]*")
    normalized = _normalize_version(exact_tag)
    if normalized is not None:
        return normalized

    latest_tag = _git_output(root, "describe", "--tags", "--abbrev=0", "--match", "v[0-9]*")
    return _normalize_version(latest_tag)


@lru_cache(maxsize=1)
def resolve_build_version() -> str:
    for env_name in (
        "PERFECTHASH_PYTHON_VERSION",
        "PERFECTHASH_VERSION_OVERRIDE",
        "PERFECTHASH_CONDA_VERSION",
    ):
        normalized = _normalize_version(os.environ.get(env_name))
        if normalized is not None:
            return normalized

    root = _source_checkout_root()
    if root is not None:
        normalized = _resolve_version_from_git(root)
        if normalized is not None:
            return normalized

    return FALLBACK_VERSION


@lru_cache(maxsize=1)
def resolve_runtime_version() -> str:
    for env_name in (
        "PERFECTHASH_PYTHON_VERSION",
        "PERFECTHASH_VERSION_OVERRIDE",
        "PERFECTHASH_CONDA_VERSION",
    ):
        normalized = _normalize_version(os.environ.get(env_name))
        if normalized is not None:
            return normalized

    try:
        normalized = _normalize_version(package_version(PACKAGE_DISTRIBUTION_NAME))
        if normalized is not None:
            return normalized
    except PackageNotFoundError:
        pass

    return resolve_build_version()


__version__ = resolve_runtime_version()
