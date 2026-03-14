from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from perfecthash.c_cli import (
    CliBinaryNotFoundError,
    _candidate_roots,
    find_default_bulk_create_binary_path,
    find_default_create_binary_path,
    format_command,
    render_bulk_create_command,
    render_create_command,
    run_bulk_create_command,
    run_create_command,
)
from perfecthash.hash_functions import GoodHashFunction
from perfecthash.models import BulkCreateRequest, CreateRequest


def test_candidate_roots_include_repo_root() -> None:
    roots = _candidate_roots()

    expected_root = Path(__file__).resolve().parents[2]
    assert roots[0] == expected_root
    assert (roots[0] / "pyproject.toml").is_file()
    assert (roots[0] / "python" / "perfecthash").is_dir()


def test_render_create_command_replaces_program_name(tmp_path: Path) -> None:
    create_binary = tmp_path / "PerfectHashCreate"
    create_binary.write_text("", encoding="utf-8")
    request = CreateRequest(
        keys_path=Path("keys/example.keys"),
        output_dir=Path("out"),
        hash_function=GoodHashFunction.MultiplyShiftR,
    )

    argv = render_create_command(request, create_binary=create_binary)

    assert argv[0] == str(create_binary.resolve())
    assert argv[1:] == [
        "keys/example.keys",
        "out",
        "Chm01",
        "MultiplyShiftR",
        "And",
        "0",
    ]


def test_render_create_command_raises_if_binary_missing() -> None:
    request = CreateRequest(
        keys_path=Path("keys/example.keys"),
        output_dir=Path("out"),
        hash_function=GoodHashFunction.MultiplyShiftR,
    )

    with pytest.raises(CliBinaryNotFoundError):
        render_create_command(
            request,
            create_binary=Path("/tmp/does-not-exist/PerfectHashCreate"),
        )


def test_format_command_quotes_paths() -> None:
    command = format_command(
        ["/tmp/PerfectHashCreate", "keys/example.keys", "dir with spaces"]
    )

    assert command == "/tmp/PerfectHashCreate keys/example.keys 'dir with spaces'"


def test_run_create_command_invokes_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    create_binary = tmp_path / "PerfectHashCreate"
    create_binary.write_text("", encoding="utf-8")
    request = CreateRequest(
        keys_path=Path("keys/example.keys"),
        output_dir=Path("out"),
        hash_function=GoodHashFunction.MultiplyShiftR,
    )
    recorded: dict[str, object] = {}

    def fake_run(
        argv: list[str], *, check: bool, env: dict[str, str]
    ) -> subprocess.CompletedProcess[bytes]:
        recorded["argv"] = argv
        recorded["check"] = check
        recorded["env"] = env
        return subprocess.CompletedProcess(args=argv, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_create_command(request, create_binary=create_binary)

    assert result.returncode == 0
    assert recorded["check"] is False
    assert isinstance(recorded["env"], dict)
    assert recorded["argv"] == [
        str(create_binary.resolve()),
        "keys/example.keys",
        "out",
        "Chm01",
        "MultiplyShiftR",
        "And",
        "0",
    ]


def test_find_default_create_binary_path_env_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    create_binary = tmp_path / "PerfectHashCreate"
    create_binary.write_text("", encoding="utf-8")

    monkeypatch.setenv("PERFECTHASH_CREATE_BINARY", str(create_binary))

    assert find_default_create_binary_path() == create_binary.resolve()


def test_find_default_create_binary_path_prefix_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    create_binary = tmp_path / "bin" / "PerfectHashCreate"
    create_binary.parent.mkdir(parents=True)
    create_binary.write_text("", encoding="utf-8")

    monkeypatch.delenv("PERFECTHASH_CREATE_BINARY", raising=False)
    monkeypatch.delenv("PERFECTHASH_CREATE_EXE", raising=False)
    monkeypatch.setenv("PERFECTHASH_PREFIX", str(tmp_path))

    assert find_default_create_binary_path() == create_binary.resolve()


def test_render_bulk_create_command_replaces_program_name(tmp_path: Path) -> None:
    bulk_create_binary = tmp_path / "PerfectHashBulkCreate"
    bulk_create_binary.write_text("", encoding="utf-8")
    request = BulkCreateRequest(
        keys_dir=Path("keys"),
        output_dir=Path("out"),
        hash_function=GoodHashFunction.MultiplyShiftR,
    )

    argv = render_bulk_create_command(
        request,
        bulk_create_binary=bulk_create_binary,
    )

    assert argv[0] == str(bulk_create_binary.resolve())
    assert argv[1:] == [
        "keys",
        "out",
        "Chm01",
        "MultiplyShiftR",
        "And",
        "0",
    ]


def test_run_bulk_create_command_invokes_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bulk_create_binary = tmp_path / "PerfectHashBulkCreate"
    bulk_create_binary.write_text("", encoding="utf-8")
    request = BulkCreateRequest(
        keys_dir=Path("keys"),
        output_dir=Path("out"),
        hash_function=GoodHashFunction.MultiplyShiftR,
    )
    recorded: dict[str, object] = {}

    def fake_run(
        argv: list[str], *, check: bool, env: dict[str, str]
    ) -> subprocess.CompletedProcess[bytes]:
        recorded["argv"] = argv
        recorded["check"] = check
        recorded["env"] = env
        return subprocess.CompletedProcess(args=argv, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_bulk_create_command(
        request,
        bulk_create_binary=bulk_create_binary,
    )

    assert result.returncode == 0
    assert recorded["check"] is False
    assert isinstance(recorded["env"], dict)
    assert recorded["argv"] == [
        str(bulk_create_binary.resolve()),
        "keys",
        "out",
        "Chm01",
        "MultiplyShiftR",
        "And",
        "0",
    ]


def test_find_default_bulk_create_binary_path_env_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bulk_create_binary = tmp_path / "PerfectHashBulkCreate"
    bulk_create_binary.write_text("", encoding="utf-8")

    monkeypatch.setenv("PERFECTHASH_BULK_CREATE_BINARY", str(bulk_create_binary))

    assert find_default_bulk_create_binary_path() == bulk_create_binary.resolve()


def test_find_default_bulk_create_binary_path_prefix_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bulk_create_binary = tmp_path / "bin" / "PerfectHashBulkCreate"
    bulk_create_binary.parent.mkdir(parents=True)
    bulk_create_binary.write_text("", encoding="utf-8")

    monkeypatch.delenv("PERFECTHASH_BULK_CREATE_BINARY", raising=False)
    monkeypatch.delenv("PERFECTHASH_BULK_CREATE_EXE", raising=False)
    monkeypatch.setenv("PERFECTHASH_PREFIX", str(tmp_path))

    assert find_default_bulk_create_binary_path() == bulk_create_binary.resolve()
