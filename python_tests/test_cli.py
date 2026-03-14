from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

import perfecthash
from perfecthash.cli import app, main
from perfecthash.models import BulkCreateRequest, CreateRequest

runner = CliRunner()


def test_main_returns_zero_for_empty_argv() -> None:
    assert main([]) == 0


def test_module_version_output() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "perfecthash", "--version"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == f"ph {perfecthash.__version__}"


def test_create_help_includes_curated_hash_functions() -> None:
    result = runner.invoke(app, ["create", "--help"])

    assert result.exit_code == 0
    assert "MultiplyShiftR" in result.stdout
    assert "MultiplyShiftRX" in result.stdout
    assert "Mulshrolate1RX" in result.stdout
    assert "Mulshrolate2RX" in result.stdout
    assert "Mulshrolate3RX" in result.stdout
    assert "Mulshrolate4RX" in result.stdout


def test_create_emit_c_argv_preserves_hash_function_name() -> None:
    result = runner.invoke(
        app,
        [
            "create",
            "keys/example.keys",
            "out",
            "--hash-function",
            "MultiplyShiftR",
            "--maximum-concurrency",
            "0",
            "--graph-impl",
            "3",
            "--max-solve-time-in-seconds",
            "20",
            "--do-not-try-use-hash16-impl",
            "--emit-c-argv",
        ],
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == (
        "PerfectHashCreate keys/example.keys out Chm01 MultiplyShiftR "
        "And 0 --DoNotTryUseHash16Impl --GraphImpl=3 "
        "--MaxSolveTimeInSeconds=20"
    )


def test_create_dry_run_renders_resolved_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import perfecthash.cli as cli_module

    def fake_render_create_command(*args: object, **kwargs: object) -> list[str]:
        del args, kwargs
        return ["/tmp/PerfectHashCreate", "keys/example.keys", "out"]

    monkeypatch.setattr(
        cli_module,
        "render_create_command",
        fake_render_create_command,
    )

    result = runner.invoke(
        app,
        [
            "create",
            "keys/example.keys",
            "out",
            "--hash-function",
            "MultiplyShiftR",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "/tmp/PerfectHashCreate keys/example.keys out"


def test_create_executes_c_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    import perfecthash.cli as cli_module

    recorded_request: CreateRequest | None = None
    recorded_kwargs: dict[str, object] | None = None

    def fake_run_create_command(
        *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        nonlocal recorded_request
        nonlocal recorded_kwargs
        assert isinstance(args[0], CreateRequest)
        recorded_request = args[0]
        recorded_kwargs = dict(kwargs)
        return subprocess.CompletedProcess(args=["fake"], returncode=0)

    monkeypatch.setattr(cli_module, "run_create_command", fake_run_create_command)

    result = runner.invoke(
        app,
        [
            "create",
            "keys/example.keys",
            "out",
            "--hash-function",
            "MultiplyShiftR",
            "--create-binary",
            str(Path("/tmp/PerfectHashCreate")),
        ],
    )

    assert result.exit_code == 0
    assert recorded_request is not None
    assert recorded_kwargs is not None
    request = recorded_request
    assert request.keys_path == Path("keys/example.keys")
    assert request.output_dir == Path("out")
    assert request.hash_function.value == "MultiplyShiftR"
    assert recorded_kwargs == {
        "create_binary": Path("/tmp/PerfectHashCreate"),
    }


def test_bulk_create_emit_c_argv_preserves_hash_function_name() -> None:
    result = runner.invoke(
        app,
        [
            "bulk-create",
            "keys",
            "out",
            "--hash-function",
            "MultiplyShiftR",
            "--quiet",
            "--skip-test-after-create",
            "--emit-c-argv",
        ],
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == (
        "PerfectHashBulkCreate keys out Chm01 MultiplyShiftR And 0 "
        "--SkipTestAfterCreate --Quiet"
    )


def test_bulk_create_dry_run_renders_resolved_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import perfecthash.cli as cli_module

    def fake_render_bulk_create_command(
        *args: object,
        **kwargs: object,
    ) -> list[str]:
        del args, kwargs
        return ["/tmp/PerfectHashBulkCreate", "keys", "out"]

    monkeypatch.setattr(
        cli_module,
        "render_bulk_create_command",
        fake_render_bulk_create_command,
    )

    result = runner.invoke(
        app,
        [
            "bulk-create",
            "keys",
            "out",
            "--hash-function",
            "MultiplyShiftR",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "/tmp/PerfectHashBulkCreate keys out"


def test_bulk_create_executes_c_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    import perfecthash.cli as cli_module

    recorded_request: BulkCreateRequest | None = None
    recorded_kwargs: dict[str, object] | None = None

    def fake_run_bulk_create_command(
        *args: object,
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        nonlocal recorded_request
        nonlocal recorded_kwargs
        assert isinstance(args[0], BulkCreateRequest)
        recorded_request = args[0]
        recorded_kwargs = dict(kwargs)
        return subprocess.CompletedProcess(args=["fake"], returncode=0)

    monkeypatch.setattr(
        cli_module,
        "run_bulk_create_command",
        fake_run_bulk_create_command,
    )

    result = runner.invoke(
        app,
        [
            "bulk-create",
            "keys",
            "out",
            "--hash-function",
            "MultiplyShiftR",
            "--bulk-create-binary",
            str(Path("/tmp/PerfectHashBulkCreate")),
        ],
    )

    assert result.exit_code == 0
    assert recorded_request is not None
    assert recorded_kwargs is not None
    request = recorded_request
    assert request.keys_dir == Path("keys")
    assert request.output_dir == Path("out")
    assert request.hash_function.value == "MultiplyShiftR"
    assert recorded_kwargs == {
        "bulk_create_binary": Path("/tmp/PerfectHashBulkCreate"),
    }
