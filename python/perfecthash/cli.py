from __future__ import annotations

import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer

from ._version import __version__
from .c_argv import render_bulk_create_c_argv, render_create_c_argv
from .c_cli import (
    format_command,
    render_bulk_create_command,
    render_create_command,
    run_bulk_create_command,
    run_create_command,
)
from .hash_functions import GOOD_HASH_FUNCTION_NAMES, GoodHashFunction
from .models import BulkCreateRequest, CreateRequest

app = typer.Typer(
    add_completion=True,
    help="PerfectHash Python CLI.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if not value:
        return

    typer.echo(f"ph {__version__}")
    raise typer.Exit()


@app.callback()
def callback(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=_version_callback,
            help="Show the package version and exit.",
            is_eager=True,
        ),
    ] = None,
) -> None:
    del version


@app.command()
def create(
    keys_path: Annotated[
        Path,
        typer.Argument(help="Path to the input .keys file."),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(help="Directory for generated outputs."),
    ],
    hash_function: Annotated[
        GoodHashFunction,
        typer.Option(
            "--hash-function",
            case_sensitive=True,
            help=(
                "Curated supported hash function name. "
                f"Supported values: {', '.join(GOOD_HASH_FUNCTION_NAMES)}."
            ),
        ),
    ],
    maximum_concurrency: Annotated[
        int,
        typer.Option(
            "--maximum-concurrency",
            min=0,
            help="Maps to the C CLI MaximumConcurrency positional argument.",
        ),
    ] = 0,
    compile_: Annotated[
        bool,
        typer.Option(
            "--compile",
            help="Emit the C CLI --Compile flag.",
        ),
    ] = False,
    disable_csv_output_file: Annotated[
        bool,
        typer.Option(
            "--disable-csv-output-file",
            help="Emit the C CLI --DisableCsvOutputFile flag.",
        ),
    ] = False,
    do_not_try_use_hash16_impl: Annotated[
        bool,
        typer.Option(
            "--do-not-try-use-hash16-impl",
            help="Emit the C CLI --DoNotTryUseHash16Impl flag.",
        ),
    ] = False,
    graph_impl: Annotated[
        int | None,
        typer.Option(
            "--graph-impl",
            min=1,
            help="Emit the C CLI --GraphImpl=<N> parameter.",
        ),
    ] = None,
    max_solve_time_in_seconds: Annotated[
        int | None,
        typer.Option(
            "--max-solve-time-in-seconds",
            min=1,
            help="Emit the C CLI --MaxSolveTimeInSeconds=<N> parameter.",
        ),
    ] = None,
    emit_c_argv: Annotated[
        bool,
        typer.Option(
            "--emit-c-argv",
            help="Print the current C CLI translation and exit.",
        ),
    ] = False,
    create_binary: Annotated[
        Path | None,
        typer.Option(
            "--create-binary",
            help="Path to the offline PerfectHash create binary.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Print the resolved executable command and exit.",
        ),
    ] = False,
) -> None:
    """Generate an offline table via the existing C create workflow."""
    request = CreateRequest(
        keys_path=keys_path,
        output_dir=output_dir,
        hash_function=hash_function,
        maximum_concurrency=maximum_concurrency,
        compile=compile_,
        disable_csv_output_file=disable_csv_output_file,
        do_not_try_use_hash16_impl=do_not_try_use_hash16_impl,
        graph_impl=graph_impl,
        max_solve_time_in_seconds=max_solve_time_in_seconds,
    )

    if emit_c_argv:
        typer.echo(
            " ".join(shlex.quote(part) for part in render_create_c_argv(request))
        )
        raise typer.Exit()

    if dry_run:
        typer.echo(
            format_command(
                render_create_command(
                    request,
                    create_binary=create_binary,
                )
            )
        )
        raise typer.Exit()

    result = run_create_command(
        request,
        create_binary=create_binary,
    )
    raise typer.Exit(code=result.returncode)


@app.command("bulk-create")
def bulk_create(
    keys_dir: Annotated[
        Path,
        typer.Argument(help="Directory containing input .keys files."),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(help="Directory for generated outputs."),
    ],
    hash_function: Annotated[
        GoodHashFunction,
        typer.Option(
            "--hash-function",
            case_sensitive=True,
            help=(
                "Curated supported hash function name. "
                f"Supported values: {', '.join(GOOD_HASH_FUNCTION_NAMES)}."
            ),
        ),
    ],
    maximum_concurrency: Annotated[
        int,
        typer.Option(
            "--maximum-concurrency",
            min=0,
            help="Maps to the C CLI MaximumConcurrency positional argument.",
        ),
    ] = 0,
    compile_: Annotated[
        bool,
        typer.Option(
            "--compile",
            help="Emit the C CLI --Compile flag.",
        ),
    ] = False,
    skip_test_after_create: Annotated[
        bool,
        typer.Option(
            "--skip-test-after-create",
            help="Emit the C CLI --SkipTestAfterCreate flag.",
        ),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            help="Emit the C CLI --Quiet flag.",
        ),
    ] = False,
    disable_csv_output_file: Annotated[
        bool,
        typer.Option(
            "--disable-csv-output-file",
            help="Emit the C CLI --DisableCsvOutputFile flag.",
        ),
    ] = False,
    omit_csv_row_if_table_create_failed: Annotated[
        bool,
        typer.Option(
            "--omit-csv-row-if-table-create-failed",
            help="Emit the C CLI --OmitCsvRowIfTableCreateFailed flag.",
        ),
    ] = False,
    omit_csv_row_if_table_create_succeeded: Annotated[
        bool,
        typer.Option(
            "--omit-csv-row-if-table-create-succeeded",
            help="Emit the C CLI --OmitCsvRowIfTableCreateSucceeded flag.",
        ),
    ] = False,
    do_not_try_use_hash16_impl: Annotated[
        bool,
        typer.Option(
            "--do-not-try-use-hash16-impl",
            help="Emit the C CLI --DoNotTryUseHash16Impl flag.",
        ),
    ] = False,
    graph_impl: Annotated[
        int | None,
        typer.Option(
            "--graph-impl",
            min=1,
            help="Emit the C CLI --GraphImpl=<N> parameter.",
        ),
    ] = None,
    max_solve_time_in_seconds: Annotated[
        int | None,
        typer.Option(
            "--max-solve-time-in-seconds",
            min=1,
            help="Emit the C CLI --MaxSolveTimeInSeconds=<N> parameter.",
        ),
    ] = None,
    emit_c_argv: Annotated[
        bool,
        typer.Option(
            "--emit-c-argv",
            help="Print the current C CLI translation and exit.",
        ),
    ] = False,
    bulk_create_binary: Annotated[
        Path | None,
        typer.Option(
            "--bulk-create-binary",
            help="Path to the offline PerfectHash bulk-create binary.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Print the resolved executable command and exit.",
        ),
    ] = False,
) -> None:
    """Generate offline tables for a directory of keys via C bulk-create."""
    request = BulkCreateRequest(
        keys_dir=keys_dir,
        output_dir=output_dir,
        hash_function=hash_function,
        maximum_concurrency=maximum_concurrency,
        compile=compile_,
        skip_test_after_create=skip_test_after_create,
        quiet=quiet,
        disable_csv_output_file=disable_csv_output_file,
        omit_csv_row_if_table_create_failed=omit_csv_row_if_table_create_failed,
        omit_csv_row_if_table_create_succeeded=omit_csv_row_if_table_create_succeeded,
        do_not_try_use_hash16_impl=do_not_try_use_hash16_impl,
        graph_impl=graph_impl,
        max_solve_time_in_seconds=max_solve_time_in_seconds,
    )

    if emit_c_argv:
        typer.echo(
            " ".join(shlex.quote(part) for part in render_bulk_create_c_argv(request))
        )
        raise typer.Exit()

    if dry_run:
        typer.echo(
            format_command(
                render_bulk_create_command(
                    request,
                    bulk_create_binary=bulk_create_binary,
                )
            )
        )
        raise typer.Exit()

    result = run_bulk_create_command(
        request,
        bulk_create_binary=bulk_create_binary,
    )
    raise typer.Exit(code=result.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv) if argv is not None else None
    if args == []:
        args = ["--help"]

    result: object = app(
        prog_name="ph",
        args=args,
        standalone_mode=False,
    )

    if isinstance(result, int):
        return result

    return 0
