from __future__ import annotations

from .models import BulkCreateRequest, CreateRequest


def render_create_c_argv(request: CreateRequest) -> list[str]:
    argv = [
        "PerfectHashCreate",
        str(request.keys_path),
        str(request.output_dir),
        request.algorithm,
        request.hash_function.value,
        request.mask_function,
        str(request.maximum_concurrency),
    ]

    if request.compile:
        argv.append("--Compile")

    if request.disable_csv_output_file:
        argv.append("--DisableCsvOutputFile")

    if request.do_not_try_use_hash16_impl:
        argv.append("--DoNotTryUseHash16Impl")

    if request.graph_impl is not None:
        argv.append(f"--GraphImpl={request.graph_impl}")

    if request.max_solve_time_in_seconds is not None:
        argv.append(f"--MaxSolveTimeInSeconds={request.max_solve_time_in_seconds}")

    return argv


def render_bulk_create_c_argv(request: BulkCreateRequest) -> list[str]:
    argv = [
        "PerfectHashBulkCreate",
        str(request.keys_dir),
        str(request.output_dir),
        request.algorithm,
        request.hash_function.value,
        request.mask_function,
        str(request.maximum_concurrency),
    ]

    if request.compile:
        argv.append("--Compile")

    if request.skip_test_after_create:
        argv.append("--SkipTestAfterCreate")

    if request.quiet:
        argv.append("--Quiet")

    if request.disable_csv_output_file:
        argv.append("--DisableCsvOutputFile")

    if request.omit_csv_row_if_table_create_failed:
        argv.append("--OmitCsvRowIfTableCreateFailed")

    if request.omit_csv_row_if_table_create_succeeded:
        argv.append("--OmitCsvRowIfTableCreateSucceeded")

    if request.do_not_try_use_hash16_impl:
        argv.append("--DoNotTryUseHash16Impl")

    if request.graph_impl is not None:
        argv.append(f"--GraphImpl={request.graph_impl}")

    if request.max_solve_time_in_seconds is not None:
        argv.append(f"--MaxSolveTimeInSeconds={request.max_solve_time_in_seconds}")

    return argv
