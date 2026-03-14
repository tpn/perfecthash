from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt

from .hash_functions import GoodHashFunction


class CreateRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    keys_path: Path
    output_dir: Path
    hash_function: GoodHashFunction
    algorithm: Literal["Chm01"] = "Chm01"
    mask_function: Literal["And"] = "And"
    maximum_concurrency: NonNegativeInt = 0
    compile: bool = False
    disable_csv_output_file: bool = False
    do_not_try_use_hash16_impl: bool = False
    graph_impl: PositiveInt | None = None
    max_solve_time_in_seconds: PositiveInt | None = None


class BulkCreateRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    keys_dir: Path
    output_dir: Path
    hash_function: GoodHashFunction
    algorithm: Literal["Chm01"] = "Chm01"
    mask_function: Literal["And"] = "And"
    maximum_concurrency: NonNegativeInt = 0
    compile: bool = False
    skip_test_after_create: bool = False
    quiet: bool = False
    disable_csv_output_file: bool = False
    omit_csv_row_if_table_create_failed: bool = False
    omit_csv_row_if_table_create_succeeded: bool = False
    do_not_try_use_hash16_impl: bool = False
    graph_impl: PositiveInt | None = None
    max_solve_time_in_seconds: PositiveInt | None = None
