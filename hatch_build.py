from __future__ import annotations

import os
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from packaging.tags import platform_tags


def _shared_library_suffixes() -> tuple[str, ...]:
    return (".so", ".dylib", ".dll")


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        del version

        if self.target_name != "wheel":
            return

        native_root_value = self.config.get("native-root") or os.environ.get(
            "PERFECTHASH_PYTHON_NATIVE_ROOT"
        )
        if not native_root_value:
            return

        native_root = Path(native_root_value).expanduser().resolve()
        if not native_root.is_dir():
            raise RuntimeError(f"PerfectHash native root does not exist: {native_root}")

        force_include = dict(build_data.get("force_include", {}))
        binary_dir = native_root / "bin"
        library_dir = native_root / "lib"

        if binary_dir.is_dir():
            for path in sorted(binary_dir.iterdir()):
                if not path.is_file():
                    continue
                force_include[str(path)] = f"perfecthash/_native/bin/{path.name}"

        if library_dir.is_dir():
            for path in sorted(library_dir.iterdir()):
                if not path.is_file():
                    continue
                if path.suffix not in _shared_library_suffixes():
                    continue
                force_include[str(path)] = f"perfecthash/_native/lib/{path.name}"

        build_data["force_include"] = force_include
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{next(platform_tags())}"
