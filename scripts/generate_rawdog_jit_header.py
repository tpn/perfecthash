#!/usr/bin/env python3
"""
Generate a C header that embeds a raw byte blob as a static array.
"""

from __future__ import annotations

import argparse
import pathlib


def emit_array(data: bytes, symbol: str) -> str:
    lines = [
        "//",
        "// Auto-generated. Do not edit.",
        "//",
        "#pragma once",
        "",
        f"static const unsigned char {symbol}[] = {{",
    ]

    for offset in range(0, len(data), 12):
        chunk = data[offset : offset + 12]
        hex_bytes = ", ".join(f"0x{b:02x}" for b in chunk)
        lines.append(f"    {hex_bytes},")

    lines.append("};")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--symbol", required=True)
    args = parser.parse_args()

    data = args.input.read_bytes()
    args.output.write_text(emit_array(data, args.symbol), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
