# Offline Build + Disassembly Workflow

Use the offline codegen path to produce a compiler-emitted Index() routine, then
inspect the `.so` to compare with RawDog assembly (useful for AVX2/AVX-512).

## Generate an offline project

1) Run the create exe to generate a project:

```bash
./build/bin/Release/PerfectHashCreateExe \
  keys/HologramWorld-31016.keys /tmp/ph-out \
  Chm01 Mulshrolate1RX And 1 --Quiet
```

2) Find the generated project directory and project name:

```bash
rg -n "project\(" /tmp/ph-out -g 'CMakeLists.txt'
```

## Build the generated library

```bash
cmake -S /tmp/ph-out/<generated-dir> -B /tmp/ph-out/_build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/ph-out/_build
```

The resulting shared library is usually under `/tmp/ph-out/_build` with the
project name (e.g., `lib<TableName>.so`).

## Disassemble the index routine

```bash
objdump -d -M intel /tmp/ph-out/_build/lib<TableName>.so | rg -n "Index|Mulshrolate"
```

If `objdump` is too noisy, try `llvm-objdump` and restrict to the symbol:

```bash
llvm-objdump -d --no-show-raw-insn -M intel \
  --disassemble-symbols=PerfectHashTableIndex \
  /tmp/ph-out/_build/lib<TableName>.so
```

Notes:
- Use the output as a guide, not a source of truth. The online JIT logic in
  `src/PerfectHash/ChmOnline01.c` is authoritative.
- If you need a non-inlined Index, ensure the generated project is built in
  Release and confirm the symbol exists before disassembling.
