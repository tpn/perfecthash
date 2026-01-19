# Perfect Hash

<!-- Disable this whilst cuda-dev2 is in flux. 
<img src="https://ci.appveyor.com/api/projects/status/github/tpn/perfecthash?svg=true&retina=true" alt="Appveyor Badge">
-->
[![macos](https://github.com/tpn/perfecthash/actions/workflows/macos.yml/badge.svg)](https://github.com/tpn/perfecthash/actions/workflows/macos.yml)

[Helper Utility for Generating Command Line Syntax](https://tpn.github.io/perfecthash-ui/)
The `ui/` directory contains the companion web UI (submodule) for generating
command line syntax. See [ui/README.md](ui/README.md) for details.


## Overview

This project is a library for creating perfect hash tables from 32-bit key sets.
It is based on the acyclic random 2-part hypergraph algorithm.  Its
primary goal is finding the fastest possible runtime solution.

It is geared toward offline table generation: a command line application is used
to generate a small C library that implements an `Index()` routine, which, given
an input key, will return the order-preserved index of that key within the
original key set, e.g.:

```c
    uint32_t ix;
    uint32_t key = 0x20190903;

    ix = Index(key);
```

This allows the efficient implementation of key-value tables, e.g.:

```c
    extern uint32_t Table[];

    uint32_t
    Lookup(uint32_t Key)
    {
        return Table[Index(Key)]
    };

    void
    Insert(uint32_t Key, uint32_t Value)
    {
        Table[Index(Key)] = Value;
    }

    void
    Delete(uint32_t Key)
    {
        Table[Index(Key)] = 0;
    }

```

The fastest `Index()` routine is the [MultiplyShiftRX](https://github.com/tpn/perfecthash/blob/main/src/CompiledPerfectHashTable/CompiledPerfectHashTableChm01IndexMultiplyShiftRXAnd.c)
routine, which clocks in at about 6 cycles in practice, and boils down to
something like this:

```c
extern uint32_t Assigned[];

uint32_t
Index(uint32_t Key)
{
    uint32_t Vertex1;
    uint32_t Vertex2;

    Vertex1 = ((Key * Seed1) >> Shift);
    Vertex2 = ((Key * Seed2) >> Shift);

    return ((Assigned[Vertex1] + Assigned[Vertex2]) & IndexMask);
}
```
N.B. `Seed1`, `Seed2`, `Shift`, and `IndexMask` will all be literal
constants in the final source code, not variables.

This compiles down to:
```assembly
lea     r8, ptr [rip+0x23fc0]
imul    edx, ecx, 0xe8d9cdf9
shr     rdx, 0x10
movzx   eax, word ptr [r8+rdx*2]
imul    edx, ecx, 0xc2e3c0b7
shr     rdx, 0x10
movzx   ecx, word ptr [r8+rdx*2]
add     eax, ecx
ret
```

The IACA profile reports 8 uops:

```
Intel(R) Architecture Code Analyzer Version -  v3.0-28-g1ba2cbb build date: 2017-10-23;17:30:24
Analyzed File -  .\x64\Release\HologramWorld_31016_Chm01_MultiplyShiftRX_And.dll
Binary Format - 64Bit
Architecture  -  SKL
Analysis Type - Throughput

Throughput Analysis Report
--------------------------
Block Throughput: 10.00 Cycles       Throughput Bottleneck: Backend
Loop Count:  26
Port Binding In Cycles Per Iteration:
----------------------------------------------------------------------------
| Port   |  0  - DV  |  1  |  2  - D   |  3  - D   |  4  |  5  |  6  |  7  |
----------------------------------------------------------------------------
| Cycles | 1.3   0.0 | 2.0 | 1.0   1.0 | 1.0   1.0 | 0.0 | 1.3 | 1.3 | 0.0 |
----------------------------------------------------------------------------

| # Of |         Ports pressure in cycles                     |
| Uops |0 - DV | 1   | 2 - D   | 3 - D    | 4 | 5   | 6   | 7 |
---------------------------------------------------------------
|  1   |       |     |         |          |   | 1.0 |     |   | lea r8, ptr [rip+0x23fc0]
|  1   |       | 1.0 |         |          |   |     |     |   | imul edx, ecx, 0xe8d9cdf9
|  1   | 0.3   |     |         |          |   |     | 0.7 |   | shr rdx, 0x10
|  1   |       |     | 1.0 1.0 |          |   |     |     |   | movzx eax, word ptr [r8+rdx*2]
|  1   |       | 1.0 |         |          |   |     |     |   | imul edx, ecx, 0xc2e3c0b7
|  1   | 0.7   |     |         |          |   |     | 0.3 |   | shr rdx, 0x10
|  1   |       |     |         | 1.0  1.0 |   |     |     |   | movzx ecx, word ptr [r8+rdx*2]
|  1   | 0.3   |     |         |          |   | 0.3 | 0.3 |   | add eax, ecx
Total Num of Uops: 8
```

The "cost" behind the perfect hash table is the `Assigned` array.  The size of
this array will be the number of keys, rounded up to a power of two, and then
doubled.  E.g. `HologramWorld-31016.keys` has 31,016 keys.  Rounded up to a
power of two is 32,768, then doubled: 65,336.

The data type used by the `Assigned` array is the smallest C data type that
can hold the number of keys rounded up to a power of two.  Thus, a 16-bit
`unsigned short int` can be used for the `HologramWorld-31016.keys` array:

```c
unsigned short int Assigned[65336] = { ... };
```
Thus, `sizeof(Assigned)` will be 131,072, or 128KB.

The `Index()` routine will perform two memory lookups into this array per call.
No pointer chasing or indirection is required.  The most frequent keys will have
both locations in L1 cache; the worst-case scenario is two memory lookups for
both locations for cold or infrequent keys.

## Quick Guide

Initially, all development on this project was done exclusively on Windows, with
Visual Studio 2022.  Linux support has recently been added, although it is still
quite rough around the edges.  For some context: about 1,700 hours have been
spent on the Windows portion.  The Linux support was hacked together in about
two weeks of evening and weekend work.

The generated compiled perfect hash tables are cross-platform, and will work on
Windows, Mac, Linux, x86, x64, and ARM64.

### Building
#### Windows

```
mkdir c:\src
cd src
git clone https://github.com/tpn/perfecthash
git clone https://github.com/tpn/perfecthash-keys

cd perfecthash/src
```
The `PerfectHash.sln` file lives in `perfecthash/src`.  You can either build
this directly via Visual Studio, use one of the `build-*.bat` files, or just
use `msbuild` from a Visual Studio 2022 command prompt:

```
msbuild /nologo /m /t:Rebuild /p:Configuration=Release;Platform=x64
```

You can also download the latest binaries from the [Releases](https://github.com/tpn/perfecthash/releases/)
page.  The `PGO` zip files refer to profile-guided optimization builds, and are
generally faster than the `Release` builds by up to 30-40%.

Once built or downloaded, there are two main command line executables:
`PerfectHashCreate.exe`, and `PerfectHashBulkCreate.exe`.  The former is for
creating a single table, and it takes a single input key file.  The latter
can be pointed at a directory of keys, and it will create tables for all of
them.

#### Linux

Prerequisites: C compiler (GCC 10 tested), CMake.  Optional: Ninja.

Recommended (mamba/conda) environment:

```
# x86_64 (pre-generated)
mamba env create -f conda/environments/dev-linux_os-linux_arch-x86_64_py-313_cuda-none_compiler-llvm.yaml
mamba activate dev-linux_os-linux_arch-x86_64_py-313_cuda-none_compiler-llvm

# ARM64 / aarch64 (pre-generated)
mamba env create -f conda/environments/dev-linux-arm64_os-linux_arch-aarch64_py-313_cuda-none_compiler-llvm.yaml
mamba activate dev-linux-arm64_os-linux_arch-aarch64_py-313_cuda-none_compiler-llvm
```

Generated environment files live under `conda/environments/` and are produced
from `dependencies.yaml` using `rapids-dependency-file-generator`.

If you need a different Python version, pick the matching `py-314` environment
file from `conda/environments/`.  You can also create a minimal dev/test
environment manually:

```
mamba create -y -n perfecthash-dev -c conda-forge \
  python=3.12 rust cmake ninja make pkg-config clang clangxx lld llvmdev pytest
mamba activate perfecthash-dev
```

```
mkdir -p ~/src && cd ~/src
git clone https://github.com/tpn/perfecthash
git clone https://github.com/tpn/perfecthash-keys

cd perfecthash
cmake -S . -B build -G"Ninja Multi-Config"
cmake --build build --config Release
cmake --build build --config Debug
```
Note: the default build enables `-march=native` for required SIMD intrinsics.
Use `-DPERFECTHASH_ENABLE_NATIVE_ARCH=OFF` if you need a portable binary.

CUDA build (Ninja Multi-Config):

```
cmake -S . -B build-cuda -G"Ninja Multi-Config" \
    -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda --config Release
```
CUDA builds require CUDAToolkit on PATH. Set
`CMAKE_CUDA_ARCHITECTURES` to your GPU (e.g., 86, 89, 90).
For normal Makefile support:

```
cd perfecthash
cmake -S . -B build -G"Unix Makefiles"
cmake --build build
```

#### Tests (Linux/macOS)

Unit tests and CLI integration tests are available via CTest.  They require a
build with tests enabled and use the `keys/HologramWorld-31016.keys` fixture.
The codegen tests require `cargo` (from the Rust toolchain) to be available.

```
cmake -S . -B build-tests -G Ninja -DPERFECTHASH_ENABLE_TESTS=ON -DBUILD_TESTING=ON
cmake --build build-tests
ctest --test-dir build-tests --output-on-failure
```

#### Mac

Prerequisites: Xcode Command Line Tools, CMake, Ninja. Optional: mamba/conda.

Recommended (mamba/conda) environment:

```
# Apple Silicon (arm64)
mamba env create -f conda/environments/dev-macos_os-macos_arch-arm64_py-313_cuda-none_compiler-llvm.yaml
mamba activate dev-macos_os-macos_arch-arm64_py-313_cuda-none_compiler-llvm

# Intel (x86_64)
mamba env create -f conda/environments/dev-macos_os-macos_arch-x86_64_py-313_cuda-none_compiler-llvm.yaml
mamba activate dev-macos_os-macos_arch-x86_64_py-313_cuda-none_compiler-llvm
```

```
mkdir -p ~/src && cd ~/src
git clone https://github.com/tpn/perfecthash
git clone https://github.com/tpn/perfecthash-keys

cd perfecthash
cmake -S . -B build-macos -G"Ninja Multi-Config"
cmake --build build-macos --config Release
cmake --build build-macos --config Debug
```

For Intel macOS CI (or cross-build on Apple Silicon), you can use the preset:

```
cmake --preset ninja-multi-macos-x86_64
cmake --build --preset ninja-macos-x86_64-release
```

Tests:

```
ctest --test-dir build-macos --output-on-failure -C Release
```

### Usage

The usage options are almost identical for both programs.  If you run either one
without arguments, it will print detailed usage instructions, also available
[here](https://github.com/tpn/perfecthash/blob/main/USAGE.txt).

The main usage follows:

```
PerfectHashBulkCreate.exe Usage:
    <KeysDirectory> <OutputDirectory>
    <Algorithm> <HashFunction> <MaskFunction>
    <MaximumConcurrency>
    [BulkCreateFlags] [KeysLoadFlags] [TableCreateFlags]
    [TableCompileFlags] [TableCreateParameters]

PerfectHashCreate.exe Usage:
    <KeysPath> <OutputDirectory>
    <Algorithm> <HashFunction> <MaskFunction>
    <MaximumConcurrency>
    [CreateFlags] [KeysLoadFlags] [TableCreateFlags]
    [TableCompileFlags] [TableCreateParameters]
```

Assuming you have built a `Release` version of the library, from a Visual Studio
2022 command prompt (i.e. so the compiler is available in your `PATH`):

```
mkdir c:\Temp\ph.out
cd c:\src\perfecthash\src
..\bin\timemem.exe x64\Release\PerfectHashCreate.exe c:\src\perfecthash\keys\HologramWorld-31016.keys c:\Temp\ph.out Chm01 MultiplyShiftR And 0 --Compile
```

On Linux this would look like:
```
mkdir -p ~/tmp/ph.out
cd ~/src/perfecthash/src
time ../x64/Release/PerfectHashCreateExe $HOME/src/perfecthash-keys/sys32/HologramWorld-31016.keys ~/tmp/ph.out Chm01 MultiplyShiftR And 0 --DisableCsvOutputFile
```

This should result in some output that looks like this:
```
c:\src\perfecthash\src>..\bin\timemem x64\Release\PerfectHashCreate.exe c:\src\perfecthash\keys\HologramWorld-31016.keys c:\Temp\ph.out Chm01 MultiplyShiftR And 0 --Compile

Keys File Name:                                    HologramWorld-31016.keys
Number of Keys:                                    31016
Number of Table Resize Events:                     0
Keys to Edges Ratio:                               0.946533203125
Duration:                                          0 hours, 0 mins, 0 secs
Duration Since Last Best Graph:
Attempts:                                          8633
Attempts Per Second:                               53956.250000000
Current Attempts:                                  8633
Current Attempts Per Second:                       53956.250000000
Successful Attempts:                               1
Failed Attempts:                                   8625
First Attempt Solved:                              0
Most Recent Attempt Solved:                        8562
Predicted Attempts to Solve:                       8634
Predicted Attempts Remaining until next Solve:     8563
Estimated Seconds until next Solve:                0.15870265261206998
New Best Graph Count:                              0
Equal Best Graph Count:                            0
Solutions Found Ratio:                             0.00011583458820803892
Vertex Collision Failures:                         4294
Cyclic Graph Failures:                             4331
Vertex Collision to Cyclic Graph Failure Ratio:    0.99145693835142
Highest Deleted Edges Count:                       31008
[r] Refresh [f] Finish [e] Resize [c] Toggle Callback [?] More Help
.
Exit code      : 0
Elapsed time   : 2.89
Kernel time    : 0.00 (0.0%)
User time      : 2.97 (102.7%)
page fault #   : 6504
Working set    : 24164 KB
Paged pool     : 167 KB
Non-paged pool : 52 KB
Page file size : 32280 KB
```
N.B. Console output isn't supported on Linux yet.

N.B. If you get an error like this, it means `msbuild` couldn't be found on
your path; make sure to launch a Visual Studio 2022 command prompt:

```
C:\src\perfecthash\src\PerfectHash\PerfectHashTableCompile.c: 217: CreateProcessW failed with error: 2 (0x2).  The system cannot find the file specified.
C:\src\perfecthash\src\PerfectHash\PerfectHashContextTableCreate.c: 492: PerfectHashTableCompile failed with error: 3758359076 (0xe0040224).  System call failed.
```

If you look in `C:\Temp\ph.out`, you'll see something along the following lines.
Intermediate build files have been omitted for brevity.

```
C:\Temp\ph.out>tree /f
Folder PATH listing for volume Windows
Volume serial number is 2490-4F03
C:.
│   CompiledPerfectHash.h
│   CompiledPerfectHash.props
│   CompiledPerfectHashMacroGlue.h
│   no_sal2.h
│   PerfectHashTableCreate_10A0ED40.csv
│
├───HologramWorld_31016_Chm01_MultiplyShiftR_And
│       HologramWorld_31016_Chm01_MultiplyShiftR_And.c
│       HologramWorld_31016_Chm01_MultiplyShiftR_And.def
│       HologramWorld_31016_Chm01_MultiplyShiftR_And.h
│       HologramWorld_31016_Chm01_MultiplyShiftR_And.pht1
│       HologramWorld_31016_Chm01_MultiplyShiftR_And.sln
│       HologramWorld_31016_Chm01_MultiplyShiftR_And_BenchmarkFull.c
│       HologramWorld_31016_Chm01_MultiplyShiftR_And_BenchmarkFull.mk
│       HologramWorld_31016_Chm01_MultiplyShiftR_And_BenchmarkFullExe.c
│       HologramWorld_31016_Chm01_MultiplyShiftR_And_BenchmarkFullExe.vcxproj
│       HologramWorld_31016_Chm01_MultiplyShiftR_And_BenchmarkIndex.c
│       HologramWorld_31016_Chm01_MultiplyShiftR_And_BenchmarkIndex.mk
│       HologramWorld_31016_Chm01_MultiplyShiftR_And_BenchmarkIndexExe.c
│       HologramWorld_31016_Chm01_MultiplyShiftR_And_BenchmarkIndexExe.vcxproj
│       HologramWorld_31016_Chm01_MultiplyShiftR_And_Build.bat
│       HologramWorld_31016_Chm01_MultiplyShiftR_And_Dll.vcxproj
│       HologramWorld_31016_Chm01_MultiplyShiftR_And_Keys.c
│       HologramWorld_31016_Chm01_MultiplyShiftR_And_Lib.mk
│       HologramWorld_31016_Chm01_MultiplyShiftR_And_So.mk
│       HologramWorld_31016_Chm01_MultiplyShiftR_And_StdAfx.c
│       HologramWorld_31016_Chm01_MultiplyShiftR_And_StdAfx.h
│       HologramWorld_31016_Chm01_MultiplyShiftR_And_Support.c
│       HologramWorld_31016_Chm01_MultiplyShiftR_And_Support.h
│       HologramWorld_31016_Chm01_MultiplyShiftR_And_TableData.c
│       HologramWorld_31016_Chm01_MultiplyShiftR_And_TableValues.c
│       HologramWorld_31016_Chm01_MultiplyShiftR_And_Test.c
│       HologramWorld_31016_Chm01_MultiplyShiftR_And_Test.mk
│       HologramWorld_31016_Chm01_MultiplyShiftR_And_TestExe.c
│       HologramWorld_31016_Chm01_MultiplyShiftR_And_TestExe.vcxproj
│       HologramWorld_31016_Chm01_MultiplyShiftR_And_Types.h
│       main.mk
│       Makefile
│
└───x64
    └───Release
            BenchmarkFull_HologramWorld_31016_Chm01_MultiplyShiftR_And.exe
            BenchmarkFull_HologramWorld_31016_Chm01_MultiplyShiftR_And.pdb
            BenchmarkIndex_HologramWorld_31016_Chm01_MultiplyShiftR_And.exe
            BenchmarkIndex_HologramWorld_31016_Chm01_MultiplyShiftR_And.pdb
            HologramWorld_31016_Chm01_MultiplyShiftR_And.dll
            HologramWorld_31016_Chm01_MultiplyShiftR_And.lib
            HologramWorld_31016_Chm01_MultiplyShiftR_And.pdb
            Test_HologramWorld_31016_Chm01_MultiplyShiftR_And.exe
            Test_HologramWorld_31016_Chm01_MultiplyShiftR_And.pdb
```

The main library implementing the perfect hash table is the `.dll` file.  There
are three helper `.exe` utilities also compiled for benchmarking and testing.
You can assess the performance of just the `Index()` routine via:
`BenchmarkIndex_HologramWorld_31016_Chm01_MultiplyShiftR_And.exe`:

```
cd /d c:\Temp\ph.out\x64\Release
c:\src\perfecthash\bin\timemem.exe BenchmarkIndex_HologramWorld_31016_Chm01_MultiplyShiftR_And.exe
```

On Windows, the output will look like this:
```
C:\Temp\ph.out\x64\Release>c:\src\perfecthash\bin\timemem.exe BenchmarkIndex_HologramWorld_31016_Chm01_MultiplyShiftR_And.exe
Exit code      : 7106
Elapsed time   : 0.01
Kernel time    : 0.00 (0.0%)
User time      : 0.00 (0.0%)
page fault #   : 849
Working set    : 3220 KB
Paged pool     : 22 KB
Non-paged pool : 5 KB
Page file size : 520 KB
```

The exit code, 7106 in this case, is the minimum number of cycles it took, out
of 1000 attempts, to do 1000 calls to the `Index()` routine.  So, you can
divide by 1000 to get the approximate number of cycles per call, in this case,
about 7.

The `BenchmarkFull` executable returns the minimum number of cycles it took, out
of 100 attempts, to do 10 iterations of the following:

- For each key, call `Insert(Key, RotateLeft(Key, 15))`.
- For each key, call `Value = Lookup(Key)`.
- For each key, call `Previous = Delete(Key)`.

```
C:\Temp\ph.out\x64\Release>c:\src\perfecthash\bin\timemem.exe BenchmarkFull_HologramWorld_31016_Chm01_MultiplyShiftR_And.exe
Exit code      : 7321346
Elapsed time   : 0.28
Kernel time    : 0.00 (0.0%)
User time      : 0.16 (55.7%)
page fault #   : 1036
Working set    : 3832 KB
Paged pool     : 31 KB
Non-paged pool : 5 KB
Page file size : 564 KB
```

The `.dll` files are compiled with special `IACA` versions of each `Index()`
routine (i.e. `IndexIaca()`), so you can call `iaca.exe` on them to get an
analysis of the generated code, e.g.:

```
C:\Temp\ph.out\x64\Release>c:\src\perfecthash\bin\iaca.exe HologramWorld_31016_Chm01_MultiplyShiftR_And.dll
Intel(R) Architecture Code Analyzer Version -  v3.0-28-g1ba2cbb build date: 2017-10-23;17:30:24
Analyzed File -  HologramWorld_31016_Chm01_MultiplyShiftR_And.dll
Binary Format - 64Bit
Architecture  -  SKL
Analysis Type - Throughput

Throughput Analysis Report
--------------------------
Block Throughput: 8.93 Cycles       Throughput Bottleneck: Backend
Loop Count:  30
Port Binding In Cycles Per Iteration:
--------------------------------------------------------------------------------------------------
|  Port  |   0   -  DV   |   1   |   2   -  D    |   3   -  D    |   4   |   5   |   6   |   7   |
--------------------------------------------------------------------------------------------------
| Cycles |  1.0     0.0  |  1.0  |  1.0     1.0  |  1.0     1.0  |  0.0  |  1.0  |  1.0  |  0.0  |
--------------------------------------------------------------------------------------------------

DV - Divider pipe (on port 0)
D - Data fetch pipe (on ports 2 and 3)
F - Macro Fusion with the previous instruction occurred
* - instruction micro-ops not bound to a port
^ - Micro Fusion occurred
# - ESP Tracking sync uop was issued
@ - SSE instruction followed an AVX256/AVX512 instruction, dozens of cycles penalty is expected
X - instruction not supported, was not accounted in Analysis

| Num Of   |                    Ports pressure in cycles                         |      |
|  Uops    |  0  - DV    |  1   |  2  -  D    |  3  -  D    |  4   |  5   |  6   |  7   |
-----------------------------------------------------------------------------------------
|   1      |             | 1.0  |             |             |      |      |      |      | imul ecx, ecx, 0xff8d672d
|   1      | 1.0         |      |             |             |      |      |      |      | shr rax, 0x9
|   1*     |             |      |             |             |      |      |      |      | movzx edx, ax
|   1      |             |      |             |             |      |      | 1.0  |      | shr rcx, 0xd
|   1      |             |      | 1.0     1.0 |             |      |      |      |      | movzx eax, word ptr [r8+rdx*2]
|   1*     |             |      |             |             |      |      |      |      | movzx edx, cx
|   1      |             |      |             | 1.0     1.0 |      |      |      |      | movzx ecx, word ptr [r8+rdx*2]
|   1      |             |      |             |             |      | 1.0  |      |      | add eax, ecx
Total Num Of Uops: 8
Analysis Notes:
Backend allocation was stalled due to unavailable allocation resources.
```

Although note that this isn't an exact science, sometimes the compiler
reorders the `IACA_VC_START()` and `IACA_VC_END()` markers such that you end up
missing a couple of instructions in the analysis.  In the example above, the
actual assembly for the `Index()` routine is as follows:

```
C:\Temp\ph.out\x64\Release>dumpbin /disasm HologramWorld_31016_Chm01_MultiplyShiftR_And.dll
Microsoft (R) COFF/PE Dumper Version 14.34.31935.0
Copyright (C) Microsoft Corporation.  All rights reserved.


Dump of file HologramWorld_31016_Chm01_MultiplyShiftR_And.dll

File Type: DLL

CompiledPerfectHash_HologramWorld_31016_Chm01_MultiplyShiftR_And_Index:
  0000000180001000: 69 C1 DF 0E AD FF  imul        eax,ecx,0FFAD0EDFh
  0000000180001006: 4C 8D 05 F3 3F 02  lea         r8,[HologramWorld_31016_Chm01_MultiplyShiftR_And_TableData]
                    00
  000000018000100D: 69 C9 2D 67 8D FF  imul        ecx,ecx,0FF8D672Dh
  0000000180001013: 48 C1 E8 09        shr         rax,9
  0000000180001017: 0F B7 D0           movzx       edx,ax
  000000018000101A: 48 C1 E9 0D        shr         rcx,0Dh
  000000018000101E: 41 0F B7 04 50     movzx       eax,word ptr [r8+rdx*2]
  0000000180001023: 0F B7 D1           movzx       edx,cx
  0000000180001026: 41 0F B7 0C 50     movzx       ecx,word ptr [r8+rdx*2]
  000000018000102B: 03 C1              add         eax,ecx
  000000018000102D: 25 FF 7F 00 00     and         eax,7FFFh
  0000000180001032: C3                 ret
```

### Linux Compilation
To compile the hash table on Linux (using WSL1 and GCC 9 as an example):
```
% cd /mnt/c/Temp/ph.out/HologramWorld_31016_Chm01_MultiplyShiftR_And
% make
% export LD_LIBRARY_PATH=.
% ./BenchmarkIndex_HologramWorld_31016_Chm01_MultiplyShiftR_And
8094
```

With clang (version 10):
```
% make clean
% CC=clang make
% ./BenchmarkIndex_HologramWorld_31016_Chm01_MultiplyShiftR_And
7068
```

### Mac Compilation
Identical to Linux, except you don't need `export LD_LIBRARY_PATH=.`:
```
% make
% ./BenchmarkIndex_HologramWorld_31016_Chm01_MultiplyShiftR_And
```
