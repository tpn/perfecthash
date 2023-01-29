# Perfect Hash

<img src="https://ci.appveyor.com/api/projects/status/github/tpn/perfecthash?svg=true&retina=true" alt="Appveyor Badge">

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
routine, which clocks in at about 6 cycles in practice, boils down to something
like this (`Seed1`, `Seed2`, `Shift`, and `IndexMask` will all be literal
constants in the final source code, not variables):

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
```

The IACA profile is as follows:

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
