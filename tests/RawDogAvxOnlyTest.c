#include "stdafx.h"
#include <PerfectHash/PerfectHashOnlineJit.h>
#include "PerfectHashOnlineJitPrivate.h"

// Exercise AVX-only dispatch on hosts that also support AVX2 or AVX-512.
// The test owns the table and compiles it synchronously.
int32_t
CompileRawDogX4WithoutAvx2(
    PH_ONLINE_JIT_CONTEXT *Context,
    PH_ONLINE_JIT_TABLE *Table,
    PH_ONLINE_JIT_MAX_ISA MaxIsa
    )
{
    PRTL_CPU_FEATURES Features = &Table->Table->Rtl->CpuFeatures;
    BOOLEAN Avx2 = Features->AVX2;
    BOOLEAN Avx512F = Features->AVX512F;
    int32_t Result;

    Features->AVX2 = FALSE;
    Features->AVX512F = FALSE;
    Result = PhOnlineJitCompileTableEx(
        Context, Table, PhOnlineJitBackendRawDogJit, 4, MaxIsa,
        PH_ONLINE_JIT_COMPILE_FLAG_STRICT_VECTOR_WIDTH, NULL, NULL
    );
    Features->AVX2 = Avx2;
    Features->AVX512F = Avx512F;
    return Result;
}
