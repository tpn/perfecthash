#include "stdafx.h"
#include <PerfectHash/PerfectHashOnlineJit.h>
#include "PerfectHashOnlineJitPrivate.h"

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("xsave")))
#endif
int
CanExecuteAvx512F(
    PH_ONLINE_JIT_TABLE *Table
    )
{
    PRTL_CPU_FEATURES Features = &Table->Table->Rtl->CpuFeatures;
    return Features->AVX512F && Features->OSXSAVE && (_xgetbv(0) & 0xe6) == 0xe6;
}

// Mask the optional extensions beyond the x4 or x16 kernel's requirements.
// The test owns the table and compiles it synchronously.
int32_t
CompileRawDogWithMinimumFeatures(
    PH_ONLINE_JIT_CONTEXT *Context,
    PH_ONLINE_JIT_TABLE *Table,
    uint32_t Width,
    PH_ONLINE_JIT_MAX_ISA MaxIsa
    )
{
    PRTL_CPU_FEATURES Features = &Table->Table->Rtl->CpuFeatures;
    RTL_CPU_FEATURES_F7_EBX Saved = Features->F7Ebx;
    int32_t Result;

    if (Width == 4) {
        Features->AVX512F = FALSE;
    } else if (Width == 16) {
        Features->AVX512DQ = FALSE;
        Features->AVX512BW = FALSE;
        Features->AVX512VL = FALSE;
    } else {
        return E_INVALIDARG;
    }
    Features->AVX2 = FALSE;
    Result = PhOnlineJitCompileTableEx(
        Context, Table, PhOnlineJitBackendRawDogJit, Width, MaxIsa,
        PH_ONLINE_JIT_COMPILE_FLAG_STRICT_VECTOR_WIDTH, NULL, NULL
    );
    Features->F7Ebx = Saved;
    return Result;
}
