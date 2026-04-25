/*++

Copyright (c) 2026 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashOnlineJit.h

Abstract:

    This header exposes a minimal public API for runtime creation of
    32-bit and 64-bit perfect hash tables via online mode with RawDog JIT and
    LLVM JIT.

    The 64-bit wrapper path currently supports 64-bit inputs that downsize to
    a 32-bit runtime table representation. Full-width 64-bit JIT index support
    is not exposed via this minimal wrapper surface.

--*/

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if (defined(_WIN32) || defined(_WIN64)) && !defined(PH_COMPAT)
#if defined(PERFECT_HASH_ONLINE_JIT_BUILD)
#define PH_ONLINE_JIT_API __declspec(dllexport)
#elif defined(PERFECT_HASH_ONLINE_JIT_STATIC)
#define PH_ONLINE_JIT_API
#else
#define PH_ONLINE_JIT_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define PH_ONLINE_JIT_API __attribute__((visibility("default")))
#else
#define PH_ONLINE_JIT_API
#endif

typedef struct PH_ONLINE_JIT_CONTEXT PH_ONLINE_JIT_CONTEXT;
typedef struct PH_ONLINE_JIT_TABLE PH_ONLINE_JIT_TABLE;

typedef enum PH_ONLINE_JIT_HASH_FUNCTION {
    PhOnlineJitHashMultiplyShiftR = 0,
    PhOnlineJitHashMultiplyShiftLR,
    PhOnlineJitHashMultiplyShiftRMultiply,
    PhOnlineJitHashMultiplyShiftR2,
    PhOnlineJitHashMultiplyShiftRX,
    PhOnlineJitHashMulshrolate1RX,
    PhOnlineJitHashMulshrolate2RX,
    PhOnlineJitHashMulshrolate3RX,
    PhOnlineJitHashMulshrolate4RX,
} PH_ONLINE_JIT_HASH_FUNCTION;

typedef enum PH_ONLINE_JIT_BACKEND {
    PhOnlineJitBackendAuto = 0,
    PhOnlineJitBackendRawDogJit,
    PhOnlineJitBackendLlvmJit,
} PH_ONLINE_JIT_BACKEND;

typedef enum PH_ONLINE_JIT_MAX_ISA {
    PhOnlineJitMaxIsaAuto = 0,
    PhOnlineJitMaxIsaAvx,
    PhOnlineJitMaxIsaAvx2,
    PhOnlineJitMaxIsaAvx512,
    PhOnlineJitMaxIsaNeon,
    PhOnlineJitMaxIsaSve,
    PhOnlineJitMaxIsaSve2,
} PH_ONLINE_JIT_MAX_ISA;

#define PH_ONLINE_JIT_COMPILE_FLAG_STRICT_VECTOR_WIDTH (1u << 0)

PH_ONLINE_JIT_API
int32_t
PhOnlineJitOpen(
    PH_ONLINE_JIT_CONTEXT **ContextPointer
    );

PH_ONLINE_JIT_API
void
PhOnlineJitClose(
    PH_ONLINE_JIT_CONTEXT *Context
    );

PH_ONLINE_JIT_API
int32_t
PhOnlineJitCreateTable32(
    PH_ONLINE_JIT_CONTEXT *Context,
    PH_ONLINE_JIT_HASH_FUNCTION HashFunction,
    const uint32_t *Keys,
    uint64_t NumberOfKeys,
    PH_ONLINE_JIT_TABLE **TablePointer
    );

//
// Creates a downsized 64-bit table.  Hash functions in the curated "good" set
// are intentionally created with GraphImpl4 so assigned16/assigned32 JIT paths
// can use the compact-key backend.  Other accepted hash functions keep the
// default graph implementation.
//

PH_ONLINE_JIT_API
int32_t
PhOnlineJitCreateTable64(
    PH_ONLINE_JIT_CONTEXT *Context,
    PH_ONLINE_JIT_HASH_FUNCTION HashFunction,
    const uint64_t *Keys,
    uint64_t NumberOfKeys,
    PH_ONLINE_JIT_TABLE **TablePointer
    );

PH_ONLINE_JIT_API
int32_t
PhOnlineJitCompileTable(
    PH_ONLINE_JIT_CONTEXT *Context,
    PH_ONLINE_JIT_TABLE *Table,
    PH_ONLINE_JIT_BACKEND Backend,
    uint32_t VectorWidth,
    PH_ONLINE_JIT_MAX_ISA JitMaxIsa
    );

PH_ONLINE_JIT_API
int32_t
PhOnlineJitCompileTableEx(
    PH_ONLINE_JIT_CONTEXT *Context,
    PH_ONLINE_JIT_TABLE *Table,
    PH_ONLINE_JIT_BACKEND Backend,
    uint32_t VectorWidth,
    PH_ONLINE_JIT_MAX_ISA JitMaxIsa,
    uint32_t Flags,
    PH_ONLINE_JIT_BACKEND *EffectiveBackend,
    uint32_t *EffectiveVectorWidth
    );

PH_ONLINE_JIT_API
int32_t
PhOnlineJitIndex32(
    PH_ONLINE_JIT_TABLE *Table,
    uint32_t Key,
    uint32_t *Index
    );

PH_ONLINE_JIT_API
int32_t
PhOnlineJitIndex64(
    PH_ONLINE_JIT_TABLE *Table,
    uint64_t Key,
    uint32_t *Index
    );

PH_ONLINE_JIT_API
int32_t
PhOnlineJitIndex32x4(
    PH_ONLINE_JIT_TABLE *Table,
    const uint32_t *Keys,
    uint32_t *Indexes
    );

PH_ONLINE_JIT_API
int32_t
PhOnlineJitIndex32x8(
    PH_ONLINE_JIT_TABLE *Table,
    const uint32_t *Keys,
    uint32_t *Indexes
    );

PH_ONLINE_JIT_API
int32_t
PhOnlineJitIndex32x16(
    PH_ONLINE_JIT_TABLE *Table,
    const uint32_t *Keys,
    uint32_t *Indexes
    );

PH_ONLINE_JIT_API
void
PhOnlineJitReleaseTable(
    PH_ONLINE_JIT_TABLE *Table
    );

#ifdef __cplusplus
}
#endif
