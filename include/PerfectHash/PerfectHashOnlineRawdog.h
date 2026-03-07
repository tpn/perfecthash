/*++

Copyright (c) 2026 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashOnlineRawdog.h

Abstract:

    This header exposes a minimal public API for runtime creation of
    32-bit perfect hash tables via online mode and RawDog JIT.

--*/

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if (defined(_WIN32) || defined(_WIN64)) && !defined(PH_COMPAT)
#if defined(PERFECT_HASH_ONLINE_RAWDOG_BUILD)
#define PH_ONLINE_RAWDOG_API __declspec(dllexport)
#elif defined(PERFECT_HASH_ONLINE_RAWDOG_STATIC)
#define PH_ONLINE_RAWDOG_API
#else
#define PH_ONLINE_RAWDOG_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define PH_ONLINE_RAWDOG_API __attribute__((visibility("default")))
#else
#define PH_ONLINE_RAWDOG_API
#endif

typedef struct PH_ONLINE_RAWDOG_CONTEXT PH_ONLINE_RAWDOG_CONTEXT;
typedef struct PH_ONLINE_RAWDOG_TABLE PH_ONLINE_RAWDOG_TABLE;

typedef enum PH_ONLINE_RAWDOG_HASH_FUNCTION {
    PhOnlineRawdogHashMultiplyShiftR = 0,
    PhOnlineRawdogHashMultiplyShiftLR,
    PhOnlineRawdogHashMultiplyShiftRMultiply,
    PhOnlineRawdogHashMultiplyShiftR2,
    PhOnlineRawdogHashMultiplyShiftRX,
    PhOnlineRawdogHashMulshrolate1RX,
    PhOnlineRawdogHashMulshrolate2RX,
    PhOnlineRawdogHashMulshrolate3RX,
    PhOnlineRawdogHashMulshrolate4RX,
} PH_ONLINE_RAWDOG_HASH_FUNCTION;

typedef enum PH_ONLINE_RAWDOG_JIT_MAX_ISA {
    PhOnlineRawdogJitMaxIsaAuto = 0,
    PhOnlineRawdogJitMaxIsaAvx,
    PhOnlineRawdogJitMaxIsaAvx2,
    PhOnlineRawdogJitMaxIsaAvx512,
    PhOnlineRawdogJitMaxIsaNeon,
    PhOnlineRawdogJitMaxIsaSve,
    PhOnlineRawdogJitMaxIsaSve2,
} PH_ONLINE_RAWDOG_JIT_MAX_ISA;

PH_ONLINE_RAWDOG_API
int32_t
PhOnlineRawdogOpen(
    PH_ONLINE_RAWDOG_CONTEXT **ContextPointer
    );

PH_ONLINE_RAWDOG_API
void
PhOnlineRawdogClose(
    PH_ONLINE_RAWDOG_CONTEXT *Context
    );

PH_ONLINE_RAWDOG_API
int32_t
PhOnlineRawdogCreateTable32(
    PH_ONLINE_RAWDOG_CONTEXT *Context,
    PH_ONLINE_RAWDOG_HASH_FUNCTION HashFunction,
    const uint32_t *Keys,
    uint64_t NumberOfKeys,
    PH_ONLINE_RAWDOG_TABLE **TablePointer
    );

PH_ONLINE_RAWDOG_API
int32_t
PhOnlineRawdogCompileTable(
    PH_ONLINE_RAWDOG_CONTEXT *Context,
    PH_ONLINE_RAWDOG_TABLE *Table,
    uint32_t VectorWidth,
    PH_ONLINE_RAWDOG_JIT_MAX_ISA JitMaxIsa
    );

PH_ONLINE_RAWDOG_API
int32_t
PhOnlineRawdogIndex32(
    PH_ONLINE_RAWDOG_TABLE *Table,
    uint32_t Key,
    uint32_t *Index
    );

PH_ONLINE_RAWDOG_API
void
PhOnlineRawdogReleaseTable(
    PH_ONLINE_RAWDOG_TABLE *Table
    );

#ifdef __cplusplus
}
#endif
