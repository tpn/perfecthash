/*++

Copyright (c) 2020-2023 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashCu.h

Abstract:

    This is the private header file for PerfectHash-specific parts of the
    NVIDIA CUDA component.

--*/

#pragma once

#include "stdafx.h"

//
// Defaults
//

#define PERFECT_HASH_CU_BLOCKS_PER_GRID 1024
#define PERFECT_HASH_CU_THREADS_PER_BLOCK 128
#define PERFECT_HASH_CU_WDDM_KERNEL_RUNTIME_TARGET_IN_MILLISECONDS 1500
#define PERFECT_HASH_CU_RANDOM_NUMBER_BATCH_SIZE 16384
#define PERFECT_HASH_CU_RNG_DEFAULT PerfectHashCuRngPhilox43210Id

//
// Error handling function and helper macro.
//

typedef
_Success_(return >= 0)
_Check_return_opt_
HRESULT
(NTAPI PERFECT_HASH_PRINT_CU_ERROR)(
    _In_ PCU Cu,
    _In_ PCSZ FunctionName,
    _In_ PCSZ FileName,
    _In_ ULONG LineNumber,
    _In_ CU_RESULT Error
    );
typedef PERFECT_HASH_PRINT_CU_ERROR *PPERFECT_HASH_PRINT_CU_ERROR;
extern PERFECT_HASH_PRINT_CU_ERROR PerfectHashPrintCuError;

#define CU_ERROR(Name, CuResult)             \
    PerfectHashPrintCuError(Cu,              \
                            #Name,           \
                            __FILE__,        \
                            __LINE__,        \
                            (ULONG)CuResult)

#define CU_CHECK(CuResult, Name)                   \
    if (CU_FAILED(CuResult)) {                     \
        CU_ERROR(Name, CuResult);                  \
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED; \
        goto Error;                                \
    }

#ifndef PH_WINDOWS
#define CU_PRINT_JIT_ERROR_LOG_BUFFER() \
    fprintf(stderr, "%s\n", &JitErrorLogBuffer[0])
#else
#define CU_PRINT_JIT_ERROR_LOG_BUFFER() \
        PRINT_CSTR(&JitErrorLogBuffer[0])
#endif

#define CU_LINK_CHECK(CuResult, Name)              \
    if (CU_FAILED(CuResult)) {                     \
        CU_ERROR(Name, CuResult);                  \
        CU_PRINT_JIT_ERROR_LOG_BUFFER();           \
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED; \
        goto Error;                                \
    }

//
// CUDA device information.
//

typedef struct _PH_CU_DEVICE {

    //
    // Ordinal of the device.
    //

    LONG Ordinal;

    //
    // Device driver handle.
    //

    CU_DEVICE Handle;

    //
    // Name of the device.
    //

    STRING Name;
    CHAR NameBuffer[32];

    //
    // Device attributes.
    //

    CU_DEVICE_ATTRIBUTES Attributes;

} PH_CU_DEVICE;
typedef PH_CU_DEVICE *PPH_CU_DEVICE;

typedef struct _PH_CU_DEVICES {
    LONG NumberOfDevices;
    LONG Padding;

    _Writable_elements_(NumberOfDevices)
    PPH_CU_DEVICE Devices;
} PH_CU_DEVICES;
typedef PH_CU_DEVICES *PPH_CU_DEVICES;

//
// Define an X-macro for the CUDA kernels.
//
// The entry callback parameters are as follows.
//
//  1. The name of the kernel.
//  2. The number of blocks per grid.
//  3. The number of threads per block.
//  4. Runtime target in milliseconds for WDDM drivers only.
//

#if 0
#define PERFECT_HASH_CUDA_KERNELS_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
                                                                        \
    FIRST_ENTRY(                                                        \
        LoadKeyStats,                                                   \
        PERFECT_HASH_CU_BLOCKS_PER_GRID,                                \
        PERFECT_HASH_CU_THREADS_PER_BLOCK,                              \
        PERFECT_HASH_CU_WDDM_KERNEL_RUNTIME_TARGET_IN_MILLISECONDS      \
    )                                                                   \
                                                                        \
    ENTRY(                                                              \
        HashKeys,                                                       \
        PERFECT_HASH_CU_BLOCKS_PER_GRID,                                \
        PERFECT_HASH_CU_THREADS_PER_BLOCK,                              \
        PERFECT_HASH_CU_WDDM_KERNEL_RUNTIME_TARGET_IN_MILLISECONDS      \
    )                                                                   \
                                                                        \
    ENTRY(                                                              \
        AddKeysToGraph,                                                 \
        PERFECT_HASH_CU_BLOCKS_PER_GRID,                                \
        PERFECT_HASH_CU_THREADS_PER_BLOCK,                              \
        PERFECT_HASH_CU_WDDM_KERNEL_RUNTIME_TARGET_IN_MILLISECONDS      \
    )                                                                   \
                                                                        \
    LAST_ENTRY(                                                         \
        IsGraphAcyclic,                                                 \
        PERFECT_HASH_CU_BLOCKS_PER_GRID,                                \
        PERFECT_HASH_CU_THREADS_PER_BLOCK,                              \
        PERFECT_HASH_CU_WDDM_KERNEL_RUNTIME_TARGET_IN_MILLISECONDS      \
    )
#else
#define PERFECT_HASH_CUDA_KERNELS_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
                                                                        \
    FIRST_ENTRY(                                                        \
        HashKeys,                                                       \
        PERFECT_HASH_CU_BLOCKS_PER_GRID,                                \
        PERFECT_HASH_CU_THREADS_PER_BLOCK,                              \
        PERFECT_HASH_CU_WDDM_KERNEL_RUNTIME_TARGET_IN_MILLISECONDS      \
    )

#define PERFECT_HASH_CUDA_KERNELS_TABLE_ENTRY(ENTRY) \
    PERFECT_HASH_CUDA_KERNELS_TABLE(ENTRY, ENTRY, ENTRY)

#define EXPAND_AS_CUDA_KERNEL_ENUM(Name, BlocksPerGrid, ThreadsPerBlock, \
                                   RuntimeTargetInMilliseconds)          \
    PerfectHashCudaKernel##Name##Id,
#endif

typedef enum _PERFECT_HASH_CUDA_KERNEL_ID {

    PerfectHashCudaKernelNullId = 0,

    PERFECT_HASH_CUDA_KERNELS_TABLE_ENTRY(EXPAND_AS_CUDA_KERNEL_ENUM)

    PerfectHashCudaKernelInvalidId,

} PERFECT_HASH_CUDA_KERNEL_ID;

extern const STRING PerfectHashCuKernelNames[];
static const BYTE NumberOfPerfectHashCuKernels =
    PerfectHashCudaKernelInvalidId - 1;

static
INLINE
const STRING*
PerfectHashGetCudaKernelName(
    _In_ PERFECT_HASH_CUDA_KERNEL_ID Id
    )
{
    return &PerfectHashCuKernelNames[Id];
}

//
// Each CUDA kernel is represented by the following PH_CU_KERNEL structure.  It
// captures the kernel name and the kernel entry point, as well as occupancy
// stats and kernel launch parameters.  Each device participating in solving
// gets a unique instance of this structure for each kernel (as different
// devices may have different capabilities and thus different launch
// parameters).
//

typedef struct _PH_CU_KERNEL {
    const STRING* Name;
    PCU_FUNCTION Function;
    CU_OCCUPANCY Occupancy;
    CU_STREAM Stream;
    PERFECT_HASH_CUDA_KERNEL_ID Id;
    ULONG BlocksPerGrid;
    ULONG ThreadsPerBlock;
    ULONG RuntimeTargetInMilliseconds;
} PH_CU_KERNEL, *PPH_CU_KERNEL;

//
// Each unique device in the system that will be participating in graph solving
// will have an instance of PH_CU_DEVICE_CONTEXT allocated.  This structure is
// responsible for encapsulating a per-device CUDA context, plus references to
// the CUDA module and graph solving kernel entry function.
//

typedef struct _PH_CU_DEVICE_CONTEXT {

    //
    // Pointer to an RTL instance.
    //

    PRTL Rtl;

    //
    // Device ordinal and driver handle.  Invariant: these two fields will
    // always match the corresponding fields in the Device member.
    //

    LONG Ordinal;
    CU_DEVICE Handle;

    //
    // Parent device this context points at.
    //

    PPH_CU_DEVICE Device;

    //
    // Pointer to the parent CU instance.
    //

    PCU Cu;

    //
    // CUDA context and module.
    //

    PCU_CONTEXT Context;
    PCU_MODULE Module;

#define EXPAND_AS_FIRST_CUDA_KERNEL(Name, BlocksPerGrid, ThreadsPerBlock, \
                                    RuntimeTargetInMilliseconds)          \
    union {                                                               \
        PH_CU_KERNEL Name##Kernel;                                        \
        PH_CU_KERNEL FirstKernel;                                         \
    };

#define EXPAND_AS_CUDA_KERNEL(Name, BlocksPerGrid, ThreadsPerBlock, \
                              RuntimeTargetInMilliseconds)          \
    PH_CU_KERNEL Name##Kernel;

#define EXPAND_AS_LAST_CUDA_KERNEL(Name, BlocksPerGrid, ThreadsPerBlock, \
                                   RuntimeTargetInMilliseconds)          \
    union {                                                              \
        PH_CU_KERNEL Name##Kernel;                                       \
        PH_CU_KERNEL LastKernel;                                         \
    };

    PERFECT_HASH_CUDA_KERNELS_TABLE(
        EXPAND_AS_FIRST_CUDA_KERNEL,
        EXPAND_AS_CUDA_KERNEL,
        EXPAND_AS_LAST_CUDA_KERNEL
    )

    //
    // CUDA stream for per-device activities unrelated to specific kernels
    // (e.g. copying keys, graph info, etc.).
    //

    CU_STREAM Stream;

    //
    // Base address of the keys currently copied to the device.
    //

    CU_DEVICE_POINTER KeysBaseAddress;

    //
    // Size of the keys array in bytes.
    //

    SIZE_T KeysSizeInBytes;

    //
    // Device address of the CU_DEVICE_ATTRIBUTES struct.
    //

    CU_DEVICE_POINTER DeviceAttributes;

    //
    // Device address of the GRAPH_INFO structure.
    //

    CU_DEVICE_POINTER DeviceGraphInfoAddress;

    //
    // Number of solving contexts associated with this device.
    //

    ULONG NumberOfSolveContexts;

    //
    // Each graph associated with this device gets an index assigned to its
    // Graph->CuDeviceIndex field, which is obtained by InterlockIncrement()
    // against the following counter.
    //

    volatile LONG NextDeviceIndex;

    //
    // Base address of array of device graphs (one per solve context).
    //

    struct _GRAPH *DeviceGraphs;

} PH_CU_DEVICE_CONTEXT;
typedef PH_CU_DEVICE_CONTEXT *PPH_CU_DEVICE_CONTEXT;

typedef struct _PH_CU_DEVICE_CONTEXTS {
    LONG NumberOfDeviceContexts;
    LONG Padding;

    PH_CU_DEVICE_CONTEXT DeviceContexts[ANYSIZE_ARRAY];
} PH_CU_DEVICE_CONTEXTS;
typedef PH_CU_DEVICE_CONTEXTS *PPH_CU_DEVICE_CONTEXTS;

//
// A PERFECT_HASH_CONTEXT will have a single instance of PH_CU_RUNTIME_CONTEXT.
//

typedef union _PH_CU_RUNTIME_FLAGS {
    struct {
        ULONG Initialized:1;
        ULONG WantsRandomHostSeeds:1;
        ULONG SawCuRngSeed:1;
        ULONG SawCuConcurrency:1;
        ULONG Unused:28;
    };
    ULONG AsULong;
} PH_CU_RUNTIME_FLAGS, *PPH_CU_RUNTIME_FLAGS;

typedef struct _PH_CU_RUNTIME_CONTEXT {
    PCU Cu;
    PH_CU_RUNTIME_FLAGS Flags;
    ULONG NumberOfDevices;
    ULONG NumberOfContexts;
    ULONG NumberOfRandomHostSeeds;
    PCUNICODE_STRING CuRngName;
    PERFECT_HASH_CU_RNG_ID CuRngId;
    ULONG Padding1;
    ULONGLONG CuRngSeed;
    ULONGLONG CuRngSubsequence;
    ULONGLONG CuRngOffset;
    PVALUE_ARRAY Ordinals;
    PVALUE_ARRAY BlocksPerGrid;
    PVALUE_ARRAY ThreadsPerBlock;
    PVALUE_ARRAY KernelRuntimeTarget;
    PUNICODE_STRING CuPtxPath;
    PUNICODE_STRING CuCudaDevRuntimeLibPath;

    //
    // CUDA devices.
    //

    PH_CU_DEVICES CuDevices;

    //
    // CUDA device contexts.
    //

    PPH_CU_DEVICE_CONTEXTS CuDeviceContexts;

} PH_CU_RUNTIME_CONTEXT, *PPH_CU_RUNTIME_CONTEXT;

//
// Each solver GPU thread gets an instance of PH_CU_SOLVE_CONTEXT.  This
// structure ties together the host and device graphs, as well as the device
// context.
//

typedef struct _PH_CU_SOLVE_CONTEXT {

    //
    // Pointer to the owning device context.
    //

    PPH_CU_DEVICE_CONTEXT DeviceContext;

    //
    // Streams for this solve context.
    //

    union {
        CU_STREAM Stream;
        CU_STREAM Stream1;
        CU_STREAM FirstStream;
    };
    CU_STREAM Stream2;
    CU_STREAM Stream3;
    union {
        CU_STREAM Stream4;
        CU_STREAM LastStream;
    };

    //
    // Host and device graph instances.
    //

    struct _GRAPH *HostGraph;
    struct _GRAPH *DeviceGraph;

    //
    // Spare host and device graphs.
    //

    struct _GRAPH *HostSpareGraph;
    struct _GRAPH *DeviceSpareGraph;

} PH_CU_SOLVE_CONTEXT;
typedef PH_CU_SOLVE_CONTEXT *PPH_CU_SOLVE_CONTEXT;

typedef struct _PH_CU_SOLVE_CONTEXTS {
    ULONG NumberOfSolveContexts;
    ULONG Padding;

    PH_CU_SOLVE_CONTEXT SolveContexts[ANYSIZE_ARRAY];
} PH_CU_SOLVE_CONTEXTS;
typedef PH_CU_SOLVE_CONTEXTS *PPH_CU_SOLVE_CONTEXTS;


_Must_inspect_result_
_Success_(return >= 0)
HRESULT
CreateCuInstance(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _COM_Outptr_ PCU *CuInstance
    );

_Must_inspect_result_
_Success_(return >= 0)
HRESULT
CreateCuRuntimeContext(
    _In_ PCU Cu,
    _Out_ PPH_CU_RUNTIME_CONTEXT *CuRuntimeContextPointer
    );

VOID
DestroyCuRuntimeContext(
    _Inout_ PPH_CU_RUNTIME_CONTEXT *CuRuntimeContextPointer
    );

_Must_inspect_result_
_Success_(return >= 0)
HRESULT
InitializeCuRuntimeContext(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters,
    _Inout_ PPH_CU_RUNTIME_CONTEXT CuRuntimeContext
    );

_Must_inspect_result_
_Success_(return >= 0)
HRESULT
CreatePerfectHashCuDevices(
    _In_ PCU Cu,
    _In_ PALLOCATOR Allocator,
    _Inout_ PPH_CU_DEVICES Devices
    );

_Must_inspect_result_
_Success_(return >= 0)
HRESULT
CuDeviceContextInitializeKernels(
    _In_ PPH_CU_DEVICE_CONTEXT CuDeviceContext
    );

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
