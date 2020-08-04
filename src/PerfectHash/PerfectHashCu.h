/*++

Copyright (c) 2020 Trent Nelson <trent@trent.me>

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

#define PH_CU_BLOCKS_PER_GRID 32
#define PH_CU_THREADS_PER_BLOCK 512
#define PH_CU_WDDM_KERNEL_RUNTIME_TARGET_IN_MILLISECONDS 1500
#define PH_CU_RANDOM_NUMBER_BATCH_SIZE 16384

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
    _In_opt_ ULONG LineNumber,
    _In_opt_ CU_RESULT Error
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
        CU_ERROR(__FUNCTION__ ## Name, CuResult);  \
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

    ULONG Padding2;

} PH_CU_DEVICE;
typedef PH_CU_DEVICE *PPH_CU_DEVICE;

typedef struct _PH_CU_DEVICES {
    LONG NumberOfDevices;
    LONG Padding;

    _Writable_elements_(NumberOfDevices)
    PPH_CU_DEVICE Devices;
} PH_CU_DEVICES;
typedef PH_CU_DEVICES *PPH_CU_DEVICES;

_Must_inspect_result_
_Success_(return >= 0)
HRESULT
CreatePerfectHashCuDevices(
    _In_ PCU Cu,
    _In_ PALLOCATOR Allocator,
    _Inout_ PPH_CU_DEVICES Devices
    );

//
// Each unique device in the system that will be participating in graph solving
// will have an instance of PH_CU_DEVICE_CONTEXT allocated.  This structure is
// responsible for encapsulating a per-device CUDA context, plus references to
// the CUDA module and graph solving kernel entry function.
//

typedef struct _PH_CU_DEVICE_CONTEXT {

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
    // CUDA context, module, solver function entry point, and occupancy helper.
    //

    PCU_CONTEXT Context;
    PCU_MODULE Module;
    PCU_FUNCTION Function;
    CU_OCCUPANCY Occupancy;

    //
    // CUDA stream for per-device activities (like copying keys).
    //

    PCU_STREAM Stream;

    //
    // Base address of the keys currently copied to the device.
    //

    CU_DEVICE_POINTER KeysBaseAddress;

    //
    // Device address of the CU_DEVICE_ATTRIBUTES struct.
    //

    CU_DEVICE_POINTER DeviceAttributes;

    //
    // XXX: I don't think we need this for GPU solving.
    //

#if 0
    //
    // Best and spare graphs.
    //

    CRITICAL_SECTION BestGraphCriticalSection;

    _Guarded_by_(BestGraphCriticalSection)
    struct _GRAPH *BestGraph;

    //
    // The following counter is incremented every time a new "best graph" is
    // registered.
    //

    _Guarded_by_(BestGraphCriticalSection)
    volatile LONG NewBestGraphCount;

    //
    // The following counter is incremented every time a graph is found whose
    // coverage matches the existing best graph's coverage (for the given
    // predicate when in "find best graph" mode).
    //

    _Guarded_by_(BestGraphCriticalSection)
    volatile LONG EqualBestGraphCount;
#endif

    //
    // Number of solving contexts associated with this device.
    //

    ULONG NumberOfSolveContexts;
    ULONG Padding1;

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
// Each solver GPU thread gets an instance of PH_CU_SOLVE_CONTEXT.
//

typedef struct _PH_CU_SOLVE_CONTEXT {

    //
    // Pointer to the owning device context.
    //

    PPH_CU_DEVICE_CONTEXT DeviceContext;

    //
    // Kernel launch stream.
    //

    PCU_STREAM Stream;

    //
    // Host and device graph instances.
    //

    struct _GRAPH *HostGraph;
    struct _GRAPH *DeviceGraph;

    //
    // Spare host and device graphs.
    //

    struct _GRAPH *SpareHostGraph;
    struct _GRAPH *SpareDeviceGraph;

    //
    // Kernel launch parameters.
    //

    ULONG BlocksPerGrid;
    ULONG ThreadsPerBlock;
    ULONG KernelRuntimeTargetInMilliseconds;
    ULONG JitMaxNumberOfRegisters;

} PH_CU_SOLVE_CONTEXT;
typedef PH_CU_SOLVE_CONTEXT *PPH_CU_SOLVE_CONTEXT;

typedef struct _PH_CU_SOLVE_CONTEXTS {

    ULONG NumberOfSolveContexts;
    ULONG Padding;

    PH_CU_SOLVE_CONTEXT SolveContexts[ANYSIZE_ARRAY];
} PH_CU_SOLVE_CONTEXTS;
typedef PH_CU_SOLVE_CONTEXTS *PPH_CU_SOLVE_CONTEXTS;


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
