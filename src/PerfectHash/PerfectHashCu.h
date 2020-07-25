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
        CU_ERROR(__FUNCTION__##Name, CuResult);    \
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
    // Device identifier (obtained via Cu->DeviceGet(&Device->Id, Ordinal)).
    //

    CU_DEVICE Id;

    //
    // Pad out to an 8-byte boundary.
    //

    ULONG Padding1;

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
    PCU Cu;
    CU_CONTEXT Context;
    CU_MODULE Module;
    CU_FUNCTION Function;
    CU_OCCUPANCY Occupancy;
    PPH_CU_DEVICE Device;

    //
    // Kernel launch parameters.
    //

    ULONG BlocksPerGrid;
    ULONG ThreadsPerBlock;
    ULONG KernelRuntimeTargetInMilliseconds;
    ULONG JitMaxNumberOfRegisters;

    //
    // Buffer for output of JIT compilation.
    //

    CHAR JitLogBuffer[PERFECT_HASH_CU_JIT_LOG_BUFFER_SIZE_IN_BYTES];

} PH_CU_DEVICE_CONTEXT;
typedef PH_CU_DEVICE_CONTEXT *PPH_CU_DEVICE_CONTEXT;

typedef struct _PH_CU_DEVICE_CONTEXTS {
    LONG NumberOfDeviceContexts;
    LONG Padding;

    PH_CU_DEVICE_CONTEXTS DeviceContexts[ANYSIZE_ARRAY];
} PH_CU_DEVICES;
typedef PH_CU_DEVICE_CONTEXTS *PPH_CU_DEVICE_CONTEXTS;


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
