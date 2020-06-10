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

#define CU_ERROR(Name, CuResult)            \
    PerfectHashPrintCuError(Cu,             \
                            #Name,          \
                            __FILE__,       \
                            __LINE__,       \
                            (ULONG)CuResult)

#define CU_CHECK(CuResult, Name) \
    if (CU_FAILED(CuResult)) { \
        CU_ERROR(__FUNCTION__##Name, CuResult); \
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED; \
        goto Error; \
    }

//
// CUDA device information.
//

typedef struct _PH_CU_DEVICE {

    //
    // Ordinal of the device.
    //

    CU_DEVICE Ordinal;

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

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
