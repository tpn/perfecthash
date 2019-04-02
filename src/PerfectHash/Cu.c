/*++

Copyright (c) 2019 Trent Nelson <trent@trent.me>

Module Name:

    Cu.c

Abstract:

    This is the implementation file for the NVIDIA CUDA component.  It
    implements the InitCu() function.
--*/

#include "stdafx.h"
#include "Cu.h"

extern LOAD_SYMBOLS LoadSymbols;

HRESULT
InitCu(
    _In_ PRTL Rtl,
    _In_ PCU Cu
    )
{
    HMODULE Module;
    BOOLEAN Success;
    ULONG NumberOfNames;
    ULONG NumberOfResolvedSymbols;

#define EXPAND_AS_CU_NAME(Upper, Name) "cu" # Name,
    CONST PCSZ Names[] = {
        CU_FUNCTION_TABLE_ENTRY(EXPAND_AS_CU_NAME)
    };

    //
    // Define an appropriately sized bitmap we can passed to LoadSymbols().
    //

    ULONG BitmapBuffer[(ALIGN_UP(ARRAYSIZE(Names),
                        sizeof(ULONG) << 3) >> 5)+1] = { 0 };
    RTL_BITMAP FailedBitmap;

    FailedBitmap.SizeOfBitMap = ARRAYSIZE(Names) + 1;
    FailedBitmap.Buffer = (PULONG)&BitmapBuffer;

    NumberOfNames = ARRAYSIZE(Names);

    ASSERT(Cu->SizeOfStruct == sizeof(*Cu));
    if (Cu->SizeOfStruct != sizeof(*Cu)) {
        return E_INVALIDARG;
    }

    ZeroStructPointer(Cu);

    Cu->NumberOfFunctions = sizeof(Cu->CuFunctions) / sizeof(ULONG_PTR);
    ASSERT(Cu->NumberOfFunctions == NumberOfNames);

    Module = LoadLibraryA("nvcuda.dll");
    if (!IsValidHandle(Module)) {
        SYS_ERROR(LoadLibraryA);
        return E_FAIL;
    }

    Success = LoadSymbols(Names,
                          NumberOfNames,
                          (PULONG_PTR)&Cu->CuFunctions,
                          Cu->NumberOfFunctions,
                          Module,
                          &FailedBitmap,
                          &NumberOfResolvedSymbols);

    if (!Success) {
        return E_FAIL;
    }

    ASSERT(Cu->NumberOfFunctions == NumberOfResolvedSymbols);

    return S_OK;
}

CU_RESULT
LoadCuDeviceAttributes(
    _In_ PRTL Rtl,
    _In_ PCU Cu,
    _Inout_ PCU_DEVICE_ATTRIBUTES AttributesPointer,
    _In_ CU_DEVICE Device
    )
{
    LONG Index;
    LONG NumberOfAttributes;
    PLONG Attribute;
    CU_RESULT Result;

    ZeroStructPointer(AttributesPointer);

    Attribute = (PLONG)AttributesPointer;
    NumberOfAttributes = sizeof(*AttributesPointer) / sizeof(ULONG);

    for (Index = 0; Index < NumberOfAttributes; Index++) {
        Result = Cu->DeviceGetAttribute(Attribute, Index+1, Device);
        if (CU_FAILED(Result)) {
            return Result;
        }
        Attribute++;
    }

    return CUDA_SUCCESS;
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
