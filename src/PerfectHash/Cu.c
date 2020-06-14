/*++

Copyright (c) 2019 Trent Nelson <trent@trent.me>

Module Name:

    Cu.c

Abstract:

    This is the implementation file for the NVIDIA CUDA component.  It
    implements the InitCu() function.
--*/

#include "stdafx.h"

extern LOAD_SYMBOLS LoadSymbols;

//
// COM scaffolding routines for initialization and rundown.
//

CU_INITIALIZE CuInitialize;

_Use_decl_annotations_
HRESULT
CuInitialize(
    PCU Cu
    )
/*++

Routine Description:

    Initializes a graph structure.  This is a relatively simple method that
    just primes the COM scaffolding.

Arguments:

    Cu - Supplies a pointer to a CU structure for which initialization
        is to be performed.

Return Value:

    S_OK - Success.

    E_POINTER - Cu is NULL.

    E_UNEXPECTED - All other errors.

--*/
{
    HRESULT Result;
    HMODULE Module;
    BOOLEAN Success;
    CU_RESULT CuResult;
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

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Cu)) {
        return E_POINTER;
    }

    Cu->SizeOfStruct = sizeof(*Cu);

    //
    // Continue initialization.  Wire up the FailedBitmap.
    //

    FailedBitmap.SizeOfBitMap = ARRAYSIZE(Names) + 1;
    FailedBitmap.Buffer = (PULONG)&BitmapBuffer;

    NumberOfNames = ARRAYSIZE(Names);

    Cu->NumberOfFunctions = sizeof(Cu->CuFunctions) / sizeof(ULONG_PTR);
    ASSERT(Cu->NumberOfFunctions == NumberOfNames);

    Module = LoadLibraryA("nvcuda.dll");
    if (!IsValidHandle(Module)) {
        Result = PH_E_NVCUDA_DLL_LOAD_LIBRARY_FAILED;
        goto Error;
    }

    Success = LoadSymbols(Names,
                          NumberOfNames,
                          (PULONG_PTR)&Cu->CuFunctions,
                          Cu->NumberOfFunctions,
                          Module,
                          &FailedBitmap,
                          &NumberOfResolvedSymbols);

    if (!Success) {
        Result = PH_E_NVCUDA_DLL_LOAD_SYMBOLS_FAILED;
        goto Error;
    }

    if (Cu->NumberOfFunctions != NumberOfResolvedSymbols) {
        Result = PH_E_NVCUDA_DLL_LOAD_SYMBOLS_FAILED_TO_LOAD_ALL_SYMBOLS;
        goto Error;
    }

    //
    // Initialize CUDA.
    //

    CuResult = Cu->Init(0);
    if (CU_FAILED(CuResult)) {
        CU_ERROR(CuInitialize_Init, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    //
    // We're done!  Indicate success and finish up.
    //

    Result = S_OK;
    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}


CU_RUNDOWN CuRundown;

_Use_decl_annotations_
VOID
CuRundown(
    PCU Cu
    )
/*++

Routine Description:

    Release all resources associated with a CU instance.

Arguments:

    Cu - Supplies a pointer to a CU structure to rundown.

Return Value:

    None.

--*/
{
    //
    // Sanity check structure size.
    //

    ASSERT(Cu->SizeOfStruct == sizeof(*Cu));

    //
    // Release the nvcuda.dll module if applicable.
    //

    if (Cu->NvCudaModule != NULL) {
        FreeLibrary(Cu->NvCudaModule);
        Cu->NvCudaModule = NULL;
    }

    //
    // Release applicable COM references.
    //

    RELEASE(Cu->Rtl);
    RELEASE(Cu->Allocator);

    return;
}

_Use_decl_annotations_
CU_RESULT
LoadCuDeviceAttributes(
    PCU Cu,
    PCU_DEVICE_ATTRIBUTES AttributesPointer,
    CU_DEVICE Device
    )
/*++

Routine Description:

    Loads CUDA device attributes for a given device.

Arguments:

    Cu - Supplies the CU instance.

    AttributesPointer - Supplies a pointer to a CU_DEVICE_ATTRIBUTES struct
        that will receive the attributes for the given device.

    Device - Supplies the CUDA device.

Return Value:

    CUDA_SUCCESS on success, otherwise a CUDA error code.

--*/
{
    PRTL Rtl;
    LONG Index;
    LONG NumberOfAttributes;
    PLONG Attribute;
    CU_RESULT Result;

    Rtl = Cu->Rtl;
    ZeroStructPointer(AttributesPointer);

    Attribute = (PLONG)AttributesPointer;
    NumberOfAttributes = sizeof(*AttributesPointer) / sizeof(ULONG);

    for (Index = 0; Index < NumberOfAttributes; Index++) {
        Result = Cu->DeviceGetAttribute(Attribute, Index+1, Device);
        if (CU_FAILED(Result)) {
            continue;
        }
        Attribute++;
    }

    return CUDA_SUCCESS;
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
