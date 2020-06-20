/*++

Copyright (c) 2020 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashCu.c

Abstract:

    This is the module for PerfectHash-specifc CUDA functionality.

--*/

#include "stdafx.h"

PERFECT_HASH_PRINT_CU_ERROR PerfectHashPrintCuError;

_Use_decl_annotations_
HRESULT
PerfectHashPrintCuError(
    PCU Cu,
    PCSZ FunctionName,
    PCSZ FileName,
    ULONG LineNumber,
    CU_RESULT Error
    )
{
    BOOL Success;
    ULONG Flags;
    ULONG Count;
    PCHAR Buffer;
    PCHAR BaseBuffer;
    PCHAR EndBuffer;
    PCHAR CuErrorName;
    PCHAR CuErrorString;
    CU_RESULT CuResult;
    ULONG LanguageId;
    ULONG BytesWritten;
    HRESULT Result = S_OK;
    ULONG_PTR BytesToWrite;
    LONG_PTR SizeOfBufferInBytes;
    LONG_PTR SizeOfBufferInChars;
    CHAR LocalBuffer[1024];
    ULONG_PTR Args[7];
    const STRING Prefix = RTL_CONSTANT_STRING(
        "%1: %2!lu!: %3 failed with error: %4!lu! (0x%4!lx!): %5: %6\n"
    );

    CuResult = Cu->GetErrorName(Error, &CuErrorName);
    if (CU_FAILED(CuResult)) {
        OutputDebugStringA("PhPrintCuError: CuGetErrorName() failed.\n");
        goto Error;
    }

    CuResult = Cu->GetErrorString(Error, &CuErrorString);
    if (CU_FAILED(CuResult)) {
        OutputDebugStringA("PhPrintCuError: CuGetErrorString() failed.\n");
        goto Error;
    }

    Args[0] = (ULONG_PTR)FileName,
    Args[1] = (ULONG_PTR)LineNumber;
    Args[2] = (ULONG_PTR)FunctionName;
    Args[3] = (ULONG_PTR)Error;
    Args[4] = (ULONG_PTR)CuErrorName;
    Args[5] = (ULONG_PTR)CuErrorString;
    Args[6] = 0;

    SizeOfBufferInBytes = sizeof(LocalBuffer);
    BaseBuffer = Buffer = (PCHAR)&LocalBuffer;
    EndBuffer = Buffer + SizeOfBufferInBytes;

    //
    // The following is unnecessary when dealing with bytes, but will allow
    // easy conversion into a WCHAR version at a later date.
    //

    SizeOfBufferInChars = (LONG_PTR)(
        SizeOfBufferInBytes *
        (LONG_PTR)sizeof(*Buffer)
    );

    Flags = FORMAT_MESSAGE_FROM_STRING | FORMAT_MESSAGE_ARGUMENT_ARRAY;

    LanguageId = MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT);

    Count = FormatMessageA(Flags,
                           Prefix.Buffer,
                           0,
                           LanguageId,
                           (PSTR)Buffer,
                           (ULONG)SizeOfBufferInChars,
                           (va_list *)Args);

    if (!Count) {
        OutputDebugStringA("PhPrintCuError: FormatMessageA().\n");
        goto Error;
    }

    Buffer += Count;
    SizeOfBufferInChars -= Count;

    //
    // We want at least two characters left in the buffer for the \n and
    // trailing NULL.
    //

    ASSERT(SizeOfBufferInChars >= 2);
    ASSERT((ULONG_PTR)Buffer <= (ULONG_PTR)(EndBuffer - 2));

    *Buffer += '\n';
    *Buffer += '\0';

    ASSERT((ULONG_PTR)Buffer <= (ULONG_PTR)EndBuffer);

    BytesToWrite = RtlPointerToOffset(BaseBuffer, Buffer);
    ASSERT(BytesToWrite <= sizeof(LocalBuffer));

    Success = WriteFile(GetStdHandle(STD_ERROR_HANDLE),
                        BaseBuffer,
                        (ULONG)BytesToWrite,
                        &BytesWritten,
                        NULL);

    if (!Success) {
        OutputDebugStringA("PhPrintCuError: WriteFile() failed.\n");
        goto Error;
    }

    //
    // We're done, finish up and return.
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

_Must_inspect_result_
_Success_(return >= 0)
HRESULT
CreatePerfectHashCuDevices(
    _In_ PCU Cu,
    _In_ PALLOCATOR Allocator,
    _Inout_ PPH_CU_DEVICES Devices
    )
/*++

Routine Description:

    Allocates and initializes an array of CU devices.

Arguments:

    Cu - Supplies the CU instance.

    Allocator - Supplies the allocator.

    Devices - Supplies a pointer to the devices to update.  Caller is
        responsible for freeing Devices->Devices.

Return Value:

    S_OK - Success.

    E_OUTOFMEMORY - Out of memory.

--*/
{
    LONG Index;
    HRESULT Result;
    PPH_CU_DEVICE Device;
    CU_RESULT CuResult;
    PVOID Address = NULL;
    PSTRING Name;
    PCHAR Buffer;

    //
    // Get the number of devices.
    //

    CuResult = Cu->DeviceGetCount(&Devices->NumberOfDevices);
    if (CU_FAILED(CuResult)) {
        CU_ERROR(CuDeviceGetCount, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto End;
    }

    Address = Allocator->Vtbl->Calloc(Allocator,
                                      Devices->NumberOfDevices,
                                      sizeof(Devices->Devices[0]));
    if (!Address) {
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    Devices->Devices = (PPH_CU_DEVICE)Address;
    for (Index = 0; Index < Devices->NumberOfDevices; Index++) {
        Device = &Devices->Devices[Index];

        //
        // Initialize the ordinal to the current index.
        //

        Device->Ordinal = Index;

        //
        // Obtain the device identifier.
        //

        CuResult = Cu->DeviceGet(&Device->Handle, Device->Ordinal);
        if (CU_FAILED(CuResult)) {
            CU_ERROR(CreatePerfectHashCuDevices_DeviceGet, CuResult);
            Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
            goto Error;
        }

        //
        // Load the device name.
        //

        Name = &Device->Name;
        Buffer = (PCHAR)&Device->NameBuffer;
        Name->Length = 0;
        Name->MaximumLength = sizeof(Device->NameBuffer);
        CuResult = Cu->DeviceGetName(Buffer,
                                     (LONG)Name->MaximumLength,
                                     Device->Handle);
        if (CU_FAILED(CuResult)) {
            CU_ERROR(CuDeviceGetName, CuResult);
            Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
            goto Error;
        }

        Name->Length = (USHORT)strlen(Buffer);
        Name->Buffer = Buffer;

        //
        // Load device attributes.
        //

        CuResult = Cu->Vtbl->LoadCuDeviceAttributes(Cu,
                                                    &Device->Attributes,
                                                    Device->Handle);
        if (CU_FAILED(CuResult)) {
            CU_ERROR(CuDeviceGetAttribute, CuResult);
            Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
            goto Error;
        }
    }

    //
    // We're done, indicate success and finish up.
    //

    Result = S_OK;
    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    Allocator->Vtbl->FreePointer(Allocator, &Devices->Devices);

    //
    // Intentional follow-on to End.
    //

End:

    return Result;

}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
