/*++

Copyright (c) 2020-2023 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashCu.c

Abstract:

    This is the module for PerfectHash-specifc CUDA functionality.

--*/

#include "stdafx.h"

CHAR JitInfoLogBuffer[PERFECT_HASH_CU_JIT_LOG_BUFFER_SIZE_IN_BYTES];
CHAR JitErrorLogBuffer[PERFECT_HASH_CU_JIT_LOG_BUFFER_SIZE_IN_BYTES];

#define EXPAND_AS_CU_KERNEL_NAME(Name,            \
                                 BlocksPerGrid,   \
                                 ThreadsPerBlock, \
                                 RuntimeTarget)   \
    RTL_CONSTANT_STRING(#Name),

const STRING PerfectHashCuKernelNames[] = {
    RTL_CONSTANT_STRING(""), // Null
    PERFECT_HASH_CUDA_KERNELS_TABLE(EXPAND_AS_CU_KERNEL_NAME,
                                    EXPAND_AS_CU_KERNEL_NAME,
                                    EXPAND_AS_CU_KERNEL_NAME)
    RTL_CONSTANT_STRING(""), // Invalid
};

//
// Forward decls.
//

_Must_inspect_result_
_Success_(return >= 0)
HRESULT
CuRuntimeContextProcessTableCreateParameters(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PPH_CU_RUNTIME_CONTEXT CuRuntimeContext,
    _In_ PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    );

_Must_inspect_result_
_Success_(return >= 0)
HRESULT
CuRuntimeContextProcessOrdinals(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PPH_CU_RUNTIME_CONTEXT CuRuntimeContext
    );

_Must_inspect_result_
_Success_(return >= 0)
HRESULT
CuRuntimeContextInitializeDevices(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PPH_CU_RUNTIME_CONTEXT CuRuntimeContext
    );

PERFECT_HASH_PRINT_CU_ERROR PerfectHashPrintCuError;

#ifdef PH_WINDOWS
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
        "%1:%2!lu!: %3 failed with error: %4!lu! (0x%4!lx!): %5: %6\n"
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

#else // PH_WINDOWS

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
    PCHAR CuErrorName;
    PCHAR CuErrorString;
    CU_RESULT CuResult;
    HRESULT Result = PH_E_SYSTEM_CALL_FAILED;

    const STRING Prefix = RTL_CONSTANT_STRING(
        "%s:%u: %s failed with error: 0x%x: %s: %s\n"
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

    fprintf(stderr,
            Prefix.Buffer,
            FileName,
            LineNumber,
            FunctionName,
            Error,
            CuErrorName,
            CuErrorString);

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

#endif // !PH_WINDOWS

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

//
// Disable `warning C6001: Using uninitialized memory '*CuPointer'.`; I can't
// figure out why it's being reported for this function.
//

#pragma warning(push)
#pragma warning(disable: 6001)

_Must_inspect_result_
_Success_(return >= 0)
HRESULT
CreateCuInstance(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _COM_Outptr_ PCU *CuPointer
    )
/*++

Routine Description:

    This routine creates an instance of the CU component.

Arguments:

    Context - Supplies a pointer to a PERFECT_HASH_CONTEXT structure.

    CuPointer - Supplies a pointer to a variable that receives a pointer to the
        newly-created CU instance.

Return Value:

    S_OK if successful, or an appropriate error code otherwise.

--*/
{
    PCU Cu;
    HRESULT Result;

    //
    // Clear the caller's pointer up-front.
    //

    *CuPointer = NULL;

    Result = Context->Vtbl->CreateInstance(Context,
                                           NULL,
                                           &IID_PERFECT_HASH_CU,
                                           PPV(&Cu));

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashCuCreateInstance, Result);
        return Result;
    }

    //
    // The CU component is a global component, which means it can't create
    // instances of other global components like Rtl and Allocator during its
    // initialization function.  So, we manually set them now.
    //

    Cu->Rtl = Context->Rtl;
    Cu->Rtl->Vtbl->AddRef(Cu->Rtl);

    Cu->Allocator = Context->Allocator;
    Cu->Allocator->Vtbl->AddRef(Cu->Allocator);

    //
    // Copy the instance to the caller's pointer and return success.
    //

    *CuPointer = Cu;

    return S_OK;
}
#pragma warning(pop)

_Must_inspect_result_
_Success_(return >= 0)
HRESULT
CreateCuRuntimeContext(
    _In_ PCU Cu,
    _Out_ PPH_CU_RUNTIME_CONTEXT *CuRuntimeContextInstance
    )
{
    HRESULT Result;
    PPH_CU_RUNTIME_CONTEXT CuRuntimeContext;

    //
    // Clear the caller's pointer up-front.
    //

    *CuRuntimeContextInstance = NULL;

    //
    // Create the runtime context.
    //

    CuRuntimeContext = Cu->Allocator->Vtbl->Calloc(Cu->Allocator,
                                                   1,
                                                   sizeof(*CuRuntimeContext));
    if (!CuRuntimeContext) {
        return E_OUTOFMEMORY;
    }

    //
    // Copy the CU instance pointer and add a reference.
    //

    CuRuntimeContext->Cu = Cu;
    Cu->Vtbl->AddRef(Cu);

    //
    // Create the array of PH_CU_DEVICE structs.
    //

    Result = CreatePerfectHashCuDevices(Cu,
                                        Cu->Allocator,
                                        &CuRuntimeContext->CuDevices);
    if (FAILED(Result)) {
        PH_ERROR(CreatePerfectHashCuDevices, Result);
        return Result;
    }

    *CuRuntimeContextInstance = CuRuntimeContext;

    return S_OK;
}

VOID
DestroyCuRuntimeContext(
    _Inout_ PPH_CU_RUNTIME_CONTEXT *CuRuntimeContextPointer
    )
{
    PALLOCATOR Allocator;
    PPH_CU_RUNTIME_CONTEXT CuRuntimeContext;

    if (!ARGUMENT_PRESENT(CuRuntimeContextPointer)) {
        return;
    }

    CuRuntimeContext = *CuRuntimeContextPointer;
    if (CuRuntimeContext == NULL) {
        return;
    }

    Allocator = CuRuntimeContext->Cu->Allocator;

    //
    // TODO: loop through the device contexts and free any device-specific allocs.
    //

    //
    // Free the array of PH_CU_DEVICE structs.
    //

    Allocator->Vtbl->FreePointer(Allocator,
                                 &CuRuntimeContext->CuDevices.Devices);

    //
    // Free the device contexts.
    //

    Allocator->Vtbl->FreePointer(Allocator,
                                 &CuRuntimeContext->CuDeviceContexts);

    //
    // Release the CU instance.
    //

    RELEASE(CuRuntimeContext->Cu);

    //
    // Clear the caller's pointer and return.
    //

    *CuRuntimeContextPointer = NULL;
}

_Must_inspect_result_
_Success_(return >= 0)
HRESULT
InitializeCuRuntimeContext(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters,
    _Inout_ PPH_CU_RUNTIME_CONTEXT CuRuntimeContext
    )
/**

Routine Description:

    Initializes a CU_RUNTIME_CONTEXT structure from the supplied
    PERFECT_HASH_TABLE_CREATE_PARAMETERS structure.

Arguments:

    Context - Supplies a pointer to a PERFECT_HASH_CONTEXT structure.

    TableCreateParameters - Supplies a pointer to a
        PERFECT_HASH_TABLE_CREATE_PARAMETERS structure.

    CuRuntimeContext - Supplies a pointer to a CU_RUNTIME_CONTEXT structure
        that will be initialized.

Return Value:

    HRESULT - S_OK on success, appropriate HRESULT otherwise.

--*/

{
    PRTL Rtl;
    HRESULT Result;

    ASSERT(CuRuntimeContext->Flags.Initialized == FALSE);

    Rtl = Context->Rtl;

    //
    // Process the table create parameters.
    //

    Result = CuRuntimeContextProcessTableCreateParameters(
        Context,
        CuRuntimeContext,
        TableCreateParameters
    );
    if (FAILED(Result)) {
        PH_ERROR(CuRuntimeContextProcessTableCreateParameters, Result);
        goto Error;
    }

    //
    // Process the ordinals supplied via --CuDevices, if applicable.
    //

    Result = CuRuntimeContextProcessOrdinals(Context, CuRuntimeContext);
    if (FAILED(Result)) {
        PH_ERROR(CuRuntimeContextProcessOrdinals, Result);
        goto Error;
    }

    //
    // Initialize the device contexts.
    //

    Result = CuRuntimeContextInitializeDevices(Context,
                                               CuRuntimeContext);
    if (FAILED(Result)) {
        PH_ERROR(CuRuntimeContextInitializeDeviceContexts, Result);
        goto Error;
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

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}

_Must_inspect_result_
_Success_(return >= 0)
HRESULT
CuRuntimeContextProcessTableCreateParameters(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PPH_CU_RUNTIME_CONTEXT CuRuntimeContext,
    _In_ PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    )
/*++

Routine Description:

    Processes the Cu-related table create parameters supplied via the command
    line and updates the CuRuntimeContext structure accordingly.

Arguments:

    CuRuntimeContext - Supplies a pointer to a CU_RUNTIME_CONTEXT structure.

    TableCreateParameters - Supplies a pointer to a
        PERFECT_HASH_TABLE_CREATE_PARAMETERS structure.

Return Value:

    HRESULT - S_OK on success, appropriate HRESULT otherwise.

--*/
{
    PRTL Rtl;
    ULONG Index;
    ULONG Count;
    HRESULT Result;
    BOOLEAN SawCuRngSeed;
    BOOLEAN SawCuConcurrency;
    BOOLEAN IsRngImplemented;
    BOOLEAN WantsRandomHostSeeds;
    PPERFECT_HASH_TABLE_CREATE_PARAMETER Param;

    Rtl = Context->Rtl;

    //
    // Disable "enum not handled in switch statement" warning.
    //

#pragma warning(push)
#pragma warning(disable: 4061)

    //
    // Loop through the table create parameters and process CU-related ones.
    //

    SawCuRngSeed = FALSE;
    SawCuConcurrency = FALSE;
    WantsRandomHostSeeds = FALSE;

    Count = TableCreateParameters->NumberOfElements;
    Param = TableCreateParameters->Params;

    for (Index = 0; Index < Count; Index++, Param++) {

        switch (Param->Id) {

            case TableCreateParameterCuRngId:
                CuRuntimeContext->CuRngId = Param->AsCuRngId;
                break;

            case TableCreateParameterCuRngSeedId:
                CuRuntimeContext->CuRngSeed = Param->AsULongLong;
                SawCuRngSeed = TRUE;
                break;

            case TableCreateParameterCuRngSubsequenceId:
                CuRuntimeContext->CuRngSubsequence = Param->AsULongLong;
                break;

            case TableCreateParameterCuRngOffsetId:
                CuRuntimeContext->CuRngOffset = Param->AsULongLong;
                break;

            case TableCreateParameterCuConcurrencyId:
                Context->CuConcurrency = Param->AsULong;
                SawCuConcurrency = TRUE;
                break;

            case TableCreateParameterCuDevicesId:
                CuRuntimeContext->Ordinals = &Param->AsValueArray;
                break;

            case TableCreateParameterCuDevicesBlocksPerGridId:
                CuRuntimeContext->BlocksPerGrid = &Param->AsValueArray;
                break;

            case TableCreateParameterCuDevicesThreadsPerBlockId:
                CuRuntimeContext->ThreadsPerBlock = &Param->AsValueArray;
                break;

            case TableCreateParameterCuDevicesKernelRuntimeTargetInMillisecondsId:
                CuRuntimeContext->KernelRuntimeTarget = &Param->AsValueArray;
                break;

            case TableCreateParameterCuPtxPathId:
                CuRuntimeContext->CuPtxPath = &Param->AsUnicodeString;
                break;

            case TableCreateParameterCuCudaDevRuntimeLibPathId:
                CuRuntimeContext->CuCudaDevRuntimeLibPath =
                    &Param->AsUnicodeString;
                break;

            case TableCreateParameterCuNumberOfRandomHostSeedsId:
                CuRuntimeContext->NumberOfRandomHostSeeds = Param->AsULong;
                WantsRandomHostSeeds = TRUE;
                break;

            default:
                break;
        }
    }

#pragma warning(pop)

    //
    // Validate --CuRng.  We only implement a subset of algorithms.
    //

    if (!IsValidPerfectHashCuRngId(CuRuntimeContext->CuRngId)) {
        CuRuntimeContext->CuRngId = PERFECT_HASH_CU_RNG_DEFAULT;
    }

    Result = PerfectHashLookupNameForId(Rtl,
                                        PerfectHashCuRngEnumId,
                                        CuRuntimeContext->CuRngId,
                                        &CuRuntimeContext->CuRngName);
    if (FAILED(Result)) {
        PH_ERROR(InitializeCuRuntimeContext_LookupNameForId, Result);
        goto Error;
    }

    IsRngImplemented = FALSE;

#define EXPAND_AS_CU_RNG_ID_CASE(Name, Upper, Implemented) \
    case PerfectHashCuRng##Name##Id:                       \
        IsRngImplemented = Implemented;                    \
        break;

    switch (CuRuntimeContext->CuRngId) {

        case PerfectHashNullCuRngId:
        case PerfectHashInvalidCuRngId:
            PH_RAISE(PH_E_UNREACHABLE_CODE);
            break;

        PERFECT_HASH_CU_RNG_TABLE_ENTRY(EXPAND_AS_CU_RNG_ID_CASE);

        default:
            PH_RAISE(PH_E_UNREACHABLE_CODE);
            break;
    }

    if (!IsRngImplemented) {
        Result = PH_E_UNIMPLEMENTED_CU_RNG_ID;
        goto Error;
    }

    //
    // If no seed has been supplied, use the default.
    //

    if (!SawCuRngSeed) {
        CuRuntimeContext->CuRngSeed = RNG_DEFAULT_SEED;
    }

#if 0
    //
    // Ensure a runtime path was supplied.
    //

    if (CuRuntimeContext->CuCudaDevRuntimeLibPath == NULL) {
        Result = PH_E_CU_CUDA_DEV_RUNTIME_LIB_PATH_MANDATORY;
        goto Error;
    }
#endif

    //
    // Validate --CuConcurrency.  It's mandatory, it must be greater than zero,
    // and less than or equal to the maximum concurrency.  (When CuConcurrency
    // is less than max concurrency, the difference between the two will be the
    // number of CPU solving threads launched.  E.g. if --CuConcurrency=16 and
    // max concurrency is 18; there will be two CPU solving threads launched in
    // addition to the 16 GPU solver threads.)
    //

    //
    // N.B. The fact that we have to mutate Context here is hacky.  Leaky
    //      abstraction that's an artifact of bolting on GPU/CUDA support
    //      after the fact.
    //

    if (!SawCuConcurrency) {
        Result = PH_E_CU_CONCURRENCY_MANDATORY_FOR_SELECTED_ALGORITHM;
        goto Error;
    }

    if (Context->CuConcurrency == 0) {
        Result = PH_E_INVALID_CU_CONCURRENCY;
        goto Error;
    }

    if (Context->CuConcurrency > Context->MaximumConcurrency) {
        Result = PH_E_CU_CONCURRENCY_EXCEEDS_MAX_CONCURRENCY;
        goto Error;
    }

    //
    // Calculate the number of CPU solving threads; this may be zero.
    //

    Context->NumberOfCpuThreads = (
        Context->MaximumConcurrency -
        Context->CuConcurrency
    );

    //
    // Initialize the number of graphs to use for CPU/GPU solving.  Initially,
    // this will match the desired respective concurrency level.
    //

    Context->NumberOfGpuGraphs = Context->CuConcurrency;
    Context->NumberOfCpuGraphs = Context->NumberOfCpuThreads;

    if (FindBestGraph(Context)) {

        //
        // Double the graph count if we're in "find best graph" mode to account
        // for the spare graphs (one per solve context).
        //

        Context->NumberOfGpuGraphs *= 2;

        //
        // Only increment the number of CPU graphs if the number of CPU threads
        // is greater than zero.  (We only need one extra spare graph for all
        // CPU solver threads; this is a side-effect of the original Chm01 CPu
        // solver implementation.)
        //

        if (Context->NumberOfCpuThreads > 0) {
            Context->NumberOfCpuGraphs += 1;
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

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}

_Must_inspect_result_
_Success_(return >= 0)
HRESULT
CuRuntimeContextProcessOrdinals(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PPH_CU_RUNTIME_CONTEXT CuRuntimeContext
    )
/*++

Routine Description:

    Process the device ordinals optionally supplied via --CuDevices.

    This parameter is a bit quirky: it can be a single value or list of comma-
    separated values.  Each value represents a device ordinal, and any device
    ordinal can appear one or more times.  The number of *unique* ordinals
    dictates the number of CUDA contexts we create.  (We only want one context
    per device; multiple contexts would impede performance.)

    If only one device ordinal is supplied, then all GPU solver threads will
    use this device.  If more than one ordinal is supplied, there must be at
    least two unique ordinals present in the entire set.  E.g.:

         Valid:      --CuDevices=0,1
         Invalid:    --CuDevices=0,0

    Additionally, if more than one ordinal is supplied, the dependent params
    like --CuDevicesBlocksPerGrid and --CuDevicesThreadsPerBlock must have
    the same number of values supplied.  E.g.:

         Valid:      --CuDevices=0,1 --CuDevicesBlocksPerGrid=32,16
         Invalid:    --CuDevices=0,1 --CuDevicesBlocksPerGrid=32

    In this situation, the order of the device ordinal in the value list will
    be correlated with the identically-offset value in the dependent list.
    In the example above, the CUDA contexts for devices 0 and 1 will use 32
    and 16 respectively as their blocks-per-grid value.


Arguments:

    Context - Supplies a pointer to a PERFECT_HASH_CONTEXT structure.

    CuRuntimeContext - Supplies a pointer to a CU_RUNTIME_CONTEXT structure.

Return Value:

    HRESULT - S_OK on success, appropriate HRESULT otherwise.

--*/
{
    PCU Cu;
    PRTL Rtl;
    ULONG Index;
    LONG Ordinal;
    HRESULT Result;
    RTL_BITMAP Bitmap;
    CU_RESULT CuResult;
    CU_DEVICE DeviceId;
    CU_DEVICE MinDeviceId;
    CU_DEVICE MaxDeviceId;
    ULONG NumberOfDevices;
    ULONG NumberOfContexts;
    PVALUE_ARRAY Ordinals;
    PVALUE_ARRAY BlocksPerGrid;
    PVALUE_ARRAY ThreadsPerBlock;
    PVALUE_ARRAY KernelRuntimeTarget;
    ULARGE_INTEGER BitmapBufferSizeInBytes;
    PALLOCATOR Allocator = NULL;
    PULONG BitmapBuffer = NULL;

    //
    // Initialize local aliases.
    //

    Ordinals = CuRuntimeContext->Ordinals;
    BlocksPerGrid = CuRuntimeContext->BlocksPerGrid;
    ThreadsPerBlock = CuRuntimeContext->ThreadsPerBlock;
    KernelRuntimeTarget = CuRuntimeContext->KernelRuntimeTarget;

    //
    // First, if --CuDevices (local variable `Ordinals`) has not been supplied,
    // verify no dependent params are present.
    //

    if (Ordinals == NULL) {

        if (BlocksPerGrid != NULL) {
            Result = PH_E_CU_BLOCKS_PER_GRID_REQUIRES_CU_DEVICES;
            goto Error;
        }

        if (ThreadsPerBlock != NULL) {
            Result = PH_E_CU_THREADS_PER_BLOCK_REQUIRES_CU_DEVICES;
            goto Error;
        }

        if (KernelRuntimeTarget != NULL) {
            Result = PH_E_CU_KERNEL_RUNTIME_TARGET_IN_MILLISECONDS_REQUIRES_CU_DEVICES;
            goto Error;
        }

        //
        // We default the number of contexts and devices to 1 in the absence of any
        // user-supplied values.
        //

        NumberOfContexts = 1;
        NumberOfDevices = 1;
        goto FinalizeOrdinalsProcessing;
    }

    //
    // Ordinals have been supplied.  Verify the number of values matches the
    // supplied value for --CuConcurrency, then verify that if any dependent
    // parameters have been supplied, they have the same number of values.
    //

#if 0
    if (0 && Context->CuConcurrency != Ordinals->NumberOfValues) {
        Result = PH_E_CU_DEVICES_COUNT_MUST_MATCH_CU_CONCONCURRENCY;
        goto Error;
    }
#endif

    if ((BlocksPerGrid != NULL) &&
        (BlocksPerGrid->NumberOfValues != Ordinals->NumberOfValues))
    {
        Result = PH_E_CU_BLOCKS_PER_GRID_COUNT_MUST_MATCH_CU_DEVICES_COUNT;
        goto Error;
    }

    if ((ThreadsPerBlock != NULL) &&
        (ThreadsPerBlock->NumberOfValues != Ordinals->NumberOfValues))
    {
        Result = PH_E_CU_THREADS_PER_BLOCK_COUNT_MUST_MATCH_CU_DEVICES_COUNT;
        goto Error;
    }

    if ((KernelRuntimeTarget != NULL) &&
        (KernelRuntimeTarget->NumberOfValues != Ordinals->NumberOfValues))
    {
        Result = PH_E_CU_KERNEL_RUNTIME_TARGET_IN_MILLISECONDS_COUNT_MUST_MATCH_CU_DEVICES_COUNT;
        goto Error;
    }

    //
    // Initialize the min and max device IDs, then enumerate the supplied
    // ordinals, validating each one as we go and updating the min/max values
    // accordingly.
    //

    MinDeviceId = 1 << 30;
    MaxDeviceId = 0;
    Cu = CuRuntimeContext->Cu;

    for (Index = 0; Index < Ordinals->NumberOfValues; Index++) {
        Ordinal = (LONG)Ordinals->Values[Index];
        CuResult = Cu->DeviceGet(&DeviceId, Ordinal);
        if (CU_FAILED(CuResult)) {
            CU_ERROR(CuDeviceGet, CuResult);
            Result = PH_E_INVALID_CU_DEVICES;
            goto Error;
        }
        if (DeviceId > MaxDeviceId) {
            MaxDeviceId = DeviceId;
        }
        if (DeviceId < MinDeviceId) {
            MinDeviceId = DeviceId;
        }
    }

    //
    // We use a bitmap to count the number of unique devices supplied in the
    // --CuDevices parameter.  Calculate the bitmap buffer size in bytes.
    //

    BitmapBufferSizeInBytes.QuadPart = ALIGN_UP_POINTER(
        ALIGN_UP((MaxDeviceId + 1ULL), 8) >> 3
    );

    //
    // Sanity check we haven't overflowed.
    //

    if (BitmapBufferSizeInBytes.HighPart != 0) {
        Result = PH_E_TOO_MANY_BITS_FOR_BITMAP;
        goto Error;
    }

    ASSERT(BitmapBufferSizeInBytes.LowPart > 0);

    //
    // Allocate sufficient bitmap buffer space.
    //

    Allocator = Cu->Allocator;
    BitmapBuffer = Allocator->Vtbl->Calloc(Allocator,
                                           1,
                                           BitmapBufferSizeInBytes.LowPart);
    if (BitmapBuffer == NULL) {
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    //
    // Wire-up the device bitmap.
    //

    Bitmap.Buffer = BitmapBuffer;
    Bitmap.SizeOfBitMap = (MaxDeviceId + 1);

    //
    // Enumerate the ordinals again, setting a corresponding bit for each
    // ordinal we see.
    //

    for (Index = 0; Index < Ordinals->NumberOfValues; Index++) {
        Ordinal = (LONG)Ordinals->Values[Index];
        ASSERT(Ordinal >= 0);
        _Analysis_assume_(Ordinal >= 0);
        SetBit32(Bitmap.Buffer, Ordinal);
    }

    //
    // Count the number of bits set, this will represent the number of unique
    // devices we encountered.  Sanity check the number doesn't exceed the
    // total number of devices reported in the system.
    //

    Rtl = Context->Rtl;
#if 0
    NumberOfContexts = Rtl->RtlNumberOfSetBits(&Bitmap);
    NumberOfDevices = CuRuntimeContext->CuDevices.NumberOfDevices;
#else
    NumberOfContexts = Context->CuConcurrency;
    NumberOfDevices = CuRuntimeContext->CuDevices.NumberOfDevices;
#endif

    //
    // Intentional follow-on to FinalizeOrdinalsProcessing.
    //

FinalizeOrdinalsProcessing:

#if 0
    if (NumberOfContexts > NumberOfDevices) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(PerfectHashContextInitializeCuda_SetBitsExceedsNumDevices,
                 Result);
        goto Error;
    } else if (NumberOfContexts == 0) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(PerfectHashContextInitializeCuda_NumContextsIsZero, Result);
        goto Error;
    }
#endif

    //Context->NumberOfCuContexts = NumberOfContexts;

    //
    // Copy local aliases back to the runtime context.
    //

    CuRuntimeContext->NumberOfContexts = NumberOfContexts;
    CuRuntimeContext->NumberOfDevices = NumberOfDevices;

    //
    // We're done, indicate success and finish up.
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

    //
    // If we allocated a bitmap buffer, free it now.
    //

    if (BitmapBuffer != NULL) {
        Allocator->Vtbl->FreePointer(Allocator, &BitmapBuffer);
    }

    return Result;
}

_Must_inspect_result_
_Success_(return >= 0)
HRESULT
CuRuntimeContextInitializeDevices(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PPH_CU_RUNTIME_CONTEXT CuRuntimeContext
    )
/*++

Routine Description:

    Initializes the devices for the supplied CU_RUNTIME_CONTEXT structure.

Arguments:

    Context - Supplies a pointer to a PERFECT_HASH_CONTEXT structure.

    CuRuntimeContext - Supplies a pointer to a CU_RUNTIME_CONTEXT structure.

Return Value:

    HRESULT - S_OK on success, appropriate HRESULT otherwise.

--*/
{
    PRTL Rtl;
    ULONG Index;
    HRESULT Result;
    RTL_BITMAP Bitmap;
    PALLOCATOR Allocator;
    PVALUE_ARRAY Ordinals;
    ULONG NumberOfContexts;
    ULARGE_INTEGER AllocSizeInBytes;
    PPH_CU_DEVICE_CONTEXT DeviceContext;
    PPH_CU_DEVICE_CONTEXTS DeviceContexts = NULL;

    //
    // Initialize local aliases.
    //

    Rtl = Context->Rtl;
    Allocator = Context->Allocator;
    NumberOfContexts = CuRuntimeContext->NumberOfContexts;

    //
    // Allocate memory for the device contexts structs.
    //

    AllocSizeInBytes.QuadPart = sizeof(*CuRuntimeContext->CuDeviceContexts);

    if (NumberOfContexts > 1) {

        //
        // Account for additional device context structures if we're creating
        // more than one.  (We get one for free via ANYSIZE_ARRAY.)
        //

        AllocSizeInBytes.QuadPart += (
            (NumberOfContexts - 1) *
            sizeof(CuRuntimeContext->CuDeviceContexts->DeviceContexts[0])
        );

        if (FindBestGraph(Context)) {

            //
            // Sanity check our graph counts line up.
            //

            ASSERT((NumberOfContexts * 2) == Context->NumberOfGpuGraphs);
        }
    }

    //
    // Sanity check we haven't overflowed.
    //

    if (AllocSizeInBytes.HighPart > 0) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(CuRuntimeContextInitializeDevices_Overflow, Result);
        PH_RAISE(Result);
    }

    DeviceContexts = Allocator->Vtbl->Calloc(Allocator,
                                             1,
                                             AllocSizeInBytes.LowPart);
    if (DeviceContexts == NULL) {
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    CuRuntimeContext->CuDeviceContexts = DeviceContexts;
    DeviceContexts->NumberOfDeviceContexts = NumberOfContexts;

    Ordinals = CuRuntimeContext->Ordinals;

    //
    // First pass: set each device context's ordinal to the value obtained via
    // the --CuDevices parameter.  (The logic we use to do this is a little
    // different if we're dealing with one context versus more than one.)
    //

    if (NumberOfContexts == 1) {

        DeviceContext = &DeviceContexts->DeviceContexts[0];
        DeviceContext->Rtl = Rtl;

        if (Ordinals != NULL) {
            ASSERT(Ordinals->NumberOfValues == 1);
            DeviceContext->Ordinal = (LONG)Ordinals->Values[0];
        } else {

            //
            // If no --CuDevices parameter has been supplied, default to 0 for
            // the device ordinal.
            //

            DeviceContext->Ordinal = 0;
        }

    } else if (NumberOfContexts > 1 && Ordinals != NULL &&
               Ordinals->NumberOfValues == 1) {

        //
        // Multiple contexts for a single device.
        //

        for (Index = 0; Index < NumberOfContexts; Index++) {
            DeviceContext = &DeviceContexts->DeviceContexts[Index];
            DeviceContext->Rtl = Rtl;
            DeviceContext->Ordinal = (LONG)Ordinals->Values[0];
        }

    } else {

        ULONG Bit = 0;
        const ULONG FindOneBit = 1;

        for (Index = 0; Index < NumberOfContexts; Index++) {
            DeviceContext = &DeviceContexts->DeviceContexts[Index];
            DeviceContext->Rtl = Rtl;

            //
            // Get the device ordinal from the first set/next set bit of the
            // bitmap.
            //

            Bit = Rtl->RtlFindSetBits(&Bitmap, FindOneBit, Bit);

            if (Bit == BITS_NOT_FOUND) {
                Result = PH_E_INVARIANT_CHECK_FAILED;
                PH_ERROR(PerfectHashContextInitializeCuda_BitsNotFound,
                         Result);
                PH_RAISE(Result);
            }

            DeviceContext->Ordinal = (LONG)Bit;
            Bit += 1;
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

    //
    // With the way this routine is currently written, we shouldn't hit this
    // code with a non-NULL DeviceContexts pointer.  Assert this, but also
    // include the free logic in case the situation changes.
    //

    if (DeviceContexts != NULL) {
        ASSERT(DeviceContexts == NULL);
        Allocator->Vtbl->FreePointer(Allocator, &DeviceContexts);
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
CuDeviceContextInitializeKernels(
    _In_ PPH_CU_DEVICE_CONTEXT DeviceContext
    )
{
    PCU Cu;
    HRESULT Result;
    CU_RESULT CuResult;
    PPH_CU_KERNEL Kernel;
    PCU_OCCUPANCY Occupancy;
    CU_STREAM_FLAGS StreamFlags;

    Cu = DeviceContext->Cu;
    StreamFlags = CU_STREAM_NON_BLOCKING;

#ifdef PH_WINDOWS
    //
    // Required for DO_OUTPUT/PRINT_CSTR.
    //

    BOOLEAN Silent = FALSE;
    DWORD BytesWritten = 0;
    HANDLE OutputHandle = DeviceContext->Rtl->SysErrorOutputHandle;
#endif

#define EXPAND_AS_INIT_CUDA_KERNEL(Name_, BlocksPerGrid_, ThreadsPerBlock_, \
                                   RuntimeTargetInMilliseconds_)            \
    Kernel = &DeviceContext->Name_##Kernel;                                 \
    Kernel->Id = PerfectHashCudaKernel##Name_##Id;                          \
    Kernel->Name = PerfectHashGetCudaKernelName(Kernel->Id);                \
    Kernel->BlocksPerGrid = BlocksPerGrid_;                                 \
    Kernel->ThreadsPerBlock = ThreadsPerBlock_;                             \
    Kernel->RuntimeTargetInMilliseconds = RuntimeTargetInMilliseconds_;     \
    CuResult = Cu->ModuleGetFunction(&Kernel->Function,                     \
                                     DeviceContext->Module,                 \
                                     Kernel->Name->Buffer);                 \
    CU_LINK_CHECK(CuResult, CuModuleGetFunction_##Name_);                   \
                                                                            \
    Occupancy = &Kernel->Occupancy;                                         \
    CuResult = Cu->OccupancyMaxPotentialBlockSizeWithFlags(                 \
        &Occupancy->MinimumGridSize,                                        \
        &Occupancy->BlockSize,                                              \
        Kernel->Function,                                                   \
        NULL,                                                               \
        0,                                                                  \
        0,                                                                  \
        0                                                                   \
    );                                                                      \
    CU_CHECK(CuResult,                                                      \
             OccupancyMaxPotentialBlockSizeWithFlags_##Name_);              \
                                                                            \
    CuResult = Cu->OccupancyMaxActiveBlocksPerMultiprocessorWithFlags(      \
        &Occupancy->NumBlocks,                                              \
        Kernel->Function,                                                   \
        Occupancy->BlockSize,                                               \
        0,                                                                  \
        0                                                                   \
    );                                                                      \
    CU_CHECK(CuResult,                                                      \
             OccupancyMaxActiveBlocksPerMultiprocessorWithFlags_##Name_);   \
                                                                            \
    CuResult = Cu->StreamCreate(&Kernel->Stream, StreamFlags);              \
    CU_CHECK(CuResult, StreamCreate_##Name_);

    PERFECT_HASH_CUDA_KERNELS_TABLE_ENTRY(EXPAND_AS_INIT_CUDA_KERNEL)

    //
    // Finally, create a stream for the device context.
    //

    CuResult = Cu->StreamCreate(&DeviceContext->Stream, StreamFlags);
    CU_CHECK(CuResult, StreamCreate_DeviceContext);

    //
    // We're done, indicate success and finish up.
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

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
