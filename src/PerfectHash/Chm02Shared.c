/*++

Copyright (c) 2023 Trent Nelson <trent@trent.me>

Module Name:

    Chm02Shared.c

Abstract:

    Logic shared between the PH_WINDOWS and PH_COMPAT Chm02 implementations.

--*/

#include "stdafx.h"
#include "Chm01.h"
#include "Chm02Private.h"
#include "Graph_Ptx_RawCString.h"

_Use_decl_annotations_
HRESULT
InitializeCudaAndGraphsChm02(
    PPERFECT_HASH_CONTEXT Context,
    PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    )
/*++

Routine Description:

    Attempts to initialize CUDA and all supporting graphs for the given context.

Arguments:

    Context - Supplies a pointer to the PERFECT_HASH_CONTEXT instance for which
        the routine will try and initialize CUDA and all supporting graphs.

    TableCreateParameters - Supplies a pointer to the table create parameters.

Return Value:

    S_OK - Initialized successfully.

    S_FALSE - Already initialized.

    Otherwise, an appropriate error code.

--*/
{
    PCU Cu;
    PRTL Rtl;
    ULONG Index;
    ULONG Inner;
    ULONG Count;
    LONG Ordinal;
    BOOLEAN Found;
    ULONG NumberOfDevices;
    ULONG NumberOfContexts;
    PULONG BitmapBuffer = NULL;
    RTL_BITMAP Bitmap;
    HRESULT Result;
    PCHAR PtxString;
    CU_RESULT CuResult;
    CU_DEVICE DeviceId;
    CU_DEVICE MinDeviceId;
    CU_DEVICE MaxDeviceId;
    PALLOCATOR Allocator;
    PGRAPH Graph;
    PGRAPH *Graphs;
    PGRAPH *CpuGraphs;
    PGRAPH *GpuGraphs;
    PGRAPH DeviceGraph;
    PGRAPH DeviceGraphs;
    PCHAR LinkedModule;
    SIZE_T LinkedModuleSizeInBytes;
    SIZE_T PtxSizeInBytes;
    PPH_CU_DEVICE Device;
    PCU_OCCUPANCY Occupancy;
    PCU_LINK_STATE LinkState;
    BOOLEAN SawCuRngSeed;
    BOOLEAN SawCuConcurrency;
    BOOLEAN WantsRandomHostSeeds;
    BOOLEAN IsRngImplemented;
    PUNICODE_STRING CuPtxPath;
    PUNICODE_STRING CuCudaDevRuntimeLibPath;
    ULONG NumberOfGpuGraphs;
    ULONG NumberOfCpuGraphs;
    ULONG TotalNumberOfGraphs;
    ULONG NumberOfRandomHostSeeds;
    ULONG SpareGraphCount;
    ULONG MatchedGraphCount;
    ULONG BlocksPerGridValue;
    ULONG ThreadsPerBlockValue;
    ULONG KernelRuntimeTargetValue;
    ULONG NumberOfGraphsForDevice;
    ULONG NumberOfSolveContexts;
    PVALUE_ARRAY Ordinals;
    PVALUE_ARRAY BlocksPerGrid;
    PVALUE_ARRAY ThreadsPerBlock;
    PVALUE_ARRAY KernelRuntimeTarget;
    CU_STREAM_FLAGS StreamFlags;
    ULARGE_INTEGER AllocSizeInBytes;
    ULARGE_INTEGER BitmapBufferSizeInBytes;
    PPH_CU_DEVICE_CONTEXT DeviceContext;
    PPH_CU_DEVICE_CONTEXTS DeviceContexts;
    PPH_CU_SOLVE_CONTEXT SolveContext;
    PPH_CU_SOLVE_CONTEXTS SolveContexts;
    //PERFECT_HASH_CU_RNG_ID CuRngId = PerfectHashCuNullRngId;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_PATH PtxPath;
    PPERFECT_HASH_FILE PtxFile;
    PPERFECT_HASH_PATH RuntimeLibPath;
    PPERFECT_HASH_FILE RuntimeLibFile;
    PPERFECT_HASH_TABLE_CREATE_PARAMETER Param;
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags;
    PERFECT_HASH_FILE_LOAD_FLAGS FileLoadFlags = { 0 };
    LARGE_INTEGER EndOfFile = { 0 };

#ifdef PH_WINDOWS
    //
    // Required for DO_OUTPUT/PRINT_CSTR.
    //

    BOOLEAN Silent = FALSE;
    DWORD BytesWritten = 0;
    HANDLE OutputHandle = Context->Rtl->SysErrorOutputHandle;
#endif

    STRING KernelFunctionName =
        RTL_CONSTANT_STRING("PerfectHashCudaEnterSolvingLoop");

    CU_JIT_OPTION JitOptions[] = {
        CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
#ifdef _DEBUG
        CU_JIT_GENERATE_DEBUG_INFO,
#else
        CU_JIT_GENERATE_LINE_INFO,
#endif
        CU_JIT_LOG_VERBOSE,
    };

    PVOID JitOptionValues[ARRAYSIZE(JitOptions)];
    USHORT NumberOfJitOptions = ARRAYSIZE(JitOptions);

    CHAR JitInfoLogBuffer[PERFECT_HASH_CU_JIT_LOG_BUFFER_SIZE_IN_BYTES];
    CHAR JitErrorLogBuffer[PERFECT_HASH_CU_JIT_LOG_BUFFER_SIZE_IN_BYTES];

    //
    // If we've already got a CU instance, assume we're already initialized.
    //

    if (Context->Cu != NULL) {
        return S_FALSE;
    }

    PtxFile = NULL;
    PtxPath = NULL;
    CuPtxPath = NULL;
    SolveContexts = NULL;
    DeviceContexts = NULL;
    RuntimeLibPath = NULL;
    RuntimeLibFile = NULL;
    CuCudaDevRuntimeLibPath = NULL;

    Table = Context->Table;
    TableCreateFlags.AsULongLong = Table->TableCreateFlags.AsULongLong;

    //
    // Try create a CU instance.
    //

    Result = Context->Vtbl->CreateInstance(Context,
                                           NULL,
                                           &IID_PERFECT_HASH_CU,
                                           &Context->Cu);

    if (FAILED(Result)) {
        goto Error;
    }

    //
    // The CU component is a global component, which means it can't create
    // instances of other global components like Rtl and Allocator during its
    // initialization function.  So, we manually set them now.
    //

    Cu = Context->Cu;

    Rtl = Cu->Rtl = Context->Rtl;
    Cu->Rtl->Vtbl->AddRef(Cu->Rtl);

    Cu->Allocator = Allocator = Context->Allocator;
    Cu->Allocator->Vtbl->AddRef(Cu->Allocator);

    Result = CreatePerfectHashCuDevices(Cu,
                                        Cu->Allocator,
                                        &Context->CuDevices);
    if (FAILED(Result)) {
        PH_ERROR(CreatePerfectHashCuDevices, Result);
        goto Error;
    }

    Count = TableCreateParameters->NumberOfElements;
    Param = TableCreateParameters->Params;

    //
    // Clear our local aliases.
    //

    Ordinals = NULL;
    BlocksPerGrid = NULL;
    ThreadsPerBlock = NULL;
    KernelRuntimeTarget = NULL;
    SawCuConcurrency = FALSE;
    SawCuRngSeed = FALSE;

    //
    // Disable "enum not handled in switch statement" warning.
    //
    //      warning C4061: enumerator 'TableCreateParameterNullId' in switch
    //                     of enum 'PERFECT_HASH_TABLE_CREATE_PARAMETER_ID'
    //                     is not explicitly handled by a case label
    //

#pragma warning(push)
#pragma warning(disable: 4061)

    for (Index = 0; Index < Count; Index++, Param++) {

        switch (Param->Id) {

            case TableCreateParameterCuRngId:
                Context->CuRngId = Param->AsCuRngId;
                break;

            case TableCreateParameterCuRngSeedId:
                Context->CuRngSeed = Param->AsULongLong;
                SawCuRngSeed = TRUE;
                break;

            case TableCreateParameterCuRngSubsequenceId:
                Context->CuRngSubsequence = Param->AsULongLong;
                break;

            case TableCreateParameterCuRngOffsetId:
                Context->CuRngOffset = Param->AsULongLong;
                break;

            case TableCreateParameterCuConcurrencyId:
                Context->CuConcurrency = Param->AsULong;
                SawCuConcurrency = TRUE;
                break;

            case TableCreateParameterCuDevicesId:
                Ordinals = &Param->AsValueArray;
                break;

            case TableCreateParameterCuDevicesBlocksPerGridId:
                BlocksPerGrid = &Param->AsValueArray;
                break;

            case TableCreateParameterCuDevicesThreadsPerBlockId:
                ThreadsPerBlock = &Param->AsValueArray;
                break;

            case TableCreateParameterCuDevicesKernelRuntimeTargetInMillisecondsId:
                KernelRuntimeTarget = &Param->AsValueArray;
                break;

            case TableCreateParameterCuPtxPathId:
                CuPtxPath = &Param->AsUnicodeString;
                break;

            case TableCreateParameterCuCudaDevRuntimeLibPathId:
                CuCudaDevRuntimeLibPath = &Param->AsUnicodeString;
                break;

            case TableCreateParameterCuNumberOfRandomHostSeedsId:
                NumberOfRandomHostSeeds = Param->AsULong;
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

    if (!IsValidPerfectHashCuRngId(Context->CuRngId)) {
        Context->CuRngId = CU_RNG_DEFAULT;
    }

    Result = PerfectHashLookupNameForId(Rtl,
                                        PerfectHashCuRngEnumId,
                                        Context->CuRngId,
                                        &Context->CuRngName);
    if (FAILED(Result)) {
        PH_ERROR(InitializeCudaAndGraphsChm02_LookupNameForId, Result);
        goto Error;
    }

    IsRngImplemented = FALSE;

#define EXPAND_AS_CU_RNG_ID_CASE(Name, Upper, Implemented) \
    case PerfectHashCuRng##Name##Id:                       \
        IsRngImplemented = Implemented;                    \
        break;

    switch (Context->CuRngId) {

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
        Context->CuRngSeed = RNG_DEFAULT_SEED;
    }

    //
    // Validate --CuConcurrency.  It's mandatory, it must be greater than zero,
    // and less than or equal to the maximum concurrency.  (When CuConcurrency
    // is less than max concurrency, the difference between the two will be the
    // number of CPU solving threads launched.  E.g. if --CuConcurrency=16 and
    // max concurrency is 18; there will be two CPU solving threads launched in
    // addition to the 16 GPU solver threads.)
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

    if (CuCudaDevRuntimeLibPath == NULL) {
        Result = PH_E_CU_CUDA_DEV_RUNTIME_LIB_PATH_MANDATORY;
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
    // Validate device ordinals optionally supplied via --CuDevices.  This
    // parameter is a bit quirky: it can be a single value or list of comma-
    // separated values.  Each value represents a device ordinal, and any
    // device ordinal can appear one or more times.  The number of *unique*
    // ordinals dictates the number of CUDA contexts we create.  (We only want
    // one context per device; multiple contexts would impede performance.)
    //
    // If only one device ordinal is supplied, then all GPU solver threads will
    // use this device.  If more than one ordinal is supplied, there must be at
    // least two unique ordinals present in the entire set.  E.g.:
    //
    //      Valid:      --CuDevices=0,1
    //      Invalid:    --CuDevices=0,0
    //
    // Additionally, if more than one ordinal is supplied, the dependent params
    // like --CuDevicesBlocksPerGrid and --CuDevicesThreadsPerBlock must have
    // the same number of values supplied.  E.g.:
    //
    //      Valid:      --CuDevices=0,1 --CuDevicesBlocksPerGrid=32,16
    //      Invalid:    --CuDevices=0,1 --CuDevicesBlocksPerGrid=32
    //
    // In this situation, the order of the device ordinal in the value list will
    // be correlated with the identically-offset value in the dependent list.
    // In the example above, the CUDA contexts for devices 0 and 1 will use 32
    // and 16 respectively as their blocks-per-grid value.
    //

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
        goto FinishedOrdinalsProcessing;

    }

    //
    // Ordinals have been supplied.  Verify the number of values matches the
    // supplied value for --CuConcurrency, then verify that if any dependent
    // parameters have been supplied, they have the same number of values.
    //

    if (Context->CuConcurrency != Ordinals->NumberOfValues) {
        Result = PH_E_CU_DEVICES_COUNT_MUST_MATCH_CU_CONCONCURRENCY;
        goto Error;
    }

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
    NumberOfContexts = Rtl->RtlNumberOfSetBits(&Bitmap);
    NumberOfDevices = Context->CuDevices.NumberOfDevices;

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

    Context->NumberOfCuContexts = NumberOfContexts;

    //
    // Intentional follow-on to FinishedOrdinalsProcessing.
    //

FinishedOrdinalsProcessing:

    //
    // Allocate memory for the device contexts structs.
    //

    AllocSizeInBytes.QuadPart = sizeof(*Context->CuDeviceContexts);

    if (NumberOfContexts > 1) {

        //
        // Account for additional device context structures if we're creating
        // more than one.  (We get one for free via ANYSIZE_ARRAY.)
        //

        AllocSizeInBytes.QuadPart += (
            (NumberOfContexts - 1) *
            sizeof(Context->CuDeviceContexts->DeviceContexts[0])
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
        PH_ERROR(PerfectHashContextInitializeCuda_DeviceContextAllocOverflow,
                 Result);
        PH_RAISE(Result);
    }

    DeviceContexts = Allocator->Vtbl->Calloc(Allocator,
                                             1,
                                             AllocSizeInBytes.LowPart);
    if (DeviceContexts == NULL) {
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    Context->CuDeviceContexts = DeviceContexts;
    DeviceContexts->NumberOfDeviceContexts = NumberOfContexts;

    //
    // First pass: set each device context's ordinal to the value obtained via
    // the --CuDevices parameter.  (The logic we use to do this is a little
    // different if we're dealing with one context versus more than one.)
    //

    if (NumberOfContexts == 1) {

        DeviceContext = &DeviceContexts->DeviceContexts[0];

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

    } else {

        ULONG Bit = 0;
        const ULONG FindOneBit = 1;

        for (Index = 0; Index < NumberOfContexts; Index++) {
            DeviceContext = &DeviceContexts->DeviceContexts[Index];

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

    if (CuPtxPath == NULL) {

        //
        // No --CuPtxPath supplied; use the embedded PTX string.
        //

        PtxString = (PCHAR)GraphPtxRawCStr;
        PtxSizeInBytes = sizeof(GraphPtxRawCStr);

    } else {

        //
        // --CuPtxPath was supplied.  Create a path instance to encapsulate the
        // argument, then a corresponding file object that can be loaded, such
        // that we can access the PTX as a raw C string.
        //

        //
        // Construct a path instance.
        //

        Result = Context->Vtbl->CreateInstance(Context,
                                               NULL,
                                               &IID_PERFECT_HASH_PATH,
                                               &PtxPath);

        if (FAILED(Result)) {
            PH_ERROR(InitializeCudaAndGraphsChm02_CreatePtxPath, Result);
            goto Error;
        }

        Result = PtxPath->Vtbl->Copy(PtxPath, CuPtxPath, NULL, NULL);
        if (FAILED(Result)) {
            PH_ERROR(InitializeCudaAndGraphsChm02_PtxPathCopy, Result);
            goto Error;
        }

        //
        // Create a file instance.
        //

        Result = Context->Vtbl->CreateInstance(Context,
                                               NULL,
                                               &IID_PERFECT_HASH_FILE,
                                               &PtxFile);

        if (FAILED(Result)) {
            PH_ERROR(InitializeCudaAndGraphsChm02_CreatePtxFile, Result);
            goto Error;
        }

        //
        // Load the PTX file (map it into memory).  We can then use the base
        // address as the PTX string.  EndOfFile will capture the PTX size in
        // bytes.
        //

        Result = PtxFile->Vtbl->Load(PtxFile,
                                     PtxPath,
                                     &EndOfFile,
                                     &FileLoadFlags);
        if (FAILED(Result)) {
            PH_ERROR(InitializeCudaAndGraphsChm02_LoadPtxFile, Result);
            goto Error;
        }

        PtxString = (PCHAR)PtxFile->BaseAddress;
        PtxSizeInBytes = EndOfFile.QuadPart;

        RELEASE(PtxPath);
    }

    //
    // Open the cudadevrt.lib path.
    //

    Result = Context->Vtbl->CreateInstance(Context,
                                           NULL,
                                           &IID_PERFECT_HASH_PATH,
                                           &RuntimeLibPath);

    if (FAILED(Result)) {
        PH_ERROR(InitializeCudaAndGraphsChm02_CreateRuntimeLibPath, Result);
        goto Error;
    }

    Result = RuntimeLibPath->Vtbl->Copy(RuntimeLibPath,
                                        CuCudaDevRuntimeLibPath,
                                        NULL,
                                        NULL);
    if (FAILED(Result)) {
        PH_ERROR(InitializeCudaAndGraphsChm02_RuntimeLibPathCopy, Result);
        goto Error;
    }

    //
    // Create a file instance.
    //

    Result = Context->Vtbl->CreateInstance(Context,
                                           NULL,
                                           &IID_PERFECT_HASH_FILE,
                                           &RuntimeLibFile);

    if (FAILED(Result)) {
        PH_ERROR(InitializeCudaAndGraphsChm02_CreateRuntimeLibFile, Result);
        goto Error;
    }

    EndOfFile.QuadPart = 0;
    Result = RuntimeLibFile->Vtbl->Load(RuntimeLibFile,
                                        RuntimeLibPath,
                                        &EndOfFile,
                                        &FileLoadFlags);

    if (FAILED(Result)) {
        PH_ERROR(InitializeCudaAndGraphsChm02_LoadRuntimeLibFile, Result);
        goto Error;
    }

    //
    // Initialize the JIT options.
    //

    JitOptionValues[0] = (PVOID)JitInfoLogBuffer;
    JitOptionValues[1] = (PVOID)sizeof(JitInfoLogBuffer);

    JitOptionValues[2] = (PVOID)JitErrorLogBuffer;
    JitOptionValues[3] = (PVOID)sizeof(JitErrorLogBuffer);

    //
    // Debug or lineinfo mode.
    //

    JitOptionValues[4] = (PVOID)1;

    //
    // Verbose linking.
    //

    JitOptionValues[5] = (PVOID)1;

    //
    // Second pass: wire-up each device context (identified by ordinal, set
    // in the first pass above) to the corresponding PH_CU_DEVICE instance
    // for that device, create CUDA contexts for each device context, load
    // the module, get the solver entry function, and calculate occupancy.
    //

    Device = NULL;
    StreamFlags = CU_STREAM_NON_BLOCKING;

    for (Index = 0; Index < NumberOfContexts; Index++) {
        DeviceContext = &DeviceContexts->DeviceContexts[Index];

        //
        // Find the PH_CU_DEVICE instance with the same ordinal.
        //

        Found = FALSE;
        Device = NULL;
        for (Inner = 0; Inner < NumberOfDevices; Inner++) {
            Device = &Context->CuDevices.Devices[Inner];
            if (Device->Ordinal == DeviceContext->Ordinal) {
                DeviceContext->Device = Device;
                Found = TRUE;
                break;
            }
        }

        if (!Found) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(PerfectHashContextInitializeCuda_OrdinalNotFound, Result);
            PH_RAISE(Result);
        }

        ASSERT(Device != NULL);

        DeviceContext->Handle = Device->Handle;
        DeviceContext->Cu = Cu;

        //
        // Create the context for the device.
        //

        CuResult = Cu->CtxCreate(&DeviceContext->Context,
                                 CU_CTX_SCHED_BLOCKING_SYNC,
                                 Device->Handle);
        CU_CHECK(CuResult, CtxCreate);

#ifndef PH_WINDOWS
#define CU_LINK_CHECK(CuResult, Name)                   \
    if (CU_FAILED(CuResult)) {                          \
        CU_ERROR(Name, CuResult);                       \
        fprintf(stderr, "%s\n", &JitErrorLogBuffer[0]); \
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;      \
        goto Error;                                     \
    }
#else
#define CU_LINK_CHECK(CuResult, Name)              \
    if (CU_FAILED(CuResult)) {                     \
        CU_ERROR(Name, CuResult);                  \
        PRINT_CSTR(&JitErrorLogBuffer[0]);         \
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED; \
        goto Error;                                \
    }
#endif

        //
        // Our solver kernel uses dynamic parallelism (that is, it launches
        // other kernels).  For this to work, we can't just load the PTX
        // directly; we need to perform a linking step which also adds the
        // cudadevrt.lib (CUDA device runtime static library) into the mix.
        // We do this by issuing a cuLinkCreate(), cuLinkAddData() for the
        // static .lib and our .ptx string, then cuLinkComplete() to get the
        // final module that we can pass to cuLoadModuleEx().
        //

        CuResult = Cu->LinkCreate(NumberOfJitOptions,
                                  JitOptions,
                                  JitOptionValues,
                                  &LinkState);
        CU_CHECK(CuResult, LinkCreate);

        //
        // Add cudadevrt.lib.
        //

        CuResult = Cu->LinkAddData(LinkState,
                                   CU_JIT_INPUT_LIBRARY,
                                   RuntimeLibFile->BaseAddress,
                                   EndOfFile.QuadPart,
                                   "cudadevrt.lib",
                                   0,
                                   NULL,
                                   NULL);
        CU_LINK_CHECK(CuResult, LinkAddData);

        //
        // Add the PTX file.
        //

        CuResult = Cu->LinkAddData(LinkState,
                                   CU_JIT_INPUT_PTX,
                                   PtxString,
                                   PtxSizeInBytes,
                                   "Graph.ptx",
                                   0,
                                   NULL,
                                   NULL);
        CU_LINK_CHECK(CuResult, LinkAddData);

        //
        // Complete the link.
        //

        CuResult = Cu->LinkComplete(LinkState,
                                    &LinkedModule,
                                    &LinkedModuleSizeInBytes);
        CU_LINK_CHECK(CuResult, LinkComplete);

        //
        // Load the module from the embedded PTX.
        //

        CuResult = Cu->ModuleLoadDataEx(&DeviceContext->Module,
                                        LinkedModule,
                                        NumberOfJitOptions,
                                        JitOptions,
                                        JitOptionValues);
        CU_LINK_CHECK(CuResult, ModuleLoadDataEx);

        //
        // Module loaded successfully, resolve the kernel.
        //

        CuResult = Cu->ModuleGetFunction(&DeviceContext->Function,
                                         DeviceContext->Module,
                                         (PCSZ)KernelFunctionName.Buffer);
        CU_LINK_CHECK(CuResult, ModuleGetFunction);

        //
        // We can now destroy the linker state.
        //

        CuResult = Cu->LinkDestroy(LinkState);
        CU_CHECK(CuResult, LinkDestroy);

        //
        // Get the occupancy stats.
        //

        Occupancy = &DeviceContext->Occupancy;
        CuResult = Cu->OccupancyMaxPotentialBlockSizeWithFlags(
            &Occupancy->MinimumGridSize,
            &Occupancy->BlockSize,
            DeviceContext->Function,
            NULL,   // OccupancyBlockSizeToDynamicMemSize
            0,      // DynamicSharedMemorySize
            0,      // BlockSizeLimit
            0       // Flags
        );
        CU_CHECK(CuResult, OccupancyMaxPotentialBlockSizeWithFlags);

        CuResult = Cu->OccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
            &Occupancy->NumBlocks,
            DeviceContext->Function,
            Occupancy->BlockSize,
            0, // DynamicSharedMemorySize
            0  // Flags
        );
        CU_CHECK(CuResult, OccupancyMaxActiveBlocksPerMultiprocessorWithFlags);

        //
        // Create the stream to use for per-device activies (like copying keys).
        //

        CuResult = Cu->StreamCreate(&DeviceContext->Stream, StreamFlags);
        CU_CHECK(CuResult, StreamCreate);

        //
        // Pop the context off this thread (required before it can be used by
        // other threads).
        //

        CuResult = Cu->CtxPopCurrent(NULL);
        CU_CHECK(CuResult, CtxPopCurrent);

    }

    //
    // Allocate space for solver contexts; one per CUDA solving thread.
    //

    AllocSizeInBytes.QuadPart = sizeof(*Context->CuSolveContexts);

    //
    // Account for additional solve context structures if we're creating more
    // than one.  (We get one for free via ANYSIZE_ARRAY.)
    //

    if (Context->CuConcurrency > 1) {
        AllocSizeInBytes.QuadPart += (
            (Context->CuConcurrency - 1) *
            sizeof(Context->CuSolveContexts->SolveContexts[0])
        );
    }

    //
    // Sanity check we haven't overflowed.
    //

    if (AllocSizeInBytes.HighPart > 0) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(PerfectHashContextInitializeCuda_SolveContextAllocOverflow,
                 Result);
        PH_RAISE(Result);
    }

    SolveContexts = Allocator->Vtbl->Calloc(Allocator,
                                            1,
                                            AllocSizeInBytes.LowPart);
    if (SolveContexts == NULL) {
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    Context->CuSolveContexts = SolveContexts;
    SolveContexts->NumberOfSolveContexts = Context->CuConcurrency;
    NumberOfSolveContexts = Context->CuConcurrency;

    //
    // Wire up the solve contexts to their respective device context.
    //

    for (Index = 0; Index < NumberOfSolveContexts; Index++) {

        SolveContext = &SolveContexts->SolveContexts[Index];

        //
        // Resolve the ordinal and kernel launch parameters.
        //

        if (Ordinals == NULL) {
            Ordinal = 0;
        } else {
            Ordinal = (LONG)Ordinals->Values[Index];
        }

        if (BlocksPerGrid == NULL) {
            BlocksPerGridValue = 0;
        } else {
            BlocksPerGridValue = BlocksPerGrid->Values[Index];
        }
        if (BlocksPerGridValue == 0) {
            BlocksPerGridValue = PERFECT_HASH_CU_DEFAULT_BLOCKS_PER_GRID;
        }

        if (ThreadsPerBlock == NULL) {
            ThreadsPerBlockValue = 0;
        } else {
            ThreadsPerBlockValue = ThreadsPerBlock->Values[Index];
        }
        if (ThreadsPerBlockValue == 0) {
            ThreadsPerBlockValue = PERFECT_HASH_CU_DEFAULT_THREADS_PER_BLOCK;
        }

        if (KernelRuntimeTarget == NULL) {
            KernelRuntimeTargetValue = 0;
        } else {
            KernelRuntimeTargetValue = KernelRuntimeTarget->Values[Index];
        }
        if (KernelRuntimeTargetValue == 0) {
            KernelRuntimeTargetValue =
                PERFECT_HASH_CU_DEFAULT_KERNEL_RUNTIME_TARGET_IN_MILLISECONDS;
        }

        //
        // Find the device context for this device ordinal.
        //

        Found = FALSE;
        DeviceContext = NULL;
        for (Inner = 0; Inner < NumberOfContexts; Inner++) {
            DeviceContext = &DeviceContexts->DeviceContexts[Inner];
            if (DeviceContext->Ordinal == Ordinal) {
                Found = TRUE;
                break;
            }
        }

        if (!Found) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(PerfectHashContextInitializeCuda_ContextOrdinalNotFound,
                     Result);
            PH_RAISE(Result);
        }

        ASSERT(DeviceContext != NULL);

        //
        // Increment the count of solve contexts for this device context.
        //

        DeviceContext->NumberOfSolveContexts++;

        //
        // Link this solve context to the corresponding device context, then
        // fill in the kernel launch parameters.
        //

        SolveContext->DeviceContext = DeviceContext;

        SolveContext->BlocksPerGrid = BlocksPerGridValue;
        SolveContext->ThreadsPerBlock = ThreadsPerBlockValue;
        SolveContext->KernelRuntimeTargetInMilliseconds =
            KernelRuntimeTargetValue;

        //
        // Activate this context, create a stream, then deactivate it.
        //

        CuResult = Cu->CtxPushCurrent(DeviceContext->Context);
        CU_CHECK(CuResult, CtxPushCurrent);

        //
        // Create the stream for this solve context.
        //

        CuResult = Cu->StreamCreate(&SolveContext->Stream, StreamFlags);
        CU_CHECK(CuResult, StreamCreate);

        //
        // Pop the context off this thread.
        //

        CuResult = Cu->CtxPopCurrent(NULL);
        CU_CHECK(CuResult, CtxPopCurrent);

    }

    //
    // For each device context, allocate device memory to hold sufficient
    // graphs.  If we're in "find best graph" mode, there are two graphs per
    // solve context (to account for the spare graph); otherwise, there is one.
    //

    for (Index = 0; Index < NumberOfContexts; Index++) {

        DeviceContext = &DeviceContexts->DeviceContexts[Index];

        ASSERT(DeviceContext->NumberOfSolveContexts > 0);

        NumberOfGraphsForDevice = DeviceContext->NumberOfSolveContexts;

        if (FindBestGraph(Context)) {

            //
            // Account for the spare graphs.
            //

            NumberOfGraphsForDevice *= 2;
        }

        AllocSizeInBytes.QuadPart = NumberOfGraphsForDevice * sizeof(GRAPH);
        ASSERT(AllocSizeInBytes.HighPart == 0);

        //
        // Set the context, then allocate the array of graphs.
        //

        CuResult = Cu->CtxPushCurrent(DeviceContext->Context);
        CU_CHECK(CuResult, CtxPushCurrent);

        CuResult = Cu->MemAlloc((PCU_DEVICE_POINTER)&DeviceGraphs,
                                AllocSizeInBytes.LowPart);
        CU_CHECK(CuResult, MemAlloc);

        DeviceContext->DeviceGraphs = DeviceGraph = DeviceGraphs;

        //
        // Loop over the solve contexts pointing to this device context and
        // wire up matching ones to the device graph memory we just allocated.
        //

        MatchedGraphCount = 0;
        for (Inner = 0; Inner < NumberOfSolveContexts; Inner++) {
            SolveContext = &SolveContexts->SolveContexts[Inner];

            if (SolveContext->DeviceContext == DeviceContext) {
                SolveContext->DeviceGraph = DeviceGraph++;
                MatchedGraphCount++;
                if (FindBestGraph(Context)) {
                    SolveContext->DeviceSpareGraph = DeviceGraph++;
                    MatchedGraphCount++;
                }
            }
        }

        ASSERT(MatchedGraphCount == NumberOfGraphsForDevice);

        //
        // Allocate device memory to hold the CU_DEVICE_ATTRIBUTES structure.
        //

        CuResult = Cu->MemAlloc(&DeviceContext->DeviceAttributes,
                                sizeof(CU_DEVICE_ATTRIBUTES));
        CU_CHECK(CuResult, MemAlloc);

        //
        // Copy the host attributes to the device.
        //

        CuResult = Cu->MemcpyHtoDAsync(DeviceContext->DeviceAttributes,
                                       &DeviceContext->Device->Attributes,
                                       sizeof(CU_DEVICE_ATTRIBUTES),
                                       DeviceContext->Stream);
        CU_CHECK(CuResult, MemcpyHtoDAsync);

        //
        // Finally, pop the context.
        //

        CuResult = Cu->CtxPopCurrent(NULL);
        CU_CHECK(CuResult, CtxPopCurrent);

    }

    //
    // Time to allocate graph instances.
    //

    Graph = NULL;
    GpuGraphs = NULL;
    CpuGraphs = NULL;

    NumberOfGpuGraphs = Context->NumberOfGpuGraphs;
    NumberOfCpuGraphs = Context->NumberOfCpuGraphs;
    TotalNumberOfGraphs = NumberOfCpuGraphs + NumberOfGpuGraphs;
    Context->TotalNumberOfGraphs = TotalNumberOfGraphs;

    Graphs = Allocator->Vtbl->Calloc(Allocator,
                                     TotalNumberOfGraphs,
                                     sizeof(Graphs[0]));
    if (Graphs == NULL) {
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    GpuGraphs = Graphs;
    CpuGraphs = Graphs + NumberOfGpuGraphs;

    Context->GpuGraphs = GpuGraphs;
    Context->CpuGraphs = CpuGraphs;

    //
    // Create GPU graphs and assign one to each solve context.
    //

    SpareGraphCount = 0;
    DeviceContext = &DeviceContexts->DeviceContexts[0];
    SolveContext = &SolveContexts->SolveContexts[0];

    for (Index = 0; Index < NumberOfGpuGraphs; Index++) {

        Result = Context->Vtbl->CreateInstance(Context,
                                               NULL,
                                               &IID_PERFECT_HASH_GRAPH_CU,
                                               &Graph);

        if (FAILED(Result)) {

            //
            // Suppress logging for out-of-memory errors (as we communicate
            // memory issues back to the caller via informational return codes).
            //

            if (Result != E_OUTOFMEMORY) {
                PH_ERROR(InitializeCudaAndGraphsChm02_CreateGpuGraph, Result);
            }

            goto Error;
        }

        ASSERT(IsCuGraph(Graph));

        Graph->Index = Index;
        Graphs[Index] = Graph;

        Graph->CuDeviceIndex =
            InterlockedIncrement(&SolveContext->DeviceContext->NextDeviceIndex);

        Graph->CuSolveContext = SolveContext;

        ASSERT(SolveContext->DeviceGraph != NULL);

        Graph->CuRngId = Context->CuRngId;
        Graph->CuRngSeed = Context->CuRngSeed;
        Graph->CuRngSubsequence = Context->CuRngSubsequence;
        Graph->CuRngOffset = Context->CuRngOffset;

        if (!FindBestGraph(Context)) {
            SolveContext->HostGraph = Graph;
            SolveContext++;
        } else {

            ASSERT(SolveContext->DeviceSpareGraph != NULL);

            //
            // If the index is even (least significant bit is not set), this is
            // a normal graph.  If it's odd (LSB is 1), it's a spare graph.  We
            // advance the solve context pointer after every spare graph.  E.g.
            // if we have two solve contexts, we'll have four GPU graphs, which
            // will be mapped as follows:
            //
            //      Graph #0            -> SolveContext #0
            //      Graph #1 (spare)    -> SolveContext #0; SolveContext++
            //      Graph #2            -> SolveContext #1
            //      Graph #3 (spare)    -> SolveContext #1; SolveContext++
            //      etc.
            //

            if ((Index & 0x1) == 0) {

                //
                // This is a normal graph.
                //

                Graph->Flags.IsSpare = FALSE;
                SolveContext->HostGraph = Graph;
                ASSERT(SolveContext->HostSpareGraph == NULL);

            } else {

                //
                // This is a spare graph.
                //

                Graph->Flags.IsSpare = TRUE;
                SolveContext->HostSpareGraph = Graph;
                ASSERT(SolveContext->HostGraph != NULL);

                //
                // Advance the solve context.
                //

                SolveContext++;
            }
        }
    }

    if (FAILED(Result)) {
        goto Error;
    }

    //
    // If there are CPU solver threads, create CPU graph instances.
    //

    if (Context->NumberOfCpuThreads > 0) {

        SpareGraphCount = 0;

        for (Index = NumberOfGpuGraphs;
             Index < (NumberOfGpuGraphs + NumberOfCpuGraphs);
             Index++)
        {

            Result = Context->Vtbl->CreateInstance(Context,
                                                   NULL,
                                                   &IID_PERFECT_HASH_GRAPH,
                                                   &Graph);

            if (FAILED(Result)) {

                //
                // Suppress logging for out-of-memory errors (as we communicate
                // memory issues back to the caller via informational return
                // codes).
                //

                if (Result != E_OUTOFMEMORY) {
                    PH_ERROR(InitializeCudaAndGraphsChm02_CreateCpuGraph,
                             Result);
                }

                goto Error;
            }

            ASSERT(!IsCuGraph(Graph));

            Graph->Index = Index;
            Graphs[Index] = Graph;

            if (Index == NumberOfGpuGraphs) {

                //
                // This is the first CPU graph, verify we've captured the
                // correct CPU graph starting point.
                //

                ASSERT(&Graphs[Index] == CpuGraphs);
            }

            if (FindBestGraph(Context)) {

                if ((Index - NumberOfGpuGraphs) < Context->NumberOfCpuThreads) {

                    NOTHING;

                } else {

                    //
                    // There should only ever be one spare CPU graph.
                    //

                    SpareGraphCount++;
                    ASSERT(SpareGraphCount == 1);

                    Graph->Flags.IsSpare = TRUE;

                    //
                    // Context->SpareGraph is guarded by the best graph critical
                    // section.  We know that no worker threads will be running
                    // at this point; inform SAL accordingly by suppressing the
                    // concurrency warnings.
                    //

                    _Benign_race_begin_
                    Context->SpareGraph = Graph;
                    _Benign_race_end_
                }
            }

            //
            // Copy relevant flags over.
            //

            Graph->Flags.SkipVerification =
                (TableCreateFlags.SkipGraphVerification != FALSE);

            Graph->Flags.WantsWriteCombiningForVertexPairsArray =
                (TableCreateFlags.EnableWriteCombineForVertexPairs != FALSE);

            Graph->Flags.RemoveWriteCombineAfterSuccessfulHashKeys =
                (TableCreateFlags.RemoveWriteCombineAfterSuccessfulHashKeys
                 != FALSE);
        }
    }

    if (FAILED(Result)) {
        goto Error;
    }

    //
    // We're done, finish up.
    //

    Result = S_OK;
    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // TODO: loop through any device contexts here and free?
    //

    //
    // Intentional follow-on to End.
    //

End:

    //
    // Release any temporary component references.
    //

    RELEASE(PtxPath);
    RELEASE(PtxFile);
    RELEASE(RuntimeLibPath);
    RELEASE(RuntimeLibFile);

    Allocator = Context->Allocator;
    if (BitmapBuffer != NULL) {
        Allocator->Vtbl->FreePointer(Allocator, &BitmapBuffer);
    }

    return Result;
}


_Use_decl_annotations_
HRESULT
LoadPerfectHashTableImplChm02(
    PPERFECT_HASH_TABLE Table
    )
/*++

Routine Description:

    Loads a previously created perfect hash table.

Arguments:

    Table - Supplies a pointer to a partially-initialized PERFECT_HASH_TABLE
        structure.

Return Value:

    S_OK - Table was loaded successfully.

--*/
{
    return LoadPerfectHashTableImplChm01(Table);
}

_Use_decl_annotations_
HRESULT
PrepareGraphInfoChm02(
    PPERFECT_HASH_TABLE Table,
    PGRAPH_INFO Info,
    PGRAPH_INFO PrevInfo
    )
/*++

Routine Description:

    Prepares the GRAPH_INFO structure for a given table.

Arguments:

    Table - Supplies a pointer to the table.

    Info - Supplies a pointer to the graph info structure to prepare.

    PrevInfo - Optionally supplies a pointer to the previous info structure
        if this is not the first time the routine is being called.

Return Value:

    S_OK - Graph info prepared successfully.

    The following error codes may also be returned.  Note that this list is not
    guaranteed to be exhaustive; that is, error codes other than the ones listed
    below may also be returned.

    E_POINTER - Table or Info were NULL.

    E_UNEXPECTED - Catastrophic internal error.

    PH_E_TOO_MANY_KEYS - Too many keys.

    PH_E_TOO_MANY_EDGES - Too many edges.

    PH_E_TOO_MANY_TOTAL_EDGES - Too many total edges.

    PH_E_TOO_MANY_VERTICES - Too many vertices.

--*/
{
    HRESULT Result;
    ULONG NumberOfKeys;
    ULONG NumberOfVertices;

    //
    // Call out to the Chm01 preparation version first.
    //

    Result = PrepareGraphInfoChm01(Table, Info, PrevInfo);
    if (FAILED(Result)) {
        return Result;
    }

    //
    // CUDA-specific logic.
    //

    NumberOfKeys = Table->Keys->NumberOfKeys.LowPart;
    NumberOfVertices = Info->Dimensions.NumberOfVertices;

    Info->VertexPairsSizeInBytes = (
        RTL_ELEMENT_SIZE(GRAPH, VertexPairs) *
        (ULONGLONG)NumberOfKeys
    );

#if 0

    Info->DeletedSizeInBytes = (
        RTL_ELEMENT_SIZE(GRAPH, Deleted) *
        (ULONGLONG)NumberOfVertices
    );

    Info->VisitedSizeInBytes = (
        RTL_ELEMENT_SIZE(GRAPH, Visited) *
        (ULONGLONG)NumberOfVertices
    );

    Info->CountsSizeInBytes = (
        RTL_ELEMENT_SIZE(GRAPH, Counts) *
        (ULONGLONG)NumberOfVertices
    );

#endif

    return Result;
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
