/*++

Copyright (c) 2020-2023 Trent Nelson <trent@trent.me>

Module Name:

    Chm02Shared.c

Abstract:

    Logic shared between the PH_WINDOWS and PH_COMPAT Chm02 implementations.

--*/

#include "stdafx.h"
#include "Chm01.h"
#include "Chm02Private.h"
#include "Graph_Ptx_RawCString.h"

//
// Main initialization routine.
//

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
    ULONG Index;
    ULONG Inner;
    LONG Ordinal;
    BOOLEAN Found;
    CU_RESULT CuResult;
    ULONG NumberOfDevices;
    ULONG NumberOfContexts;
    PULONG BitmapBuffer = NULL;
    HRESULT Result;
    PCHAR PtxString;
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
    PCU_LINK_STATE LinkState;
    PUNICODE_STRING CuPtxPath;
    PUNICODE_STRING CuCudaDevRuntimeLibPath;
    PVALUE_ARRAY Ordinals;
    ULONG NumberOfGpuGraphs;
    ULONG NumberOfCpuGraphs;
    ULONG TotalNumberOfGraphs;
    ULONG SpareGraphCount;
    ULONG MatchedGraphCount;
    ULONG NumberOfGraphsForDevice;
    ULONG NumberOfSolveContexts;
    CU_STREAM_FLAGS StreamFlags;
    ULARGE_INTEGER AllocSizeInBytes;
    PPH_CU_DEVICE_CONTEXT DeviceContext;
    PPH_CU_DEVICE_CONTEXTS DeviceContexts;
    PPH_CU_SOLVE_CONTEXT SolveContext;
    PPH_CU_SOLVE_CONTEXTS SolveContexts;
    PPH_CU_RUNTIME_CONTEXT CuRuntimeContext;
    //PERFECT_HASH_CU_RNG_ID CuRngId = PerfectHashCuNullRngId;
    PPERFECT_HASH_PATH PtxPath;
    PPERFECT_HASH_FILE PtxFile;
    PPERFECT_HASH_PATH RuntimeLibPath;
    PPERFECT_HASH_FILE RuntimeLibFile;
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

    TableCreateFlags.AsULongLong =
        Context->Table->TableCreateFlags.AsULongLong;

    //
    // If we've already got a CU instance, assume we're already initialized.
    //

    if (Context->Cu != NULL) {
        return S_FALSE;
    }

    PtxPath = NULL;
    PtxFile = NULL;
    RuntimeLibFile = NULL;
    RuntimeLibPath = NULL;
    SolveContexts = NULL;
    DeviceContexts = NULL;
    CuCudaDevRuntimeLibPath = NULL;

    Result = CreateCuInstance(Context, &Context->Cu);
    if (FAILED(Result)) {
        PH_ERROR(InitializeCudaAndGraphsChm02_CreateCuInstance, Result);
        goto Error;
    }

    //
    // Create and initialize the CU runtime context.
    //

    Cu = Context->Cu;
    Result = CreateCuRuntimeContext(Cu, &Context->CuRuntimeContext);
    if (FAILED(Result)) {
        PH_ERROR(InitializeCudaAndGraphsChm02_CreateCuRuntimeContext, Result);
        goto Error;
    }
    CuRuntimeContext = Context->CuRuntimeContext;

    Result = InitializeCuRuntimeContext(Context,
                                        TableCreateParameters,
                                        Context->CuRuntimeContext);
    if (FAILED(Result)) {
        PH_ERROR(InitializeCudaAndGraphsChm02_InitializeCuRuntimeContext,
                 Result);
        goto Error;
    }

    //
    // TODO: continue hoisting out logic below into separate routines.
    //

    CuPtxPath = CuRuntimeContext->CuPtxPath;

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

    CuCudaDevRuntimeLibPath = CuRuntimeContext->CuCudaDevRuntimeLibPath;
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
    NumberOfDevices = CuRuntimeContext->NumberOfDevices;
    NumberOfContexts = CuRuntimeContext->NumberOfContexts;
    DeviceContexts = CuRuntimeContext->CuDeviceContexts;

    for (Index = 0; Index < NumberOfContexts; Index++) {
        DeviceContext = &DeviceContexts->DeviceContexts[Index];

        DeviceContext->Rtl = Context->Rtl;

        //
        // Find the PH_CU_DEVICE instance with the same ordinal.
        //

        Found = FALSE;
        Device = NULL;
        for (Inner = 0; Inner < NumberOfDevices; Inner++) {
            Device = &CuRuntimeContext->CuDevices.Devices[Inner];
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
        // We can now destroy the linker state.
        //

        CuResult = Cu->LinkDestroy(LinkState);
        CU_CHECK(CuResult, LinkDestroy);

        //
        // Module loaded successfully, resolve the kernels.
        //

        Result = CuDeviceContextInitializeKernels(DeviceContext);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashContextInitializeCuda_InitializeKernels,
                     Result);
            goto Error;
        }

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

    Allocator = Cu->Allocator;
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

    Ordinals = CuRuntimeContext->Ordinals;

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
        // Link this solve context to the corresponding device context.
        //

        SolveContext->DeviceContext = DeviceContext;

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

        Graph->CuRngId = CuRuntimeContext->CuRngId;
        Graph->CuRngSeed = CuRuntimeContext->CuRngSeed;
        Graph->CuRngSubsequence = CuRuntimeContext->CuRngSubsequence;
        Graph->CuRngOffset = CuRuntimeContext->CuRngOffset;

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

    DestroyCuRuntimeContext(&Context->CuRuntimeContext);

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

HRESULT
CopyKeysToDevices(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PPERFECT_HASH_KEYS Keys
    )
/*++

Routine Description:

    Copies the keys to each device.

Arguments:

    Context - Supplies a pointer to a PERFECT_HASH_CONTEXT structure.

    Keys - Supplies a pointer to a PERFECT_HASH_KEYS structure.

Return Value:

    HRESULT - S_OK on success, appropriate HRESULT otherwise.

--*/
{
    PCU Cu;
    USHORT Index;
    HRESULT Result;
    CU_RESULT CuResult;
    PVOID KeysBaseAddress;
    SIZE_T KeysSizeInBytes;
    ULONG NumberOfDeviceContexts;
    PPH_CU_DEVICE_CONTEXT DeviceContext;
    PPH_CU_DEVICE_CONTEXTS DeviceContexts;
    PPH_CU_RUNTIME_CONTEXT CuRuntimeContext;

    //
    // Initialize aliases.
    //

    CuRuntimeContext = Context->CuRuntimeContext;
    Cu = CuRuntimeContext->Cu;
    DeviceContexts = CuRuntimeContext->CuDeviceContexts;
    NumberOfDeviceContexts = DeviceContexts->NumberOfDeviceContexts;

    KeysBaseAddress = Keys->KeyArrayBaseAddress;
    KeysSizeInBytes = Keys->NumberOfKeys.QuadPart * Keys->KeySizeInBytes;

    for (Index = 0; Index < NumberOfDeviceContexts; Index++) {

        DeviceContext = &DeviceContexts->DeviceContexts[Index];

        //
        // Active the context.
        //

        CuResult = Cu->CtxPushCurrent(DeviceContext->Context);
        if (CU_FAILED(CuResult)) {
            CU_ERROR(CtxPushCurrent, CuResult);
            Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
            goto Error;
        }

        if (DeviceContext->KeysBaseAddress == 0) {

            //
            // No device memory has been allocated for keys before, so,
            // allocate some now.
            //

            ASSERT(DeviceContext->KeysSizeInBytes == 0);

            CuResult = Cu->MemAlloc(&DeviceContext->KeysBaseAddress,
                                    KeysSizeInBytes);
            CU_CHECK(CuResult, MemAlloc);

            DeviceContext->KeysSizeInBytes = KeysSizeInBytes;

        } else {

            //
            // Device memory has already been allocated.  If it's less than what
            // we need, free what's there and allocate new memory.
            //

            ASSERT(DeviceContext->KeysSizeInBytes > 0);

            if (DeviceContext->KeysSizeInBytes < KeysSizeInBytes) {

                CuResult = Cu->MemFree(DeviceContext->KeysBaseAddress);
                CU_CHECK(CuResult, MemFree);

                DeviceContext->KeysBaseAddress = 0;

                CuResult = Cu->MemAlloc(&DeviceContext->KeysBaseAddress,
                                        KeysSizeInBytes);
                CU_CHECK(CuResult, MemAlloc);

                DeviceContext->KeysSizeInBytes = KeysSizeInBytes;

            } else {

                //
                // The existing device memory will fit the keys array, so
                // there's nothing more to do here.
                //

                ASSERT(DeviceContext->KeysSizeInBytes >= KeysSizeInBytes);
            }
        }

        //
        // Copy the keys over.
        //

        CuResult = Cu->MemcpyHtoDAsync(DeviceContext->KeysBaseAddress,
                                       KeysBaseAddress,
                                       KeysSizeInBytes,
                                       DeviceContext->Stream);
        CU_CHECK(CuResult, MemcpyHtoDAsync);

        //
        // Pop the context off this thread.
        //

        CuResult = Cu->CtxPopCurrent(NULL);
        CU_CHECK(CuResult, CtxPopCurrent);
    }

    //
    // If we get here, we're done.  Indicate success and finish up.
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


HRESULT
CopyGraphInfoToDevices(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PGRAPH_INFO Info
    )
/*++

Routine Description:

    Copies the graph info to each device.

Arguments:

    Context - Supplies a pointer to a PERFECT_HASH_CONTEXT structure.

    Info - Supplies a pointer to a GRAPH_INFO structure.

Return Value:

    HRESULT - S_OK on success, appropriate HRESULT otherwise.

--*/
{
    PCU Cu;
    USHORT Index;
    HRESULT Result;
    CU_RESULT CuResult;
    ULONG NumberOfDeviceContexts;
    PPH_CU_DEVICE_CONTEXT DeviceContext;
    PPH_CU_DEVICE_CONTEXTS DeviceContexts;
    PPH_CU_RUNTIME_CONTEXT CuRuntimeContext;

    //
    // Initialize aliases.
    //

    CuRuntimeContext = Context->CuRuntimeContext;
    Cu = CuRuntimeContext->Cu;
    DeviceContexts = CuRuntimeContext->CuDeviceContexts;
    NumberOfDeviceContexts = DeviceContexts->NumberOfDeviceContexts;

    for (Index = 0; Index < NumberOfDeviceContexts; Index++) {

        DeviceContext = &DeviceContexts->DeviceContexts[Index];

        //
        // Activate the context.
        //

        CuResult = Cu->CtxPushCurrent(DeviceContext->Context);
        if (CU_FAILED(CuResult)) {
            CU_ERROR(CtxPushCurrent, CuResult);
            Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
            goto Error;
        }

        if (DeviceContext->DeviceGraphInfoAddress == 0) {

            //
            // Allocate memory for the graph info.
            //

            CuResult = Cu->MemAlloc(&DeviceContext->DeviceGraphInfoAddress,
                                    sizeof(*Info));
            CU_CHECK(CuResult, MemAlloc);
        }

        //
        // Copy the graph info over.
        //

        CuResult = Cu->MemcpyHtoDAsync(DeviceContext->DeviceGraphInfoAddress,
                                       Info,
                                       sizeof(*Info),
                                       DeviceContext->Stream);

        CU_CHECK(CuResult, MemcpyHtoDAsync);

        //
        // Pop the context off this thread.
        //

        CuResult = Cu->CtxPopCurrent(NULL);
        CU_CHECK(CuResult, CtxPopCurrent);
    }

    //
    // If we get here, we're done.  Indicate success and finish up.
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
