/*++

Copyright (c) 2020 Trent Nelson <trent@trent.me>

Module Name:

    GraphCu.c

Abstract:

    This module implements a CUDA version of the original GRAPH module.

--*/

#include "stdafx.h"

//
// COM scaffolding routines for initialization and rundown.
//

GRAPH_INITIALIZE GraphCuInitialize;

_Use_decl_annotations_
HRESULT
GraphCuInitialize(
    PGRAPH Graph
    )
/*++

Routine Description:

    Initializes a graph structure.  This is a relatively simple method that
    just primes the COM scaffolding.

Arguments:

    Graph - Supplies a pointer to a GRAPH structure for which initialization
        is to be performed.

Return Value:

    S_OK - Success.

    E_POINTER - Graph is NULL.

    E_UNEXPECTED - All other errors.

--*/
{
    HRESULT Result = S_OK;

    if (!ARGUMENT_PRESENT(Graph)) {
        return E_POINTER;
    }

    Graph->SizeOfStruct = sizeof(*Graph);

    //
    // Create Rtl and Allocator components.
    //

    Result = Graph->Vtbl->CreateInstance(Graph,
                                         NULL,
                                         &IID_PERFECT_HASH_RTL,
                                         &Graph->Rtl);

    if (FAILED(Result)) {
        goto Error;
    }

    Result = Graph->Vtbl->CreateInstance(Graph,
                                         NULL,
                                         &IID_PERFECT_HASH_ALLOCATOR,
                                         &Graph->Allocator);

    if (FAILED(Result)) {
        goto Error;
    }

    //
    // Set the IsCuGraph flag indicating we're a CUDA graph.
    //

    Graph->Flags.IsCuGraph = TRUE;

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


GRAPH_RUNDOWN GraphCuRundown;

_Use_decl_annotations_
VOID
GraphCuRundown(
    PGRAPH Graph
    )
/*++

Routine Description:

    Release all resources associated with a graph.

Arguments:

    Graph - Supplies a pointer to a GRAPH structure for which rundown is to
        be performed.

Return Value:

    None.

--*/
{
    //
    // Sanity check structure size.
    //

    ASSERT(Graph->SizeOfStruct == sizeof(*Graph));

    //
    // Release applicable COM references.
    //

    RELEASE(Graph->Rtl);
    RELEASE(Graph->Allocator);

    return;
}

//
// Main interface entry points.
//

GRAPH_SET_INFO GraphCuSetInfo;

_Use_decl_annotations_
HRESULT
GraphCuSetInfo(
    PGRAPH Graph,
    PGRAPH_INFO Info
    )
/*++

Routine Description:

    Registers information about a graph with an individual graph instance.
    As table resizing isn't supported with GPU graphs, this routine will only
    ever be called once per graph.

Arguments:

    Graph - Supplies a pointer to the graph instance.

    Info - Supplies a pointer to the graph info instance.

Return Value:

    S_OK - Success.

    E_POINTER - Graph or Info were NULL.

--*/
{
    if (!ARGUMENT_PRESENT(Graph)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Info)) {
        return E_POINTER;
    }

    Graph->Info = Info;
    Graph->Flags.IsInfoSet = TRUE;

    return S_OK;
}

GRAPH_LOAD_INFO GraphCuLoadInfo;

_Use_decl_annotations_
HRESULT
GraphCuLoadInfo(
    PGRAPH Graph
    )
/*++

Routine Description:

    This routine is called by graph solving worker threads prior to attempting
    any solving; it is responsible for initializing the graph structure and
    allocating the necessary buffers required for graph solving, using the sizes
    indicated by the info structure previously set by the main thread via
    SetInfo().

Arguments:

    Graph - Supplies a pointer to the graph instance.

Return Value:

    S_OK - Success.

    E_POINTER - Graph was NULL.

    E_OUTOFMEMORY - Out of memory.

    PH_E_GRAPH_NO_INFO_SET - No graph information has been set for this graph.

    PH_E_GRAPH_INFO_ALREADY_LOADED - Graph information has already been loaded
        for this graph.

--*/
{
    PCU Cu;
    PRTL Rtl;
    HRESULT Result;
    PGRAPH_INFO Info;
    PGRAPH_INFO PrevInfo;
    PALLOCATOR Allocator;
    ULONG ProtectionFlags;
    PPERFECT_HASH_TABLE Table;
    SIZE_T VertexPairsSizeInBytes;
    PPERFECT_HASH_CONTEXT Context;
    BOOLEAN LargePagesForVertexPairs;
    PASSIGNED_MEMORY_COVERAGE Coverage;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;
    PRTL_TRY_LARGE_PAGE_VIRTUAL_ALLOC Alloc;
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Graph)) {
        return E_POINTER;
    }

    if (!IsGraphInfoSet(Graph)) {
        return PH_E_GRAPH_NO_INFO_SET;
    } else if (IsGraphInfoLoaded(Graph)) {
        return PH_E_GRAPH_INFO_ALREADY_LOADED;
    } else {
        Info = Graph->Info;
    }

    //
    // Sanity check the graph size is correct.
    //

    ASSERT(sizeof(*Graph) == Info->SizeOfGraphStruct);

    //
    // Initialize aliases.
    //

    Context = Info->Context;
    Rtl = Context->Rtl;
    PrevInfo = Info->PrevInfo;
    Allocator = Graph->Allocator;
    Table = Context->Table;
    TableInfoOnDisk = Table->TableInfoOnDisk;
    TableCreateFlags.AsULong = Table->TableCreateFlags.AsULong;

    //
    // Set the relevant graph fields based on the provided info.
    //

    Graph->Context = Context;
    Graph->NumberOfSeeds = Table->TableInfoOnDisk->NumberOfSeeds;
    Graph->NumberOfKeys = Table->Keys->NumberOfElements.LowPart;

    Graph->ThreadId = GetCurrentThreadId();
    Graph->ThreadAttempt = 0;

    Graph->EdgeMask = Table->IndexMask;
    Graph->VertexMask = Table->HashMask;
    Graph->EdgeModulus = Table->IndexModulus;
    Graph->VertexModulus = Table->HashModulus;
    Graph->MaskFunctionId = Info->Context->MaskFunctionId;

    Graph->Flags.Paranoid = IsParanoid(Table);

    CopyInline(&Graph->Dimensions,
               &Info->Dimensions,
               sizeof(Graph->Dimensions));

    Result = S_OK;

    //
    // Allocate (or reallocate) arrays.
    //

#define ALLOC_ARRAY(Name, Type)                       \
    if (!Graph->##Name) {                             \
        Graph->##Name = (Type)(                       \
            Allocator->Vtbl->AlignedMalloc(           \
                Allocator,                            \
                (ULONG_PTR)Info->##Name##SizeInBytes, \
                YMMWORD_ALIGNMENT                     \
            )                                         \
        );                                            \
    } else {                                          \
        Graph->##Name## = (Type)(                     \
            Allocator->Vtbl->AlignedReAlloc(          \
                Allocator,                            \
                Graph->##Name,                        \
                (ULONG_PTR)Info->##Name##SizeInBytes, \
                YMMWORD_ALIGNMENT                     \
            )                                         \
        );                                            \
    }                                                 \
    if (!Graph->##Name) {                             \
        Result = E_OUTOFMEMORY;                       \
        goto Error;                                   \
    }

    ALLOC_ARRAY(Edges, PEDGE);
    ALLOC_ARRAY(Next, PEDGE);
    ALLOC_ARRAY(First, PVERTEX);
    ALLOC_ARRAY(Assigned, PASSIGNED);

    //
    // If we're hashing all keys first, prepare the vertex pairs array if it
    // hasn't already been prepared.  (This array is sized off the number of
    // keys, which never changes upon subsequent table resize events, so it
    // never needs to be reallocated to a larger size (unlike the other arrays
    // above, which grow larger upon each resize event).)
    //

    if (TableCreateFlags.HashAllKeysFirst) {

        ASSERT(Info->VertexPairsSizeInBytes != 0);
        VertexPairsSizeInBytes = (SIZE_T)Info->VertexPairsSizeInBytes;

        if (Graph->VertexPairs == NULL) {

            LargePagesForVertexPairs = (BOOLEAN)(
                TableCreateFlags.TryLargePagesForVertexPairs != FALSE
            );

            ProtectionFlags = PAGE_READWRITE;

            if (Graph->Flags.WantsWriteCombiningForVertexPairsArray) {

                //
                // Large pages and write-combine are incompatible.  (This will
                // have been weeded out by IsValidTableCreateFlags(), so we can
                // just ASSERT() instead here.)
                //

                ASSERT(!LargePagesForVertexPairs);

                ProtectionFlags |= PAGE_WRITECOMBINE;
            }

            //
            // Proceed with allocation of the vertex pairs array.
            //

            Alloc = Rtl->Vtbl->TryLargePageVirtualAlloc;
            Graph->VertexPairs = Alloc(Rtl,
                                       NULL,
                                       VertexPairsSizeInBytes,
                                       MEM_RESERVE | MEM_COMMIT,
                                       ProtectionFlags,
                                       &LargePagesForVertexPairs);

            if (Graph->VertexPairs == NULL) {
                Result = E_OUTOFMEMORY;
                goto Error;
            }

            //
            // Update the graph flags indicating whether or not large pages
            // were used, and if write-combining is active.
            //

            Graph->Flags.VertexPairsArrayUsesLargePages =
                LargePagesForVertexPairs;

            Graph->Flags.VertexPairsArrayIsWriteCombined =
                Graph->Flags.WantsWriteCombiningForVertexPairsArray;

        }
    }

    //
    // Set the bitmap sizes and then allocate (or reallocate) the bitmap
    // buffers.
    //

    Graph->DeletedEdgesBitmap.SizeOfBitMap = Graph->TotalNumberOfEdges;
    Graph->VisitedVerticesBitmap.SizeOfBitMap = Graph->NumberOfVertices;
    Graph->AssignedBitmap.SizeOfBitMap = Graph->NumberOfVertices;
    Graph->IndexBitmap.SizeOfBitMap = Graph->NumberOfVertices;

#define ALLOC_BITMAP_BUFFER(Name)                          \
    if (!Graph->##Name##.Buffer) {                         \
        Graph->##Name##.Buffer = (PULONG)(                 \
            Allocator->Vtbl->Malloc(                       \
                Allocator,                                 \
                (ULONG_PTR)Info->##Name##BufferSizeInBytes \
            )                                              \
        );                                                 \
    } else {                                               \
        Graph->##Name##.Buffer = (PULONG)(                 \
            Allocator->Vtbl->ReAlloc(                      \
                Allocator,                                 \
                Graph->##Name##.Buffer,                    \
                (ULONG_PTR)Info->##Name##BufferSizeInBytes \
            )                                              \
        );                                                 \
    }                                                      \
    if (!Graph->##Name##.Buffer) {                         \
        Result = E_OUTOFMEMORY;                            \
        goto Error;                                        \
    }

    ALLOC_BITMAP_BUFFER(DeletedEdgesBitmap);
    ALLOC_BITMAP_BUFFER(VisitedVerticesBitmap);
    ALLOC_BITMAP_BUFFER(AssignedBitmap);
    ALLOC_BITMAP_BUFFER(IndexBitmap);

    //
    // Check to see if we're in "first graph wins" mode, and have also been
    // asked to skip memory coverage information.  If so, we can jump straight
    // to the end and finish up.
    //

    if (FirstSolvedGraphWinsAndSkipMemoryCoverage(Context)) {
        Graph->Flags.WantsAssignedMemoryCoverage = FALSE;
        goto End;
    }

    if (FirstSolvedGraphWins(Context)) {

        Graph->Flags.WantsAssignedMemoryCoverage = TRUE;

    } else {

        if (DoesBestCoverageTypeRequireKeysSubset(Context->BestCoverageType)) {
            Graph->Flags.WantsAssignedMemoryCoverageForKeysSubset = TRUE;
        } else {
            Graph->Flags.WantsAssignedMemoryCoverage = TRUE;
        }

    }

    //
    // Fill out the assigned memory coverage structure and allocate buffers.
    //

    Coverage = &Graph->AssignedMemoryCoverage;

    Coverage->TotalNumberOfPages = Info->AssignedArrayNumberOfPages;
    Coverage->TotalNumberOfLargePages = Info->AssignedArrayNumberOfLargePages;
    Coverage->TotalNumberOfCacheLines = Info->AssignedArrayNumberOfCacheLines;

#define ALLOC_ASSIGNED_ARRAY(Name, Type)               \
    if (!Coverage->##Name) {                           \
        Coverage->##Name = (PASSIGNED_##Type##_COUNT)( \
            Allocator->Vtbl->AlignedMalloc(            \
                Allocator,                             \
                (ULONG_PTR)Info->##Name##SizeInBytes,  \
                YMMWORD_ALIGNMENT                      \
            )                                          \
        );                                             \
    } else {                                           \
        Coverage->##Name = (PASSIGNED_##Type##_COUNT)( \
            Allocator->Vtbl->AlignedReAlloc(           \
                Allocator,                             \
                Coverage->##Name,                      \
                (ULONG_PTR)Info->##Name##SizeInBytes,  \
                YMMWORD_ALIGNMENT                      \
            )                                          \
        );                                             \
    }                                                  \
    if (!Coverage->##Name) {                           \
        Result = E_OUTOFMEMORY;                        \
        goto Error;                                    \
    }

    ALLOC_ASSIGNED_ARRAY(NumberOfAssignedPerPage, PAGE);
    ALLOC_ASSIGNED_ARRAY(NumberOfAssignedPerCacheLine, CACHE_LINE);

    //
    // The number of large pages consumed may not change between resize events;
    // avoid a realloc if unnecessary by checking the previous info's number of
    // large pages if applicable.
    //

#define ALLOC_ASSIGNED_LARGE_PAGE_ARRAY(Name)                                 \
    if (!Coverage->##Name) {                                                  \
        Coverage->##Name = (PASSIGNED_LARGE_PAGE_COUNT)(                      \
            Allocator->Vtbl->AlignedMalloc(                                   \
                Allocator,                                                    \
                (ULONG_PTR)Info->##Name##SizeInBytes,                         \
                YMMWORD_ALIGNMENT                                             \
            )                                                                 \
        );                                                                    \
    } else {                                                                  \
        BOOLEAN DoReAlloc = TRUE;                                             \
        if (PrevInfo) {                                                       \
            if (PrevInfo->##Name##SizeInBytes == Info->##Name##SizeInBytes) { \
                DoReAlloc = FALSE;                                            \
            }                                                                 \
        }                                                                     \
        if (DoReAlloc) {                                                      \
            Coverage->##Name = (PASSIGNED_LARGE_PAGE_COUNT)(                  \
                Allocator->Vtbl->AlignedReAlloc(                              \
                    Allocator,                                                \
                    Coverage->##Name,                                         \
                    (ULONG_PTR)Info->##Name##SizeInBytes,                     \
                    YMMWORD_ALIGNMENT                                         \
                )                                                             \
            );                                                                \
        }                                                                     \
    }                                                                         \
    if (!Coverage->##Name) {                                                  \
        Result = E_OUTOFMEMORY;                                               \
        goto Error;                                                           \
    }

    ALLOC_ASSIGNED_LARGE_PAGE_ARRAY(NumberOfAssignedPerLargePage);

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    if (SUCCEEDED(Result)) {
        Graph->Flags.IsInfoLoaded = TRUE;
        Graph->LastLoadedNumberOfVertices = Graph->NumberOfVertices;
    }

    return Result;

}


GRAPH_ENTER_SOLVING_LOOP GraphCuEnterSolvingLoop;

_Use_decl_annotations_
HRESULT
GraphCuEnterSolvingLoop(
    PGRAPH Graph
    )
/*++

Routine Description:

    Enters the graph solving loop.

Arguments:

    Graph - Supplies a pointer to a graph instance.

Return Value:

    S_OK - Success.

    E_POINTER - Graph was NULL.

    E_OUTOFMEMORY - Out of memory.

    Non-exhaustive list of additional errors that may be returned:

    PH_E_GRAPH_NO_INFO_SET - No graph information was set.

    PH_E_NO_MORE_SEEDS - No more seed data is available.

--*/
{
    //PGRAPH NewGraph;
    HRESULT Result = S_OK;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Graph)) {
        return E_POINTER;
    }

    //
    // Acquire the exclusive graph lock for the duration of the routine.  The
    // graph should never be locked at this point; if it is, consider it a
    // fatal error.
    //

    if (!TryAcquireGraphLockExclusive(Graph)) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(GraphEnterSolvingLoop_GraphLocked, Result);
        PH_RAISE(Result);
    }

    /*
    //
    // Load the graph info.
    //

    Result = Graph->Vtbl->LoadInfo(Graph);

    if (FAILED(Result)) {

        if (Result != E_OUTOFMEMORY) {

            //
            // Anything other than an out-of-memory indication from LoadInfo()
            // indicates an internal error somewhere; log the error, then raise.
            //

            PH_ERROR(GraphLoadInfo, Result);
            PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);

        }

        //
        // We failed to allocate sufficient memory for the graph.  Check for the
        // edge case where *all* threads failed to allocate memory, and set the
        // context state flag and FailedEvent accordingly.
        //

        if (InterlockedDecrement(&Graph->Context->GraphMemoryFailures) == 0) {
            Graph->Context->State.AllGraphsFailedMemoryAllocation = TRUE;
            SetStopSolving(Graph->Context);
            if (!SetEvent(Graph->Context->FailedEvent)) {
                SYS_ERROR(SetEvent);
                Result = PH_E_SYSTEM_CALL_FAILED;
            }
        }

        goto End;
    }

    //
    // Begin the solving loop.
    //

    while (Graph->Vtbl->ShouldWeContinueTryingToSolve(Graph)) {

        Result = Graph->Vtbl->LoadNewSeeds(Graph);
        if (FAILED(Result)) {

            //
            // N.B. This will need to be adjusted when we support the notion
            //      of no more seed data (PH_E_NO_MORE_SEEDS).
            //

            PH_ERROR(GraphLoadNewSeeds, Result);
            break;
        }

        Result = Graph->Vtbl->Reset(Graph);
        if (FAILED(Result)) {
            PH_ERROR(GraphReset, Result);
            break;
        } else if (Result != PH_S_CONTINUE_GRAPH_SOLVING) {
            break;
        }

        NewGraph = NULL;
        Result = Graph->Vtbl->Solve(Graph, &NewGraph);
        if (FAILED(Result)) {
            PH_ERROR(GraphSolve, Result);
            break;
        }

        if (Result == PH_S_STOP_GRAPH_SOLVING ||
            Result == PH_S_GRAPH_SOLVING_STOPPED) {
            if (NewGraph != NULL) {
                PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
            }
            break;
        }

        if (Result == PH_S_USE_NEW_GRAPH_FOR_SOLVING) {

            if (NewGraph == NULL) {
                PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
            }

            //
            // Acquire the new graph's lock and release the existing
            // graph's lock.
            //

            AcquireGraphLockExclusive(NewGraph);
            ReleaseGraphLockExclusive(Graph);

            if (!IsGraphInfoLoaded(NewGraph) ||
                NewGraph->LastLoadedNumberOfVertices <
                Graph->NumberOfVertices) {

                Result = NewGraph->Vtbl->LoadInfo(NewGraph);
                if (FAILED(Result)) {
                    PH_ERROR(GraphLoadInfo_NewGraph, Result);
                    goto End;
                }
            }

            Graph = NewGraph;
            continue;
        }

        //
        // Invariant check: result should always be PH_S_CONTINUE_GRAPH_SOLVING
        // at this point.
        //

        ASSERT(Result == PH_S_CONTINUE_GRAPH_SOLVING);

        //
        // Continue the loop and attempt another solve.
        //

    }

    //
    // Intentional follow-on to End.
    //

End:

    */

    if (SUCCEEDED(Result)) {

        //
        // Normalize the success error codes (e.g. PH_S_STOP_GRAPH_SOLVING)
        // into a single S_OK return value.
        //

        Result = S_OK;
    }

    ReleaseGraphLockExclusive(Graph);

    return Result;
}

GRAPH_VERIFY GraphCuVerify;

_Use_decl_annotations_
HRESULT
GraphCuVerify(
    PGRAPH Graph
    )
{
    return GraphVerify(Graph);
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
