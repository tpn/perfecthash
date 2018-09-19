/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Chm_01.c

Abstract:

    This module implements the CHM perfect hash table algorithm.

--*/

#include "stdafx.h"
#include "Chm_01.h"

//
// Define the threshold for how many attempts need to be made at finding a
// perfect hash solution before we double our number of vertices and try again.
//
// N.B. 100 is quite generous; normally, solutions are found on average within
//      3 attempts, and there's a 99.9% chance a solution will be found by the
//      18th attempt.
//

#define GRAPH_SOLVING_ATTEMPTS_THRESHOLD 100

//
// Define a limit for how many times the table resizing will be attempted before
// giving up.  For large table sizes and large concurrency values, note that we
// may hit memory limits before we hit this resize limit.
//

#define GRAPH_SOLVING_RESIZE_TABLE_LIMIT 10


_Use_decl_annotations_
HRESULT
CreatePerfectHashTableImplChm01(
    PPERFECT_HASH_TABLE Table
    )
/*++

Routine Description:

    Attempts to create a perfect hash table using the CHM algorithm and a
    2-part random hypergraph.

Arguments:

    Table - Supplies a pointer to a partially-initialized PERFECT_HASH_TABLE
        structure.

Return Value:

    S_OK - Table created successfully.


    The following error codes may also be returned.  Note that this list is not
    guaranteed to be exhaustive; that is, error codes other than the ones listed
    below may also be returned.

    E_POINTER - Table was NULL.

    E_UNEXPECTED - Catastrophic internal error.

    E_OUTOFMEMORY - Out of memory.

    PH_E_SYSTEM_CALL_FAILED - A system call failed.

    PH_E_REQUESTED_NUMBER_OF_TABLE_ELEMENTS_TOO_LARGE - The requested number
        of table elements exceeded limits.  If a table resize event occurrs,
        the number of requested table elements is doubled.  If this number
        exceeds MAX_ULONG, this error will be returned.

    PH_E_ERROR_PREPARING_FILE - An error occurred whilst preparing a file to
        use for saving the perfect hash table.

    PH_E_ERROR_SAVING_FILE - An error occurred whilst trying to save the perfect
        hash table to the file prepared earlier.

    PH_E_TABLE_VERIFICATION_FAILED - The winning perfect hash table solution
        failed internal verification.  The primary cause of this is typically
        when collisions are detected during verification.

--*/
{
    PRTL Rtl;
    USHORT Index;
    PULONG Keys;
    PGRAPH Graph;
    PBYTE Buffer;
    BOOLEAN Success;
    BOOLEAN PreparedHeader = FALSE;
    USHORT PageSize;
    USHORT PageShift;
    ULONG_PTR LastPage;
    ULONG_PTR ThisPage;
    BYTE NumberOfEvents;
    HRESULT Result = S_OK;
    PVOID BaseAddress = NULL;
    ULONG WaitResult;
    GRAPH_INFO Info;
    PBYTE Unusable;
    ULONG NumberOfKeys;
    BOOLEAN CaughtException;
    PALLOCATOR Allocator;
    HANDLE ProcessHandle = NULL;
    PHANDLE Event;
    USHORT NumberOfGraphs;
    USHORT NumberOfGuardPages;
    ULONG NumberOfPagesPerGraph;
    ULONG TotalNumberOfPages;
    USHORT NumberOfBitmaps;
    PGRAPH_DIMENSIONS Dim;
    PSLIST_ENTRY ListEntry;
    SYSTEM_INFO SystemInfo;
    FILE_WORK_ITEM SaveTableFile;
    FILE_WORK_ITEM PrepareTableFile;
    FILE_WORK_ITEM SaveHeaderFile;
    FILE_WORK_ITEM PrepareHeaderFile;
    PGRAPH_INFO_ON_DISK GraphInfoOnDisk;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;
    ULONGLONG Closest;
    ULONGLONG LastClosest;
    ULONGLONG NextSizeInBytes;
    ULONGLONG PrevSizeInBytes;
    ULONGLONG FirstSizeInBytes;
    ULONGLONG EdgesSizeInBytes;
    ULONGLONG ValuesSizeInBytes;
    ULONGLONG AssignedSizeInBytes;
    ULONGLONG TotalBufferSizeInBytes;
    ULONGLONG UsableBufferSizeInBytesPerBuffer;
    ULONGLONG ExpectedTotalBufferSizeInBytes;
    ULONGLONG ExpectedUsableBufferSizeInBytesPerBuffer;
    ULONGLONG GraphSizeInBytesIncludingGuardPage;
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId;
    ULARGE_INTEGER AllocSize;
    ULARGE_INTEGER NumberOfEdges;
    ULARGE_INTEGER NumberOfVertices;
    ULARGE_INTEGER TotalNumberOfEdges;
    ULARGE_INTEGER DeletedEdgesBitmapBufferSizeInBytes;
    ULARGE_INTEGER VisitedVerticesBitmapBufferSizeInBytes;
    ULARGE_INTEGER AssignedBitmapBufferSizeInBytes;
    ULARGE_INTEGER IndexBitmapBufferSizeInBytes;
    PPERFECT_HASH_CONTEXT Context = Table->Context;
    BOOL WaitForAllEvents;
    HANDLE Events[5];
    HANDLE SaveFileEvents[2];

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    //
    // The following label is jumped to by code later in this routine when we
    // detect that we've exceeded a plausible number of attempts at finding a
    // graph solution with the given number of vertices, and have bumped up
    // the vertex count (by adjusting Table->RequestedNumberOfElements) and
    // want to try again.
    //

RetryWithLargerTableSize:

    //
    // Initialize aliases.
    //

    Rtl = Table->Rtl;
    Keys = (PULONG)Table->Keys->BaseAddress;
    Allocator = Table->Allocator;
    Context = Table->Context;
    MaskFunctionId = Context->MaskFunctionId;
    ProcessHandle = GetCurrentProcess();

    //
    // If no threshold has been set, use the default.
    //

    if (!Context->ResizeTableThreshold) {
        Context->ResizeTableThreshold = GRAPH_SOLVING_ATTEMPTS_THRESHOLD;
        Context->ResizeLimit = GRAPH_SOLVING_RESIZE_TABLE_LIMIT;
    }

    //
    // Explicitly reset all events*.  This ensures everything is back in the
    // starting state if we happen to be attempting to solve the graph after
    // a resize event.
    //
    // [*]: Except the PreparedFileHeaderEvent, which is only set once
    //      regardless of how many table resize events occur.
    //

    Event = (PHANDLE)&Context->FirstEvent;
    NumberOfEvents = GetNumberOfContextEvents(Context);

    for (Index = 0; Index < NumberOfEvents; Index++, Event++) {

        if (*Event == Context->PreparedHeaderFileEvent) {
            continue;
        }

        if (!ResetEvent(*Event)) {
            SYS_ERROR(ResetEvent);
            Result = PH_E_SYSTEM_CALL_FAILED;
            goto Error;
        }
    }

    //
    // Prepare the initial "header file preparation" work callback.  This will
    // write the common code shared by all headers (i.e. everything that can
    // be written prior to the perfect hash table solution being solved).
    //

    if (!PreparedHeader) {

        ZeroStruct(PrepareHeaderFile);
        PrepareHeaderFile.FileWorkId = FileWorkPrepareHeaderId;
        InterlockedPushEntrySList(&Context->FileWorkListHead,
                                  &PrepareHeaderFile.ListEntry);

        CONTEXT_START_TIMERS(PrepareHeaderFile);

        SubmitThreadpoolWork(Context->FileWork);

        PreparedHeader = TRUE;
    }

    //
    // The number of edges in our graph is equal to the number of keys in the
    // input data set if modulus masking is in use.  It will be rounded up to
    // a power of 2 otherwise.
    //

    NumberOfEdges.QuadPart = Table->Keys->NumberOfElements.QuadPart;

    //
    // Sanity check we're under MAX_ULONG.
    //

    ASSERT(!NumberOfEdges.HighPart);

    NumberOfKeys = NumberOfEdges.LowPart;

    //
    // Determine the number of vertices.  If we've reached here due to a resize
    // event, Table->RequestedNumberOfTableElements will be non-zero, and takes
    // precedence.  Otherwise, determine the vertices heuristically.
    //

    if (Table->RequestedNumberOfTableElements.QuadPart) {

        NumberOfVertices.QuadPart = (
            Table->RequestedNumberOfTableElements.QuadPart
        );

        if (IsModulusMasking(MaskFunctionId)) {

            //
            // Nothing more to do with modulus masking; we'll verify the number
            // of vertices below.
            //

            NOTHING;

        } else {

            //
            // For non-modulus masking, make sure the number of vertices are
            // rounded up to a power of 2.  The number of edges will be rounded
            // up to a power of 2 from the number of keys.
            //

            NumberOfVertices.QuadPart = (
                RoundUpPowerOf2(NumberOfVertices.LowPart)
            );

            NumberOfEdges.QuadPart = (
                RoundUpPowerOf2(NumberOfEdges.LowPart)
            );

        }

    } else {

        //
        // No table size was requested, so we need to determine how many
        // vertices to use heuristically.  The main factor is what type of
        // masking has been requested.  The chm.c implementation, which is
        // modulus based, uses a size multiplier (c) of 2.09, and calculates
        // the final size via ceil(nedges * (double)2.09).  We can avoid the
        // need for doubles and linking with a math library (to get ceil())
        // and just use ~2.25, which we can calculate by adding the result
        // of right shifting the number of edges by 1 to the result of left
        // shifting said edge count by 2 (simulating multiplication by 0.25).
        //
        // If we're dealing with modulus masking, this will be the exact number
        // of vertices used.  For other types of masking, we need the edges size
        // to be a power of 2, and the vertices size to be the next power of 2.
        //

        if (IsModulusMasking(MaskFunctionId)) {

            NumberOfVertices.QuadPart = NumberOfEdges.QuadPart << 1ULL;
            NumberOfVertices.QuadPart += NumberOfEdges.QuadPart >> 2ULL;

        } else {

            //
            // Round up the edges to a power of 2.
            //

            NumberOfEdges.QuadPart = RoundUpPowerOf2(NumberOfEdges.LowPart);

            //
            // Make sure we haven't overflowed.
            //

            ASSERT(!NumberOfEdges.HighPart);

            //
            // For the number of vertices, round the number of edges up to the
            // next power of 2.
            //

            NumberOfVertices.QuadPart = (
                RoundUpNextPowerOf2(NumberOfEdges.LowPart)
            );

        }
    }

    //
    // Another sanity check we haven't exceeded MAX_ULONG.
    //

    ASSERT(!NumberOfVertices.HighPart);

    //
    // The r-graph (r = 2) nature of this implementation results in various
    // arrays having twice the number of elements indicated by the edge count.
    // Capture this number now, as we need it in various size calculations.
    //

    TotalNumberOfEdges.QuadPart = NumberOfEdges.QuadPart;
    TotalNumberOfEdges.QuadPart <<= 1ULL;

    //
    // Another overflow sanity check.
    //

    ASSERT(!TotalNumberOfEdges.HighPart);

    //
    // Make sure vertices > edges.
    //

    ASSERT(NumberOfVertices.QuadPart > NumberOfEdges.QuadPart);

    //
    // Calculate the size required for the DeletedEdges bitmap buffer.  One
    // bit is used per TotalNumberOfEdges.  Convert the bits into bytes by
    // shifting right 3 (dividing by 8) then align it up to a 16 byte boundary.
    // We add 1 before shifting to account 1-based bitmaps vs 0-based indices.
    //

    DeletedEdgesBitmapBufferSizeInBytes.QuadPart = (
        ALIGN_UP(((ALIGN_UP(TotalNumberOfEdges.QuadPart + 1, 8)) >> 3), 16)
    );

    ASSERT(!DeletedEdgesBitmapBufferSizeInBytes.HighPart);

    //
    // Calculate the size required for the VisitedVertices bitmap buffer.  One
    // bit is used per NumberOfVertices.  Convert the bits into bytes by
    // shifting right 3 (dividing by 8) then align it up to a 16 byte boundary.
    // We add 1 before shifting to account 1-based bitmaps vs 0-based indices.
    //

    VisitedVerticesBitmapBufferSizeInBytes.QuadPart = (
        ALIGN_UP(((ALIGN_UP(NumberOfVertices.QuadPart + 1, 8)) >> 3), 16)
    );

    ASSERT(!VisitedVerticesBitmapBufferSizeInBytes.HighPart);

    //
    // Calculate the size required for the AssignedBitmap bitmap buffer.  One
    // bit is used per NumberOfVertices.  Convert the bits into bytes by shifting
    // right 3 (dividing by 8) then align it up to a 16 byte boundary.
    // We add 1 before shifting to account 1-based bitmaps vs 0-based indices.
    //

    AssignedBitmapBufferSizeInBytes.QuadPart = (
        ALIGN_UP(((ALIGN_UP(NumberOfVertices.QuadPart + 1, 8)) >> 3), 16)
    );

    ASSERT(!AssignedBitmapBufferSizeInBytes.HighPart);

    //
    // Calculate the size required for the IndexBitmap bitmap buffer.  One
    // bit is used per NumberOfVertices.  Convert the bits into bytes by shifting
    // right 3 (dividing by 8) then align it up to a 16 byte boundary.
    // We add 1 before shifting to account 1-based bitmaps vs 0-based indices.
    //

    IndexBitmapBufferSizeInBytes.QuadPart = (
        ALIGN_UP(((ALIGN_UP(NumberOfVertices.QuadPart + 1, 8)) >> 3), 16)
    );

    ASSERT(!IndexBitmapBufferSizeInBytes.HighPart);

    //
    // Calculate the sizes required for each of the arrays.  We collect them
    // into independent variables as it makes carving up the allocated buffer
    // easier down the track.
    //

    EdgesSizeInBytes = (
        ALIGN_UP_POINTER(sizeof(*Graph->Edges) * TotalNumberOfEdges.QuadPart)
    );

    NextSizeInBytes = (
        ALIGN_UP_POINTER(sizeof(*Graph->Next) * TotalNumberOfEdges.QuadPart)
    );

    FirstSizeInBytes = (
        ALIGN_UP_POINTER(sizeof(*Graph->First) * NumberOfVertices.QuadPart)
    );

    PrevSizeInBytes = (
        ALIGN_UP_POINTER(sizeof(*Graph->Prev) * TotalNumberOfEdges.QuadPart)
    );

    AssignedSizeInBytes = (
        ALIGN_UP_POINTER(sizeof(*Graph->Assigned) * NumberOfVertices.QuadPart)
    );

    //
    // Calculate the size required for the values array.  This is used as part
    // of verification, where we essentially do Insert(Key, Key) in combination
    // with bitmap tracking of assigned indices, which allows us to detect if
    // there are any colliding indices, and if so, what was the previous key
    // that mapped to the same index.
    //

    ValuesSizeInBytes = (
        ALIGN_UP_POINTER(sizeof(*Graph->Values) * NumberOfVertices.QuadPart)
    );

    //
    // Calculate the total size required for the underlying graph, such that
    // we can allocate memory via a single call to the allocator.
    //

    AllocSize.QuadPart = ALIGN_UP_POINTER(

        //
        // Account for the size of the graph structure.
        //

        sizeof(GRAPH) +

        //
        // Account for the size of the Graph->Edges array, which is double
        // sized.
        //

        EdgesSizeInBytes +

        //
        // Account for the size of the Graph->Next array; also double sized.
        //

        NextSizeInBytes +

        //
        // Account for the size of the Graph->First array.  This is sized
        // proportional to the number of vertices.
        //

        FirstSizeInBytes +

        //
        // Account for the size of the Graph->Prev array, also double sized.
        //

        PrevSizeInBytes +

        //
        // Account for Graph->Assigned array of vertices.
        //

        AssignedSizeInBytes +

        //
        // Account for the Table->Values array of values for the perfect hash
        // table, indexed via the result of the table's Index() method.
        //

        ValuesSizeInBytes +

        //
        // Account for the size of the bitmap buffer for Graph->DeletedEdges.
        //

        DeletedEdgesBitmapBufferSizeInBytes.QuadPart +

        //
        // Account for the size of the bitmap buffer for Graph->VisitedVertices.
        //

        VisitedVerticesBitmapBufferSizeInBytes.QuadPart +

        //
        // Account for the size of the bitmap buffer for Graph->AssignedBitmap.
        //

        AssignedBitmapBufferSizeInBytes.QuadPart +

        //
        // Account for the size of the bitmap buffer for Graph->IndexBitmap.
        //

        IndexBitmapBufferSizeInBytes.QuadPart +

        //
        // Keep a dummy 0 at the end such that the last item above can use an
        // addition sign at the end of it, which minimizes the diff churn when
        // adding a new size element.
        //

        0

    );

    //
    // Capture the number of bitmaps here, where it's close to the lines above
    // that indicate how many bitmaps we're dealing with.  The number of bitmaps
    // accounted for above should match this number.  Visually confirm this any
    // time a new bitmap buffer is accounted for.
    //
    // N.B. We ASSERT() in InitializeGraph() if we detect a mismatch between
    //      Info->NumberOfBitmaps and a local counter incremented each time
    //      we initialize a bitmap.
    //

    NumberOfBitmaps = 4;

    //
    // Sanity check the size hasn't overflowed.
    //

    ASSERT(!AllocSize.HighPart);

    //
    // Calculate the number of pages required by each graph, then extrapolate
    // the number of guard pages and total number of pages.  We currently use
    // 4KB for the page size (i.e. we're not using large pages).
    //

    PageSize = PAGE_SIZE;
    PageShift = (USHORT)TrailingZeros(PageSize);
    NumberOfGraphs = (USHORT)Context->MaximumConcurrency;
    NumberOfPagesPerGraph = BYTES_TO_PAGES(AllocSize.LowPart);
    NumberOfGuardPages = (USHORT)Context->MaximumConcurrency;
    TotalNumberOfPages = (
        (NumberOfGraphs * NumberOfPagesPerGraph) +
        NumberOfGuardPages
    );
    GraphSizeInBytesIncludingGuardPage = (
        (ULONGLONG)PageSize +
        ((ULONGLONG)NumberOfPagesPerGraph * (ULONGLONG)PageSize)
    );

    //
    // Create multiple buffers separated by guard pages using a single call
    // to VirtualAllocEx().
    //

    Result = Rtl->Vtbl->CreateMultipleBuffers(Rtl,
                                              &ProcessHandle,
                                              PageSize,
                                              NumberOfGraphs,
                                              NumberOfPagesPerGraph,
                                              NULL,
                                              NULL,
                                              &UsableBufferSizeInBytesPerBuffer,
                                              &TotalBufferSizeInBytes,
                                              &BaseAddress);

    if (FAILED(Result)) {
        SYS_ERROR(VirtualAlloc);
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    //
    // N.B. Subsequent errors must 'goto Error' at this point to ensure our
    //      cleanup logic kicks in.
    //

    //
    // Assert the sizes returned by the buffer allocation match what we're
    // expecting.
    //

    ExpectedTotalBufferSizeInBytes = (
        (ULONGLONG)TotalNumberOfPages *
        (ULONGLONG)PageSize
    );

    ExpectedUsableBufferSizeInBytesPerBuffer = (
        (ULONGLONG)NumberOfPagesPerGraph *
        (ULONGLONG)PageSize
    );

    ASSERT(TotalBufferSizeInBytes == ExpectedTotalBufferSizeInBytes);
    ASSERT(UsableBufferSizeInBytesPerBuffer ==
           ExpectedUsableBufferSizeInBytesPerBuffer);

    //
    // Initialize the GRAPH_INFO structure with all the sizes captured earlier.
    // (We zero it first just to ensure any of the padding fields are cleared.)
    //

    ZeroStruct(Info);

    Info.PageSize = PageSize;
    Info.AllocSize = AllocSize.QuadPart;
    Info.Context = Context;
    Info.BaseAddress = BaseAddress;
    Info.NumberOfPagesPerGraph = NumberOfPagesPerGraph;
    Info.NumberOfGraphs = NumberOfGraphs;
    Info.NumberOfBitmaps = NumberOfBitmaps;
    Info.SizeOfGraphStruct = sizeof(GRAPH);
    Info.EdgesSizeInBytes = EdgesSizeInBytes;
    Info.NextSizeInBytes = NextSizeInBytes;
    Info.FirstSizeInBytes = FirstSizeInBytes;
    Info.PrevSizeInBytes = PrevSizeInBytes;
    Info.AssignedSizeInBytes = AssignedSizeInBytes;
    Info.ValuesSizeInBytes = ValuesSizeInBytes;
    Info.AllocSize = AllocSize.QuadPart;
    Info.FinalSize = UsableBufferSizeInBytesPerBuffer;

    Info.DeletedEdgesBitmapBufferSizeInBytes = (
        DeletedEdgesBitmapBufferSizeInBytes.QuadPart
    );

    Info.VisitedVerticesBitmapBufferSizeInBytes = (
        VisitedVerticesBitmapBufferSizeInBytes.QuadPart
    );

    Info.AssignedBitmapBufferSizeInBytes = (
        AssignedBitmapBufferSizeInBytes.QuadPart
    );

    Info.IndexBitmapBufferSizeInBytes = (
        IndexBitmapBufferSizeInBytes.QuadPart
    );

    //
    // Capture the system allocation granularity.  This is used to align the
    // backing memory maps used for the table array.
    //

    GetSystemInfo(&SystemInfo);
    Info.AllocationGranularity = SystemInfo.dwAllocationGranularity;

    //
    // Copy the dimensions over.
    //

    Dim = &Info.Dimensions;
    Dim->NumberOfEdges = NumberOfEdges.LowPart;
    Dim->TotalNumberOfEdges = TotalNumberOfEdges.LowPart;
    Dim->NumberOfVertices = NumberOfVertices.LowPart;

    Dim->NumberOfEdgesPowerOf2Exponent = (BYTE)(
        TrailingZeros64(RoundUpPowerOf2(NumberOfEdges.LowPart))
    );

    Dim->NumberOfEdgesNextPowerOf2Exponent = (BYTE)(
        TrailingZeros64(RoundUpNextPowerOf2(NumberOfEdges.LowPart))
    );

    Dim->NumberOfVerticesPowerOf2Exponent = (BYTE)(
        TrailingZeros64(RoundUpPowerOf2(NumberOfVertices.LowPart))
    );

    Dim->NumberOfVerticesNextPowerOf2Exponent = (BYTE)(
        TrailingZeros64(RoundUpNextPowerOf2(NumberOfVertices.LowPart))
    );

    //
    // If non-modulus masking is active, initialize the edge and vertex masks.
    //

    if (!IsModulusMasking(MaskFunctionId)) {

        Info.EdgeMask = NumberOfEdges.LowPart - 1;
        Info.VertexMask = NumberOfVertices.LowPart - 1;

        //
        // Sanity check our masks are correct: their popcnts should match the
        // exponent value identified above whilst filling out the dimensions
        // structure.
        //

        ASSERT(PopulationCount32(Info.EdgeMask) ==
               Dim->NumberOfEdgesPowerOf2Exponent);

        ASSERT(PopulationCount32(Info.VertexMask) ==
               Dim->NumberOfVerticesPowerOf2Exponent);

    }

    //
    // Set the Modulus, Size, Shift, Mask and Fold fields of the table, such
    // that the Hash and Mask vtbl functions operate correctly.
    //
    // N.B. Shift, Mask and Fold are meaningless for modulus masking.
    //
    // N.B. If you change these fields, you'll probably need to change something
    //      in LoadPerfectHashTableImplChm01() too.
    //

    Table->HashModulus = NumberOfVertices.LowPart;
    Table->IndexModulus = NumberOfEdges.LowPart;
    Table->HashSize = NumberOfVertices.LowPart;
    Table->IndexSize = NumberOfEdges.LowPart;
    Table->HashShift = TrailingZeros(Table->HashSize);
    Table->IndexShift = TrailingZeros(Table->IndexSize);
    Table->HashMask = (Table->HashSize - 1);
    Table->IndexMask = (Table->IndexSize - 1);
    Table->HashFold = Table->HashShift >> 3;
    Table->IndexFold = Table->IndexShift >> 3;

    //
    // Save the on-disk representation of the graph information.  This is a
    // smaller subset of data needed in order to load a previously-solved
    // graph as a perfect hash table.  The data resides in an NTFS stream named
    // :Info off the main perfect hash table file.  It will have been mapped for
    // us already at Table->InfoStreamBaseAddress.
    //

    GraphInfoOnDisk = (PGRAPH_INFO_ON_DISK)Table->InfoStreamBaseAddress;
    ASSERT(GraphInfoOnDisk);
    TableInfoOnDisk = &GraphInfoOnDisk->TableInfoOnDisk;
    TableInfoOnDisk->Magic.LowPart = TABLE_INFO_ON_DISK_MAGIC_LOWPART;
    TableInfoOnDisk->Magic.HighPart = TABLE_INFO_ON_DISK_MAGIC_HIGHPART;
    TableInfoOnDisk->SizeOfStruct = sizeof(*GraphInfoOnDisk);
    TableInfoOnDisk->Flags.AsULong = 0;
    TableInfoOnDisk->Concurrency = Context->MaximumConcurrency;
    TableInfoOnDisk->AlgorithmId = Context->AlgorithmId;
    TableInfoOnDisk->MaskFunctionId = Context->MaskFunctionId;
    TableInfoOnDisk->HashFunctionId = Context->HashFunctionId;
    TableInfoOnDisk->KeySizeInBytes = sizeof(ULONG);
    TableInfoOnDisk->HashSize = Table->HashSize;
    TableInfoOnDisk->IndexSize = Table->IndexSize;
    TableInfoOnDisk->HashShift = Table->HashShift;
    TableInfoOnDisk->IndexShift = Table->IndexShift;
    TableInfoOnDisk->HashMask = Table->HashMask;
    TableInfoOnDisk->IndexMask = Table->IndexMask;
    TableInfoOnDisk->HashFold = Table->HashFold;
    TableInfoOnDisk->IndexFold = Table->IndexFold;
    TableInfoOnDisk->HashModulus = Table->HashModulus;
    TableInfoOnDisk->IndexModulus = Table->IndexModulus;
    TableInfoOnDisk->NumberOfKeys.QuadPart = (
        Table->Keys->NumberOfElements.QuadPart
    );
    TableInfoOnDisk->NumberOfSeeds = ((
        FIELD_OFFSET(GRAPH, LastSeed) -
        FIELD_OFFSET(GRAPH, FirstSeed)
    ) / sizeof(ULONG)) + 1;

    //
    // This will change based on masking type and whether or not the caller
    // has provided a value for NumberOfTableElements.  For now, keep it as
    // the number of vertices.
    //

    TableInfoOnDisk->NumberOfTableElements.QuadPart = (
        NumberOfVertices.QuadPart
    );

    CopyMemory(&GraphInfoOnDisk->Dimensions, Dim, sizeof(*Dim));

    //
    // Set the context's main work callback to our worker routine, and the algo
    // context to our graph info structure.
    //

    Context->MainWorkCallback = ProcessGraphCallbackChm01;
    Context->AlgorithmContext = &Info;

    //
    // Set the context's file work callback to our worker routine.
    //

    Context->FileWorkCallback = FileWorkCallbackChm01;

    //
    // Prepare the initial "table file preparation" work callback.  This will
    // extend the backing file to the appropriate size.
    //

    ZeroStruct(PrepareTableFile);
    PrepareTableFile.FileWorkId = FileWorkPrepareTableId;
    InterlockedPushEntrySList(&Context->FileWorkListHead,
                              &PrepareTableFile.ListEntry);

    CONTEXT_START_TIMERS(PrepareTableFile);

    SubmitThreadpoolWork(Context->FileWork);

    //
    // Capture initial cycles as reported by __rdtsc() and the performance
    // counter.  The former is used to report a raw cycle count, the latter
    // is used to convert to microseconds reliably (i.e. unaffected by turbo
    // boosting).
    //

    QueryPerformanceFrequency(&Context->Frequency);

    CONTEXT_START_TIMERS(Solve);

    //
    // We're ready to create threadpool work for the graph.
    //

    Buffer = (PBYTE)BaseAddress;
    Unusable = Buffer;

    for (Index = 0; Index < NumberOfGraphs; Index++) {

        //
        // Invariant check: at the top of the loop, Buffer and Unusable should
        // point to the same address (which will be the base of the current
        // graph being processed).  Assert this invariant now.
        //

        ASSERT(Buffer == Unusable);

        //
        // Carve out the graph pointer, and bump the unusable pointer past the
        // graph's pages, such that it points to the first byte of the guard
        // page.
        //

        Graph = (PGRAPH)Buffer;
        Unusable = Buffer + UsableBufferSizeInBytesPerBuffer;

        //
        // Sanity check the page alignment logic.  If we subtract 1 byte from
        // Unusable, it should reside on a different page.  Additionally, the
        // two pages should be separated by at most a single page size.
        //

        ThisPage = ALIGN_DOWN(Unusable,   PageSize);
        LastPage = ALIGN_DOWN(Unusable-1, PageSize);
        ASSERT(LastPage < ThisPage);
        ASSERT((ThisPage - LastPage) == PageSize);

        //
        // Verify the guard page is working properly by wrapping an attempt to
        // write to it in a structured exception handler that will catch the
        // access violation trap.
        //
        // N.B. We only do this if we're not actively being debugged, as the
        //      traps get dispatched to the debugger engine first as part of
        //      the "first-pass" handling logic of the kernel.
        //

        if (!IsDebuggerPresent()) {

            CaughtException = FALSE;

            TRY_PROBE_MEMORY {

                *Unusable = 1;

            } CATCH_EXCEPTION_ACCESS_VIOLATION{

                CaughtException = TRUE;

            }

            ASSERT(CaughtException);
        }

        //
        // Guard page is working properly.  Push the graph onto the context's
        // main work list head and submit the corresponding threadpool work.
        //

        InterlockedPushEntrySList(&Context->MainWorkListHead,
                                  &Graph->ListEntry);
        SubmitThreadpoolWork(Context->MainWork);

        //
        // Advance the buffer past the graph size and guard page.  Copy the
        // same address to the Unusable variable as well, such that our top
        // of the loop invariants hold true.
        //

        Buffer += GraphSizeInBytesIncludingGuardPage;
        Unusable = Buffer;

        //
        // If our key set size is small and our maximum concurrency is large,
        // we may have already solved the graph, in which case, we can stop
        // submitting new solver attempts and just break out of the loop here.
        //

        if (!ShouldWeContinueTryingToSolveGraphChm01(Context)) {
            break;
        }
    }

    //
    // Wait on the context's events.
    //

    Events[0] = Context->SucceededEvent;
    Events[1] = Context->CompletedEvent;
    Events[2] = Context->ShutdownEvent;
    Events[3] = Context->FailedEvent;
    Events[4] = Context->TryLargerTableSizeEvent;

    WaitResult = WaitForMultipleObjects(ARRAYSIZE(Events),
                                        Events,
                                        FALSE,
                                        INFINITE);

    //
    // If the wait result indicates the try larger table size event was set,
    // deal with that, first.
    //

    if (WaitResult == WAIT_OBJECT_0+4) {

        //
        // The number of attempts at solving this graph have exceeded the
        // threshold.  Set the shutdown event in order to trigger all worker
        // threads to abort their current attempts and wait on the main thread
        // work to complete.
        //

        SetEvent(Context->ShutdownEvent);
        WaitForThreadpoolWorkCallbacks(Context->MainWork, TRUE);

        //
        // Perform a blocking wait for the prepare table file work to complete.
        // (It would be highly unlikely that this event hasn't been set yet.)
        //

        WaitResult = WaitForSingleObject(Context->PreparedTableFileEvent,
                                         INFINITE);

        if (WaitResult != WAIT_OBJECT_0) {
            SYS_ERROR(WaitForSingleObject);
            Result = PH_E_SYSTEM_CALL_FAILED;
            goto Error;
        }

        //
        // N.B. We don't need to wait for the C header file preparation to
        //      complete here as the size of that file isn't coupled so tightly
        //      with the underlying table size.
        //

        //
        // There are no more threadpool callbacks running.  However, a thread
        // could have finished a solution between the time the try larger table
        // size event was set, and this point.  So, check the finished count
        // first.  If it indicates a solution, jump to that handler code.
        //

        if (Context->FinishedCount > 0) {
            goto FinishedSolution;
        }

        //
        // Check to see if we've exceeded the maximum number of resize events.
        //

        if (TableInfoOnDisk->NumberOfTableResizeEvents >= Context->ResizeLimit) {
            Result = PH_E_MAXIMUM_NUMBER_OF_TABLE_RESIZE_EVENTS_REACHED;
            goto Error;
        }

        //
        // Destroy the existing buffer we allocated for this attempt.  We'll
        // need a new, larger one to accommodate the resize.
        //

        Result = Rtl->Vtbl->DestroyBuffer(Rtl, ProcessHandle, &BaseAddress);
        if (FAILED(Result)) {
            SYS_ERROR(VirtualFree);
            Result = PH_E_SYSTEM_CALL_FAILED;
            goto Error;
        }

        //
        // Increment the resize counter and update the total number of attempts
        // in the header.  Then, determine how close we came to solving the
        // graph, and store that in the header as well if it's the best so far
        // (or no previous version is present).
        //

        TableInfoOnDisk->NumberOfTableResizeEvents++;
        TableInfoOnDisk->TotalNumberOfAttemptsWithSmallerTableSizes += (
            Context->Attempts
        );

        Closest = NumberOfEdges.LowPart - Context->HighestDeletedEdgesCount;
        LastClosest = (
            TableInfoOnDisk->ClosestWeCameToSolvingGraphWithSmallerTableSizes
        );

        if (!LastClosest || Closest < LastClosest) {
            TableInfoOnDisk->ClosestWeCameToSolvingGraphWithSmallerTableSizes = (
                Closest
            );
        }

        //
        // If this is our first resize, capture the initial size we used.
        //

        if (!TableInfoOnDisk->InitialTableSize) {
            TableInfoOnDisk->InitialTableSize = NumberOfVertices.QuadPart;
        }

        //
        // Reset the remaining counters.
        //

        Context->Attempts = 0;
        Context->FailedAttempts = 0;
        Context->HighestDeletedEdgesCount = 0;

        //
        // Double the vertex count.  If we have overflowed max ULONG, abort.
        //

        Table->RequestedNumberOfTableElements.QuadPart = (
            NumberOfVertices.QuadPart
        );

        Table->RequestedNumberOfTableElements.QuadPart <<= 1ULL;

        if (Table->RequestedNumberOfTableElements.HighPart) {
            Result = PH_E_REQUESTED_NUMBER_OF_TABLE_ELEMENTS_TOO_LARGE;
            goto Error;
        }

        //
        // Unmap the existing mapping and close the section.
        //

        _Analysis_assume_(Table->BaseAddress != NULL);
        if (!UnmapViewOfFile(Table->BaseAddress)) {
            SYS_ERROR(UnmapViewOfFile);
            Result = PH_E_SYSTEM_CALL_FAILED;
            goto Error;
        }
        Table->BaseAddress = NULL;

        _Analysis_assume_(Table->MappingHandle != NULL);
        if (!CloseHandle(Table->MappingHandle)) {
            SYS_ERROR(CloseHandle);
            Result = PH_E_SYSTEM_CALL_FAILED;
            goto Error;
        }
        Table->MappingHandle = NULL;

        //
        // Jump back to the start and try again with a larger vertex count.
        //

        goto RetryWithLargerTableSize;
    }

    //
    // The wait result did not indicate a resize event.  Ignore the wait
    // result for now; determine if the graph solving was successful by the
    // finished count of the context.  We'll corroborate that with whatever
    // events have been signaled shortly.
    //

    Success = (Context->FinishedCount > 0);

    if (!Success) {

        BOOL CancelPending = TRUE;

        //
        // Invariant check: if no worker thread registered a solved graph (i.e.
        // Context->FinishedCount > 0), then verify that the shutdown event was
        // set.  If our WaitResult above indicates WAIT_OBJECT_2, we're done.
        // If not, verify explicitly.
        //

        if (WaitResult != WAIT_OBJECT_0+2) {

            //
            // Manually test that the shutdown event has been signaled.
            //

            WaitResult = WaitForSingleObject(Context->ShutdownEvent, 0);

            if (WaitResult != WAIT_OBJECT_0) {
                SYS_ERROR(WaitForSingleObject);
                Result = PH_E_SYSTEM_CALL_FAILED;
                goto Error;
            }
        }

        //
        // Wait for the main thread work group members.  This will block until
        // all the worker threads have returned.  We need to put this in place
        // prior to jumping to the End: label as that step will destroy the
        // buffer we allocated earlier for the parallel graphs, which we mustn't
        // do if any threads are still working.
        //

        WaitForThreadpoolWorkCallbacks(Context->MainWork, CancelPending);

        //
        // Perform the same operation for the file work threadpool.  Note that
        // the only work we've dispatched to this pool at this point is the
        // initial table and header file preparation work.
        //

        WaitForThreadpoolWorkCallbacks(Context->FileWork, CancelPending);

        goto End;
    }

    //
    // Pop the winning graph off the finished list head.
    //

FinishedSolution:

    ListEntry = InterlockedPopEntrySList(&Context->FinishedWorkListHead);
    ASSERT(ListEntry);

    Graph = CONTAINING_RECORD(ListEntry, GRAPH, ListEntry);

    //
    // Note this graph as the one solved to the context.  This is used by the
    // save file work callback we dispatch below.
    //

    Context->SolvedContext = Graph;

    //
    // Graphs always pass verification in normal circumstances.  The only time
    // they don't is if there's an internal bug in our code.  So, knowing that
    // the graph is probably correct, we can dispatch the file work required to
    // save it to disk to the file work threadpool whilst we verify it has been
    // solved correctly.
    //

    ZeroStruct(SaveTableFile);
    SaveTableFile.FileWorkId = FileWorkSaveTableId;

    //
    // Before we dispatch the save file work, make sure the preparation has
    // completed.
    //

    WaitResult = WaitForSingleObject(Context->PreparedTableFileEvent, INFINITE);
    if (WaitResult != WAIT_OBJECT_0) {
        SYS_ERROR(WaitForSingleObject);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    if (Context->TableFileWorkErrors > 0) {
        Result = Context->TableFileWorkLastResult;
        if (Result == S_OK) {
            Result = PH_E_ERROR_PREPARING_TABLE_FILE;
        }
        goto Error;
    }

    //
    // Push this work item to the file work list head and submit the threadpool
    // work for it.
    //

    CONTEXT_START_TIMERS(SaveTableFile);

    InterlockedPushEntrySList(&Context->FileWorkListHead,
                              &SaveTableFile.ListEntry);
    SubmitThreadpoolWork(Context->FileWork);

    //
    // As above, dispatch a save header file work item in parallel to graph
    // verification.
    //

    CONTEXT_START_TIMERS(SaveHeaderFile);

    ZeroStruct(SaveHeaderFile);
    SaveHeaderFile.FileWorkId = FileWorkSaveHeaderId;

    WaitResult = WaitForSingleObject(Context->PreparedHeaderFileEvent,
                                     INFINITE);

    if (WaitResult != WAIT_OBJECT_0) {
        SYS_ERROR(WaitForSingleObject);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    if (Context->HeaderFileWorkErrors > 0) {
        Result = Context->HeaderFileWorkLastResult;
        if (Result == S_OK) {
            Result = PH_E_ERROR_PREPARING_HEADER_FILE;
        }
        goto Error;
    }

    CONTEXT_START_TIMERS(SaveHeaderFile);

    InterlockedPushEntrySList(&Context->FileWorkListHead,
                              &SaveHeaderFile.ListEntry);
    SubmitThreadpoolWork(Context->FileWork);

    //
    // Capture another round of cycles and performance counter values, then
    // continue with verification of the solution.
    //

    CONTEXT_START_TIMERS(Verify);

    Result = VerifySolvedGraph(Graph);

    CONTEXT_END_TIMERS(Verify);

    //
    // Set the verified table event (regardless of whether or not we succeeded
    // in verification).  The save file work will be waiting upon it in order to
    // write the final timing details to the on-disk header.
    //

    if (!SetEvent(Context->VerifiedTableEvent)) {
        SYS_ERROR(SetEvent);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    if (FAILED(Result)) {
        Result = PH_E_TABLE_VERIFICATION_FAILED;
        goto Error;
    }

    //
    // Wait on the saved file events before returning.
    //

    WaitForAllEvents = TRUE;
    SaveFileEvents[0] = Context->SavedTableFileEvent;
    SaveFileEvents[1] = Context->SavedHeaderFileEvent;

    WaitResult = WaitForMultipleObjects(ARRAYSIZE(SaveFileEvents),
                                        SaveFileEvents,
                                        WaitForAllEvents,
                                        INFINITE);

    if (WaitResult != WAIT_OBJECT_0) {
        SYS_ERROR(WaitForSingleObject);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    if (Context->TableFileWorkErrors > 0) {
        Result = Context->TableFileWorkLastResult;
        if (Result == S_OK) {
            Result = PH_E_ERROR_SAVING_TABLE_FILE;
        }
        goto Error;
    }

    if (Context->HeaderFileWorkErrors > 0) {
        Result = Context->HeaderFileWorkLastResult;
        if (Result == S_OK) {
            Result = PH_E_ERROR_SAVING_HEADER_FILE;
        }
        goto Error;
    }

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

    //
    // Destroy the buffer we created earlier.
    //
    // N.B. Although we used Rtl->CreateMultipleBuffers(), we can still free
    //      the underlying buffer via Rtl->DestroyBuffer(), as only a single
    //      VirtualAllocEx() call was dispatched for the entire buffer.
    //

    if (BaseAddress && ProcessHandle) {
        Result = Rtl->Vtbl->DestroyBuffer(Rtl, ProcessHandle, &BaseAddress);
        if (FAILED(Result)) {
            SYS_ERROR(VirtualFree);
            Result = PH_E_SYSTEM_CALL_FAILED;
        }
    }

    //
    // Explicitly reset all events before returning.
    //

    Event = (PHANDLE)&Context->FirstEvent;
    NumberOfEvents = GetNumberOfContextEvents(Context);

    for (Index = 0; Index < NumberOfEvents; Index++, Event++) {

        if (!ResetEvent(*Event)) {
            SYS_ERROR(ResetEvent);
            Result = PH_E_SYSTEM_CALL_FAILED;
        }
    }

    return Result;
}

_Use_decl_annotations_
HRESULT
LoadPerfectHashTableImplChm01(
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
    PTABLE_INFO_ON_DISK OnDisk;

    OnDisk = Table->TableInfoOnDisk;

    Table->HashSize = OnDisk->HashSize;
    Table->IndexSize = OnDisk->IndexSize;
    Table->HashShift = OnDisk->HashShift;
    Table->IndexShift = OnDisk->IndexShift;
    Table->HashMask = OnDisk->HashMask;
    Table->IndexMask = OnDisk->IndexMask;
    Table->HashFold = OnDisk->HashFold;
    Table->IndexFold = OnDisk->IndexFold;
    Table->HashModulus = OnDisk->HashModulus;
    Table->IndexModulus = OnDisk->IndexModulus;

    return S_OK;
}

//
// The entry point into the actual per-thread solving attempts is the following
// routine.
//

_Use_decl_annotations_
VOID
ProcessGraphCallbackChm01(
    PTP_CALLBACK_INSTANCE Instance,
    PPERFECT_HASH_CONTEXT Context,
    PSLIST_ENTRY ListEntry
    )
/*++

Routine Description:

    This routine is the callback entry point for graph solving threads.  It
    will enter an infinite loop attempting to solve the graph; terminating
    only when the graph is solved or we detect another thread has solved it.

Arguments:

    Instance - Supplies a pointer to the callback instance for this invocation.

    Context - Supplies a pointer to the active context for the graph solving.

    ListEntry - Supplies a pointer to the list entry that was popped off the
        context's main work interlocked singly-linked list head.  The list
        entry will be the address of Graph->ListEntry, and thus, the Graph
        address can be obtained via the following CONTAINING_RECORD() construct:

            Graph = CONTAINING_RECORD(ListEntry, GRAPH, ListEntry);


Return Value:

    None.

--*/
{
    PRTL Rtl;
    PGRAPH Graph;
    ULONG Attempt = 0;
    PGRAPH_INFO Info;
    PRTL_FILL_PAGES FillPages;

    UNREFERENCED_PARAMETER(Instance);

    //
    // Resolve the graph base address from the list entry.  Nothing will be
    // filled in initially.
    //

    Graph = CONTAINING_RECORD(ListEntry, GRAPH, ListEntry);

    //
    // Resolve aliases.
    //

    Rtl = Context->Rtl;
    FillPages = Rtl->Vtbl->FillPages;

    //
    // The graph info structure will be stashed in the algo context field.
    //

    Info = (PGRAPH_INFO)Context->AlgorithmContext;

    //
    // Begin the solving loop.  InitializeGraph() generates new seed data,
    // so each loop iteration will be attempting to solve the graph uniquely.
    //

    while (ShouldWeContinueTryingToSolveGraphChm01(Context)) {

        InitializeGraph(Info, Graph);

        Graph->ThreadAttempt = ++Attempt;

        if (SolveGraph(Graph)) {

            //
            // Hey, we were the ones to solve it, great!
            //

            break;
        }

        //
        // Our attempt at solving failed.  Zero all pages associated with the
        // graph and then try again with new seed data.
        //

        FillPages(Rtl, (PCHAR)Graph, 0, Info->NumberOfPagesPerGraph);

    }

    return;
}

_Use_decl_annotations_
VOID
FileWorkCallbackChm01(
    PTP_CALLBACK_INSTANCE Instance,
    PPERFECT_HASH_CONTEXT Context,
    PSLIST_ENTRY ListEntry
    )
/*++

Routine Description:

    This routine is the callback entry point for file-oriented work we want
    to perform in the file work threadpool context.

Arguments:

    Instance - Supplies a pointer to the callback instance for this invocation.

    Context - Supplies a pointer to the active context for the graph solving.

    ListEntry - Supplies a pointer to the list entry that was popped off the
        context's file work interlocked singly-linked list head.

Return Value:

    None.

--*/
{
    PHANDLE Event;
    volatile HRESULT *Result;
    volatile LONG *Errors;
    volatile LONG *LastError;
    PFILE_WORK_ITEM Item;

    //
    // Resolve the work item base address from the list entry.
    //

    Item = CONTAINING_RECORD(ListEntry, FILE_WORK_ITEM, ListEntry);

    ASSERT(IsValidFileWorkId(Item->FileWorkId));

    switch (Item->FileWorkId) {

        case FileWorkPrepareTableId:

            Event = &Context->PreparedTableFileEvent;
            Result = &Context->TableFileWorkLastResult;
            Errors = &Context->TableFileWorkErrors;
            LastError = &Context->TableFileWorkLastError;
            *Result = PrepareTableCallbackChm01(Context);
            break;

        case FileWorkSaveTableId:

            Event = &Context->SavedTableFileEvent;
            Result = &Context->TableFileWorkLastResult;
            Errors = &Context->TableFileWorkErrors;
            LastError = &Context->TableFileWorkLastError;
            *Result = SaveTableCallbackChm01(Context);
            break;

        case FileWorkPrepareHeaderId:

            Event = &Context->PreparedHeaderFileEvent;
            Result = &Context->HeaderFileWorkLastResult;
            Errors = &Context->HeaderFileWorkErrors;
            LastError = &Context->HeaderFileWorkLastError;
            *Result = PrepareHeaderCallbackChm01(Context);
            break;

        case FileWorkSaveHeaderId:

            Event = &Context->SavedHeaderFileEvent;
            Result = &Context->HeaderFileWorkLastResult;
            Errors = &Context->HeaderFileWorkErrors;
            LastError = &Context->HeaderFileWorkLastError;
            *Result = SaveHeaderCallbackChm01(Context);
            break;

        default:

            //
            // Should never get here.
            //

            ASSERT(FALSE);
            return;
    }

    if (FAILED(*Result)) {
        InterlockedIncrement(Errors);
        *LastError = GetLastError();
    }

    //
    // Register the relevant event to be set when this threadpool callback
    // returns, then return.
    //

    SetEventWhenCallbackReturns(Instance, *Event);

    return;
}

_Use_decl_annotations_
HRESULT
PrepareTableCallbackChm01(
    PPERFECT_HASH_CONTEXT Context
    )
{
    PRTL Rtl;
    HRESULT Result = S_OK;
    PGRAPH_INFO Info;
    PVOID BaseAddress;
    HANDLE MappingHandle;
    PPERFECT_HASH_TABLE Table;
    ULARGE_INTEGER SectorAlignedSize;
    PGRAPH_INFO_ON_DISK GraphInfoOnDisk;

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Table = Context->Table;
    Info = (PGRAPH_INFO)Context->AlgorithmContext;
    GraphInfoOnDisk = (PGRAPH_INFO_ON_DISK)Table->InfoStreamBaseAddress;

    //
    // We need to extend the file to accommodate for the solved graph.
    //

    SectorAlignedSize.QuadPart = ALIGN_UP(Info->AssignedSizeInBytes,
                                          Info->AllocationGranularity);

    //
    // Create the file mapping for the sector-aligned size.  This will
    // extend the underlying file size accordingly.
    //

    MappingHandle = CreateFileMappingW(Table->FileHandle,
                                       NULL,
                                       PAGE_READWRITE,
                                       SectorAlignedSize.HighPart,
                                       SectorAlignedSize.LowPart,
                                       NULL);

    Table->MappingHandle = MappingHandle;

    if (!MappingHandle || MappingHandle == INVALID_HANDLE_VALUE) {
        SYS_ERROR(CreateFileMappingW);
        goto Error;
    }

    BaseAddress = MapViewOfFile(MappingHandle,
                                FILE_MAP_READ | FILE_MAP_WRITE,
                                0,
                                0,
                                SectorAlignedSize.QuadPart);

    Table->BaseAddress = BaseAddress;

    if (!BaseAddress) {
        SYS_ERROR(MapViewOfFile);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    CONTEXT_END_TIMERS(PrepareTableFile);

    //
    // We've successfully mapped an area of sufficient space to store
    // the underlying table array if a perfect hash table solution is
    // found.  Nothing more to do.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = PH_E_ERROR_PREPARING_TABLE_FILE;
    }

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}

_Use_decl_annotations_
HRESULT
SaveTableCallbackChm01(
    PPERFECT_HASH_CONTEXT Context
    )
{
    PRTL Rtl;
    BOOL Success;
    PULONG Dest;
    PGRAPH Graph;
    PULONG Source;
    ULONG WaitResult;
    HRESULT Result = S_OK;
    ULONGLONG SizeInBytes;
    LARGE_INTEGER EndOfFile;
    PPERFECT_HASH_TABLE Table;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Table = Context->Table;
    Table = Context->Table;
    Dest = (PULONG)Table->BaseAddress;
    Graph = (PGRAPH)Context->SolvedContext;
    Source = Graph->Assigned;
    TableInfoOnDisk = Table->TableInfoOnDisk;

    SizeInBytes = (
        TableInfoOnDisk->NumberOfTableElements.QuadPart *
        TableInfoOnDisk->KeySizeInBytes
    );

    //
    // The graph has been solved.  Copy the array of assigned values
    // to the mapped area we prepared earlier (above).
    //

    CopyMemory(Dest, Source, SizeInBytes);

    //
    // Save the seed values used by this graph.  (Everything else in
    // the on-disk info representation was saved earlier.)
    //

    TableInfoOnDisk->Seed1 = Graph->Seed1;
    TableInfoOnDisk->Seed2 = Graph->Seed2;
    TableInfoOnDisk->Seed3 = Graph->Seed3;
    TableInfoOnDisk->Seed4 = Graph->Seed4;

    //
    // Kick off a flush file buffers now before we wait on the verified
    // event.  The flush will be a blocking call.  The wait on verified
    // will be blocking if the event isn't signaled.  So, we may as well
    // get some useful blocking work done, before potentially going into
    // another wait state where we're not doing anything useful.
    //

    if (!FlushFileBuffers(Table->FileHandle)) {
        SYS_ERROR(FlushFileBuffers);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // Stop the save file timer here, after flushing the file buffers,
    // but before we potentially wait on the verified state.
    //

    CONTEXT_END_TIMERS(SaveTableFile);

    //
    // Wait on the verification complete event.  This is done in the
    // main thread straight after it dispatches our file work callback
    // (that ended up here).  We need to block on this event as we want
    // to save the timings for verification to the header.
    //

    WaitResult = WaitForSingleObject(Context->VerifiedTableEvent, INFINITE);
    if (WaitResult != WAIT_OBJECT_0) {
        SYS_ERROR(WaitForSingleObject);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // When we mapped the array in the work item above, we used a size
    // that was aligned with the system allocation granularity.  We now
    // want to set the end of file explicitly to the exact size of the
    // underlying array.  To do this, we unmap the view, delete the
    // section, set the file pointer to where we want, set the end of
    // file (which will apply the file pointer position as EOF), then
    // close the file handle.
    //

    if (!UnmapViewOfFile(Table->BaseAddress)) {
        SYS_ERROR(UnmapViewOfFile);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }
    Table->BaseAddress = NULL;

    if (!CloseHandle(Table->MappingHandle)) {
        SYS_ERROR(UnmapViewOfFile);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }
    Table->MappingHandle = NULL;

    EndOfFile.QuadPart = SizeInBytes;

    Success = SetFilePointerEx(Table->FileHandle,
                               EndOfFile,
                               NULL,
                               FILE_BEGIN);

    if (!Success) {
        SYS_ERROR(SetFilePointerEx);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    if (!SetEndOfFile(Table->FileHandle)) {
        SYS_ERROR(SetEndOfFile);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    if (!CloseHandle(Table->FileHandle)) {
        SYS_ERROR(CloseHandle);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    Table->FileHandle = NULL;

    //
    // Save the number of attempts and number of finished solutions.
    //

    TableInfoOnDisk->NumberOfAttempts = Context->Attempts;
    TableInfoOnDisk->NumberOfFailedAttempts = Context->FailedAttempts;
    TableInfoOnDisk->NumberOfSolutionsFound = Context->FinishedCount;

    //
    // Copy timer values for everything except the save header event.
    //

    CONTEXT_SAVE_TIMERS_TO_TABLE_INFO_ON_DISK(Solve);
    CONTEXT_SAVE_TIMERS_TO_TABLE_INFO_ON_DISK(Verify);
    CONTEXT_SAVE_TIMERS_TO_TABLE_INFO_ON_DISK(PrepareTableFile);
    CONTEXT_SAVE_TIMERS_TO_TABLE_INFO_ON_DISK(SaveTableFile);
    CONTEXT_SAVE_TIMERS_TO_TABLE_INFO_ON_DISK(PrepareHeaderFile);

    //
    // We need to wait on the header saved event before we can capture
    // the header timers.
    //

    WaitResult = WaitForSingleObject(Context->SavedHeaderFileEvent,
                                     INFINITE);

    if (WaitResult != WAIT_OBJECT_0) {
        SYS_ERROR(WaitForSingleObject);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    CONTEXT_SAVE_TIMERS_TO_TABLE_INFO_ON_DISK(SaveHeaderFile);

    //
    // Finalize the :Info stream the same way we handled the backing
    // file above; unmap, delete section, set file pointer, set eof,
    // close file.
    //

    if (!UnmapViewOfFile(Table->InfoStreamBaseAddress)) {
        SYS_ERROR(UnmapViewOfFile);
        goto Error;
    }
    Table->InfoStreamBaseAddress = NULL;

    if (!CloseHandle(Table->InfoStreamMappingHandle)) {
        SYS_ERROR(CloseHandle);
        goto Error;
    }
    Table->InfoStreamMappingHandle = NULL;

    //
    // The file size for the :Info stream will be the size of our
    // on-disk graph info structure.
    //

    EndOfFile.QuadPart = sizeof(GRAPH_INFO_ON_DISK);

    Success = SetFilePointerEx(Table->InfoStreamFileHandle,
                               EndOfFile,
                               NULL,
                               FILE_BEGIN);

    if (!Success) {
        SYS_ERROR(SetFilePointerEx);
        goto Error;
    }

    if (!SetEndOfFile(Table->InfoStreamFileHandle)) {
        SYS_ERROR(SetEndOfFile);
        goto Error;
    }

    if (!CloseHandle(Table->InfoStreamFileHandle)) {
        SYS_ERROR(CloseHandle);
        goto Error;
    }

    Table->InfoStreamFileHandle = NULL;

    //
    // We're done, jump to the end.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = PH_E_ERROR_SAVING_TABLE_FILE;
    }

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}

_Use_decl_annotations_
HRESULT
PrepareHeaderCallbackChm01(
    PPERFECT_HASH_CONTEXT Context
    )
{
    PRTL Rtl;
    PCHAR Base;
    PCHAR Output;
    ULONG Count;
    PULONG Long;
    ULONG Key;
    ULONGLONG Index;
    ULONGLONG NumberOfKeys;
    PCSTRING Name;
    HRESULT Result = S_OK;
    PPERFECT_HASH_KEYS Keys;
    PPERFECT_HASH_TABLE Table;
    const ULONG Indent = 0x20202020;

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Table = Context->Table;
    Keys = Table->Keys;
    Name = &Table->TableNameA;
    Base = (PCHAR)Table->HeaderBaseAddress;
    Output = Base;

    //
    // Write the keys.
    //

#define INDENT() {            \
    Long = (PULONG)Output;    \
    *Long = Indent;           \
    Output += sizeof(Indent); \
}

    OUTPUT_RAW("//\n// Compiled Perfect Hash Table.  Auto-generated.\n//\n\n");

    OUTPUT_RAW("#ifdef COMPILED_PERFECT_HASH_TABLE_INCLUDE_KEYS\n");
    OUTPUT_RAW("#pragma const_seg(\".cpht_keys\")\n");
    OUTPUT_RAW("static const unsigned long TableKeys[");
    OUTPUT_INT(Keys->NumberOfElements.QuadPart);
    OUTPUT_RAW("] = {\n");

    Count = 0;
    NumberOfKeys = Keys->NumberOfElements.QuadPart;

    for (Index = 0; Index < NumberOfKeys; Index++) {

        if (Count == 0) {
            INDENT();
        }

        Key = Keys->Keys[Index];

        OUTPUT_HEX(Key);

        *Output++ = ',';

        if (++Count == 4) {
            Count = 0;
            *Output++ = '\n';
        } else {
            *Output++ = ' ';
        }
    }

    //
    // If the last character written was a trailing space, replace
    // it with a newline.
    //

    if (*(Output - 1) == ' ') {
        *(Output - 1) = '\n';
    }

    OUTPUT_RAW("};\n#endif "
               "/* COMPILED_PERFECT_HASH_TABLE_INCLUDE_KEYS */\n\n");

    Table->HeaderSizeInBytes = ((ULONG_PTR)Output - (ULONG_PTR)Base);

    return Result;
}

_Use_decl_annotations_
HRESULT
SaveHeaderCallbackChm01(
    PPERFECT_HASH_CONTEXT Context
    )
{
    PRTL Rtl;
    BOOL Success;
    HRESULT Result = S_OK;
    LARGE_INTEGER EndOfFile;
    PPERFECT_HASH_TABLE Table;

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Table = Context->Table;

    //
    // XXX TODO: write table data.
    //


    //
    // Save the header file: flush file buffers, unmap the view, close the
    // mapping handle, set the file pointer, set EOF, and then close the handle.
    //

    if (!FlushFileBuffers(Table->HeaderFileHandle)) {
        SYS_ERROR(FlushFileBuffers);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    if (!UnmapViewOfFile(Table->HeaderBaseAddress)) {
        SYS_ERROR(UnmapViewOfFile);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }
    Table->HeaderBaseAddress = NULL;

    if (!CloseHandle(Table->HeaderMappingHandle)) {
        SYS_ERROR(UnmapViewOfFile);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }
    Table->HeaderMappingHandle = NULL;

    EndOfFile.QuadPart = Table->HeaderSizeInBytes;

    Success = SetFilePointerEx(Table->HeaderFileHandle,
                               EndOfFile,
                               NULL,
                               FILE_BEGIN);

    if (!Success) {
        SYS_ERROR(SetFilePointerEx);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    if (!SetEndOfFile(Table->HeaderFileHandle)) {
        SYS_ERROR(SetEndOfFile);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    if (!CloseHandle(Table->HeaderFileHandle)) {
        SYS_ERROR(CloseHandle);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    Table->HeaderFileHandle = NULL;

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = PH_E_ERROR_SAVING_HEADER_FILE;
    }

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}

SHOULD_WE_CONTINUE_TRYING_TO_SOLVE_GRAPH
    ShouldWeContinueTryingToSolveGraphChm01;

_Use_decl_annotations_
BOOLEAN
ShouldWeContinueTryingToSolveGraphChm01(
    PPERFECT_HASH_CONTEXT Context
    )
{
    ULONG WaitResult;
    HANDLE Events[4];
    USHORT NumberOfEvents = ARRAYSIZE(Events);

    Events[0] = Context->ShutdownEvent;
    Events[1] = Context->SucceededEvent;
    Events[2] = Context->FailedEvent;
    Events[3] = Context->CompletedEvent;

    //
    // Fast-path exit: if the finished count is not 0, then someone has already
    // solved the solution, and we don't need to wait on any of the events.
    //

    if (Context->FinishedCount > 0) {
        return FALSE;
    }

    //
    // N.B. We should probably switch this to simply use volatile field of the
    //      context structure to indicate whether or not the context is active.
    //      WaitForMultipleObjects() on four events seems a bit... excessive.
    //

    WaitResult = WaitForMultipleObjects(NumberOfEvents,
                                        Events,
                                        FALSE,
                                        0);

    //
    // The only situation where we continue attempting to solve the graph is
    // if the result from the wait is WAIT_TIMEOUT, which indicates none of
    // the events have been set.  We treat any other situation as an indication
    // to stop processing.  (This includes wait failures and abandonment.)
    //

    return (WaitResult == WAIT_TIMEOUT ? TRUE : FALSE);
}

_Use_decl_annotations_
HRESULT
PerfectHashTableIndexImplChm01(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    PULONG Index
    )
/*++

Routine Description:

    Looks up given key in a perfect hash table and returns its index.

    N.B. If Key did not appear in the original set the hash table was created
         from, the behavior of this routine is undefined.  (In practice, the
         key will hash to either an existing key's location or an empty slot,
         so there is potential for returning a non-unique index.)

Arguments:

    Table - Supplies a pointer to the table for which the key lookup is to be
        performed.

    Key - Supplies the key to look up.

    Index - Receives the index associated with this key.  The index will be
        between 0 and Table->HashSize-1, and can be safely used to offset
        directly into an appropriately sized array (e.g. Table->Values[]).

Return Value:

    S_OK on success, E_FAIL if the underlying hash function returned a failure.
    This will happen if the two hash values for a key happen to be identical.
    It shouldn't happen once a perfect graph has been created (i.e. it only
    happens when attempting to solve the graph).  The Index parameter will
    be cleared in the case of E_FAIL.

--*/
{
    ULONG Masked;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG MaskedLow;
    ULONG MaskedHigh;
    PULONG Assigned;
    ULONGLONG Combined;
    ULARGE_INTEGER Hash;

    //
    // Hash the incoming key into the 64-bit representation, which is two
    // 32-bit ULONGs in disguise, each one driven by a separate seed value.
    //

    if (FAILED(Table->Vtbl->Hash(Table, Key, &Hash.QuadPart))) {
        goto Error;
    }

    //
    // Mask each hash value such that it falls within the confines of the
    // number of vertices.  That is, make sure the value is between 0 and
    // Table->NumberOfVertices-1.
    //

    if (FAILED(Table->Vtbl->MaskHash(Table, Hash.LowPart, &MaskedLow))) {
        goto Error;
    }

    if (FAILED(Table->Vtbl->MaskHash(Table, Hash.HighPart, &MaskedHigh))) {
        goto Error;
    }

    //
    // Obtain the corresponding vertex values for the masked high and low hash
    // values.  These are derived from the "assigned" array that we construct
    // during the creation routine's assignment step (GraphAssign()).
    //

    Assigned = Table->Data;

    Vertex1 = Assigned[MaskedLow];
    Vertex2 = Assigned[MaskedHigh];

    //
    // Combine the two values, then perform the index masking operation, such
    // that our final index into the array falls within the confines of the
    // number of edges, or keys, in the table.  That is, make sure the index
    // value is between 0 and Table->Keys->NumberOfElements-1.
    //

    Combined = (ULONGLONG)Vertex1 + (ULONGLONG)Vertex2;

    if (FAILED(Table->Vtbl->MaskIndex(Table, Combined, &Masked))) {
        goto Error;
    }

    //
    // Update the caller's pointer and return success.  The resulting index
    // value represents the array offset index for this given key in the
    // underlying table, and is guaranteed to be unique amongst the original
    // keys in the input set.
    //

    *Index = Masked;
    return S_OK;

Error:

    //
    // Clear the caller's pointer and return failure.  We should only hit this
    // point if the caller supplies a key that both: a) wasn't in the original
    // input set, and b) happens to result in a hash value where both the high
    // part and low part are identical, which is rare, but not impossible.
    //

    *Index = 0;
    return E_FAIL;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableFastIndexImplChm01Crc32RotateHashAndMask(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    PULONG Index
    )
/*++

Routine Description:

    Looks up given key in a perfect hash table and returns its index.  This
    is a fast version of the normal Index() routine that inlines the Crc32Rotate
    hash function and AND masking.

    N.B. If Key did not appear in the original set the hash table was created
         from, the behavior of this routine is undefined.  (In practice, the
         key will hash to either an existing key's location or an empty slot,
         so there is potential for returning a non-unique index.)

Arguments:

    Table - Supplies a pointer to the table for which the key lookup is to be
        performed.

    Key - Supplies the key to look up.

    Index - Receives the index associated with this key.  The index will be
        between 0 and Table->HashSize-1, and can be safely used to offset
        directly into an appropriately sized array (e.g. Table->Values[]).

Return Value:

    S_OK on success, E_FAIL if the underlying hash function returned a failure.
    This will happen if the two hash values for a key happen to be identical.
    It shouldn't happen once a perfect graph has been created (i.e. it only
    happens when attempting to solve the graph).  The Index parameter will
    be cleared in the case of E_FAIL.

--*/
{
    ULONG A;
    ULONG B;
    ULONG C;
    ULONG D;
    ULONG Seed1;
    ULONG Seed2;
    ULONG Seed3;
    ULONG Input;
    PULONG Seeds;
    ULONG Masked;
    ULONG Vertex1;
    ULONG Vertex2;
    PULONG Assigned;
    ULONG MaskedLow;
    ULONG MaskedHigh;
    ULONGLONG Combined;

    //IACA_VC_START();

    //
    // Initialize aliases.
    //

    Seeds = &Table->TableInfoOnDisk->FirstSeed;
    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3 = Seeds[2];
    Input = Key;
    Assigned = Table->Data;

    //
    // Calculate the individual hash parts.
    //

    A = _mm_crc32_u32(Seed1, Input);
    B = _mm_crc32_u32(Seed2, _rotl(Input, 15));
    C = Seed3 ^ Input;
    D = _mm_crc32_u32(B, C);

    //IACA_VC_END();

    Vertex1 = A;
    Vertex2 = D;

    if (Vertex1 == Vertex2) {
        goto Error;
    }

    //
    // Mask each hash value such that it falls within the confines of the
    // number of vertices.
    //

    MaskedLow = Vertex1 & Table->HashMask;
    MaskedHigh = Vertex2 & Table->HashMask;

    //
    // Obtain the corresponding vertex values for the masked high and low hash
    // values.  These are derived from the "assigned" array that we construct
    // during the creation routine's assignment step (GraphAssign()).
    //

    Vertex1 = Assigned[MaskedLow];
    Vertex2 = Assigned[MaskedHigh];

    //
    // Combine the two values, then perform the index masking operation, such
    // that our final index into the array falls within the confines of the
    // number of edges, or keys, in the table.  That is, make sure the index
    // value is between 0 and Table->Keys->NumberOfElements-1.
    //

    Combined = (ULONGLONG)Vertex1 + (ULONGLONG)Vertex2;

    Masked = Combined & Table->IndexMask;

    //
    // Update the caller's pointer and return success.  The resulting index
    // value represents the array offset index for this given key in the
    // underlying table, and is guaranteed to be unique amongst the original
    // keys in the input set.
    //

    *Index = Masked;

    //IACA_VC_END();

    return S_OK;

Error:

    //
    // Clear the caller's pointer and return failure.  We should only hit this
    // point if the caller supplies a key that both: a) wasn't in the original
    // input set, and b) happens to result in a hash value where both the high
    // part and low part are identical, which is rare, but not impossible.
    //

    *Index = 0;
    return E_FAIL;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableFastIndexImplChm01JenkinsHashAndMask(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    PULONG Index
    )
/*++

Routine Description:

    Looks up given key in a perfect hash table and returns its index.  This
    is a fast version of the normal Index() routine that inlines the Jenkins
    hash function and AND masking.

    N.B. If Key did not appear in the original set the hash table was created
         from, the behavior of this routine is undefined.  (In practice, the
         key will hash to either an existing key's location or an empty slot,
         so there is potential for returning a non-unique index.)

Arguments:

    Table - Supplies a pointer to the table for which the key lookup is to be
        performed.

    Key - Supplies the key to look up.

    Index - Receives the index associated with this key.  The index will be
        between 0 and Table->HashSize-1, and can be safely used to offset
        directly into an appropriately sized array (e.g. Table->Values[]).

Return Value:

    S_OK on success, E_FAIL if the underlying hash function returned a failure.
    This will happen if the two hash values for a key happen to be identical.
    It shouldn't happen once a perfect graph has been created (i.e. it only
    happens when attempting to solve the graph).  The Index parameter will
    be cleared in the case of E_FAIL.

--*/
{
    ULONG A;
    ULONG B;
    ULONG C;
    ULONG D;
    ULONG E;
    ULONG F;
    PBYTE Byte;
    ULONG Seed1;
    ULONG Seed2;
    ULONG Input;
    PULONG Seeds;
    ULONG Masked;
    ULONG Vertex1;
    ULONG Vertex2;
    PULONG Assigned;
    ULONG MaskedLow;
    ULONG MaskedHigh;
    ULONGLONG Combined;

    //
    // Initialize aliases.
    //

    //IACA_VC_START();

    Seeds = &Table->TableInfoOnDisk->FirstSeed;
    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Input = Key;

    Byte = (PBYTE)&Input;

    //
    // Generate the first hash.
    //

    A = B = 0x9e3779b9;
    C = Seed1;

    A += (((ULONG)Byte[3]) << 24);
    A += (((ULONG)Byte[2]) << 16);
    A += (((ULONG)Byte[1]) <<  8);
    A += ((ULONG)Byte[0]);

    A -= B; A -= C; A ^= (C >> 13);
    B -= C; B -= A; B ^= (A <<  8);
    C -= A; C -= B; C ^= (B >> 13);
    A -= B; A -= C; A ^= (C >> 12);
    B -= C; B -= A; B ^= (A << 16);
    C -= A; C -= B; C ^= (B >>  5);
    A -= B; A -= C; A ^= (C >>  3);
    B -= C; B -= A; B ^= (A << 10);
    C -= A; C -= B; C ^= (B >> 15);

    Vertex1 = C;

    //
    // Generate the second hash.
    //

    D = E = 0x9e3779b9;
    F = Seed2;

    D += (((ULONG)Byte[3]) << 24);
    D += (((ULONG)Byte[2]) << 16);
    D += (((ULONG)Byte[1]) <<  8);
    D += ((ULONG)Byte[0]);

    D -= E; D -= F; D ^= (F >> 13);
    E -= F; E -= D; E ^= (D <<  8);
    F -= D; F -= E; F ^= (E >> 13);
    D -= E; D -= F; D ^= (F >> 12);
    E -= F; E -= D; E ^= (D << 16);
    F -= D; F -= E; F ^= (E >>  5);
    D -= E; D -= F; D ^= (F >>  3);
    E -= F; E -= D; E ^= (D << 10);
    F -= D; F -= E; F ^= (E >> 15);

    //IACA_VC_END();

    Vertex2 = F;

    if (Vertex1 == Vertex2) {
        goto Error;
    }

    //
    // Mask each hash value such that it falls within the confines of the
    // number of vertices.
    //

    MaskedLow = Vertex1 & Table->HashMask;
    MaskedHigh = Vertex2 & Table->HashMask;

    //
    // Obtain the corresponding vertex values for the masked high and low hash
    // values.  These are derived from the "assigned" array that we construct
    // during the creation routine's assignment step (GraphAssign()).
    //

    Assigned = Table->Data;

    Vertex1 = Assigned[MaskedLow];
    Vertex2 = Assigned[MaskedHigh];

    //
    // Combine the two values, then perform the index masking operation, such
    // that our final index into the array falls within the confines of the
    // number of edges, or keys, in the table.  That is, make sure the index
    // value is between 0 and Table->Keys->NumberOfElements-1.
    //

    Combined = (ULONGLONG)Vertex1 + (ULONGLONG)Vertex2;

    Masked = Combined & Table->IndexMask;

    //
    // Update the caller's pointer and return success.  The resulting index
    // value represents the array offset index for this given key in the
    // underlying table, and is guaranteed to be unique amongst the original
    // keys in the input set.
    //

    *Index = Masked;

    //IACA_VC_END();

    return S_OK;

Error:

    //
    // Clear the caller's pointer and return failure.  We should only hit this
    // point if the caller supplies a key that both: a) wasn't in the original
    // input set, and b) happens to result in a hash value where both the high
    // part and low part are identical, which is rare, but not impossible.
    //

    *Index = 0;
    return E_FAIL;
}

PERFECT_HASH_TABLE_INDEX PerfectHashTableFastIndexImplChm01JenkinsHashModMask;

_Use_decl_annotations_
HRESULT
PerfectHashTableFastIndexImplChm01JenkinsHashModMask(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    PULONG Index
    )
/*++

Routine Description:

    Looks up given key in a perfect hash table and returns its index.

    N.B. This version is based off the Jenkins hash function and modulus
         masking.  As we don't use modulus masking at all, it's not intended
         to be used in reality.  However, it's useful to feed to IACA to see
         the impact of the modulus operation.

Arguments:

    Table - Supplies a pointer to the table for which the key lookup is to be
        performed.

    Key - Supplies the key to look up.

    Index - Receives the index associated with this key.  The index will be
        between 0 and Table->HashSize-1, and can be safely used to offset
        directly into an appropriately sized array (e.g. Table->Values[]).

Return Value:

    S_OK on success, E_FAIL if the underlying hash function returned a failure.
    This will happen if the two hash values for a key happen to be identical.
    It shouldn't happen once a perfect graph has been created (i.e. it only
    happens when attempting to solve the graph).  The Index parameter will
    be cleared in the case of E_FAIL.

--*/
{
    ULONG A;
    ULONG B;
    ULONG C;
    ULONG D;
    ULONG E;
    ULONG F;
    PBYTE Byte;
    ULONG Seed1;
    ULONG Seed2;
    ULONG Input;
    PULONG Seeds;
    ULONG Masked;
    ULONG Vertex1;
    ULONG Vertex2;
    PULONG Assigned;
    ULONG MaskedLow;
    ULONG MaskedHigh;
    ULONGLONG Combined;

    //
    // Initialize aliases.
    //

    //IACA_VC_START();

    Seeds = &Table->TableInfoOnDisk->FirstSeed;
    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Input = Key;

    Byte = (PBYTE)&Input;

    //
    // Generate the first hash.
    //

    A = B = 0x9e3779b9;
    C = Seed1;

    A += (((ULONG)Byte[3]) << 24);
    A += (((ULONG)Byte[2]) << 16);
    A += (((ULONG)Byte[1]) <<  8);
    A += ((ULONG)Byte[0]);

    A -= B; A -= C; A ^= (C >> 13);
    B -= C; B -= A; B ^= (A <<  8);
    C -= A; C -= B; C ^= (B >> 13);
    A -= B; A -= C; A ^= (C >> 12);
    B -= C; B -= A; B ^= (A << 16);
    C -= A; C -= B; C ^= (B >>  5);
    A -= B; A -= C; A ^= (C >>  3);
    B -= C; B -= A; B ^= (A << 10);
    C -= A; C -= B; C ^= (B >> 15);

    Vertex1 = C;

    //
    // Generate the second hash.
    //

    D = E = 0x9e3779b9;
    F = Seed2;

    D += (((ULONG)Byte[3]) << 24);
    D += (((ULONG)Byte[2]) << 16);
    D += (((ULONG)Byte[1]) <<  8);
    D += ((ULONG)Byte[0]);

    D -= E; D -= F; D ^= (F >> 13);
    E -= F; E -= D; E ^= (D <<  8);
    F -= D; F -= E; F ^= (E >> 13);
    D -= E; D -= F; D ^= (F >> 12);
    E -= F; E -= D; E ^= (D << 16);
    F -= D; F -= E; F ^= (E >>  5);
    D -= E; D -= F; D ^= (F >>  3);
    E -= F; E -= D; E ^= (D << 10);
    F -= D; F -= E; F ^= (E >> 15);

    //IACA_VC_END();

    Vertex2 = F;

    if (Vertex1 == Vertex2) {
        goto Error;
    }

    //
    // Mask each hash value such that it falls within the confines of the
    // number of vertices.
    //

    MaskedLow = Vertex1 % Table->HashModulus;
    MaskedHigh = Vertex2 % Table->HashModulus;

    //
    // Obtain the corresponding vertex values for the masked high and low hash
    // values.  These are derived from the "assigned" array that we construct
    // during the creation routine's assignment step (GraphAssign()).
    //

    Assigned = Table->Data;

    Vertex1 = Assigned[MaskedLow];
    Vertex2 = Assigned[MaskedHigh];

    //
    // Combine the two values, then perform the index masking operation, such
    // that our final index into the array falls within the confines of the
    // number of edges, or keys, in the table.  That is, make sure the index
    // value is between 0 and Table->Keys->NumberOfElements-1.
    //

    Combined = (ULONGLONG)Vertex1 + (ULONGLONG)Vertex2;

    Masked = Combined % Table->IndexModulus;

    //
    // Update the caller's pointer and return success.  The resulting index
    // value represents the array offset index for this given key in the
    // underlying table, and is guaranteed to be unique amongst the original
    // keys in the input set.
    //

    *Index = Masked;

    //IACA_VC_END();

    return S_OK;

Error:

    //
    // Clear the caller's pointer and return failure.  We should only hit this
    // point if the caller supplies a key that both: a) wasn't in the original
    // input set, and b) happens to result in a hash value where both the high
    // part and low part are identical, which is rare, but not impossible.
    //

    *Index = 0;
    return E_FAIL;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
