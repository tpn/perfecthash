/*++

Copyright (c) 2023 Trent Nelson <trent@trent.me>

Module Name:

    Chm01Shared.c

Abstract:

    Logic shared between the PH_WINDOWS and PH_COMPAT Chm01 implementations.

--*/

#include "stdafx.h"
#include "Chm01.h"
#include "Chm01Private.h"

PREPARE_GRAPH_INFO PrepareGraphInfoChm01;

_Use_decl_annotations_
HRESULT
PrepareGraphInfoChm01(
    PPERFECT_HASH_TABLE Table,
    PGRAPH_INFO Info,
    PGRAPH_INFO PrevInfo
    )
/*++

Routine Description:

    Prepares the GRAPH_INFO structure for a given table.

    N.B. This routine was created by lifting all of the logic from the body of
         the CreatePerfectHashTableImplChm01() routine, which is where it was
         originally based.  It could do with an overhaul; it uses way too many
         local variables unnecessarily, for example.

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
    PRTL Rtl;
    HRESULT Result = S_OK;
    BYTE AssignedShift;
    ULONG GraphImpl;
    ULONG NumberOfKeys;
    BOOLEAN UseAssigned16;
    USHORT NumberOfBitmaps;
    PGRAPH_DIMENSIONS Dim;
    SYSTEM_INFO SystemInfo;
    PCSTRING TypeNames;
    PGRAPH_INFO_ON_DISK GraphInfoOnDisk;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;
    ULONG NumberOfEdgeMaskBits;
    ULONG NumberOfVertexMaskBits;
    ULONGLONG NextSizeInBytes;
    ULONGLONG FirstSizeInBytes;
    ULONGLONG OrderSizeInBytes;
    ULONGLONG EdgesSizeInBytes;
    ULONGLONG Vertices3SizeInBytes;
    ULONGLONG ValuesSizeInBytes;
    ULONGLONG AssignedSizeInBytes;
    ULONGLONG VertexPairsSizeInBytes;
    PPERFECT_HASH_CONTEXT Context;
    ULARGE_INTEGER AllocSize;
    ULARGE_INTEGER NumberOfEdges;
    ULARGE_INTEGER NumberOfVertices;
    ULARGE_INTEGER TotalNumberOfEdges;
    ULARGE_INTEGER DeletedEdgesBitmapBufferSizeInBytes;
    ULARGE_INTEGER VisitedVerticesBitmapBufferSizeInBytes;
    ULARGE_INTEGER AssignedBitmapBufferSizeInBytes;
    ULARGE_INTEGER IndexBitmapBufferSizeInBytes;
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags;
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId;
    PTRAILING_ZEROS_32 TrailingZeros32;
    PTRAILING_ZEROS_64 TrailingZeros64;
    PPOPULATION_COUNT_32 PopulationCount32;
    PROUND_UP_POWER_OF_TWO_32 RoundUpPowerOfTwo32;
    PROUND_UP_NEXT_POWER_OF_TWO_32 RoundUpNextPowerOfTwo32;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Info)) {
        return E_POINTER;
    }

    //
    // Initialize aliases.
    //

    Rtl = Table->Rtl;
    Context = Table->Context;
    GraphImpl = Table->GraphImpl;
    MaskFunctionId = Table->MaskFunctionId;
    GraphInfoOnDisk = Context->GraphInfoOnDisk;
    TableInfoOnDisk = &GraphInfoOnDisk->TableInfoOnDisk;
    TypeNames = Table->CTypeNames;
    TrailingZeros32 = Rtl->TrailingZeros32;
    TrailingZeros64 = Rtl->TrailingZeros64;
    PopulationCount32 = Rtl->PopulationCount32;
    RoundUpPowerOfTwo32 = Rtl->RoundUpPowerOfTwo32;
    RoundUpNextPowerOfTwo32 = Rtl->RoundUpNextPowerOfTwo32;
    TableCreateFlags.AsULongLong = Table->TableCreateFlags.AsULongLong;

    //
    // If a previous Info struct pointer has been passed, copy the current
    // Info contents into it.
    //

    if (ARGUMENT_PRESENT(PrevInfo)) {
        CopyInline(PrevInfo, Info, sizeof(*PrevInfo));
    }

    //
    // Clear our Info struct and wire up the PrevInfo pointer (which may be
    // NULL).
    //

    ZeroStructPointerInline(Info);
    Info->PrevInfo = PrevInfo;

    //
    // Ensure the number of keys are under MAX_ULONG, then take a local copy.
    //

    if (Table->Keys->NumberOfKeys.HighPart) {
        return PH_E_TOO_MANY_KEYS;
    }

    NumberOfKeys = Table->Keys->NumberOfKeys.LowPart;

    //
    // The number of edges in our graph is equal to the number of keys in the
    // input data set if modulus masking is in use.  It will be rounded up to
    // a power of 2 otherwise.
    //

    NumberOfEdges.QuadPart = NumberOfKeys;

    //
    // Make sure we have at least 8 edges; this ensures the assigned array
    // will consume at least one cache line, which is required for our memory
    // coverage routine (see Graph.c) to work correctly, as it operates on
    // cache line sized strides.
    //

    if (NumberOfEdges.QuadPart < 8) {
        NumberOfEdges.QuadPart = 8;
    }

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
            // rounded up to a power of 2.
            //

            NumberOfVertices.QuadPart = (
                RoundUpPowerOfTwo32(NumberOfVertices.LowPart)
            );

            //
            // If we're clamping number of edges, use number of keys rounded up
            // to a power of two.  Otherwise, use the number of vertices shifted
            // right by one (divided by two).
            //

            if (Table->TableCreateFlags.ClampNumberOfEdges) {
                NumberOfEdges.QuadPart = RoundUpPowerOfTwo32(NumberOfKeys);
            } else {
                NumberOfEdges.QuadPart = NumberOfVertices.QuadPart >> 1ULL;
            }

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
        // Update: attempt to use primes for modulus masking.  Still doesn't
        //         work.
        //

        if (IsModulusMasking(MaskFunctionId)) {
            SHORT PrimeIndex;
            ULONGLONG Value;
            ULONGLONG Prime;

            //
            // Find a prime greater than or equal to the number of edges.
            //

            Value = NumberOfEdges.QuadPart;

            PrimeIndex = FindIndexForFirstPrimeGreaterThanOrEqual(Value);
            if (PrimeIndex == -1) {
                PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
            }

            Prime = Primes[PrimeIndex];
            NumberOfEdges.QuadPart = Prime;

            //
            // Double the number of edges, then find a prime greater than or
            // equal to this new value.
            //

            Value <<= 1;

            PrimeIndex = FindIndexForFirstPrimeGreaterThanOrEqual(Value);
            if (PrimeIndex == -1) {
                PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
            }

            Prime = Primes[PrimeIndex];
            NumberOfVertices.QuadPart = Prime;

        } else {

            //
            // Round up the edges to a power of 2.
            //

            NumberOfEdges.QuadPart = RoundUpPowerOfTwo32(NumberOfEdges.LowPart);

            //
            // Make sure we haven't overflowed.
            //

            if (NumberOfEdges.HighPart) {
                Result = PH_E_TOO_MANY_EDGES;
                goto Error;
            }

            //
            // For the number of vertices, round the number of edges up to the
            // next power of 2.
            //

            NumberOfVertices.QuadPart = (
                RoundUpNextPowerOfTwo32(NumberOfEdges.LowPart)
            );

        }
    }

    //
    // Another sanity check we haven't exceeded MAX_ULONG.
    //

    if (NumberOfVertices.HighPart) {
        Result = PH_E_TOO_MANY_VERTICES;
        goto Error;
    }

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

    if (TotalNumberOfEdges.HighPart) {
        Result = PH_E_TOO_MANY_TOTAL_EDGES;
        goto Error;
    }

    //
    // Make sure vertices > edges.
    //

    if (NumberOfVertices.QuadPart <= NumberOfEdges.QuadPart) {
        Result = PH_E_NUM_VERTICES_LESS_THAN_OR_EQUAL_NUM_EDGES;
        goto Error;
    }

    //
    // Invariant check: sure vertices shifted right once == edges if applicable.
    //

    if ((!IsModulusMasking(MaskFunctionId)) &&
        (Table->TableCreateFlags.ClampNumberOfEdges == FALSE)) {

        if ((NumberOfVertices.QuadPart >> 1) != NumberOfEdges.QuadPart) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(PrepareGraphInfoChm01_NumEdgesNotNumVerticesDiv2, Result);
            goto Error;
        }
    }

    //
    // If our graph impl is v3, our vertex count - 1 is less than or equal to
    // MAX_USHORT (i.e. 65535), and the user hasn't supplied the flag to the
    // contrary, use the 16-bit hash/assigned implementation.
    //

    if ((GraphImpl == 3) &&
        ((NumberOfVertices.LowPart-1) <= 0x0000ffff) &&
        (TableCreateFlags.DoNotTryUseHash16Impl == FALSE)) {

        UseAssigned16 = TRUE;
        AssignedShift = ASSIGNED16_SHIFT;
        Table->State.UsingAssigned16 = TRUE;

        //
        // Overwrite the vtbl Index routines accordingly.
        //

        Table->Vtbl->Index = PerfectHashTableIndex16ImplChm01;
        Table->Vtbl->FastIndex = NULL;
        Table->Vtbl->SlowIndex = NULL;

    } else {
        UseAssigned16 = FALSE;
        AssignedShift = ASSIGNED_SHIFT;
        Table->State.UsingAssigned16 = FALSE;
    }

    //
    // Calculate the size required for our bitmap buffers.
    //

    DeletedEdgesBitmapBufferSizeInBytes.QuadPart = ALIGN_UP_POINTER(
        ALIGN_UP(TotalNumberOfEdges.QuadPart, 8) >> 3
    );

    if (DeletedEdgesBitmapBufferSizeInBytes.HighPart) {
        Result = PH_E_TOO_MANY_BITS_FOR_BITMAP;
        PH_ERROR(PrepareGraphInfoChm01_DeletedEdgesBitmap, Result);
        goto Error;
    }

    VisitedVerticesBitmapBufferSizeInBytes.QuadPart = ALIGN_UP_POINTER(
        ALIGN_UP(NumberOfVertices.QuadPart, 8) >> 3
    );

    if (VisitedVerticesBitmapBufferSizeInBytes.HighPart) {
        Result = PH_E_TOO_MANY_BITS_FOR_BITMAP;
        PH_ERROR(PrepareGraphInfoChm01_VisitedVerticesBitmap, Result);
        goto Error;
    }

    AssignedBitmapBufferSizeInBytes.QuadPart = ALIGN_UP_POINTER(
        ALIGN_UP(NumberOfVertices.QuadPart, 8) >> 3
    );

    if (AssignedBitmapBufferSizeInBytes.HighPart) {
        Result = PH_E_TOO_MANY_BITS_FOR_BITMAP;
        PH_ERROR(PrepareGraphInfoChm01_AssignedBitmap, Result);
        goto Error;
    }

    IndexBitmapBufferSizeInBytes.QuadPart = ALIGN_UP_POINTER(
        ALIGN_UP(NumberOfVertices.QuadPart, 8) >> 3
    );

    if (IndexBitmapBufferSizeInBytes.HighPart) {
        Result = PH_E_TOO_MANY_BITS_FOR_BITMAP;
        PH_ERROR(PrepareGraphInfoChm01_IndexBitmap, Result);
        goto Error;
    }

    //
    // Calculate the sizes required for each of the arrays.
    //

    if (GraphImpl == 1 || GraphImpl == 2) {

        EdgesSizeInBytes = ALIGN_UP_YMMWORD(
            RTL_ELEMENT_SIZE(GRAPH, Edges) * TotalNumberOfEdges.QuadPart
        );

        NextSizeInBytes = ALIGN_UP_YMMWORD(
            RTL_ELEMENT_SIZE(GRAPH, Next) * TotalNumberOfEdges.QuadPart
        );

        FirstSizeInBytes = ALIGN_UP_YMMWORD(
            RTL_ELEMENT_SIZE(GRAPH, First) * NumberOfVertices.QuadPart
        );

        Vertices3SizeInBytes = 0;

        if (TableCreateFlags.HashAllKeysFirst == FALSE) {
            VertexPairsSizeInBytes = 0;
        } else {
            VertexPairsSizeInBytes = ALIGN_UP_ZMMWORD(
                RTL_ELEMENT_SIZE(GRAPH, VertexPairs) * (ULONGLONG)NumberOfKeys
            );
        }

    } else {

        ASSERT(GraphImpl == 3);

        EdgesSizeInBytes = 0;
        NextSizeInBytes = 0;
        FirstSizeInBytes = 0;

        if (!UseAssigned16) {

            VertexPairsSizeInBytes = ALIGN_UP_YMMWORD(
                RTL_ELEMENT_SIZE(GRAPH, VertexPairs) *
                (ULONGLONG)NumberOfKeys
            );

            Vertices3SizeInBytes = ALIGN_UP_YMMWORD(
                RTL_ELEMENT_SIZE(GRAPH, Vertices3) *
                NumberOfVertices.QuadPart
            );

        } else {

            VertexPairsSizeInBytes = ALIGN_UP_YMMWORD(
                RTL_ELEMENT_SIZE(GRAPH, Vertex16Pairs) *
                (ULONGLONG)NumberOfKeys
            );

            Vertices3SizeInBytes = ALIGN_UP_YMMWORD(
                RTL_ELEMENT_SIZE(GRAPH, Vertices163) *
                NumberOfVertices.QuadPart
            );

        }

        DeletedEdgesBitmapBufferSizeInBytes.QuadPart = 0;
    }

    if (!UseAssigned16) {

        OrderSizeInBytes = ALIGN_UP_YMMWORD(
            RTL_ELEMENT_SIZE(GRAPH, Order) * NumberOfEdges.QuadPart
        );

        AssignedSizeInBytes = ALIGN_UP_YMMWORD(
            RTL_ELEMENT_SIZE(GRAPH, Assigned) * NumberOfVertices.QuadPart
        );

    } else {

        OrderSizeInBytes = ALIGN_UP_YMMWORD(
            RTL_ELEMENT_SIZE(GRAPH, Order16) * NumberOfEdges.QuadPart
        );

        AssignedSizeInBytes = ALIGN_UP_YMMWORD(
            RTL_ELEMENT_SIZE(GRAPH, Assigned16) * NumberOfVertices.QuadPart
        );

    }

    //
    // Calculate the size required for the values array.  This is used as part
    // of verification, where we essentially do Insert(Key, Key) in combination
    // with bitmap tracking of assigned indices, which allows us to detect if
    // there are any colliding indices, and if so, what was the previous key
    // that mapped to the same index.
    //

    ValuesSizeInBytes = ALIGN_UP_YMMWORD(
        RTL_ELEMENT_SIZE(GRAPH, Values) * NumberOfVertices.QuadPart
    );

    //
    // Calculate the number of cache lines, pages and large pages covered by
    // the assigned array, plus the respective buffer sizes for each array in
    // the ASSIGNED_MEMORY_COVERAGE structure that captures counts.
    //
    // N.B. We don't use AssignedSizeInBytes here as it is subject to being
    //      aligned up to a YMMWORD boundary, and thus, may not represent the
    //      exact number of cache lines used strictly for vertices.
    //

    //
    // Element counts.
    //

    Info->AssignedArrayNumberOfPages = (ULONG)(
        BYTES_TO_PAGES(NumberOfVertices.QuadPart << AssignedShift)
    );

    Info->AssignedArrayNumberOfLargePages = (ULONG)(
        BYTES_TO_LARGE_PAGES(NumberOfVertices.QuadPart << AssignedShift)
    );

    Info->AssignedArrayNumberOfCacheLines = (ULONG)(
        BYTES_TO_CACHE_LINES(NumberOfVertices.QuadPart << AssignedShift)
    );

    //
    // Array sizes for the number of assigned per page, large page and cache
    // line.
    //

    if (!UseAssigned16) {

        Info->NumberOfAssignedPerPageSizeInBytes = (
            Info->AssignedArrayNumberOfPages *
            RTL_ELEMENT_SIZE(ASSIGNED_MEMORY_COVERAGE,
                             NumberOfAssignedPerPage)
        );

        Info->NumberOfAssignedPerLargePageSizeInBytes = (
            Info->AssignedArrayNumberOfLargePages *
            RTL_ELEMENT_SIZE(ASSIGNED_MEMORY_COVERAGE,
                             NumberOfAssignedPerLargePage)
        );

        Info->NumberOfAssignedPerCacheLineSizeInBytes = (
            Info->AssignedArrayNumberOfCacheLines *
            RTL_ELEMENT_SIZE(ASSIGNED_MEMORY_COVERAGE,
                             NumberOfAssignedPerCacheLine)
        );

    } else {

        Info->NumberOfAssignedPerPageSizeInBytes = (
            Info->AssignedArrayNumberOfPages *
            RTL_ELEMENT_SIZE(ASSIGNED16_MEMORY_COVERAGE,
                             NumberOfAssignedPerPage)
        );

        Info->NumberOfAssignedPerLargePageSizeInBytes = (
            Info->AssignedArrayNumberOfLargePages *
            RTL_ELEMENT_SIZE(ASSIGNED16_MEMORY_COVERAGE,
                             NumberOfAssignedPerLargePage)
        );

        Info->NumberOfAssignedPerCacheLineSizeInBytes = (
            Info->AssignedArrayNumberOfCacheLines *
            RTL_ELEMENT_SIZE(ASSIGNED16_MEMORY_COVERAGE,
                             NumberOfAssignedPerCacheLine)
        );

    }

    //
    // Calculate the total size required for the underlying arrays, bitmap
    // buffers and assigned array counts, rounded up to the nearest page size.
    //

    AllocSize.QuadPart = ROUND_TO_PAGES(
        EdgesSizeInBytes +
        NextSizeInBytes +
        OrderSizeInBytes +
        FirstSizeInBytes +
        Vertices3SizeInBytes +
        AssignedSizeInBytes +
        VertexPairsSizeInBytes +
        ValuesSizeInBytes +

        Info->NumberOfAssignedPerPageSizeInBytes +
        Info->NumberOfAssignedPerLargePageSizeInBytes +
        Info->NumberOfAssignedPerCacheLineSizeInBytes +

        //
        // Begin bitmaps.
        //

        DeletedEdgesBitmapBufferSizeInBytes.QuadPart +
        VisitedVerticesBitmapBufferSizeInBytes.QuadPart +
        AssignedBitmapBufferSizeInBytes.QuadPart +
        IndexBitmapBufferSizeInBytes.QuadPart +

        //
        // End bitmaps.
        //

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
    // Initialize the GRAPH_INFO structure with all the sizes captured earlier.
    // (We zero it first just to ensure any of the padding fields are cleared.)
    //

    Info->Context = Context;
    Info->AllocSize = AllocSize.QuadPart;
    Info->NumberOfBitmaps = NumberOfBitmaps;
    Info->SizeOfGraphStruct = sizeof(GRAPH);
    Info->EdgesSizeInBytes = EdgesSizeInBytes;
    Info->NextSizeInBytes = NextSizeInBytes;
    Info->OrderSizeInBytes = OrderSizeInBytes;
    Info->FirstSizeInBytes = FirstSizeInBytes;
    Info->Vertices3SizeInBytes = Vertices3SizeInBytes;
    Info->AssignedSizeInBytes = AssignedSizeInBytes;
    Info->VertexPairsSizeInBytes = VertexPairsSizeInBytes;
    Info->ValuesSizeInBytes = ValuesSizeInBytes;

    Info->DeletedEdgesBitmapBufferSizeInBytes = (
        DeletedEdgesBitmapBufferSizeInBytes.QuadPart
    );

    Info->VisitedVerticesBitmapBufferSizeInBytes = (
        VisitedVerticesBitmapBufferSizeInBytes.QuadPart
    );

    Info->AssignedBitmapBufferSizeInBytes = (
        AssignedBitmapBufferSizeInBytes.QuadPart
    );

    Info->IndexBitmapBufferSizeInBytes = (
        IndexBitmapBufferSizeInBytes.QuadPart
    );

    //
    // Capture the system allocation granularity.  This is used to align the
    // backing memory maps used for the table array.
    //

    GetSystemInfo(&SystemInfo);
    Info->AllocationGranularity = SystemInfo.dwAllocationGranularity;

    //
    // Copy the dimensions over.
    //

    Dim = &Info->Dimensions;
    Dim->NumberOfEdges = NumberOfEdges.LowPart;
    Dim->TotalNumberOfEdges = TotalNumberOfEdges.LowPart;
    Dim->NumberOfVertices = NumberOfVertices.LowPart;

    Dim->NumberOfEdgesPowerOf2Exponent = (BYTE)(
        TrailingZeros64(RoundUpPowerOfTwo32(NumberOfEdges.LowPart))
    );

    Dim->NumberOfEdgesNextPowerOf2Exponent = (BYTE)(
        TrailingZeros64(RoundUpNextPowerOfTwo32(NumberOfEdges.LowPart))
    );

    Dim->NumberOfVerticesPowerOf2Exponent = (BYTE)(
        TrailingZeros64(RoundUpPowerOfTwo32(NumberOfVertices.LowPart))
    );

    Dim->NumberOfVerticesNextPowerOf2Exponent = (BYTE)(
        TrailingZeros64(RoundUpNextPowerOfTwo32(NumberOfVertices.LowPart))
    );

    //
    // If non-modulus masking is active, initialize the edge and vertex masks
    // and underlying table data type.
    //

    if (!IsModulusMasking(MaskFunctionId)) {

        ULONG_PTR EdgeValue;

        Info->EdgeMask = NumberOfEdges.LowPart - 1;
        Info->VertexMask = NumberOfVertices.LowPart - 1;

        NumberOfEdgeMaskBits = PopulationCount32(Info->EdgeMask);
        NumberOfVertexMaskBits = PopulationCount32(Info->VertexMask);

        //
        // Sanity check our masks are correct: their popcnts should match the
        // exponent value identified above whilst filling out the dimensions
        // structure.
        //

        if (NumberOfEdgeMaskBits != Dim->NumberOfEdgesPowerOf2Exponent) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(PrepareGraphInfoChm01_EdgeMaskPopcountMismatch, Result);
            goto Error;
        }

        if (NumberOfVertexMaskBits != Dim->NumberOfVerticesPowerOf2Exponent) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(PrepareGraphInfoChm01_VertexMaskPopcountMismatch, Result);
            goto Error;
        }

        //
        // Use the number of bits set in the edge mask to derive an appropriate
        // containing data type (i.e. USHORT if there are less than 65k keys,
        // ULONG otherwise, etc).  The Table->TableDataArrayType field is used
        // as the type for the TableData/Assigned array; if we have less than
        // 65k keys, we know that the two values derived from this table will
        // never exceed 65536, so the array can be USHORT, saving 2 bytes per
        // entry vs keeping it at ULONG.  The 256->65536 USHORT sweet spot is
        // common enough (based on the typical number of keys we're targeting)
        // to warrant this logic.
        //

        EdgeValue = (ULONG_PTR)1 << NumberOfEdgeMaskBits;
        Result = GetContainingType(Rtl, EdgeValue, &Table->TableDataArrayType);
        if (FAILED(Result)) {
            PH_ERROR(PrepareGraphInfoChm01_GetContainingType, Result);
            goto Error;
        }

        Table->TableDataArrayTypeName = &TypeNames[Table->TableDataArrayType];

        //
        // Default the table values type name to ULONG for now until we add
        // more comprehensive support for varying the type name (i.e. wire up
        // a command line parameter to it at the very least).  Ditto for keys.
        //

        Table->ValueType = LongType;
        Table->TableValuesArrayTypeName = &TypeNames[LongType];
        Table->KeysArrayTypeName = &TypeNames[LongType];

    } else {

        //
        // Default names to something sensible if modulus masking is active.
        //

        Table->ValueType = LongType;
        Table->TableDataArrayTypeName = &TypeNames[LongType];
        Table->TableValuesArrayTypeName = &TypeNames[LongType];
        Table->KeysArrayTypeName = &TypeNames[LongType];
    }

    Table->SeedTypeName = &TypeNames[LongType];
    Table->IndexTypeName = &TypeNames[LongType];
    Table->ValueTypeName = &TypeNames[Table->ValueType];
    Table->KeySizeTypeName = &TypeNames[Table->Keys->KeySizeType];
    Table->OriginalKeySizeTypeName =
        &TypeNames[Table->Keys->OriginalKeySizeType];

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
    Table->HashShift = 32 - Rtl->TrailingZeros32(Table->HashSize);
    Table->IndexShift = 32 - Rtl->TrailingZeros32(Table->IndexSize);
    Table->HashMask = (Table->HashSize - 1);
    Table->IndexMask = (Table->IndexSize - 1);
    Table->HashFold = Table->HashShift >> 3;
    Table->IndexFold = Table->IndexShift >> 3;

    //
    // Fill out the in-memory representation of the on-disk table/graph info.
    // This is a smaller subset of data needed in order to load a previously
    // solved graph as a perfect hash table.  The data will eventually be
    // written into the NTFS stream :Info.
    //

    ZeroStructPointerInline(GraphInfoOnDisk);
    TableInfoOnDisk->Magic.LowPart = TABLE_INFO_ON_DISK_MAGIC_LOWPART;
    TableInfoOnDisk->Magic.HighPart = TABLE_INFO_ON_DISK_MAGIC_HIGHPART;
    TableInfoOnDisk->SizeOfStruct = sizeof(*GraphInfoOnDisk);
    TableInfoOnDisk->Flags.AsULong = 0;
    TableInfoOnDisk->Flags.UsingAssigned16 = (UseAssigned16 != FALSE);
    TableInfoOnDisk->Concurrency = Context->MaximumConcurrency;
    TableInfoOnDisk->AlgorithmId = Context->AlgorithmId;
    TableInfoOnDisk->MaskFunctionId = Context->MaskFunctionId;
    TableInfoOnDisk->HashFunctionId = Context->HashFunctionId;
    TableInfoOnDisk->KeySizeInBytes = Table->Keys->KeySizeInBytes;
    TableInfoOnDisk->OriginalKeySizeInBytes = (
        Table->Keys->OriginalKeySizeInBytes
    );
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
    TableInfoOnDisk->TableDataArrayType = Table->TableDataArrayType;
    TableInfoOnDisk->NumberOfKeys.QuadPart = (
        Table->Keys->NumberOfKeys.QuadPart
    );
    TableInfoOnDisk->NumberOfSeeds = (
        HashRoutineNumberOfSeeds[Table->HashFunctionId]
    );
    TableInfoOnDisk->AssignedElementSizeInBytes = UseAssigned16 ? 2 : 4;

    //
    // This will change based on masking type and whether or not the caller
    // has provided a value for NumberOfTableElements.  For now, keep it as
    // the number of vertices.
    //

    TableInfoOnDisk->NumberOfTableElements.QuadPart = (
        NumberOfVertices.QuadPart
    );

    CopyInline(&GraphInfoOnDisk->Dimensions, Dim, sizeof(*Dim));

    //
    // Capture ratios.
    //

    Table->KeysToEdgesRatio = Table->Keys->KeysToEdgesRatio;

    Table->KeysToVerticesRatio = (DOUBLE)(
        ((DOUBLE)Table->Keys->NumberOfKeys.QuadPart) /
        ((DOUBLE)NumberOfVertices.QuadPart)
    );

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

    return Result;
}


PREPARE_TABLE_OUTPUT_DIRECTORY PrepareTableOutputDirectory;

_Use_decl_annotations_
HRESULT
PrepareTableOutputDirectory(
    PPERFECT_HASH_TABLE Table
    )
{
    HRESULT Result = S_OK;
    ULONG NumberOfResizeEvents;
    ULARGE_INTEGER NumberOfTableElements;
    PPERFECT_HASH_CONTEXT Context;
    PPERFECT_HASH_PATH OutputPath = NULL;
    PPERFECT_HASH_DIRECTORY OutputDir = NULL;
    PPERFECT_HASH_DIRECTORY BaseOutputDirectory;
    PCUNICODE_STRING BaseOutputDirectoryPath;
    const UNICODE_STRING EmptyString = { 0 };

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    //
    // Invariant check: if Table->OutputDirectory is set, ensure the table
    // requires renames after table resize events.
    //

    if (Table->OutputDirectory) {
        if (!TableResizeRequiresRename(Table)) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(PrepareTableOutputDirectory_NoRenameRequired, Result);
            goto Error;
        }
    }

    //
    // Initialize aliases.
    //

    Context = Table->Context;
    BaseOutputDirectory = Context->BaseOutputDirectory;
    BaseOutputDirectoryPath = &BaseOutputDirectory->Path->FullPath;
    NumberOfResizeEvents = (ULONG)Context->NumberOfTableResizeEvents;
    NumberOfTableElements.QuadPart = (
        Table->TableInfoOnDisk->NumberOfTableElements.QuadPart
    );

    //
    // Create an output directory path name.
    //

    Result = PerfectHashTableCreatePath(Table,
                                        Table->Keys->File->Path,
                                        &NumberOfResizeEvents,
                                        &NumberOfTableElements,
                                        Table->AlgorithmId,
                                        Table->HashFunctionId,
                                        Table->MaskFunctionId,
                                        BaseOutputDirectoryPath,
                                        NULL,           // NewBaseName
                                        NULL,           // AdditionalSuffix
                                        &EmptyString,   // NewExtension
                                        NULL,           // NewStreamName
                                        &OutputPath,
                                        NULL);

    if (FAILED(Result)) {
        PH_ERROR(PrepareTableOutputDirectory_CreatePath, Result);
        goto Error;
    }

    ASSERT(IsValidUnicodeString(&OutputPath->FullPath));

    //
    // Release the existing output path, if applicable.  (This will already
    // have a value if we're being called for the second or more time due to
    // a resize event.)
    //

    RELEASE(Table->OutputPath);

    Table->OutputPath = OutputPath;

    //
    // Either create a new directory instance if this is our first pass, or
    // schedule a rename if not.
    //

    if (!Table->OutputDirectory) {

        PERFECT_HASH_DIRECTORY_CREATE_FLAGS DirectoryCreateFlags = { 0 };

        //
        // No output directory has been set; this is the first attempt at
        // trying to solve the graph.  Create a new directory instance, then
        // issue a Create() call against the output path we constructed above.
        //

        Result = Table->Vtbl->CreateInstance(Table,
                                             NULL,
                                             &IID_PERFECT_HASH_DIRECTORY,
                                             (PVOID *)&OutputDir);

        if (FAILED(Result)) {
            PH_ERROR(PerfectHashDirectoryCreateInstance, Result);
            goto Error;
        }

        Result = OutputDir->Vtbl->Create(OutputDir,
                                         OutputPath,
                                         &DirectoryCreateFlags);

        if (FAILED(Result)) {
            PH_ERROR(PerfectHashDirectoryCreate, Result);
            goto Error;
        }

        //
        // Directory creation was successful.
        //

        Table->OutputDirectory = OutputDir;

    } else {

        //
        // Directory already exists; a resize event must have occurred.
        // Schedule a rename of the directory to the output path constructed
        // above.
        //

        OutputDir = Table->OutputDirectory;
        Result = OutputDir->Vtbl->ScheduleRename(OutputDir, OutputPath);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashDirectoryScheduleRename, Result);
            goto Error;
        }

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

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
