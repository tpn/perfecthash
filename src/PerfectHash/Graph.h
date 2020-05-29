/*++

Copyright (c) 2018-2020 Trent Nelson <trent@trent.me>

Module Name:

    Graph.h

Abstract:

    This is the header file for the Graph module of the perfect hash library.

    N.B. This module is actively undergoing disruptive refactoring.  (It is
         one of the oldest modules yet to receive any attention and thus, it
         lags behind other modules with regards to common design patterns,
         especially regarding exposing the functionality via a COM interface.)

         It will likely be broken into two modules down the track.  Graph.c
         will contain the common "component" scaffolding (i.e. the COM vtbl
         methods), and a GraphImpl.c file will contain the implementation
         specific routines (adding edges, determining if it's acyclic, etc.)

    N.B. The bulk of this work has been done.  Graph.h (this file) now contains
         the COM-related and generic graph functionality (as cared about by
         consumers of the graph component, i.e. Chm01.c and PerfectHashContext).

         The specifics of the hypergraph logic are abstracted into GraphImpl.h,
         and the original implementation, based on the chmd project, live in
         GraphImpl1.[ch].

--*/

#include "stdafx.h"

//
// Define the primitive key, edge and vertex types and pointers to said types.
//

typedef ULONG KEY;
typedef ULONG EDGE;
typedef ULONG VERTEX;
typedef KEY *PKEY;
typedef EDGE *PEDGE;
typedef VERTEX *PVERTEX;
typedef union _VERTEX_PAIR {
    struct {
        VERTEX Vertex1;
        VERTEX Vertex2;
    };
    ULONGLONG AsULongLong;
    ULARGE_INTEGER AsULargeInteger;
} VERTEX_PAIR, *PVERTEX_PAIR;

//
// A core concept of the 2-part hypergraph algorithm for generating a perfect
// hash solution is the "assigned" array.  The size of this array is equal to
// the number of vertices in the graph.  The number of vertices in the graph is
// equal to the number of edges in the graph, rounded up to the next power of
// 2.  The number of edges in the graph is equal to the number of keys in the
// input set, rounded up to the next power of 2.  That is:
//
//  a) Number of keys in the input set.
//  b) Number of edges = number of keys, rounded up to a power of 2.
//  c) Number of vertices = number of edges, rounded up to the next power of 2.
//  d) Number of assigned elements = number of vertices.
//
// For example, KernelBase-2415.keys has 2415 keys, 4096 edges, and 8192
// vertices.  Thus, the assigned array will feature 8192 elements of the given
// underlying type (currently an unsigned 32-bit integer; ULONG).
//
// If, after a certain number of attempts, no graph solution has been found,
// the edges and vertices (and thus, assigned array) get doubled, and more
// attempts are made.  This is referred to as a table resize event.
//
// Poorly performing hash functions will often require numerous table resize
// events before finding a solution.  On the opposite side of the spectrum,
// good hash functions will often require no table resize events.  (We've yet
// to observe the Jenkins hash function resulting in a table resize event;
// however, its latency is 10x that of our best performing routines based on
// the crc32 hardware instruction.)
//
// Typically, the more complex the hash function, the better it performs, and
// vice versa.  Complexity in this case usually correlates directly to the
// number of CPU instructions required to calculate the hash, and thus, the
// latency of the routine.  Less instructions, lower latency, and vice versa.
// (Ignoring outlier instructions like idiv which can take upward of 90 cycles
//  to execute.)
//
// Hash routines with more instructions usually have long dependency chains,
// too, which further inhibits the performance.  That is, each calculation is
// dependent upon the results of the previous calculation, limiting the out-of-
// order and speculative execution capabilities of the CPU.
//
// Evaluating the latency of a given hash function is straight forward: you just
// measure the latency to perform the Index() routine for a given table (which
// results in two executions of the hash function (generating two hash codes),
// each with a different seed, and then two memory lookups into the assigned
// array, using the masked version of each hash code as the index).
//
// The performance of the memory lookups, however, can be highly variable, as
// they are at the mercy of the CPU cache.  The best scenario we can hope for is
// if the containing cache lines for both indices are in the L1 cache.  If we're
// benchmarking a single Index() call (with a constant key), we're guaranteed to
// get L1 cache hits after the first call.  So, the measured latency will tell
// us the best possible performance we can expect; micro-benchmarking at its
// finest.
//
// From a macro-benchmarking perspective, though, the size of the assigned array
// plays a large role.  In fact, it is not the size of the array per se, but the
// distribution of values throughout the array that will ultimately govern the
// likelihood of a cache hit or miss for a given key, assuming the keys being
// looked up are essentially random during the lifetime of the table (i.e. no
// key is any more probable of being looked up than another key).
//
// Are truly random key lookups a good measure of what happens in the real world
// though?  One could argue that a common use case would be hash tables where
// 90% of the lookups are performed against 10% of the keys, for example.  This
// may mean that under realistic workloads, the number of cache lines used by
// those 10% of keys is the critical factor; the remaining 90% of the keys are
// looked up so infrequently that the cost of a cache miss at all levels and
// subsequent memory fetch (so hundreds of cycles instead of 10-20 cycles) is
// ultimately irrelevant.  So, a table with 65536 elements may perform just as
// good, perhaps even better, than a table with 8192 elements, depending upon
// the actual real world workload.
//
// In order to quantify the impact of memory coverage of the assigned array,
// we need to be able to measure it.  That is the role of the following types
// being defined.
//
// N.B. Memory coverage is an active work-in-progress.
//

typedef VERTEX ASSIGNED;
typedef ASSIGNED *PASSIGNED;

#define ASSIGNED_SHIFT 2

#ifndef PAGE_SHIFT
#define PAGE_SHIFT 12
#endif

#ifndef PAGE_SIZE
#define PAGE_SIZE (1 << PAGE_SHIFT) // 4096
#endif

#ifndef LARGE_PAGE_SHIFT
#define LARGE_PAGE_SHIFT 21
#endif

#ifndef LARGE_PAGE_SIZE
#define LARGE_PAGE_SIZE (1 << LARGE_PAGE_SHIFT) // 2097152, or 2MB.
#endif

#ifndef CACHE_LINE_SHIFT
#define CACHE_LINE_SHIFT 6
#endif

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE (1 << CACHE_LINE_SHIFT) // 64
#endif

#define NUM_ASSIGNED_PER_PAGE       (PAGE_SIZE       / sizeof(ASSIGNED))
#define NUM_ASSIGNED_PER_LARGE_PAGE (LARGE_PAGE_SIZE / sizeof(ASSIGNED))
#define NUM_ASSIGNED_PER_CACHE_LINE (CACHE_LINE_SIZE / sizeof(ASSIGNED))

//
// For the human readers that don't like doing C preprocessor mental math,
// some C_ASSERTs to clarify the sizes above:
//

C_ASSERT(NUM_ASSIGNED_PER_PAGE       == 1024);      // Fits within USHORT.
C_ASSERT(NUM_ASSIGNED_PER_LARGE_PAGE == 524288);    // Fits within ULONG.
C_ASSERT(NUM_ASSIGNED_PER_CACHE_LINE == 16);        // Fits within BYTE.

typedef ASSIGNED ASSIGNED_PAGE[NUM_ASSIGNED_PER_PAGE];
typedef ASSIGNED_PAGE *PASSIGNED_PAGE;

typedef ASSIGNED ASSIGNED_LARGE_PAGE[NUM_ASSIGNED_PER_LARGE_PAGE];
typedef ASSIGNED_LARGE_PAGE *PASSIGNED_LARGE_PAGE;

typedef ASSIGNED ASSIGNED_CACHE_LINE[NUM_ASSIGNED_PER_CACHE_LINE];
typedef ASSIGNED_CACHE_LINE *PASSIGNED_CACHE_LINE;

typedef USHORT ASSIGNED_PAGE_COUNT;
typedef ASSIGNED_PAGE_COUNT *PASSIGNED_PAGE_COUNT;

typedef ULONG ASSIGNED_LARGE_PAGE_COUNT;
typedef ASSIGNED_LARGE_PAGE_COUNT *PASSIGNED_LARGE_PAGE_COUNT;

typedef BYTE ASSIGNED_CACHE_LINE_COUNT;
typedef ASSIGNED_CACHE_LINE_COUNT *PASSIGNED_CACHE_LINE_COUNT;

typedef struct _ASSIGNED_MEMORY_COVERAGE {

    ULONG TotalNumberOfPages;
    ULONG TotalNumberOfLargePages;
    ULONG TotalNumberOfCacheLines;

    union {
        ULONG NumberOfUsedPages;
        ULONG NumberOfPagesUsedByKeysSubset;
    };

    union {
        ULONG NumberOfUsedLargePages;
        ULONG NumberOfLargePagesUsedByKeysSubset;
    };

    union {
        ULONG NumberOfUsedCacheLines;
        ULONG NumberOfCacheLinesUsedByKeysSubset;
    };

    ULONG NumberOfEmptyPages;
    ULONG NumberOfEmptyLargePages;
    ULONG NumberOfEmptyCacheLines;

    ULONG FirstPageUsed;
    ULONG FirstLargePageUsed;
    ULONG FirstCacheLineUsed;

    ULONG LastPageUsed;
    ULONG LastLargePageUsed;
    ULONG LastCacheLineUsed;

    ULONG TotalNumberOfAssigned;

    _Writable_elements_(TotalNumberOfPages)
    PASSIGNED_PAGE_COUNT NumberOfAssignedPerPage;

    _Writable_elements_(TotalNumberOfLargePages)
    PASSIGNED_LARGE_PAGE_COUNT NumberOfAssignedPerLargePage;

    _Writable_elements_(TotalNumberOfCacheLines)
    PASSIGNED_CACHE_LINE_COUNT NumberOfAssignedPerCacheLine;

    //
    // Histogram of cache line counts.  The +1 accounts for the fact that we
    // want to count the number of times 0 occurs, as well, so we need 17 array
    // elements, not 16.
    //

    ULONG NumberOfAssignedPerCacheLineCounts[NUM_ASSIGNED_PER_CACHE_LINE + 1];
    union {
        ULONG MaxAssignedPerCacheLineCount;
        ULONG MaxAssignedPerCacheLineCountForKeysSubset;
    };

    //
    // If we're calculating memory coverage for a subset of keys, the following
    // counts will reflect the situation where the two vertices for a given key
    // are co-located within the same page, large page and cache line.
    //

    ULONG NumberOfKeysWithVerticesMappingToSamePage;
    ULONG NumberOfKeysWithVerticesMappingToSameLargePage;
    ULONG NumberOfKeysWithVerticesMappingToSameCacheLine;

    ULONG MaxGraphTraversalDepth;
    ULONG TotalGraphTraversals;
    ULONG NumberOfEmptyVertices;
    ULONG NumberOfCollisionsDuringAssignment;

    //
    // Pad out to an 8-byte boundary.
    //

    ULONG Padding;

    //
    // Stores Graph->Attempt at the time the memory coverage was captured.
    //

    LONGLONG Attempt;

} ASSIGNED_MEMORY_COVERAGE;
typedef ASSIGNED_MEMORY_COVERAGE *PASSIGNED_MEMORY_COVERAGE;
typedef const ASSIGNED_MEMORY_COVERAGE *PCASSIGNED_MEMORY_COVERAGE;

FORCEINLINE
VOID
CopyCoverage(
    _Out_writes_bytes_all_(sizeof(*Dest)) PASSIGNED_MEMORY_COVERAGE Dest,
    _In_reads_(sizeof(*Source)) PCASSIGNED_MEMORY_COVERAGE Source
    )
{
    //
    // Copy the structure, then clear the pointers.
    //

    CopyInline(Dest, Source, sizeof(*Dest));

    Dest->NumberOfAssignedPerPage = NULL;
    Dest->NumberOfAssignedPerLargePage = NULL;
    Dest->NumberOfAssignedPerCacheLine = NULL;
}

//
// Define a graph iterator structure use to facilitate graph traversal.
//

typedef struct _GRAPH_ITERATOR {
    VERTEX Vertex;
    EDGE Edge;
} GRAPH_ITERATOR;
typedef GRAPH_ITERATOR *PGRAPH_ITERATOR;

//
// Define graph flags.
//

typedef union _GRAPH_FLAGS {
    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // Indicates we've started deletion of edges from the graph.
        //

        ULONG Shrinking:1;

        //
        // When set, indicates the graph has been determined to be acyclic.
        // (This bit is set by IsGraphAcyclic() if the graph is acyclic; it
        //  is checked in GraphAssign().)
        //

        ULONG IsAcyclic:1;

        //
        // When set, indicates graph information has been set via the SetInfo()
        // call.
        //

        ULONG IsInfoSet:1;

        //
        // When set, indicates graph information has been loaded (at least once)
        // via a call to LoadInfo().
        //

        ULONG IsInfoLoaded:1;

        //
        // When set, turns the Graph->Vtbl->Verify() step into a no-op.
        //

        ULONG SkipVerification:1;

        //
        // When set, indicates this graph is the "spare graph" for the context.
        // When attempting to find the best graph solution, a worker thread may
        // register its graph as the current best solution, then use the spare
        // graph (or previous best graph) to continue solving attempts.
        //

        ULONG IsSpare:1;

        //
        // When set, indicates the graph should calculate assigned memory
        // coverage after a solution has been found.
        //

        ULONG WantsAssignedMemoryCoverage:1;

        //
        // When set, indicates the graph wants assigned memory coverage
        // information for a subset of keys (Graph->Context->KeysSubset).
        //

        ULONG WantsAssignedMemoryCoverageForKeysSubset:1;

        //
        // When set, enables additional redundant checks in IsGraphAcyclic()
        // with regards to counting deleted edges.
        //

        ULONG Paranoid:1;

        //
        // When set, indicates the VertexPairs array was successfully allocated
        // with large pages.  (Only applies when the --HashAllKeysFirst table
        // create flag is present.)
        //

        ULONG VertexPairsArrayUsesLargePages:1;

        //
        // When set, indicates the vertex pairs array wants write-combined
        // memory, when applicable.
        //

        ULONG WantsWriteCombiningForVertexPairsArray:1;

        //
        // When set, indicates the vertex pairs array is currently allocated
        // with write-combined page protection.
        //

        ULONG VertexPairsArrayIsWriteCombined:1;

        //
        // When set, indicates that the vertex pairs array wants to have the
        // write-combine page protection removed after all keys have been hashed
        // and prior to adding the vertices to the graph.
        //

        ULONG RemoveWriteCombineAfterSuccessfulHashKeys:1;

        //
        // Unused bits.
        //

        ULONG Unused:19;
    };
    LONG AsLong;
    ULONG AsULong;
} GRAPH_FLAGS;
typedef GRAPH_FLAGS *PGRAPH_FLAGS;
C_ASSERT(sizeof(GRAPH_FLAGS) == sizeof(ULONG));

#define IsGraphInfoSet(Graph) ((Graph)->Flags.IsInfoSet == TRUE)
#define IsGraphInfoLoaded(Graph) ((Graph)->Flags.IsInfoLoaded == TRUE)
#define IsSpareGraph(Graph) ((Graph)->Flags.IsSpare == TRUE)
#define SkipGraphVerification(Graph) ((Graph)->Flags.SkipVerification == TRUE)
#define WantsAssignedMemoryCoverage(Graph) \
    ((Graph)->Flags.WantsAssignedMemoryCoverage)
#define WantsAssignedMemoryCoverageForKeysSubset(Graph) \
    ((Graph)->Flags.WantsAssignedMemoryCoverageForKeysSubset)
#define IsGraphParanoid(Graph) ((Graph)->Flags.Paranoid == TRUE)

#define SetSpareGraph(Graph) (Graph->Flags.IsSpareGraph = TRUE)

DEFINE_UNUSED_STATE(GRAPH);

//
// Define the primary dimensions governing the graph size.
//

typedef struct _GRAPH_DIMENSIONS {

    //
    // Number of edges in the graph.  This corresponds to the number of keys
    // in our input set.  If modulus masking is active, the number of keys and
    // the number of edges will be identical.  Otherwise, the number of edges
    // will be the number of keys rounded up to a power of 2.
    //

    ULONG NumberOfEdges;

    //
    // Total number of edges in the graph.  This will be twice the size of the
    // NumberOfEdges value above, due to the quirky way the underlying r-graph
    // algorithm captures two hash values in the same list and offsets the
    // second set after the first, e.g.:
    //
    //      Edge2 = Edge1 + Graph->NumberOfEdges;
    //

    ULONG TotalNumberOfEdges;

    //
    // Number of vertices in the graph.  This will vary based on the masking
    // type.  It is doubled every time a graph resize event is encountered.
    //

    ULONG NumberOfVertices;

    //
    // The number of edges in the graph, rounded up to a power of 2, and then
    // shifted the appropriate amount to extract the exponent part for 2^n.
    //

    BYTE NumberOfEdgesPowerOf2Exponent;

    //
    // As above, but rounded up to the next power of 2 first.
    //

    BYTE NumberOfEdgesNextPowerOf2Exponent;

    //
    // The same exponent logic applied to the number of vertices as per the
    // NumberOfVertices field above.
    //

    BYTE NumberOfVerticesPowerOf2Exponent;

    //
    // And again for the next value.
    //

    BYTE NumberOfVerticesNextPowerOf2Exponent;

} GRAPH_DIMENSIONS;
C_ASSERT(sizeof(GRAPH_DIMENSIONS) == 16);
typedef GRAPH_DIMENSIONS *PGRAPH_DIMENSIONS;

//
// Define various memory offsets associated with a given graph structure.
// This allows parallel worker threads to reset their local GRAPH instance
// back to the initial state each time they want to try a new random seed.
//

typedef struct _GRAPH_INFO {

    //
    // Total number of graphs created.  This will match the maximum concurrency
    // level of the upstream context.
    //

    ULONG NumberOfGraphs;

    //
    // Number of RTL_BITMAP structures used by the graph.
    //

    USHORT NumberOfBitmaps;

    //
    // Pad out to a 4 byte boundary.
    //

    USHORT Padding;

    //
    // Size of the GRAPH structure.
    //

    ULONG SizeOfGraphStruct;

    //
    // System allocation granularity.  We align the memory map for the on-disk
    // structure using this value initially.
    //

    ULONG AllocationGranularity;

    //
    // If a masking type other than modulus is active, the AbsoluteEdge() needs
    // a way to mask edge values that exceed the number of edges in the table.
    // It does this via EdgeMask, which is initialized to the number of edges
    // (which will be power-of-2 sized for non-modulus masking), minus 1, such
    // that all lower bits will be set.
    //

    ULONG EdgeMask;

    //
    // Also capture the mask required to isolate vertices.
    //

    ULONG VertexMask;

    //
    // Number of pages, large pages and cache lines covered by the assigned
    // array.
    //

    ULONG AssignedArrayNumberOfPages;
    ULONG AssignedArrayNumberOfLargePages;
    ULONG AssignedArrayNumberOfCacheLines;

    //
    // Graph dimensions.  This information is duplicated in the graph due to
    // it being accessed frequently.
    //

    GRAPH_DIMENSIONS Dimensions;

    //
    // Pad out to an 8 byte boundary.
    //

    ULONG Padding2;

    //
    // Pointer to the owning context.
    //

    struct _PERFECT_HASH_CONTEXT *Context;

    //
    // Pointer to the previous Info, if applicable.
    //

    struct _GRAPH_INFO *PrevInfo;

    //
    // Array sizes.
    //

    ULONGLONG EdgesSizeInBytes;
    ULONGLONG NextSizeInBytes;
    ULONGLONG FirstSizeInBytes;
    ULONGLONG AssignedSizeInBytes;
    ULONGLONG VertexPairsSizeInBytes;
    ULONGLONG ValuesSizeInBytes;

    //
    // Bitmap buffer sizes.
    //

    ULONGLONG DeletedEdgesBitmapBufferSizeInBytes;
    ULONGLONG VisitedVerticesBitmapBufferSizeInBytes;
    ULONGLONG AssignedBitmapBufferSizeInBytes;
    ULONGLONG IndexBitmapBufferSizeInBytes;

    //
    // Assigned memory coverage buffer sizes for counts.
    //

    ULONGLONG NumberOfAssignedPerCacheLineSizeInBytes;
    ULONGLONG NumberOfAssignedPerPageSizeInBytes;
    ULONGLONG NumberOfAssignedPerLargePageSizeInBytes;

    //
    // The allocation size of all the arrays, bitmap buffers, and memory
    // coverage arrays, rounded up to the nearest page size.
    //

    ULONGLONG AllocSize;

} GRAPH_INFO;
typedef GRAPH_INFO *PGRAPH_INFO;

//
// Declare graph component and define vtbl methods.
//

DECLARE_COMPONENT(Graph, GRAPH);

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(&Graph->Lock)
HRESULT
(STDAPICALLTYPE GRAPH_SET_INFO)(
    _In_ PGRAPH Graph,
    _In_ PGRAPH_INFO Info
    );
typedef GRAPH_SET_INFO *PGRAPH_SET_INFO;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_lock_not_held_(Graph->Lock)
HRESULT
(STDAPICALLTYPE GRAPH_ENTER_SOLVING_LOOP)(
    _In_ PGRAPH Graph
    );
typedef GRAPH_ENTER_SOLVING_LOOP *PGRAPH_ENTER_SOLVING_LOOP;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
(STDAPICALLTYPE GRAPH_SOLVE)(
    _In_ PGRAPH Graph,
    _Inout_ PGRAPH *NewGraphPointer
    );
typedef GRAPH_SOLVE *PGRAPH_SOLVE;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
(STDAPICALLTYPE GRAPH_LOAD_INFO)(
    _In_ PGRAPH Graph
    );
typedef GRAPH_LOAD_INFO *PGRAPH_LOAD_INFO;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
(STDAPICALLTYPE GRAPH_RESET)(
    _In_ PGRAPH Graph
    );
typedef GRAPH_RESET *PGRAPH_RESET;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
(STDAPICALLTYPE GRAPH_LOAD_NEW_SEEDS)(
    _In_ PGRAPH Graph
    );
typedef GRAPH_LOAD_NEW_SEEDS *PGRAPH_LOAD_NEW_SEEDS;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
(STDAPICALLTYPE GRAPH_VERIFY)(
    _In_ PGRAPH Graph
    );
typedef GRAPH_VERIFY *PGRAPH_VERIFY;

typedef
_Requires_exclusive_lock_held_(Graph->Lock)
VOID
(STDAPICALLTYPE GRAPH_CALCULATE_ASSIGNED_MEMORY_COVERAGE)(
    _In_ PGRAPH Graph
    );
typedef GRAPH_CALCULATE_ASSIGNED_MEMORY_COVERAGE
      *PGRAPH_CALCULATE_ASSIGNED_MEMORY_COVERAGE;

typedef
_Requires_exclusive_lock_held_(Graph->Lock)
VOID
(STDAPICALLTYPE GRAPH_CALCULATE_ASSIGNED_MEMORY_COVERAGE_FOR_KEYS_SUBSET)(
    _In_ PGRAPH Graph
    );
typedef GRAPH_CALCULATE_ASSIGNED_MEMORY_COVERAGE_FOR_KEYS_SUBSET
      *PGRAPH_CALCULATE_ASSIGNED_MEMORY_COVERAGE_FOR_KEYS_SUBSET;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
(STDAPICALLTYPE GRAPH_REGISTER_SOLVED)(
    _In_ PGRAPH Graph,
    _Inout_ PGRAPH *NewGraphPointer
    );
typedef GRAPH_REGISTER_SOLVED *PGRAPH_REGISTER_SOLVED;

typedef
_Must_inspect_result_
_Requires_exclusive_lock_held_(Graph->Lock)
BOOLEAN
(STDAPICALLTYPE GRAPH_SHOULD_WE_CONTINUE_TRYING_TO_SOLVE)(
    _In_ struct _GRAPH *Graph
    );
typedef GRAPH_SHOULD_WE_CONTINUE_TRYING_TO_SOLVE
      *PGRAPH_SHOULD_WE_CONTINUE_TRYING_TO_SOLVE;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
(STDAPICALLTYPE GRAPH_ADD_KEYS)(
    _In_ PGRAPH Graph,
    _In_ ULONG NumberOfKeys,
    _In_reads_(NumberOfKeys) PKEY Keys
    );
typedef GRAPH_ADD_KEYS *PGRAPH_ADD_KEYS;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
(STDAPICALLTYPE GRAPH_HASH_KEYS)(
    _In_ PGRAPH Graph,
    _In_ ULONG NumberOfKeys,
    _In_reads_(NumberOfKeys) PKEY Keys
    );
typedef GRAPH_HASH_KEYS *PGRAPH_HASH_KEYS;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
(STDAPICALLTYPE GRAPH_ADD_HASHED_KEYS)(
    _In_ PGRAPH Graph,
    _In_ ULONG NumberOfKeys,
    _In_reads_(NumberOfKeys) PVERTEX_PAIR VertexPairs
    );
typedef GRAPH_ADD_HASHED_KEYS *PGRAPH_ADD_HASHED_KEYS;

typedef struct _GRAPH_VTBL {
    DECLARE_COMPONENT_VTBL_HEADER(GRAPH);
    PGRAPH_SET_INFO SetInfo;
    PGRAPH_ENTER_SOLVING_LOOP EnterSolvingLoop;
    PGRAPH_LOAD_INFO LoadInfo;
    PGRAPH_RESET Reset;
    PGRAPH_LOAD_NEW_SEEDS LoadNewSeeds;
    PGRAPH_SOLVE Solve;
    PGRAPH_VERIFY Verify;
    PGRAPH_CALCULATE_ASSIGNED_MEMORY_COVERAGE CalculateAssignedMemoryCoverage;
    PGRAPH_CALCULATE_ASSIGNED_MEMORY_COVERAGE_FOR_KEYS_SUBSET
        CalculateAssignedMemoryCoverageForKeysSubset;
    PGRAPH_REGISTER_SOLVED RegisterSolved;
    PGRAPH_SHOULD_WE_CONTINUE_TRYING_TO_SOLVE ShouldWeContinueTryingToSolve;
    PGRAPH_ADD_KEYS AddKeys;
} GRAPH_VTBL;
typedef GRAPH_VTBL *PGRAPH_VTBL;

//
// Define the graph structure.
//

typedef struct _Struct_size_bytes_(SizeOfStruct) _GRAPH {
    COMMON_COMPONENT_HEADER(GRAPH);

    //
    // Edge and vertex masks that can be used when non-modulus masking is in
    // place.  Both of these values are duplicated from the info structure as
    // they are accessed frequently.
    //
    //

    ULONG EdgeMask;
    ULONG VertexMask;

    //
    // Ditto for modulus-masking equivalents.
    //

    ULONG EdgeModulus;
    ULONG VertexModulus;

    //
    // Duplicate the mask type, as well, as this directs AbsoluteEdge()'s
    // decision to use the two masks above.
    //

    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId;

    //
    // Duplicate the number of keys, as this is also frequently referenced.
    //

    ULONG NumberOfKeys;

    //
    // Pointer to the info structure describing various sizes.
    //

    PGRAPH_INFO Info;

    //
    // Graph attempt.  This ID is derived from an interlocked increment against
    // Context->Attempts, and represents the attempt number across all threads.
    //

    ULONGLONG Attempt;

    //
    // A localized attempt number that reflects the number of attempts made
    // by just this thread.
    //

    ULONG ThreadAttempt;

    //
    // Thread ID of the thread that owns us.  Each callback thread is provided
    // a single graph, and will attempt to solve the perfect hash table until
    // told otherwise.  Thus, there's a 1:1 relationship between graph instance
    // and owning thread.
    //

    ULONG ThreadId;

    //
    // Counter that is incremented each time we delete an edge during the
    // acyclic graph detection stage.
    //

    ULONG DeletedEdgeCount;

    //
    // Counter that is incremented each time we visit a vertex during the
    // assignment stage.
    //

    ULONG VisitedVerticesCount;

    //
    // Capture collisions during assignment step.
    //

    ULONG Collisions;

    //
    // Inline the GRAPH_DIMENSIONS structure.  This is available from the
    // GRAPH_INFO structure, however, it's accessed frequently, so we inline
    // it to avoid the extra level of indirection.
    //

    union {

        struct {
            ULONG NumberOfEdges;
            ULONG TotalNumberOfEdges;
            ULONG NumberOfVertices;
            BYTE NumberOfEdgesPowerOf2Exponent;
            BYTE NumberOfEdgesNextPowerOf2Exponent;
            BYTE NumberOfVerticesPowerOf2Exponent;
            BYTE NumberOfVerticesNextPowerOf2Exponent;
        };

        GRAPH_DIMENSIONS Dimensions;
    };

    //
    // The number of vertices when LoadInfo() was last called.
    //

    ULONG LastLoadedNumberOfVertices;

    //
    // Captures the 0-based index of this graph in the graphs array allocated
    // by the parent table creation thread.
    //

    ULONG Index;

    //
    // Number of empty vertices encountered during the assignment step.
    //

    ULONG NumberOfEmptyVertices;

    //
    // Duplicate the context pointer.  (This is also available from Info.)
    //

    struct _PERFECT_HASH_CONTEXT *Context;

    //
    // Edges array.
    //

    _Writable_elements_(TotalNumberOfEdges)
    PEDGE Edges;

    //
    // Array of the "next" edge array, as per the referenced papers.
    //

    _Writable_elements_(TotalNumberOfEdges)
    PEDGE Next;

    //
    // Array of vertices.
    //

    _Writable_elements_(NumberOfVertices)
    PVERTEX First;

    //
    // Array of assigned vertices.
    //

    _Writable_elements_(NumberOfVertices)
    PVERTEX Assigned;

    //
    // Optional array of vertex pairs, indexed by number of keys.
    //

    _Writable_elements_(NumberOfKeys)
    PVERTEX_PAIR VertexPairs;

    //
    // Array of values indexed by the offsets in the Assigned array.  This
    // essentially allows us to simulate a loaded table that supports the
    // Insert(), Index() and Lookup() routines as part of graph validation.
    //

    PULONG Values;

    //
    // Bitmap used to capture deleted edges as part of the acyclic detection
    // stage.  The SizeOfBitMap will reflect TotalNumberOfEdges.
    //

    RTL_BITMAP DeletedEdgesBitmap;

    //
    // Bitmap used to capture vertices visited as part of the assignment stage.
    // The SizeOfBitMap will reflect NumberOfVertices.
    //

    RTL_BITMAP VisitedVerticesBitmap;

    //
    // Bitmap used to test the correctness of the Assigned array.
    //

    RTL_BITMAP AssignedBitmap;

    //
    // Bitmap used to track indices during the assignment step.
    //

    RTL_BITMAP IndexBitmap;

    //
    // Memory coverage information for the assigned array.
    //

    ASSIGNED_MEMORY_COVERAGE AssignedMemoryCoverage;

    //
    // Elapsed cycles of the GraphAddKeys() routine.
    //

    LARGE_INTEGER AddKeysElapsedCycles;

    //
    // Elapsed cycles of the GraphHashKeys() routine, if used.
    //

    LARGE_INTEGER HashKeysElapsedCycles;

    //
    // Elapsed cycles of the GraphAddHashedKeys() routine, if used.
    //

    LARGE_INTEGER AddHashedKeysElapsedCycles;

    //
    // Elapsed microseconds of the GraphAddKeys() routine.
    //

    LARGE_INTEGER AddKeysElapsedMicroseconds;

    //
    // Elapsed microseconds of the GraphHashKeys() routine, if used.
    //

    LARGE_INTEGER HashKeysElapsedMicroseconds;

    //
    // Elapsed microseconds of the GraphAddHashedKeys() routine, if used.
    //

    LARGE_INTEGER AddHashedKeysElapsedMicroseconds;

    //
    // The graph interface.
    //

    GRAPH_VTBL Interface;

    //
    // The current recursive traversal depth during assignment.
    //

    ULONG TraversalDepth;

    //
    // Maximum recursive traversal depth observed during assignment.
    //

    ULONG MaximumTraversalDepth;

    //
    // Total number of graph traversals performed during assignment.
    //

    ULONG TotalTraversals;

    //
    // Capture the seeds used for each hash function employed by the graph.
    //

    ULONG NumberOfSeeds;

    union {
        ULONG Seeds[MAX_NUMBER_OF_SEEDS];
        struct {
            union {
                struct {
                    union {
                        ULONG Seed1;
                        ULONG FirstSeed;
                    };
                    ULONG Seed2;
                };
                ULARGE_INTEGER Seeds12;
            };
            union {
                struct {
                    ULONG Seed3;
                    ULONG Seed4;
                };
                ULARGE_INTEGER Seeds34;
            };
            union {
                struct {
                    ULONG Seed5;
                    ULONG Seed6;
                };
                ULARGE_INTEGER Seeds56;
            };
            union {
                struct {
                    ULONG Seed7;
                    union {
                        ULONG Seed8;
                        ULONG LastSeed;
                    };
                };
                ULARGE_INTEGER Seeds78;
            };
        };
    };

} GRAPH;
typedef GRAPH *PGRAPH;

//
// Locking macro helpers.
//

#define TryAcquireGraphLockExclusive(Graph) \
    TryAcquireSRWLockExclusive(&Graph->Lock)

#define AcquireGraphLockExclusive(Graph) \
    AcquireSRWLockExclusive(&Graph->Lock)

#define ReleaseGraphLockExclusive(Graph) \
    ReleaseSRWLockExclusive(&Graph->Lock)

#define TryAcquireGraphLockShared(Graph) \
    TryAcquireSRWLockShared(&Graph->Lock)

#define AcquireGraphLockShared(Graph) \
    AcquireSRWLockShared(&Graph->Lock)

#define ReleaseGraphLockShared(Graph) \
    ReleaseSRWLockShared(&Graph->Lock)

//
// Bitmap macro helpers.
//

#define TestGraphBit(Name, BitNumber) \
    BitTest64((PLONGLONG)Graph->##Name##.Buffer, (LONGLONG)BitNumber)

#define SetGraphBit(Name, BitNumber) \
    BitTestAndSet64((PLONGLONG)Graph->##Name##.Buffer, (LONGLONG)BitNumber)

//
// Private non-vtbl methods.
//

typedef
HRESULT
(NTAPI GRAPH_INITIALIZE)(
    _In_ PGRAPH Graph
    );
typedef GRAPH_INITIALIZE *PGRAPH_INITIALIZE;

typedef
VOID
(NTAPI GRAPH_RUNDOWN)(
    _In_ _Post_ptr_invalid_ PGRAPH Path
    );
typedef GRAPH_RUNDOWN *PGRAPH_RUNDOWN;

typedef
_Success_(return >= 0)
HRESULT
(NTAPI GRAPH_APPLY_USER_SEEDS)(
    _In_ PGRAPH Graph
    );
typedef GRAPH_APPLY_USER_SEEDS *PGRAPH_APPLY_USER_SEEDS;

typedef
_Success_(return >= 0)
HRESULT
(NTAPI GRAPH_APPLY_SEED_MASKS)(
    _In_ PGRAPH Graph
    );
typedef GRAPH_APPLY_SEED_MASKS *PGRAPH_APPLY_SEED_MASKS;

typedef
_Success_(return >= 0)
HRESULT
(NTAPI GRAPH_APPLY_WEIGHTED_SEED_MASKS)(
    _In_ PGRAPH Graph,
    _In_opt_ PCSEED_MASK_COUNTS SeedMaskCounts
    );
typedef GRAPH_APPLY_WEIGHTED_SEED_MASKS *PGRAPH_APPLY_WEIGHTED_SEED_MASKS;


#ifndef __INTELLISENSE__
extern GRAPH_INITIALIZE GraphInitialize;
extern GRAPH_RUNDOWN GraphRundown;
extern GRAPH_APPLY_USER_SEEDS GraphApplyUserSeeds;
extern GRAPH_APPLY_SEED_MASKS GraphApplySeedMasks;
extern GRAPH_APPLY_WEIGHTED_SEED_MASKS GraphApplyWeightedSeedMasks;

//
// Private vtbl methods.
//
// N.B. These need to come after the GRAPH structure definition in order for
//      the SAL concurrency annotations to work (i.e. _Requires_lock_not_held_).
//

extern GRAPH_SET_INFO GraphSetInfo;
extern GRAPH_ENTER_SOLVING_LOOP GraphEnterSolvingLoop;
extern GRAPH_LOAD_INFO GraphLoadInfo;
extern GRAPH_LOAD_NEW_SEEDS GraphLoadNewSeeds;
extern GRAPH_RESET GraphReset;
extern GRAPH_SOLVE GraphSolve;
extern GRAPH_VERIFY GraphVerify;
extern GRAPH_CALCULATE_ASSIGNED_MEMORY_COVERAGE
    GraphCalculateAssignedMemoryCoverage;
extern GRAPH_CALCULATE_ASSIGNED_MEMORY_COVERAGE_FOR_KEYS_SUBSET
    GraphCalculateAssignedMemoryCoverageForKeysSubset;
extern GRAPH_REGISTER_SOLVED GraphRegisterSolved;
#ifdef _M_X64
extern GRAPH_REGISTER_SOLVED GraphRegisterSolvedTsx;
#endif
extern GRAPH_SHOULD_WE_CONTINUE_TRYING_TO_SOLVE
    GraphShouldWeContinueTryingToSolve;
extern GRAPH_ADD_KEYS GraphAddKeys;
extern GRAPH_HASH_KEYS GraphHashKeys;
extern GRAPH_ADD_HASHED_KEYS GraphAddHashedKeys;
#endif

//
// Define a helper macro for hashing keys during graph creation.  Assumes a
// variable named Graph is in scope.  We can't use the Table->Vtbl->Hash()
// vtbl function during graph creation because it obtains the seed data
// from the table header -- which is the appropriate place to find it once
// we're dealing with a previously-created table that has been loaded, but
// won't have a value during the graph solving step because the seed data
// is located in the graph itself.
//

#define SEEDED_HASH(Key, Result)                             \
    if (FAILED(Table->Vtbl->SeededHash(Table,                \
                                       Key,                  \
                                       Graph->NumberOfSeeds, \
                                       &Graph->FirstSeed,    \
                                       Result))) {           \
        goto Error;                                          \
    }

//
// Define an on-disk representation of the graph's information.  This is stored
// in the NTFS stream extending from the backing file named :Info.  It is
// responsible for storing information about the on-disk mapping such that it
// can be reloaded from disk and used as a perfect hash table.  The structure
// must always embed the TABLE_INFO_ON_DISK_HEADER structure such that the
// generic loader routine can access on-disk versions saved by different algos
// in order to extract the algorithm ID and determine a suitable loader func to
// use.
//

typedef struct _Struct_size_bytes_(Header.SizeOfStruct) _GRAPH_INFO_ON_DISK {

    //
    // Include the required header.
    //

    TABLE_INFO_ON_DISK TableInfoOnDisk;

    //
    // Additional information we capture is mostly just for informational
    // and debugging purposes.
    //

    //
    // Inline the GRAPH_DIMENSIONS structure.
    //

    union {

        struct {
            ULONG NumberOfEdges;
            ULONG TotalNumberOfEdges;
            ULONG NumberOfVertices;
            BYTE NumberOfEdgesPowerOf2Exponent;
            BYTE NumberOfEdgesNextPowerOf2Exponent;
            BYTE NumberOfVerticesPowerOf2Exponent;
            BYTE NumberOfVerticesNextPowerOf2Exponent;
        };

        GRAPH_DIMENSIONS Dimensions;
    };

} GRAPH_INFO_ON_DISK;
C_ASSERT(sizeof(GRAPH_INFO_ON_DISK) <= PAGE_SIZE);
typedef GRAPH_INFO_ON_DISK *PGRAPH_INFO_ON_DISK;

//
// Define a helper macro for checking whether or not graph solving should stop.
//

#define MAYBE_STOP_GRAPH_SOLVING(Graph)                               \
    if (Graph->Vtbl->ShouldWeContinueTryingToSolve(Graph) == FALSE) { \
        return PH_S_GRAPH_SOLVING_STOPPED;                            \
    }

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
