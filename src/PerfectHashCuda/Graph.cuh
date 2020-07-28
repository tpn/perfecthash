/*++

Copyright (c) 2020 Trent Nelson <trent@trent.me>

Module Name:

    Graph.cuh

Abstract:

    CUDA graph implementation.

--*/

#pragma once

extern "C" {

#include <no_sal2.h>
#include "../PerfectHash/Cu.cuh"

#define MAX_NUMBER_OF_SEEDS 8

//
// Define helper macros for referring to seed constants stored in local
// variables by their uppercase names.  This allows easy copy-and-pasting of
// the algorithm "guts" between the "compiled" perfect hash table routines in
// ../CompiledPerfectHashTable and the SeededHashEx() implementations here.
//

#define SEED1 Seed1
#define SEED2 Seed2
#define SEED3 Seed3
#define SEED4 Seed4
#define SEED5 Seed5
#define SEED6 Seed6
#define SEED7 Seed7
#define SEED8 Seed8

#define SEED3_BYTE1 Seed3.Byte1
#define SEED3_BYTE2 Seed3.Byte2
#define SEED3_BYTE3 Seed3.Byte3
#define SEED3_BYTE4 Seed3.Byte4

#define SEED6_BYTE1 Seed6.Byte1
#define SEED6_BYTE2 Seed6.Byte2
#define SEED6_BYTE3 Seed6.Byte3
#define SEED6_BYTE4 Seed6.Byte4

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
        // When set, indicates that this is a CUDA graph.
        //

        ULONG IsCuGraph:1;

        //
        // When set, indicates this graph is the CUDA "spare graph" for the
        // context.  When attempting to find the best graph solution, a worker
        // thread may register its graph as the current best solution, then use
        // the spare graph (or previous best graph) to continue solving
        // attempts.
        //

        ULONG IsCuSpare:1;

        //
        // Unused bits.
        //

        ULONG Unused:17;
    };
    LONG AsLong;
    ULONG AsULong;
} GRAPH_FLAGS;
typedef GRAPH_FLAGS *PGRAPH_FLAGS;

#define IsGraphInfoSet(Graph) ((Graph)->Flags.IsInfoSet != FALSE)
#define IsGraphInfoLoaded(Graph) ((Graph)->Flags.IsInfoLoaded != FALSE)
#define IsSpareGraph(Graph) ((Graph)->Flags.IsSpare != FALSE)
#define IsSpareCuGraph(Graph) ((Graph)->Flags.IsCuSpare != FALSE)
#define SkipGraphVerification(Graph) ((Graph)->Flags.SkipVerification != FALSE)
#define WantsAssignedMemoryCoverage(Graph) \
    ((Graph)->Flags.WantsAssignedMemoryCoverage)
#define WantsAssignedMemoryCoverageForKeysSubset(Graph) \
    ((Graph)->Flags.WantsAssignedMemoryCoverageForKeysSubset)
#define IsGraphParanoid(Graph) ((Graph)->Flags.Paranoid != FALSE)
#define IsCuGraph(Graph) ((Graph)->Flags.IsCuGraph != FALSE)

#define SetSpareGraph(Graph) (Graph->Flags.IsSpareGraph = TRUE)
#define SetSpareCuGraph(Graph) (Graph->Flags.IsSpareCuGraph = TRUE)

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
    ULONGLONG VertexPairsSizeInBytes;
    ULONGLONG ValuesSizeInBytes;

    //
    // We use a union for the Assigned size in order to work with macros in the
    // CUDA GraphCuLoadInfo() routine.
    //

    union {
        ULONGLONG AssignedSizeInBytes;
        ULONGLONG AssignedHostSizeInBytes;
        ULONGLONG AssignedDeviceSizeInBytes;
    };

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

#define COMMON_COMPONENT_HEADER(Name) \
    PVOID Vtbl;                       \
    PVOID Lock;                       \
    LIST_ENTRY ListEntry;             \
    struct _RTL *Rtl;                 \
    struct _ALLOCATOR *Allocator;     \
    PVOID OuterUnknown;               \
    volatile LONG ReferenceCount;     \
    ULONG Id;                         \
    ULONG SizeOfStruct;               \
    Name##_STATE State;               \
    Name##_FLAGS Flags;               \
    ULONG Reserved

#define DEFINE_UNUSED_STATE(Name)                  \
typedef union _##Name##_STATE {                    \
    struct {                                       \
        ULONG Unused:32;                           \
    };                                             \
    LONG AsLong;                                   \
    ULONG AsULong;                                 \
} Name##_STATE;                                    \
typedef Name##_STATE *P##Name##_STATE

DEFINE_UNUSED_STATE(GRAPH);

typedef struct _GRAPH_VTBL {
    PVOID QueryInterface;
    PVOID AddRef;
    PVOID Release;
    PVOID CreateInstance;
    PVOID LockServer;

    PVOID SetInfo;
    PVOID EnterSolvingLoop;
    PVOID Verify;

    PVOID LoadInfo;
    PVOID Reset;
    PVOID LoadNewSeeds;
    PVOID Solve;
    PVOID CalculateAssignedMemoryCoverage;
    PVOID CalculateAssignedMemoryCoverageForKeysSubset;
    PVOID RegisterSolved;
    PVOID ShouldWeContinueTryingToSolve;
    PVOID AddKeys;
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

    ULONG MaskFunctionId;

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
    // If this is the host memory graph for a CUDA graph, this is the
    // corresponding device memory address for the graph.
    //

    struct _GRAPH *CuDeviceGraph;

    struct _PH_CU_SOLVE_CONTEXT *CuSolveContext;

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

    union {
        _Writable_elements_(NumberOfVertices)
        PVERTEX Assigned;

        _Writable_elements_(NumberOfVertices)
        PVERTEX AssignedHost;
    };

    //
    // Device address of assigned vertices.
    //

    _Writable_elements_(NumberOfVertices)
    PVERTEX AssignedDevice;

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

    //
    // The graph interface.
    //

    GRAPH_VTBL Interface;

} GRAPH;
typedef GRAPH *PGRAPH;

} // extern "C"

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
