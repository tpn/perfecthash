/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Graph.h

Abstract:

    This is the header file for the Graph module of the perfect hash library.

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

//
// Define a graph iterator structure use to facilitate graph traversal.
//

typedef struct _GRAPH_ITERATOR {
    VERTEX Vertex;
    EDGE Edge;
} GRAPH_ITERATOR;
typedef GRAPH_ITERATOR *PGRAPH_ITERATOR;

//
// Define helper macros for EMPTY and GRAPH_NO_NEIGHBOR constants.
//
// N.B. I'm not sure why they don't use NULL/0 for the empty edge case.  Using
//      -1 means the edge array needs to be filled with -1s as part of graph
//      initialization, which seems inefficient and unnecessary.
//

#define EMPTY ((ULONG)-1)
#define IsEmpty(Value) ((ULONG)Value == EMPTY)
#define GRAPH_NO_NEIGHBOR ((ULONG)-1)
#define IsNeighborEmpty(Neighbor) ((ULONG)Neighbor == EMPTY)

//
// Define graph flags.
//

typedef union _GRAPH_FLAGS {
    struct {

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
        // Unused bits.
        //

        ULONG Unused:30;
    };
    LONG AsLong;
    ULONG AsULong;
} GRAPH_FLAGS;
typedef GRAPH_FLAGS *PGRAPH_FLAGS;

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
    // Number of pages consumed by the entire graph and all backing arrays.
    //

    ULONG NumberOfPagesPerGraph;

    //
    // Page size (e.g. 4096, 2MB).
    //

    ULONG PageSize;

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
    // Size of the GRAPH structure.
    //

    USHORT SizeOfGraphStruct;

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
    // Graph dimensions.  This information is duplicated in the graph due to
    // it being accessed frequently.
    //

    GRAPH_DIMENSIONS Dimensions;

    ULONG Padding;

    //
    // Pointer to the owning context.
    //

    struct _PERFECT_HASH_CONTEXT *Context;

    //
    // Base address of the entire graph allocation.
    //

    union {
        PVOID BaseAddress;
        struct _GRAPH *FirstGraph;
    };

    //
    // Array sizes.
    //

    ULONGLONG EdgesSizeInBytes;
    ULONGLONG NextSizeInBytes;
    ULONGLONG FirstSizeInBytes;
    ULONGLONG PrevSizeInBytes;
    ULONGLONG AssignedSizeInBytes;
    ULONGLONG ValuesSizeInBytes;

    //
    // Deleted edges bitmap buffer size.
    //

    ULONGLONG DeletedEdgesBitmapBufferSizeInBytes;

    //
    // Visited vertices bitmap buffer size.
    //

    ULONGLONG VisitedVerticesBitmapBufferSizeInBytes;

    //
    // Assigned bitmap buffer size.
    //

    ULONGLONG AssignedBitmapBufferSizeInBytes;

    //
    // Index bitmap buffer size.
    //

    ULONGLONG IndexBitmapBufferSizeInBytes;

    //
    // The allocation size of the graph, including structure size and all
    // array and bitmap buffer sizes.
    //

    ULONGLONG AllocSize;

    //
    // Allocation size rounded up to the nearest page size multiple.
    //

    ULONGLONG FinalSize;

} GRAPH_INFO;
typedef GRAPH_INFO *PGRAPH_INFO;

//
// Define the graph structure.  This represents an r-graph, or a hypergraph,
// or an r-partite 2-uniform graph, or any other seemingly unlimited number
// of names floating around in academia for what appears to be exactly the
// same thing.
//

typedef struct _Struct_size_bytes_(SizeOfStruct) _GRAPH {

    //
    // List entry used to push the graph onto the context's work list.
    //

    SLIST_ENTRY ListEntry;

    //
    // Edge and vertex masks that can be used when non-modulus masking is in
    // place.  Both of these values are duplicated from the info structure as
    // they are accessed frequently.
    //
    //

    ULONG EdgeMask;
    ULONG VertexMask;

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
    // Structure size, in bytes.
    //

    _Field_range_(== , sizeof(struct _GRAPH)) ULONG SizeOfStruct;

    //
    // Graph flags.
    //

    GRAPH_FLAGS Flags;

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

    ULONG Padding;

    //
    // Duplicate the context pointer.  (This is also available from Info.)
    //

    struct _PERFECT_HASH_CONTEXT *Context;

    //
    // Edges array.  The number of elements in this array is governed by the
    // TotalNumberOfEdges field, and will be twice the number of edges.
    //

    PEDGE Edges;

    //
    // Array of the "next" edge array, as per the referenced papers.  The number
    // of elements in this array is also governed by TotalNumberOfEdges.
    //

    PEDGE Next;

    //
    // Array of vertices.  Number of elements is governed by the
    // NumberOfVertices field.
    //

    PVERTEX First;

    //
    // The original CHM paper in 1996 references a "prev" array to "facilitate
    // fast deletion".  However, the chmp project appears to have switched to
    // using bitmaps.  Let's reserve a slot for the "prev" array anyway.
    //

    PVERTEX Prev;

    //
    // Array of assigned vertices.  Number of elements is governed by the
    // NumberOfVertices field.
    //

    PVERTEX Assigned;

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

    RTL_BITMAP DeletedEdges;

    //
    // Bitmap used to capture vertices visited as part of the assignment stage.
    // The SizeOfBitMap will reflect NumberOfVertices.
    //

    RTL_BITMAP VisitedVertices;

    //
    // Bitmap used to test the correctness of the Assigned array.
    //

    RTL_BITMAP AssignedBitmap;

    //
    // Bitmap used to track indices during the assignment step.
    //

    RTL_BITMAP IndexBitmap;

    //
    // Capture the seeds used for each hash function employed by the graph.
    //

    ULONG NumberOfSeeds;

    ULONG Padding2;

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
                union {
                    ULONG Seed4;
                    ULONG LastSeed;
                };
            };
            ULARGE_INTEGER Seeds34;
        };
    };

} GRAPH;
typedef GRAPH *PGRAPH;

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

    TABLE_INFO_ON_DISK_HEADER Header;

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
// Function pointer typedefs.
//

typedef
VOID
(NTAPI INITIALIZE_GRAPH)(
    _In_ PGRAPH_INFO Info,
    _In_ PGRAPH Graph
    );
typedef INITIALIZE_GRAPH *PINITIALIZE_GRAPH;

typedef
BOOLEAN
(NTAPI SOLVE_GRAPH)(
    _In_ PGRAPH Graph
    );
typedef SOLVE_GRAPH *PSOLVE_GRAPH;

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI VERIFY_SOLVED_GRAPH)(
    _In_ PGRAPH Graph
    );
typedef VERIFY_SOLVED_GRAPH *PVERIFY_SOLVED_GRAPH;

extern SOLVE_GRAPH SolveGraph;
extern VERIFY_SOLVED_GRAPH VerifySolvedGraph;
extern INITIALIZE_GRAPH InitializeGraph;

////////////////////////////////////////////////////////////////////////////////
// Algorithm Implementation Typedefs
////////////////////////////////////////////////////////////////////////////////

typedef
VOID
(NTAPI GRAPH_ADD_EDGE)(
    _In_ PGRAPH Graph,
    _In_ EDGE Edge,
    _In_ VERTEX Vertex1,
    _In_ VERTEX Vertex2
    );
typedef GRAPH_ADD_EDGE *PGRAPH_ADD_EDGE;
extern GRAPH_ADD_EDGE GraphAddEdge;

typedef
VOID
(NTAPI GRAPH_CYCLIC_DELETE_EDGE)(
    _In_ PGRAPH Graph,
    _In_ VERTEX Vertex
    );
typedef GRAPH_CYCLIC_DELETE_EDGE *PGRAPH_CYCLIC_DELETE_EDGE;
extern GRAPH_CYCLIC_DELETE_EDGE GraphCyclicDeleteEdge;

typedef
_Success_(return != 0)
BOOLEAN
(NTAPI GRAPH_FIND_DEGREE1_EDGE)(
    _In_ PGRAPH Graph,
    _In_ VERTEX Vertex,
    _Out_ PEDGE Edge
    );
typedef GRAPH_FIND_DEGREE1_EDGE *PGRAPH_FIND_DEGREE1_EDGE;
extern GRAPH_FIND_DEGREE1_EDGE GraphFindDegree1Edge;

typedef
_Check_return_
BOOLEAN
(NTAPI IS_GRAPH_ACYCLIC)(
    _In_ PGRAPH Graph
    );
typedef IS_GRAPH_ACYCLIC *PIS_GRAPH_ACYCLIC;
extern IS_GRAPH_ACYCLIC IsGraphAcyclic;

typedef
_Check_return_
GRAPH_ITERATOR
(NTAPI GRAPH_NEIGHBORS_ITERATOR)(
    _In_ PGRAPH Graph,
    _In_ VERTEX Vertex
    );
typedef GRAPH_NEIGHBORS_ITERATOR *PGRAPH_NEIGHBORS_ITERATOR;
extern GRAPH_NEIGHBORS_ITERATOR GraphNeighborsIterator;

typedef
VERTEX
(NTAPI GRAPH_NEXT_NEIGHBOR)(
    _In_ PGRAPH Graph,
    _Inout_ PGRAPH_ITERATOR Iterator
    );
typedef GRAPH_NEXT_NEIGHBOR *PGRAPH_NEXT_NEIGHBOR;
extern GRAPH_NEXT_NEIGHBOR GraphNextNeighbor;

typedef
EDGE
(NTAPI GRAPH_EDGE_ID)(
    _In_ PGRAPH Graph,
    _In_ VERTEX Vertex1,
    _In_ VERTEX Vertex2
    );
typedef GRAPH_EDGE_ID *PGRAPH_EDGE_ID;
extern GRAPH_EDGE_ID GraphEdgeId;

typedef
VOID
(NTAPI GRAPH_DELETE_EDGE)(
    _In_ PGRAPH Graph,
    _In_ EDGE Edge
    );
typedef GRAPH_DELETE_EDGE *PGRAPH_DELETE_EDGE;
extern GRAPH_DELETE_EDGE GraphDeleteEdge;

typedef
BOOLEAN
(NTAPI GRAPH_CHECK_EDGE)(
    _In_ PGRAPH Graph,
    _In_ EDGE Edge,
    _In_ VERTEX Vertex1,
    _In_ VERTEX Vertex2
    );
typedef GRAPH_CHECK_EDGE *PGRAPH_CHECK_EDGE;
extern GRAPH_CHECK_EDGE GraphCheckEdge;

typedef
VOID
(NTAPI GRAPH_TRAVERSE)(
    _In_ PGRAPH Graph,
    _In_ EDGE Edge,
    _Inout_ PULONG Depth,
    _Inout_ PULONG MaximumDepth
    );
typedef GRAPH_TRAVERSE *PGRAPH_TRAVERSE;
extern GRAPH_TRAVERSE GraphTraverse;

typedef
VOID
(NTAPI GRAPH_ASSIGN)(
    _In_ PGRAPH Graph
    );
typedef GRAPH_ASSIGN *PGRAPH_ASSIGN;
extern GRAPH_ASSIGN GraphAssign;

//
// Inline function helpers.
//

FORCEINLINE
EDGE
AbsoluteEdge(
    _In_ PGRAPH Graph,
    _In_ EDGE Edge,
    _In_ ULONG Index
    )
{
    ULONG AbsEdge;
    ULONG MaskedEdge;
    ULONG NumberOfEdges;

    NumberOfEdges = Graph->NumberOfEdges;

    if (IsModulusMasking(Graph->MaskFunctionId)) {

        MaskedEdge = Edge % NumberOfEdges;

    } else {

        MaskedEdge = Edge & Graph->EdgeMask;

    }

    AbsEdge = (MaskedEdge + (Index * Graph->NumberOfEdges));
    return AbsEdge;
}

#define TestGraphBit(Name, BitNumber) \
    BitTest64((PLONGLONG)Graph->##Name##.Buffer, (LONGLONG)BitNumber)

FORCEINLINE
BOOLEAN
IsDeletedEdge(
    _In_ PGRAPH Graph,
    _In_ EDGE Edge
    )
{
    return TestGraphBit(DeletedEdges, Edge + 1);
}

FORCEINLINE
BOOLEAN
IsVisitedVertex(
    _In_ PGRAPH Graph,
    _In_ VERTEX Vertex
    )
{
    return TestGraphBit(VisitedVertices, Vertex + 1);
}

#define SetGraphBit(Name, BitNumber) \
    BitTestAndSet64((PLONGLONG)Graph->##Name##.Buffer, (LONGLONG)BitNumber)

FORCEINLINE
VOID
RegisterEdgeDeletion(
    _In_ PGRAPH Graph,
    _In_ EDGE Edge
    )
{
    //
    // We add 1 to the edge to account for the fact that they're 0-based, but
    // the bitmaps are 1-based.
    //

    SetGraphBit(DeletedEdges, Edge + 1);
    Graph->DeletedEdgeCount++;
    ASSERT(Graph->DeletedEdgeCount <= Graph->TotalNumberOfEdges);
}

FORCEINLINE
VOID
RegisterVertexVisit(
    _In_ PGRAPH Graph,
    _In_ VERTEX Vertex
    )
{
    //
    // We add 1 to the vertex to account for the fact that they're 0-based, but
    // the bitmaps are 1-based.
    //

    SetGraphBit(VisitedVertices, Vertex + 1);
    Graph->VisitedVerticesCount++;
    ASSERT(Graph->VisitedVerticesCount <= Graph->NumberOfVertices);
}

FORCEINLINE
BOOLEAN
GraphCheckEdge(
    PGRAPH Graph,
    EDGE Edge,
    VERTEX Vertex1,
    VERTEX Vertex2
    )
{
    EDGE Edge1;
    EDGE Edge2;

    Edge1 = AbsoluteEdge(Graph, Edge, 0);
    Edge2 = AbsoluteEdge(Graph, Edge, 1);

    if (Graph->Edges[Edge1] == Vertex1 && Graph->Edges[Edge2] == Vertex2) {
        return TRUE;
    }

    if (Graph->Edges[Edge1] == Vertex2 && Graph->Edges[Edge2] == Vertex1) {
        return TRUE;
    }

    return FALSE;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
