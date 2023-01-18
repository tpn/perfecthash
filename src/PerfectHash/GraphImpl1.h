/*++

Copyright (c) 2018-2023 Trent Nelson <trent@trent.me>

Module Name:

    GraphImpl1.h

Abstract:

    This is the private header file for our first implementation of the graph
    module for the perfect hash library.  This implementation attempts to be
    as close to the original CHM implementation as possible.

--*/

#include "stdafx.h"

//
// Define a graph iterator structure used to facilitate graph traversal.
//

typedef struct _GRAPH_ITERATOR {
    VERTEX Vertex;
    EDGE Edge;
} GRAPH_ITERATOR;
typedef GRAPH_ITERATOR *PGRAPH_ITERATOR;

//
// Define helper macros.
//

#define EMPTY ((VERTEX)-1)
#define GRAPH_NO_NEIGHBOR ((VERTEX)-1)

#define IsEmpty(Value) ((ULONG)Value == EMPTY)
#define IsNeighborEmpty(Neighbor) ((ULONG)Neighbor == EMPTY)

//
// When a solution has been found and the assignment step begins, the initial
// value assigned to a vertex is govered by the following macro.
//

#define INITIAL_ASSIGNMENT_VALUE 0

//
// Define function types specific to our implementation.
//

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
    _In_ EDGE Edge
    );
typedef GRAPH_TRAVERSE *PGRAPH_TRAVERSE;
extern GRAPH_TRAVERSE GraphTraverseRecursive;

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

    //
    // Re-enable this if modulus ever works again.
    //
#if 0
    if (IsModulusMasking(Graph->MaskFunctionId)) {
        MaskedEdge = Edge % Graph->EdgeModulus;
    } else {
        MaskedEdge = Edge & Graph->EdgeMask;
    }
#else
    MaskedEdge = Edge & Graph->EdgeMask;
#endif

    AbsEdge = (MaskedEdge + (Index * Graph->NumberOfEdges));
    return AbsEdge;
}

FORCEINLINE
BOOLEAN
IsDeletedEdge(
    _In_ PGRAPH Graph,
    _In_ EDGE Edge
    )
{
    return TestGraphBit(DeletedEdgesBitmap, Edge);
}

FORCEINLINE
BOOLEAN
IsVisitedVertex(
    _In_ PGRAPH Graph,
    _In_ VERTEX Vertex
    )
{
    return TestGraphBit(VisitedVerticesBitmap, Vertex);
}

#ifdef _DEBUG

FORCEINLINE
VOID
RegisterEdgeDeletion(
    _In_ PGRAPH Graph,
    _In_ EDGE Edge
    )
{
    LONG OrderIndex;
    ASSERT(!TestGraphBit(DeletedEdgesBitmap, Edge));
    SetGraphBit(DeletedEdgesBitmap, Edge);
    Graph->DeletedEdgeCount++;
    ASSERT(Graph->DeletedEdgeCount <= Graph->TotalNumberOfEdges);
    OrderIndex = --Graph->OrderIndex;
    ASSERT(OrderIndex >= 0);
    Graph->Order[OrderIndex] = Edge;
}

#else

FORCEINLINE
VOID
RegisterEdgeDeletion(
    _In_ PGRAPH Graph,
    _In_ EDGE Edge
    )
{
    ULONG OrderIndex;
    SetGraphBit(DeletedEdgesBitmap, Edge);
    Graph->DeletedEdgeCount++;
    OrderIndex = --Graph->OrderIndex;
    Graph->Order[OrderIndex] = Edge;
}

#endif

FORCEINLINE
VOID
RegisterVertexVisit(
    _In_ PGRAPH Graph,
    _In_ VERTEX Vertex
    )
{
    SetGraphBit(VisitedVerticesBitmap, Vertex);
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
