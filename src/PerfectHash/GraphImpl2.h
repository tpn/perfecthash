/*++

Copyright (c) 2018-2023 Trent Nelson <trent@trent.me>

Module Name:

    GraphImpl2.h

Abstract:

    This is the private header file for our second implementation of the graph
    module for the perfect hash library.  This implementation differs from our
    first attempt in GraphImpl1.c in the graph assignment step.

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
// Inline function helpers.
//

FORCEINLINE
BOOLEAN
IsVisitedVertex2(
    _In_ PGRAPH Graph,
    _In_ VERTEX Vertex
    )
{
    return TestGraphBit(VisitedVerticesBitmap, Vertex);
}

FORCEINLINE
VOID
RegisterVertexVisit2(
    _In_ PGRAPH Graph,
    _In_ VERTEX Vertex
    )
{
    SetGraphBit(VisitedVerticesBitmap, Vertex);
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
