/*++

Copyright (c) 2020 Trent Nelson <trent@trent.me>

Module Name:

    GraphImpl2.c

Abstract:

    This module is our second implementation of graph innards for the CHM
    algorithm of the perfect hash library.  It implements GraphAssign2(),
    which is a more efficient implementation of the graph assignment step
    present in GraphImpl1.c.

--*/

#include "stdafx.h"
#include "GraphImpl2.h"
#include "PerfectHashEventsPrivate.h"

GRAPH_ASSIGN GraphAssign2;

_Use_decl_annotations_
VOID
GraphAssign2(
    PGRAPH Graph
    )
/*++

Routine Description:

    This routine is called after a graph has determined to be acyclic.  It is
    responsible for walking the graph and assigning values to edges in order to
    complete the perfect hash solution.

Arguments:

    Graph - Supplies a pointer to the graph to operate on.

Return Value:

    None.

--*/
{
    EDGE Edge1;
    EDGE Edge2;
    ULONG Index;
    ULONG Order;
    ULONG Assigned1;
    ULONG Assigned2;
    ULONG IndexMask;
    VERTEX Vertex1;
    VERTEX Vertex2;
    ULONG NumberOfKeys;
    ULONG NumberOfEdges;
    PPERFECT_HASH_TABLE Table;

    DECL_GRAPH_COUNTER_LOCAL_VARS();

    //
    // Initialize aliases.
    //

    Table = Graph->Context->Table;
    IndexMask = Table->IndexMask;
    NumberOfKeys = Graph->NumberOfKeys;
    NumberOfEdges = Graph->NumberOfEdges;

    //
    // Invariant check: we should only be called on graphs that have already
    // been determined to be invariant.
    //

    ASSERT(Graph->Flags.IsAcyclic);

    START_GRAPH_COUNTER();

    //
    // Walk the graph and assign values.
    //

    for (Index = 0; Index < NumberOfKeys; Index++) {

        //
        // Obtain the deletion order for the edge at this index.
        //

        Order = Graph->Order[Index];

        //
        // Resolve the edges.
        //

        Edge1 = Order;
        Edge2 = Order + Graph->NumberOfEdges;

        //
        // Obtain the vertices.
        //

        Vertex1 = Graph->Edges[Edge1];
        Vertex2 = Graph->Edges[Edge2];

        //
        // If the first vertex hasn't been visited, construct an appropriate
        // assignment value based on the second vertex's assignment value.
        // Otherwise, perform the reverse.  The final result of this operation
        // will allow the index to be reconstructed by adding the two assigned
        // values together for each vertex, e.g.:
        //
        //      Index = (
        //          Graph->Assigned[Vertex1] +
        //          Graph->Assigned[Vertex2]
        //      ) & IndexMask;
        //

        if (!IsVisitedVertex(Graph, Vertex1)) {
            Assigned2 = Graph->Assigned[Vertex2];
            Assigned1 = (((NumberOfEdges + Order) - Assigned2) & IndexMask);
            Graph->Assigned[Vertex1] = Assigned1;
        } else {
            Assigned1 = Graph->Assigned[Vertex1];
            Assigned2 = (((NumberOfEdges + Order) - Assigned1) & IndexMask);
            Graph->Assigned[Vertex2] = Assigned2;
        }

        //
        // Set both vertices as visited.
        //

        RegisterVertexVisit(Graph, Vertex1);
        RegisterVertexVisit(Graph, Vertex2);
    }

    STOP_GRAPH_COUNTER(Assign);

    EventWriteGraphAssignResult(
        NULL,
        Graph->Attempt,
        Table->GraphImpl,
        Cycles,
        Microseconds,
        Graph->NumberOfKeys,
        Graph->NumberOfVertices
    );

    return;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
