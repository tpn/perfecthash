/*++

Copyright (c) 2018-2020 Trent Nelson <trent@trent.me>

Module Name:

    GraphImpl1.c

Abstract:

    This module is our first implementation of the graph module for
    the perfect hash library, which attempts to be as close to the
    original CHM implementation as possible.

    It implements the three main required graph functions: GraphAddEdge(),
    IsGraphAcyclic() and GraphAssign().

    These three functions collectively call the following implementation
    specific functions also defined in this file: GraphCyclicDeleteEdge(),
    GraphFindDegree1Edge(), GraphNeighborsIterator(), GraphNextNeighbor(),
    GraphEdgeId(), GraphDeleteEdge(), GraphCheckEdge(), and
    GraphTraverseRecursive().

--*/

#include "stdafx.h"
#include "GraphImpl1.h"
#include "PerfectHashEventsPrivate.h"

//
// Forward decls.
//

GRAPH_CYCLIC_DELETE_EDGE GraphCyclicDeleteEdge;
GRAPH_FIND_DEGREE1_EDGE GraphFindDegree1Edge;
GRAPH_NEIGHBORS_ITERATOR GraphNeighborsIterator;
GRAPH_NEXT_NEIGHBOR GraphNextNeighbor;
GRAPH_EDGE_ID GraphEdgeId;
GRAPH_DELETE_EDGE GraphDeleteEdge;
GRAPH_CHECK_EDGE GraphCheckEdge;
GRAPH_TRAVERSE GraphTraverseRecursive;

#define IsEmpty(Value) ((ULONG)Value == EMPTY)
#define IsNeighborEmpty(Neighbor) ((ULONG)Neighbor == EMPTY)

//
// When a solution has been found and the assignment step begins, the initial
// value assigned to a vertex is govered by the following macro.
//

#define INITIAL_ASSIGNMENT_VALUE 0

//
// The algorithm is as follows:
//
//  For each key:
//      Generate unique hash 1 (h1/v1) and hash 2 (h2/v2)
//      Add edge to graph for h1<->h2
//  Determine if graph is cyclic.  If so, restart.
//  If not, we've found a solution; perform assignment and finish up.
//


GRAPH_ADD_EDGE GraphAddEdge;

_Use_decl_annotations_
VOID
GraphAddEdge(
    PGRAPH Graph,
    EDGE Edge,
    VERTEX Vertex1,
    VERTEX Vertex2
    )
/*++

Routine Description:

    This routine adds an edge to the hypergraph for two vertices.

Arguments:

    Graph - Supplies a pointer to the graph for which the edge is to be added.

    Edge - Supplies the edge to add to the graph.

    Vertex1 - Supplies the first vertex.

    Vertex2 - Supplies the second vertex.

Return Value:

    None.

--*/
{
    EDGE Edge1;
    EDGE Edge2;
    EDGE First1;
    EDGE First2;

    Edge1 = Edge;
    Edge2 = Edge1 + Graph->NumberOfEdges;

#ifdef _DEBUG
    //
    // Invariant checks:
    //
    //      - Vertex1 should be less than the number of vertices.
    //      - Vertex2 should be less than the number of vertices.
    //      - Edge1 should be less than the number of edges.
    //      - The graph must not have started deletions.
    //

    ASSERT(Vertex1 < Graph->NumberOfVertices);
    ASSERT(Vertex2 < Graph->NumberOfVertices);
    ASSERT(Edge1 < Graph->NumberOfEdges);
    ASSERT(!Graph->Flags.Shrinking);
#endif

    //
    // Insert the first edge.
    //

    First1 = Graph->First[Vertex1];
    Graph->Next[Edge1] = First1;
    Graph->First[Vertex1] = Edge1;
    Graph->Edges[Edge1] = Vertex2;

    //
    // Insert the second edge.
    //

    First2 = Graph->First[Vertex2];
    Graph->Next[Edge2] = First2;
    Graph->First[Vertex2] = Edge2;
    Graph->Edges[Edge2] = Vertex1;

}


IS_GRAPH_ACYCLIC IsGraphAcyclic;

_Use_decl_annotations_
BOOLEAN
IsGraphAcyclic(
    PGRAPH Graph
    )
/*++

Routine Description:

    This routine determines whether or not the graph is acyclic.  An acyclic
    graph is one where, after deletion of all edges in the graph with vertices
    of degree 1, no edges remain.

Arguments:

    Graph - Supplies a pointer to the graph to operate on.

Return Value:

    TRUE if the graph is acyclic, FALSE if it's cyclic.

--*/
{
    EDGE Edge;
    VERTEX Vertex;
    BOOLEAN IsAcyclic;
    BOOLEAN IsAcyclicSlow;
    ULONG NumberOfKeys;
    ULONG NumberOfEdges;
    ULONG NumberOfVertices;
    ULONG NumberOfEdgesDeleted;
    PRTL_NUMBER_OF_SET_BITS RtlNumberOfSetBits;

    DECL_GRAPH_COUNTER_LOCAL_VARS();

    START_GRAPH_COUNTER();

    //
    // Resolve aliases.
    //

    NumberOfKeys = Graph->NumberOfKeys;
    NumberOfEdges = Graph->NumberOfEdges;
    NumberOfVertices = Graph->NumberOfVertices;

    //
    // Invariant check: we should not be shrinking prior to this point, and our
    // deleted edge count should be 0.
    //

    ASSERT(!Graph->Flags.Shrinking);
    ASSERT(Graph->DeletedEdgeCount == 0);

    //
    // Toggle the shrinking bit to indicate we've started edge deletion.
    //

    Graph->Flags.Shrinking = TRUE;

    //
    // Enumerate through all vertices in the graph and attempt to delete those
    // connected by edges that have degree 1.
    //

    for (Vertex = 0; Vertex < NumberOfVertices; Vertex++) {
        GraphCyclicDeleteEdge(Graph, Vertex);
    }

    ASSERT(Graph->OrderIndex >= 0);

    if (!IsGraphParanoid(Graph)) {

        NumberOfEdgesDeleted = Graph->DeletedEdgeCount;
        IsAcyclic = (NumberOfKeys == NumberOfEdgesDeleted);

    } else {

        RtlNumberOfSetBits = Graph->Context->Rtl->RtlNumberOfSetBits;

        //
        // As each edge of degree 1 is deleted, a bit is set in the deleted
        // bitmap, indicating the edge at that bit offset was deleted.  Thus,
        // we can simply count the number of set bits in the bitmap and compare
        // that to the number of edges in the graph.  If the values do not
        // match, the graph is cyclic; if they do match, the graph is acyclic.
        //

        NumberOfEdgesDeleted = RtlNumberOfSetBits(&Graph->DeletedEdgesBitmap);

        //
        // Ensure number of bits set matches the graph deleted edge count.
        //

        ASSERT(NumberOfEdgesDeleted == Graph->DeletedEdgeCount);

        IsAcyclic = (NumberOfKeys == NumberOfEdgesDeleted);

        //
        // Verify our assumption about counting bits is correct.
        //

        IsAcyclicSlow = TRUE;

        for (Edge = 0; Edge < NumberOfKeys; Edge++) {
            if (!IsDeletedEdge(Graph, Edge)) {
                IsAcyclicSlow = FALSE;
                break;
            }
        }

        ASSERT(IsAcyclic == IsAcyclicSlow);

    }

    //
    // Invariant check: if we're acyclic, the order index should be zero (as all
    // keys were deleted once), greater than zero otherwise.
    //

    if (IsAcyclic) {
        ASSERT(Graph->OrderIndex == 0);
    } else {
        ASSERT(Graph->OrderIndex > 0);
    }

    //
    // Make a note that we're acyclic if applicable in the graph's flags.
    // This is checked by GraphAssign() to ensure we only operate on acyclic
    // graphs.
    //

    if (IsAcyclic) {

        Graph->Flags.IsAcyclic = TRUE;

    } else {

        ULONG HighestDeletedEdges;
        PPERFECT_HASH_CONTEXT Context;

        Context = Graph->Info->Context;

        if (NumberOfEdgesDeleted > Context->HighestDeletedEdgesCount) {

            //
            // Register as the highest deleted edges count if applicable.
            //

            while (TRUE) {

                HighestDeletedEdges = Context->HighestDeletedEdgesCount;

                if (NumberOfEdgesDeleted <= HighestDeletedEdges) {
                    break;
                }

                InterlockedCompareExchange(
                    (PLONG)&Context->HighestDeletedEdgesCount,
                    NumberOfEdgesDeleted,
                    HighestDeletedEdges
                );

            }
        }
    }

    STOP_GRAPH_COUNTER(IsAcyclic);

    //EVENT_WRITE_GRAPH_IS_ACYCLIC();

    return IsAcyclic;
}

GRAPH_IS_ACYCLIC GraphIsAcyclic;

_Use_decl_annotations_
HRESULT
GraphIsAcyclic(
    PGRAPH Graph
    )
{
    BOOLEAN Success;

    Success = IsGraphAcyclic(Graph);
    return (Success ? S_OK : PH_E_GRAPH_CYCLIC_FAILURE);
}

GRAPH_ASSIGN GraphAssign;

_Use_decl_annotations_
HRESULT
GraphAssign(
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

    S_OK.

--*/
{
    PRTL Rtl;
    VERTEX Vertex;
    ULONG NumberOfSetBits;

    DECL_GRAPH_COUNTER_LOCAL_VARS();

    //
    // Invariant check: the acyclic flag should be set.  (Indicating that
    // IsGraphAcyclic() successfully determined that, yes, the graph is
    // acyclic.)
    //

    ASSERT(Graph->Flags.IsAcyclic);

    EVENT_WRITE_GRAPH_ASSIGN_START();

    //
    // Walk the graph and assign values.
    //

    START_GRAPH_COUNTER();

    for (Vertex = 0; Vertex < Graph->NumberOfVertices; Vertex++) {

        if (!IsGraphParanoid(Graph)) {
            if (IsEmpty(Graph->First[Vertex])) {
                Graph->NumberOfEmptyVertices++;
                continue;
            }
        }

        if (!IsVisitedVertex(Graph, Vertex)) {

            //
            // This is a "root" vertex that we'll perform a depth-first search
            // on as part of assignment.  Sanity check the assigned value at
            // this location matches our initial value, then traverse the graph.
            //

            ASSERT(Graph->Assigned[Vertex] == INITIAL_ASSIGNMENT_VALUE);
            GraphTraverseRecursive(Graph, Vertex);
        }
    }

    if (IsGraphParanoid(Graph)) {
        Rtl = Graph->Context->Rtl;
        NumberOfSetBits = Rtl->RtlNumberOfSetBits(&Graph->VisitedVerticesBitmap);

        ASSERT(Graph->VisitedVerticesCount == NumberOfSetBits);
        ASSERT(Graph->VisitedVerticesCount == Graph->NumberOfVertices);
    }

    STOP_GRAPH_COUNTER(Assign);

    EVENT_WRITE_GRAPH_ASSIGN_STOP();

    EVENT_WRITE_GRAPH_ASSIGN_RESULT();

    return S_OK;
}

//
// The following methods are internal methods specific to this implementation.
//

GRAPH_FIND_DEGREE1_EDGE GraphFindDegree1Edge;

_Use_decl_annotations_
BOOLEAN
GraphFindDegree1Edge(
    PGRAPH Graph,
    VERTEX Vertex,
    PEDGE EdgePointer
    )
/*++

Routine Description:

    This routine determines if a vertex has degree 1 within the graph, and if
    so, returns the edge associated with it.

Arguments:

    Graph - Supplies a pointer to the graph.

    Vertex - Supplies the vertex for which the degree 1 test is made.

    EdgePointer - Supplies the address of a variable that receives the EDGE
        owning this vertex if it degree 1.

Return Value:

    TRUE if the vertex has degree 1, FALSE otherwise.  EdgePointer will be
    updated if TRUE is returned.

    N.B. Actually, in the CHM implementation, they seem to update the edge
         regardless if it was a degree 1 connection.  I guess we should mirror
         that behavior now too.

--*/
{
    EDGE Edge;
    EDGE AbsEdge;
    BOOLEAN Found = FALSE;

    //
    // Get the edge for this vertex.
    //

    Edge = Graph->First[Vertex];

    //
    // If edge is empty, we're done.
    //

    if (IsEmpty(Edge)) {
        return FALSE;
    }

    AbsEdge = AbsoluteEdge(Graph, Edge, 0);

    //
    // AbsEdge should always be less than or equal to Edge here.
    //

    ASSERT(AbsEdge <= Edge);

    //
    // If the edge has not been deleted, capture it.
    //

    if (!IsDeletedEdge(Graph, AbsEdge)) {
        Found = TRUE;
        *EdgePointer = Edge;
    }

    //
    // Determine if this is a degree 1 connection.
    //

    while (TRUE) {

        //
        // Load the next edge.
        //

        Edge = Graph->Next[Edge];

        if (IsEmpty(Edge)) {
            break;
        }

        //
        // Obtain the absolute edge for this edge.
        //

        AbsEdge = AbsoluteEdge(Graph, Edge, 0);

        //
        // If we've already deleted this edge, we can skip it and look at the
        // next edge in the graph.
        //

        if (IsDeletedEdge(Graph, AbsEdge)) {
            continue;
        }

        if (Found) {

            //
            // If we've already found an edge by this point, we're not 1 degree.
            //

            return FALSE;
        }

        //
        // We've found the first edge.
        //

        *EdgePointer = Edge;
        Found = TRUE;
    }

    return Found;
}


GRAPH_CYCLIC_DELETE_EDGE GraphCyclicDeleteEdge;

_Use_decl_annotations_
VOID
GraphCyclicDeleteEdge(
    PGRAPH Graph,
    VERTEX Vertex
    )
/*++

Routine Description:

    This routine deletes edges from a graph connected by vertices of degree 1.

Arguments:

    Graph - Supplies a pointer to the graph for which the edge is to be deleted.

    Vertex - Supplies the vertex for which the initial edge is obtained.

Return Value:

    None.

    N.B. If an edge is deleted, its corresponding bit will be set in the bitmap
         Graph->DeletedEdges.

--*/
{
    EDGE Edge = 0;
    EDGE PrevEdge;
    EDGE AbsEdge;
    VERTEX Vertex1;
    VERTEX Vertex2;
    BOOLEAN IsDegree1;

    //
    // Determine if the vertex has a degree of 1, and if so, obtain the edge.
    //

    IsDegree1 = GraphFindDegree1Edge(Graph, Vertex, &Edge);

    //
    // If this isn't a degree 1 edge, there's nothing left to do.
    //

    if (!IsDegree1) {
        return;
    }

    //
    // We've found an edge of degree 1 to delete.
    //

    Vertex1 = Vertex;
    Vertex2 = 0;

    while (TRUE) {

        //
        // Obtain the absolute edge and register it as deleted.
        //

        AbsEdge = AbsoluteEdge(Graph, Edge, 0);

        //
        // Invariant check: Edge should always be greater than or
        // equal to AbsEdge here.
        //

        ASSERT(Edge >= AbsEdge);

        //
        // Invariant check: AbsEdge should not have been deleted yet.
        //

        ASSERT(!IsDeletedEdge(Graph, AbsEdge));

        //
        // Register the deletion of this edge.
        //

        RegisterEdgeDeletion(Graph, AbsEdge);

        //
        // Find the other vertex the edge is connecting.
        //

        Vertex2 = Graph->Edges[AbsEdge];

        if (Vertex2 == Vertex1) {

            //
            // We had the first vertex; get the second one.
            //

            AbsEdge = AbsoluteEdge(Graph, Edge, 1);
            Vertex2 = Graph->Edges[AbsEdge];
        }

        //
        // If the second vertex is empty, break.
        //

        if (IsEmpty(Vertex2)) {
            break;
        }

        //
        // Stash a copy of the current edge before it gets potentially
        // mutated by GraphFindDegree1Edge().
        //

        PrevEdge = Edge;

        //
        // Determine if the other vertex is degree 1.
        //

        IsDegree1 = GraphFindDegree1Edge(Graph, Vertex2, &Edge);

        if (!IsDegree1) {

            //
            // Other vertex isn't degree 1, we can stop the search.
            //

            break;
        }

        //
        // Invariant check: Edge should have been modified by
        // GraphFindDegree1Edge().
        //

        ASSERT(PrevEdge != Edge);

        //
        // This vertex is also degree 1, so continue the deletion.
        //

        Vertex1 = Vertex2;
    }
}


GRAPH_NEIGHBORS_ITERATOR GraphNeighborsIterator;

_Use_decl_annotations_
GRAPH_ITERATOR
GraphNeighborsIterator(
    PGRAPH Graph,
    VERTEX Vertex
    )
/*++

Routine Description:

    For a given vertex in graph, create an iterator such that the neighboring
    vertices can be iterated over.

Arguments:

    Graph - Supplies a pointer to the graph to operate on.

    Vertex - Supplies the vertex for which the iterator will be initialized.

Return Value:

    An instance of a GRAPH_ITERATOR with the Vertex member set to the Vertex
    parameter, and the Edge member set to the first edge in the graph for the
    given vertex.

--*/
{
    GRAPH_ITERATOR Iterator;

    Iterator.Vertex = Vertex;
    Iterator.Edge = Graph->First[Vertex];

    return Iterator;
}


GRAPH_NEXT_NEIGHBOR GraphNextNeighbor;

_Use_decl_annotations_
VERTEX
GraphNextNeighbor(
    PGRAPH Graph,
    PGRAPH_ITERATOR Iterator
    )
/*++

Routine Description:

    Return the next vertex for a given graph iterator.

Arguments:

    Graph - Supplies a pointer to the graph to operate on.

    Iterator - Supplies a pointer to the graph iterator structure to use.

Return Value:

    The neighboring vertex, or GRAPH_NO_NEIGHBOR if no vertices remain.

--*/
{
    EDGE Edge;
    VERTEX Vertex;
    VERTEX Neighbor;

    //
    // If the edge is empty, the graph iteration has finished.
    //

    Edge = Iterator->Edge;

    if (IsEmpty(Edge)) {
        return GRAPH_NO_NEIGHBOR;
    }

    //
    // Find the vertex for this edge.
    //

    Vertex = Graph->Edges[Edge];

    //
    // If the vertex matches the one in our iterator, the edge we've been
    // provided is the first edge.  Otherwise, it's the second edge.
    //

    if (Vertex == Iterator->Vertex) {

        ASSERT(Edge < Graph->NumberOfEdges);
        Neighbor = Graph->Edges[Edge + Graph->NumberOfEdges];

    } else {

        Neighbor = Vertex;
    }

    //
    // Update the edge and return the neighbor.
    //

    Iterator->Edge = Graph->Next[Edge];

    return Neighbor;
}


GRAPH_EDGE_ID GraphEdgeId;

_Use_decl_annotations_
EDGE
GraphEdgeId(
    PGRAPH Graph,
    VERTEX Vertex1,
    VERTEX Vertex2
    )
/*++

Routine Description:

    Generates an ID for two vertices as part of the assignment step.

Arguments:

    Graph - Supplies a pointer to the graph for which the edge is to be added.

    Vertex1 - Supplies the first vertex.

    Vertex2 - Supplies the second vertex.

Return Value:

    An EDGE value.

--*/
{
    EDGE Edge;
    EDGE EdgeId;
    ULONG Iterations = 0;

    //
    // Obtain the first edge for this vertex from the Graph->First array.
    // Subsequent edges are obtained from the Graph->Next array.
    //

    Edge = Graph->First[Vertex1];

    ASSERT(!IsEmpty(Edge));

    //
    // Find the first ID of the edge where the first part contains vertex 1 and
    // the second part contains vertex 2.  This is achieved via the check edge
    // call.  If this returns TRUE, the resulting absolute edge is our ID.
    //

    if (GraphCheckEdge(Graph, Edge, Vertex1, Vertex2)) {

        //
        // XXX: in the chm.c implementation, they call abs_edge() here.
        // However, if I do that, I trigger endless collisions during the
        // verification stage.  Reverting to just returning the edge appears
        // to work, though.
        //

        //
        // EdgeId = AbsoluteEdge(Graph, Edge, 0);
        //

        EdgeId = Edge;

    } else {

        //
        // Continue looking for an edge in the graph that satisfies the edge
        // check condition.  Track the number of iterations for debugging
        // purposes.
        //

        do {

            Iterations++;
            Edge = Graph->Next[Edge];
            ASSERT(!IsEmpty(Edge));

        } while (!GraphCheckEdge(Graph, Edge, Vertex1, Vertex2));

        //
        // Ditto for here (see comment above).
        //

        //
        // EdgeId = AbsoluteEdge(Graph, Edge, 0);
        //

        EdgeId = Edge;
    }

    return EdgeId;
}


GRAPH_TRAVERSE GraphTraverseRecursive;

_Use_decl_annotations_
VOID
GraphTraverseRecursive(
    PGRAPH Graph,
    VERTEX Vertex
    )
/*++

Routine Description:

    This routine is called as part of graph assignment.  It is responsible for
    doing a depth-first traversal of the graph and obtaining edge IDs that can
    be saved in the Graph->Assigned array.

Arguments:

    Graph - Supplies a pointer to the graph to operate on.

    Vertex - Supplies the vertex to traverse.

Return Value:

    None.

--*/
{
    PRTL Rtl;
    ULONG Bit;
    LONG EdgeId;
    LONG ThisId;
    LONG FinalId;
    ULONG IndexMask;
    ULONG MaskedEdgeId;
    ULONG MaskedThisId;
    ULONG MaskedFinalId;
    LONG ExistingId;
    LONG OriginalExistingId;
    VERTEX Neighbor;
    GRAPH_ITERATOR Iterator;
    PPERFECT_HASH_TABLE Table;

    //
    // Initialize aliases.
    //

    Rtl = Graph->Context->Rtl;
    Table = Graph->Context->Table;
    IndexMask = Table->IndexMask;

    //
    // Invariant check: vertex should not be empty.
    //

    ASSERT(!IsEmpty(Vertex));

    //
    // Register the vertex as visited.
    //

    RegisterVertexVisit(Graph, Vertex);

    //
    // Initialize a graph iterator for visiting neighbors.
    //

    Iterator = GraphNeighborsIterator(Graph, Vertex);

    //
    // Update the total traversals counter and current traversal depth.  If
    // the current depth is the deepest we've seen so far, update the maximum.
    //

    Graph->TotalTraversals++;
    Graph->TraversalDepth++;
    if (Graph->TraversalDepth > Graph->MaximumTraversalDepth) {
        Graph->MaximumTraversalDepth = Graph->TraversalDepth;
    }

    //
    // N.B. This routine has been especially problematic.  In conjunction
    //      with what appeared to be a faulty GraphEdgeId() implementation,
    //      the algorithm was just not generating solutions that passed
    //      validation when written as per the chm.c implementation.  So,
    //      the current logic is a little overly-defensive, however, it
    //      does work (and passes validation and then separate testing),
    //      so, eh.
    //

    while (TRUE) {

        Neighbor = GraphNextNeighbor(Graph, &Iterator);

        if (IsNeighborEmpty(Neighbor)) {
            break;
        }

        //
        // If the neighbor has already been visited, skip it.
        //

        if (IsVisitedVertex(Graph, Neighbor)) {
            continue;
        }

        //
        // Construct the unique ID for this particular visit.  We break it out
        // into three distinct steps in order to assist with debugging.
        //

        EdgeId = GraphEdgeId(Graph, Vertex, Neighbor);
        MaskedEdgeId = EdgeId & IndexMask;

        OriginalExistingId = ExistingId = Graph->Assigned[Vertex];
        ASSERT(ExistingId >= 0);

        ThisId = EdgeId - ExistingId;
        MaskedThisId = ThisId & IndexMask;
        ASSERT(MaskedThisId <= Graph->NumberOfVertices);

        FinalId = EdgeId + ExistingId;
        MaskedFinalId = FinalId & IndexMask;
        ASSERT(MaskedFinalId <= Graph->NumberOfVertices);

        Bit = MaskedFinalId;

        if (Bit >= Graph->NumberOfVertices) {

            //
            // Invariant check: this should never be hit.
            //

            PH_RAISE(PH_E_UNREACHABLE_CODE);
        }

        if (TestGraphBit(IndexBitmap, Bit)) {

            Graph->Collisions++;

        } else {

            SetGraphBit(IndexBitmap, Bit);

        }

        Graph->Assigned[Neighbor] = (ULONG)MaskedThisId;

        //
        // Recursively traverse the neighbor.
        //

        GraphTraverseRecursive(Graph, Neighbor);

    }

    //
    // Decrement depth and return.
    //

    Graph->TraversalDepth--;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
