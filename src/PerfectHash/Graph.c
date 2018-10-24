/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Graph.c

Abstract:

    This module implements generic graph functionality.

--*/

#include "stdafx.h"

//
// Temporary typedefs for new memory coverage logic.
//

typedef
VOID
(NTAPI GRAPH_CALCULATE_MEMORY_COVERAGE)(
    _In_ PGRAPH Graph
    );
typedef GRAPH_CALCULATE_MEMORY_COVERAGE *PGRAPH_CALCULATE_MEMORY_COVERAGE;

#ifdef _M_X64
extern GRAPH_CALCULATE_MEMORY_COVERAGE GraphCalculateMemoryCoverageAvx2;
#define GraphCalculateMemoryCoverage GraphCalculateMemoryCoverageAvx2
#else
#error Non-AVX2 GraphCalculateMemoryCoverage not yet implemented.
#endif


typedef
VOID
(NTAPI GRAPH_REGISTER_SOLVED)(
    _In_ PGRAPH Graph
    );
typedef GRAPH_REGISTER_SOLVED *PGRAPH_REGISTER_SOLVED;

extern GRAPH_REGISTER_SOLVED GraphRegisterSolved;

//
// COM scaffolding routines for initialization and rundown.
//

GRAPH_INITIALIZE GraphInitialize;

_Use_decl_annotations_
HRESULT
GraphInitialize(
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

GRAPH_RUNDOWN GraphRundown;

_Use_decl_annotations_
VOID
GraphRundown(
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
    // Todo: release individual buffers.
    //

    //
    // Release applicable COM references.
    //

    RELEASE(Graph->Rtl);
    RELEASE(Graph->Allocator);

    return;
}

//
// Private methods.
//

#define IsEmpty(Value) ((ULONG)Value == EMPTY)
#define IsNeighborEmpty(Neighbor) ((ULONG)Neighbor == EMPTY)

#if 0

INITIALIZE_GRAPH InitializeGraphObsolete;

_Use_decl_annotations_
VOID
InitializeGraphObsolete(
    PGRAPH_INFO Info,
    PGRAPH Graph
    )
/*++

    WIP - this routine is in the process of being made obsolete.

Routine Description:

    Initialize a graph structure given the provided information.  This routine
    is called at the top of each worker thread's infinite loop around the solve
    graph function.  It is responsible for resetting the block of memory for the
    graph back to its initial state, including generation of new random seed
    values that can be used by the hash function when generating vertices.

Arguments:

    Info - Supplies a pointer to the graph info to use for initialization.

    Graph - Supplies a pointer to the graph to be initialized.

Return Value:

    None.

--*/
{
    UNREFERENCED_PARAMETER(Info);
    UNREFERENCED_PARAMETER(Graph);

    PH_RAISE(PH_E_UNREACHABLE_CODE);

#if 0

    PRTL Rtl;
    ULONG Index;
    PCHAR Buffer;
    PCHAR ExpectedBuffer;
    USHORT BitmapCount = 0;
    PPERFECT_HASH_TABLE Table;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;
    PPERFECT_HASH_CONTEXT Context;

    PH_RAISE(PH_E_UNREACHABLE_CODE);

    //
    // Initialize aliases.
    //

    Context = Info->Context;
    Table = Context->Table;
    TableInfoOnDisk = Table->TableInfoOnDisk;
    Rtl = Context->Rtl;

    Graph->Attempt = InterlockedIncrement64(&Context->Attempts);

    //
    // If this attempt has exceeded our threshold, inform our parent.
    //

    if (Graph->Attempt == Context->ResizeTableThreshold) {

        if (!SetEvent(Context->TryLargerTableSizeEvent)) {
            SYS_ERROR(SetEvent);
        }

    }

    //
    // Obtain new seed data for the first two seeds and initialize the number
    // of seeds.
    //

    GetRandomSeedsBlocking(&Graph->Seeds12);
    Graph->NumberOfSeeds = Table->TableInfoOnDisk->NumberOfSeeds;

    //
    // Initialize the number of keys.
    //

    Graph->NumberOfKeys = Table->Keys->NumberOfElements.LowPart;

    //
    // Carve out the backing memory structures for arrays and bitmap buffers.
    // Use the PBYTE Buffer here to make pointer arithmetic a tad easier.  We
    // initialize it to the start of the memory immediately following the graph
    // structure, and then bump it each time we carve out an array or bitmap
    // buffer.
    //

    ASSERT(sizeof(*Graph) == Info->SizeOfGraphStruct);
    Buffer = RtlOffsetToPointer(Graph, ALIGN_UP_YMMWORD(sizeof(*Graph)));

    //
    // Carve out the Graph->Edges array.
    //

    Graph->Edges = (PEDGE)Buffer;
    Buffer += Info->EdgesSizeInBytes;

    //
    // Carve out the Graph->Next array.
    //

    Graph->Next = (PEDGE)Buffer;
    Buffer += Info->NextSizeInBytes;

    //
    // Carve out the Graph->First array.
    //

    Graph->First = (PVERTEX)Buffer;
    Buffer += Info->FirstSizeInBytes;

    //
    // Carve out the Graph->Prev array.
    //

    Graph->Prev = (PVERTEX)Buffer;
    Buffer += Info->PrevSizeInBytes;

    //
    // Carve out the Graph->Assigned array.
    //

    Graph->Assigned = (PVERTEX)Buffer;
    Buffer += Info->AssignedSizeInBytes;

    //
    // Carve out the Graph->Values array.
    //

    Graph->Values = (PVERTEX)Buffer;
    Buffer += Info->ValuesSizeInBytes;

    //
    // Replicate the graph dimensions.
    //

    CopyMemory(&Graph->Dimensions,
               &Info->Dimensions,
               sizeof(Graph->Dimensions));

    //
    // Carve out the bitmap buffer for Graph->DeletedEdges.
    //

    Graph->DeletedEdges.Buffer = (PULONG)Buffer;
    Graph->DeletedEdges.SizeOfBitMap = Graph->TotalNumberOfEdges + 1;
    Buffer += Info->DeletedEdgesBitmapBufferSizeInBytes;
    BitmapCount++;

    //
    // Carve out the bitmap buffer for Graph->VisitedEdges.
    //

    Graph->VisitedVertices.Buffer = (PULONG)Buffer;
    Graph->VisitedVertices.SizeOfBitMap = Graph->NumberOfVertices + 1;
    Buffer += Info->VisitedVerticesBitmapBufferSizeInBytes;
    BitmapCount++;

    //
    // Carve out the bitmap buffer for Graph->AssignedBitmap.
    //

    Graph->AssignedBitmap.Buffer = (PULONG)Buffer;
    Graph->AssignedBitmap.SizeOfBitMap = Graph->NumberOfVertices + 1;
    Buffer += Info->AssignedBitmapBufferSizeInBytes;
    BitmapCount++;

    //
    // Carve out the bitmap buffer for Graph->IndexBitmap.
    //

    Graph->IndexBitmap.Buffer = (PULONG)Buffer;
    Graph->IndexBitmap.SizeOfBitMap = Graph->NumberOfVertices + 1;
    Buffer += Info->IndexBitmapBufferSizeInBytes;
    BitmapCount++;

    //
    // Verify we visited the number of bitmaps we were expecting to visit.
    //

    ASSERT(Info->NumberOfBitmaps == BitmapCount);

    //
    // If our pointer arithmetic was correct, Buffer should match the base
    // address of the graph plus the total allocation size at this point.
    // Assert this invariant now.
    //

    ExpectedBuffer = RtlOffsetToPointer(Graph, Info->AllocSize);
    ASSERT(Buffer == ExpectedBuffer);

    //
    // Set the current thread ID and capture an attempt number from context.
    // Save the info address.
    //

    Graph->Info = Info;
    Graph->ThreadId = GetCurrentThreadId();

    //
    // Copy the edge and vertex masks, and the masking type.
    //

    Graph->EdgeMask = Table->IndexMask;
    Graph->VertexMask = Table->HashMask;
    Graph->MaskFunctionId = Info->Context->MaskFunctionId;

    //
    // Set the context.
    //

    Graph->Context = Info->Context;

    //
    // "Empty" all of the nodes; which they've chosen to mean setting them
    // all to -1.  (Can't we use 0 here?  This seems unnecessarily inefficient.)
    //

    for (Index = 0; Index < Graph->NumberOfVertices; Index++) {
        Graph->First[Index] = EMPTY;
    }

    for (Index = 0; Index < Graph->TotalNumberOfEdges; Index++) {
        Graph->Prev[Index] = EMPTY;
        Graph->Next[Index] = EMPTY;
        Graph->Edges[Index] = EMPTY;
    }

    //
    // Obtain seed data for the last two seeds.
    //

    GetRandomSeedsBlocking(&Graph->Seeds34);

    //
    // Initialization complete!
    //

    return;

#endif
}
#endif


////////////////////////////////////////////////////////////////////////////////
// Algorithm Implementation
////////////////////////////////////////////////////////////////////////////////

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

    //
    // Insert the first edge.  If we've already seen this edge value, insert it
    // into the previous edges array.
    //

    First1 = Graph->First[Vertex1];
    if (!IsEmpty(First1)) {
        Graph->Prev[First1] = Edge1;
    }

    Graph->Next[Edge1] = First1;
    Graph->First[Vertex1] = Edge1;
    Graph->Edges[Edge1] = Vertex2;
    Graph->Prev[Edge1] = EMPTY;

    //
    // Insert the second edge.  If we've already seen this edge value, insert it
    // into the previous edges array.
    //

    First2 = Graph->First[Vertex2];
    if (!IsEmpty(First2)) {
        Graph->Prev[First2] = Edge2;
    }

    Graph->Next[Edge2] = First2;
    Graph->First[Vertex2] = Edge2;
    Graph->Edges[Edge2] = Vertex1;
    Graph->Prev[Edge2] = EMPTY;

}

GRAPH_DELETE_EDGE GraphDeleteEdge;

_Use_decl_annotations_
VOID
GraphDeleteEdge(
    PGRAPH Graph,
    EDGE Edge
    )
/*++

Routine Description:

    This routine deletes an edge from the hypergraph.

Arguments:

    Graph - Supplies a pointer to the graph for which the edge is to be deleted.

    Edge - Supplies the edge to delete from the graph.

Return Value:

    None.

--*/
{
    EDGE Prev;
    EDGE Next;
    VERTEX Vertex;

    Vertex = Graph->Edges[Edge];

    Prev = Graph->Prev[Vertex];

    if (IsEmpty(Prev)) {

        //
        // This is the initial edge.
        //

        Graph->First[Vertex] = Graph->Next[Edge];

    } else {

        //
        // Not the initial edge.
        //

        Graph->Next[Prev] = Graph->Next[Edge];

    }

    Next = Graph->Next[Edge];

    if (!IsEmpty(Next)) {

        //
        // Not at the end.
        //

        Graph->Prev[Next] = Prev;
    }
}

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

    //
    // Resolve aliases.
    //

    NumberOfKeys = Graph->NumberOfKeys;
    NumberOfEdges = Graph->NumberOfEdges;
    NumberOfVertices = Graph->NumberOfVertices;
    RtlNumberOfSetBits = Graph->Context->Rtl->RtlNumberOfSetBits;

    //
    // Invariant check: we should not be shrinking prior to this point.
    //

    ASSERT(!Graph->Flags.Shrinking);

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

    //
    // As each edge of degree 1 is deleted, a bit is set in the deleted bitmap,
    // indicating the edge at that bit offset was deleted.  Thus, we can simply
    // count the number of set bits in the bitmap and compare that to the number
    // of edges in the graph.  If the values do not match, the graph is cyclic;
    // if they do match, the graph is acyclic.
    //

    NumberOfEdgesDeleted = RtlNumberOfSetBits(&Graph->DeletedEdgesBitmap);

    //
    // Temporary assert to determine if the number of edges deleted will always
    // meet our deleted edge count.  (If so, we can just test this value,
    // instead of having to count the bitmap bits.)
    //

    ASSERT(NumberOfEdgesDeleted == Graph->DeletedEdgeCount);

    IsAcyclic = (NumberOfKeys == NumberOfEdgesDeleted);

    //
    // Temporary slow version to verify our assumption about counting bits is
    // correct.
    //

    IsAcyclicSlow = TRUE;

    for (Edge = 0; Edge < NumberOfKeys; Edge++) {
        if (!IsDeletedEdge(Graph, Edge)) {
            IsAcyclicSlow = FALSE;
            break;
        }
    }

    ASSERT(IsAcyclic == IsAcyclicSlow);

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

    return IsAcyclic;
}

GRAPH_ASSIGN GraphAssign;

_Use_decl_annotations_
VOID
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

    TRUE if the graph is acyclic, FALSE if it's cyclic.

--*/
{
    PRTL Rtl;
    VERTEX Vertex;
    ULONG Depth;
    ULONG MaximumDepth;
    ULONG NumberOfSetBits;

    //
    // Invariant check: the acyclic flag should be set.  (Indicating that
    // IsGraphAcyclic() successfully determined that, yes, the graph is
    // acyclic.)
    //

    ASSERT(Graph->Flags.IsAcyclic);

    //
    // Initialize the depth and maximum depth counters.
    //

    Depth = 0;
    MaximumDepth = 0;

    //
    // Walk the graph and assign values.
    //

    for (Vertex = 0; Vertex < Graph->NumberOfVertices; Vertex++) {

        if (!IsVisitedVertex(Graph, Vertex)) {

            //
            // Assign an initial value of 0, then walk the subgraph.
            //

            Graph->Assigned[Vertex] = 0;
            GraphTraverse(Graph, Vertex, &Depth, &MaximumDepth);
        }
    }

    Rtl = Graph->Context->Rtl;
    NumberOfSetBits = Rtl->RtlNumberOfSetBits(&Graph->VisitedVerticesBitmap);

    ASSERT(Graph->VisitedVerticesCount == NumberOfSetBits);
    ASSERT(Graph->VisitedVerticesCount == Graph->NumberOfVertices);

    return;
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
        // verification stage.  Reverting to just returning the edge at least
        // allows the CRC32 hash + AND mask combo to work satisfactorily.
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

GRAPH_TRAVERSE GraphTraverse;

_Use_decl_annotations_
VOID
GraphTraverse(
    PGRAPH Graph,
    VERTEX Vertex,
    PULONG Depth,
    PULONG MaximumDepth
    )
/*++

Routine Description:

    This routine is called as part of graph assignment.  It is responsible for
    doing a depth-first traversal of the graph and obtaining edge IDs that can
    be saved in the Graph->Assigned array.

Arguments:

    Graph - Supplies a pointer to the graph to operate on.

    Vertex - Supplies the vertex to traverse.

    Depth - Supplies a pointer to a variable that is used to capture the call
        stack depth of the recursive graph traversal.  This is incremented
        on entry and decremented on exit.

    MaximumDepth - Supplies a pointer to a variable that will be used to track
        the maximum depth observed during recursive graph traversal.

Return Value:

    None.

--*/
{
    PRTL Rtl;
    ULONG Bit;
    LONG EdgeId;
    LONG ThisId;
    LONG FinalId;
    ULONG MaskedEdgeId;
    ULONG MaskedThisId;
    ULONG MaskedFinalId;
    LONG ExistingId;
    LONG OriginalExistingId;
    VERTEX Neighbor;
    GRAPH_ITERATOR Iterator;
    PPERFECT_HASH_TABLE Table;

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
    // Initialize aliases.
    //

    Rtl = Graph->Context->Rtl;
    Table = Graph->Context->Table;

    //
    // Update the depth.
    //

    *Depth += 1;
    if (*Depth > *MaximumDepth) {
        *MaximumDepth = *Depth;
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

        MASK_INDEX(EdgeId, &MaskedEdgeId);

        OriginalExistingId = ExistingId = Graph->Assigned[Vertex];

        ASSERT(ExistingId >= 0);

        ThisId = EdgeId - ExistingId;

        MASK_INDEX(ThisId, &MaskedThisId);

        ASSERT(MaskedThisId <= Graph->NumberOfVertices);

        FinalId = EdgeId + ExistingId;

        MASK_INDEX(FinalId, &MaskedFinalId);

        ASSERT(MaskedFinalId <= Graph->NumberOfVertices);

        Bit = MaskedFinalId + 1;

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

        GraphTraverse(Graph, Neighbor, Depth, MaximumDepth);

    }

    //
    // We need an Error: label for the MASK_INDEX() et al macros.
    //

    goto End;

Error:

    //
    // We shouldn't ever reach here.
    //

    PH_RAISE(PH_E_UNREACHABLE_CODE);

End:

    //
    // Decrement depth and return.
    //

    *Depth -= 1;
}

SOLVE_GRAPH SolveGraph;

_Use_decl_annotations_
BOOLEAN
SolveGraph(
    _In_ PGRAPH Graph
    )
/*++

Routine Description:

    Add all keys to the hypergraph using the unique seeds to hash each key into
    two vertex values, connected by a "hyper-edge".  Determine if the graph is
    acyclic, if it is, we've "solved" the graph.  If not, we haven't, return
    FALSE such that another attempt can be made with new random unique seeds.

Arguments:

    Graph - Supplies a pointer to the graph to be solved.

Return Value:

    TRUE if the graph was solved successfully, FALSE otherwise.

--*/
{
    KEY Key;
    PKEY Keys;
    EDGE Edge;
    PEDGE Edges;
    VERTEX Vertex1;
    VERTEX Vertex2;
    PGRAPH_INFO Info;
    ULONG Iterations;
    ULONG NumberOfKeys;
    ULARGE_INTEGER Hash;
    LONGLONG FinishedCount;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_CONTEXT Context;
    const ULONG CheckForTerminationAfterIterations = 1024;

    Info = Graph->Info;
    Context = Info->Context;
    Table = Context->Table;
    NumberOfKeys = Table->Keys->NumberOfElements.LowPart;
    Edges = Keys = (PKEY)Table->Keys->File->BaseAddress;

    //
    // Enumerate all keys in the input set, hash them into two unique vertices,
    // then add them to the hypergraph.
    //

    Iterations = CheckForTerminationAfterIterations;

    for (Edge = 0; Edge < NumberOfKeys; Edge++) {
        Key = *Edges++;

        SEEDED_HASH(Key, &Hash.QuadPart);

        ASSERT(Hash.HighPart != Hash.LowPart);

        //
        // Mask the individual vertices.
        //

        MASK_HASH(Hash.LowPart, &Vertex1);
        MASK_HASH(Hash.HighPart, &Vertex2);

        //
        // We can't have two vertices point to the same location.
        // Abort this graph attempt.
        //

        if (Vertex1 == Vertex2) {
            goto Failed;
        }

        //
        // Add the edge to the graph connecting these two vertices.
        //

        GraphAddEdge(Graph, Edge, Vertex1, Vertex2);

        //
        // If we're in "first graph wins" mode, every 1024 iterations, check
        // to see if someone else has already solved the graph, and if so, do
        // a fast-path exit.
        //

        if (FirstSolvedGraphWins(Context)) {

            if (!--Iterations) {
                if (Context->FinishedCount > 0) {
                    return FALSE;
                }

                //
                // Reset the iteration counter.
                //

                Iterations = CheckForTerminationAfterIterations;
            }
        }
    }

    //
    // We've added all of the vertices to the graph.  Determine if the graph
    // is acyclic.
    //

    if (!IsGraphAcyclic(Graph)) {

        //
        // Failed to create an acyclic graph.
        //

        goto Failed;
    }

    //
    // We created an acyclic graph.
    //

    //
    // Increment the finished count.  If the context indicates "first solved
    // graph wins", and the value is 1, we're the winning thread, so continue
    // with graph assignment.  Otherwise, just return TRUE immediately and let
    // the other thread finish up (i.e. perform the assignment step and then
    // persist the result).
    //
    // If the context does not indicate "first solved graph wins", perform
    // the assignment step regardless.
    //

    FinishedCount = InterlockedIncrement64(&Context->FinishedCount);

    if (FirstSolvedGraphWins(Context)) {

        if (FinishedCount != 1) {

            //
            // Some other thread beat us.  Nothing left to do.
            //

            return TRUE;
        }
    }

    //
    // Perform the assignment step.
    //

    GraphAssign(Graph);

    //
    // Calculate memory coverage information.
    //

    GraphCalculateMemoryCoverage(Graph);

    //
    // Stop the solve timers here.
    //

    CONTEXT_END_TIMERS(Solve);

    //
    // If we're in "first graph wins" mode and we reach this point, we're the
    // winning thread, so, push the graph onto the finished list head, then
    // submit the relevant finished threadpool work item and return TRUE.
    //

    if (FirstSolvedGraphWins(Context)) {
        InsertTailFinishedWork(Context, &Graph->ListEntry);
        SubmitThreadpoolWork(Context->FinishedWork);
        return TRUE;
    }

    //
    // If we reach this mode, we're in "find best memory coverage" mode, so,
    // register the solved graph and return FALSE in order to kick off another
    // attempt.
    //

    ASSERT(FindBestMemoryCoverage(Context));

    GraphRegisterSolved(Graph);
    return FALSE;

Failed:

    //
    // Increment the failed attempts counter.
    //

    InterlockedIncrement64(&Context->FailedAttempts);

    //
    // Intentional follow-on to Error.
    //

Error:

    //
    // If any of the HASH/MASK macros fail, they'll jump to this Error: label.
    //

    return FALSE;
}

_Use_decl_annotations_
HRESULT
VerifySolvedGraph(
    _In_ PGRAPH Graph
    )
/*++

Routine Description:

    Verify a solved graph is working correctly.  This walks through the entire
    original key set, captures the index that is returned when the key is hashed
    (i.e. simulates the Index() method), sets a bit in a bitmap for each index,
    verifying that we never see the same index twice, as this would indicate a
    collision, and then finally, verify that the number of set bits in the
    bitmap exactly equals the number of keys we saw.

    N.B. The original chm.c-style modulus-oriented solution fails to pass this
         step, which actually matches the experience I had with it during my
         initial evaluation.

Arguments:

    Graph - Supplies a pointer to the graph to be verified.

Return Value:

    S_OK - Graph was solved successfully.

    E_UNEXPECTED - Internal error during graph validation.

    PH_E_COLLISIONS_ENCOUNTERED_DURING_GRAPH_VALIDATION - Collisions were
        detected during graph validation.

    PH_E_NUM_ASSIGNMENTS_NOT_EQUAL_TO_NUM_KEYS_DURING_GRAPH_VALIDATION -
        The number of value assignments did not equal the number of keys
        during graph validation.

--*/
{
    PRTL Rtl;
    KEY Key;
    KEY PreviousKey;
    PKEY Keys;
    EDGE Edge;
    PEDGE Edges;
    ULONG Bit;
    ULONG Index;
    ULONG PrevIndex;
    PULONG Values;
    VERTEX Vertex1;
    VERTEX Vertex2;
    VERTEX MaskedLow;
    VERTEX MaskedHigh;
    VERTEX PrevVertex1;
    VERTEX PrevVertex2;
    VERTEX PrevMaskedLow;
    VERTEX PrevMaskedHigh;
    PVERTEX Assigned;
    PGRAPH_INFO Info;
    ULONG NumberOfKeys;
    ULONG NumberOfAssignments;
    ULONG Collisions = 0;
    LONGLONG Combined;
    LONGLONG PrevCombined;
    ULARGE_INTEGER Hash;
    ULARGE_INTEGER PrevHash;
    HRESULT Result = S_OK;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_CONTEXT Context;

    Info = Graph->Info;
    Context = Info->Context;
    Rtl = Context->Rtl;
    Table = Context->Table;
    NumberOfKeys = Graph->NumberOfKeys;
    Edges = Keys = (PKEY)Table->Keys->File->BaseAddress;
    Assigned = Graph->Assigned;

    NumberOfAssignments = Rtl->RtlNumberOfSetBits(&Graph->AssignedBitmap);
    ASSERT(NumberOfAssignments == 0);

    Values = Graph->Values;

    //
    // Enumerate all keys in the input set and verify they can be resolved
    // correctly from the assigned vertex array.
    //

    for (Edge = 0; Edge < NumberOfKeys; Edge++) {
        Key = *Edges++;

        //
        // Hash the key.
        //

        SEEDED_HASH(Key, &Hash.QuadPart);

        ASSERT(Hash.QuadPart);
        ASSERT(Hash.HighPart != Hash.LowPart);

        //
        // Mask the high and low parts of the hash.
        //

        MASK_HASH(Hash.LowPart, &MaskedLow);
        MASK_HASH(Hash.HighPart, &MaskedHigh);

        //
        // Extract the individual vertices.
        //

        Vertex1 = Assigned[MaskedLow];
        Vertex2 = Assigned[MaskedHigh];

        //
        // Mask the result.
        //

        Combined = (LONGLONG)Vertex1 + (LONGLONG)Vertex2;

        MASK_INDEX(Combined, &Index);

        Bit = Index + 1;

        //
        // Make sure we haven't seen this bit before.
        //

        if (TestGraphBit(AssignedBitmap, Bit)) {

            //
            // We've seen this index before!  Get the key that previously
            // mapped to it.
            //

            PreviousKey = Values[Index];

            SEEDED_HASH(PreviousKey, &PrevHash.QuadPart);

            MASK_HASH(PrevHash.LowPart, &PrevMaskedLow);
            MASK_HASH(PrevHash.HighPart, &PrevMaskedHigh);

            PrevVertex1 = Assigned[MaskedLow];
            PrevVertex2 = Assigned[MaskedHigh];

            PrevCombined = (LONGLONG)PrevVertex1 + (LONGLONG)PrevVertex2;

            MASK_INDEX(PrevCombined, &PrevIndex);

            Collisions++;

        }

        //
        // Set the bit and store this key in the underlying values array.
        //

        SetGraphBit(AssignedBitmap, Bit);
        Values[Index] = Key;

    }

    if (Collisions) {
        Result = PH_E_COLLISIONS_ENCOUNTERED_DURING_GRAPH_VALIDATION;
        goto Error;
    }

    NumberOfAssignments = Rtl->RtlNumberOfSetBits(&Graph->AssignedBitmap);

    if (NumberOfAssignments != NumberOfKeys) {
        Result =
            PH_E_NUM_ASSIGNMENTS_NOT_EQUAL_TO_NUM_KEYS_DURING_GRAPH_VALIDATION;
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

    return Result;
}

//
// Work-in-progress memory coverage routines.
//

typedef struct _MEMORY_COVERAGE {
    ULONG NumberOfEmptyCacheLines;
    ULONG NumberOfEmptyPages;
    ULONG NumberOfEmptyLargePages;
    ULONG Unused;
} MEMORY_COVERAGE;

typedef ULONG ASSIGNED_CACHE_LINE[16];
typedef ASSIGNED_CACHE_LINE  *PASSIGNED_CACHE_LINE;

GRAPH_CALCULATE_MEMORY_COVERAGE GraphCalculateMemoryCoverageAvx2;

_Use_decl_annotations_
VOID
GraphCalculateMemoryCoverageAvx2(
    PGRAPH Graph
    )
/*++

Routine Description:

    Calculate the memory coverage of a solved, assigned graph.

    Work in progress.

Arguments:

    Graph - Supplies a pointer to the graph for which memory coverage is to
        be calculated.

Return Value:

    None.

--*/
{
    PRTL Rtl;
    ULONG Index;
    ULONG Trailing;
    ULONG NumberOfCacheLines;
    ULONG AssignedLowHighSizeInBytes;
    ULONG PageSizeBytesProcessed;
    ULONG LargePageSizeBytesProcessed;
    ULONG TotalBytesProcessed;
    ULONG Low1Mask;
    ULONG Low2Mask;
    ULONG High1Mask;
    ULONG High2Mask;
    ULONG Low1Count;
    ULONG Low2Count;
    ULONG High1Count;
    ULONG High2Count;
    ULONG LowCount;
    ULONG HighCount;
    ULONG LowEmptyCacheLinesThisPage;
    ULONG LowEmptyCacheLinesThisLargePage;
    ULONG HighEmptyCacheLinesThisPage;
    ULONG HighEmptyCacheLinesThisLargePage;
    ULONG LowUsedCacheLinesThisPage;
    ULONG LowUsedCacheLinesThisLargePage;
    ULONG HighUsedCacheLinesThisPage;
    ULONG HighUsedCacheLinesThisLargePage;
    const ULONG PageSize = PAGE_SIZE;
    const ULONG LargePageSize = (ULONG)GetLargePageMinimum();
    const YMMWORD AllZeros = _mm256_set1_epi8(0);
    YMMWORD Low1;
    YMMWORD Low2;
    YMMWORD High1;
    YMMWORD High2;
    YMMWORD Low1Zeros;
    YMMWORD Low2Zeros;
    YMMWORD High1Zeros;
    YMMWORD High2Zeros;
    ULONG Alignment;
    PVERTEX AssignedLow;
    PVERTEX AssignedHigh;
    PBYTE AssignedLowBuffer;
    PBYTE AssignedHighBuffer;
    MEMORY_COVERAGE CoverageLow;
    MEMORY_COVERAGE CoverageHigh;
    PASSIGNED_CACHE_LINE LowLine;
    PASSIGNED_CACHE_LINE HighLine;

    Rtl = Graph->Context->Rtl;

    AssignedLow = Graph->Assigned;
    AssignedHigh = (PVERTEX)(
        RtlOffsetToPointer(
            AssignedLow,
            Graph->NumberOfVertices
        )
    );

    AssignedLowHighSizeInBytes = Graph->NumberOfVertices * sizeof(VERTEX);
    NumberOfCacheLines = AssignedLowHighSizeInBytes >> 6;
    Trailing = AssignedLowHighSizeInBytes - (NumberOfCacheLines << 6);

    if (Trailing != 0) {
        PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
    }

    AssignedLowBuffer = (PBYTE)AssignedLow;
    AssignedHighBuffer = (PBYTE)AssignedHigh;

    TotalBytesProcessed = 0;
    PageSizeBytesProcessed = 0;
    LargePageSizeBytesProcessed = 0;

    LowEmptyCacheLinesThisPage = 0;
    LowEmptyCacheLinesThisLargePage = 0;
    HighEmptyCacheLinesThisPage = 0;
    HighEmptyCacheLinesThisLargePage = 0;

    LowUsedCacheLinesThisPage = 0;
    LowUsedCacheLinesThisLargePage = 0;
    HighUsedCacheLinesThisPage = 0;
    HighUsedCacheLinesThisLargePage = 0;

    ZeroStruct(CoverageLow);
    ZeroStruct(CoverageHigh);

    Alignment = (ULONG)GetAddressAlignment((ULONG_PTR)AssignedLow);
    AssertAligned32(Alignment);

    for (Index = 0; Index < NumberOfCacheLines; Index++) {

        LowLine = (PASSIGNED_CACHE_LINE)AssignedLowBuffer;

        Low1 = _mm256_stream_load_si256((PYMMWORD)AssignedLowBuffer);
        AssignedLowBuffer += 32;
        Low2 = _mm256_stream_load_si256((PYMMWORD)AssignedLowBuffer);
        AssignedLowBuffer += 32;

        Low1Zeros = _mm256_cmpgt_epi32(Low1, AllZeros);
        Low2Zeros = _mm256_cmpgt_epi32(Low2, AllZeros);

        Low1Mask = _mm256_movemask_epi8(Low1Zeros);
        Low2Mask = _mm256_movemask_epi8(Low2Zeros);

        Low1Count = PopulationCount32(Low1Mask);
        Low2Count = PopulationCount32(Low2Mask);

        LowCount = Low1Count + Low2Count;

        if (LowCount == 0) {
            CoverageLow.NumberOfEmptyCacheLines++;
            LowEmptyCacheLinesThisPage++;
            LowEmptyCacheLinesThisLargePage++;
        } else {
            LowUsedCacheLinesThisPage += LowCount;
            LowUsedCacheLinesThisLargePage += LowCount;
        }

        HighLine = (PASSIGNED_CACHE_LINE)AssignedHighBuffer;

        High1 = _mm256_stream_load_si256((PYMMWORD)AssignedHighBuffer);
        AssignedHighBuffer += 32;
        High2 = _mm256_stream_load_si256((PYMMWORD)AssignedHighBuffer);
        AssignedHighBuffer += 32;

        High1Zeros = _mm256_cmpgt_epi32(High1, AllZeros);
        High2Zeros = _mm256_cmpgt_epi32(High2, AllZeros);

        High1Mask = _mm256_movemask_epi8(High1Zeros);
        High2Mask = _mm256_movemask_epi8(High2Zeros);

        High1Count = PopulationCount32(High1Mask);
        High2Count = PopulationCount32(High2Mask);

        HighCount = High1Count + High2Count;

        if (HighCount == 0) {
            CoverageHigh.NumberOfEmptyCacheLines++;
            HighEmptyCacheLinesThisPage++;
            HighEmptyCacheLinesThisLargePage++;
        } else {
            HighUsedCacheLinesThisPage += HighCount;
            HighUsedCacheLinesThisLargePage += HighCount;
        }

        TotalBytesProcessed += 64;

        PageSizeBytesProcessed += 64;

        if (PageSizeBytesProcessed == PageSize) {

            if (LowEmptyCacheLinesThisPage == 0) {
                CoverageLow.NumberOfEmptyPages++;
            } else {
                LowEmptyCacheLinesThisPage = 0;
            }

            if (HighEmptyCacheLinesThisPage == 0) {
                CoverageHigh.NumberOfEmptyPages++;
            } else {
                HighEmptyCacheLinesThisPage = 0;
            }

            PageSizeBytesProcessed = 0;
        }

        if (LargePageSizeBytesProcessed == LargePageSize) {

            if (LowEmptyCacheLinesThisLargePage == 0) {
                CoverageLow.NumberOfEmptyLargePages++;
            } else {
                LowEmptyCacheLinesThisLargePage = 0;
            }

            if (HighEmptyCacheLinesThisLargePage == 0) {
                CoverageHigh.NumberOfEmptyLargePages++;
            } else {
                HighEmptyCacheLinesThisLargePage = 0;
            }

            LargePageSizeBytesProcessed = 0;
        }

    }

}

GRAPH_REGISTER_SOLVED GraphRegisterSolved;

_Use_decl_annotations_
VOID
GraphRegisterSolved(
    PGRAPH Graph
    )
/*++

Routine Description:

    Attempts to register a solved graph with a context if the graph's memory
    coverage is the best that's been encountered so far.

Arguments:

    Graph - Supplies a pointer to the solved graph to register.

Return Value:

    None.

--*/
{
    DBG_UNREFERENCED_PARAMETER(Graph);
}


GRAPH_SET_INFO GraphSetInfo;

_Use_decl_annotations_
HRESULT
GraphSetInfo(
    PGRAPH Graph,
    PGRAPH_INFO Info
    )
/*++

Routine Description:

    Registers information about a graph with an individual graph instance.
    This routine is called once per unique graph info (that is, if a table
    resize event occurs it will be called again with the new graph info).
    The LoadInfo() routine will use the provided info for allocating or
    reallocating the necessary buffers required for graph solving.

    N.B. This routine is intended to be called from the "main" thread, whereas
         LoadInfo() is intended to be called as the first operation by graph
         solving worker threads.  Thus, this routine is pretty simple.

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


GRAPH_ENTER_SOLVING_LOOP GraphEnterSolvingLoop;

_Use_decl_annotations_
HRESULT
GraphEnterSolvingLoop(
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
    HRESULT Result = S_OK;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Graph)) {
        return E_POINTER;
    }

    //
    // Acquire the exclusive graph lock for the duration of the routine.
    //

    AcquireGraphLockExclusive(Graph);

    //
    // Load the graph info.
    //

    Result = Graph->Vtbl->LoadInfo(Graph);

    if (FAILED(Result)) {
        PH_ERROR(GraphLoadInfo, Result);
        goto End;
    }

    //
    // Begin the solving loop.
    //

    while (ShouldWeContinueTryingToSolveGraphChm01(Graph->Context)) {

        Result = Graph->Vtbl->LoadNewSeeds(Graph);
        if (FAILED(Result)) {
            if (Result != PH_E_NO_MORE_SEEDS) {
                PH_ERROR(GraphLoadNewSeeds, Result);
            }
            break;
        }

        Result = Graph->Vtbl->Reset(Graph);
        if (FAILED(Result)) {
            PH_ERROR(GraphReset, Result);
            break;
        }

        Result = Graph->Vtbl->Solve(Graph);
        if (FAILED(Result)) {

            //
            // If the error code indicates anything other than an imminent
            // table resize, log it.
            //

            if (Result != PH_E_TABLE_RESIZE_IMMINENT) {
                PH_ERROR(GraphSolve, Result);
            }

            break;
        }

        if (Result == S_FALSE) {

            //
            // No failure was encountered, but no more solving needs to be done.
            //

            Result = S_OK;
            break;

        } else {

            //
            // The only permissible result here is S_OK.  Verify this, then
            // allow the loop to continue.
            //

            ASSERT(Result == S_OK);
        }

    }

    //
    // Intentional follow-on to End.
    //

End:

    ReleaseGraphLockExclusive(Graph);

    return Result;
}


GRAPH_LOAD_INFO GraphLoadInfo;

_Use_decl_annotations_
HRESULT
GraphLoadInfo(
    PGRAPH Graph
    )
/*++

Routine Description:

    This routine is called by graph solving worker threads prior to attempting
    any solving; it is responsible for initializing the graph structure and
    allocating (or reallocating) the necessary buffers required for graph
    solving, using the sizes indicated by the info structure previously set
    by the main thread via SetInfo().

Arguments:

    Graph - Supplies a pointer to the graph instance.

Return Value:

    S_OK - Success.

    E_POINTER - Graph was NULL.

    E_OUTOFMEMORY - Out of memory.

    PH_E_GRAPH_NO_INFO_SET - No graph information has been set for this graph.

--*/
{
    PRTL Rtl;
    HRESULT Result = S_OK;
    BOOLEAN ZeroMemory;
    PGRAPH_INFO Info;
    PALLOCATOR Allocator;
    PPERFECT_HASH_TABLE Table;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;
    PPERFECT_HASH_CONTEXT Context;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Graph)) {
        return E_POINTER;
    }

    if (!IsGraphInfoSet(Graph)) {
        return PH_E_GRAPH_NO_INFO_SET;
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
    Allocator = Graph->Allocator;
    Table = Context->Table;
    TableInfoOnDisk = Table->TableInfoOnDisk;

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
    Graph->MaskFunctionId = Info->Context->MaskFunctionId;

    CopyInline(&Graph->Dimensions,
               &Info->Dimensions,
               sizeof(Graph->Dimensions));

    //
    // We don't need zerod memory as the Reset() call will be responsible for
    // getting all of the arrays and bitmap buffers into an appropriate state.
    //

    ZeroMemory = FALSE;

    //
    // Allocate (or reallocate) arrays.
    //

#define ALLOC_ARRAY(Name, Type)                      \
    if (!Graph->##Name) {                            \
        Graph->##Name = (Type)(                      \
            Allocator->Vtbl->Malloc(                 \
                Allocator,                           \
                (ULONG_PTR)Info->##Name##SizeInBytes \
            )                                        \
        );                                           \
    } else {                                         \
        Graph->##Name## = (Type)(                    \
            Allocator->Vtbl->ReAlloc(                \
                Allocator,                           \
                ZeroMemory,                          \
                Graph->##Name,                       \
                (ULONG_PTR)Info->##Name##SizeInBytes \
            )                                        \
        );                                           \
    }                                                \
    if (!Graph->##Name) {                            \
        Result = E_OUTOFMEMORY;                      \
        goto Error;                                  \
    }

    ALLOC_ARRAY(Edges, PEDGE);
    ALLOC_ARRAY(Next, PEDGE);
    ALLOC_ARRAY(First, PVERTEX);
    ALLOC_ARRAY(Prev, PVERTEX);
    ALLOC_ARRAY(Assigned, PVERTEX);

    //
    // Set the bitmap sizes and then allocate (or reallocate) the bitmap
    // buffers.
    //

    Graph->DeletedEdgesBitmap.SizeOfBitMap = Graph->TotalNumberOfEdges + 1;
    Graph->VisitedVerticesBitmap.SizeOfBitMap = Graph->NumberOfVertices + 1;
    Graph->AssignedBitmap.SizeOfBitMap = Graph->NumberOfVertices + 1;
    Graph->IndexBitmap.SizeOfBitMap = Graph->NumberOfVertices + 1;

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
                ZeroMemory,                                \
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
    // We're done, finish up.
    //

    Graph->Flags.IsInfoSet = TRUE;
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


GRAPH_RESET GraphReset;

_Use_decl_annotations_
HRESULT
GraphReset(
    PGRAPH Graph
    )
/*++

Routine Description:

    Resets the state of a graph instance after a solving attempt, such that it
    can be used for a subsequent attempt.

Arguments:

    Graph - Supplies a pointer to the graph instance to reset.

Return Value:

    S_OK - Success.

    PH_E_TABLE_RESIZE_IMMINENT - The reset was not performed as a table resize
        is imminent (and thus, attempts at solving this current graph can be
        stopped).

    PH_E_SYSTEM_CALL_FAILED - A system call failed.

--*/
{
    ULONG Index;
    HRESULT Result = S_OK;
    PGRAPH_INFO Info;
    PPERFECT_HASH_CONTEXT Context;

    //
    // Initialize aliases.
    //

    Context = Graph->Context;
    Info = Graph->Info;

    //
    // Increment the thread attempt counter, and interlocked-increment the
    // global context counter.  If the global attempt is equal to the resize
    // table threshold, signal the event to try a larger table size.
    //

    ++Graph->ThreadAttempt;

    Graph->Attempt = InterlockedIncrement64(&Context->Attempts);

    if (Graph->Attempt == Context->ResizeTableThreshold) {
        if (!SetEvent(Context->TryLargerTableSizeEvent)) {
            SYS_ERROR(SetEvent);
            Result = PH_E_SYSTEM_CALL_FAILED;
            goto Error;
        }
        return PH_E_TABLE_RESIZE_IMMINENT;
    }

#ifdef _M_X64
#define ZeroInline(Dest, Size) __stosq((PDWORD64)Dest, 0, (Size >> 3))
#else
#define ZeroInline(Dest, Size) ZeroMemory(Dest, Size)
#endif

#define ZERO_BITMAP_BUFFER(Name) \
    ZeroInline(Graph->##Name##.Buffer, Info->##Name##BufferSizeInBytes)

    //
    // Clear the bitmap buffers.
    //

    ZERO_BITMAP_BUFFER(DeletedEdgesBitmap);
    ZERO_BITMAP_BUFFER(VisitedVerticesBitmap);
    ZERO_BITMAP_BUFFER(AssignedBitmap);
    ZERO_BITMAP_BUFFER(IndexBitmap);

    //
    // "Empty" all of the nodes.
    //

    for (Index = 0; Index < Graph->NumberOfVertices; Index++) {
        Graph->First[Index] = EMPTY;
    }

    for (Index = 0; Index < Graph->TotalNumberOfEdges; Index++) {
        Graph->Prev[Index] = EMPTY;
        Graph->Next[Index] = EMPTY;
        Graph->Edges[Index] = EMPTY;
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


GRAPH_LOAD_NEW_SEEDS GraphLoadNewSeeds;

_Use_decl_annotations_
HRESULT
GraphLoadNewSeeds(
    PGRAPH Graph
    )
/*++

Routine Description:

    Loads new seed data for a graph instance.  This is called prior to each
    solving attempt.

Arguments:

    Graph - Supplies a pointer to the graph instance for which the new seed
        data will be loaded.

Return Value:

    S_OK - Success.

    E_POINTER - Graph was NULL.

    PH_E_NO_MORE_SEEDS - No more seed data is available.  (Not currently
        returned for this implementation.)

--*/
{
    PRTL Rtl;
    HRESULT Result;
    ULONG SizeInBytes;

    if (!ARGUMENT_PRESENT(Graph)) {
        return E_POINTER;
    }

    SizeInBytes = Graph->NumberOfSeeds * sizeof(Graph->FirstSeed);

    Rtl = Graph->Rtl;

    Result = Rtl->Vtbl->GenerateRandomBytes(Rtl,
                                            SizeInBytes,
                                            (PBYTE)&Graph->FirstSeed);

    return Result;
}


GRAPH_SOLVE GraphSolve;

_Use_decl_annotations_
HRESULT
GraphSolve(
    PGRAPH Graph
    )
/*++

Routine Description:

    Attempts to solve the graph.

Arguments:

    Graph - Supplies a pointer to a graph instance.

Return Value:

    S_OK - Success.

    E_POINTER - Graph was NULL.

    PH_E_NO_MORE_SEEDS - No more seed data is available.  (Not currently
        returned for this implementation.)

--*/
{
    if (!ARGUMENT_PRESENT(Graph)) {
        return E_POINTER;
    }

    //
    // XXX stop-gap measure until SolveGraph() is updated.
    //

    if (SolveGraph(Graph)) {
        return S_FALSE;
    } else {
        return S_OK;
    }
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
