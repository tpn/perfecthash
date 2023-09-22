/*++

Copyright (c) 2018-2023 Trent Nelson <trent@trent.me>

Module Name:

    GraphImpl3.c

Abstract:

    This module is our third implementation of the graph module for the perfect
    hash library.  It is derived from improvements made to NetBSD's nbperf
    module in late 2020/early 2021.

    I've included the copyright from NetBSD nbperf's nbperf-chm.c* for this
    reason; thanks Joerg Sonnenberger <joerg at bec dot de>!

    [*]: https://github.com/tpn/nbperf/blob/master/v2/nbperf-chm.c

--*/

/*	$NetBSD: graph2.c,v 1.5 2021/01/07 16:03:08 joerg Exp $	*/
/*-
 * Copyright (c) 2009 The NetBSD Foundation, Inc.
 * All rights reserved.
 *
 * This code is derived from software contributed to The NetBSD Foundation
 * by Joerg Sonnenberger.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
 * COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include "stdafx.h"
#include "GraphImpl3.h"
#include "PerfectHashEventsPrivate.h"

//
// Define helper macros.
//

#define EMPTY ((VERTEX)-1)
#define IsEmpty(Value) ((ULONG)Value == EMPTY)

//
// When a solution has been found and the assignment step begins, the initial
// value assigned to a vertex is govered by the following macro.
//

#define INITIAL_ASSIGNMENT_VALUE 0

GRAPH_ADD_EDGE GraphAddEdge3;

_Use_decl_annotations_
VOID
GraphAddEdge3(
    PGRAPH Graph,
    EDGE Edge,
    VERTEX Vertex1Index,
    VERTEX Vertex2Index
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
    PVERTEX3 Vertex1;
    PVERTEX3 Vertex2;

#ifdef _DEBUG
    //
    // Invariant checks:
    //
    //      - Vertex1Index should be less than the number of vertices.
    //      - Vertex2Index should be less than the number of vertices.
    //      - Edge should be less than the number of edges.
    //      - The graph must not have started deletions.
    //

    ASSERT(Vertex1Index < Graph->NumberOfVertices);
    ASSERT(Vertex2Index < Graph->NumberOfVertices);
    ASSERT(Edge < Graph->NumberOfEdges);
    ASSERT(!Graph->Flags.Shrinking);
#endif

    //
    // Insert the first edge.
    //

    Vertex1 = &Graph->Vertices3[Vertex1Index];
    Vertex1->Edges ^= Edge;
    ++Vertex1->Degree;

    //
    // Insert the second edge.
    //

    Vertex2 = &Graph->Vertices3[Vertex2Index];
    Vertex2->Edges ^= Edge;
    ++Vertex2->Degree;
}

GRAPH_ADD_EDGE16 GraphAddEdge16;

_Use_decl_annotations_
VOID
GraphAddEdge16(
    PGRAPH Graph,
    EDGE16 Edge,
    VERTEX16 Vertex1Index,
    VERTEX16 Vertex2Index
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
    PVERTEX163 Vertex1;
    PVERTEX163 Vertex2;

#ifdef _DEBUG
    //
    // Invariant checks:
    //
    //      - Vertex1Index should be less than the number of vertices.
    //      - Vertex2Index should be less than the number of vertices.
    //      - Edge should be less than the number of edges.
    //      - The graph must not have started deletions.
    //

    ASSERT(Vertex1Index < Graph->NumberOfVertices);
    ASSERT(Vertex2Index < Graph->NumberOfVertices);
    ASSERT(Edge < Graph->NumberOfEdges);
    ASSERT(!Graph->Flags.Shrinking);
#endif

    //
    // Insert the first edge.
    //

    Vertex1 = &Graph->Vertices163[Vertex1Index];
    Vertex1->Edges ^= Edge;
#ifdef _DEBUG
    ASSERT(Vertex1->Degree <= (0x00007fff - 1));
#endif
    ++Vertex1->Degree;

    //
    // Insert the second edge.
    //

    Vertex2 = &Graph->Vertices163[Vertex2Index];
    Vertex2->Edges ^= Edge;
#ifdef _DEBUG
    ASSERT(Vertex2->Degree <= (0x00007fff - 1));
#endif
    ++Vertex2->Degree;
}


GRAPH_ADD_KEYS GraphAddKeys3;

_Use_decl_annotations_
HRESULT
GraphAddKeys3(
    PGRAPH Graph,
    ULONG NumberOfKeys,
    PKEY Keys
    )
/*++

Routine Description:

    Add all keys to the hypergraph using the unique seeds to hash each key into
    two vertex values, connected by a "hyper-edge".  This implementation uses
    the newer "Ex" version of the seeded hash routines.

Arguments:

    Graph - Supplies a pointer to the graph for which the keys will be added.

    NumberOfKeys - Supplies the number of keys.

    Keys - Supplies the base address of the keys array.

Return Value:

    S_OK - Success.

    PH_E_GRAPH_VERTEX_COLLISION_FAILURE - The graph encountered two vertices
        that, when masked, were identical.

--*/
{
    KEY Key = 0;
    EDGE Edge;
    PEDGE Edges;
    ULONG Mask;
    HRESULT Result;
    ULARGE_INTEGER Hash;
    PULONGLONG VertexPairs;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_SEEDED_HASH_EX SeededHashEx;

    DECL_GRAPH_COUNTER_LOCAL_VARS();

    //
    // Initialize aliases.
    //

    Table = Graph->Context->Table;
    Mask = Table->HashMask;
    SeededHashEx = SeededHashExRoutines[Table->HashFunctionId];
    Edges = (PEDGE)Keys;

    //
    // Enumerate all keys in the input set, hash them into two unique vertices,
    // then add them to the hypergraph.
    //

    Result = S_OK;
    VertexPairs = (PULONGLONG)Graph->VertexPairs;

    START_GRAPH_COUNTER();

    for (Edge = 0; Edge < NumberOfKeys; Edge++) {
        Key = *Edges++;

        Hash.QuadPart = SeededHashEx(Key, &Graph->FirstSeed, Mask);

        if (Hash.HighPart == Hash.LowPart) {
            Result = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
            break;
        }

        *VertexPairs++ = Hash.QuadPart;

        //
        // Add the edge to the graph connecting these two vertices.
        //

        GraphAddEdge3(Graph, Edge, Hash.LowPart, Hash.HighPart);
    }

    STOP_GRAPH_COUNTER(AddKeys);

    EVENT_WRITE_GRAPH(AddKeys);

    return Result;
}

GRAPH_ADD_KEYS GraphAddKeys16;

_Use_decl_annotations_
HRESULT
GraphAddKeys16(
    PGRAPH Graph,
    ULONG NumberOfKeys,
    PKEY Keys
    )
/*++

Routine Description:

    Add all keys to the hypergraph using the unique seeds to hash each key into
    two vertex values, connected by a "hyper-edge".  This implementation uses
    the newer "Ex" version of the seeded hash routines.

Arguments:

    Graph - Supplies a pointer to the graph for which the keys will be added.

    NumberOfKeys - Supplies the number of keys.

    Keys - Supplies the base address of the keys array.

Return Value:

    S_OK - Success.

    PH_E_GRAPH_VERTEX_COLLISION_FAILURE - The graph encountered two vertices
        that, when masked, were identical.

--*/
{
    KEY Key = 0;
    EDGE16 Edge;
    PEDGE Edges;
    USHORT Mask;
    HRESULT Result;
    ULONG_INTEGER Hash;
    PULONG VertexPairs;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_SEEDED_HASH16_EX SeededHashEx;

    DECL_GRAPH_COUNTER_LOCAL_VARS();

    //
    // Initialize aliases.
    //

    Table = Graph->Context->Table;
    Mask = (USHORT)Table->HashMask;
    SeededHashEx = SeededHash16ExRoutines[Table->HashFunctionId];
    Edges = (PEDGE)Keys;

    //
    // Enumerate all keys in the input set, hash them into two unique vertices,
    // then add them to the hypergraph.
    //

    Result = S_OK;
    VertexPairs = (PULONG)Graph->Vertex16Pairs;

    START_GRAPH_COUNTER();

    for (Edge = 0; Edge < NumberOfKeys; Edge++) {
        Key = *Edges++;

        Hash.LongPart = SeededHashEx(Key, &Graph->FirstSeed, Mask);

        if (Hash.HighPart == Hash.LowPart) {
            Result = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
            break;
        }

        *VertexPairs++ = Hash.LongPart;

        //
        // Add the edge to the graph connecting these two vertices.
        //

        GraphAddEdge16(Graph, Edge, Hash.LowPart, Hash.HighPart);
    }

    STOP_GRAPH_COUNTER(AddKeys);

    EVENT_WRITE_GRAPH(AddKeys);

    return Result;
}

GRAPH_ADD_KEYS GraphHashKeysThenAdd3;

_Use_decl_annotations_
HRESULT
GraphHashKeysThenAdd3(
    PGRAPH Graph,
    ULONG NumberOfKeys,
    PKEY Keys
    )
/*++

Routine Description:

    This routine is a drop-in replacement for Graph->Vtbl->AddKeys (handled by
    GraphInitialize()), and is responsible for hashing all keys into vertices
    first, then adding all resulting vertices to the graph.  This differs from
    the normal GraphAddKeys() behavior, which hashes a key into two vertices
    and immediately adds them to the graph via GraphAddEdge().  (This routine
    loops over the keys twice; once to construct all the vertices, then again
    to add them all to the graph.)

    The motivation behind this routine is to separate out the action of hashing
    keys versus adding them to the graph to better analyze performance.

Arguments:

    Graph - Supplies a pointer to the graph for which the keys will be added.

    NumberOfKeys - Supplies the number of keys.

    Keys - Supplies the base address of the keys array.

Return Value:

    S_OK - Success.

    PH_E_GRAPH_VERTEX_COLLISION_FAILURE - The graph encountered two vertices
        that, when masked, were identical.

        N.B. Unlike GraphAddKeys(), when this code is returned, none of the
             vertices will have been added to the graph at this point (versus
             having the graph in a partially-constructed state).  This has no
             impact on the behavior of the graph solving, other than potentially
             being faster overall for graphs encountering a lot of collisions
             (because the overhead of writing to all the graph's First/Next
             arrays will have been avoided).

--*/
{
    EDGE Edge;
    HRESULT Result;
    VERTEX_PAIR VertexPair;
    PVERTEX_PAIR VertexPairs;

    DECL_GRAPH_COUNTER_LOCAL_VARS();

    //
    // Attempt to hash the keys first.
    //

    Result = Graph->Vtbl->HashKeys(Graph, NumberOfKeys, Keys);
    if (FAILED(Result)) {
        return Result;
    }

    //
    // No vertex collisions were encountered.  All the vertex pairs have been
    // written to Graph->VertexPairs, indexed by Edge.  Loop through the number
    // of edges and add the vertices to the graph.
    //

    VertexPairs = Graph->VertexPairs;

    START_GRAPH_COUNTER();

    for (Edge = 0; Edge < NumberOfKeys; Edge++) {
        VertexPair = *(VertexPairs++);
        GraphAddEdge3(Graph, Edge, VertexPair.Vertex1, VertexPair.Vertex2);
    }

    STOP_GRAPH_COUNTER(AddHashedKeys);

    EVENT_WRITE_GRAPH_ADD_HASHED_KEYS();

    return S_OK;
}

GRAPH_HASH_KEYS GraphHashKeys16;

_Use_decl_annotations_
HRESULT
GraphHashKeys16(
    PGRAPH Graph,
    ULONG NumberOfKeys,
    PKEY Keys
    )
/*++

Routine Description:

    This routine hashes all keys into vertices without adding the resulting
    vertices to the graph.  It is used by GraphHashKeysThenAdd().

Arguments:

    Graph - Supplies a pointer to the graph for which the hash values will be
        created.

    NumberOfKeys - Supplies the number of keys.

    Keys - Supplies the base address of the keys array.

Return Value:

    S_OK - Success.

    PH_E_GRAPH_VERTEX_COLLISION_FAILURE - The graph encountered two vertices
        that, when masked, were identical.

--*/
{
    KEY Key = 0;
    EDGE Edge;
    USHORT Mask;
    PEDGE Edges;
    HRESULT Result;
    VERTEX16_PAIR Hash;
    PULONG VertexPairs;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_SEEDED_HASH16_EX SeededHashEx;

    DECL_GRAPH_COUNTER_LOCAL_VARS();

    //
    // Initialize aliases.
    //

    Result = S_OK;
    Table = Graph->Context->Table;
    Mask = (USHORT)Table->HashMask;
    SeededHashEx = SeededHash16ExRoutines[Table->HashFunctionId];
    Edges = (PEDGE)Keys;

    //
    // Sanity check we can enumerate over the vertex pair elements via a
    // ULONG pointer.
    //

    C_ASSERT(sizeof(*VertexPairs) == sizeof(*Graph->Vertex16Pairs));
    VertexPairs = (PULONG)Graph->Vertex16Pairs;

    //
    // Enumerate all keys in the input set and hash them into the vertex arrays.
    //

    START_GRAPH_COUNTER();

    for (Edge = 0; Edge < NumberOfKeys; Edge++) {
        Key = *Edges++;

        Hash.AsULong = SeededHashEx(Key, &Graph->FirstSeed, Mask);

        if (Hash.Vertex1 == Hash.Vertex2) {
            Result = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
            break;
        }

        *VertexPairs++ = Hash.AsULong;
    }

    STOP_GRAPH_COUNTER(HashKeys);

    EVENT_WRITE_GRAPH(HashKeys);

    Result = GraphPostHashKeys(Result, Graph);

    return Result;
}


GRAPH_ADD_KEYS GraphHashKeysThenAdd16;

_Use_decl_annotations_
HRESULT
GraphHashKeysThenAdd16(
    PGRAPH Graph,
    ULONG NumberOfKeys,
    PKEY Keys
    )
/*++

Routine Description:

    This routine is a drop-in replacement for Graph->Vtbl->AddKeys (handled by
    GraphInitialize()), and is responsible for hashing all keys into vertices
    first, then adding all resulting vertices to the graph.  This differs from
    the normal GraphAddKeys() behavior, which hashes a key into two vertices
    and immediately adds them to the graph via GraphAddEdge().  (This routine
    loops over the keys twice; once to construct all the vertices, then again
    to add them all to the graph.)

    The motivation behind this routine is to separate out the action of hashing
    keys versus adding them to the graph to better analyze performance.

Arguments:

    Graph - Supplies a pointer to the graph for which the keys will be added.

    NumberOfKeys - Supplies the number of keys.

    Keys - Supplies the base address of the keys array.

Return Value:

    S_OK - Success.

    PH_E_GRAPH_VERTEX_COLLISION_FAILURE - The graph encountered two vertices
        that, when masked, were identical.

        N.B. Unlike GraphAddKeys(), when this code is returned, none of the
             vertices will have been added to the graph at this point (versus
             having the graph in a partially-constructed state).  This has no
             impact on the behavior of the graph solving, other than potentially
             being faster overall for graphs encountering a lot of collisions
             (because the overhead of writing to all the graph's First/Next
             arrays will have been avoided).

--*/
{
    EDGE16 Edge;
    HRESULT Result;
    VERTEX16_PAIR VertexPair;
    PVERTEX16_PAIR VertexPairs;

    DECL_GRAPH_COUNTER_LOCAL_VARS();

    //
    // Attempt to hash the keys first.
    //

    Result = Graph->Vtbl->HashKeys(Graph, NumberOfKeys, Keys);
    if (FAILED(Result)) {
        return Result;
    }

    //
    // No vertex collisions were encountered.  All the vertex pairs have been
    // written to Graph->VertexPairs, indexed by Edge.  Loop through the number
    // of edges and add the vertices to the graph.
    //

    VertexPairs = Graph->Vertex16Pairs;

    START_GRAPH_COUNTER();

    for (Edge = 0; Edge < NumberOfKeys; Edge++) {
        VertexPair = *(VertexPairs++);
        GraphAddEdge16(Graph, Edge, VertexPair.Vertex1, VertexPair.Vertex2);
    }

    STOP_GRAPH_COUNTER(AddHashedKeys);

    EVENT_WRITE_GRAPH_ADD_HASHED_KEYS();

    return S_OK;
}


VOID
GraphRemoveVertex3(
    _In_ PGRAPH Graph,
    _In_ VERTEX VertexIndex
    )
{
    EDGE Edge;
    PEDGE3 Edge3;
    LONG OrderIndex;
    PVERTEX3 Vertex;
    PVERTEX3 Vertex1;
    PVERTEX3 Vertex2;

    Vertex = &Graph->Vertices3[VertexIndex];
    if (Vertex->Degree != 1) {
        return;
    }

    ASSERT(Vertex->Degree > 0);

    Edge = Vertex->Edges;
    Edge3 = &Graph->Edges3[Edge];

    if (IsEmpty(Edge3->Vertex1)) {
        ASSERT(IsEmpty(Edge3->Vertex2));
        return;
    } else if (IsEmpty(Edge3->Vertex2)) {
        ASSERT(IsEmpty(Edge3->Vertex1));
        return;
    }

    Vertex1 = &Graph->Vertices3[Edge3->Vertex1];
    if (Vertex1->Degree >= 1) {
        Vertex1->Edges ^= Edge;
        --Vertex1->Degree;
    }

    Vertex2 = &Graph->Vertices3[Edge3->Vertex2];
    if (Vertex2->Degree >= 1) {
        Vertex2->Edges ^= Edge;
        --Vertex2->Degree;
    }

    Graph->DeletedEdgeCount++;
    OrderIndex = --Graph->OrderIndex;
#ifdef _DEBUG
    ASSERT(Graph->DeletedEdgeCount <= Graph->NumberOfEdges);
    ASSERT(OrderIndex >= 0);
#endif
    Graph->Order[OrderIndex] = Edge;
}

VOID
GraphRemoveVertex16(
    _In_ PGRAPH Graph,
    _In_ VERTEX VertexIndex
    )
{
    EDGE Edge;
    PEDGE163 Edge3;
    ORDER16 OrderIndex;
    PVERTEX163 Vertex;
    PVERTEX163 Vertex1;
    PVERTEX163 Vertex2;

    Vertex = &Graph->Vertices163[VertexIndex];
    if (Vertex->Degree != 1) {
        return;
    }

    ASSERT(Vertex->Degree > 0);

    Edge = Vertex->Edges;
    Edge3 = &Graph->Edges163[Edge];

    if (IsEmpty(Edge3->Vertex1)) {
        ASSERT(IsEmpty(Edge3->Vertex2));
        return;
    } else if (IsEmpty(Edge3->Vertex2)) {
        ASSERT(IsEmpty(Edge3->Vertex1));
        return;
    }

    Vertex1 = &Graph->Vertices163[Edge3->Vertex1];
    if (Vertex1->Degree >= 1) {
        Vertex1->Edges ^= Edge;
        --Vertex1->Degree;
    }

    Vertex2 = &Graph->Vertices163[Edge3->Vertex2];
    if (Vertex2->Degree >= 1) {
        Vertex2->Edges ^= Edge;
        --Vertex2->Degree;
    }

    Graph->DeletedEdgeCount++;
    OrderIndex = --Graph->Order16Index;
#ifdef _DEBUG
    ASSERT(Graph->DeletedEdgeCount <= Graph->NumberOfEdges);
    ASSERT(OrderIndex >= 0);
    ASSERT(Edge <= 0x00007fff);
#endif
    Graph->Order16[OrderIndex] = (SHORT)Edge;
}

GRAPH_IS_ACYCLIC GraphIsAcyclic3;

_Use_decl_annotations_
HRESULT
GraphIsAcyclic3(
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
    LONG Index;
    ULONG EdgeIndex;
    VERTEX Vertex;
    PEDGE3 OtherEdge;
    BOOLEAN IsAcyclic;
    ULONG NumberOfKeys;
    ULONG NumberOfEdges;
    ULONG NumberOfVertices;
    ULONG NumberOfEdgesDeleted;

    DECL_GRAPH_COUNTER_LOCAL_VARS();

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

    Graph->OrderIndex = (LONG)NumberOfKeys;

    //
    // Toggle the shrinking bit to indicate we've started edge deletion.
    //

    Graph->Flags.Shrinking = TRUE;

    //
    // Enumerate through all vertices in the graph and attempt to delete those
    // connected by edges that have degree 1.
    //

    START_GRAPH_COUNTER();

    for (Vertex = 0; Vertex < NumberOfVertices; Vertex++) {
        GraphRemoveVertex3(Graph, Vertex);
    }

    for (Index = (LONG)NumberOfKeys;
         Graph->OrderIndex > 0 && Index > Graph->OrderIndex;
         NOTHING)
    {
        EdgeIndex = Graph->Order[--Index];
        OtherEdge = &Graph->Edges3[EdgeIndex];
        GraphRemoveVertex3(Graph, OtherEdge->Vertex1);
        GraphRemoveVertex3(Graph, OtherEdge->Vertex2);
    }

    ASSERT(Graph->OrderIndex >= 0);

    NumberOfEdgesDeleted = Graph->DeletedEdgeCount;
    IsAcyclic = (NumberOfKeys == NumberOfEdgesDeleted);

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

    EVENT_WRITE_GRAPH_IS_ACYCLIC();

    return (IsAcyclic ? S_OK : PH_E_GRAPH_CYCLIC_FAILURE);
}

GRAPH_IS_ACYCLIC GraphIsAcyclic16;

_Use_decl_annotations_
HRESULT
GraphIsAcyclic16(
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
    SHORT Index;
    USHORT EdgeIndex;
    VERTEX Vertex;
    PEDGE163 OtherEdge;
    BOOLEAN IsAcyclic;
    ULONG NumberOfKeys;
    ULONG NumberOfEdges;
    ULONG NumberOfVertices;
    ULONG NumberOfEdgesDeleted;

    DECL_GRAPH_COUNTER_LOCAL_VARS();

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

    Graph->Order16Index = (SHORT)NumberOfKeys;

    //
    // Toggle the shrinking bit to indicate we've started edge deletion.
    //

    Graph->Flags.Shrinking = TRUE;

    //
    // Enumerate through all vertices in the graph and attempt to delete those
    // connected by edges that have degree 1.
    //

    START_GRAPH_COUNTER();

    for (Vertex = 0; Vertex < NumberOfVertices; Vertex++) {
        GraphRemoveVertex16(Graph, Vertex);
    }

    for (Index = (SHORT)NumberOfKeys;
         Graph->Order16Index > 0 && Index > Graph->Order16Index;
         NOTHING)
    {
        EdgeIndex = Graph->Order16[--Index];
        OtherEdge = &Graph->Edges163[EdgeIndex];
        GraphRemoveVertex16(Graph, OtherEdge->Vertex1);
        GraphRemoveVertex16(Graph, OtherEdge->Vertex2);
    }

    ASSERT(Graph->Order16Index >= 0);

    NumberOfEdgesDeleted = Graph->DeletedEdgeCount;
    IsAcyclic = (NumberOfKeys == NumberOfEdgesDeleted);

    //
    // Invariant check: if we're acyclic, the order index should be zero (as all
    // keys were deleted once), greater than zero otherwise.
    //

    if (IsAcyclic) {
        ASSERT(Graph->Order16Index == 0);
    } else {
        ASSERT(Graph->Order16Index > 0);
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

    EVENT_WRITE_GRAPH_IS_ACYCLIC();

    return (IsAcyclic ? S_OK : PH_E_GRAPH_CYCLIC_FAILURE);
}

GRAPH_ASSIGN GraphAssign3;

_Use_decl_annotations_
HRESULT
GraphAssign3(
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
    PEDGE3 Edge3;
    ULONG Index;
    LONG Order;
    ULONG Assigned;
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

    EVENT_WRITE_GRAPH_ASSIGN_START();

    //
    // Walk the graph and assign values.
    //

    START_GRAPH_COUNTER();

    for (Index = 0; Index < NumberOfKeys; Index++) {

        //
        // Obtain the edge for the deletion order at this index.
        //

        Order = Graph->Order[Index];
        Edge3 = &Graph->Edges3[Order];

        if (!IsVisitedVertex3(Graph, Edge3->Vertex1)) {
            Vertex1 = Edge3->Vertex1;
            Vertex2 = Edge3->Vertex2;
        } else {
            Vertex1 = Edge3->Vertex2;
            Vertex2 = Edge3->Vertex1;
        }

        Assigned = Order - Graph->Assigned[Vertex2];
        if (Assigned >= NumberOfEdges) {
            Assigned += NumberOfEdges;
        }

        ASSERT(Graph->Assigned[Vertex1] == INITIAL_ASSIGNMENT_VALUE);
        Graph->Assigned[Vertex1] = Assigned;

        //
        // Set both vertices as visited.
        //

        RegisterVertexVisit3(Graph, Vertex1);
        RegisterVertexVisit3(Graph, Vertex2);
    }

    STOP_GRAPH_COUNTER(Assign);

    EVENT_WRITE_GRAPH_ASSIGN_STOP();

    EVENT_WRITE_GRAPH_ASSIGN_RESULT();

    return S_OK;
}

GRAPH_ASSIGN GraphAssign16;

_Use_decl_annotations_
HRESULT
GraphAssign16(
    PGRAPH Graph
    )
/*++

Routine Description:

    This routine is called after a graph has determined to be acyclic.  It is
    responsible for walking the graph and assigning values to edges in order to
    complete the perfect hash solution.  This implementation deals with USHORT
    assigned arrays, i.e. tables with vertex counts less than or equal to 65535.

Arguments:

    Graph - Supplies a pointer to the graph to operate on.

Return Value:

    None.

--*/
{
    PEDGE163 Edge3;
    SHORT Order;
    USHORT Index;
    VERTEX16 Vertex1;
    VERTEX16 Vertex2;
    USHORT Assigned;
    USHORT NumberOfKeys;
    USHORT NumberOfEdges;
    PPERFECT_HASH_TABLE Table;

    DECL_GRAPH_COUNTER_LOCAL_VARS();

    //
    // Initialize aliases.
    //

    Table = Graph->Context->Table;
    NumberOfKeys = (USHORT)Graph->NumberOfKeys;
    NumberOfEdges = (USHORT)Graph->NumberOfEdges;

    //
    // Invariant check: we should only be called on graphs that have already
    // been determined to be invariant.
    //

    ASSERT(Graph->Flags.IsAcyclic);

    EVENT_WRITE_GRAPH_ASSIGN_START();

    //
    // Walk the graph and assign values.
    //

    START_GRAPH_COUNTER();

    for (Index = 0; Index < NumberOfKeys; Index++) {

        //
        // Obtain the edge for the deletion order at this index.
        //

        Order = Graph->Order16[Index];
        Edge3 = &Graph->Edges163[Order];

        if (!IsVisitedVertex16(Graph, Edge3->Vertex1)) {
            Vertex1 = Edge3->Vertex1;
            Vertex2 = Edge3->Vertex2;
        } else {
            Vertex1 = Edge3->Vertex2;
            Vertex2 = Edge3->Vertex1;
        }

        Assigned = Order - Graph->Assigned16[Vertex2];
        if (Assigned >= NumberOfEdges) {
            Assigned += NumberOfEdges;
        }

        ASSERT(Graph->Assigned16[Vertex1] == INITIAL_ASSIGNMENT_VALUE);
        Graph->Assigned16[Vertex1] = Assigned;

        //
        // Set both vertices as visited.
        //

        RegisterVertex16Visit(Graph, Vertex1);
        RegisterVertex16Visit(Graph, Vertex2);
    }

    STOP_GRAPH_COUNTER(Assign);

    EVENT_WRITE_GRAPH_ASSIGN_STOP();

    EVENT_WRITE_GRAPH_ASSIGN_RESULT();

    return S_OK;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
