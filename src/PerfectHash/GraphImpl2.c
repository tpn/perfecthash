/*++

Copyright (c) 2020-2021 Trent Nelson <trent@trent.me>

Module Name:

    GraphImpl2.c

Abstract:

    This module is our second implementation of graph innards for the CHM
    algorithm of the perfect hash library.  It implements GraphAssign2(),
    which is a more efficient implementation of the graph assignment step
    present in GraphImpl1.c, inspired by NetBSD's nbperf utility.

    I've included the copyright from NetBSD nbperf's nbperf-chm.c* for this
    reason; thanks Joerg Sonnenberger <joerg at bec dot de>!

    [*]: https://github.com/tpn/nbperf/blob/master/nbperf-chm.c

--*/

/*	$NetBSD: nbperf-chm.c,v 1.3 2011/10/21 23:47:11 joerg Exp $	*/
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

    EventWriteGraphAssignStart(
        NULL,
        Graph->Attempt,
        Graph->NumberOfKeys,
        Graph->NumberOfVertices
    );

    //
    // Walk the graph and assign values.
    //

    START_GRAPH_COUNTER();

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

    EventWriteGraphAssignStop(
        NULL,
        Graph->Attempt,
        Graph->NumberOfKeys,
        Graph->NumberOfVertices,
        Graph->NumberOfEmptyVertices,
        0, // MaximumTraversalDepth
        0  // TotalTraversals
    );

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
