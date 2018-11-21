/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    GraphImpl.h

Abstract:

    This is the header file for the implementation of the graph module for the
    perfect hash library.  It defines the function typedefs required by a graph
    implementation: GraphAddEdge(), IsGraphAcyclic(), and GraphAssign().

    N.B. The refactoring of the original Graph.[ch] modules into separate
         implementation-specific modules is a work-in-progress.  The general
         idea is that the Graph.[ch] modules contain the top-level defs plus
         all the COM/vtbl scaffolding; whereas the GraphImpl*.[ch] modules
         focus solely on implementing the hypergraph requirements.

--*/

#include "stdafx.h"

//
// Define the top-level function types required by the graph implementation.
//

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
_Must_inspect_result_
BOOLEAN
(NTAPI IS_GRAPH_ACYCLIC)(
    _In_ PGRAPH Graph
    );
typedef IS_GRAPH_ACYCLIC *PIS_GRAPH_ACYCLIC;
extern IS_GRAPH_ACYCLIC IsGraphAcyclic;

typedef
VOID
(NTAPI GRAPH_ASSIGN)(
    _In_ PGRAPH Graph
    );
typedef GRAPH_ASSIGN *PGRAPH_ASSIGN;
extern GRAPH_ASSIGN GraphAssign;

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
