/*++

Copyright (c) 2018-2021 Trent Nelson <trent@trent.me>

Module Name:

    GraphImpl3.h

Abstract:

    This is the private header file for the 3rd implementation of graph
    functionality for the perfect hash library.

--*/

#include "stdafx.h"

FORCEINLINE
BOOLEAN
IsVisitedVertex3(
    _In_ PGRAPH Graph,
    _In_ VERTEX Vertex
    )
{
    return TestGraphBit(VisitedVerticesBitmap, Vertex);
}

FORCEINLINE
VOID
RegisterVertexVisit3(
    _In_ PGRAPH Graph,
    _In_ VERTEX Vertex
    )
{
    SetGraphBit(VisitedVerticesBitmap, Vertex);
}

#if 0
#ifdef _DEBUG

FORCEINLINE
VOID
RegisterEdgeDeletion3(
    _In_ PGRAPH Graph,
    _In_ EDGE Edge
    )
{
    LONG OrderIndex;
    //ASSERT(!TestGraphBit(DeletedEdgesBitmap, Edge));
    //SetGraphBit(DeletedEdgesBitmap, Edge);
    Graph->DeletedEdgeCount++;
    ASSERT(Graph->DeletedEdgeCount <= Graph->NumberOfEdges);
    OrderIndex = --Graph->OrderIndex;
    ASSERT(OrderIndex >= 0);
    Graph->Order[OrderIndex] = Edge;
}

#else

FORCEINLINE
VOID
RegisterEdgeDeletion3(
    _In_ PGRAPH Graph,
    _In_ EDGE Edge
    )
{
    ULONG OrderIndex;
    //SetGraphBit(DeletedEdgesBitmap, Edge);
    Graph->DeletedEdgeCount++;
    OrderIndex = --Graph->OrderIndex;
    Graph->Order[OrderIndex] = Edge;
}

#endif
#endif


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
