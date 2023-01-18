/*++

Copyright (c) 2018-2023 Trent Nelson <trent@trent.me>

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

FORCEINLINE
BOOLEAN
IsVisitedVertex16(
    _In_ PGRAPH Graph,
    _In_ VERTEX16 Vertex
    )
{
    return TestGraphBit(VisitedVerticesBitmap, Vertex);
}

FORCEINLINE
VOID
RegisterVertex16Visit(
    _In_ PGRAPH Graph,
    _In_ VERTEX16 Vertex
    )
{
    SetGraphBit(VisitedVerticesBitmap, Vertex);
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
