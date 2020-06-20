/*++

Copyright (c) 2020 Trent Nelson <trent@trent.me>

Module Name:

    GraphCu.h

Abstract:

    This is the header file for the GraphCu module of the perfect hash library.

--*/

#include "stdafx.h"

//
// Initially, the GRAPH_CU_VTBL differed from the GRAPH_VTBL.  We've since
// combined them, so, just typedef now.
//

typedef GRAPH_VTBL GRAPH_CU_VTBL;
typedef GRAPH_CU_VTBL *PGRAPH_CU_VTBL;

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
