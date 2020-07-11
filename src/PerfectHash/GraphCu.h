/*++

Copyright (c) 2020 Trent Nelson <trent@trent.me>

Module Name:

    GraphCu.h

Abstract:

    This is the header file for the GraphCu module of the perfect hash library.

--*/

#include "stdafx.h"

//
// GraphCu vtbl.
//

typedef struct _GRAPH_CU_VTBL {
    DECLARE_COMPONENT_VTBL_HEADER(GRAPH);

    //
    // These methods are common to both CPU and GPU graph implementations.
    //

    PGRAPH_SET_INFO SetInfo;
    PGRAPH_ENTER_SOLVING_LOOP EnterSolvingLoop;
    PGRAPH_VERIFY Verify;

} GRAPH_CU_VTBL;
typedef GRAPH_CU_VTBL *PGRAPH_CU_VTBL;

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
