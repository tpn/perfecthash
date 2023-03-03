/*++

Copyright (c) 2023 Trent Nelson <trent@trent.me>

Module Name:

    Chm02Private.h

Abstract:

    Private header file for the Chm02 components.

--*/

#include "stdafx.h"
#include "Chm01Private.h"

#define CU_RNG_DEFAULT PerfectHashCuRngPhilox43210Id

PREPARE_GRAPH_INFO PrepareGraphInfoChm02;
PERFECT_HASH_MAIN_WORK_CALLBACK ProcessGraphCallbackChm02;

//
// Internal methods private to Chm02.c and Chm02Compat.c.
//

HRESULT
InitializeCudaAndGraphsChm02(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    );


HRESULT
LoadPerfectHashTableImplChm02(
    _In_ PPERFECT_HASH_TABLE Table
    );

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
