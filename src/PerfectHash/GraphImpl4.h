/*++

Copyright (c) 2026 Trent Nelson <trent@trent.me>

Module Name:

    GraphImpl4.h

Abstract:

    Private declarations for the experimental C++ graph implementation used by
    the CPU solver backend.

--*/

#pragma once

#include "stdafx.h"

EXTERN_C HRESULT NTAPI GraphAddKeys4(PGRAPH Graph, ULONG NumberOfKeys, PKEY Keys);
EXTERN_C HRESULT NTAPI GraphHashKeys4(PGRAPH Graph, ULONG NumberOfKeys, PKEY Keys);
EXTERN_C HRESULT NTAPI GraphHashKeysThenAdd4(PGRAPH Graph,
                                             ULONG NumberOfKeys,
                                             PKEY Keys);
EXTERN_C HRESULT NTAPI GraphIsAcyclic4(PGRAPH Graph);
EXTERN_C HRESULT NTAPI GraphAssign4(PGRAPH Graph);
EXTERN_C HRESULT NTAPI GraphVerify4(PGRAPH Graph);
EXTERN_C HRESULT NTAPI PerfectHashTableIndexImpl4Chm01(PPERFECT_HASH_TABLE Table,
                                                       ULONG Key,
                                                       PULONG Index);
//
// The Key argument must already be the GraphImpl4 effective key.  For loaded
// downsized 64-bit tables, callers must pass the result of the persisted
// composed outer-bitmap ExtractBits64 step, not the original raw input key.
//
EXTERN_C HRESULT NTAPI
PerfectHashTableIndexImpl4EffectiveKeyChm01(PPERFECT_HASH_TABLE Table,
                                            ULONG Key,
                                            PULONG Index);

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
