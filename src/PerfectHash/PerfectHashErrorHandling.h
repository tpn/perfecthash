/*++

Copyright (c) 2016 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTableErrorHandling.h

Abstract:

    This is the private header file for the error handling component of the
    perfect hash table library.

--*/

#pragma once

#include "stdafx.h"

typedef
_Success_(return >= 0)
_Check_return_opt_
HRESULT
(NTAPI PERFECT_HASH_TABLE_PRINT_ERROR)(
    _In_ PCSZ FunctionName,
    _In_ PCSZ FileName,
    _In_opt_ ULONG LineNumber,
    _In_opt_ ULONG Error
    );
typedef PERFECT_HASH_TABLE_PRINT_ERROR *PPERFECT_HASH_TABLE_PRINT_ERROR;

extern PERFECT_HASH_TABLE_PRINT_ERROR PerfectHashTablePrintError;

#define SYS_ERROR(Name) \
    PerfectHashTablePrintError(#Name, __FILE__, __LINE__, GetLastError())

#define PH_ERROR(Name, Result) \
    PerfectHashTablePrintError(#Name, __FILE__, __LINE__, (ULONG)Result)

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
