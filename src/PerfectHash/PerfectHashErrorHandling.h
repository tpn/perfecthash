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

//
// warning C4820: '<unnamed-tag>': '4' bytes padding added after
//      data member 'MessageId'
//

#pragma warning(push)
#pragma warning(disable: 4820)

typedef struct _ERROR_CODE_SYMBOL_NAME {
    HRESULT MessageId;
    PCSZ SymbolicName;
} ERROR_CODE_SYMBOL_NAME;
typedef ERROR_CODE_SYMBOL_NAME *PERROR_CODE_SYMBOL_NAME;
typedef const ERROR_CODE_SYMBOL_NAME *PCERROR_CODE_SYMBOL_NAME;

#pragma warning(pop)

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI PERFECT_HASH_GET_ERROR_CODE_STRING)(
    _In_ PRTL Rtl,
    _In_ HRESULT Code,
    _Out_ PCSZ *String
    );
typedef PERFECT_HASH_GET_ERROR_CODE_STRING *PPERFECT_HASH_GET_ERROR_CODE_STRING;

#ifndef __INTELLISENSE__
extern PERFECT_HASH_PRINT_ERROR PerfectHashPrintError;
extern PERFECT_HASH_PRINT_MESSAGE PerfectHashPrintMessage;
extern PERFECT_HASH_GET_ERROR_CODE_STRING PerfectHashGetErrorCodeString;
#endif


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
