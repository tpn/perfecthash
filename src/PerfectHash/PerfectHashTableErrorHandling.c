/*++

Copyright (c) 2016 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTableErrorHandling.c

Abstract:

    This module is responsible for error handling related to the perfect hash
    table library.

--*/

#include "stdafx.h"

PERFECT_HASH_TABLE_PRINT_ERROR PerfectHashTablePrintError;

_Use_decl_annotations_
HRESULT
PerfectHashTablePrintError(
    PRTL Rtl,
    PCSZ FunctionName,
    PCSZ FileName,
    ULONG LineNumber,
    ULONG Error
    )
{
    BOOL Success;
    LONG Result1;
    ULONG Result2;
    ULONG Flags;
    PCHAR Buffer;
    PCHAR BaseBuffer;
    PCHAR EndBuffer;
    ULONG LanguageId;
    ULONG BytesWritten;
    HRESULT Result = S_OK;
    ULONG_PTR BytesToWrite;
    LONG_PTR SizeOfBufferInBytes;
    LONG_PTR SizeOfBufferInChars;
    CHAR LocalBuffer[1024];

    SizeOfBufferInBytes = sizeof(LocalBuffer);
    BaseBuffer = Buffer = (PCHAR)&LocalBuffer;
    EndBuffer = Buffer + SizeOfBufferInBytes;

    //
    // The following is unnecessary when dealing with bytes, but will allow
    // easy conversion into a WCHAR version at a later date.
    //

    SizeOfBufferInChars = (LONG_PTR)(
        SizeOfBufferInBytes *
        (LONG_PTR)sizeof(*Buffer)
    );

    Result1 = Rtl->sprintf_s(Buffer,
                             (ULONG)SizeOfBufferInChars,
                             "%s: %lu: %s failed with error: %lu.  ",
                             FileName,
                             LineNumber,
                             FunctionName,
                             Error);

    if (Result1 <= 0) {
        OutputDebugStringA("PhtPrintError: Rtl->sprintf_s() failed.\n");
        goto Error;
    }

    Buffer += Result1;
    SizeOfBufferInChars -= Result1;

    Flags = (
        FORMAT_MESSAGE_FROM_HMODULE |
        FORMAT_MESSAGE_FROM_SYSTEM  |
        FORMAT_MESSAGE_IGNORE_INSERTS
    );

    LanguageId = MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),

    Result2 = FormatMessageA(Flags,
                             PerfectHashTableModule,
                             Error,
                             LanguageId,
                             (PSTR)Buffer,
                             (ULONG)SizeOfBufferInChars,
                             NULL);

    if (!Result2) {
        OutputDebugStringA("PhtPrintError: FormatMessageA() failed.\n");
        goto Error;
    }

    Buffer += Result2;
    SizeOfBufferInChars -= Result2;

    //
    // We want at least two characters left in the buffer for the \n and
    // trailing NULL.
    //

    ASSERT(SizeOfBufferInChars >= 2);
    ASSERT((ULONG_PTR)Buffer <= (ULONG_PTR)(EndBuffer - 2));

    *Buffer += '\n';
    *Buffer += '\0';

    ASSERT((ULONG_PTR)Buffer <= (ULONG_PTR)EndBuffer);

    BytesToWrite = RtlPointerToOffset(BaseBuffer, Buffer);
    ASSERT(BytesToWrite <= sizeof(LocalBuffer));

    Success = WriteFile(GetStdHandle(STD_ERROR_HANDLE),
                        BaseBuffer,
                        (ULONG)BytesToWrite,
                        &BytesWritten,
                        NULL);

    if (!Success) {
        OutputDebugStringA("PhtPrintError: WriteFile() failed.\n");
        goto Error;
    }

    //
    // We're done, finish up and return.
    //

    Result = S_OK;
    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
