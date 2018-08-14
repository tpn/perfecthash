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
    PCSZ FunctionName,
    PCSZ FileName,
    ULONG LineNumber,
    ULONG Error
    )
{
    BOOL Success;
    ULONG Flags;
    ULONG Count;
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
    ULONG_PTR Args[5];
    const STRING Prefix = RTL_CONSTANT_STRING(
        "%1: %2!lu!: %3 failed with error: %4!lu! (0x%4!lx!).  "
    );

    Args[0] = (ULONG_PTR)FileName,
    Args[1] = (ULONG_PTR)LineNumber;
    Args[2] = (ULONG_PTR)FunctionName;
    Args[3] = (ULONG_PTR)Error;
    Args[4] = 0;

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

    Flags = FORMAT_MESSAGE_FROM_STRING | FORMAT_MESSAGE_ARGUMENT_ARRAY;

    LanguageId = MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),

    Count = FormatMessageA(Flags,
                           Prefix.Buffer,
                           0,
                           LanguageId,
                           (PSTR)Buffer,
                           (ULONG)SizeOfBufferInChars,
                           (va_list *)Args);

    if (!Count) {
        OutputDebugStringA("PhtPrintError: FormatMessageA() 1 failed.\n");
        goto Error;
    }

    Buffer += Count;
    SizeOfBufferInChars -= Count;

    Flags = (
        FORMAT_MESSAGE_FROM_HMODULE |
        FORMAT_MESSAGE_FROM_SYSTEM  |
        FORMAT_MESSAGE_IGNORE_INSERTS
    );

    Count = FormatMessageA(Flags,
                           PerfectHashModule,
                           Error,
                           LanguageId,
                           (PSTR)Buffer,
                           (ULONG)SizeOfBufferInChars,
                           NULL);

    if (!Count) {
        OutputDebugStringA("PhtPrintError: FormatMessageA() 2 failed.\n");
        goto Error;
    }

    Buffer += Count;
    SizeOfBufferInChars -= Count;

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
