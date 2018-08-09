/*++

Copyright (c) 2016 Trent Nelson <trent@trent.me>

Module Name:

    RtlErrorHandling.c

Abstract:

    This module is responsible for generic error handling routines.

--*/

#include "stdafx.h"

RTL_PRINT_SYS_ERROR RtlPrintSysError;

_Use_decl_annotations_
HRESULT
RtlPrintSysError(
    PRTL Rtl,
    PCSZ FunctionName,
    PCSZ FileName,
    ULONG LineNumber
    )
{
    BOOL Success;
    LONG Result1;
    ULONG Result2;
    ULONG Flags;
    PCHAR Buffer;
    PCHAR BaseBuffer;
    PCHAR EndBuffer;
    ULONG LastError;
    ULONG LanguageId;
    ULONG BytesWritten;
    HRESULT Result = S_OK;
    ULONG_PTR BytesToWrite;
    LONG_PTR SizeOfBufferInChars;

    LastError = GetLastError();

    AcquireRtlSysErrorMessageBufferLock(Rtl);

    Buffer = BaseBuffer = Rtl->SysErrorMessageBuffer;
    EndBuffer = Buffer + Rtl->SizeOfSysErrorMessageBufferInBytes;

    //
    // The following is unnecessary when dealing with bytes, but will allow
    // easy conversion into a WCHAR version at a later date.
    //

    SizeOfBufferInChars = (LONG_PTR)(
        (LONG_PTR)Rtl->SizeOfSysErrorMessageBufferInBytes *
        (LONG_PTR)sizeof(*Buffer)
    );

    Result1 = Rtl->sprintf_s(Buffer,
                             (ULONG)SizeOfBufferInChars,
                             "%s: %lu: %s failed with error: %lu.  ",
                             FileName,
                             LineNumber,
                             FunctionName,
                             LastError);

    if (Result1 <= 0) {
        OutputDebugStringA("RtlPrintSysError: Rtl->sprintf_s() failed.\n");
        goto Error;
    }

    Buffer += Result1;
    SizeOfBufferInChars -= Result1;

    Flags = (FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS);

    LanguageId = MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),

    Result2 = FormatMessageA(Flags,
                             NULL,
                             LastError,
                             LanguageId,
                             (PSTR)Buffer,
                             (ULONG)SizeOfBufferInChars,
                             NULL);

    if (!Result2) {
        OutputDebugStringA("RtlPrintSysError: FormatMessageA() failed.\n");
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
    ASSERT(BytesToWrite <= Rtl->SizeOfSysErrorMessageBufferInBytes);

    Success = WriteFile(Rtl->SysErrorOutputHandle,
                        BaseBuffer,
                        (ULONG)BytesToWrite,
                        &BytesWritten,
                        NULL);

    if (!Success) {
        OutputDebugStringA("RtlPrintSysError: WriteFile() failed.\n");
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

    SecureZeroMemory(Rtl->SysErrorMessageBuffer,
                     Rtl->SizeOfSysErrorMessageBufferInBytes);

    ReleaseRtlSysErrorMessageBufferLock(Rtl);

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
