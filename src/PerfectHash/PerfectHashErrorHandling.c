/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashErrorHandling.c

Abstract:

    This module is responsible for error handling related to the perfect hash
    library.  Routines are provided for printing errors and messages, and
    getting string representations of error codes.

--*/

#include "stdafx.h"

PERFECT_HASH_PRINT_ERROR PerfectHashPrintError;

_Use_decl_annotations_
HRESULT
PerfectHashPrintError(
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

    LanguageId = MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT);

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

FORCEINLINE
BOOLEAN
DoesErrorCodeWantAlgoHashMaskTableAppended(
    _In_ ULONG ErrorCode
    )
{
    HRESULT Code;
    BOOLEAN Result;

    Code = (HRESULT)ErrorCode;

    //
    // We append the algo/hash/mask table text for usage strings, currently.
    //

    Result = (
        Code == PH_MSG_PERFECT_HASH_BULK_CREATE_EXE_USAGE ||
        Code == PH_MSG_PERFECT_HASH_CREATE_EXE_USAGE
    );

    return Result;
}

PERFECT_HASH_PRINT_MESSAGE PerfectHashPrintMessage;

_Use_decl_annotations_
HRESULT
PerfectHashPrintMessage(
    ULONG Code,
    ...
    )
{
    BOOL Success;
    ULONG Flags;
    ULONG Count;
    PSTR Buffer;
    ULONG LanguageId;
    ULONG BytesWritten;
    HRESULT Result = S_OK;
    LONG_PTR SizeOfBufferInBytes;
    va_list Args;

    va_start(Args, Code);

    LanguageId = MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT);

    Flags = (
        FORMAT_MESSAGE_ALLOCATE_BUFFER |
        FORMAT_MESSAGE_FROM_HMODULE    |
        FORMAT_MESSAGE_FROM_SYSTEM
    );

    Count = FormatMessageA(Flags,
                           PerfectHashModule,
                           Code,
                           LanguageId,
                           (PSTR)&Buffer,
                           0,
                           &Args);

    if (!Count) {
        SYS_ERROR(FormatMessageA);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // The following is unnecessary when dealing with bytes, but will allow
    // easy conversion into a WCHAR version at a later date.
    //

    SizeOfBufferInBytes = Count * sizeof(*Buffer);

    Success = WriteFile(GetStdHandle(STD_ERROR_HANDLE),
                        Buffer,
                        (ULONG)SizeOfBufferInBytes,
                        &BytesWritten,
                        NULL);

    if (!Success) {
        SYS_ERROR(FormatMessageA);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    if (BytesWritten != (ULONG)SizeOfBufferInBytes) {
        PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
    }

    //
    // If the incoming error code was a usage string, print the table of
    // algorithms, hash functions and masking types.
    //

    if (DoesErrorCodeWantAlgoHashMaskTableAppended(Code)) {
        ULONG NewCode;

        NewCode = PH_MSG_PERFECT_HASH_ALGO_HASH_MASK_NAMES;
        Result = PerfectHashPrintMessage(NewCode);
        if (FAILED(Result)) {
            goto Error;
        }
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

    if (Buffer) {
        if (LocalFree(Buffer)) {
            SYS_ERROR(LocalFree);
            Result = PH_E_SYSTEM_CALL_FAILED;
        }
        Buffer = NULL;
    }

    return Result;
}


static const ERROR_CODE_SYMBOL_NAME CommonErrorCodes[] = {
    (HRESULT) S_OK, "S_OK",
    (HRESULT) S_FALSE, "S_FALSE",
    (HRESULT) E_OUTOFMEMORY, "E_OUTOFMEMORY",
    (HRESULT) E_UNEXPECTED, "E_UNEXPECTED",
    (HRESULT) E_POINTER, "E_POINTER",
    (HRESULT) E_INVALIDARG, "E_INVALIDARG",
    (HRESULT) E_FAIL, "E_FAIL",
    (HRESULT) 0xFFFFFFFF, NULL
};

//
// warning C4820: '<unnamed-tag>': '4' bytes padding added after
//      data member 'MessageId'
//

#pragma warning(push)
#pragma warning(disable: 4820)
#include "PerfectHashErrors.dbg"
#pragma warning(pop)

static const PCSZ UnknownErrorCode = "Unknown";

PERFECT_HASH_GET_ERROR_CODE_STRING PerfectHashGetErrorCodeString;

_Use_decl_annotations_
HRESULT
PerfectHashGetErrorCodeString(
    PRTL Rtl,
    HRESULT Code,
    PCSZ *StringPointer
    )
/*++

Routine Description:

    Gets the ASCII representation of an error code.

Arguments:

    Rtl - Supplies a pointer to an Rtl instance.

    Code - Supplies the error code for which the string representation is to
        be obtained.

    StringPointer - Receives the string representation of the error code.

Return Value:

    S_OK - Found error code.

    S_FALSE - Could not find error code.  *StringPointer will be set to a
        string "Unknown".

    E_POINTER - Rtl or StringPointer were NULL.

--*/
{
    ULONG Index;
    ULONG Count;
    PCERROR_CODE_SYMBOL_NAME Entry;

    //
    // Rtl isn't currently used, however, we may use it down the track if we
    // want to do a binary search via Rtl->bsearch() instead of the linear scan
    // on the larger PerfectHashErrors.dbg array.
    //

    UNREFERENCED_PARAMETER(Rtl);

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Rtl)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(StringPointer)) {
        return E_POINTER;
    }

    //
    // Linear scan through the common error codes first.
    //

    Entry = CommonErrorCodes;
    Count = ARRAYSIZE(CommonErrorCodes);

    for (Index = 0; Index < Count; Index++, Entry++) {
        if (Entry->MessageId == Code) {
            *StringPointer = Entry->SymbolicName;
            return S_OK;
        }
    }

    //
    // Linear scan through the PerfectHashErrors.dbg array.
    //

    Entry = (PCERROR_CODE_SYMBOL_NAME)PerfectHashErrorsSymbolicNames;
    Count = ARRAYSIZE(PerfectHashErrorsSymbolicNames);

    for (Index = 0; Index < Count; Index++, Entry++) {
        if (Entry->MessageId == Code) {
            *StringPointer = Entry->SymbolicName;
            return S_OK;
        }
    }

    //
    // If we get here, we haven't found a match, so return the unknown string.
    //

    *StringPointer = UnknownErrorCode;
    return S_FALSE;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
