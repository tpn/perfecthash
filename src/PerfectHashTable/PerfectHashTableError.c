/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    HordeThump.c

Abstract:

    This module implements common/miscellaneous routines related to the
    HordeThump component.

--*/

#include "stdafx.h"

_Use_decl_annotations_
VOID
PrintError(
    LPWSTR FunctionName,
    LPWSTR FileName,
    ULONG LineNumber
    )
{
    ULONG Flags;
    ULONG Result;
    ULONG LastError;
    ULONG LanguageId;
    LPWSTR MessageBuffer = NULL;

    LastError = GetLastError();

    printf("%S: %lu: %S failed with error: %lu.  ",
           FileName,
           LineNumber,
           FunctionName,
           LastError);

    Flags = (
        FORMAT_MESSAGE_FROM_SYSTEM     |
        FORMAT_MESSAGE_IGNORE_INSERTS  |
        FORMAT_MESSAGE_ALLOCATE_BUFFER
    );

    LanguageId = MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),

    Result = FormatMessageW(Flags,
                            NULL,
                            LastError,
                            LanguageId,
                            (LPWSTR)&MessageBuffer,
                            0,
                            NULL);

    if (Result) {
        printf("%S", MessageBuffer);
        LocalFree(MessageBuffer);
    }

    printf("\n");

}

RUNDOWN_PIPE_HANDLES RundownPipeHandles;

_Use_decl_annotations_
HRESULT
RundownPipeHandles(
    HANDLE HeapHandle,
    PHANDLE PipeHandles,
    ULONG NumberOfPipeHandles
    )
{
    ULONG Index;
    HANDLE *Handle;
    HRESULT Result = S_OK;
    BOOLEAN LastHandle = FALSE;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(HeapHandle)) {
        return E_INVALIDARG;
    }

    if (!ARGUMENT_PRESENT(PipeHandles)) {
        return E_POINTER;
    }

    if (NumberOfPipeHandles == 0) {
        return E_INVALIDARG;
    }

    //
    // Validation complete, continue with closing all handles.
    //

    Handle = PipeHandles;

    for (Index = 0; Index < NumberOfPipeHandles; Index++, Handle++) {

        //
        // Invariant check: once we've seen a NULL handle value, we shouldn't
        // see any non-NULL handles.  We never allocate handles out of order,
        // so this would indicate a severe problem somewhere.
        //

        if (LastHandle) {

            ASSERT(!*Handle);

        } else {

            if (*Handle) {
                if (!CloseHandle(*Handle)) {
                    REPORT_ERROR(CloseHandle);
                    Result = S_FALSE;
                }
            } else {
                LastHandle = TRUE;
            }
        }
    }

    if (!HeapFree(HeapHandle, 0, PipeHandles)) {
        REPORT_ERROR(HeapFree);
        Result = S_FALSE;
    }

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
