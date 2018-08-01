/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    DestroyPerfectHashTableContext.c

Abstract:

    This module implements the runtime context destroy routine for the
    PerfectHashTable component.

--*/

#include "stdafx.h"

_Use_decl_annotations_
BOOLEAN
DestroyPerfectHashTableContext(
    PPERFECT_HASH_TABLE_CONTEXT *ContextPointer,
    PBOOLEAN IsProcessTerminating
    )
/*++

Routine Description:

    Destroys a previously created PERFECT_HASH_TABLE_CONTEXT structure, freeing
    all memory unless the IsProcessTerminating flag is TRUE.

Arguments:

    ContextPointer - Supplies the address of a variable that contains the
        address of the PERFECT_HASH_TABLE_CONTEXT structure to destroy.  This
        variable will be cleared (i.e. the pointer will be set to NULL) if the
        routine destroys the structure successfully (returns TRUE).

    IsProcessTerminating - Optionally supplies a pointer to a boolean flag
        indicating whether or not the process is terminating.  If the pointer
        is non-NULL and the underlying value is TRUE, the method returns success
        immediately.  (If the process is terminating, there is no need to walk
        any internal data structures and individually free elements.)

Return Value:

    TRUE on success, FALSE on failure.  If successful, a NULL pointer will be
    written to the ContextPointer parameter.

--*/
{
    PRTL Rtl;
    BYTE Index;
    BYTE NumberOfEvents;
    BOOLEAN Success;
    PHANDLE Event;
    PALLOCATOR Allocator;
    PPERFECT_HASH_TABLE_CONTEXT Context;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(ContextPointer)) {
        goto Error;
    }

    if (ARGUMENT_PRESENT(IsProcessTerminating)) {
        if (*IsProcessTerminating) {

            //
            // Fast-path exit.  Clear the caller's pointer and return success.
            //

            *ContextPointer = NULL;
            return TRUE;
        }
    }

    //
    // A valid pointer has been provided, and the process isn't terminating.
    // Initialize aliases and continue with destroy logic.
    //

    Context = *ContextPointer;
    Rtl = Context->Rtl;
    Allocator = Context->Allocator;

    //
    // Sanity check the perfect hash structure size matches what we expect.
    //

    ASSERT(Context->SizeOfStruct == sizeof(*Context));

    //
    // Loop through all the events associated with the context and check if
    // they need to be closed.  (We do this instead of explicit calls to each
    // named event (e.g. CloseHandle(Context->ShutdownEvent)) as it means we
    // don't have to add new destructor code here every time we add an event.)
    //
    //

    Event = (PHANDLE)&Context->FirstEvent;
    NumberOfEvents = GetNumberOfContextEvents(Context);

    for (Index = 0; Index < NumberOfEvents; Index++, Event++) {

        if (*Event && *Event != INVALID_HANDLE_VALUE) {
            CloseHandle(*Event);
            *Event = NULL;
        }
    }

    //
    // Continue with cleanup of remaining resources.
    //

    if (Context->ObjectNamesWideBuffer) {
        Allocator->FreePointer(Allocator->Context,
                               &Context->ObjectNamesWideBuffer);
    }

    if (Context->MainCleanupGroup) {

        CloseThreadpoolCleanupGroupMembers(Context->MainCleanupGroup,
                                           TRUE,
                                           NULL);
        Context->MainWork = NULL;
        CloseThreadpoolCleanupGroup(Context->MainCleanupGroup);
        Context->MainCleanupGroup = NULL;

    } else {

        //
        // Invariant check: Context->MainWork should never be set if
        // MainCleanupGroup is not set.
        //

        ASSERT(!Context->MainWork);
    }

    if (Context->MainThreadpool) {
        CloseThreadpool(Context->MainThreadpool);
        Context->MainThreadpool = NULL;
    }

    //
    // The Finished and Error threadpools do not have a cleanup group associated
    // with them, so we can close their work items directly, if applicable.
    //

    if (Context->FinishedWork) {
        CloseThreadpoolWork(Context->FinishedWork);
        Context->FinishedWork = NULL;
    }

    if (Context->FinishedThreadpool) {
        CloseThreadpool(Context->FinishedThreadpool);
        Context->FinishedThreadpool = NULL;
    }

    if (Context->ErrorWork) {
        CloseThreadpoolWork(Context->ErrorWork);
        Context->ErrorWork = NULL;
    }

    if (Context->ErrorThreadpool) {
        CloseThreadpool(Context->ErrorThreadpool);
        Context->ErrorThreadpool = NULL;
    }

    //
    // Free the underlying memory and clear the caller's pointer.
    //

    Allocator->FreePointer(Allocator->Context, ContextPointer);

    Success = TRUE;

    goto End;

Error:

    Success = FALSE;

    //
    // Intentional follow-on to End.
    //

End:

    return Success;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
