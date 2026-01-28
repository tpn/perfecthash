/*++

Copyright (c) 2025 Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWorkStub.c

Abstract:

    This module provides stub file work callbacks for online-only builds.

--*/

#include "stdafx.h"

#ifdef PH_ONLINE_ONLY

#ifdef PH_WINDOWS
PERFECT_HASH_FILE_WORK_ITEM_CALLBACK FileWorkItemCallbackChm01;

PERFECT_HASH_FILE_WORK_CALLBACK FileWorkCallbackChm01;

_Use_decl_annotations_
VOID
FileWorkCallbackChm01(
    PTP_CALLBACK_INSTANCE Instance,
    PPERFECT_HASH_CONTEXT Context,
    PLIST_ENTRY ListEntry
    )
{
    PFILE_WORK_ITEM Item;

    if (!ARGUMENT_PRESENT(ListEntry)) {
        return;
    }

    Item = CONTAINING_RECORD(ListEntry, FILE_WORK_ITEM, ListEntry);
    Item->Instance = Instance;
    Item->Context = Context;

    FileWorkItemCallbackChm01(Item);
}
#endif

_Use_decl_annotations_
VOID
FileWorkItemCallbackChm01(
    PFILE_WORK_ITEM Item
    )
{
    if (!ARGUMENT_PRESENT(Item)) {
        return;
    }

    Item->LastResult = PH_E_NOT_IMPLEMENTED;
    Item->LastError = 0;
    Item->NumberOfErrors = 1;
}

#endif // PH_ONLINE_ONLY

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
