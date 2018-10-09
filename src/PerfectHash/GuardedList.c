/*++

Copyright (c) 2016 Trent Nelson <trent@trent.me>

Module Name:

    RtlGuardedList.c

Abstract:

    This module implements routines related to the GUARDED_LIST component.
    The GUARDED_LIST structure encapsulates the doubly-linked list NT-style
    functionality (LIST_ENTRY); routines are implemented mirroring the same
    functionality present in NT: appending to head, append to tail, remove
    from head, remove from tail, is list empty, and list traversal.

--*/

#include "stdafx.h"

GUARDED_LIST_INITIALIZE GuardedListInitialize;

_Use_decl_annotations_
HRESULT
GuardedListInitialize(
    PGUARDED_LIST List
    )
/*++

Routine Description:

    Initializes a guarded list structure.

Arguments:

    List - Supplies a pointer to a GUARDED_LIST structure for which
        initialization is to be performed.

Return Value:

    S_OK - Success.

    E_POINTER - List is NULL.

    E_UNEXPECTED - All other errors.

--*/
{
    HRESULT Result = S_OK;

    if (!ARGUMENT_PRESENT(List)) {
        return E_POINTER;
    }

    List->SizeOfStruct = sizeof(*List);

    //
    // Create Rtl and Allocator components.
    //

    Result = List->Vtbl->CreateInstance(List,
                                        NULL,
                                        &IID_PERFECT_HASH_RTL,
                                        &List->Rtl);

    if (FAILED(Result)) {
        goto Error;
    }

    Result = List->Vtbl->CreateInstance(List,
                                        NULL,
                                        &IID_PERFECT_HASH_ALLOCATOR,
                                        &List->Allocator);

    if (FAILED(Result)) {
        goto Error;
    }

    AcquireGuardedListLockExclusive(List);
    InitializeListHead(&List->ListHead);
    List->NumberOfEntries = 0;
    ReleaseGuardedListLockExclusive(List);

    //
    // We're done!  Indicate success and finish up.
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

GUARDED_LIST_RUNDOWN GuardedListRundown;

_Use_decl_annotations_
VOID
GuardedListRundown(
    PGUARDED_LIST List
    )
/*++

Routine Description:

    Release all resources associated with a guarded list.

Arguments:

    List - Supplies a pointer to a GUARDED_LIST structure for which rundown is
        to be performed.

Return Value:

    None.

--*/
{
    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(List)) {
        return;
    }

    AcquireGuardedListLockExclusive(List);

    //
    // Sanity check structure size.
    //

    ASSERT(List->SizeOfStruct == sizeof(*List));

    if (!IsListEmpty(&List->ListHead)) {
        PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
    } else if (List->NumberOfEntries > 0) {
        PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
    }

    //
    // Release COM references.
    //

    RELEASE(List->Allocator);
    RELEASE(List->Rtl);

    //
    // Release lock and return.
    //

    ReleaseGuardedListLockExclusive(List);

    return;
}


GUARDED_LIST_IS_EMPTY GuardedListIsEmpty;

_Use_decl_annotations_
BOOLEAN
GuardedListIsEmpty(
    PGUARDED_LIST List
    )
{
    BOOLEAN IsEmpty;
    AcquireGuardedListLockShared(List);
    IsEmpty = IsListEmpty(&List->ListHead);
    ReleaseGuardedListLockShared(List);
    return IsEmpty;
}


GUARDED_LIST_QUERY_DEPTH GuardedListQueryDepth;

_Use_decl_annotations_
ULONG_PTR
GuardedListQueryDepth(
    PGUARDED_LIST List
    )
{
    ULONG_PTR Depth;
    AcquireGuardedListLockShared(List);
    Depth = List->NumberOfEntries;
    ReleaseGuardedListLockShared(List);
    return Depth;
}


GUARDED_LIST_INSERT_HEAD GuardedListInsertHead;

_Use_decl_annotations_
VOID
GuardedListInsertHead(
    PGUARDED_LIST List,
    PLIST_ENTRY Entry
    )
{
    AcquireGuardedListLockExclusive(List);
    InsertHeadList(&List->ListHead, Entry);
    InterlockedIncrementULongPtr(&List->NumberOfEntries);
    ReleaseGuardedListLockExclusive(List);
}


GUARDED_LIST_INSERT_TAIL GuardedListInsertTail;

_Use_decl_annotations_
VOID
GuardedListInsertTail(
    PGUARDED_LIST List,
    PLIST_ENTRY Entry
    )
{
    AcquireGuardedListLockExclusive(List);
    InsertTailList(&List->ListHead, Entry);
    InterlockedIncrementULongPtr(&List->NumberOfEntries);
    ReleaseGuardedListLockExclusive(List);
}


GUARDED_LIST_APPEND_TAIL GuardedListAppendTail;

_Use_decl_annotations_
VOID
GuardedListAppendTail(
    PGUARDED_LIST List,
    PLIST_ENTRY ListHeadToAppend,
    ULONG_PTR NumberOfEntries
    )
{
    AcquireGuardedListLockExclusive(List);
    AppendTailList(&List->ListHead, ListHeadToAppend);
    InterlockedAddULongPtr(&List->NumberOfEntries, NumberOfEntries);
    ReleaseGuardedListLockExclusive(List);
}


GUARDED_LIST_REMOVE_HEAD GuardedListRemoveHead;

_Use_decl_annotations_
PLIST_ENTRY
GuardedListRemoveHead(
    PGUARDED_LIST List
    )
{
    PLIST_ENTRY Entry;

    AcquireGuardedListLockExclusive(List);
    Entry = RemoveHeadList(&List->ListHead);
    InterlockedDecrementULongPtr(&List->NumberOfEntries);
    ReleaseGuardedListLockExclusive(List);

    return Entry;
}


GUARDED_LIST_REMOVE_HEAD GuardedListRemoveTail;

_Use_decl_annotations_
PLIST_ENTRY
GuardedListRemoveTail(
    PGUARDED_LIST List
    )
{
    PLIST_ENTRY Entry;

    AcquireGuardedListLockExclusive(List);
    Entry = RemoveTailList(&List->ListHead);
    InterlockedDecrementULongPtr(&List->NumberOfEntries);
    ReleaseGuardedListLockExclusive(List);

    return Entry;
}


GUARDED_LIST_REMOVE_ENTRY GuardedListRemoveEntry;

_Use_decl_annotations_
BOOLEAN
GuardedListRemoveEntry(
    PGUARDED_LIST List,
    PLIST_ENTRY Entry
    )
{
    BOOLEAN IsEmpty;

    AcquireGuardedListLockExclusive(List);
    IsEmpty = RemoveEntryList(Entry);
    InterlockedDecrementULongPtr(&List->NumberOfEntries);
    ReleaseGuardedListLockExclusive(List);

    return IsEmpty;
}


GUARDED_LIST_REMOVE_HEAD_EX GuardedListRemoveHeadEx;

_Use_decl_annotations_
BOOLEAN
GuardedListRemoveHeadEx(
    PGUARDED_LIST List,
    PLIST_ENTRY *EntryPointer
    )
{
    BOOLEAN NotEmpty;
    PLIST_ENTRY Entry = NULL;

    AcquireGuardedListLockExclusive(List);
    NotEmpty = (List->NumberOfEntries > 0 ? TRUE : FALSE);
    if (NotEmpty) {
        Entry = RemoveHeadList(&List->ListHead);
        InterlockedDecrementULongPtr(&List->NumberOfEntries);
    }
    ReleaseGuardedListLockExclusive(List);

    *EntryPointer = Entry;
    return NotEmpty;
}

GUARDED_LIST_RESET GuardedListReset;

_Use_decl_annotations_
VOID
GuardedListReset(
    PGUARDED_LIST List
    )
{
    AcquireGuardedListLockExclusive(List);
    InitializeListHead(&List->ListHead);
    List->NumberOfEntries = 0;
    ReleaseGuardedListLockExclusive(List);
}

//
// TSX versions of the above.
//

#ifdef _M_AMD64

#define BEGIN_TSX(Fallback)                 \
    ULONG Status;                           \
                                            \
Retry:                                      \
    Status = _xbegin();                     \
    if (Status & _XABORT_RETRY) {           \
        goto Retry;                         \
    } else if (Status != _XBEGIN_STARTED) { \
        return Fallback;                    \
    }

#define BEGIN_TSX_VOID_RETURN(Fallback)     \
    ULONG Status;                           \
                                            \
Retry:                                      \
    Status = _xbegin();                     \
    if (Status & _XABORT_RETRY) {           \
        goto Retry;                         \
    } else if (Status != _XBEGIN_STARTED) { \
        Fallback;                           \
        return;                             \
    }


#define END_TSX() _xend()

GUARDED_LIST_IS_EMPTY GuardedListIsEmptyTsx;

_Use_decl_annotations_
BOOLEAN
GuardedListIsEmptyTsx(
    PGUARDED_LIST List
    )
{
    BOOLEAN IsEmpty;

    BEGIN_TSX(GuardedListIsEmpty(List));
    IsEmpty = IsListEmpty(&List->ListHead);
    END_TSX();
    return IsEmpty;
}


GUARDED_LIST_QUERY_DEPTH GuardedListQueryDepthTsx;

_Use_decl_annotations_
ULONG_PTR
GuardedListQueryDepthTsx(
    PGUARDED_LIST List
    )
{
    ULONG_PTR Depth;

    BEGIN_TSX(GuardedListQueryDepth(List));
    Depth = List->NumberOfEntries;
    END_TSX();
    return Depth;
}


GUARDED_LIST_INSERT_HEAD GuardedListInsertHeadTsx;

_Use_decl_annotations_
VOID
GuardedListInsertHeadTsx(
    PGUARDED_LIST List,
    PLIST_ENTRY Entry
    )
{
    BEGIN_TSX_VOID_RETURN(GuardedListInsertHead(List, Entry));
    InsertHeadList(&List->ListHead, Entry);
    List->NumberOfEntries++;
    END_TSX();
}


GUARDED_LIST_INSERT_TAIL GuardedListInsertTailTsx;

_Use_decl_annotations_
VOID
GuardedListInsertTailTsx(
    PGUARDED_LIST List,
    PLIST_ENTRY Entry
    )
{
    BEGIN_TSX_VOID_RETURN(GuardedListInsertTail(List, Entry));
    InsertTailList(&List->ListHead, Entry);
    List->NumberOfEntries++;
    END_TSX();
}


GUARDED_LIST_APPEND_TAIL GuardedListAppendTailTsx;

_Use_decl_annotations_
VOID
GuardedListAppendTailTsx(
    PGUARDED_LIST List,
    PLIST_ENTRY ListHeadToAppend,
    ULONG_PTR NumberOfEntries
    )
{

    BEGIN_TSX_VOID_RETURN(
        GuardedListAppendTail(
            List,
            ListHeadToAppend,
            NumberOfEntries
        )
    );

    AppendTailList(&List->ListHead, ListHeadToAppend);
    List->NumberOfEntries += NumberOfEntries;

    END_TSX();
}


GUARDED_LIST_REMOVE_HEAD GuardedListRemoveHeadTsx;

_Use_decl_annotations_
PLIST_ENTRY
GuardedListRemoveHeadTsx(
    PGUARDED_LIST List
    )
{
    PLIST_ENTRY Entry;

    BEGIN_TSX(GuardedListRemoveHead(List));
    Entry = RemoveHeadList(&List->ListHead);
    List->NumberOfEntries--;
    END_TSX();

    return Entry;
}


GUARDED_LIST_REMOVE_HEAD GuardedListRemoveTailTsx;

_Use_decl_annotations_
PLIST_ENTRY
GuardedListRemoveTailTsx(
    PGUARDED_LIST List
    )
{
    PLIST_ENTRY Entry;

    BEGIN_TSX(GuardedListRemoveTail(List));
    Entry = RemoveTailList(&List->ListHead);
    List->NumberOfEntries--;
    END_TSX();
    return Entry;
}


GUARDED_LIST_REMOVE_ENTRY GuardedListRemoveEntryTsx;

_Use_decl_annotations_
BOOLEAN
GuardedListRemoveEntryTsx(
    PGUARDED_LIST List,
    PLIST_ENTRY Entry
    )
{
    BOOLEAN IsEmpty;

    BEGIN_TSX(GuardedListRemoveEntry(List, Entry));
    IsEmpty = RemoveEntryList(Entry);
    List->NumberOfEntries--;
    END_TSX();

    return IsEmpty;
}


GUARDED_LIST_REMOVE_HEAD_EX GuardedListRemoveHeadExTsx;

_Use_decl_annotations_
BOOLEAN
GuardedListRemoveHeadExTsx(
    PGUARDED_LIST List,
    PLIST_ENTRY *EntryPointer
    )
{
    BOOLEAN NotEmpty = FALSE;
    PLIST_ENTRY Entry = NULL;

    BEGIN_TSX(GuardedListRemoveHeadEx(List, EntryPointer));
    NotEmpty = (List->NumberOfEntries > 0 ? TRUE : FALSE);
    if (NotEmpty) {
        Entry = RemoveHeadList(&List->ListHead);
        List->NumberOfEntries--;
    }
    END_TSX();

    *EntryPointer = Entry;
    return NotEmpty;
}

#endif

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
