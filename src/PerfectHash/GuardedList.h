/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    RtlGuardedList.h

Abstract:

    This is the private header file for the guarded (thread safe) doubly-linked
    list component.  It defines the GUARDED_LIST structure and relevant function
    pointer typedefs for manipulating the structure.

--*/

#pragma once

#include "stdafx.h"

DECLARE_COMPONENT(List, GUARDED_LIST);

DEFINE_UNUSED_STATE(GUARDED_LIST);
DEFINE_UNUSED_FLAGS(GUARDED_LIST);

typedef
VOID
(CALLBACK GUARDED_LIST_CALLBACK)(
    _In_ PGUARDED_LIST List,
    _In_ PLIST_ENTRY Entry,
    _In_opt_ PVOID Context
    );
typedef GUARDED_LIST_CALLBACK *PGUARDED_LIST_CALLBACK;

//
// Vtbl methods.
//

typedef
HRESULT
(STDAPICALLTYPE GUARDED_LIST_SET_CALLBACKS)(
    _In_ PGUARDED_LIST List,
    _In_opt_ PGUARDED_LIST_CALLBACK AddEntryCallback,
    _In_opt_ PGUARDED_LIST_CALLBACK RemoveEntryCallback,
    _In_opt_ PVOID CallbackContext
    );
typedef GUARDED_LIST_SET_CALLBACKS *PGUARDED_LIST_SET_CALLBACKS;

typedef
BOOLEAN
(STDAPICALLTYPE GUARDED_LIST_IS_EMPTY)(
    _In_ PGUARDED_LIST List
    );
typedef GUARDED_LIST_IS_EMPTY *PGUARDED_LIST_IS_EMPTY;

typedef
ULONG_PTR
(STDAPICALLTYPE GUARDED_LIST_QUERY_DEPTH)(
    _In_ PGUARDED_LIST List
    );
typedef GUARDED_LIST_QUERY_DEPTH *PGUARDED_LIST_QUERY_DEPTH;

typedef
VOID
(STDAPICALLTYPE GUARDED_LIST_INSERT_HEAD)(
    _In_ PGUARDED_LIST List,
    _In_ PLIST_ENTRY Entry
    );
typedef GUARDED_LIST_INSERT_HEAD *PGUARDED_LIST_INSERT_HEAD;

typedef
VOID
(STDAPICALLTYPE GUARDED_LIST_INSERT_TAIL)(
    _In_ PGUARDED_LIST List,
    _In_ PLIST_ENTRY Entry
    );
typedef GUARDED_LIST_INSERT_TAIL *PGUARDED_LIST_INSERT_TAIL;

typedef
VOID
(STDAPICALLTYPE GUARDED_LIST_APPEND_TAIL)(
    _In_ PGUARDED_LIST List,
    _In_ PLIST_ENTRY ListHeadToAppend,
    _In_ ULONG_PTR NumberOfEntries
    );
typedef GUARDED_LIST_APPEND_TAIL *PGUARDED_LIST_APPEND_TAIL;

typedef
PLIST_ENTRY
(STDAPICALLTYPE GUARDED_LIST_REMOVE_HEAD)(
    _In_ PGUARDED_LIST List
    );
typedef GUARDED_LIST_REMOVE_HEAD *PGUARDED_LIST_REMOVE_HEAD;

typedef
PLIST_ENTRY
(STDAPICALLTYPE GUARDED_LIST_REMOVE_TAIL)(
    _In_ PGUARDED_LIST List
    );
typedef GUARDED_LIST_REMOVE_TAIL *PGUARDED_LIST_REMOVE_TAIL;

typedef
BOOLEAN
(STDAPICALLTYPE GUARDED_LIST_REMOVE_ENTRY)(
    _In_ PGUARDED_LIST List,
    _In_ PLIST_ENTRY Entry
    );
typedef GUARDED_LIST_REMOVE_ENTRY *PGUARDED_LIST_REMOVE_ENTRY;

typedef struct _GUARDED_LIST_VTBL {
    DECLARE_COMPONENT_VTBL_HEADER(GUARDED_LIST);
    PGUARDED_LIST_SET_CALLBACKS SetCallbacks;
    PGUARDED_LIST_IS_EMPTY IsEmpty;
    PGUARDED_LIST_QUERY_DEPTH QueryDepth;
    PGUARDED_LIST_INSERT_HEAD InsertHead;
    PGUARDED_LIST_INSERT_TAIL InsertTail;
    PGUARDED_LIST_APPEND_TAIL AppendTail;
    PGUARDED_LIST_REMOVE_HEAD RemoveHead;
    PGUARDED_LIST_REMOVE_TAIL RemoveTail;
    PGUARDED_LIST_REMOVE_ENTRY RemoveEntry;
} GUARDED_LIST_VTBL;
typedef GUARDED_LIST_VTBL *PGUARDED_LIST_VTBL;

typedef struct _Struct_size_bytes_(SizeOfStruct) _GUARDED_LIST {
    COMMON_COMPONENT_HEADER(GUARDED_LIST);
    volatile ULONG_PTR NumberOfEntries;
    PGUARDED_LIST_CALLBACK AddEntryCallback;
    PGUARDED_LIST_CALLBACK RemoveEntryCallback;
    PVOID CallbackContext;
    LIST_ENTRY ListHead;
    GUARDED_LIST_VTBL Interface;
} GUARDED_LIST;
typedef GUARDED_LIST *PGUARDED_LIST;

#define TryAcquireGuardedListLockExclusive(GuardedList) \
    TryAcquireSRWLockExclusive(&GuardedList->Lock)

#define AcquireGuardedListLockExclusive(GuardedList) \
    AcquireSRWLockExclusive(&GuardedList->Lock)

#define ReleaseGuardedListLockExclusive(GuardedList) \
    ReleaseSRWLockExclusive(&GuardedList->Lock)

#define TryAcquireGuardedListLockShared(GuardedList) \
    TryAcquireSRWLockShared(&GuardedList->Lock)

#define AcquireGuardedListLockShared(GuardedList) \
    AcquireSRWLockShared(&GuardedList->Lock)

#define ReleaseGuardedListLockShared(GuardedList) \
    ReleaseSRWLockShared(&GuardedList->Lock)

//
// Private non-vtbl methods.
//

typedef
HRESULT
(NTAPI GUARDED_LIST_INITIALIZE)(
    _In_ PGUARDED_LIST List
    );
typedef GUARDED_LIST_INITIALIZE *PGUARDED_LIST_INITIALIZE;

typedef
VOID
(NTAPI GUARDED_LIST_RUNDOWN)(
    _In_ _Post_ptr_invalid_ PGUARDED_LIST Path
    );
typedef GUARDED_LIST_RUNDOWN *PGUARDED_LIST_RUNDOWN;

extern GUARDED_LIST_INITIALIZE GuardedListInitialize;
extern GUARDED_LIST_RUNDOWN GuardedListRundown;

extern GUARDED_LIST_SET_CALLBACKS GuardedListSetCallbacks;
extern GUARDED_LIST_IS_EMPTY GuardedListIsEmpty;
extern GUARDED_LIST_QUERY_DEPTH GuardedListQueryDepth;
extern GUARDED_LIST_INSERT_HEAD GuardedListInsertHead;
extern GUARDED_LIST_INSERT_TAIL GuardedListInsertTail;
extern GUARDED_LIST_APPEND_TAIL GuardedListAppendTail;
extern GUARDED_LIST_REMOVE_HEAD GuardedListRemoveHead;
extern GUARDED_LIST_REMOVE_TAIL GuardedListRemoveTail;
extern GUARDED_LIST_REMOVE_ENTRY GuardedListRemoveEntry;

extern GUARDED_LIST_IS_EMPTY GuardedListIsEmptyTsx;
extern GUARDED_LIST_QUERY_DEPTH GuardedListQueryDepthTsx;
extern GUARDED_LIST_INSERT_HEAD GuardedListInsertHeadTsx;
extern GUARDED_LIST_INSERT_TAIL GuardedListInsertTailTsx;
extern GUARDED_LIST_APPEND_TAIL GuardedListAppendTailTsx;
extern GUARDED_LIST_REMOVE_HEAD GuardedListRemoveHeadTsx;
extern GUARDED_LIST_REMOVE_TAIL GuardedListRemoveTailTsx;
extern GUARDED_LIST_REMOVE_ENTRY GuardedListRemoveEntryTsx;

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
