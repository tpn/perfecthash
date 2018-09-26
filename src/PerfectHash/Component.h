/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Component.h

Abstract:

    This is the private header file for the supporting COM glue used by the
    perfect hash table library.

--*/

#pragma once

#include "stdafx.h"

#define COMMON_COMPONENT_HEADER(Name) \
    P##Name##_VTBL Vtbl;              \
    SRWLOCK Lock;                     \
    struct _RTL *Rtl;                 \
    struct _ALLOCATOR *Allocator;     \
    PIUNKNOWN OuterUnknown;           \
    volatile LONG ReferenceCount;     \
    PERFECT_HASH_INTERFACE_ID Id;     \
    ULONG SizeOfStruct;               \
    Name##_STATE State;               \
    Name##_FLAGS Flags;               \
    ULONG Reserved

#define DEFINE_UNUSED_STATE(Name)                  \
typedef union _##Name##_STATE {                    \
    struct {                                       \
        ULONG Unused:32;                           \
    };                                             \
    LONG AsLong;                                   \
    ULONG AsULong;                                 \
} Name##_STATE;                                    \
C_ASSERT(sizeof(Name##_STATE) == sizeof(ULONG));   \
typedef Name##_STATE *P##Name##_STATE

#define DEFINE_UNUSED_FLAGS(Name)                  \
typedef union _##Name##_FLAGS {                    \
    struct {                                       \
        ULONG Unused:32;                           \
    };                                             \
    LONG AsLong;                                   \
    ULONG AsULong;                                 \
} Name##_FLAGS;                                    \
C_ASSERT(sizeof(Name##_FLAGS) == sizeof(ULONG));   \
typedef Name##_FLAGS *P##Name##_FLAGS

//
// IUnknown
//

DEFINE_UNUSED_STATE(IUNKNOWN);
DEFINE_UNUSED_FLAGS(IUNKNOWN);

typedef struct _Struct_size_bytes_(SizeOfStruct) _IUNKNOWN {
    COMMON_COMPONENT_HEADER(IUNKNOWN);
    IUNKNOWN_VTBL Interface;
} IUNKNOWN;
typedef IUNKNOWN *PIUNKNOWN;

//
// IClassFactory
//

DEFINE_UNUSED_STATE(ICLASSFACTORY);
DEFINE_UNUSED_FLAGS(ICLASSFACTORY);

typedef struct _Struct_size_bytes_(SizeOfStruct) _ICLASSFACTORY {
    COMMON_COMPONENT_HEADER(ICLASSFACTORY);
    ICLASSFACTORY_VTBL Interface;
} ICLASSFACTORY;
typedef ICLASSFACTORY *PICLASSFACTORY;

//
// Component
//

typedef union _COMPONENT COMPONENT;
typedef COMPONENT *PCOMPONENT;

typedef
HRESULT
(STDAPICALLTYPE COMPONENT_QUERY_INTERFACE)(
    _In_ PCOMPONENT Component,
    _In_ REFIID InterfaceId,
    _COM_Outptr_ PVOID *Interface
    );
typedef COMPONENT_QUERY_INTERFACE *PCOMPONENT_QUERY_INTERFACE;

typedef
ULONG
(STDAPICALLTYPE COMPONENT_ADD_REF)(
    _In_ PCOMPONENT Component
    );
typedef COMPONENT_ADD_REF *PCOMPONENT_ADD_REF;

typedef
ULONG
(STDAPICALLTYPE COMPONENT_RELEASE)(
    _In_ PCOMPONENT Component
    );
typedef COMPONENT_RELEASE *PCOMPONENT_RELEASE;

typedef
HRESULT
(STDAPICALLTYPE COMPONENT_CREATE_INSTANCE)(
    _In_ PCOMPONENT Component,
    _In_opt_ PIUNKNOWN UnknownOuter,
    _In_ REFIID InterfaceId,
    _COM_Outptr_ PVOID *Instance
    );
typedef COMPONENT_CREATE_INSTANCE *PCOMPONENT_CREATE_INSTANCE;

typedef
HRESULT
(STDAPICALLTYPE COMPONENT_LOCK_SERVER)(
    _In_ PCOMPONENT Component,
    _In_opt_ BOOL Lock
    );
typedef COMPONENT_LOCK_SERVER *PCOMPONENT_LOCK_SERVER;

typedef union _COMPONENT_VTBL {
    IUNKNOWN_VTBL UnknownVtbl;
    ICLASSFACTORY_VTBL ClassFactoryVtbl;
} COMPONENT_VTBL;
typedef COMPONENT_VTBL *PCOMPONENT_VTBL;

DEFINE_UNUSED_STATE(COMPONENT);
DEFINE_UNUSED_FLAGS(COMPONENT);

typedef union _COMPONENT {
    struct {
        COMMON_COMPONENT_HEADER(COMPONENT);
    };
    IUNKNOWN Unknown;
    ICLASSFACTORY ClassFactory;
} COMPONENT;
typedef COMPONENT *PCOMPONENT;

typedef
_Success_(return != 0)
PCOMPONENT
(STDAPICALLTYPE CREATE_COMPONENT)(
    _In_ PERFECT_HASH_INTERFACE_ID Id,
    _In_opt_ PIUNKNOWN OuterUnknown
    );
typedef CREATE_COMPONENT *PCREATE_COMPONENT;

typedef
HRESULT
(STDAPICALLTYPE COMPONENT_INITIALIZE)(
    _In_ PCOMPONENT Component
    );
typedef COMPONENT_INITIALIZE *PCOMPONENT_INITIALIZE;

typedef
VOID
(STDAPICALLTYPE COMPONENT_RUNDOWN)(
    _In_ _Post_invalid_ PCOMPONENT Component
    );
typedef COMPONENT_RUNDOWN *PCOMPONENT_RUNDOWN;

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
