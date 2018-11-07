/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHash.h

Abstract:

    This is the main public header file for the PerfectHash component.
    It defines structures and functions related to creating perfect hash
    tables, contexts, loading keys, testing and benchmarking.

--*/

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

//
// N.B. The warning disable glue is necessary to get the system headers to
//      include with all errors enabled (/Wall).
//

//
// Disable the "function selected for inlining" and "function not inlined"
// warnings.
//

#pragma warning(disable: 4710 4711)

//
// 4255:
//      winuser.h(6502): warning C4255: 'EnableMouseInPointerForThread':
//          no function prototype given: converting '()' to '(void)'
//
// 4668:
//      winioctl.h(8910): warning C4668: '_WIN32_WINNT_WIN10_TH2'
//          is not defined as a preprocessor macro, replacing with
//          '0' for '#if/#elif'
//
//

#pragma warning(push)
#pragma warning(disable: 4255)
#pragma warning(disable: 4668)
#include <Windows.h>
#pragma warning(pop)

#include <sal.h>
#include <specstrings.h>

//
// Disable the anonymous union/struct warning.
//

#pragma warning(disable: 4201)

//
// Disable "bit field types other than int" warning.
//

#pragma warning(disable: 4214)

//
// NT DDK types.
//

typedef struct _STRING {
    USHORT Length;
    USHORT MaximumLength;
#ifdef _WIN64
    union {
        LONG Hash;
        LONG Padding;
    };
#endif
    PCHAR Buffer;
} STRING, ANSI_STRING, *PSTRING, *PANSI_STRING, **PPSTRING, **PPANSI_STRING;
typedef const STRING *PCSTRING;

typedef struct _UNICODE_STRING {
    USHORT Length;
    USHORT MaximumLength;
#ifdef _WIN64
    union {
        LONG Hash;
        LONG Padding;
    };
#endif
    PWSTR Buffer;
} UNICODE_STRING, *PUNICODE_STRING, **PPUNICODE_STRING, ***PPPUNICODE_STRING;
typedef const UNICODE_STRING *PCUNICODE_STRING;
#define UNICODE_NULL ((WCHAR)0)

typedef _Null_terminated_ CONST CHAR *PCSZ;

typedef union ULONG_BYTES {
    struct _Struct_size_bytes_(sizeof(ULONG)) {
        BYTE Byte1;
        BYTE Byte2;
        BYTE Byte3;
        BYTE Byte4;
    };

    struct _Struct_size_bytes_(sizeof(ULONG)) {
        USHORT Word1;
        USHORT Word2;
    };

    LONG AsLong;
    ULONG AsULong;
} ULONG_BYTES;
C_ASSERT(sizeof(ULONG_BYTES) == sizeof(ULONG));
typedef ULONG_BYTES *PULONG_BYTES;

#ifndef ARGUMENT_PRESENT
#define ARGUMENT_PRESENT(ArgumentPointer) (                  \
    (CHAR *)((ULONG_PTR)(ArgumentPointer)) != (CHAR *)(NULL) \
)
#endif

#define IsValidHandle(Handle) (Handle != NULL && Handle != INVALID_HANDLE_VALUE)

#ifdef _WIN64
#define InterlockedIncrementULongPtr(Ptr) InterlockedIncrement64((PLONG64)Ptr)
#define InterlockedDecrementULongPtr(Ptr) InterlockedDecrement64((PLONG64)Ptr)
#define InterlockedAddULongPtr(Ptr, Val) \
    InterlockedAdd64((PLONG64)Ptr, (LONG64)Val)
#else
#define InterlockedIncrementULongPtr(Ptr) InterlockedIncrement((PLONG)Ptr)
#define InterlockedDecrementULongPtr(Ptr) InterlockedDecrement((PLONG)Ptr)
#define InterlockedAddULongPtr(Ptr, Val) \
    InterlockedAdd((PLONG)Ptr, (LONG)Val)
#endif

//
// Define start/end markers for IACA.
//

#define IACA_VC_START() __writegsbyte(111, 111)
#define IACA_VC_END()   __writegsbyte(222, 222)

//
// Define an enumeration for identifying CPU architectures.
//

typedef enum _Enum_is_bitflag_ _PERFECT_HASH_CPU_ARCH_ID {

    //
    // Explicitly define a null ID to take the 0-index slot.  This makes enum
    // validation easier.
    //

    PerfectHashNullCpuArchId                = 0,

    //
    // Begin valid CPU architectures.
    //

    PerfectHashx86CpuArchId                 = 1,
    PerfectHashx64CpuArchId                 = 2,
    PerfectHashArmCpuArchId                 = 3,
    PerfectHashArm64CpuArchId               = 4,

    //
    // End valid CPU architectures.
    //

    //
    // N.B. Keep the next value last.
    //

    PerfectHashInvalidCpuArchId

} PERFECT_HASH_CPU_ARCH_ID;
typedef PERFECT_HASH_CPU_ARCH_ID *PPERFECT_HASH_CPU_ARCH_ID;

//
// Provide a simple inline CPU architecture enum validation routine.
//

FORCEINLINE
BOOLEAN
IsValidPerfectHashCpuArchId(
    _In_ PERFECT_HASH_CPU_ARCH_ID CpuArchId
    )
{
    return (
        CpuArchId > PerfectHashNullCpuArchId &&
        CpuArchId < PerfectHashInvalidCpuArchId
    );
}

//
// Provide a simple inline routine for obtaining the current CPU architecture.
//

FORCEINLINE
PERFECT_HASH_CPU_ARCH_ID
PerfectHashGetCurrentCpuArch(
    VOID
    )
{
#ifdef _M_AMD64
    return PerfectHashx64CpuArchId;
#elif defined(_M_IX86)
    return PerfectHashx86CpuArchId;
#elif defined(_M_ARM64)
    return PerfectHashArm64CpuArchId;
#elif defined(_M_ARM)
    return PerfectHashArmCpuArchId;
#else
#error Unknown CPU architecture.
#endif
}

//
// Define an enumeration for identifying COM interfaces.
//

typedef enum _PERFECT_HASH_INTERFACE_ID {

    //
    // Explicitly define a null ID to take the 0-index slot.  This makes enum
    // validation easier.
    //

    PerfectHashNullInterfaceId             = 0,

    //
    // Begin valid interfaces.
    //

    PerfectHashUnknownInterfaceId          =  1,
    PerfectHashClassFactoryInterfaceId     =  2,
    PerfectHashKeysInterfaceId             =  3,
    PerfectHashContextInterfaceId          =  4,
    PerfectHashTableInterfaceId            =  5,
    PerfectHashRtlInterfaceId              =  6,
    PerfectHashAllocatorInterfaceId        =  7,
    PerfectHashFileInterfaceId             =  8,
    PerfectHashPathInterfaceId             =  9,
    PerfectHashDirectoryInterfaceId        = 10,
    PerfectHashGuardedListInterfaceId      = 11,
    PerfectHashGraphInterfaceId            = 12,

    PerfectHashLastInterfaceId             = PerfectHashGraphInterfaceId,

    //
    // End valid interfaces.
    //

    //
    // N.B. Keep the next value last.
    //

    PerfectHashInvalidInterfaceId,

} PERFECT_HASH_INTERFACE_ID;

//
// Provide a simple inline interface enum validation routine.
//

FORCEINLINE
BOOLEAN
IsValidPerfectHashInterfaceId(
    _In_ PERFECT_HASH_INTERFACE_ID InterfaceId
    )
{
    return (
        InterfaceId > PerfectHashNullInterfaceId &&
        InterfaceId < PerfectHashInvalidInterfaceId
    );
}

//
// COM-related typedefs.
//

typedef
HRESULT
(CO_INITIALIZE_EX)(
    _In_opt_ LPVOID Reserved,
    _In_ DWORD CoInit
    );
typedef CO_INITIALIZE_EX *PCO_INITIALIZE_EX;

typedef
_Check_return_
HRESULT
(STDAPICALLTYPE DLL_GET_CLASS_OBJECT)(
    _In_ REFCLSID ClassId,
    _In_ REFIID InterfaceId,
    _COM_Outptr_ LPVOID *Interface
    );
typedef DLL_GET_CLASS_OBJECT *PDLL_GET_CLASS_OBJECT;

typedef
HRESULT
(STDAPICALLTYPE DLL_CAN_UNLOAD_NOW)(
    VOID
    );
typedef DLL_CAN_UNLOAD_NOW *PDLL_CAN_UNLOAD_NOW;

#define DEFINE_GUID_EX(Name, l, w1, w2, b1, b2, b3, b4, b5, b6, b7, b8) \
    static const GUID Name                                              \
        = { l, w1, w2, { b1, b2,  b3,  b4,  b5,  b6,  b7,  b8 } }

typedef GUID *PGUID;
typedef const GUID CGUID;
typedef GUID const *PCGUID;

//
// IID_IUNKNOWN: 00000000-0000-0000-C000-000000000046
//

DEFINE_GUID_EX(IID_PERFECT_HASH_IUNKNOWN, 0x00000000, 0x0000, 0x0000,
               0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46);

//
// IID_ICLASSFACTORY: 00000001-0000-0000-C000-000000000046
//

DEFINE_GUID_EX(IID_PERFECT_HASH_ICLASSFACTORY, 0x00000001, 0x0000, 0x0000,
               0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46);

//
// CLSID_PERFECT_HASH: 402045FD-72F4-4A05-902E-D22B7C1877B4
//

DEFINE_GUID_EX(CLSID_PERFECT_HASH, 0x402045fd, 0x72f4, 0x4a05,
               0x90, 0x2e, 0xd2, 0x2b, 0x7c, 0x18, 0x77, 0xb4);

//
// IID_PERFECT_HASH_KEYS: 7E43EBEA-8671-47BA-B844-760B7A9EA921
//

DEFINE_GUID_EX(IID_PERFECT_HASH_KEYS, 0x7e43ebea, 0x8671, 0x47ba,
               0xb8, 0x44, 0x76, 0xb, 0x7a, 0x9e, 0xa9, 0x21);

//
// IID_PERFECT_HASH_CONTEXT: D4B24571-99D7-44BA-8A27-63D8739F9B81
//

DEFINE_GUID_EX(IID_PERFECT_HASH_CONTEXT, 0xd4b24571, 0x99d7, 0x44ba,
               0x8a, 0x27, 0x63, 0xd8, 0x73, 0x9f, 0x9b, 0x81);

//
// IID_PERFECT_HASH_TABLE: C265816F-C6A9-4B44-BCEE-EC5A12ABE1EF
//

DEFINE_GUID_EX(IID_PERFECT_HASH_TABLE, 0xc265816f, 0xc6a9, 0x4b44,
               0xbc, 0xee, 0xec, 0x5a, 0x12, 0xab, 0xe1, 0xef);

//
// IID_PERFECT_HASH_RTL: 9C05A3D6-BC30-45E6-BEA6-504FCC9243A8
//

DEFINE_GUID_EX(IID_PERFECT_HASH_RTL, 0x9c05a3d6, 0xbc30, 0x45e6,
               0xbe, 0xa6, 0x50, 0x4f, 0xcc, 0x92, 0x43, 0xa8);

//
// IID_PERFECT_HASH_ALLOCATOR: F87564D2-B3C7-4CCA-9013-EB59C1E253B7
//

DEFINE_GUID_EX(IID_PERFECT_HASH_ALLOCATOR,
               0xf87564d2, 0xb3c7, 0x4cca,
               0x90, 0x13, 0xeb, 0x59, 0xc1, 0xe2, 0x53, 0xb7);

//
// IID_PERFECT_HASH_FILE 27549274-968A-499A-8349-3133E3D5E649
//

DEFINE_GUID_EX(IID_PERFECT_HASH_FILE,
               0x27549274, 0x968a, 0x499a,
               0x83, 0x49, 0x31, 0x33, 0xe3, 0xd5, 0xe6, 0x49);

//
// IID_PERFECT_HASH_PATH: 267623B1-0C5D-47B1-A297-DF0E5467AFD1
//

DEFINE_GUID_EX(IID_PERFECT_HASH_PATH,
               0x267623b1, 0xc5d, 0x47b1,
               0xa2, 0x97, 0xdf, 0xe, 0x54, 0x67, 0xaf, 0xd1);

//
// IID_PERFECT_HASH_DIRECTORY: {5D673839-1686-411E-9902-46C6E97CA567}
//

DEFINE_GUID_EX(IID_PERFECT_HASH_DIRECTORY,
               0x5d673839, 0x1686, 0x411e,
               0x99, 0x2, 0x46, 0xc6, 0xe9, 0x7c, 0xa5, 0x67);

//
// IID_PERFECT_HASH_GUARDED_LIST: {14A25BA2-3C18-413F-8C76-A7A91EC88C2A}
//

DEFINE_GUID_EX(IID_PERFECT_HASH_GUARDED_LIST,
               0x14a25ba2, 0x3c18, 0x413f,
               0x8c, 0x76, 0xa7, 0xa9, 0x1e, 0xc8, 0x8c, 0x2a);

//
// IID_PERFECT_HASH_GRAPH: {B906F824-CB59-4696-8477-44D4BA09DA94}
//

DEFINE_GUID_EX(IID_PERFECT_HASH_GRAPH,
               0xb906f824, 0xcb59, 0x4696,
               0x84, 0x77, 0x44, 0xd4, 0xba, 0x9, 0xda, 0x94);

//
// GUID array.
//

static const PCGUID PerfectHashInterfaceGuids[] = {

    NULL,

    &IID_PERFECT_HASH_IUNKNOWN,
    &IID_PERFECT_HASH_ICLASSFACTORY,
    &IID_PERFECT_HASH_KEYS,
    &IID_PERFECT_HASH_CONTEXT,
    &IID_PERFECT_HASH_TABLE,
    &IID_PERFECT_HASH_RTL,
    &IID_PERFECT_HASH_ALLOCATOR,
    &IID_PERFECT_HASH_FILE,
    &IID_PERFECT_HASH_PATH,
    &IID_PERFECT_HASH_DIRECTORY,
    &IID_PERFECT_HASH_GUARDED_LIST,
    &IID_PERFECT_HASH_GRAPH,

    NULL
};

static const BYTE NumberOfPerfectHashInterfaceGuids =
    ARRAYSIZE(PerfectHashInterfaceGuids);

//
// Convert a GUID to an interface ID.
//

FORCEINLINE
PERFECT_HASH_INTERFACE_ID
PerfectHashInterfaceGuidToId(
    _In_ REFIID Guid
    )
{
    BYTE Index;
    BYTE Count;
    PERFECT_HASH_INTERFACE_ID Id = PerfectHashNullInterfaceId;

    if (!Guid) {
        return PerfectHashInvalidInterfaceId;
    }

    //
    // We start the index at 1 in order to skip the first NULL entry.
    //

    Count = NumberOfPerfectHashInterfaceGuids;

    for (Index = 1; Index < Count; Index++) {
        if (InlineIsEqualGUID(Guid, PerfectHashInterfaceGuids[Index])) {
            Id = (PERFECT_HASH_INTERFACE_ID)Index;
            break;
        }
    }

    return Id;
}

//
//
// Define helper macros to handle COM interface glue.
//
// N.B. We abuse the COM spec a bit here in that all of our components,
//      including IUnknown, actually implement IClassFactory.
//


#define DECLARE_COMPONENT(Name, Upper)                           \
    typedef struct _##Upper Upper;                               \
    typedef Upper *P##Upper;                                     \
    typedef const Upper *PC##Upper;                              \
                                                                 \
    typedef                                                      \
    _Must_inspect_result_                                        \
    _Success_(return >= 0)                                       \
    HRESULT                                                      \
    (STDAPICALLTYPE Upper##_QUERY_INTERFACE)(                    \
        _In_ P##Upper Name,                                      \
        _In_ REFIID InterfaceId,                                 \
        _COM_Outptr_ PVOID *Interface                            \
        );                                                       \
    typedef Upper##_QUERY_INTERFACE *P##Upper##_QUERY_INTERFACE; \
                                                                 \
    typedef                                                      \
    ULONG                                                        \
    (STDAPICALLTYPE Upper##_ADD_REF)(                            \
        _In_ P##Upper Name                                       \
        );                                                       \
    typedef Upper##_ADD_REF *P##Upper##_ADD_REF;                 \
                                                                 \
    typedef                                                      \
    ULONG                                                        \
    (STDAPICALLTYPE Upper##_RELEASE)(                            \
        _In_ P##Upper Name                                       \
        );                                                       \
    typedef Upper##_RELEASE *P##Upper##_RELEASE;                 \
                                                                 \
    typedef                                                      \
    _Must_inspect_result_                                        \
    _Success_(return >= 0)                                       \
    HRESULT                                                      \
    (STDAPICALLTYPE Upper##_CREATE_INSTANCE)(                    \
        _In_ P##Upper Name,                                      \
        _In_opt_ PIUNKNOWN UnknownOuter,                         \
        _In_ REFIID InterfaceId,                                 \
        _COM_Outptr_ PVOID *Interface                            \
        );                                                       \
    typedef Upper##_CREATE_INSTANCE *P##Upper##_CREATE_INSTANCE; \
                                                                 \
    typedef                                                      \
    _Must_inspect_result_                                        \
    _Success_(return >= 0)                                       \
    HRESULT                                                      \
    (STDAPICALLTYPE Upper##_LOCK_SERVER)(                        \
        _In_ P##Upper Name,                                      \
        _In_opt_ BOOL Lock                                       \
        );                                                       \
    typedef Upper##_LOCK_SERVER *P##Upper##_LOCK_SERVER


#define DECLARE_COMPONENT_VTBL_HEADER(Upper)   \
    P##Upper##_QUERY_INTERFACE QueryInterface; \
    P##Upper##_ADD_REF AddRef;                 \
    P##Upper##_RELEASE Release;                \
    P##Upper##_CREATE_INSTANCE CreateInstance; \
    P##Upper##_LOCK_SERVER LockServer

//
// Define our COM interfaces.  We include IUnknown and IClassFactory just so
// we're consistent with our Cutler-Normal-Form naming scheme.
//

//
// IUnknown
//

DECLARE_COMPONENT(Unknown, IUNKNOWN);

typedef struct _IUNKNOWN_VTBL {
    DECLARE_COMPONENT_VTBL_HEADER(IUNKNOWN);
} IUNKNOWN_VTBL;
typedef IUNKNOWN_VTBL *PIUNKNOWN_VTBL;

#ifndef _PERFECT_HASH_INTERNAL_BUILD
typedef struct _IUNKNOWN {
    PIUNKNOWN_VTBL Vtbl;
} IUNKNOWN;
typedef IUNKNOWN *PIUNKNOWN;
#endif

//
// IClassFactory
//

DECLARE_COMPONENT(ClassFactory, ICLASSFACTORY);

typedef struct _ICLASSFACTORY_VTBL {
    DECLARE_COMPONENT_VTBL_HEADER(ICLASSFACTORY);
} ICLASSFACTORY_VTBL;
typedef ICLASSFACTORY_VTBL *PICLASSFACTORY_VTBL;

#ifndef _PERFECT_HASH_INTERNAL_BUILD
typedef struct _ICLASSFACTORY {
    PICLASSFACTORY_VTBL Vtbl;
} ICLASSFACTORY;
typedef ICLASSFACTORY *PICLASSFACTORY;
#endif

//
// Define the ALLOCATOR interface.
//

DECLARE_COMPONENT(Allocator, ALLOCATOR);

typedef
_Check_return_
_Ret_maybenull_
_Success_(return != 0)
_Post_writable_byte_size_(Size)
PVOID
(STDAPICALLTYPE ALLOCATOR_MALLOC)(
    _In_ PALLOCATOR Allocator,
    _In_ SIZE_T Size
    );
typedef ALLOCATOR_MALLOC *PALLOCATOR_MALLOC;

typedef
_Check_return_
_Ret_maybenull_
_Success_(return != 0)
_Post_writable_byte_size_(NumberOfElements * ElementSize)
PVOID
(STDAPICALLTYPE ALLOCATOR_CALLOC)(
    _In_ PALLOCATOR Allocator,
    _In_ SIZE_T NumberOfElements,
    _In_ SIZE_T ElementSize
    );
typedef ALLOCATOR_CALLOC *PALLOCATOR_CALLOC;

typedef
_Check_return_
_Ret_reallocated_bytes_(Address, Size)
PVOID
(STDAPICALLTYPE ALLOCATOR_REALLOC)(
    _In_ PALLOCATOR Allocator,
    _Frees_ptr_opt_ PVOID Address,
    _In_ SIZE_T Size
    );
typedef ALLOCATOR_REALLOC *PALLOCATOR_REALLOC;

typedef
_Check_return_
_Ret_reallocated_bytes_(Address, NumberOfElements * ElementSize)
PVOID
(STDAPICALLTYPE ALLOCATOR_RECALLOC)(
    _In_ PALLOCATOR Allocator,
    _Frees_ptr_opt_ PVOID Address,
    _In_ SIZE_T NumberOfElements,
    _In_ SIZE_T ElementSize
    );
typedef ALLOCATOR_RECALLOC *PALLOCATOR_RECALLOC;

typedef
VOID
(STDAPICALLTYPE ALLOCATOR_FREE)(
    _In_ PALLOCATOR Allocator,
    _Frees_ptr_opt_ PVOID Address
    );
typedef ALLOCATOR_FREE *PALLOCATOR_FREE;

typedef
VOID
(STDAPICALLTYPE ALLOCATOR_FREE_POINTER)(
    _In_ PALLOCATOR Allocator,
    _Inout_ PVOID *AddressPointer
    );
typedef ALLOCATOR_FREE_POINTER *PALLOCATOR_FREE_POINTER;

typedef
VOID
(STDAPICALLTYPE ALLOCATOR_FREE_STRING_BUFFER)(
    _In_ PALLOCATOR Allocator,
    _In_ PSTRING String
    );
typedef ALLOCATOR_FREE_STRING_BUFFER
      *PALLOCATOR_FREE_STRING_BUFFER;

typedef
VOID
(STDAPICALLTYPE ALLOCATOR_FREE_UNICODE_STRING_BUFFER)(
    _In_ PALLOCATOR Allocator,
    _In_ PUNICODE_STRING String
    );
typedef ALLOCATOR_FREE_UNICODE_STRING_BUFFER
      *PALLOCATOR_FREE_UNICODE_STRING_BUFFER;

typedef
_Check_return_
_Success_(return != 0)
_Ret_maybenull_
_Post_writable_byte_size_(Size)
PVOID
(STDAPICALLTYPE ALLOCATOR_ALIGNED_MALLOC)(
    _In_ PALLOCATOR Allocator,
    _In_ SIZE_T Size,
    _In_ SIZE_T Alignment
    );
typedef ALLOCATOR_ALIGNED_MALLOC *PALLOCATOR_ALIGNED_MALLOC;

typedef
_Check_return_
_Success_(return != 0)
_Ret_maybenull_
_Post_writable_byte_size_(NumberOfElements * ElementSize)
PVOID
(STDAPICALLTYPE ALLOCATOR_ALIGNED_CALLOC)(
    _In_ PALLOCATOR Allocator,
    _In_ SIZE_T NumberOfElements,
    _In_ SIZE_T ElementSize,
    _In_ SIZE_T Alignment
    );
typedef ALLOCATOR_ALIGNED_CALLOC *PALLOCATOR_ALIGNED_CALLOC;

typedef
_Check_return_
_Ret_maybenull_
_Ret_reallocated_bytes_(Address, Size)
PVOID
(STDAPICALLTYPE ALLOCATOR_ALIGNED_REALLOC)(
    _In_ PALLOCATOR Allocator,
    _Frees_ptr_opt_ PVOID Address,
    _In_ SIZE_T Size,
    _In_ SIZE_T Alignment
    );
typedef ALLOCATOR_ALIGNED_REALLOC *PALLOCATOR_ALIGNED_REALLOC;

typedef
_Check_return_
_Ret_maybenull_
_Ret_reallocated_bytes_(Address, NumberOfElements * ElementSize)
PVOID
(STDAPICALLTYPE ALLOCATOR_ALIGNED_RECALLOC)(
    _In_ PALLOCATOR Allocator,
    _Frees_ptr_opt_ PVOID Address,
    _In_ SIZE_T NumberOfElements,
    _In_ SIZE_T ElementSize,
    _In_ SIZE_T Alignment
    );
typedef ALLOCATOR_ALIGNED_RECALLOC *PALLOCATOR_ALIGNED_RECALLOC;

typedef
VOID
(STDAPICALLTYPE ALLOCATOR_ALIGNED_FREE)(
    _In_ PALLOCATOR Allocator,
    _Frees_ptr_opt_ PVOID Address
    );
typedef ALLOCATOR_ALIGNED_FREE *PALLOCATOR_ALIGNED_FREE;

typedef
VOID
(STDAPICALLTYPE ALLOCATOR_ALIGNED_FREE_POINTER)(
    _In_ PALLOCATOR Allocator,
    _Inout_ PVOID *AddressPointer
    );
typedef ALLOCATOR_ALIGNED_FREE_POINTER *PALLOCATOR_ALIGNED_FREE_POINTER;

typedef
_Check_return_
_Success_(return != 0)
_Ret_maybenull_
_Post_writable_byte_size_(Size)
PVOID
(STDAPICALLTYPE ALLOCATOR_ALIGNED_OFFSET_MALLOC)(
    _In_ PALLOCATOR Allocator,
    _In_ SIZE_T Size,
    _In_ SIZE_T Alignment,
    _In_opt_ SIZE_T Offset
    );
typedef ALLOCATOR_ALIGNED_OFFSET_MALLOC *PALLOCATOR_ALIGNED_OFFSET_MALLOC;

typedef
_Check_return_
_Success_(return != 0)
_Ret_maybenull_
_Post_writable_byte_size_(NumberOfElements * ElementSize)
PVOID
(STDAPICALLTYPE ALLOCATOR_ALIGNED_OFFSET_CALLOC)(
    _In_ PALLOCATOR Allocator,
    _In_ SIZE_T NumberOfElements,
    _In_ SIZE_T ElementSize,
    _In_ SIZE_T Alignment,
    _In_opt_ SIZE_T Offset
    );
typedef ALLOCATOR_ALIGNED_OFFSET_CALLOC *PALLOCATOR_ALIGNED_OFFSET_CALLOC;

typedef
_Check_return_
_Ret_maybenull_
_Ret_reallocated_bytes_(Address, Size)
PVOID
(STDAPICALLTYPE ALLOCATOR_ALIGNED_OFFSET_REALLOC)(
    _In_ PALLOCATOR Allocator,
    _Frees_ptr_opt_ PVOID Address,
    _In_ SIZE_T Size,
    _In_ SIZE_T Alignment,
    _In_opt_ SIZE_T Offset
    );
typedef ALLOCATOR_ALIGNED_OFFSET_REALLOC *PALLOCATOR_ALIGNED_OFFSET_REALLOC;

typedef
_Check_return_
_Ret_maybenull_
_Ret_reallocated_bytes_(Address, NumberOfElements * ElementSize)
PVOID
(STDAPICALLTYPE ALLOCATOR_ALIGNED_OFFSET_RECALLOC)(
    _In_ PALLOCATOR Allocator,
    _Frees_ptr_opt_ PVOID Address,
    _In_ SIZE_T NumberOfElements,
    _In_ SIZE_T ElementSize,
    _In_ SIZE_T Alignment,
    _In_opt_ SIZE_T Offset
    );
typedef ALLOCATOR_ALIGNED_OFFSET_RECALLOC *PALLOCATOR_ALIGNED_OFFSET_RECALLOC;

typedef struct _ALLOCATOR_VTBL {
    DECLARE_COMPONENT_VTBL_HEADER(ALLOCATOR);
    PALLOCATOR_MALLOC Malloc;
    PALLOCATOR_CALLOC Calloc;
    PALLOCATOR_REALLOC ReAlloc;
    PALLOCATOR_RECALLOC ReCalloc;
    PALLOCATOR_FREE Free;
    PALLOCATOR_FREE_POINTER FreePointer;
    PALLOCATOR_FREE_STRING_BUFFER FreeStringBuffer;
    PALLOCATOR_FREE_UNICODE_STRING_BUFFER FreeUnicodeStringBuffer;
    PALLOCATOR_ALIGNED_MALLOC AlignedMalloc;
    PALLOCATOR_ALIGNED_CALLOC AlignedCalloc;
    PALLOCATOR_ALIGNED_REALLOC AlignedReAlloc;
    PALLOCATOR_ALIGNED_RECALLOC AlignedReCalloc;
    PALLOCATOR_ALIGNED_FREE AlignedFree;
    PALLOCATOR_ALIGNED_FREE_POINTER AlignedFreePointer;
    PALLOCATOR_ALIGNED_OFFSET_MALLOC AlignedOffsetMalloc;
    PALLOCATOR_ALIGNED_OFFSET_CALLOC AlignedOffsetCalloc;
    PALLOCATOR_ALIGNED_OFFSET_REALLOC AlignedOffsetReAlloc;
    PALLOCATOR_ALIGNED_OFFSET_RECALLOC AlignedOffsetReCalloc;
} ALLOCATOR_VTBL;
typedef ALLOCATOR_VTBL *PALLOCATOR_VTBL;

#ifndef _PERFECT_HASH_INTERNAL_BUILD
typedef struct _ALLOCATOR {
    PALLOCATOR_VTBL Vtbl;
} ALLOCATOR;
typedef ALLOCATOR *PALLOCATOR;
#endif

//
// Define the PERFECT_HASH_PATH interface.
//

DECLARE_COMPONENT(Path, PERFECT_HASH_PATH);

typedef struct _PERFECT_HASH_PATH_PARTS {

    //
    // Fully-qualified, NULL-terminated path of the file.  Path.Buffer is
    // owned by this component, and allocated via the Allocator.
    //

    UNICODE_STRING FullPath;

    //
    // N.B. The following path component fields all point within Path.Buffer.
    //

    //
    // Drive (e.g. "C" or "\\??\\...".
    //

    UNICODE_STRING Drive;

    //
    // Fully-qualified directory, including drive.
    //

    UNICODE_STRING Directory;

    //
    // Base name of the file (i.e. file name excluding extension).
    //

    UNICODE_STRING BaseName;

    //
    // File name (includes extension).
    //

    UNICODE_STRING FileName;

    //
    // File extension (e.g. ".keys").
    //

    UNICODE_STRING Extension;

    //
    // Stream name if applicable (e.g. ":Info").
    //

    UNICODE_STRING StreamName;

    //
    // If the base name (above) is a valid C identifier, an ASCII-encoded
    // version of the string will be available in the following field.  This
    // is guaranteed to only use ASCII (7-bit) characters and will be a valid
    // C identifier (only starts with _, a-z or A-Z, and only contains _, 0-9,
    // a-z and A-Z characters).
    //

    STRING BaseNameA;

    //
    // As above, but captures just the "table name" part of the file.  This
    // will exclude any additional suffix string data appended to the file name
    // during creation.  This buffer is identical to BaseNameA.Buffer, however,
    // the lengths differ.
    //

    STRING TableNameA;

    //
    // As above, but an uppercase version of the base name and table name if
    // applicable.
    //

    STRING BaseNameUpperA;
    STRING TableNameUpperA;

} PERFECT_HASH_PATH_PARTS;
typedef PERFECT_HASH_PATH_PARTS *PPERFECT_HASH_PATH_PARTS;
typedef const PERFECT_HASH_PATH_PARTS *PCPERFECT_HASH_PATH_PARTS;

typedef
_Check_return_
_Success_(return >= 0)
_Requires_lock_not_held_(Path->Lock)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_PATH_COPY)(
    _In_ PPERFECT_HASH_PATH Path,
    _In_ PCUNICODE_STRING Source,
    _Out_opt_ PCPERFECT_HASH_PATH_PARTS *Parts,
    _Reserved_ PVOID Reserved
    );
typedef PERFECT_HASH_PATH_COPY *PPERFECT_HASH_PATH_COPY;

//
// Define path creation method and supporting flags.
//

typedef
_Check_return_
_Success_(return >= 0)
_Requires_lock_not_held_(Path->Lock)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_PATH_CREATE)(
    _In_ PPERFECT_HASH_PATH Path,
    _In_ PPERFECT_HASH_PATH ExistingPath,
    _In_opt_ PCUNICODE_STRING NewDirectory,
    _In_opt_ PCUNICODE_STRING DirectorySuffix,
    _In_opt_ PCUNICODE_STRING NewBaseName,
    _In_opt_ PCUNICODE_STRING BaseNameSuffix,
    _In_opt_ PCUNICODE_STRING NewExtension,
    _In_opt_ PCUNICODE_STRING NewStreamName,
    _Out_opt_ PCPERFECT_HASH_PATH_PARTS *Parts,
    _Reserved_ PVOID Reserved
    );
typedef PERFECT_HASH_PATH_CREATE *PPERFECT_HASH_PATH_CREATE;

typedef
_Check_return_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Path->Lock)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_PATH_RESET)(
    _In_ PPERFECT_HASH_PATH Path
    );
typedef PERFECT_HASH_PATH_RESET *PPERFECT_HASH_PATH_RESET;

typedef
_Check_return_
_Success_(return >= 0)
_Requires_lock_not_held_(Path->Lock)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_PATH_GET_PARTS)(
    _In_ PPERFECT_HASH_PATH Path,
    _Out_ PCPERFECT_HASH_PATH_PARTS *Parts
    );
typedef PERFECT_HASH_PATH_GET_PARTS *PPERFECT_HASH_PATH_GET_PARTS;

#ifndef _PERFECT_HASH_INTERNAL_BUILD
typedef struct _PERFECT_HASH_PATH_VTBL {
    DECLARE_COMPONENT_VTBL_HEADER(PERFECT_HASH_PATH);
    PPERFECT_HASH_PATH_COPY Copy;
    PPERFECT_HASH_PATH_CREATE Create;
    PPERFECT_HASH_PATH_RESET Reset;
    PPERFECT_HASH_PATH_GET_PARTS GetParts;
} PERFECT_HASH_PATH_VTBL;
typedef PERFECT_HASH_PATH_VTBL *PPERFECT_HASH_PATH_VTBL;

typedef struct _PERFECT_HASH_PATH {
    PPERFECT_HASH_PATH_VTBL Vtbl;
    SRWLOCK Lock;
} PERFECT_HASH_PATH;
typedef PERFECT_HASH_PATH *PPERFECT_HASH_PATH;
#endif

//
// Define helper macros for acquiring and releasing the perfect hash path lock
// in shared and exclusive mode.
//

#define TryAcquirePerfectHashPathLockExclusive(Path) \
    TryAcquireSRWLockExclusive(&Path->Lock)

#define AcquirePerfectHashPathLockExclusive(Path) \
    AcquireSRWLockExclusive(&Path->Lock)

#define ReleasePerfectHashPathLockExclusive(Path) \
    ReleaseSRWLockExclusive(&Path->Lock)

#define TryAcquirePerfectHashPathLockShared(Path) \
    TryAcquireSRWLockShared(&Path->Lock)

#define AcquirePerfectHashPathLockShared(Path) \
    AcquireSRWLockShared(&Path->Lock)

#define ReleasePerfectHashPathLockShared(Path) \
    ReleaseSRWLockShared(&Path->Lock)

//
// Define the PERFECT_HASH_DIRECTORY interface.
//

DECLARE_COMPONENT(Directory, PERFECT_HASH_DIRECTORY);

typedef union _PERFECT_HASH_DIRECTORY_OPEN_FLAGS {

    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // Unused bits.
        //

        ULONG Unused:32;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_DIRECTORY_OPEN_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_DIRECTORY_OPEN_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_DIRECTORY_OPEN_FLAGS *PPERFECT_HASH_DIRECTORY_OPEN_FLAGS;

FORCEINLINE
HRESULT
IsValidDirectoryOpenFlags(
    _In_ PPERFECT_HASH_DIRECTORY_OPEN_FLAGS DirectoryOpenFlags
    )
{

    if (!ARGUMENT_PRESENT(DirectoryOpenFlags)) {
        return E_POINTER;
    }

    if (DirectoryOpenFlags->Unused != 0) {
        return E_FAIL;
    }

    return S_OK;
}

typedef
_Check_return_
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_DIRECTORY_OPEN)(
    _In_ PPERFECT_HASH_DIRECTORY Directory,
    _In_ PPERFECT_HASH_PATH SourcePath,
    _In_opt_ PPERFECT_HASH_DIRECTORY_OPEN_FLAGS DirectoryOpenFlags
    );
typedef PERFECT_HASH_DIRECTORY_OPEN *PPERFECT_HASH_DIRECTORY_OPEN;

//
// Define Create() method and supporting flags.
//

typedef union _PERFECT_HASH_DIRECTORY_CREATE_FLAGS {

    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // Unused bits.
        //

        ULONG Unused:32;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_DIRECTORY_CREATE_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_DIRECTORY_CREATE_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_DIRECTORY_CREATE_FLAGS
      *PPERFECT_HASH_DIRECTORY_CREATE_FLAGS;

FORCEINLINE
HRESULT
IsValidDirectoryCreateFlags(
    _In_ PPERFECT_HASH_DIRECTORY_CREATE_FLAGS DirectoryCreateFlags
    )
{

    if (!ARGUMENT_PRESENT(DirectoryCreateFlags)) {
        return E_POINTER;
    }

    if (DirectoryCreateFlags->Unused != 0) {
        return E_FAIL;
    }

    return S_OK;
}

typedef
_Check_return_
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_DIRECTORY_CREATE)(
    _In_ PPERFECT_HASH_DIRECTORY Directory,
    _In_ PPERFECT_HASH_PATH SourcePath,
    _In_opt_ PPERFECT_HASH_DIRECTORY_CREATE_FLAGS DirectoryCreateFlags
    );
typedef PERFECT_HASH_DIRECTORY_CREATE *PPERFECT_HASH_DIRECTORY_CREATE;

//
// Define the PERFECT_HASH_DIRECTORY_FLAGS structure.
//

typedef union _PERFECT_HASH_DIRECTORY_FLAGS {
    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, indicates the directory was initialized via Open().
        //
        // Invariant:
        //
        //      If Opened == TRUE:
        //          Assert Created == FALSE
        //

        ULONG Opened:1;

        //
        // When set, indicates the directory was initialized via Create().
        //
        // Invariant:
        //
        //      If Created == TRUE:
        //          Assert Loaded == FALSE
        //

        ULONG Created:1;

        //
        // Unused bits.
        //

        ULONG Unused:30;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_DIRECTORY_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_DIRECTORY_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_DIRECTORY_FLAGS *PPERFECT_HASH_DIRECTORY_FLAGS;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_DIRECTORY_GET_FLAGS)(
    _In_ PPERFECT_HASH_DIRECTORY Directory,
    _In_ ULONG SizeOfFlags,
    _Out_writes_bytes_(SizeOfFlags) PPERFECT_HASH_DIRECTORY_FLAGS Flags
    );
typedef PERFECT_HASH_DIRECTORY_GET_FLAGS *PPERFECT_HASH_DIRECTORY_GET_FLAGS;

typedef
_Check_return_
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_DIRECTORY_GET_PATH)(
    _In_ PPERFECT_HASH_DIRECTORY Directory,
    _Inout_opt_ PPERFECT_HASH_PATH *Path
    );
typedef PERFECT_HASH_DIRECTORY_GET_PATH *PPERFECT_HASH_DIRECTORY_GET_PATH;

#ifndef _PERFECT_HASH_INTERNAL_BUILD
typedef struct _PERFECT_HASH_DIRECTORY_VTBL {
    DECLARE_COMPONENT_VTBL_HEADER(PERFECT_HASH_DIRECTORY);
    PPERFECT_HASH_DIRECTORY_OPEN Open;
    PPERFECT_HASH_DIRECTORY_CREATE Create;
    PPERFECT_HASH_DIRECTORY_GET_FLAGS GetFlags;
    PPERFECT_HASH_DIRECTORY_GET_PATH GetPath;
} PERFECT_HASH_DIRECTORY_VTBL;
typedef PERFECT_HASH_DIRECTORY_VTBL *PPERFECT_HASH_DIRECTORY_VTBL;

typedef struct _PERFECT_HASH_DIRECTORY {
    PPERFECT_HASH_DIRECTORY_VTBL Vtbl;
} PERFECT_HASH_DIRECTORY;
typedef PERFECT_HASH_DIRECTORY *PPERFECT_HASH_DIRECTORY;
#endif

//
// Define the PERFECT_HASH_FILE interface.
//

DECLARE_COMPONENT(File, PERFECT_HASH_FILE);

typedef union _PERFECT_HASH_FILE_LOAD_FLAGS {

    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, tries to allocate the file buffer using large pages.  The
        // caller is responsible for ensuring the process can create large pages
        // first by enabling the lock memory privilege.  If large pages can't be
        // allocated (because the lock memory privilege hasn't been enabled, or
        // there are insufficient large pages available to the system), the file
        // will be accessed via the normal memory-mapped address of the
        // underlying file.
        //
        // To determine whether or not the large page allocation was successful,
        // check the UsesLargePages bit of the PERFECT_HASH_FILE_FLAGS enum
        // enum (the flags can be obtained via the GetFlags() vtbl function).
        //

        ULONG TryLargePagesForFileData:1;

        //
        // Unused bits.
        //

        ULONG Unused:31;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_FILE_LOAD_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_FILE_LOAD_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_FILE_LOAD_FLAGS *PPERFECT_HASH_FILE_LOAD_FLAGS;

FORCEINLINE
HRESULT
IsValidFileLoadFlags(
    _In_ PPERFECT_HASH_FILE_LOAD_FLAGS FileLoadFlags
    )
{

    if (!ARGUMENT_PRESENT(FileLoadFlags)) {
        return E_POINTER;
    }

    if (FileLoadFlags->Unused != 0) {
        return E_FAIL;
    }

    return S_OK;
}

typedef
_Check_return_
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_FILE_LOAD)(
    _In_ PPERFECT_HASH_FILE File,
    _In_ PPERFECT_HASH_PATH SourcePath,
    _Out_opt_ PLARGE_INTEGER EndOfFile,
    _In_opt_ PPERFECT_HASH_FILE_LOAD_FLAGS FileLoadFlags
    );
typedef PERFECT_HASH_FILE_LOAD *PPERFECT_HASH_FILE_LOAD;

//
// Define Create() method and supporting flags.
//

typedef union _PERFECT_HASH_FILE_CREATE_FLAGS {

    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, tries to allocate the file buffer using large pages.  The
        // caller is responsible for ensuring the process can create large pages
        // first by enabling the lock memory privilege.  If large pages can't be
        // allocated (because the lock memory privilege hasn't been enabled, or
        // there are insufficient large pages available to the system), the file
        // will be accessed via the normal memory-mapped address of the
        // underlying file.
        //
        // To determine whether or not the large page allocation was successful,
        // check the UsesLargePages bit of the PERFECT_HASH_FILE_FLAGS enum
        // enum (the flags can be obtained via the GetFlags() vtbl function).
        //

        ULONG TryLargePagesForFileData:1;

        //
        // Unused bits.
        //

        ULONG Unused:31;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_FILE_CREATE_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_FILE_CREATE_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_FILE_CREATE_FLAGS *PPERFECT_HASH_FILE_CREATE_FLAGS;

FORCEINLINE
HRESULT
IsValidFileCreateFlags(
    _In_ PPERFECT_HASH_FILE_CREATE_FLAGS FileCreateFlags
    )
{

    if (!ARGUMENT_PRESENT(FileCreateFlags)) {
        return E_POINTER;
    }

    if (FileCreateFlags->Unused != 0) {
        return E_FAIL;
    }

    return S_OK;
}

typedef
_Check_return_
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_FILE_CREATE)(
    _In_ PPERFECT_HASH_FILE File,
    _In_ PPERFECT_HASH_PATH SourcePath,
    _In_ PLARGE_INTEGER EndOfFile,
    _In_opt_ PPERFECT_HASH_DIRECTORY ParentDirectory,
    _In_opt_ PPERFECT_HASH_FILE_CREATE_FLAGS FileCreateFlags
    );
typedef PERFECT_HASH_FILE_CREATE *PPERFECT_HASH_FILE_CREATE;

//
// Define the PERFECT_HASH_FILE_FLAGS structure.
//

typedef union _PERFECT_HASH_FILE_FLAGS {
    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, indicates the file was initialized via Load().
        //
        // Invariant:
        //
        //      If Loaded == TRUE:
        //          Assert Created == FALSE
        //

        ULONG Loaded:1;

        //
        // When set, indicates the file was initialized via Create().
        //
        // Invariant:
        //
        //      If Created == TRUE:
        //          Assert Loaded == FALSE
        //

        ULONG Created:1;

        //
        // When set, indicates the caller did not set the relevant bit for
        // trying large pages in the create or load flags.
        //

        ULONG DoesNotWantLargePages:1;

        //
        // When set, indicates the file data resides in a memory allocation
        // backed by large pages.  In this case, BaseAddress represents the
        // large page address, and MappedAddress represents the original
        // address the file was mapped at.
        //

        ULONG UsesLargePages:1;

        //
        // Unused bits.
        //

        ULONG Unused:28;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_FILE_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_FILE_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_FILE_FLAGS *PPERFECT_HASH_FILE_FLAGS;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_FILE_GET_FLAGS)(
    _In_ PPERFECT_HASH_FILE File,
    _In_ ULONG SizeOfFlags,
    _Out_writes_bytes_(SizeOfFlags) PPERFECT_HASH_FILE_FLAGS Flags
    );
typedef PERFECT_HASH_FILE_GET_FLAGS *PPERFECT_HASH_FILE_GET_FLAGS;

typedef
_Check_return_
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_FILE_GET_PATH)(
    _In_ PPERFECT_HASH_FILE File,
    _Inout_opt_ PPERFECT_HASH_PATH *Path
    );
typedef PERFECT_HASH_FILE_GET_PATH *PPERFECT_HASH_FILE_GET_PATH;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_FILE_GET_RESOURCES)(
    _In_ PPERFECT_HASH_FILE File,
    _Out_opt_ PHANDLE FileHandle,
    _Out_opt_ PHANDLE MappingHandle,
    _Out_opt_ PVOID *BaseAddress,
    _Out_opt_ PVOID *MappedAddress,
    _Out_opt_ PLARGE_INTEGER EndOfFile
    );
typedef PERFECT_HASH_FILE_GET_RESOURCES
      *PPERFECT_HASH_FILE_GET_RESOURCES;

#ifndef _PERFECT_HASH_INTERNAL_BUILD
typedef struct _PERFECT_HASH_FILE_VTBL {
    DECLARE_COMPONENT_VTBL_HEADER(PERFECT_HASH_FILE);
    PPERFECT_HASH_FILE_LOAD Load;
    PPERFECT_HASH_FILE_CREATE Create;
    PPERFECT_HASH_FILE_GET_FLAGS GetFlags;
    PPERFECT_HASH_FILE_GET_PATH GetPath;
    PPERFECT_HASH_FILE_GET_RESOURCES GetResources;
} PERFECT_HASH_FILE_VTBL;
typedef PERFECT_HASH_FILE_VTBL *PPERFECT_HASH_FILE_VTBL;

typedef struct _PERFECT_HASH_FILE {
    PPERFECT_HASH_FILE_VTBL Vtbl;
} PERFECT_HASH_FILE;
typedef PERFECT_HASH_FILE *PPERFECT_HASH_FILE;
#endif


//
// Define the PERFECT_HASH_KEYS interface.
//

typedef union _PERFECT_HASH_KEYS_BITMAP_FLAGS {

    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, indicates the bitmap for the keys contains a single
        // contiguous run of set bits.  That is, between the highest set
        // bit and lowest set bit, all intermediate bits are set.  E.g.:
        //
        //  Contiguous:     00000000000111111111111111110000
        //  Not Contiguous: 00110000000111111111111111110000
        //

        ULONG Contiguous:1;

        //
        // Unused bits.
        //

        ULONG Unused:31;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_KEYS_BITMAP_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_KEYS_BITMAP_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_KEYS_BITMAP_FLAGS *PPERFECT_HASH_KEYS_BITMAP_FLAGS;

typedef struct _PERFECT_HASH_KEYS_BITMAP {

    PERFECT_HASH_KEYS_BITMAP_FLAGS Flags;

    ULONG Bitmap;

    BYTE LongestRunLength;
    BYTE LongestRunStart;
    BYTE TrailingZeros;
    BYTE LeadingZeros;

    ULONG ShiftedMask;

    CHAR String[32];

} PERFECT_HASH_KEYS_BITMAP;
typedef PERFECT_HASH_KEYS_BITMAP *PPERFECT_HASH_KEYS_BITMAP;

DECLARE_COMPONENT(Keys, PERFECT_HASH_KEYS);

typedef union _PERFECT_HASH_KEYS_LOAD_FLAGS {

    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, tries to allocate the keys buffer using large pages.  The
        // caller is responsible for ensuring the process can create large pages
        // first by enabling the lock memory privilege.  If large pages can't be
        // allocated (because the lock memory privilege hasn't been enabled, or
        // there are insufficient large pages available to the system), the keys
        // will be accessed via the normal memory-mapped address of the
        // underlying file.
        //
        // To determine whether or not the large page allocation was successful,
        // check the KeysDataUsesLargePages bit of the PERFECT_HASH_KEYS_FLAGS
        // enum (the flags can be obtained after loading via the GetFlags() vtbl
        // function).
        //

        ULONG TryLargePagesForKeysData:1;

        //
        // Unused bits.
        //

        ULONG Unused:31;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_KEYS_LOAD_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_KEYS_LOAD_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_KEYS_LOAD_FLAGS *PPERFECT_HASH_KEYS_LOAD_FLAGS;

FORCEINLINE
HRESULT
IsValidKeysLoadFlags(
    _In_ PPERFECT_HASH_KEYS_LOAD_FLAGS LoadFlags
    )
{
    if (!ARGUMENT_PRESENT(LoadFlags)) {
        return E_POINTER;
    }

    if (LoadFlags->Unused != 0) {
        return E_FAIL;
    }

    return S_OK;
}

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_KEYS_LOAD)(
    _In_ PPERFECT_HASH_KEYS Keys,
    _In_opt_ PPERFECT_HASH_KEYS_LOAD_FLAGS LoadFlags,
    _In_ PCUNICODE_STRING Path,
    _In_ ULONG KeySizeInBytes
    );
typedef PERFECT_HASH_KEYS_LOAD *PPERFECT_HASH_KEYS_LOAD;

//
// Define the keys flags union and get flags function.
//

typedef union _PERFECT_HASH_KEYS_FLAGS {
    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, indicates the keys were mapped using large pages.
        //

        ULONG KeysDataUsesLargePages:1;

        //
        // When set, indicates the keys are a sequential linear array of
        // values.
        //

        ULONG Linear:1;

        //
        // Unused bits.
        //

        ULONG Unused:30;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_KEYS_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_KEYS_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_KEYS_FLAGS *PPERFECT_HASH_KEYS_FLAGS;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_KEYS_GET_FLAGS)(
    _In_ PPERFECT_HASH_KEYS Keys,
    _In_ ULONG SizeOfFlags,
    _Out_writes_bytes_(SizeOfFlags) PPERFECT_HASH_KEYS_FLAGS Flags
    );
typedef PERFECT_HASH_KEYS_GET_FLAGS *PPERFECT_HASH_KEYS_GET_FLAGS;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_KEYS_GET_ADDRESS)(
    _In_ PPERFECT_HASH_KEYS Keys,
    _Out_ PVOID *BaseAddress,
    _Out_ PULARGE_INTEGER NumberOfElements
    );
typedef PERFECT_HASH_KEYS_GET_ADDRESS *PPERFECT_HASH_KEYS_GET_ADDRESS;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_KEYS_GET_BITMAP)(
    _In_ PPERFECT_HASH_KEYS Keys,
    _In_ ULONG SizeOfBitmap,
    _Out_writes_bytes_(SizeOfBitmap) PPERFECT_HASH_KEYS_BITMAP Bitmap
    );
typedef PERFECT_HASH_KEYS_GET_BITMAP *PPERFECT_HASH_KEYS_GET_BITMAP;

typedef
_Check_return_
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_KEYS_GET_FILE)(
    _In_ PPERFECT_HASH_KEYS Keys,
    _Inout_ PPERFECT_HASH_FILE *File
    );
typedef PERFECT_HASH_KEYS_GET_FILE *PPERFECT_HASH_KEYS_GET_FILE;

typedef struct _PERFECT_HASH_KEYS_VTBL {
    DECLARE_COMPONENT_VTBL_HEADER(PERFECT_HASH_KEYS);
    PPERFECT_HASH_KEYS_LOAD Load;
    PPERFECT_HASH_KEYS_GET_FLAGS GetFlags;
    PPERFECT_HASH_KEYS_GET_ADDRESS GetAddress;
    PPERFECT_HASH_KEYS_GET_BITMAP GetBitmap;
    PPERFECT_HASH_KEYS_GET_FILE GetFile;
} PERFECT_HASH_KEYS_VTBL;
typedef PERFECT_HASH_KEYS_VTBL *PPERFECT_HASH_KEYS_VTBL;

#ifndef _PERFECT_HASH_INTERNAL_BUILD
typedef struct _PERFECT_HASH_KEYS {
    PPERFECT_HASH_KEYS_VTBL Vtbl;
} PERFECT_HASH_KEYS;
typedef PERFECT_HASH_KEYS *PPERFECT_HASH_KEYS;
#endif

//
// Define an enumeration for identifying which backend algorithm variant to
// use for creating the perfect hash table.
//

typedef enum _PERFECT_HASH_ALGORITHM_ID {

    //
    // Explicitly define a null algorithm to take the 0-index slot.
    // This makes enum validation easier.
    //

    PerfectHashNullAlgorithmId         = 0,

    //
    // Begin valid algorithms.
    //

    PerfectHashChm01AlgorithmId        = 1,
    PerfectHashDefaultAlgorithmId      = 1,

    //
    // End valid algorithms.
    //

    //
    // N.B. Keep the next value last.
    //

    PerfectHashInvalidAlgorithmId,

} PERFECT_HASH_ALGORITHM_ID;
typedef PERFECT_HASH_ALGORITHM_ID *PPERFECT_HASH_ALGORITHM_ID;

//
// Provide a simple inline algorithm validation routine.
//

FORCEINLINE
BOOLEAN
IsValidPerfectHashAlgorithmId(
    _In_ PERFECT_HASH_ALGORITHM_ID AlgorithmId
    )
{
    return (
        AlgorithmId > PerfectHashNullAlgorithmId &&
        AlgorithmId < PerfectHashInvalidAlgorithmId
    );
}

//
// Define an enumeration for identifying which hash function variant to use.
//

typedef enum _PERFECT_HASH_HASH_FUNCTION_ID {

    //
    // Explicitly define a null algorithm to take the 0-index slot.
    // This makes enum validation easier.
    //

    PerfectHashNullHashFunctionId               = 0,

    //
    // Begin valid hash functions.
    //

    PerfectHashHashCrc32RotateFunctionId        = 1,
    PerfectHashDefaultHashFunctionId            = 1,

    PerfectHashHashJenkinsFunctionId            = 2,

    //
    // N.B. The following three hash functions are purposefully terrible from
    //      the perspective of generating a good distribution of hash values.
    //      They all have very simple operations and were intended to test the
    //      theory that even with a poor hash function, once we find the right
    //      seed, the hash quality is unimportant.
    //
    //      In practice, that hypothesis appears to be wrong.  Either we find
    //      a solution on average in sqrt(3) attempts (99.9% probability of
    //      having found the solution by attempt 18); or we never find one at
    //      that given table size (and thus, a larger table size needs to be
    //      attempted).
    //

    PerfectHashHashRotateXorFunctionId          = 3,
    PerfectHashHashAddSubXorFunctionId          = 4,
    PerfectHashHashXorFunctionId                = 5,

    //
    // A scratch hash function to use for quickly iterating on new hash
    // functionality without the overhead of having to define a new
    // implementation.
    //

    PerfectHashHashScratchFunctionId            = 6,

    //
    // Additional functions currently in development or otherwise provided
    // without commentary.
    //

    PerfectHashHashCrc32RotateXorFunctionId     = 7,
    PerfectHashHashCrc32FunctionId              = 8,
    PerfectHashHashDjbFunctionId                = 9,
    PerfectHashHashDjbXorFunctionId             = 10,
    PerfectHashHashFnvFunctionId                = 11,
    PerfectHashHashCrc32NotFunctionId           = 12,

    //
    // End valid hash functions.
    //

    //
    // N.B. Keep the next value last.
    //

    PerfectHashInvalidHashFunctionId,

} PERFECT_HASH_HASH_FUNCTION_ID;
typedef PERFECT_HASH_HASH_FUNCTION_ID *PPERFECT_HASH_HASH_FUNCTION_ID;

//
// Provide a simple inline hash function validation routine.
//

FORCEINLINE
BOOLEAN
IsValidPerfectHashHashFunctionId(
    _In_ PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId
    )
{
    return (
        HashFunctionId > PerfectHashNullHashFunctionId &&
        HashFunctionId < PerfectHashInvalidHashFunctionId
    );
}

//
// Define an enumeration for identifying the type of table masking used by the
// underlying perfect hash table.  This has performance and size implications.
// Modulus masking typically results in smaller tables at the expenses of slower
// modulus-based hash functions.  Non-modulus masking requires power-of-2 sized
// tables, which will be larger, but the resulting mask function can be done
// by logical AND instructions, which are fast.
//

typedef enum _PERFECT_HASH_MASK_FUNCTION_ID {

    //
    // Null masking type.
    //

    PerfectHashNullMaskFunctionId          = 0,

    //
    // Being valid masking types.
    //

    PerfectHashModulusMaskFunctionId       = 1,

    PerfectHashAndMaskFunctionId           = 2,
    PerfectHashDefaultMaskFunctionId       = 2,

    //
    // End valid masking types.
    //

    //
    // N.B. Keep the next value last.
    //

    PerfectHashInvalidMaskFunctionId,


} PERFECT_HASH_MASK_FUNCTION_ID;
typedef PERFECT_HASH_MASK_FUNCTION_ID
      *PPERFECT_HASH_MASK_FUNCTION_ID;

//
// Provide a simple inline masking type validation routine.
//

FORCEINLINE
BOOLEAN
IsValidPerfectHashMaskFunctionId(
    _In_ PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId
    )
{
    return (
        MaskFunctionId > PerfectHashNullMaskFunctionId &&
        MaskFunctionId < PerfectHashInvalidMaskFunctionId
    );
}

//
// Masking tends to fall into one of two buckets: modulus and not-modulus.
// Provide an inline routine that guarantees to match all current and future
// modulus masking function IDs.
//

FORCEINLINE
BOOLEAN
IsModulusMasking(
    _In_ PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId
    )
{
    return MaskFunctionId == PerfectHashModulusMaskFunctionId;
}

//
// Define an enumeration for identifying benchmark routines.
//

typedef enum _PERFECT_HASH_BENCHMARK_FUNCTION_ID {

    //
    // Explicitly define a null algorithm to take the 0-index slot.
    // This makes enum validation easier.
    //

    PerfectHashNullBenchmarkFunctionId          = 0,

    //
    // Begin valid benchmarks.
    //

    PerfectHashFastIndexBenchmarkFunctionId     = 1,

    //
    // End valid benchmarks.
    //

    //
    // N.B. Keep the next value last.
    //

    PerfectHashInvalidBenchmarkFunctionId,

} PERFECT_HASH_BENCHMARK_FUNCTION_ID;

//
// Provide a simple inline benchmark validation routine.
//

FORCEINLINE
BOOLEAN
IsValidPerfectHashBenchmarkFunctionId(
    _In_ PERFECT_HASH_BENCHMARK_FUNCTION_ID BenchmarkFunctionId
    )
{
    return (
        BenchmarkFunctionId > PerfectHashNullBenchmarkFunctionId &&
        BenchmarkFunctionId < PerfectHashInvalidBenchmarkFunctionId
    );
}

//
// Define an enumeration for identifying benchmark types.
//

typedef enum _PERFECT_HASH_BENCHMARK_TYPE {

    //
    // Explicitly define a null benchmark type to take the 0-index slot.
    // This makes enum validation easier.
    //

    PerfectHashNullBenchmarkType        = 0,

    //
    // Begin valid benchmark typess.
    //

    PerfectHashSingleBenchmarkType      = 1,
    PerfectHashAllBenchmarkType         = 2,

    //
    // End valid benchmark typess.
    //

    //
    // N.B. Keep the next value last.
    //

    PerfectHashInvalidBenchmarkType,

} PERFECT_HASH_BENCHMARK_TYPE;
typedef PERFECT_HASH_BENCHMARK_TYPE *PPERFECT_HASH_BENCHMARK_TYPE;

//
// Provide a simple inline benchmark type validation routine.
//

FORCEINLINE
BOOLEAN
IsValidPerfectHashBenchmarkType(
    _In_ PERFECT_HASH_BENCHMARK_TYPE BenchmarkType
    )
{
    return (
        BenchmarkType > PerfectHashNullBenchmarkType &&
        BenchmarkType < PerfectHashInvalidBenchmarkType
    );
}

//
// Define the PERFECT_HASH_CONTEXT interface.  This interface is
// responsible for encapsulating threadpool resources and allows perfect hash
// table solutions to be found in parallel.  An instance of this interface must
// be provided to the PERFECT_HASH_TABLE interface's creation routine.
//

DECLARE_COMPONENT(Context, PERFECT_HASH_CONTEXT);

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_CONTEXT_SET_MAXIMUM_CONCURRENCY)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ ULONG MaximumConcurrency
    );
typedef PERFECT_HASH_CONTEXT_SET_MAXIMUM_CONCURRENCY
      *PPERFECT_HASH_CONTEXT_SET_MAXIMUM_CONCURRENCY;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_CONTEXT_GET_MAXIMUM_CONCURRENCY)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _Out_ PULONG MaximumConcurrency
    );
typedef PERFECT_HASH_CONTEXT_GET_MAXIMUM_CONCURRENCY
      *PPERFECT_HASH_CONTEXT_GET_MAXIMUM_CONCURRENCY;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_CONTEXT_SET_BASE_OUTPUT_DIRECTORY)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PCUNICODE_STRING BaseOutputDirectory
    );
typedef PERFECT_HASH_CONTEXT_SET_BASE_OUTPUT_DIRECTORY
      *PPERFECT_HASH_CONTEXT_SET_BASE_OUTPUT_DIRECTORY;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_CONTEXT_GET_BASE_OUTPUT_DIRECTORY)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _Inout_ PPERFECT_HASH_DIRECTORY *BaseOutputDirectory
    );
typedef PERFECT_HASH_CONTEXT_GET_BASE_OUTPUT_DIRECTORY
      *PPERFECT_HASH_CONTEXT_GET_BASE_OUTPUT_DIRECTORY;

//
// Define the self-test flags.
//

typedef union _PERFECT_HASH_CONTEXT_SELF_TEST_FLAGS {

    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // Unused bits.
        //

        ULONG Unused:32;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_CONTEXT_SELF_TEST_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_CONTEXT_SELF_TEST_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_CONTEXT_SELF_TEST_FLAGS
      *PPERFECT_HASH_CONTEXT_SELF_TEST_FLAGS;

FORCEINLINE
HRESULT
IsValidContextSelfTestFlags(
    _In_ PPERFECT_HASH_CONTEXT_SELF_TEST_FLAGS SelfTestFlags
    )
{

    if (!ARGUMENT_PRESENT(SelfTestFlags)) {
        return E_POINTER;
    }

    if (SelfTestFlags->Unused != 0) {
        return E_FAIL;
    }

    return S_OK;
}

//
// Define the bulk-create flags.
//

typedef union _PERFECT_HASH_CONTEXT_BULK_CREATE_FLAGS {

    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // Unused bits.
        //

        ULONG Unused:32;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_CONTEXT_BULK_CREATE_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_CONTEXT_BULK_CREATE_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_CONTEXT_BULK_CREATE_FLAGS
      *PPERFECT_HASH_CONTEXT_BULK_CREATE_FLAGS;

FORCEINLINE
HRESULT
IsValidContextBulkCreateFlags(
    _In_ PPERFECT_HASH_CONTEXT_BULK_CREATE_FLAGS BulkCreateFlags
    )
{

    if (!ARGUMENT_PRESENT(BulkCreateFlags)) {
        return E_POINTER;
    }

    if (BulkCreateFlags->Unused != 0) {
        return E_FAIL;
    }

    return S_OK;
}

//
// Define the table create flags here as they're needed for SelfTest().
//

typedef union _PERFECT_HASH_TABLE_CREATE_FLAGS {

    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, disables the default "first solved graph wins" behavior
        // and enables the "find best memory coverage" behavior.  This must
        // be used in conjunction with create table parameters that specify
        // the number of attempts at finding a best solution, as well as a
        // predicate for evaluating what constitutes "best" (e.g. highest
        // number of empty cache lines in the final assigned array).  This
        // option is considerably more CPU intensive than the "first graph
        // wins" behavior, as the create table routine will not return until
        // it has attempted the requested number of solutions.
        //
        // N.B. "Best" is being used in the relative sense here, i.e. there
        //      are no guarantees that the resulting table *is* actually the
        //      best (best what?  best performing?); only that it was the
        //      table with the highest/lowest value for a given predicate.
        //
        // N.B. See ../src/PerfectHash/Graph.h for some more information about
        //      the best memory coverage behavior.
        //

        ULONG FindBestGraph:1;

        //
        // Unused bits.
        //

        ULONG Unused:31;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_TABLE_CREATE_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_TABLE_CREATE_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_TABLE_CREATE_FLAGS
      *PPERFECT_HASH_TABLE_CREATE_FLAGS;

FORCEINLINE
HRESULT
IsValidTableCreateFlags(
    _In_ PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags
    )
{

    if (!ARGUMENT_PRESENT(TableCreateFlags)) {
        return E_POINTER;
    }

    if (TableCreateFlags->Unused != 0) {
        return E_FAIL;
    }

    return S_OK;
}

//
// Define the table load flags here as they're needed for SelfTest().
//

typedef union _PERFECT_HASH_TABLE_LOAD_FLAGS {

    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, tries to allocate the table data using large pages.  The
        // caller is responsible for ensuring the process can create large pages
        // first by enabling the lock memory privilege.  If large pages can't be
        // allocated (because the lock memory privilege hasn't been enabled, or
        // there are insufficient large pages available to the system), the
        // table data will be accessed via the normal memory-mapped address of
        // the underlying file.
        //
        // To determine whether or not the large page allocation was successful,
        // check the TableDataUsesLargePages bit of PERFECT_HASH_TABLE_FLAGS
        // enum (the flags can be obtained after loading via the GetFlags() vtbl
        // function).
        //

        ULONG TryLargePagesForTableData:1;

        //
        // As above, but for the values array (i.e. the memory used to save the
        // values inserted into the table via Insert()).  Corresponds to the
        // ValuesArrayUsesLargePages bit of PERFECT_HASH_TABLE_FLAGS.
        //

        ULONG TryLargePagesForValuesArray:1;

        //
        // Unused bits.
        //

        ULONG Unused:30;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_TABLE_LOAD_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_TABLE_LOAD_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_TABLE_LOAD_FLAGS *PPERFECT_HASH_TABLE_LOAD_FLAGS;

FORCEINLINE
HRESULT
IsValidTableLoadFlags(
    _In_ PPERFECT_HASH_TABLE_LOAD_FLAGS LoadFlags
    )
{
    if (!ARGUMENT_PRESENT(LoadFlags)) {
        return E_POINTER;
    }

    if (LoadFlags->Unused != 0) {
        return E_FAIL;
    }

    return S_OK;
}

//
// Define table compilation flags here as they're needed for SelfTest().
//

typedef union _PERFECT_HASH_TABLE_COMPILE_FLAGS {

    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, ignores any optimized assembly routines available for
        // compilation and uses the standard C routines.  If no assembly
        // routine has been written for the given table configuration (i.e.
        // algorithm, hash function and masking type), setting this bit has no
        // effect.
        //

        ULONG IgnoreAssembly:1;

        //
        // Unused bits.
        //

        ULONG Unused:31;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_TABLE_COMPILE_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_TABLE_COMPILE_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_TABLE_COMPILE_FLAGS *PPERFECT_HASH_TABLE_COMPILE_FLAGS;

FORCEINLINE
HRESULT
IsValidTableCompileFlags(
    _In_ PPERFECT_HASH_TABLE_COMPILE_FLAGS CompileFlags
    )
{
    if (!ARGUMENT_PRESENT(CompileFlags)) {
        return E_POINTER;
    }

    if (CompileFlags->Unused != 0) {
        return E_FAIL;
    }

    return S_OK;
}

//
// N.B. The table create parameters are still a work-in-progress.
//

typedef enum PERFECT_HASH_TABLE_CREATE_PARAMETER_ID {
    PerfectHashTableCreateParameterNullId = 0,

    PerfectHashTableCreateParameterChm01AttemptsBeforeTableResizeId,
    PerfectHashTableCreateParameterChm01MaxNumberOfTableResizesId,
    PerfectHashTableCreateParameterChm01BestCoverageNumAttemptsId,
    PerfectHashTableCreateParameterChm01BestCoverageTypeId,

    PerfectHashTableCreateParameterInvalidId,
} PERFECT_HASH_TABLE_CREATE_PARAMETER_ID;

typedef enum _PERFECT_HASH_TABLE_BEST_COVERAGE_TYPE_ID {
    BestCoverageTypeNullId = 0,
    BestCoverageTypeHighestNumberOfEmptyCacheLinesId,
    BestCoverageTypeInvalidId,
} PERFECT_HASH_TABLE_BEST_COVERAGE_TYPE;

FORCEINLINE
BOOLEAN
IsValidBestCoverageType(
    _In_ PERFECT_HASH_TABLE_BEST_COVERAGE_TYPE CoverageType
    )
{
    return (
        CoverageType > BestCoverageTypeNullId &&
        CoverageType < BestCoverageTypeInvalidId
    );
}

//
// Disable warning C4820:
//      '<anonymous-tag>': '4' bytes padding added after data member 'Id'.
//

#pragma warning(push)
#pragma warning(disable: 4820)
typedef struct _PERFECT_HASH_TABLE_CREATE_PARAMETER {
    PERFECT_HASH_TABLE_CREATE_PARAMETER_ID Id;
    union {
        PVOID AsVoid;
        LONG AsLong;
        ULONG AsULong;
        LONGLONG AsLongLong;
        ULONGLONG AsULongLong;
        LARGE_INTEGER AsLargeInteger;
        ULARGE_INTEGER AsULargeInteger;
        PERFECT_HASH_TABLE_BEST_COVERAGE_TYPE AsBestCoverageType;
    };
} PERFECT_HASH_TABLE_CREATE_PARAMETER;
typedef PERFECT_HASH_TABLE_CREATE_PARAMETER
      *PPERFECT_HASH_TABLE_CREATE_PARAMETER;
#pragma warning(pop)


typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_CONTEXT_SELF_TEST)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PCUNICODE_STRING TestDataDirectory,
    _In_ PCUNICODE_STRING BaseOutputDirectory,
    _In_ PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    _In_ PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    _In_ PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    _In_opt_ PPERFECT_HASH_CONTEXT_SELF_TEST_FLAGS ContextSelfTestFlags,
    _In_opt_ PPERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags,
    _In_opt_ PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags,
    _In_opt_ PPERFECT_HASH_TABLE_LOAD_FLAGS TableLoadFlags,
    _In_opt_ PPERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags,
    _In_opt_ ULONG NumberOfTableCreateParameters,
    _In_opt_ PPERFECT_HASH_TABLE_CREATE_PARAMETER TableCreateParameters
    );
typedef PERFECT_HASH_CONTEXT_SELF_TEST *PPERFECT_HASH_CONTEXT_SELF_TEST;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_CONTEXT_SELF_TEST_ARGVW)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ ULONG NumberOfArguments,
    _In_ LPWSTR *ArgvW
    );
typedef PERFECT_HASH_CONTEXT_SELF_TEST_ARGVW
      *PPERFECT_HASH_CONTEXT_SELF_TEST_ARGVW;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_CONTEXT_EXTRACT_SELF_TEST_ARGS_FROM_ARGVW)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ ULONG NumberOfArguments,
    _In_ LPWSTR *ArgvW,
    _In_ PUNICODE_STRING TestDataDirectory,
    _In_ PUNICODE_STRING BaseOutputDirectory,
    _Inout_ PPERFECT_HASH_ALGORITHM_ID AlgorithmId,
    _Inout_ PPERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    _Inout_ PPERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    _Inout_ PULONG MaximumConcurrency,
    _Inout_ PPERFECT_HASH_CONTEXT_SELF_TEST_FLAGS ContextSelfTestFlags,
    _Inout_ PPERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags,
    _Inout_ PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags,
    _Inout_ PPERFECT_HASH_TABLE_LOAD_FLAGS TableLoadFlags,
    _Inout_ PPERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags,
    _Inout_ PULONG NumberOfTableCreateParameters,
    _Inout_ PPERFECT_HASH_TABLE_CREATE_PARAMETER *TableCreateParameters
    );
typedef PERFECT_HASH_CONTEXT_EXTRACT_SELF_TEST_ARGS_FROM_ARGVW
      *PPERFECT_HASH_CONTEXT_EXTRACT_SELF_TEST_ARGS_FROM_ARGVW;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_CONTEXT_BULK_CREATE)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PCUNICODE_STRING KeysDirectory,
    _In_ PCUNICODE_STRING BaseOutputDirectory,
    _In_ PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    _In_ PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    _In_ PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    _In_opt_ PPERFECT_HASH_CONTEXT_BULK_CREATE_FLAGS ContextBulkCreateFlags,
    _In_opt_ PPERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags,
    _In_opt_ PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags,
    _In_opt_ PPERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags,
    _In_opt_ ULONG NumberOfTableCreateParameters,
    _In_opt_ PPERFECT_HASH_TABLE_CREATE_PARAMETER TableCreateParameters
    );
typedef PERFECT_HASH_CONTEXT_BULK_CREATE *PPERFECT_HASH_CONTEXT_BULK_CREATE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_CONTEXT_BULK_CREATE_ARGVW)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ ULONG NumberOfArguments,
    _In_ LPWSTR *ArgvW
    );
typedef PERFECT_HASH_CONTEXT_BULK_CREATE_ARGVW
      *PPERFECT_HASH_CONTEXT_BULK_CREATE_ARGVW;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_CONTEXT_EXTRACT_BULK_CREATE_ARGS_FROM_ARGVW)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ ULONG NumberOfArguments,
    _In_ LPWSTR *ArgvW,
    _In_ PUNICODE_STRING KeysDirectory,
    _In_ PUNICODE_STRING BaseOutputDirectory,
    _Inout_ PPERFECT_HASH_ALGORITHM_ID AlgorithmId,
    _Inout_ PPERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    _Inout_ PPERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    _Inout_ PULONG MaximumConcurrency,
    _Inout_ PPERFECT_HASH_CONTEXT_BULK_CREATE_FLAGS ContextBulkCreateFlags,
    _Inout_ PPERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags,
    _Inout_ PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags,
    _Inout_ PPERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags,
    _Inout_ PULONG NumberOfTableCreateParameters,
    _Inout_ PPERFECT_HASH_TABLE_CREATE_PARAMETER *TableCreateParameters
    );
typedef PERFECT_HASH_CONTEXT_EXTRACT_BULK_CREATE_ARGS_FROM_ARGVW
      *PPERFECT_HASH_CONTEXT_EXTRACT_BULK_CREATE_ARGS_FROM_ARGVW;


typedef struct _PERFECT_HASH_CONTEXT_VTBL {
    DECLARE_COMPONENT_VTBL_HEADER(PERFECT_HASH_CONTEXT);
    PPERFECT_HASH_CONTEXT_SET_MAXIMUM_CONCURRENCY SetMaximumConcurrency;
    PPERFECT_HASH_CONTEXT_GET_MAXIMUM_CONCURRENCY GetMaximumConcurrency;
    PPERFECT_HASH_CONTEXT_SET_BASE_OUTPUT_DIRECTORY SetBaseOutputDirectory;
    PPERFECT_HASH_CONTEXT_GET_BASE_OUTPUT_DIRECTORY GetBaseOutputDirectory;
    PPERFECT_HASH_CONTEXT_SELF_TEST SelfTest;
    PPERFECT_HASH_CONTEXT_SELF_TEST_ARGVW SelfTestArgvW;
    PPERFECT_HASH_CONTEXT_EXTRACT_SELF_TEST_ARGS_FROM_ARGVW
        ExtractSelfTestArgsFromArgvW;
    PPERFECT_HASH_CONTEXT_BULK_CREATE BulkCreate;
    PPERFECT_HASH_CONTEXT_BULK_CREATE_ARGVW BulkCreateArgvW;
    PPERFECT_HASH_CONTEXT_EXTRACT_BULK_CREATE_ARGS_FROM_ARGVW
        ExtractBulkCreateArgsFromArgvW;
} PERFECT_HASH_CONTEXT_VTBL;
typedef PERFECT_HASH_CONTEXT_VTBL *PPERFECT_HASH_CONTEXT_VTBL;

#ifndef _PERFECT_HASH_INTERNAL_BUILD
typedef struct _PERFECT_HASH_CONTEXT {
    PPERFECT_HASH_CONTEXT_VTBL Vtbl;
} PERFECT_HASH_CONTEXT;
typedef PERFECT_HASH_CONTEXT *PPERFECT_HASH_CONTEXT;
#endif

//
// Define the PERFECT_HASH_TABLE interface.
//

DECLARE_COMPONENT(Table, PERFECT_HASH_TABLE);

typedef struct _PERFECT_HASH_TABLE PERFECT_HASH_TABLE;
typedef PERFECT_HASH_TABLE *PPERFECT_HASH_TABLE;

typedef
_Check_return_
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_CREATE)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    _In_ PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    _In_ PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    _In_ PPERFECT_HASH_KEYS Keys,
    _In_opt_ PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags,
    _In_opt_ ULONG NumberOfTableCreateParameters,
    _In_opt_ PPERFECT_HASH_TABLE_CREATE_PARAMETER TableCreateParameters
    );
typedef PERFECT_HASH_TABLE_CREATE *PPERFECT_HASH_TABLE_CREATE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_LOAD)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_opt_ PPERFECT_HASH_TABLE_LOAD_FLAGS LoadFlags,
    _In_ PCUNICODE_STRING Path,
    _In_opt_ PPERFECT_HASH_KEYS Keys
    );
typedef PERFECT_HASH_TABLE_LOAD *PPERFECT_HASH_TABLE_LOAD;

//
// Define the table flags enum and associated function pointer to obtain the
// flags from a loaded table.
//

typedef union _PERFECT_HASH_TABLE_FLAGS {
    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, indicates the table was created via a context.
        //
        // Invariant:
        //
        //      If Created == TRUE:
        //          Assert Loaded == FALSE
        //

        ULONG Created:1;

        //
        // When set, indicates the table was loaded from disk via a previously
        // created instance.
        //
        // Invariant:
        //
        //      If Loaded == TRUE:
        //          Assert Created == FALSE
        //

        ULONG Loaded:1;

        //
        // When set, indicates large pages are in use for the memory backing
        // the table data (e.g. the vertices array).  This will only ever be
        // set for loaded tables, not created ones.
        //

        ULONG TableDataUsesLargePages:1;

        //
        // When set, indicates the values array was allocated with large pages.
        // This will only ever be set for loaded tables, not created ones.
        //

        ULONG ValuesArrayUsesLargePages:1;

        //
        // Unused bits.
        //

        ULONG Unused:28;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_TABLE_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_TABLE_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_TABLE_FLAGS *PPERFECT_HASH_TABLE_FLAGS;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_GET_FLAGS)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ ULONG SizeOfFlags,
    _Out_writes_bytes_(SizeOfFlags) PPERFECT_HASH_TABLE_FLAGS Flags
    );
typedef PERFECT_HASH_TABLE_GET_FLAGS *PPERFECT_HASH_TABLE_GET_FLAGS;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_COMPILE)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_opt_ PPERFECT_HASH_TABLE_COMPILE_FLAGS CompileFlags,
    _In_ PERFECT_HASH_CPU_ARCH_ID CpuArchId
    );
typedef PERFECT_HASH_TABLE_COMPILE *PPERFECT_HASH_TABLE_COMPILE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_TEST)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_opt_ PPERFECT_HASH_KEYS Keys,
    _In_opt_ BOOLEAN DebugBreakOnFailure
    );
typedef PERFECT_HASH_TABLE_TEST *PPERFECT_HASH_TABLE_TEST;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_INSERT)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ ULONG Key,
    _In_ ULONG Value,
    _Out_opt_ PULONG PreviousValue
    );
typedef PERFECT_HASH_TABLE_INSERT *PPERFECT_HASH_TABLE_INSERT;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_LOOKUP)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ ULONG Key,
    _Out_ PULONG Value
    );
typedef PERFECT_HASH_TABLE_LOOKUP *PPERFECT_HASH_TABLE_LOOKUP;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_DELETE)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ ULONG Key,
    _Out_opt_ PULONG PreviousValue
    );
typedef PERFECT_HASH_TABLE_DELETE *PPERFECT_HASH_TABLE_DELETE;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_INDEX)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ ULONG Key,
    _Out_ PULONG Index
    );
typedef PERFECT_HASH_TABLE_INDEX *PPERFECT_HASH_TABLE_INDEX;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_HASH)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ ULONG Input,
    _Out_ PULONGLONG Hash
    );
typedef PERFECT_HASH_TABLE_HASH *PPERFECT_HASH_TABLE_HASH;

typedef
HRESULT
(NTAPI PERFECT_HASH_TABLE_SEEDED_HASH)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ ULONG Input,
    _In_ ULONG NumberOfSeeds,
    _In_reads_(NumberOfSeeds) PULONG Seeds,
    _Out_ PULONGLONG Hash
    );
typedef PERFECT_HASH_TABLE_SEEDED_HASH *PPERFECT_HASH_TABLE_SEEDED_HASH;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_MASK_HASH)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ ULONG Input,
    _Out_ PULONG Masked
    );
typedef PERFECT_HASH_TABLE_MASK_HASH *PPERFECT_HASH_TABLE_MASK_HASH;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_MASK_INDEX)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ ULONGLONG Input,
    _Out_ PULONG Masked
    );
typedef PERFECT_HASH_TABLE_MASK_INDEX *PPERFECT_HASH_TABLE_MASK_INDEX;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_GET_ALGORITHM_NAME)(
    _In_ PPERFECT_HASH_TABLE Table,
    _Out_ PCUNICODE_STRING *Name
    );
typedef PERFECT_HASH_TABLE_GET_ALGORITHM_NAME
      *PPERFECT_HASH_TABLE_GET_ALGORITHM_NAME;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_GET_HASH_FUNCTION_NAME)(
    _In_ PPERFECT_HASH_TABLE Table,
    _Out_ PCUNICODE_STRING *Name
    );
typedef PERFECT_HASH_TABLE_GET_HASH_FUNCTION_NAME
      *PPERFECT_HASH_TABLE_GET_HASH_FUNCTION_NAME;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_GET_MASK_FUNCTION_NAME)(
    _In_ PPERFECT_HASH_TABLE Table,
    _Out_ PCUNICODE_STRING *Name
    );
typedef PERFECT_HASH_TABLE_GET_MASK_FUNCTION_NAME
      *PPERFECT_HASH_TABLE_GET_MASK_FUNCTION_NAME;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_GET_FILE)(
    _In_ PPERFECT_HASH_TABLE Table,
    _Inout_opt_ PPERFECT_HASH_FILE *File
    );
typedef PERFECT_HASH_TABLE_GET_FILE *PPERFECT_HASH_TABLE_GET_FILE;

typedef struct _PERFECT_HASH_TABLE_VTBL {
    DECLARE_COMPONENT_VTBL_HEADER(PERFECT_HASH_TABLE);
    PPERFECT_HASH_TABLE_CREATE Create;
    PPERFECT_HASH_TABLE_LOAD Load;
    PPERFECT_HASH_TABLE_GET_FLAGS GetFlags;
    PPERFECT_HASH_TABLE_COMPILE Compile;
    PPERFECT_HASH_TABLE_TEST Test;
    PPERFECT_HASH_TABLE_INSERT Insert;
    PPERFECT_HASH_TABLE_LOOKUP Lookup;
    PPERFECT_HASH_TABLE_DELETE Delete;
    PPERFECT_HASH_TABLE_INDEX Index;
    PPERFECT_HASH_TABLE_HASH Hash;
    PPERFECT_HASH_TABLE_MASK_HASH MaskHash;
    PPERFECT_HASH_TABLE_MASK_INDEX MaskIndex;
    PPERFECT_HASH_TABLE_SEEDED_HASH SeededHash;
    PPERFECT_HASH_TABLE_INDEX FastIndex;
    PPERFECT_HASH_TABLE_INDEX SlowIndex;
    PPERFECT_HASH_TABLE_GET_ALGORITHM_NAME GetAlgorithmName;
    PPERFECT_HASH_TABLE_GET_HASH_FUNCTION_NAME GetHashFunctionName;
    PPERFECT_HASH_TABLE_GET_MASK_FUNCTION_NAME GetMaskFunctionName;
    PPERFECT_HASH_TABLE_GET_FILE GetFile;
} PERFECT_HASH_TABLE_VTBL;
typedef PERFECT_HASH_TABLE_VTBL *PPERFECT_HASH_TABLE_VTBL;

//
// Helper functions for obtaining the string representation of enumeration IDs.
//

typedef
_Success_(return != 0)
BOOLEAN
(NTAPI GET_PERFECT_HASH_TABLE_ALGORITHM_NAME)(
    _In_ PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    _Out_ PCUNICODE_STRING *Name
    );
typedef GET_PERFECT_HASH_TABLE_ALGORITHM_NAME
      *PGET_PERFECT_HASH_TABLE_ALGORITHM_NAME;

typedef
_Success_(return != 0)
BOOLEAN
(NTAPI GET_PERFECT_HASH_TABLE_HASH_FUNCTION_NAME)(
    _In_ PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    _Out_ PCUNICODE_STRING *Name
    );
typedef GET_PERFECT_HASH_TABLE_HASH_FUNCTION_NAME
      *PGET_PERFECT_HASH_TABLE_HASH_FUNCTION_NAME;

typedef
_Success_(return != 0)
BOOLEAN
(NTAPI GET_PERFECT_HASH_TABLE_MASK_FUNCTION_NAME)(
    _In_ PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    _Out_ PCUNICODE_STRING *Name
    );
typedef GET_PERFECT_HASH_TABLE_MASK_FUNCTION_NAME
      *PGET_PERFECT_HASH_TABLE_MASK_FUNCTION_NAME;

//
// Scaffolding required to support structured exception handling via __try
// blocks without having to link to the C runtime library.
//

typedef
EXCEPTION_DISPOSITION
(__cdecl __C_SPECIFIC_HANDLER)(
    PEXCEPTION_RECORD ExceptionRecord,
    ULONG_PTR Frame,
    PCONTEXT Context,
    struct _DISPATCHER_CONTEXT *Dispatch
    );
typedef __C_SPECIFIC_HANDLER *P__C_SPECIFIC_HANDLER;

typedef
EXCEPTION_DISPOSITION
(__cdecl RTL_EXCEPTION_HANDLER)(
    PEXCEPTION_RECORD ExceptionRecord,
    ULONG_PTR Frame,
    PCONTEXT Context,
    struct _DISPATCHER_CONTEXT *Dispatch
    );
typedef RTL_EXCEPTION_HANDLER *PRTL_EXCEPTION_HANDLER;

typedef RTL_EXCEPTION_HANDLER __C_SPECIFIC_HANDLER;
typedef __C_SPECIFIC_HANDLER *P__C_SPECIFIC_HANDLER;

typedef
VOID
(SET_C_SPECIFIC_HANDLER)(
    _In_ P__C_SPECIFIC_HANDLER Handler
    );
typedef SET_C_SPECIFIC_HANDLER *PSET_C_SPECIFIC_HANDLER;

typedef
VOID
(__cdecl __SECURITY_INIT_COOKIE)(
    VOID
    );
typedef __SECURITY_INIT_COOKIE *P__SECURITY_INIT_COOKIE;

extern __SECURITY_INIT_COOKIE __security_init_cookie;

//
// Define bootstrap helpers.
//

typedef
_Success_(return >= 0)
_Check_return_opt_
HRESULT
(NTAPI PERFECT_HASH_PRINT_ERROR)(
    _In_ PCSZ FunctionName,
    _In_ PCSZ FileName,
    _In_opt_ ULONG LineNumber,
    _In_opt_ ULONG Error
    );
typedef PERFECT_HASH_PRINT_ERROR *PPERFECT_HASH_PRINT_ERROR;

//
// Define helper macros for printing errors to stdout.  Requires the symbol
// PerfectHashPrintError to be in scope.
//

#define SYS_ERROR(Name) \
    PerfectHashPrintError(#Name, __FILE__, __LINE__, GetLastError())

#define PH_ERROR(Name, Result) \
    PerfectHashPrintError(#Name, __FILE__, __LINE__, (ULONG)Result)

//
// Helper macro for raising non-continuable exceptions.
//

#ifdef _DEBUG
#define PH_RAISE(Result) __debugbreak()
#else
#define PH_RAISE(Result) \
    RaiseException((DWORD)Result, EXCEPTION_NONCONTINUABLE, 0, NULL)
#endif

#ifndef _PERFECT_HASH_INTERNAL_BUILD
FORCEINLINE
_Success_(return >= 0)
HRESULT
PerfectHashBootstrap(
    _Out_ PICLASSFACTORY *ClassFactoryPointer,
    _Out_ PPERFECT_HASH_PRINT_ERROR *PrintErrorPointer,
    _Out_ HMODULE *ModulePointer
    )
/*++

Routine Description:

    This is a simple helper routine that loads the PerfectHash.dll library,
    obtains an IClassFactory interface, and an error handling function pointer.
    It is useful for using the library without needing any underlying system
    COM support (e.g. CoCreateInstance).

Arguments:

    ClassFactoryPointer - Supplies the address of a variable that will receive
        an instance of an ICLASSFACTORY interface if the routine is successful.
        Caller is responsible for calling ClassFactory->Vtbl->Release() when
        finished with the instance.

    PrintErrorPointer - Supplies the address of a variable that will receive the
        function pointer to the PerfectHashPrintError error handling routine.

    ModulePointer - Supplies the address of a variable that will receive the
        handle of the loaded module.  Callers can call FreeLibrary() against
        this handle when finished with the library.

Return Value:

    S_OK - Routine completed successfully.

    The following error codes may also be returned.  This is not an exhaustive
    list.

    E_FAIL - Failed to load PerfectHash.dll.

    E_UNEXPECTED - Failed to resolve symbol from loaded PerfectHash.dll.

--*/
{
    PROC Proc;
    HRESULT Result;
    HMODULE Module;
    PPERFECT_HASH_PRINT_ERROR PerfectHashPrintError;
    PDLL_GET_CLASS_OBJECT PhDllGetClassObject;
    PICLASSFACTORY ClassFactory;

    *ClassFactoryPointer = NULL;
    *PrintErrorPointer = NULL;
    *ModulePointer = NULL;

    Module = LoadLibraryA("PerfectHash.dll");
    if (!Module) {
        return E_FAIL;
    }

    Proc = GetProcAddress(Module, "PerfectHashPrintError");
    if (!Proc) {
        FreeLibrary(Module);
        return E_UNEXPECTED;
    }

    PerfectHashPrintError = (PPERFECT_HASH_PRINT_ERROR)Proc;

    Proc = GetProcAddress(Module, "PerfectHashDllGetClassObject");
    if (!Proc) {
        SYS_ERROR(GetProcAddress);
        FreeLibrary(Module);
        return E_UNEXPECTED;
    }

    PhDllGetClassObject = (PDLL_GET_CLASS_OBJECT)Proc;

    Result = PhDllGetClassObject(&CLSID_PERFECT_HASH,
                                 &IID_PERFECT_HASH_ICLASSFACTORY,
                                 &ClassFactory);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashDllGetClassObject, Result);
        FreeLibrary(Module);
        return Result;
    }

    *ClassFactoryPointer = ClassFactory;
    *PrintErrorPointer = PerfectHashPrintError;
    *ModulePointer = Module;

    return S_OK;
}
#endif

#ifdef __cplusplus
} // extern "C"
#endif

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
