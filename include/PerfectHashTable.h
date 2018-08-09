/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTable.h

Abstract:

    This is the main public header file for the PerfectHashTable component.
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
#include <minwindef.h>
#pragma warning(pop)

#include <sal.h>

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

//
// Define an enumeration for identifying COM interfaces.
//

typedef enum _PERFECT_HASH_TABLE_INTERFACE_ID {

    //
    // Explicitly define a null ID to take the 0-index slot.  This makes enum
    // validation easier.
    //

    PerfectHashTableNullInterfaceId             = 0,

    //
    // Begin valid interfaces.
    //

    PerfectHashTableUnknownInterfaceId          = 1,
    PerfectHashTableClassFactoryInterfaceId     = 2,
    PerfectHashTableKeysInterfaceId             = 3,
    PerfectHashTableContextInterfaceId          = 4,
    PerfectHashTableInterfaceId                 = 5,
    PerfectHashTableRtlInterfaceId              = 6,
    PerfectHashTableAllocatorInterfaceId        = 7,

    //
    // End valid interfaces.
    //

    //
    // N.B. Keep the next value last.
    //

    PerfectHashTableInvalidInterfaceId,

} PERFECT_HASH_TABLE_INTERFACE_ID;

//
// Provide a simple inline interface enum validation routine.
//

FORCEINLINE
BOOLEAN
IsValidPerfectHashTableInterfaceId(
    _In_ PERFECT_HASH_TABLE_INTERFACE_ID InterfaceId
    )
{
    return (
        InterfaceId > PerfectHashTableNullInterfaceId &&
        InterfaceId < PerfectHashTableInvalidInterfaceId
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

DEFINE_GUID_EX(IID_PERFECT_HASH_TABLE_IUNKNOWN, 0x00000000, 0x0000, 0x0000,
               0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46);

//
// IID_ICLASSFACTORY: 00000001-0000-0000-C000-000000000046
//

DEFINE_GUID_EX(IID_PERFECT_HASH_TABLE_ICLASSFACTORY, 0x00000001, 0x0000, 0x0000,
               0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46);

//
// CLSID_PERFECT_HASH_TABLE: 402045FD-72F4-4A05-902E-D22B7C1877B4
//

DEFINE_GUID_EX(CLSID_PERFECT_HASH_TABLE, 0x402045fd, 0x72f4, 0x4a05,
               0x90, 0x2e, 0xd2, 0x2b, 0x7c, 0x18, 0x77, 0xb4);

//
// IID_PERFECT_HASH_TABLE_KEYS: 7E43EBEA-8671-47BA-B844-760B7A9EA921
//

DEFINE_GUID_EX(IID_PERFECT_HASH_TABLE_KEYS, 0x7e43ebea, 0x8671, 0x47ba,
               0xb8, 0x44, 0x76, 0xb, 0x7a, 0x9e, 0xa9, 0x21);

//
// IID_PERFECT_HASH_TABLE_CONTEXT: D4B24571-99D7-44BA-8A27-63D8739F9B81
//

DEFINE_GUID_EX(IID_PERFECT_HASH_TABLE_CONTEXT, 0xd4b24571, 0x99d7, 0x44ba,
               0x8a, 0x27, 0x63, 0xd8, 0x73, 0x9f, 0x9b, 0x81);

//
// IID_PERFECT_HASH_TABLE: C265816F-C6A9-4B44-BCEE-EC5A12ABE1EF
//

DEFINE_GUID_EX(IID_PERFECT_HASH_TABLE, 0xc265816f, 0xc6a9, 0x4b44,
               0xbc, 0xee, 0xec, 0x5a, 0x12, 0xab, 0xe1, 0xef);

//
// IID_PERFECT_HASH_TABLE_RTL: 9C05A3D6-BC30-45E6-BEA6-504FCC9243A8
//

DEFINE_GUID_EX(IID_PERFECT_HASH_TABLE_RTL, 0x9c05a3d6, 0xbc30, 0x45e6,
               0xbe, 0xa6, 0x50, 0x4f, 0xcc, 0x92, 0x43, 0xa8);

//
// IID_PERFECT_HASH_TABLE_ALLOCATOR: F87564D2-B3C7-4CCA-9013-EB59C1E253B7
//

DEFINE_GUID_EX(IID_PERFECT_HASH_TABLE_ALLOCATOR,
               0xf87564d2, 0xb3c7, 0x4cca,
               0x90, 0x13, 0xeb, 0x59, 0xc1, 0xe2, 0x53, 0xb7);

//
// GUID array.
//

static const PCGUID PerfectHashTableInterfaceGuids[] = {

    NULL,

    &IID_PERFECT_HASH_TABLE_IUNKNOWN,
    &IID_PERFECT_HASH_TABLE_ICLASSFACTORY,
    &IID_PERFECT_HASH_TABLE_KEYS,
    &IID_PERFECT_HASH_TABLE_CONTEXT,
    &IID_PERFECT_HASH_TABLE,
    &IID_PERFECT_HASH_TABLE_RTL,
    &IID_PERFECT_HASH_TABLE_ALLOCATOR,

    NULL
};

static const BYTE NumberOfPerfectHashTableInterfaceGuids =
    ARRAYSIZE(PerfectHashTableInterfaceGuids);

//
// Convert a GUID to an interface ID.
//

FORCEINLINE
PERFECT_HASH_TABLE_INTERFACE_ID
PerfectHashTableInterfaceGuidToId(
    _In_ REFIID Guid
    )
{
    BYTE Index;
    BYTE Count;
    PERFECT_HASH_TABLE_INTERFACE_ID Id = PerfectHashTableNullInterfaceId;

    if (!Guid) {
        return PerfectHashTableInvalidInterfaceId;
    }

    //
    // We start the index at 1 in order to skip the first NULL entry.
    //

    Count = NumberOfPerfectHashTableInterfaceGuids;

    for (Index = 1; Index < Count; Index++) {
        if (InlineIsEqualGUID(Guid, PerfectHashTableInterfaceGuids[Index])) {
            Id = (PERFECT_HASH_TABLE_INTERFACE_ID)Index;
            break;
        }
    }

    return Id;
}

//
// IUnknown
//

typedef struct _IUNKNOWN IUNKNOWN;
typedef IUNKNOWN *PIUNKNOWN;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE IUNKNOWN_QUERY_INTERFACE)(
    _In_ PIUNKNOWN Unknown,
    _In_ REFIID InterfaceId,
    _COM_Outptr_ PVOID *Interface
    );
typedef IUNKNOWN_QUERY_INTERFACE *PIUNKNOWN_QUERY_INTERFACE;

typedef
ULONG
(STDAPICALLTYPE IUNKNOWN_ADD_REF)(
    _In_ PIUNKNOWN Unknown
    );
typedef IUNKNOWN_ADD_REF *PIUNKNOWN_ADD_REF;

typedef
ULONG
(STDAPICALLTYPE IUNKNOWN_RELEASE)(
    _In_ PIUNKNOWN Unknown
    );
typedef IUNKNOWN_RELEASE *PIUNKNOWN_RELEASE;

//
// N.B. We abuse the COM spec a bit here in that all of our components,
//      including IUnknown, actually implement IClassFactory.
//

typedef
_Success_(return != 0)
HRESULT
(STDAPICALLTYPE IUNKNOWN_CREATE_INSTANCE)(
    _In_ PIUNKNOWN Unknown,
    _In_opt_ PIUNKNOWN UnknownOuter,
    _In_ REFIID InterfaceId,
    _COM_Outptr_ PVOID *Instance
    );
typedef IUNKNOWN_CREATE_INSTANCE *PIUNKNOWN_CREATE_INSTANCE;

typedef
HRESULT
(STDAPICALLTYPE IUNKNOWN_LOCK_SERVER)(
    _In_ PIUNKNOWN Unknown,
    _In_opt_ BOOL Lock
    );
typedef IUNKNOWN_LOCK_SERVER *PIUNKNOWN_LOCK_SERVER;

typedef struct _IUNKNOWN_VTBL {
    PIUNKNOWN_QUERY_INTERFACE QueryInterface;
    PIUNKNOWN_ADD_REF AddRef;
    PIUNKNOWN_RELEASE Release;
    PIUNKNOWN_CREATE_INSTANCE CreateInstance;
    PIUNKNOWN_LOCK_SERVER LockServer;
} IUNKNOWN_VTBL;
typedef IUNKNOWN_VTBL *PIUNKNOWN_VTBL;

#ifndef _PERFECT_HASH_TABLE_INTERNAL_BUILD
typedef struct _IUNKNOWN {
    PIUNKNOWN_VTBL Vtbl;
} IUNKNOWN;
typedef IUNKNOWN *PIUNKNOWN;
#endif

//
// IClassFactory
//

typedef struct _ICLASSFACTORY ICLASSFACTORY;
typedef ICLASSFACTORY *PICLASSFACTORY;

typedef
_Must_inspect_impl_
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE ICLASSFACTORY_QUERY_INTERFACE)(
    _In_ PICLASSFACTORY ClassFactory,
    _In_ REFIID InterfaceId,
    _COM_Outptr_ PVOID *Interface
    );
typedef ICLASSFACTORY_QUERY_INTERFACE *PICLASSFACTORY_QUERY_INTERFACE;

typedef
ULONG
(STDAPICALLTYPE ICLASSFACTORY_ADD_REF)(
    _In_ PICLASSFACTORY ClassFactory
    );
typedef ICLASSFACTORY_ADD_REF *PICLASSFACTORY_ADD_REF;

typedef
ULONG
(STDAPICALLTYPE ICLASSFACTORY_RELEASE)(
    _In_ PICLASSFACTORY ClassFactory
    );
typedef ICLASSFACTORY_RELEASE *PICLASSFACTORY_RELEASE;

typedef
HRESULT
(STDAPICALLTYPE ICLASSFACTORY_CREATE_INSTANCE)(
    _In_ PICLASSFACTORY ClassFactory,
    _In_opt_ PIUNKNOWN UnknownOuter,
    _In_ REFIID InterfaceId,
    _COM_Outptr_ PVOID *Interface
    );
typedef ICLASSFACTORY_CREATE_INSTANCE *PICLASSFACTORY_CREATE_INSTANCE;

typedef
HRESULT
(STDAPICALLTYPE ICLASSFACTORY_LOCK_SERVER)(
    _In_ PICLASSFACTORY ClassFactory,
    _In_opt_ BOOL Lock
    );
typedef ICLASSFACTORY_LOCK_SERVER *PICLASSFACTORY_LOCK_SERVER;

typedef struct _ICLASSFACTORY_VTBL {
    PICLASSFACTORY_QUERY_INTERFACE QueryInterface;
    PICLASSFACTORY_ADD_REF AddRef;
    PICLASSFACTORY_RELEASE Release;
    PICLASSFACTORY_CREATE_INSTANCE CreateInstance;
    PICLASSFACTORY_LOCK_SERVER LockServer;
} ICLASSFACTORY_VTBL;
typedef ICLASSFACTORY_VTBL *PICLASSFACTORY_VTBL;

#ifndef _PERFECT_HASH_TABLE_INTERNAL_BUILD
typedef struct _ICLASSFACTORY {
    PICLASSFACTORY_VTBL Vtbl;
} ICLASSFACTORY;
typedef ICLASSFACTORY *PICLASSFACTORY;
#endif

//
// Define the ALLOCATOR interface.
//

typedef struct _ALLOCATOR ALLOCATOR;
typedef ALLOCATOR *PALLOCATOR;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE ALLOCATOR_QUERY_INTERFACE)(
    _In_ PALLOCATOR Allocator,
    _In_ REFIID InterfaceId,
    _COM_Outptr_ PVOID *Interface
    );
typedef ALLOCATOR_QUERY_INTERFACE *PALLOCATOR_QUERY_INTERFACE;

typedef
ULONG
(STDAPICALLTYPE ALLOCATOR_ADD_REF)(
    _In_ PALLOCATOR Allocator
    );
typedef ALLOCATOR_ADD_REF *PALLOCATOR_ADD_REF;

typedef
ULONG
(STDAPICALLTYPE ALLOCATOR_RELEASE)(
    _In_ PALLOCATOR Allocator
    );
typedef ALLOCATOR_RELEASE *PALLOCATOR_RELEASE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE ALLOCATOR_CREATE_INSTANCE)(
    _In_ PALLOCATOR Allocator,
    _In_opt_ PIUNKNOWN UnknownOuter,
    _In_ REFIID InterfaceId,
    _COM_Outptr_ PVOID *Instance
    );
typedef ALLOCATOR_CREATE_INSTANCE *PALLOCATOR_CREATE_INSTANCE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE ALLOCATOR_LOCK_SERVER)(
    _In_ PALLOCATOR Allocator,
    _In_opt_ BOOL Lock
    );
typedef ALLOCATOR_LOCK_SERVER *PALLOCATOR_LOCK_SERVER;

typedef
_Check_return_
_Ret_maybenull_
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
_Post_writable_byte_size_(NumberOfElements * ElementSize)
PVOID
(STDAPICALLTYPE ALLOCATOR_CALLOC)(
    _In_ PALLOCATOR Allocator,
    _In_ SIZE_T NumberOfElements,
    _In_ SIZE_T ElementSize
    );
typedef ALLOCATOR_CALLOC *PALLOCATOR_CALLOC;

typedef
VOID
(STDAPICALLTYPE ALLOCATOR_FREE)(
    _In_ PALLOCATOR Allocator,
    _Pre_maybenull_ _Post_invalid_ PVOID Address
    );
typedef ALLOCATOR_FREE *PALLOCATOR_FREE;

typedef
VOID
(STDAPICALLTYPE ALLOCATOR_FREE_POINTER)(
    _In_ PALLOCATOR Allocator,
    _Inout_ PVOID *AddressPointer
    );
typedef ALLOCATOR_FREE_POINTER *PALLOCATOR_FREE_POINTER;

typedef struct _ALLOCATOR_VTBL {
    PALLOCATOR_QUERY_INTERFACE QueryInterface;
    PALLOCATOR_ADD_REF AddRef;
    PALLOCATOR_RELEASE Release;
    PALLOCATOR_CREATE_INSTANCE CreateInstance;
    PALLOCATOR_LOCK_SERVER LockServer;
    PALLOCATOR_MALLOC Malloc;
    PALLOCATOR_CALLOC Calloc;
    PALLOCATOR_FREE Free;
    PALLOCATOR_FREE_POINTER FreePointer;
} ALLOCATOR_VTBL;
typedef ALLOCATOR_VTBL *PALLOCATOR_VTBL;

#ifndef _PERFECT_HASH_TABLE_INTERNAL_BUILD
typedef struct _ALLOCATOR {
    PALLOCATOR_VTBL Vtbl;
} ALLOCATOR;
typedef ALLOCATOR *PALLOCATOR;
#endif

//
// Define the PERFECT_HASH_TABLE_KEYS interface.
//

typedef struct _PERFECT_HASH_TABLE_KEYS PERFECT_HASH_TABLE_KEYS;
typedef PERFECT_HASH_TABLE_KEYS *PPERFECT_HASH_TABLE_KEYS;

typedef
_Must_inspect_impl_
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_KEYS_QUERY_INTERFACE)(
    _In_ PPERFECT_HASH_TABLE_KEYS Keys,
    _In_ REFIID InterfaceId,
    _COM_Outptr_ PVOID *Interface
    );
typedef PERFECT_HASH_TABLE_KEYS_QUERY_INTERFACE
      *PPERFECT_HASH_TABLE_KEYS_QUERY_INTERFACE;

typedef
ULONG
(STDAPICALLTYPE PERFECT_HASH_TABLE_KEYS_ADD_REF)(
    _In_ PPERFECT_HASH_TABLE_KEYS Keys
    );
typedef PERFECT_HASH_TABLE_KEYS_ADD_REF *PPERFECT_HASH_TABLE_KEYS_ADD_REF;

typedef
ULONG
(STDAPICALLTYPE PERFECT_HASH_TABLE_KEYS_RELEASE)(
    _In_ PPERFECT_HASH_TABLE_KEYS Keys
    );
typedef PERFECT_HASH_TABLE_KEYS_RELEASE *PPERFECT_HASH_TABLE_KEYS_RELEASE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_KEYS_CREATE_INSTANCE)(
    _In_ PPERFECT_HASH_TABLE_KEYS Keys,
    _In_opt_ PIUNKNOWN UnknownOuter,
    _In_ REFIID InterfaceId,
    _COM_Outptr_ PVOID *Interface
    );
typedef PERFECT_HASH_TABLE_KEYS_CREATE_INSTANCE
      *PPERFECT_HASH_TABLE_KEYS_CREATE_INSTANCE;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_KEYS_LOCK_SERVER)(
    _In_ PPERFECT_HASH_TABLE_KEYS Keys,
    _In_opt_ BOOL Lock
    );
typedef PERFECT_HASH_TABLE_KEYS_LOCK_SERVER
      *PPERFECT_HASH_TABLE_KEYS_LOCK_SERVER;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_KEYS_LOAD)(
    _In_ PPERFECT_HASH_TABLE_KEYS Keys,
    _In_ PCUNICODE_STRING Path
    );
typedef PERFECT_HASH_TABLE_KEYS_LOAD
      *PPERFECT_HASH_TABLE_KEYS_LOAD;

typedef struct _PERFECT_HASH_TABLE_KEYS_VTBL {
    PPERFECT_HASH_TABLE_KEYS_QUERY_INTERFACE QueryInterface;
    PPERFECT_HASH_TABLE_KEYS_ADD_REF AddRef;
    PPERFECT_HASH_TABLE_KEYS_RELEASE Release;
    PPERFECT_HASH_TABLE_KEYS_CREATE_INSTANCE CreateInstance;
    PPERFECT_HASH_TABLE_KEYS_LOCK_SERVER LockServer;
    PPERFECT_HASH_TABLE_KEYS_LOAD Load;
} PERFECT_HASH_TABLE_KEYS_VTBL;
typedef PERFECT_HASH_TABLE_KEYS_VTBL *PPERFECT_HASH_TABLE_KEYS_VTBL;

#ifndef _PERFECT_HASH_TABLE_INTERNAL_BUILD
typedef struct _PERFECT_HASH_TABLE_KEYS {
    PPERFECT_HASH_TABLE_KEYS_VTBL Vtbl;
} PERFECT_HASH_TABLE_KEYS;
typedef PERFECT_HASH_TABLE_KEYS *PPERFECT_HASH_TABLE_KEYS;
#endif

//
// Define an enumeration for identifying which backend algorithm variant to
// use for creating the perfect hash table.
//

typedef enum _PERFECT_HASH_TABLE_ALGORITHM_ID {

    //
    // Explicitly define a null algorithm to take the 0-index slot.
    // This makes enum validation easier.
    //

    PerfectHashTableNullAlgorithmId         = 0,

    //
    // Begin valid algorithms.
    //

    PerfectHashTableChm01AlgorithmId        = 1,
    PerfectHashTableDefaultAlgorithmId      = 1,

    //
    // End valid algorithms.
    //

    //
    // N.B. Keep the next value last.
    //

    PerfectHashTableInvalidAlgorithmId,

} PERFECT_HASH_TABLE_ALGORITHM_ID;
typedef PERFECT_HASH_TABLE_ALGORITHM_ID *PPERFECT_HASH_TABLE_ALGORITHM_ID;

//
// Provide a simple inline algorithm validation routine.
//

FORCEINLINE
BOOLEAN
IsValidPerfectHashTableAlgorithmId(
    _In_ PERFECT_HASH_TABLE_ALGORITHM_ID AlgorithmId
    )
{
    return (
        AlgorithmId > PerfectHashTableNullAlgorithmId &&
        AlgorithmId < PerfectHashTableInvalidAlgorithmId
    );
}

//
// Define an enumeration for identifying which hash function variant to use.
//

typedef enum _PERFECT_HASH_TABLE_HASH_FUNCTION_ID {

    //
    // Explicitly define a null algorithm to take the 0-index slot.
    // This makes enum validation easier.
    //

    PerfectHashTableNullHashFunctionId              = 0,

    //
    // Begin valid hash functions.
    //

    PerfectHashTableHashCrc32RotateFunctionId       = 1,
    PerfectHashTableDefaultHashFunctionId           = 1,

    PerfectHashTableHashJenkinsFunctionId           = 2,

    //
    // N.B. The following three hash functions are purposefully terrible from
    //      the perspective of generating a good distribution of hash values.
    //      They all have very simple operations and are intended to test the
    //      theory that even with a poor hash function, once we find the right
    //      seed, the hash quality is unimportant.
    //

    PerfectHashTableHashRotateXorFunctionId         = 3,
    PerfectHashTableHashAddSubXorFunctionId         = 4,
    PerfectHashTableHashXorFunctionId               = 5,

    //
    // End valid hash functions.
    //

    //
    // N.B. Keep the next value last.
    //

    PerfectHashTableInvalidHashFunctionId,

} PERFECT_HASH_TABLE_HASH_FUNCTION_ID;
typedef PERFECT_HASH_TABLE_HASH_FUNCTION_ID
      *PPERFECT_HASH_TABLE_HASH_FUNCTION_ID;

//
// Provide a simple inline hash function validation routine.
//

FORCEINLINE
BOOLEAN
IsValidPerfectHashTableHashFunctionId(
    _In_ PERFECT_HASH_TABLE_HASH_FUNCTION_ID HashFunctionId
    )
{
    return (
        HashFunctionId > PerfectHashTableNullHashFunctionId &&
        HashFunctionId < PerfectHashTableInvalidHashFunctionId
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

typedef enum _PERFECT_HASH_TABLE_MASK_FUNCTION_ID {

    //
    // Null masking type.
    //

    PerfectHashTableNullMaskFunctionId          = 0,

    //
    // Being valid masking types.
    //

    PerfectHashTableModulusMaskFunctionId       = 1,

    PerfectHashTableAndMaskFunctionId           = 2,
    PerfectHashTableDefaultMaskFunctionId       = 2,

    PerfectHashTableXorAndMaskFunctionId        = 3,
    PerfectHashTableFoldAutoMaskFunctionId      = 4,
    PerfectHashTableFoldOnceMaskFunctionId      = 5,
    PerfectHashTableFoldTwiceMaskFunctionId     = 6,
    PerfectHashTableFoldThriceMaskFunctionId    = 7,

    //
    // End valid masking types.
    //

    //
    // N.B. Keep the next value last.
    //

    PerfectHashTableInvalidMaskFunctionId,


} PERFECT_HASH_TABLE_MASK_FUNCTION_ID;
typedef PERFECT_HASH_TABLE_MASK_FUNCTION_ID
      *PPERFECT_HASH_TABLE_MASK_FUNCTION_ID;

//
// Provide a simple inline masking type validation routine.
//

FORCEINLINE
BOOLEAN
IsValidPerfectHashTableMaskFunctionId(
    _In_ PERFECT_HASH_TABLE_MASK_FUNCTION_ID MaskFunctionId
    )
{
    return (
        MaskFunctionId > PerfectHashTableNullMaskFunctionId &&
        MaskFunctionId < PerfectHashTableInvalidMaskFunctionId
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
    _In_ PERFECT_HASH_TABLE_MASK_FUNCTION_ID MaskFunctionId
    )
{
    return MaskFunctionId == PerfectHashTableModulusMaskFunctionId;
}

//
// Define an enumeration for identifying benchmark routines.
//

typedef enum _PERFECT_HASH_TABLE_BENCHMARK_FUNCTION_ID {

    //
    // Explicitly define a null algorithm to take the 0-index slot.
    // This makes enum validation easier.
    //

    PerfectHashTableNullBenchmarkFunctionId         = 0,

    //
    // Begin valid benchmarks.
    //

    PerfectHashTableFastIndexBenchmarkFunctionId    = 1,

    //
    // End valid benchmarks.
    //

    //
    // N.B. Keep the next value last.
    //

    PerfectHashTableInvalidBenchmarkFunctionId,

} PERFECT_HASH_TABLE_BENCHMARK_FUNCTION_ID;

//
// Provide a simple inline benchmark validation routine.
//

FORCEINLINE
BOOLEAN
IsValidPerfectHashTableBenchmarkFunctionId(
    _In_ PERFECT_HASH_TABLE_BENCHMARK_FUNCTION_ID BenchmarkFunctionId
    )
{
    return (
        BenchmarkFunctionId > PerfectHashTableNullBenchmarkFunctionId &&
        BenchmarkFunctionId < PerfectHashTableInvalidBenchmarkFunctionId
    );
}

//
// Define an enumeration for identifying benchmark types.
//

typedef enum _PERFECT_HASH_TABLE_BENCHMARK_TYPE {

    //
    // Explicitly define a null benchmark type to take the 0-index slot.
    // This makes enum validation easier.
    //

    PerfectHashTableNullBenchmarkType       = 0,

    //
    // Begin valid benchmark typess.
    //

    PerfectHashTableSingleBenchmarkType     = 1,
    PerfectHashTableAllBenchmarkType        = 2,

    //
    // End valid benchmark typess.
    //

    //
    // N.B. Keep the next value last.
    //

    PerfectHashTableInvalidBenchmarkType,

} PERFECT_HASH_TABLE_BENCHMARK_TYPE;
typedef PERFECT_HASH_TABLE_BENCHMARK_TYPE *PPERFECT_HASH_TABLE_BENCHMARK_TYPE;

//
// Provide a simple inline benchmark type validation routine.
//

FORCEINLINE
BOOLEAN
IsValidPerfectHashTableBenchmarkType(
    _In_ PERFECT_HASH_TABLE_BENCHMARK_TYPE BenchmarkType
    )
{
    return (
        BenchmarkType > PerfectHashTableNullBenchmarkType &&
        BenchmarkType < PerfectHashTableInvalidBenchmarkType
    );
}

//
// Define the PERFECT_HASH_TABLE_CONTEXT interface.  This interface is
// responsible for encapsulating threadpool resources and allows perfect hash
// table solutions to be found in parallel.  An instance of this interface must
// be provided to the PERFECT_HASH_TABLE interface's creation routine.
//

typedef struct _PERFECT_HASH_TABLE_CONTEXT PERFECT_HASH_TABLE_CONTEXT;
typedef PERFECT_HASH_TABLE_CONTEXT *PPERFECT_HASH_TABLE_CONTEXT;

typedef
_Must_inspect_impl_
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_CONTEXT_QUERY_INTERFACE)(
    _In_ PPERFECT_HASH_TABLE_CONTEXT Context,
    _In_ REFIID InterfaceId,
    _COM_Outptr_ PVOID *Interface
    );
typedef PERFECT_HASH_TABLE_CONTEXT_QUERY_INTERFACE
      *PPERFECT_HASH_TABLE_CONTEXT_QUERY_INTERFACE;

typedef
ULONG
(STDAPICALLTYPE PERFECT_HASH_TABLE_CONTEXT_ADD_REF)(
    _In_ PPERFECT_HASH_TABLE_CONTEXT Context
    );
typedef PERFECT_HASH_TABLE_CONTEXT_ADD_REF *PPERFECT_HASH_TABLE_CONTEXT_ADD_REF;

typedef
ULONG
(STDAPICALLTYPE PERFECT_HASH_TABLE_CONTEXT_RELEASE)(
    _In_ PPERFECT_HASH_TABLE_CONTEXT Context
    );
typedef PERFECT_HASH_TABLE_CONTEXT_RELEASE *PPERFECT_HASH_TABLE_CONTEXT_RELEASE;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_CONTEXT_CREATE_INSTANCE)(
    _In_ PPERFECT_HASH_TABLE_CONTEXT Context,
    _In_opt_ PIUNKNOWN UnknownOuter,
    _In_ REFIID InterfaceId,
    _COM_Outptr_ PVOID *Interface
    );
typedef PERFECT_HASH_TABLE_CONTEXT_CREATE_INSTANCE
      *PPERFECT_HASH_TABLE_CONTEXT_CREATE_INSTANCE;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_CONTEXT_LOCK_SERVER)(
    _In_ PPERFECT_HASH_TABLE_CONTEXT Context,
    _In_opt_ BOOL Lock
    );
typedef PERFECT_HASH_TABLE_CONTEXT_LOCK_SERVER
      *PPERFECT_HASH_TABLE_CONTEXT_LOCK_SERVER;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_CONTEXT_SET_MAXIMUM_CONCURRENCY)(
    _In_ PPERFECT_HASH_TABLE_CONTEXT Context,
    _In_ ULONG MaximumConcurrency
    );
typedef PERFECT_HASH_TABLE_CONTEXT_SET_MAXIMUM_CONCURRENCY
      *PPERFECT_HASH_TABLE_CONTEXT_SET_MAXIMUM_CONCURRENCY;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_CONTEXT_GET_MAXIMUM_CONCURRENCY)(
    _In_ PPERFECT_HASH_TABLE_CONTEXT Context,
    _Out_ PULONG MaximumConcurrency
    );
typedef PERFECT_HASH_TABLE_CONTEXT_GET_MAXIMUM_CONCURRENCY
      *PPERFECT_HASH_TABLE_CONTEXT_GET_MAXIMUM_CONCURRENCY;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_CONTEXT_CREATE_TABLE)(
    _In_ PPERFECT_HASH_TABLE_CONTEXT Context,
    _In_ PERFECT_HASH_TABLE_ALGORITHM_ID AlgorithmId,
    _In_ PERFECT_HASH_TABLE_MASK_FUNCTION_ID MaskFunctionId,
    _In_ PERFECT_HASH_TABLE_HASH_FUNCTION_ID HashFunctionId,
    _In_ PPERFECT_HASH_TABLE_KEYS Keys,
    _Inout_opt_ PCUNICODE_STRING HashTablePath
    );
typedef PERFECT_HASH_TABLE_CONTEXT_CREATE_TABLE
      *PPERFECT_HASH_TABLE_CONTEXT_CREATE_TABLE;


typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_CONTEXT_SELF_TEST)(
    _In_ PPERFECT_HASH_TABLE_CONTEXT Context,
    _In_ PCUNICODE_STRING TestDataDirectory,
    _In_ PCUNICODE_STRING OutputDirectory,
    _In_ PERFECT_HASH_TABLE_ALGORITHM_ID AlgorithmId,
    _In_ PERFECT_HASH_TABLE_HASH_FUNCTION_ID HashFunctionId,
    _In_ PERFECT_HASH_TABLE_MASK_FUNCTION_ID MaskFunctionId
    );
typedef PERFECT_HASH_TABLE_CONTEXT_SELF_TEST *PPERFECT_HASH_TABLE_CONTEXT_SELF_TEST;

typedef struct _PERFECT_HASH_TABLE_CONTEXT_VTBL {
    PPERFECT_HASH_TABLE_CONTEXT_QUERY_INTERFACE QueryInterface;
    PPERFECT_HASH_TABLE_CONTEXT_ADD_REF AddRef;
    PPERFECT_HASH_TABLE_CONTEXT_RELEASE Release;
    PPERFECT_HASH_TABLE_CONTEXT_CREATE_INSTANCE CreateInstance;
    PPERFECT_HASH_TABLE_CONTEXT_LOCK_SERVER LockServer;
    PPERFECT_HASH_TABLE_CONTEXT_SET_MAXIMUM_CONCURRENCY SetMaximumConcurrency;
    PPERFECT_HASH_TABLE_CONTEXT_GET_MAXIMUM_CONCURRENCY GetMaximumConcurrency;
    PPERFECT_HASH_TABLE_CONTEXT_CREATE_TABLE CreateTable;
    PPERFECT_HASH_TABLE_CONTEXT_SELF_TEST SelfTest;
} PERFECT_HASH_TABLE_CONTEXT_VTBL;
typedef PERFECT_HASH_TABLE_CONTEXT_VTBL *PPERFECT_HASH_TABLE_CONTEXT_VTBL;

#ifndef _PERFECT_HASH_TABLE_INTERNAL_BUILD
typedef struct _PERFECT_HASH_TABLE_CONTEXT {
    PPERFECT_HASH_TABLE_CONTEXT_VTBL Vtbl;
} PERFECT_HASH_TABLE_CONTEXT;
typedef PERFECT_HASH_TABLE_CONTEXT *PPERFECT_HASH_TABLE_CONTEXT;
#endif

//
// Define the PERFECT_HASH_TABLE interface.
//

typedef struct _PERFECT_HASH_TABLE PERFECT_HASH_TABLE;
typedef PERFECT_HASH_TABLE *PPERFECT_HASH_TABLE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_QUERY_INTERFACE)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ REFIID InterfaceId,
    _COM_Outptr_ PVOID *Interface
    );
typedef PERFECT_HASH_TABLE_QUERY_INTERFACE *PPERFECT_HASH_TABLE_QUERY_INTERFACE;

typedef
ULONG
(STDAPICALLTYPE PERFECT_HASH_TABLE_ADD_REF)(
    _In_ PPERFECT_HASH_TABLE Table
    );
typedef PERFECT_HASH_TABLE_ADD_REF *PPERFECT_HASH_TABLE_ADD_REF;

typedef
ULONG
(STDAPICALLTYPE PERFECT_HASH_TABLE_RELEASE)(
    _In_ PPERFECT_HASH_TABLE Table
    );
typedef PERFECT_HASH_TABLE_RELEASE *PPERFECT_HASH_TABLE_RELEASE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_CREATE_INSTANCE)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_opt_ PIUNKNOWN UnknownOuter,
    _In_ REFIID InterfaceId,
    _COM_Outptr_ PVOID *Instance
    );
typedef PERFECT_HASH_TABLE_CREATE_INSTANCE *PPERFECT_HASH_TABLE_CREATE_INSTANCE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_LOCK_SERVER)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_opt_ BOOL Lock
    );
typedef PERFECT_HASH_TABLE_LOCK_SERVER *PPERFECT_HASH_TABLE_LOCK_SERVER;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_LOAD)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ PCUNICODE_STRING Path,
    _In_opt_ PPERFECT_HASH_TABLE_KEYS Keys
    );
typedef PERFECT_HASH_TABLE_LOAD *PPERFECT_HASH_TABLE_LOAD;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_TEST)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_opt_ PPERFECT_HASH_TABLE_KEYS Keys,
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

//
// Given a key, this routine returns the relative index of the key in the
// underlying hash table.  This is guaranteed to be within the bounds of the
// table size.
//

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

typedef struct _PERFECT_HASH_TABLE_VTBL {
    PPERFECT_HASH_TABLE_QUERY_INTERFACE QueryInterface;
    PPERFECT_HASH_TABLE_ADD_REF AddRef;
    PPERFECT_HASH_TABLE_RELEASE Release;
    PPERFECT_HASH_TABLE_CREATE_INSTANCE CreateInstance;
    PPERFECT_HASH_TABLE_LOCK_SERVER LockServer;
    PPERFECT_HASH_TABLE_LOAD Load;
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
} PERFECT_HASH_TABLE_VTBL;
typedef PERFECT_HASH_TABLE_VTBL *PPERFECT_HASH_TABLE_VTBL;

//
// Helper functions for obtaining the string representation of enumeration IDs.
//

typedef
_Success_(return != 0)
BOOLEAN
(NTAPI GET_PERFECT_HASH_TABLE_ALGORITHM_NAME)(
    _In_ PERFECT_HASH_TABLE_ALGORITHM_ID AlgorithmId,
    _Out_ PCUNICODE_STRING *Name
    );
typedef GET_PERFECT_HASH_TABLE_ALGORITHM_NAME
      *PGET_PERFECT_HASH_TABLE_ALGORITHM_NAME;

typedef
_Success_(return != 0)
BOOLEAN
(NTAPI GET_PERFECT_HASH_TABLE_HASH_FUNCTION_NAME)(
    _In_ PERFECT_HASH_TABLE_HASH_FUNCTION_ID HashFunctionId,
    _Out_ PCUNICODE_STRING *Name
    );
typedef GET_PERFECT_HASH_TABLE_HASH_FUNCTION_NAME
      *PGET_PERFECT_HASH_TABLE_HASH_FUNCTION_NAME;

typedef
_Success_(return != 0)
BOOLEAN
(NTAPI GET_PERFECT_HASH_TABLE_MASK_FUNCTION_NAME)(
    _In_ PERFECT_HASH_TABLE_MASK_FUNCTION_ID MaskFunctionId,
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

#ifdef __cplusplus
} // extern "C"
#endif

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
