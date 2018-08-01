/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTable.h

Abstract:

    This is the main public header file for the PerfectHashTable component.

--*/

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "../Rtl/__C_specific_handler.h"
#include "../Rtl/Rtl.h"

//
// Define an opaque PERFECT_HASH_TABLE_KEYS structure.
//

typedef struct _PERFECT_HASH_TABLE_KEYS PERFECT_HASH_TABLE_KEYS;
typedef PERFECT_HASH_TABLE_KEYS *PPERFECT_HASH_TABLE_KEYS;
typedef const PERFECT_HASH_TABLE_KEYS *PCPERFECT_HASH_TABLE_KEYS;

//
// Define the PERFECT_HASH_TABLE_KEYS interface function pointers.
//

typedef
_Check_return_
_Success_(return != 0)
BOOLEAN
(NTAPI LOAD_PERFECT_HASH_TABLE_KEYS)(
    _In_ PRTL Rtl,
    _In_ PALLOCATOR Allocator,
    _In_ PCUNICODE_STRING Path,
    _Outptr_result_nullonfailure_ PPERFECT_HASH_TABLE_KEYS *Keys
    );
typedef LOAD_PERFECT_HASH_TABLE_KEYS *PLOAD_PERFECT_HASH_TABLE_KEYS;

typedef
_Check_return_
_Success_(return != 0)
BOOLEAN
(NTAPI DESTROY_PERFECT_HASH_TABLE_KEYS)(
    _Inout_ PPERFECT_HASH_TABLE_KEYS *Keys
    );
typedef DESTROY_PERFECT_HASH_TABLE_KEYS *PDESTROY_PERFECT_HASH_TABLE_KEYS;

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
// Define an opaque runtime context to encapsulate threadpool resources.  This
// is created via CreatePerfectHashTableContext() with a desired concurrency,
// and then passed to CreatePerfectHashTable(), allowing it to search for
// perfect hash solutions in parallel.
//

typedef struct _PERFECT_HASH_TABLE_CONTEXT PERFECT_HASH_TABLE_CONTEXT;
typedef PERFECT_HASH_TABLE_CONTEXT *PPERFECT_HASH_TABLE_CONTEXT;

//
// Define the create and destroy functions for the runtime context.
//

typedef
_Check_return_
_Success_(return != 0)
BOOLEAN
(NTAPI CREATE_PERFECT_HASH_TABLE_CONTEXT)(
    _In_ PRTL Rtl,
    _In_ PALLOCATOR Allocator,
    _In_opt_ PULONG MaximumConcurrency,
    _Outptr_opt_result_nullonfailure_ PPERFECT_HASH_TABLE_CONTEXT *Context
    );
typedef CREATE_PERFECT_HASH_TABLE_CONTEXT *PCREATE_PERFECT_HASH_TABLE_CONTEXT;

typedef
_Check_return_
_Success_(return != 0)
BOOLEAN
(NTAPI DESTROY_PERFECT_HASH_TABLE_CONTEXT)(
    _Pre_notnull_ _Post_satisfies_(*ContextPointer == 0)
        PPERFECT_HASH_TABLE_CONTEXT *ContextPointer,
    _In_opt_ PBOOLEAN IsProcessTerminating
    );
typedef DESTROY_PERFECT_HASH_TABLE_CONTEXT *PDESTROY_PERFECT_HASH_TABLE_CONTEXT;

//
// Perfect hash tables are created via the CreatePerfectHashTable() routine,
// the signature for which is defined below.  This routine returns a TRUE value
// if a perfect hash table was successfully created from the given parameters.
// If creation was successful, an on-disk representation of the table will be
// saved at the given hash table path.
//
// N.B. Perfect hash tables are used via the PERFECT_HASH_TABLE_VTBL interface,
//      which is obtained from the Vtbl field of the PERFECT_HASH_TABLE struct
//      returned by LoadPerfectHashTableInstance().
//

typedef
_Check_return_
_Success_(return != 0)
BOOLEAN
(NTAPI CREATE_PERFECT_HASH_TABLE)(
    _In_ PRTL Rtl,
    _In_ PALLOCATOR Allocator,
    _In_ PPERFECT_HASH_TABLE_CONTEXT Context,
    _In_ PERFECT_HASH_TABLE_ALGORITHM_ID AlgorithmId,
    _In_ PERFECT_HASH_TABLE_MASK_FUNCTION_ID MaskFunctionId,
    _In_ PERFECT_HASH_TABLE_HASH_FUNCTION_ID HashFunctionId,
    _Inout_opt_ PULARGE_INTEGER NumberOfTableElements,
    _In_ PPERFECT_HASH_TABLE_KEYS Keys,
    _Inout_opt_ PCUNICODE_STRING HashTablePath
    );
typedef CREATE_PERFECT_HASH_TABLE *PCREATE_PERFECT_HASH_TABLE;

//
// Forward definition of the interface.
//

typedef struct _PERFECT_HASH_TABLE_VTBL PERFECT_HASH_TABLE_VTBL;
typedef PERFECT_HASH_TABLE_VTBL *PPERFECT_HASH_TABLE_VTBL;
typedef PERFECT_HASH_TABLE_VTBL **PPPERFECT_HASH_TABLE;

//
// Define a minimal vtbl encapsulation structure if we're a public
// (i.e. non-internal) build.  The actual structure is defined in
// PerfectHashTablePrivate.h.
//

#ifndef _PERFECT_HASH_TABLE_INTERNAL_BUILD
typedef struct _PERFECT_HASH_TABLE {
    PPERFECT_HASH_TABLE_VTBL Vtbl;
} PERFECT_HASH_TABLE;
#else
typedef struct _PERFECT_HASH_TABLE PERFECT_HASH_TABLE;
#endif
typedef PERFECT_HASH_TABLE *PPERFECT_HASH_TABLE;

typedef
_Check_return_
_Success_(return != 0)
BOOLEAN
(NTAPI LOAD_PERFECT_HASH_TABLE)(
    _In_ PRTL Rtl,
    _In_ PALLOCATOR Allocator,
    _In_opt_ PPERFECT_HASH_TABLE_KEYS Keys,
    _In_ PCUNICODE_STRING Path,
    _Out_ PPERFECT_HASH_TABLE *TablePointer
    );
typedef LOAD_PERFECT_HASH_TABLE *PLOAD_PERFECT_HASH_TABLE;

//
// Define the public perfect hash table functions.
//

typedef
HRESULT
(NTAPI PERFECT_HASH_TABLE_INSERT)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ ULONG Key,
    _In_ ULONG Value,
    _Out_opt_ PULONG PreviousValue
    );
typedef PERFECT_HASH_TABLE_INSERT *PPERFECT_HASH_TABLE_INSERT;

typedef
HRESULT
(NTAPI PERFECT_HASH_TABLE_LOOKUP)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ ULONG Key,
    _Out_ PULONG Value
    );
typedef PERFECT_HASH_TABLE_LOOKUP *PPERFECT_HASH_TABLE_LOOKUP;

typedef
HRESULT
(NTAPI PERFECT_HASH_TABLE_DELETE)(
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
(NTAPI PERFECT_HASH_TABLE_INDEX)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ ULONG Key,
    _In_ PULONG Index
    );
typedef PERFECT_HASH_TABLE_INDEX *PPERFECT_HASH_TABLE_INDEX;

typedef
HRESULT
(NTAPI PERFECT_HASH_TABLE_HASH)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ ULONG Input,
    _Out_ PULONGLONG Hash
    );
typedef PERFECT_HASH_TABLE_HASH *PPERFECT_HASH_TABLE_HASH;

typedef
HRESULT
(NTAPI PERFECT_HASH_TABLE_MASK_HASH)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ ULONG Input,
    _Out_ PULONG Masked
    );
typedef PERFECT_HASH_TABLE_MASK_HASH *PPERFECT_HASH_TABLE_MASK_HASH;

typedef
HRESULT
(NTAPI PERFECT_HASH_TABLE_MASK_INDEX)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ ULONGLONG Input,
    _Out_ PULONG Masked
    );
typedef PERFECT_HASH_TABLE_MASK_INDEX *PPERFECT_HASH_TABLE_MASK_INDEX;

//
// Loaded hash tables are reference counted using the AddRef()/Release() COM
// semantics.  The number of AddRef() calls should match the number of Release()
// calls.  The resources will be released when the final Release() is called.
//

typedef
ULONG
(NTAPI PERFECT_HASH_TABLE_ADD_REF)(
    _In_ PPERFECT_HASH_TABLE Table
    );
typedef PERFECT_HASH_TABLE_ADD_REF *PPERFECT_HASH_TABLE_ADD_REF;

typedef
ULONG
(NTAPI PERFECT_HASH_TABLE_RELEASE)(
    _In_ PPERFECT_HASH_TABLE Table
    );
typedef PERFECT_HASH_TABLE_RELEASE *PPERFECT_HASH_TABLE_RELEASE;

//
// The interface as a vtbl.  Note that we're *almost* a valid COM interface,
// except for the NULL pointer that will occupy the first slot where the impl
// for QueryInterface() is meant to live.
//

typedef struct _PERFECT_HASH_TABLE_VTBL {
    PVOID Unused;
    PPERFECT_HASH_TABLE_ADD_REF AddRef;
    PPERFECT_HASH_TABLE_RELEASE Release;
    PPERFECT_HASH_TABLE_INSERT Insert;
    PPERFECT_HASH_TABLE_LOOKUP Lookup;
    PPERFECT_HASH_TABLE_DELETE Delete;
    PPERFECT_HASH_TABLE_INDEX Index;
    PPERFECT_HASH_TABLE_HASH Hash;
    PPERFECT_HASH_TABLE_MASK_HASH MaskHash;
    PPERFECT_HASH_TABLE_MASK_INDEX MaskIndex;
} PERFECT_HASH_TABLE_VTBL;
typedef PERFECT_HASH_TABLE_VTBL *PPERFECT_HASH_TABLE_VTBL;

//
// Allocator-specific typedefs.
//

typedef
_Check_return_
_Success_(return != 0)
BOOL
(NTAPI INITIALIZE_PERFECT_HASH_TABLE_ALLOCATOR_FROM_RTL_BOOTSTRAP)(
    _In_ PRTL_BOOTSTRAP RtlBootstrap,
    _In_ PALLOCATOR Allocator
    );
typedef INITIALIZE_PERFECT_HASH_TABLE_ALLOCATOR_FROM_RTL_BOOTSTRAP
      *PINITIALIZE_PERFECT_HASH_TABLE_ALLOCATOR_FROM_RTL_BOOTSTRAP;

typedef
_Check_return_
_Success_(return != 0)
BOOL
(NTAPI INITIALIZE_PERFECT_HASH_TABLE_ALLOCATOR)(
    _In_ PRTL Rtl,
    _In_ PALLOCATOR Allocator
    );
typedef INITIALIZE_PERFECT_HASH_TABLE_ALLOCATOR
      *PINITIALIZE_PERFECT_HASH_TABLE_ALLOCATOR;

//
// Self-test typedefs.
//

typedef
_Success_(return != 0)
BOOLEAN
(NTAPI SELF_TEST_PERFECT_HASH_TABLE)(
    _In_ PRTL Rtl,
    _In_ PALLOCATOR Allocator,
    _In_ struct _PERFECT_HASH_TABLE_ANY_API *AnyApi,
    _In_ PCUNICODE_STRING TestDataDirectory,
    _In_ PULONG MaximumConcurrency,
    _In_ PERFECT_HASH_TABLE_ALGORITHM_ID AlgorithmId,
    _In_ PERFECT_HASH_TABLE_HASH_FUNCTION_ID HashFunctionId,
    _In_ PERFECT_HASH_TABLE_MASK_FUNCTION_ID MaskFunctionId
    );
typedef SELF_TEST_PERFECT_HASH_TABLE *PSELF_TEST_PERFECT_HASH_TABLE;

typedef
_Success_(return != 0)
BOOLEAN
(NTAPI TEST_PERFECT_HASH_TABLE)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_opt_ BOOLEAN DebugBreakOnFailure
    );
typedef TEST_PERFECT_HASH_TABLE *PTEST_PERFECT_HASH_TABLE;

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
// Define the main PerfectHash API structure.
//

typedef struct _Struct_size_bytes_(SizeOfStruct) _PERFECT_HASH_TABLE_API {

    //
    // Size of the structure, in bytes.  This is filled in automatically by
    // LoadPerfectHashTableApi() based on the initial SizeOfAnyApi parameter.
    //

    _In_range_(sizeof(struct _PERFECT_HASH_TABLE_API),
               sizeof(struct _PERFECT_HASH_TABLE_API_EX)) ULONG SizeOfStruct;

    //
    // Number of function pointers contained in the structure.  This is filled
    // in automatically by LoadPerfectHashTableApi() based on the initial
    // SizeOfAnyApi parameter divided by the size of a function pointer.
    //

    ULONG NumberOfFunctions;

    //
    // Begin function pointers.
    //

    union {
        PVOID FirstFunctionPointer;
        PSET_C_SPECIFIC_HANDLER SetCSpecificHandler;
    };

    PLOAD_PERFECT_HASH_TABLE_KEYS LoadPerfectHashTableKeys;
    PDESTROY_PERFECT_HASH_TABLE_KEYS DestroyPerfectHashTableKeys;

    PCREATE_PERFECT_HASH_TABLE_CONTEXT CreatePerfectHashTableContext;
    PDESTROY_PERFECT_HASH_TABLE_CONTEXT DestroyPerfectHashTableContext;

    PCREATE_PERFECT_HASH_TABLE CreatePerfectHashTable;
    PLOAD_PERFECT_HASH_TABLE LoadPerfectHashTable;
    PTEST_PERFECT_HASH_TABLE TestPerfectHashTable;

    PGET_PERFECT_HASH_TABLE_ALGORITHM_NAME GetAlgorithmName;
    PGET_PERFECT_HASH_TABLE_HASH_FUNCTION_NAME GetHashFunctionName;
    PGET_PERFECT_HASH_TABLE_MASK_FUNCTION_NAME GetMaskFunctionName;

    PINITIALIZE_PERFECT_HASH_TABLE_ALLOCATOR InitializePerfectHashAllocator;

    PINITIALIZE_PERFECT_HASH_TABLE_ALLOCATOR_FROM_RTL_BOOTSTRAP
        InitializePerfectHashAllocatorFromRtlBootstrap;

} PERFECT_HASH_TABLE_API;
typedef PERFECT_HASH_TABLE_API *PPERFECT_HASH_TABLE_API;

//
// Define the extended API.
//

typedef struct _PERFECT_HASH_TABLE_API_EX {

    //
    // Inline PERFECT_HASH_TABLE_API.
    //

    //
    // Size of the structure, in bytes.  This is filled in automatically by
    // LoadPerfectHashTableApi() based on the initial SizeOfAnyApi parameter.
    //

    _In_range_(sizeof(struct _PERFECT_HASH_TABLE_API),
               sizeof(struct _PERFECT_HASH_TABLE_API_EX)) ULONG SizeOfStruct;

    //
    // Number of function pointers contained in the structure.  This is filled
    // in automatically by LoadPerfectHashTableApi() based on the initial
    // SizeOfAnyApi parameter divided by the size of a function pointer.
    //

    ULONG NumberOfFunctions;

    //
    // Begin function pointers.
    //

    union {
        PVOID FirstFunctionPointer;
        PSET_C_SPECIFIC_HANDLER SetCSpecificHandler;
    };

    PLOAD_PERFECT_HASH_TABLE_KEYS LoadPerfectHashTableKeys;
    PDESTROY_PERFECT_HASH_TABLE_KEYS DestroyPerfectHashTableKeys;

    PCREATE_PERFECT_HASH_TABLE_CONTEXT CreatePerfectHashTableContext;
    PDESTROY_PERFECT_HASH_TABLE_CONTEXT DestroyPerfectHashTableContext;

    PCREATE_PERFECT_HASH_TABLE CreatePerfectHashTable;
    PLOAD_PERFECT_HASH_TABLE LoadPerfectHashTable;
    PTEST_PERFECT_HASH_TABLE TestPerfectHashTable;

    PGET_PERFECT_HASH_TABLE_ALGORITHM_NAME GetAlgorithmName;
    PGET_PERFECT_HASH_TABLE_HASH_FUNCTION_NAME GetHashFunctionName;
    PGET_PERFECT_HASH_TABLE_MASK_FUNCTION_NAME GetMaskFunctionName;

    PINITIALIZE_PERFECT_HASH_TABLE_ALLOCATOR InitializePerfectHashAllocator;

    PINITIALIZE_PERFECT_HASH_TABLE_ALLOCATOR_FROM_RTL_BOOTSTRAP
        InitializePerfectHashAllocatorFromRtlBootstrap;

    //
    // Extended API functions used for testing and benchmarking.
    //

    PSELF_TEST_PERFECT_HASH_TABLE SelfTestPerfectHashTable;

} PERFECT_HASH_TABLE_API_EX;
typedef PERFECT_HASH_TABLE_API_EX *PPERFECT_HASH_TABLE_API_EX;

typedef union _PERFECT_HASH_TABLE_ANY_API {
    PERFECT_HASH_TABLE_API Api;
    PERFECT_HASH_TABLE_API_EX ApiEx;
} PERFECT_HASH_TABLE_ANY_API;
typedef PERFECT_HASH_TABLE_ANY_API *PPERFECT_HASH_TABLE_ANY_API;

FORCEINLINE
BOOLEAN
LoadPerfectHashTableApi(
    _In_ PRTL Rtl,
    _Inout_ HMODULE *ModulePointer,
    _In_opt_ PUNICODE_STRING ModulePath,
    _In_ ULONG SizeOfAnyApi,
    _Out_writes_bytes_all_(SizeOfAnyApi) PPERFECT_HASH_TABLE_ANY_API AnyApi
    )
/*++

Routine Description:

    Loads the perfect hash table module and resolves all API functions for
    either the PERFECT_HASH_TABLE_API or PERFECT_HASH_TABLE_API_EX structure.
    The desired API is indicated by the SizeOfAnyApi parameter.

    Example use:

        PERFECT_HASH_TABLE_API_EX GlobalApi;
        PPERFECT_HASH_TABLE_API_EX Api;

        Success = LoadPerfectHashApi(Rtl,
                                     NULL,
                                     NULL,
                                     sizeof(GlobalApi),
                                     (PPERFECT_HASH_TABLE_ANY_API)&GlobalApi);
        ASSERT(Success);
        Api = &GlobalApi;

    In this example, the extended API will be provided as our sizeof(GlobalApi)
    will indicate the structure size used by PERFECT_HASH_TABLE_API_EX.

Arguments:

    Rtl - Supplies a pointer to an initialized RTL structure.

    ModulePointer - Optionally supplies a pointer to an existing module handle
        for which the API symbols are to be resolved.  May be NULL.  If not
        NULL, but the pointed-to value is NULL, then this parameter will
        receive the handle obtained by LoadLibrary() as part of this call.
        If the string table module is no longer needed, but the program will
        keep running, the caller should issue a FreeLibrary() against this
        module handle.

    ModulePath - Optionally supplies a pointer to a UNICODE_STRING structure
        representing a path name of the string table module to be loaded.
        If *ModulePointer is not NULL, it takes precedence over this parameter.
        If NULL, and no module has been provided via *ModulePointer, loading
        will be attempted via LoadLibraryA("PerfectHashTable.dll")'.

    SizeOfAnyApi - Supplies the size, in bytes, of the underlying structure
        pointed to by the AnyApi parameter.

    AnyApi - Supplies the address of a structure which will receive resolved
        API function pointers.  The API furnished will depend on the size
        indicated by the SizeOfAnyApi parameter.

Return Value:

    TRUE on success, FALSE on failure.

--*/
{
    BOOL Success;
    HMODULE Module = NULL;
    ULONG NumberOfSymbols;
    ULONG NumberOfResolvedSymbols;

    //
    // Define the API names.
    //
    // N.B. These names must match PERFECT_HASH_TABLE_API_EX exactly (including
    //      the order).
    //

    CONST PCSTR Names[] = {
        "SetCSpecificHandler",
        "LoadPerfectHashTableKeys",
        "DestroyPerfectHashTableKeys",
        "CreatePerfectHashTableContext",
        "DestroyPerfectHashTableContext",
        "CreatePerfectHashTable",
        "LoadPerfectHashTable",
        "TestPerfectHashTable",
        "GetPerfectHashTableAlgorithmName",
        "GetPerfectHashTableHashFunctionName",
        "GetPerfectHashTableMaskFunctionName",
        "InitializePerfectHashTableAllocator",
        "InitializePerfectHashTableAllocatorFromRtlBootstrap",
        "SelfTestPerfectHashTable",
    };

    //
    // Define an appropriately sized bitmap we can passed to Rtl->LoadSymbols().
    //

    ULONG BitmapBuffer[(ALIGN_UP(ARRAYSIZE(Names), sizeof(ULONG) << 3) >> 5)+1];
    RTL_BITMAP FailedBitmap = { ARRAYSIZE(Names)+1, (PULONG)&BitmapBuffer };

    //
    // Determine the number of symbols we want to resolve based on the size of
    // the API indicated by the caller.
    //

    if (SizeOfAnyApi == sizeof(AnyApi->Api)) {
        NumberOfSymbols = sizeof(AnyApi->Api) / sizeof(ULONG_PTR);
    } else if (SizeOfAnyApi == sizeof(AnyApi->ApiEx)) {
        NumberOfSymbols = sizeof(AnyApi->ApiEx) / sizeof(ULONG_PTR);
    } else {
        return FALSE;
    }

    //
    // Subtract the structure header (size, number of symbols, etc).
    //

    NumberOfSymbols -= (
        (FIELD_OFFSET(PERFECT_HASH_TABLE_API, FirstFunctionPointer)) /
        sizeof(ULONG_PTR)
    );

    //
    // Attempt to load the underlying perfect hash table module if necessary.
    //

    if (ARGUMENT_PRESENT(ModulePointer)) {
        Module = *ModulePointer;
    }

    if (!Module) {
        if (ARGUMENT_PRESENT(ModulePath)) {
            Module = LoadLibraryW(ModulePath->Buffer);
        } else {
            Module = LoadLibraryA("PerfectHashTable.dll");
        }
    }

    if (!Module) {
        return FALSE;
    }

    //
    // We've got a handle to the perfect hash table.  Load the symbols we want
    // dynamically via Rtl->LoadSymbols().
    //

    Success = Rtl->LoadSymbols(Names,
                               NumberOfSymbols,
                               (PULONG_PTR)&AnyApi->Api.FirstFunctionPointer,
                               NumberOfSymbols,
                               Module,
                               &FailedBitmap,
                               TRUE,
                               &NumberOfResolvedSymbols);

    ASSERT(Success);

    //
    // Debug helper: if the breakpoint below is hit, then the symbol names
    // have potentially become out of sync.  Look at the value of first failed
    // symbol to assist in determining the cause.
    //

    if (NumberOfSymbols != NumberOfResolvedSymbols) {
        PCSTR FirstFailedSymbolName;
        ULONG FirstFailedSymbol;
        ULONG NumberOfFailedSymbols;

        NumberOfFailedSymbols = Rtl->RtlNumberOfSetBits(&FailedBitmap);
        FirstFailedSymbol = Rtl->RtlFindSetBits(&FailedBitmap, 1, 0);
        FirstFailedSymbolName = Names[FirstFailedSymbol-1];
        __debugbreak();
    }

    //
    // Set the C specific handler for the module, such that structured
    // exception handling will work.
    //

    AnyApi->Api.SetCSpecificHandler(Rtl->__C_specific_handler);

    //
    // Save the structure size and number of function pointers.
    //

    AnyApi->Api.SizeOfStruct = SizeOfAnyApi;
    AnyApi->Api.NumberOfFunctions = NumberOfSymbols;

    //
    // Update the caller's pointer and return success.
    //

    if (ARGUMENT_PRESENT(ModulePointer)) {
        *ModulePointer = Module;
    }

    return TRUE;
}

#ifdef __cplusplus
} // extern "C"
#endif

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
