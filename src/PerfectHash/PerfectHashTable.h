/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTable.h

Abstract:

    This is the private header file for the PerfectHashTable component.  It
    defines function typedefs and function declarations for all major (i.e. not
    local to the module) functions available for use by individual modules
    within this component.

--*/

#ifndef _PERFECT_HASH_INTERNAL_BUILD
#error PerfectHashTablePrivate.h being included but _PERFECT_HASH_INTERNAL_BUILD not set.
#endif

#pragma once

#include "stdafx.h"

//
// Define the PERFECT_HASH_TABLE_STATE structure.
//

typedef union _PERFECT_HASH_TABLE_STATE {
    struct {

        //
        // When set, indicates the table is in a valid state.
        //

        ULONG Valid:1;

        //
        // Unused bits.
        //

        ULONG Unused:31;
    };
    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_TABLE_STATE;
C_ASSERT(sizeof(PERFECT_HASH_TABLE_STATE) == sizeof(ULONG));
typedef PERFECT_HASH_TABLE_STATE *PPERFECT_HASH_TABLE_STATE;

#define IsValidTable(Table) (Table->State.Valid == TRUE)

//
// Define the PERFECT_HASH_TABLE structure.
//

typedef struct _Struct_size_bytes_(SizeOfStruct) _PERFECT_HASH_TABLE {

    COMMON_COMPONENT_HEADER(PERFECT_HASH_TABLE);

    //
    // Flags provided during table creation/loading.
    //

    PERFECT_HASH_CONTEXT_CREATE_TABLE_FLAGS CreateTableFlags;
    PERFECT_HASH_TABLE_LOAD_FLAGS LoadFlags;

    //
    // Slim read/write lock guarding the structure.
    //

    SRWLOCK Lock;

    //
    // Base address of the memory map for the backing file.
    //

    union {
        PVOID BaseAddress;
        PULONG Data;
    };

    //
    // If we were able to allocate a large page buffer of sufficient size,
    // BaseAddress above will point to it, and the following variable will
    // capture the original mapped address.
    //

    PVOID MappedAddress;

    //
    // If a table is loaded successfully, an array will be allocated for storing
    // values (as part of the Insert()/Lookup() API), the base address for which
    // is captured by the next field.
    //

    union {
        PVOID ValuesBaseAddress;
        PULONG Values;
    };

    //
    // Pointer to an initialized RTL structure.
    //

    PRTL Rtl;

    //
    // Pointer to an initialized ALLOCATOR structure.
    //

    PALLOCATOR Allocator;

    //
    // Generic singly-linked list entry.
    //

    SLIST_ENTRY ListEntry;

    //
    // Capture the number of elements in the underlying perfect hash table.
    // This refers to the number of vertices for the CHM algorithm, or can
    // mean the rounded-up power-of-2 size.  The masking implementations need
    // an agnostic way to access this value, which is why it is provided here
    // at the table level (despite being obtainable from things like the number
    // of vertices or Keys->NumberOfElements).
    //

    ULONG HashSize;
    ULONG IndexSize;

    //
    // Similarly, provide a convenient way to access the table "shift" amount
    // if a shifting mask routine is active.  This value is the number of
    // trailing zeros of the Size above for tables whose size is a power of 2.
    // It is not used if modulus masking is active.
    //

    ULONG HashShift;
    ULONG IndexShift;

    //
    // Mask.
    //

    ULONG HashMask;
    ULONG IndexMask;

    //
    // The following value represents how many times we need to XOR the high
    // part of a word with the low part of a word -- where word is being used
    // in the general computer word sense (i.e. not a 2-byte short) -- such
    // that the final value is within the bounds of the table mask.  It is
    // also the value of log2(Table->Shift).
    //

    ULONG HashFold;
    ULONG IndexFold;

    //
    // If modulus masking is active, this represents the modulus that will be
    // used for masking, e.g. Input %= Table->Modulus.
    //

    ULONG HashModulus;
    ULONG IndexModulus;

    //
    // If a resize event is triggered, this field will capture the new number
    // of vertices to use in search of a perfect hash table solution.  (This
    // will always be at least equal to or greater than the number of keys.)
    //

    ULARGE_INTEGER RequestedNumberOfTableElements;

    //
    // The algorithm in use.
    //

    PERFECT_HASH_ALGORITHM_ID AlgorithmId;

    //
    // The hash function in use.
    //

    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId;

    //
    // The masking type in use.
    //

    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId;

    ULONG Padding2;

    //
    // Pointer to the keys corresponding to this perfect hash table.  May be
    // NULL.
    //

    PPERFECT_HASH_KEYS Keys;

    //
    // Pointer to the PERFECT_HASH_CONTEXT structure in use.
    //

    PPERFECT_HASH_CONTEXT Context;

    //
    // Handle to the backing file.
    //

    HANDLE FileHandle;

    //
    // Handle to the memory mapping for the backing file.
    //

    HANDLE MappingHandle;

    //
    // Fully-qualified, NULL-terminated path of the backing file.  The path is
    // automatically derived from the keys file.
    //

    UNICODE_STRING Path;

    //
    // Capture the mapping size of the underlying array, which will be aligned
    // up to a system allocation granularity.
    //

    ULARGE_INTEGER MappingSizeInBytes;

    //
    // Handle to the info stream backing file.
    //

    HANDLE InfoStreamFileHandle;

    //
    // Handle to the memory mapping for the backing file.
    //

    HANDLE InfoStreamMappingHandle;

    //
    // Base address of the memory map for the :Info stream.
    //

    union {
        PVOID InfoStreamBaseAddress;
        struct _TABLE_INFO_ON_DISK_HEADER *Header;
    };

    //
    // Fully-qualified, NULL-terminated path of the :Info stream associated with
    // the path above.
    //

    UNICODE_STRING InfoStreamPath;

    //
    // Capture the mapping size and actual structure size for the :Info stream.
    //

    ULARGE_INTEGER InfoMappingSizeInBytes;
    ULARGE_INTEGER InfoActualStructureSizeInBytes;

    //
    // Backing vtbl.
    //

    PERFECT_HASH_TABLE_VTBL Interface;

    PVOID Padding;

} PERFECT_HASH_TABLE;
typedef PERFECT_HASH_TABLE *PPERFECT_HASH_TABLE;

#define TryAcquirePerfectHashTableLockExclusive(Table) \
    TryAcquireSRWLockExclusive(&Table->Lock)

#define ReleasePerfectHashTableLockExclusive(Table) \
    ReleaseSRWLockExclusive(&Table->Lock)

//
// Internal method typedefs.
//

typedef
HRESULT
(NTAPI PERFECT_HASH_TABLE_INITIALIZE)(
    _In_ PPERFECT_HASH_TABLE Table
    );
typedef PERFECT_HASH_TABLE_INITIALIZE
      *PPERFECT_HASH_TABLE_INITIALIZE;

typedef
VOID
(NTAPI PERFECT_HASH_TABLE_RUNDOWN)(
    _In_ _Post_ptr_invalid_ PPERFECT_HASH_TABLE Table
    );
typedef PERFECT_HASH_TABLE_RUNDOWN
      *PPERFECT_HASH_TABLE_RUNDOWN;

//
// Function decls.
//

extern PERFECT_HASH_TABLE_INITIALIZE PerfectHashTableInitialize;
extern PERFECT_HASH_TABLE_RUNDOWN PerfectHashTableRundown;
extern PERFECT_HASH_TABLE_LOAD PerfectHashTableLoad;
extern PERFECT_HASH_TABLE_GET_FLAGS PerfectHashTableGetFlags;
extern PERFECT_HASH_TABLE_TEST PerfectHashTableTest;
extern PERFECT_HASH_TABLE_INSERT PerfectHashTableInsert;
extern PERFECT_HASH_TABLE_LOOKUP PerfectHashTableLookup;
extern PERFECT_HASH_TABLE_DELETE PerfectHashTableDelete;
extern PERFECT_HASH_TABLE_INDEX PerfectHashTableIndex;
extern PERFECT_HASH_TABLE_GET_ALGORITHM_NAME
    PerfectHashTableGetAlgorithmName;
extern PERFECT_HASH_TABLE_GET_HASH_FUNCTION_NAME
    PerfectHashTableGetHashFunctionName;
extern PERFECT_HASH_TABLE_GET_MASK_FUNCTION_NAME
    PerfectHashTableGetMaskFunctionName;

//
// Add some helper macros that improve the aesthetics of using the index,
// hash and mask COM-style routines.  All macros assume a Table variable is
// in scope, as well as an Error: label that can be jumped to if the method
// fails.
//

#define INDEX(Key, Result)                                \
    if (FAILED(Table->Vtbl->Index(Table, Key, Result))) { \
        goto Error;                                       \
    }

#define HASH(Key, Result)                                \
    if (FAILED(Table->Vtbl->Hash(Table, Key, Result))) { \
        goto Error;                                      \
    }

#define MASK_HASH(Hash, Result)                               \
    if (FAILED(Table->Vtbl->MaskHash(Table, Hash, Result))) { \
        goto Error;                                           \
    }

#define MASK_INDEX(Hash, Result)                               \
    if (FAILED(Table->Vtbl->MaskIndex(Table, Hash, Result))) { \
        goto Error;                                            \
    }


PERFECT_HASH_TABLE_HASH PerfectHashTableHashCrc32Rotate;
PERFECT_HASH_TABLE_HASH PerfectHashTableHashJenkins;
PERFECT_HASH_TABLE_HASH PerfectHashTableHashRotateXor;
PERFECT_HASH_TABLE_HASH PerfectHashTableHashAddSubXor;
PERFECT_HASH_TABLE_HASH PerfectHashTableHashXor;
PERFECT_HASH_TABLE_HASH PerfectHashTableHashCrc32RotateXor;

PERFECT_HASH_TABLE_SEEDED_HASH PerfectHashTableSeededHashCrc32Rotate;
PERFECT_HASH_TABLE_SEEDED_HASH PerfectHashTableSeededHashJenkins;
PERFECT_HASH_TABLE_SEEDED_HASH PerfectHashTableSeededHashRotateXor;
PERFECT_HASH_TABLE_SEEDED_HASH PerfectHashTableSeededHashAddSubXor;
PERFECT_HASH_TABLE_SEEDED_HASH PerfectHashTableSeededHashXor;
PERFECT_HASH_TABLE_SEEDED_HASH PerfectHashTableSeededHashCrc32RotateXor;

PERFECT_HASH_TABLE_MASK_HASH PerfectHashTableMaskHashModulus;
PERFECT_HASH_TABLE_MASK_HASH PerfectHashTableMaskHashAnd;
PERFECT_HASH_TABLE_MASK_HASH PerfectHashTableMaskHashXorAnd;
PERFECT_HASH_TABLE_MASK_HASH PerfectHashTableMaskHashFoldOnce;
PERFECT_HASH_TABLE_MASK_HASH PerfectHashTableMaskHashFoldTwice;
PERFECT_HASH_TABLE_MASK_HASH PerfectHashTableMaskHashFoldThrice;

PERFECT_HASH_TABLE_MASK_INDEX PerfectHashTableMaskIndexModulus;
PERFECT_HASH_TABLE_MASK_INDEX PerfectHashTableMaskIndexAnd;
PERFECT_HASH_TABLE_MASK_INDEX PerfectHashTableMaskIndexXorAnd;
PERFECT_HASH_TABLE_MASK_INDEX PerfectHashTableMaskIndexFoldOnce;
PERFECT_HASH_TABLE_MASK_INDEX PerfectHashTableMaskIndexFoldTwice;
PERFECT_HASH_TABLE_MASK_INDEX PerfectHashTableMaskIndexFoldThrice;

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
