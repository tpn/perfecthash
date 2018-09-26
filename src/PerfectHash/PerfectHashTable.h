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
        // When set, indicates paths have been initialized via a call to
        // PerfectHashTableInitializePaths().
        //

        ULONG PathsInitialized:1;

        //
        // Unused bits.
        //

        ULONG Unused:30;
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
    // Capture any flags provided during table creation, loading and
    // compilation.
    //

    PERFECT_HASH_CONTEXT_CREATE_TABLE_FLAGS ContextCreateTableFlags;
    PERFECT_HASH_TABLE_FLAGS TableCreateFlags;
    PERFECT_HASH_TABLE_LOAD_FLAGS TableLoadFlags;
    PERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags;

    //
    // If a table is loaded or created successfully, an array will be allocated
    // for storing values (as part of the Insert()/Lookup() API), the base
    // address for which is captured by the next field.
    //

    union {
        PVOID ValuesBaseAddress;
        PULONG Values;
    };

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

    ULONG Padding3;

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
    // Pointer to the path associated with the table file.  This is constructed
    // up-front such that it is available for other file creation logic to
    // create new paths from.
    //

    PPERFECT_HASH_PATH TablePath;

    //
    // Pointers to files associated with the table.
    //

    PPERFECT_HASH_FILE TableFile;
    PPERFECT_HASH_FILE InfoStream;
    PPERFECT_HASH_FILE CHeaderFile;
    PPERFECT_HASH_FILE CSourceFile;
    PPERFECT_HASH_FILE CSourceKeysFile;
    PPERFECT_HASH_FILE CSourceTableDataFile;

    //
    // Backing vtbl.
    //

    PERFECT_HASH_TABLE_VTBL Interface;

    //PVOID Padding4;

} PERFECT_HASH_TABLE;
typedef PERFECT_HASH_TABLE *PPERFECT_HASH_TABLE;

#define TryAcquirePerfectHashTableLockExclusive(Table) \
    TryAcquireSRWLockExclusive(&Table->Lock)

#define ReleasePerfectHashTableLockExclusive(Table) \
    ReleaseSRWLockExclusive(&Table->Lock)

#define PerfectHashTableName(Table) \
    &Table->TableFile->Path->BaseNameA

//
// Internal method typedefs.
//

typedef
HRESULT
(NTAPI PERFECT_HASH_TABLE_INITIALIZE)(
    _In_ PPERFECT_HASH_TABLE Table
    );
typedef PERFECT_HASH_TABLE_INITIALIZE *PPERFECT_HASH_TABLE_INITIALIZE;

typedef
_Check_return_
_Success_(return >= 0)
HRESULT
(NTAPI PERFECT_HASH_TABLE_CREATE_PATH)(
    _In_opt_ PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    _In_opt_ PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    _In_opt_ PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    _In_opt_ PCUNICODE_STRING NewOutputDirectory,
    _In_opt_ PCUNICODE_STRING NewBaseName,
    _In_opt_ PCUNICODE_STRING AdditionalSuffix,
    _In_opt_ PCUNICODE_STRING NewExtension,
    _In_opt_ PCUNICODE_STRING NewStreamName,
    _In_ PPERFECT_HASH_PATH ExistingPath,
    _Out_ PPERFECT_HASH_PATH *Path,
    _Out_ PPERFECT_HASH_PATH_PARTS *Parts
    );
typedef PERFECT_HASH_TABLE_CREATE_PATH *PPERFECT_HASH_TABLE_CREATE_PATH;

typedef
VOID
(NTAPI PERFECT_HASH_TABLE_RUNDOWN)(
    _In_ _Post_ptr_invalid_ PPERFECT_HASH_TABLE Table
    );
typedef PERFECT_HASH_TABLE_RUNDOWN *PPERFECT_HASH_TABLE_RUNDOWN;

//
// Function decls.
//

extern PERFECT_HASH_TABLE_INITIALIZE PerfectHashTableInitialize;
extern PERFECT_HASH_TABLE_CREATE_PATH PerfectHashTableCreatePath;
extern PERFECT_HASH_TABLE_RUNDOWN PerfectHashTableRundown;
extern PERFECT_HASH_TABLE_CREATE PerfectHashTableCreate;
extern PERFECT_HASH_TABLE_LOAD PerfectHashTableLoad;
extern PERFECT_HASH_TABLE_GET_FLAGS PerfectHashTableGetFlags;
extern PERFECT_HASH_TABLE_COMPILE PerfectHashTableCompile;
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
extern PERFECT_HASH_TABLE_GET_FILE PerfectHashTableGetFile;

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

//
// Helper method for initializing a table suffix from a given algorithm, mask
// and hash function.
//

FORCEINLINE
_Check_return_
_Success_(return >= 0)
HRESULT
InitializeTableSuffix(
    _In_ PUNICODE_STRING Suffix,
    _In_ PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    _In_ PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    _In_ PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    _In_opt_ PCUNICODE_STRING AdditionalSuffix,
    _In_opt_ PULONG_INTEGER TableSize
    )
{
    PWSTR Dest;
    USHORT Index;
    USHORT Count;
    HRESULT Result;
    BOOLEAN Success;
    ULONG_PTR ExpectedDest;
    BYTE NumberOfDigits = 0;
    LONG_INTEGER TableSuffixLength = { 0 };
    PUNICODE_STRING AlgorithmName = NULL;
    PUNICODE_STRING HashFunctionName = NULL;
    PUNICODE_STRING MaskFunctionName = NULL;

    if (ARGUMENT_PRESENT(TableSize)) {
        NumberOfDigits = CountNumberOfLongLongDigitsInline(TableSize->QuadPart);
        TableSuffixLength.LongPart += (
            sizeof(L'_') +
            NumberOfDigits
        );
    }

    if (IsValidPerfectHashAlgorithmId(AlgorithmId)) {
        AlgorithmName = (PUNICODE_STRING)AlgorithmNames[AlgorithmId];
        TableSuffixLength.LongPart += (
            sizeof(L'_') +
            AlgorithmName->Length
        );
    }

    if (IsValidPerfectHashHashFunctionId(HashFunctionId)) {
        HashFunctionName = (PUNICODE_STRING)HashFunctionNames[HashFunctionId];
        TableSuffixLength.LongPart += (
            sizeof(L'_') +
            HashFunctionName->Length
        );
    }

    if (IsValidPerfectHashMaskFunctionId(MaskFunctionId)) {
        MaskFunctionName = (PUNICODE_STRING)MaskFunctionNames[MaskFunctionId];
        TableSuffixLength.LongPart += (
            sizeof(L'_') +
            HashFunctionName->Length
        );
    }

    if (ARGUMENT_PRESENT(AdditionalSuffix)) {
        if (!IsValidUnicodeString(AdditionalSuffix)) {
            return E_INVALIDARG;
        }
        TableSuffixLength.LongPart += (
            sizeof(L'_') +
            AdditionalSuffix->Length
        );
    }

    if (TableSuffixLength.HighPart) {
        return PH_E_STRING_BUFFER_OVERFLOW;
    }

    if ((ULONG)Suffix->MaximumLength <
        TableSuffixLength.LongPart + sizeof(WCHAR)) {
        return PH_E_STRING_BUFFER_OVERFLOW;
    }

    Dest = Suffix->Buffer;

    if (NumberOfDigits) {
        *Dest++ = L'_';
        Suffix->Length = 1;
        Success = AppendLongLongIntegerToUnicodeString(Suffix,
                                                       TableSize->QuadPart,
                                                       NumberOfDigits,
                                                       L'\0');
        if (!Success) {
            return PH_E_STRING_BUFFER_OVERFLOW;
        }

        Dest += NumberOfDigits;
    }

    if (AlgorithmName) {
        *Dest++ = L'_';
        Count = AlgorithmName->Length >> 1;
        CopyMemory(Dest, AlgorithmName->Buffer, AlgorithmName->Length);
        Dest += Count;
    }

    if (HashFunctionName) {
        *Dest++ = L'_';
        Count = HashFunctionName->Length >> 1;
        CopyMemory(Dest, HashFunctionName->Buffer, HashFunctionName->Length);
        Dest += Count;
    }

    if (MaskFunctionName) {
        *Dest++ = L'_';
        Count = MaskFunctionName->Length >> 1;
        CopyMemory(Dest, MaskFunctionName->Buffer, MaskFunctionName->Length);
        Dest += Count;
    }

    if (AdditionalSuffix) {
        *Dest++ = L'_';
        Count = AdditionalSuffix->Length >> 1;
        CopyMemory(Dest, AdditionalSuffix->Buffer, AdditionalSuffix->Length);
        Dest += Count;
    }

    ExpectedDest = (
        RtlPointerToOffset(
            Suffix->Buffer,
            TableSuffixLength.LowPart
        )
    );

    if ((ULONG_PTR)Dest != ExpectedDest) {
        return PH_E_INVARIANT_CHECK_FAILED;
    }

    Suffix->Length = TableSuffixLength.LowPart;

    *Dest++ = L'\0';

    return S_OK;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
