/*++

Copyright (c) 2018-2019 Trent Nelson <trent@trent.me>

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
        // When set, indicates the Table->TableInfoOnDisk structure is backed
        // by heap-allocated memory obtained from the allocator, and thus, must
        // be free'd by the allocator during rundown.
        //

        ULONG TableInfoOnDiskWasHeapAllocated:1;

        //
        // As above, but for the Assigned/TableDataBaseAddress pointer.
        //

        ULONG TableDataWasHeapAllocated:1;

        //
        // Unused bits.
        //

        ULONG Unused:29;
    };
    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_TABLE_STATE;
C_ASSERT(sizeof(PERFECT_HASH_TABLE_STATE) == sizeof(ULONG));
typedef PERFECT_HASH_TABLE_STATE *PPERFECT_HASH_TABLE_STATE;

#define IsValidTable(Table) ((Table)->State.Valid == TRUE)
#define IsTableCreateOnly(Table) ((Table)->TableCreateFlags.CreateOnly == TRUE)

#define NoFileIo(Table) ((Table)->TableCreateFlags.NoFileIo == TRUE)
#define IsParanoid(Table) ((Table)->TableCreateFlags.Paranoid == TRUE)
#define IsIndexOnly(Table) ((Table)->TableCreateFlags.IndexOnly != FALSE)
#define UseRwsSectionForTableValues(Table) \
    ((Table)->TableCreateFlags.UseRwsSectionForTableValues != FALSE)

#define UseNonTemporalAvx2Routines(Table) \
    ((Table)->TableCreateFlags.UseNonTemporalAvx2Routines != FALSE)

#define WasTableInfoOnDiskHeapAllocated(Table) \
    ((Table)->State.TableInfoOnDiskWasHeapAllocated == TRUE)

#define WasTableDataHeapAllocated(Table) \
    ((Table)->State.TableDataWasHeapAllocated == TRUE)

#define IncludeNumberOfTableResizeEventsInOutputPath(Table) (                  \
    ((Table)->TableCreateFlags.IncludeNumberOfTableResizeEventsInOutputPath == \
     TRUE)                                                                     \
)

#define IncludeNumberOfTableElementsInOutputPath(Table) (                  \
    ((Table)->TableCreateFlags.IncludeNumberOfTableElementsInOutputPath == \
     TRUE)                                                                 \
)

#define TableResizeRequiresRename(Table) (                 \
    IncludeNumberOfTableResizeEventsInOutputPath(Table) || \
    IncludeNumberOfTableElementsInOutputPath(Table)        \
)

FORCEINLINE
BOOLEAN
SkipWritingCsvRow(
    _In_ PERFECT_HASH_TABLE_CREATE_FLAGS Flags,
    _In_opt_ HRESULT TableCreateResult
    )
{
    return (
        (Flags.OmitCsvRowIfTableCreateFailed && TableCreateResult != S_OK) ||
        (Flags.OmitCsvRowIfTableCreateSucceeded && TableCreateResult == S_OK)
    );
}

//
// Define the PERFECT_HASH_TABLE structure.
//

typedef struct _Struct_size_bytes_(SizeOfStruct) _PERFECT_HASH_TABLE {

    COMMON_COMPONENT_HEADER(PERFECT_HASH_TABLE);

    //
    // Capture any flags provided during table creation, loading and
    // compilation.
    //

    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags;
    PERFECT_HASH_TABLE_LOAD_FLAGS TableLoadFlags;
    PERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags;

    //
    // Type to use for the table data/assigned array when generating the C files
    // for the compiled perfect hash table.
    //

    TYPE TableDataArrayType;

    //
    // Type to use for the table values when generating the C files for the
    // compiled perfect hash table.
    //

    TYPE ValueType;

    //
    // Size of an individual value in the hash table, in bytes.  (Must be
    // either 4 or 8 bytes.)
    //

    ULONG ValueSizeInBytes;

    //
    // Optional table creation parameters specified to Create().
    //

    PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters;

    //
    // Pointer to the active on-disk structure describing the table.  This may
    // be backed by stack-allocated memory (during creation), heap-allocated
    // memory (after creation), or memory mapped memory (if loaded from disk).
    //

    struct _TABLE_INFO_ON_DISK *TableInfoOnDisk;

    //
    // If we're in "find best coverage" mode, a pointer to an assigned memory
    // coverage structure that reflects the coverage of the winning graph.
    //

    struct _ASSIGNED_MEMORY_COVERAGE *Coverage;

    //
    // Pointer to a string representation of the Index() routine's
    // implementation in C.
    //

    PCSTRING IndexImplString;

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
    // Pointer to the base address of the table data.  During creation, this
    // is referred to as the "Assigned" array.  During the load phase, it is
    // referred to as "table data".
    //

    union {
        PULONG Assigned;
        PULONG TableData;
        PVOID TableDataBaseAddress;
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

    //
    // Maximum recursion depth observed by the graph traversal function.
    //

    ULONG MaximumGraphTraversalDepth;

    //
    // Number of empty vertices encountered during graph assignment step.
    //

    ULONG NumberOfEmptyVertices;

    //
    // Pad out to an 8-byte boundary.
    //

    ULONG Padding1;

    //
    // Pointer to the path for the output directory (below).
    //

    PPERFECT_HASH_PATH OutputPath;

    //
    // Pointer to the output directory for this table.
    //

    PPERFECT_HASH_DIRECTORY OutputDirectory;

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
    // Timestamp of instance creation.  The STRING structure's Buffer is wired
    // up to the address of the TimestampBuffer variable.
    //

    CHAR TimestampBuffer[RTL_TIMESTAMP_FORMAT_LENGTH];
    STRING TimestampString;

    //
    // Pointers to STRINGs representing the type names to use for primitive
    // variables (index, keys, seeds, values etc).
    //

    PCSTRING SeedTypeName;
    PCSTRING IndexTypeName;
    PCSTRING KeySizeTypeName;
    PCSTRING KeysArrayTypeName;
    PCSTRING TableDataArrayTypeName;
    PCSTRING OriginalKeySizeTypeName;
    union {
        PCSTRING ValueTypeName;
        PCSTRING TableValuesArrayTypeName;
    };

    //
    // Pointer to the array of C type names based on enum TYPE values.  Defaults
    // to the global constant CTypeNames.
    //

    PCSTRING CTypeNames;

    //
    // Pointers to files associated with the table.
    //

#define EXPAND_AS_FILE(               \
    Verb, VUpper, Name, Upper,        \
    EofType, EofValue,                \
    Suffix, Extension, Stream, Base   \
)                                     \
    PPERFECT_HASH_FILE Name;

#define EXPAND_AS_FIRST_FILE(         \
    Verb, VUpper, Name, Upper,        \
    EofType, EofValue,                \
    Suffix, Extension, Stream, Base   \
)                                     \
    union {                           \
        PPERFECT_HASH_FILE Name;      \
        PPERFECT_HASH_FILE FirstFile; \
    };

#define EXPAND_AS_LAST_FILE(          \
    Verb, VUpper, Name, Upper,        \
    EofType, EofValue,                \
    Suffix, Extension, Stream, Base   \
)                                     \
    union {                           \
        PPERFECT_HASH_FILE Name;      \
        PPERFECT_HASH_FILE LastFile;  \
    };

    FILE_WORK_TABLE(EXPAND_AS_FIRST_FILE,
                    EXPAND_AS_FILE,
                    EXPAND_AS_LAST_FILE)

    //
    // Benchmarking timestamps.
    //

    ULONG BenchmarkWarmups;
    ULONG BenchmarkAttempts;
    ULONG BenchmarkIterations;
    ULONG Padding4;

    TIMESTAMP SlowIndexTimestamp;
    TIMESTAMP SeededHashTimestamp;
    TIMESTAMP NullSeededHashTimestamp;

    //
    // Cycle counters and elapsed microseconds copied from the winning graph.
    //

    LARGE_INTEGER AddKeysElapsedCycles;
    LARGE_INTEGER HashKeysElapsedCycles;
    LARGE_INTEGER AddHashedKeysElapsedCycles;

    LARGE_INTEGER AddKeysElapsedMicroseconds;
    LARGE_INTEGER HashKeysElapsedMicroseconds;
    LARGE_INTEGER AddHashedKeysElapsedMicroseconds;

    //
    // Backing vtbl.
    //

    PERFECT_HASH_TABLE_VTBL Interface;

    //PVOID Padding4;

} PERFECT_HASH_TABLE;
typedef PERFECT_HASH_TABLE *PPERFECT_HASH_TABLE;

#define TryAcquirePerfectHashTableLockExclusive(Table) \
    TryAcquireSRWLockExclusive(&Table->Lock)

#define AcquirePerfectHashTableLockExclusive(Table) \
    AcquireSRWLockExclusive(&Table->Lock)

#define ReleasePerfectHashTableLockExclusive(Table) \
    ReleaseSRWLockExclusive(&Table->Lock)

#define TryAcquirePerfectHashTableLockShared(Table) \
    TryAcquireSRWLockShared(&Table->Lock)

#define AcquirePerfectHashTableLockShared(Table) \
    AcquireSRWLockShared(&Table->Lock)

#define ReleasePerfectHashTableLockShared(Table) \
    ReleaseSRWLockShared(&Table->Lock)

#define PerfectHashTableName(Table) \
    &Table->TableFile->Path->BaseNameA

//
// Helper macro for representing a table create return code as a single char.
//

#define PRINT_CHAR_FOR_TABLE_CREATE_RESULT(Result)               \
                                                                 \
    UnknownTableCreateResult = FALSE;                            \
                                                                 \
    switch (Result) {                                            \
                                                                 \
        case S_OK:                                               \
            DOT();                                               \
            break;                                               \
                                                                 \
        case PH_I_FAILED_TO_ALLOCATE_MEMORY_FOR_ALL_GRAPHS:      \
            ASTERISK();                                          \
            break;                                               \
                                                                 \
        case PH_I_OUT_OF_MEMORY:                                 \
            EXCLAMATION();                                       \
            break;                                               \
                                                                 \
        case PH_I_LOW_MEMORY:                                    \
            BIGL();                                              \
            break;                                               \
                                                                 \
        case PH_I_TABLE_CREATED_BUT_VALUES_ARRAY_ALLOC_FAILED:   \
            BIGV();                                              \
            break;                                               \
                                                                 \
        case PH_I_MAXIMUM_NUMBER_OF_TABLE_RESIZE_EVENTS_REACHED: \
            CROSS();                                             \
            break;                                               \
                                                                 \
        case PH_I_REQUESTED_NUMBER_OF_TABLE_ELEMENTS_TOO_LARGE:  \
            BIGT();                                              \
            break;                                               \
                                                                 \
        case PH_I_CREATE_TABLE_ROUTINE_FAILED_TO_FIND_SOLUTION:  \
            BIGF();                                              \
            break;                                               \
                                                                 \
        case PH_I_CREATE_TABLE_ROUTINE_RECEIVED_SHUTDOWN_EVENT:  \
            BIGS();                                              \
            break;                                               \
                                                                 \
        default:                                                 \
            QUESTION();                                          \
            UnknownTableCreateResult = TRUE;                     \
            break;                                               \
    }

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
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI PERFECT_HASH_TABLE_INITIALIZE_TABLE_SUFFIX)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ PUNICODE_STRING Suffix,
    _In_opt_ PULONG NumberOfResizeEvents,
    _In_opt_ PULARGE_INTEGER NumberOfTableElements,
    _In_opt_ PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    _In_opt_ PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    _In_opt_ PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    _In_opt_ PCUNICODE_STRING AdditionalSuffix,
    _Out_ PUSHORT AlgorithmOffset
    );
typedef PERFECT_HASH_TABLE_INITIALIZE_TABLE_SUFFIX
      *PPERFECT_HASH_TABLE_INITIALIZE_TABLE_SUFFIX;

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI PERFECT_HASH_TABLE_CREATE_PATH)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ PPERFECT_HASH_PATH ExistingPath,
    _In_opt_ PULONG NumberOfResizeEvents,
    _In_opt_ PULARGE_INTEGER NumberOfTableElements,
    _In_opt_ PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    _In_opt_ PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    _In_opt_ PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    _In_opt_ PCUNICODE_STRING NewDirectory,
    _In_opt_ PCUNICODE_STRING NewBaseName,
    _In_opt_ PCUNICODE_STRING AdditionalSuffix,
    _In_opt_ PCUNICODE_STRING NewExtension,
    _In_opt_ PCUNICODE_STRING NewStreamName,
    _Inout_opt_ PPERFECT_HASH_PATH *Path,
    _Inout_opt_ PPERFECT_HASH_PATH_PARTS *Parts
    );
typedef PERFECT_HASH_TABLE_CREATE_PATH *PPERFECT_HASH_TABLE_CREATE_PATH;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Table->Lock)
HRESULT
(NTAPI PERFECT_HASH_TABLE_CREATE_VALUES_ARRAY)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_opt_ ULONG ValueSizeInBytes
    );
typedef PERFECT_HASH_TABLE_CREATE_VALUES_ARRAY
      *PPERFECT_HASH_TABLE_CREATE_VALUES_ARRAY;

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
extern PERFECT_HASH_TABLE_INITIALIZE_TABLE_SUFFIX
    PerfectHashTableInitializeTableSuffix;
extern PERFECT_HASH_TABLE_CREATE_PATH PerfectHashTableCreatePath;
extern PERFECT_HASH_TABLE_CREATE_VALUES_ARRAY
    PerfectHashTableCreateValuesArray;
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

//
// Declare the functions.
//

#define EXPAND_AS_HASH_FUNC_DECL(Name, NumberOfSeeds, SeedMasks) \
    PERFECT_HASH_TABLE_HASH PerfectHashTableHash##Name;

PERFECT_HASH_HASH_FUNCTION_TABLE_ENTRY(EXPAND_AS_HASH_FUNC_DECL);

#define EXPAND_AS_SEEDED_HASH_FUNC_DECL(Name, NumberOfSeeds, SeedMasks) \
    PERFECT_HASH_TABLE_SEEDED_HASH PerfectHashTableSeededHash##Name;

PERFECT_HASH_HASH_FUNCTION_TABLE_ENTRY(EXPAND_AS_SEEDED_HASH_FUNC_DECL);

PERFECT_HASH_TABLE_MASK_HASH PerfectHashTableMaskHashModulus;
PERFECT_HASH_TABLE_MASK_HASH PerfectHashTableMaskHashAnd;

PERFECT_HASH_TABLE_MASK_INDEX PerfectHashTableMaskIndexModulus;
PERFECT_HASH_TABLE_MASK_INDEX PerfectHashTableMaskIndexAnd;

//
// "Ex" versions of hash and seeded hash routines.
//

#define EXPAND_AS_HASH_EX_FUNC_DECL(Name, NumberOfSeeds, SeedMasks) \
    PERFECT_HASH_TABLE_HASH_EX PerfectHashTableHashEx##Name;

PERFECT_HASH_HASH_FUNCTION_TABLE_ENTRY(EXPAND_AS_HASH_EX_FUNC_DECL);

#define EXPAND_AS_SEEDED_HASH_EX_FUNC_DECL(Name, NumberOfSeeds, SeedMasks) \
    PERFECT_HASH_TABLE_SEEDED_HASH_EX PerfectHashTableSeededHashEx##Name;

PERFECT_HASH_HASH_FUNCTION_TABLE_ENTRY(EXPAND_AS_SEEDED_HASH_EX_FUNC_DECL);

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
