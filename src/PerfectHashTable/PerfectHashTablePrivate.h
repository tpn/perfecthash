/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTablePrivate.h

Abstract:

    This is the private header file for the PerfectHashTable component.  It
    defines function typedefs and function declarations for all major (i.e. not
    local to the module) functions available for use by individual modules
    within this component.

--*/

#ifndef _PERFECT_HASH_TABLE_INTERNAL_BUILD
#error PerfectHashTablePrivate.h being included but _PERFECT_HASH_TABLE_INTERNAL_BUILD not set.
#endif

#pragma once

#include "stdafx.h"

#define PERFECT_HASH_TABLE_KEY_SIZE_IN_BYTES 4

//
// A handle to the PerfectHashTable.dll module will be captured in this variable
// via the DLL_PROCESS_ATTACH message.  This is required in order for proper
// operation of FormatMessage() when specifying FORMAT_MESSAGE_FROM_HMODULE and
// using our own internal error codes.
//

extern HMODULE PerfectHashTableModule;


//
// XXX temporary dummy error handling macro and placeholder errors.
//

#define PH_ERROR(Name, Result)

#define PH_E_CREATE_TABLE_ALREADY_IN_PROGRESS E_UNEXPECTED
#define PH_E_TOO_MANY_KEYS E_UNEXPECTED
#define PH_E_INFO_FILE_SMALLER_THAN_HEADER E_UNEXPECTED
#define PH_E_INVALID_MAGIC_VALUES E_UNEXPECTED
#define PH_E_INVALID_INFO_HEADER_SIZE E_UNEXPECTED
#define PH_E_NUM_KEYS_MISMATCH_BETWEEN_HEADER_AND_KEYS E_UNEXPECTED
#define PH_E_INVALID_ALGORITHM_ID E_UNEXPECTED
#define PH_E_INVALID_HASH_FUNCTION_ID E_UNEXPECTED
#define PH_E_INVALID_MASK_FUNCTION_ID E_UNEXPECTED
#define PH_E_HEADER_KEY_SIZE_TOO_LARGE E_UNEXPECTED
#define PH_E_NUM_KEYS_IS_ZERO E_UNEXPECTED
#define PH_E_NUM_TABLE_ELEMENTS_IS_ZERO E_UNEXPECTED
#define PH_E_NUM_KEYS_EXCEEDS_NUM_TABLE_ELEMENTS E_UNEXPECTED
#define PH_E_EXPECTED_EOF_ACTUAL_EOF_MISMATCH E_UNEXPECTED
#define PH_E_KEYS_FILE_SIZE_NOT_MULTIPLE_OF_KEY_SIZE E_UNEXPECTED
#define PH_E_KEYS_NUM_SET_BITS_NUM_KEYS_MISMATCH E_UNEXPECTED
#define PH_E_DUPLICATE_KEYS_DETECTED E_UNEXPECTED
#define PH_E_HEAP_CREATE_FAILED E_UNEXPECTED
#define PH_E_RTL_LOAD_SYMBOLS_FROM_MULTIPLE_MODULES_FAILED E_UNEXPECTED

//
//
// Cap the maximum key set size we're willing to process.
//

#define MAXIMUM_NUMBER_OF_KEYS 500000

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
// Define the PERFECT_HASH_TABLE_FLAGS structure.
//

typedef union _PERFECT_HASH_TABLE_FLAGS {
    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, indicates the table came from CreatePerfectHashTable().
        //
        // Invariant:
        //
        //  - If Created == TRUE:
        //      Assert Loaded == FALSE
        //

        ULONG Created:1;

        //
        // When set, indicates the table came from LoadPerfectHashTable().
        //
        // Invariant:
        //
        //  - If Loaded == TRUE:
        //      Assert Created == FALSE
        //

        ULONG Loaded:1;

        //
        // When set, indicates large pages are in use for the memory backing
        // Table->Data.
        //
        // N.B. This is not currently supported.
        //

        ULONG TableDataUsesLargePages:1;

        //
        // When set, indicates the Table->Values[] array was allocated with
        // large pages.
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

//
// Define the PERFECT_HASH_TABLE structure.
//

typedef struct _Struct_size_bytes_(SizeOfStruct) _PERFECT_HASH_TABLE {

    COMMON_COMPONENT_HEADER(PERFECT_HASH_TABLE);

    //
    // Base address of the memory map for the backing file.
    //

    union {
        PVOID BaseAddress;
        PULONG Data;
    };

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
    // Generic singly-linked list entry.
    //

    SLIST_ENTRY ListEntry;

    //
    // Pointer to an initialized ALLOCATOR structure.
    //

    PALLOCATOR Allocator;

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

    PERFECT_HASH_TABLE_ALGORITHM_ID AlgorithmId;

    //
    // The masking type in use.
    //

    PERFECT_HASH_TABLE_MASK_FUNCTION_ID MaskFunctionId;

    //
    // The hash function in use.
    //

    PERFECT_HASH_TABLE_HASH_FUNCTION_ID HashFunctionId;

    ULONG Padding2;

    //
    // Pointer to the keys corresponding to this perfect hash table.  May be
    // NULL.
    //

    PPERFECT_HASH_TABLE_KEYS Keys;

    //
    // Pointer to the PERFECT_HASH_TABLE_CONTEXT structure in use.
    //

    PPERFECT_HASH_TABLE_CONTEXT Context;

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

    PVOID Padding3;

} PERFECT_HASH_TABLE;
typedef PERFECT_HASH_TABLE *PPERFECT_HASH_TABLE;

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

PERFECT_HASH_TABLE_SEEDED_HASH PerfectHashTableSeededHashCrc32Rotate;
PERFECT_HASH_TABLE_SEEDED_HASH PerfectHashTableSeededHashJenkins;
PERFECT_HASH_TABLE_SEEDED_HASH PerfectHashTableSeededHashRotateXor;
PERFECT_HASH_TABLE_SEEDED_HASH PerfectHashTableSeededHashAddSubXor;
PERFECT_HASH_TABLE_SEEDED_HASH PerfectHashTableSeededHashXor;

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
// Metadata about a perfect hash table is stored in an NTFS stream named :Info
// that is tacked onto the end of the perfect hash table's file name.  Define
// a structure, TABLE_INFO_ON_DISK_HEADER, that literally represents the on-disk
// layout of this metadata.  Each algorithm implementation must write out an
// info record that conforms with this common header.  They are free to extend
// it with additional details.
//

typedef union _TABLE_INFO_ON_DISK_HEADER_FLAGS {

    struct {

        //
        // Unused bits.
        //

        ULONG Unused:32;

    };

    LONG AsLong;
    ULONG AsULong;

} TABLE_INFO_ON_DISK_HEADER_FLAGS;
C_ASSERT(sizeof(TABLE_INFO_ON_DISK_HEADER_FLAGS) == sizeof(ULONG));

typedef struct _Struct_size_bytes_(SizeOfStruct) _TABLE_INFO_ON_DISK_HEADER {

    //
    // A magic value used to identify the structure.
    //

    ULARGE_INTEGER Magic;

    //
    // Size of the structure, in bytes.
    //
    // N.B. We don't allocate this with a _Field_range_ SAL annotation as the
    //      value will vary depending on which parameters were used to create
    //      the table.
    //

    ULONG SizeOfStruct;

    //
    // Flags.
    //

    TABLE_INFO_ON_DISK_HEADER_FLAGS Flags;

    //
    // Algorithm that was used.
    //

    PERFECT_HASH_TABLE_ALGORITHM_ID AlgorithmId;

    //
    // Hash function that was used.
    //

    PERFECT_HASH_TABLE_HASH_FUNCTION_ID HashFunctionId;

    //
    // Masking type.
    //

    PERFECT_HASH_TABLE_MASK_FUNCTION_ID MaskFunctionId;

    //
    // Size of an individual key element, in bytes.
    //

    ULONG KeySizeInBytes;

    //
    // The concurrency level used to generate the hash.
    //

    ULONG Concurrency;

    //
    // Pad out to 8 bytes.
    //

    ULONG Padding;

    //
    // Number of keys in the input set.  This is used to size an appropriate
    // array for storing values.
    //

    ULARGE_INTEGER NumberOfKeys;

    //
    // Final number of elements in the underlying table.  This will vary
    // depending on how the graph was created.  If modulus masking is in use,
    // this will reflect the number of keys (unless a custom table size was
    // requested during creation).  Otherwise, this will be the number of keys
    // rounded up to the next power of 2.  (That is, take the number of keys,
    // round up to a power of 2, then round that up to the next power of 2.)
    //

    ULARGE_INTEGER NumberOfTableElements;

    //
    // Capture the hash and index details required by MaskHash(), MaskIndex()
    // and Index() routines.
    //

    ULONG HashSize;
    ULONG IndexSize;

    ULONG HashShift;
    ULONG IndexShift;

    ULONG HashMask;
    ULONG IndexMask;

    ULONG HashFold;
    ULONG IndexFold;

    ULONG HashModulus;
    ULONG IndexModulus;

    //
    // Seed data.
    //

    ULONG NumberOfSeeds;

    union {
        ULONG Seed1;
        ULONG FirstSeed;
    };

    ULONG Seed2;
    ULONG Seed3;

    union {
        ULONG Seed4;
        ULONG LastSeed;
    };

    ULONG Padding2;

    //
    // Capture statistics about the perfect hash table solution that can be
    // useful during analysis and performance comparisons.
    //

    //
    // Number of attempts at solving the solution.
    //

    ULONGLONG NumberOfAttempts;

    //
    // Number of failed attempts at solving the solution.
    //

    ULONGLONG NumberOfFailedAttempts;

    //
    // If solutions are being sought in parallel, more than one thread may
    // find a solution before it detects that someone else has already found
    // a solution (in which case, it stops solving and returns from the pool
    // callback).  This counter measures the number of solutions that were
    // found in parallel.  It corresponds to the Context->FinishedCount value.
    //
    // With a good hashing function and large concurrency value, this value
    // may often be relatively large.  E.g. with a concurrency level of 12, it
    // would not be surprising to see 12 attempts reported, and 10 solutions
    // found.  (Meaning that our concurrency level was a tad unnecessary, or
    // our hash function is unnecessarily good (which usually means expensive).)
    //
    // N.B. Finding multiple solutions in parallel is harmless, if not a little
    //      wasteful of CPU time.  The threads can detect if they are the first
    //      ones to find a solution prior to continuing with the assignment
    //      step, such that the assignment can be avoided if another thread has
    //      beaten them to that point.
    //

    ULONGLONG NumberOfSolutionsFound;

    //
    // If a solution can't be found within a configurable threshold, a "table
    // resize" event will be generated.  This results in doubling the number
    // of vertices being used for the backing table and trying the solution
    // again.  The following field captures how many times this occurred.
    //

    ULONGLONG NumberOfTableResizeEvents;

    //
    // This counter captures the sum of all prior attempts at solving the
    // solution before giving up and resizing the table.  It excludes the
    // attempts made by the winning table size (captured by NumberOfAttempts).
    // It will be zero if no resize events occurred.  We don't keep a separate
    // failed counter here, as the resize event implies all attempts failed.
    //

    ULONGLONG TotalNumberOfAttemptsWithSmallerTableSizes;

    //
    // The following counter captures the initial table size that was attempted
    // in order to solve the solution.  It will differ from the final table size
    // if there were resize events.  As we simply double the size of the table
    // on each resize event, we can extrapolate the different table sizes that
    // were tried prior to finding a winning one by looking at the initial size
    // attempted and the number of resize events.
    //

    ULONGLONG InitialTableSize;

    //
    // The following counter captures the closest we came to solving the graph
    // in previous attempts before a resize event occurred.  This is calculated
    // by taking the number of edges and subtracting the value of the context's
    // HighestDeletedEdgesCount counter.  The value represents the additional
    // number of 1 degree edges we needed to delete in order to obtain an
    // acyclic graph.  A very low number indicates that we came very close to
    // solving it as there were very few hash collisions with the seed values
    // we picked.  A very high number indicates we had no chance and there were
    // collisions galore.
    //

    ULONGLONG ClosestWeCameToSolvingGraphWithSmallerTableSizes;

    //
    // Number of cycles it took to solve the solution for the winning thread.
    // (This does not factor in the total cycle time consumed by all threads.)
    //

    ULARGE_INTEGER SolveCycles;

    //
    // Number of microseconds taken to solve the solution.
    //

    ULARGE_INTEGER SolveMicroseconds;

    //
    // Number of cycles taken to verify the solution.
    //

    ULARGE_INTEGER VerifyCycles;

    //
    // Number of microseconds taken to verify the solution.
    //

    ULARGE_INTEGER VerifyMicroseconds;

    //
    // Number of cycles taken to prepare the file.
    //

    ULARGE_INTEGER PrepareFileCycles;

    //
    // Number of microseconds taken to prepare the file.
    //

    ULARGE_INTEGER PrepareFileMicroseconds;

    //
    // Number of cycles taken to save the file.
    //

    ULARGE_INTEGER SaveFileCycles;

    //
    // Number of microseconds taken to save the file.
    //

    ULARGE_INTEGER SaveFileMicroseconds;

} TABLE_INFO_ON_DISK_HEADER;
typedef TABLE_INFO_ON_DISK_HEADER *PTABLE_INFO_ON_DISK_HEADER;

//
// Define an enumeration to capture the type of file work operations we want
// to be able to dispatch to the file work threadpool callback.
//

typedef enum _FILE_WORK_ID {

    //
    // Null ID.
    //

    FileWorkNullId = 0,

    //
    // Initial file preparation once the underlying sizes required are known.
    //

    FileWorkPrepareId = 1,

    //
    // Perfect hash solution has been solved and is ready to be saved to disk.
    //

    FileWorkSaveId,

    //
    // Invalid ID, this must come last.
    //

    FileWorkInvalidId

} FILE_WORK_ID;

FORCEINLINE
BOOLEAN
IsValidFileWorkId(
    _In_ FILE_WORK_ID FileWorkId
    )
{
    return (
        FileWorkId > FileWorkNullId &&
        FileWorkId < FileWorkInvalidId
    );
}

//
// Define a file work item structure that will be pushed to the context's
// file work list head.
//

typedef struct _FILE_WORK_ITEM {

    //
    // Singly-linked list entry for the structure.
    //

    SLIST_ENTRY ListEntry;

    //
    // Type of work requested.
    //

    FILE_WORK_ID FileWorkId;

    ULONG Padding[3];

} FILE_WORK_ITEM;
typedef FILE_WORK_ITEM *PFILE_WORK_ITEM;

typedef
BOOLEAN
(NTAPI DESTROY_PERFECT_HASH_TABLE)(
    _Pre_notnull_ _Post_satisfies_(*PerfectHashTablePointer == 0)
    PPERFECT_HASH_TABLE *PerfectHashTablePointer,
    _In_opt_ PBOOLEAN IsProcessTerminating
    );
typedef DESTROY_PERFECT_HASH_TABLE *PDESTROY_PERFECT_HASH_TABLE;
DESTROY_PERFECT_HASH_TABLE DestroyPerfectHashTable;

//
// Function typedefs for private functions.
//

//
// Each algorithm implements a creation routine that matches the following
// signature.  It is called by CreatePerfectHashTable() after it has done all
// the initial heavy-lifting (e.g. parameter validation, table allocation and
// initialization), and thus, has a much simpler function signature.
//

typedef
_Check_return_
_Success_(return != 0)
BOOLEAN
(NTAPI CREATE_PERFECT_HASH_TABLE_IMPL)(
    _Inout_ PPERFECT_HASH_TABLE Table
    );
typedef CREATE_PERFECT_HASH_TABLE_IMPL *PCREATE_PERFECT_HASH_TABLE_IMPL;

//
// For each algorithm, declare the creation impl routine.  These are gathered
// in an array named CreationRoutines[] (see PerfectHashTableConstants.[ch]).
//

CREATE_PERFECT_HASH_TABLE_IMPL CreatePerfectHashTableImplChm01;

//
// Likewise, each algorithm implements a loader routine that matches the
// following signature.  It is called by LoadPerfectHashTable() after it
// has done the initial heavy-lifting (e.g. parameter validation, table
// allocation and initialization), and, thus, has a much simpler function
// signature.
//

typedef
_Check_return_
_Success_(return != 0)
BOOLEAN
(NTAPI LOAD_PERFECT_HASH_TABLE_IMPL)(
    _Inout_ PPERFECT_HASH_TABLE Table
    );
typedef LOAD_PERFECT_HASH_TABLE_IMPL *PLOAD_PERFECT_HASH_TABLE_IMPL;

//
// For each algorithm, declare the loader impl routine.  These are gathered
// in an array named LoaderRoutines[] (see PerfectHashTableConstants.[ch]).
//

LOAD_PERFECT_HASH_TABLE_IMPL LoadPerfectHashTableImplChm01;

//
// For each algorithm, declare the index impl routine.  These are gathered in an
// array named LookupIndexRoutines[] (see PerfectHashTableConstants.[ch]).
//

PERFECT_HASH_TABLE_INDEX PerfectHashTableIndexImplChm01;

//
// For each algorithm, declare fast-index impl routines.  These differ from the
// normal index routines in that they inline the hash and mask logic (for a
// given hash and mask combo), removing the vtbl overhead.
//

PERFECT_HASH_TABLE_INDEX
    PerfectHashTableFastIndexImplChm01Crc32RotateHashAndMask;

PERFECT_HASH_TABLE_INDEX PerfectHashTableFastIndexImplChm01JenkinsHashAndMask;

//
// Define a helper structure for capturing fast index routines for a subset
// of hash/mask combinations.  These routines inline the hashing and masking
// such that they're essentially leaf entries, and avoid the COM vtbl overhead.
// This is used by our constants module.
//

typedef struct _PERFECT_HASH_TABLE_FAST_INDEX_TUPLE {
    PERFECT_HASH_TABLE_ALGORITHM_ID AlgorithmId;
    PERFECT_HASH_TABLE_HASH_FUNCTION_ID HashFunctionId;
    PERFECT_HASH_TABLE_MASK_FUNCTION_ID MaskFunctionId;
    ULONG Unused;
    PPERFECT_HASH_TABLE_INDEX FastIndex;
} PERFECT_HASH_TABLE_FAST_INDEX_TUPLE;
typedef PERFECT_HASH_TABLE_FAST_INDEX_TUPLE
      *PPERFECT_HASH_TABLE_FAST_INDEX_TUPLE;
typedef const PERFECT_HASH_TABLE_FAST_INDEX_TUPLE
           *PCPERFECT_HASH_TABLE_FAST_INDEX_TUPLE;

//
// Symbol loader helpers.
//

typedef _Null_terminated_ CONST CHAR *PCSZ;

typedef
_Check_return_
_Success_(return != 0)
BOOLEAN
(LOAD_SYMBOLS)(
    _In_count_(NumberOfSymbolNames) CONST PCSZ *SymbolNameArray,
    _In_ ULONG NumberOfSymbolNames,
    _In_count_(NumberOfSymbolAddresses) PULONG_PTR SymbolAddressArray,
    _In_ ULONG NumberOfSymbolAddresses,
    _In_ HMODULE Module,
    _In_ PRTL_BITMAP FailedSymbols,
    _Out_ PULONG NumberOfResolvedSymbolsPointer
    );
typedef LOAD_SYMBOLS *PLOAD_SYMBOLS;

typedef
_Check_return_
_Success_(return != 0)
BOOLEAN
(LOAD_SYMBOLS_FROM_MULTIPLE_MODULES)(
    _In_count_(NumberOfSymbolNames) CONST PCSZ *SymbolNameArray,
    _In_ ULONG NumberOfSymbolNames,
    _In_count_(NumberOfSymbolAddresses) PULONG_PTR SymbolAddressArray,
    _In_ ULONG NumberOfSymbolAddresses,
    _In_count_(NumberOfModules) HMODULE *ModuleArray,
    _In_ USHORT NumberOfModules,
    _In_ PRTL_BITMAP FailedSymbols,
    _Out_ PULONG NumberOfResolvedSymbolsPointer
    );
typedef LOAD_SYMBOLS_FROM_MULTIPLE_MODULES *PLOAD_SYMBOLS_FROM_MULTIPLE_MODULES;

//
// Exception helpers.
//

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
(NTAPI SET_C_SPECIFIC_HANDLER)(
    _In_ P__C_SPECIFIC_HANDLER Handler
    );
typedef SET_C_SPECIFIC_HANDLER *PSET_C_SPECIFIC_HANDLER;

typedef
VOID
(__cdecl __SECURITY_INIT_COOKIE)(
    VOID
    );
typedef __SECURITY_INIT_COOKIE *P__SECURITY_INIT_COOKIE;

//
// Inline helper functions.
//

#define MAX_RDRAND_RETRY_COUNT 10

FORCEINLINE
BOOLEAN
GetRandomSeeds(
    _Inout_ PULARGE_INTEGER Output,
    _Inout_opt_ PULARGE_INTEGER Cycles,
    _Inout_opt_ PULONG Attempts
    )
/*++

Routine Description:

    Generates a 64-bit random seed using the rdrand64 intrinisic.

Arguments:

    Output - Supplies a pointer to a ULARGE_INTEGER structure that will receive
        the random seed value.

    Cycles - Optionally supplies a pointer to a variable that will receive the
        approximate number of CPU cycles that were required in order to fulfil
        the random seed request.

        N.B. This calls __rdtsc() before and after the __rdseed64_step() call.
             If the pointer is NULL, __rdtsc() is not called either before or
             after.

    Attempts - Optionally supplies the address of a variable that receives the
        number of attempts it took before __rdseed64_step() succeeded.  (This
        is bound by the MAX_RDRAND_RETRY_COUNT constant.)

Return Value:

    TRUE on success, FALSE on failure.

--*/
{
    ULONG Index;
    BOOLEAN Success = FALSE;
    ULARGE_INTEGER Start;
    ULARGE_INTEGER End;

    if (ARGUMENT_PRESENT(Cycles)) {
        Start.QuadPart = ReadTimeStampCounter();
    }

    for (Index = 0; Index < MAX_RDRAND_RETRY_COUNT; Index++) {
        if (_rdseed64_step(&Output->QuadPart)) {
            Success = TRUE;
            break;
        }
        YieldProcessor();
    }

    if (ARGUMENT_PRESENT(Cycles)) {
        End.QuadPart = ReadTimeStampCounter();
        Cycles->QuadPart = End.QuadPart - Start.QuadPart;
    }

    if (ARGUMENT_PRESENT(Attempts)) {
        *Attempts = Index + 1;
    }

    return Success;
}

FORCEINLINE
VOID
GetRandomSeedsBlocking(
    _Inout_ PULARGE_INTEGER Output
    )
/*++

Routine Description:

    Calls __rdseed64_step() in a loop until it returns successfully.

Arguments:

    Output - Supplies a pointer to a ULARGE_INTEGER structure that will receive
        the random seed value.

Return Value:

    None.

--*/
{
    while (!_rdseed64_step(&Output->QuadPart)) {
        YieldProcessor();
    }
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
