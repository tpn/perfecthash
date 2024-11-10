/*++

Copyright (c) 2018-2024 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashPrivate.h

Abstract:

    This is the private header file for the PerfectHash component.  It defines
    function typedefs and function declarations for all major (i.e. not local to
    the module) functions available for use by individual modules within this
    component.

--*/

#ifndef PH_CUDA
#ifndef _PERFECT_HASH_INTERNAL_BUILD
#error PerfectHashPrivate.h being included but _PERFECT_HASH_INTERNAL_BUILD not set.
#endif
#endif

#pragma once

#ifndef PH_CUDA
#include "stdafx.h"
#endif

#define PERFECT_HASH_KEY_SIZE_IN_BYTES 4

//
// A handle to the PerfectHash.dll module will be captured in this variable
// via the DLL_PROCESS_ATTACH message.  This is required in order for proper
// operation of FormatMessage() when specifying FORMAT_MESSAGE_FROM_HMODULE and
// using our own internal error codes.  Additionally, MODULEINFO will be saved
// to PerfectHashModuleInfo.
//

#if (defined PH_WINDOWS && !defined PH_CUDA)
extern HMODULE PerfectHashModule;
extern MODULEINFO PerfectHashModuleInfo;
#endif

//
// Components can check this variable to determine if Ctrl-C has been pressed.
//

extern volatile ULONG CtrlCPressed;

//
// Define a helper macro for validating flags passed as parameters to routines.
//

#define VALIDATE_FLAGS(Name, Upper, Type)                        \
    if (ARGUMENT_PRESENT(Name##FlagsPointer)) {                  \
        if (FAILED(IsValid##Name##Flags(Name##FlagsPointer))) {  \
            return PH_E_INVALID_##Upper##_FLAGS;                 \
        } else {                                                 \
            Name##Flags.As##Type = Name##FlagsPointer->As##Type; \
        }                                                        \
    } else {                                                     \
        Name##Flags.As##Type = 0;                                \
    }

//
// Define a helper macro for releasing references to COM objects and clearing
// the associated pointer.  Typically used in Rundown() routines.  Eliminates
// use-after-free class of bugs.
//

#define RELEASE(Name)              \
    if (ARGUMENT_PRESENT(Name)) {  \
        Name->Vtbl->Release(Name); \
        Name = NULL;               \
    }

//
// Metadata about a perfect hash table is stored in an NTFS stream named :Info
// that is tacked onto the end of the perfect hash table's file name.  Define
// a structure, TABLE_INFO_ON_DISK, that literally represents the on-disk
// layout of this metadata.  Each algorithm implementation must write out an
// info record that conforms with this common header.  They are free to extend
// it with additional details.
//

typedef union _TABLE_INFO_ON_DISK_FLAGS {

    struct {

        //
        // When set, indicates the 16-bit hash/assigned infrastructure is
        // active.
        //

        ULONG UsingAssigned16:1;

        //
        // Unused bits.
        //

        ULONG Unused:31;

    };

    LONG AsLong;
    ULONG AsULong;

} TABLE_INFO_ON_DISK_FLAGS;
C_ASSERT(sizeof(TABLE_INFO_ON_DISK_FLAGS) == sizeof(ULONG));

typedef struct _Struct_size_bytes_(SizeOfStruct) _TABLE_INFO_ON_DISK {

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

    TABLE_INFO_ON_DISK_FLAGS Flags;

    //
    // Algorithm that was used.
    //

    PERFECT_HASH_ALGORITHM_ID AlgorithmId;

    //
    // Hash function that was used.
    //

    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId;

    //
    // Masking type.
    //

    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId;

    //
    // Size of an individual key element, in bytes.
    //

    ULONG KeySizeInBytes;

    //
    // If downsizing occurred, original size of a key in bytes.
    //

    ULONG OriginalKeySizeInBytes;

    //
    // The concurrency level used to generate the hash.
    //

    ULONG Concurrency;

    //
    // C data type of the table data/assigned array.
    //

    TYPE TableDataArrayType;

    //
    // C data type of the table values array.  Currently unused.
    //

    TYPE TableValuesArrayType;

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
    ULONG Seed4;
    ULONG Seed5;
    ULONG Seed6;
    ULONG Seed7;

    union {
        ULONG Seed8;
        ULONG LastSeed;
    };

    //
    // Size of an individual assigned element, in bytes.
    //

    ULONG AssignedElementSizeInBytes;

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

} TABLE_INFO_ON_DISK;
typedef TABLE_INFO_ON_DISK *PTABLE_INFO_ON_DISK;

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
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI CREATE_PERFECT_HASH_TABLE_IMPL)(
    _Inout_ PPERFECT_HASH_TABLE Table
    );
typedef CREATE_PERFECT_HASH_TABLE_IMPL *PCREATE_PERFECT_HASH_TABLE_IMPL;

//
// For each algorithm, declare the creation impl routine.  These are gathered
// in an array named CreationRoutines[] (see PerfectHashTableConstants.[ch]).
//

CREATE_PERFECT_HASH_TABLE_IMPL CreatePerfectHashTableImplChm01;
#ifdef PH_USE_CUDA
CREATE_PERFECT_HASH_TABLE_IMPL CreatePerfectHashTableImplChm02;
#endif

//
// Likewise, each algorithm implements a loader routine that matches the
// following signature.  It is called by LoadPerfectHashTable() after it
// has done the initial heavy-lifting (e.g. parameter validation, table
// allocation and initialization), and, thus, has a much simpler function
// signature.
//

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI LOAD_PERFECT_HASH_TABLE_IMPL)(
    _Inout_ PPERFECT_HASH_TABLE Table
    );
typedef LOAD_PERFECT_HASH_TABLE_IMPL *PLOAD_PERFECT_HASH_TABLE_IMPL;

//
// For each algorithm, declare the loader impl routine.  These are gathered
// in an array named LoaderRoutines[] (see PerfectHashTableConstants.[ch]).
//

LOAD_PERFECT_HASH_TABLE_IMPL LoadPerfectHashTableImplChm01;
#ifdef PH_USE_CUDA
LOAD_PERFECT_HASH_TABLE_IMPL LoadPerfectHashTableImplChm02;
#endif

//
// For each algorithm, declare the index impl routine.  These are gathered in an
// array named LookupIndexRoutines[] (see PerfectHashTableConstants.[ch]).
//

PERFECT_HASH_TABLE_INDEX PerfectHashTableIndexImplChm01;
PERFECT_HASH_TABLE_INDEX PerfectHashTableIndexImplChm02;

//
// For each algorithm, declare fast-index impl routines.  These differ from the
// normal index routines in that they inline the hash and mask logic (for a
// given hash and mask combo), removing the vtbl overhead.
//

PERFECT_HASH_TABLE_INDEX
    PerfectHashTableFastIndexImplChm01Crc32Rotate15HashAndMask;

PERFECT_HASH_TABLE_INDEX
    PerfectHashTableFastIndexImplChm01Crc32RotateXHashAndMask;

PERFECT_HASH_TABLE_INDEX
    PerfectHashTableFastIndexImplChm01Crc32RotateXYHashAndMask;

PERFECT_HASH_TABLE_INDEX
    PerfectHashTableFastIndexImplChm01Crc32RotateWXYZHashAndMask;

PERFECT_HASH_TABLE_INDEX PerfectHashTableFastIndexImplChm01JenkinsHashAndMask;

//
// Define a helper structure for capturing fast index routines for a subset
// of hash/mask combinations.  These routines inline the hashing and masking
// such that they're essentially leaf entries, and avoid the COM vtbl overhead.
// This is used by our constants module.
//

//
// Disable warning C4820:
//      '<anonymous-tag>': '4' bytes padding added after data member ...
//

#pragma warning(push)
#pragma warning(disable: 4820)
typedef struct _PERFECT_HASH_TABLE_FAST_INDEX_TUPLE {
    PERFECT_HASH_ALGORITHM_ID AlgorithmId;
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId;
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId;
    PPERFECT_HASH_TABLE_INDEX FastIndex;
} PERFECT_HASH_TABLE_FAST_INDEX_TUPLE;
typedef PERFECT_HASH_TABLE_FAST_INDEX_TUPLE
      *PPERFECT_HASH_TABLE_FAST_INDEX_TUPLE;
typedef const PERFECT_HASH_TABLE_FAST_INDEX_TUPLE
           *PCPERFECT_HASH_TABLE_FAST_INDEX_TUPLE;

//
// As above, but for raw C strings of the Index() implementation.
//

typedef struct _PERFECT_HASH_TABLE_INDEX_IMPL_STRING_TUPLE {
    PERFECT_HASH_ALGORITHM_ID AlgorithmId;
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId;
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId;
    PCSTRING RawCString;
} PERFECT_HASH_TABLE_INDEX_IMPL_STRING_TUPLE;
typedef PERFECT_HASH_TABLE_INDEX_IMPL_STRING_TUPLE
      *PPERFECT_HASH_TABLE_INDEX_IMPL_STRING_TUPLE;
typedef const PERFECT_HASH_TABLE_INDEX_IMPL_STRING_TUPLE
           *PCPERFECT_HASH_TABLE_INDEX_IMPL_STRING_TUPLE;

#pragma warning(pop)

#ifndef __CUDA_ARCH__

//
// Symbol loader helpers.
//

typedef
_Must_inspect_result_
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
_Must_inspect_result_
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

#ifdef PH_WINDOWS
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
#endif // PH_WINDOWS


FORCEINLINE
BOOLEAN
DoesErrorCodeWantAlgoHashMaskTableAppended(
    _In_ ULONG ErrorCode
    )
{
    HRESULT Code;
    BOOLEAN Result;

    Code = (HRESULT)ErrorCode;

    //
    // We append the algo/hash/mask table text for the usage string.
    //

    Result = (Code == PH_MSG_PERFECT_HASH_USAGE_CONTINUED_1);

    return Result;
}


#endif // ifndef __CUDA_ARCH__

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
