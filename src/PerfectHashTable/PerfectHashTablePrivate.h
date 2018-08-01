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

#ifndef ASSERT
#define ASSERT(Condition) \
    if (!(Condition)) {   \
        __debugbreak();   \
    }
#endif

//
//
// Cap the maximum key set size we're willing to process.
//

#define MAXIMUM_NUMBER_OF_KEYS 500000

//
// Define the PERFECT_HASH_TABLE_KEYS_FLAGS structure.
//

typedef union _PERFECT_HASH_TABLE_FLAGS_KEYS {
    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, indicates the keys were mapped using large pages.
        //

        ULONG MappedWithLargePages:1;

        //
        // Unused bits.
        //

        ULONG Unused:31;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_TABLE_KEYS_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_TABLE_KEYS_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_TABLE_KEYS_FLAGS *PPERFECT_HASH_TABLE_KEYS_FLAGS;

//
// Define the PERFECT_HASH_TABLE_KEYS structure.
//

typedef struct _Struct_size_bytes_(SizeOfStruct) _PERFECT_HASH_TABLE_KEYS {

    //
    // Reserve a slot for a vtable.  Currently unused.
    //

    PPVOID Vtbl;

    //
    // Size of the structure, in bytes.
    //

    _Field_range_(==, sizeof(struct _PERFECT_HASH_TABLE_KEYS))
        ULONG SizeOfStruct;

    //
    // Flags.
    //

    PERFECT_HASH_TABLE_KEYS_FLAGS Flags;

    //
    // Pointer to an initialized RTL structure.
    //

    PRTL Rtl;

    //
    // Pointer to an initialized ALLOCATOR structure.
    //

    PALLOCATOR Allocator;

    //
    // Pointer to the API structure in use.
    //

    PPERFECT_HASH_TABLE_ANY_API AnyApi;

    //
    // Number of keys in the mapping.
    //

    ULARGE_INTEGER NumberOfElements;

    //
    // Handle to the underlying keys file.
    //

    HANDLE FileHandle;

    //
    // Handle to the memory mapping for the keys file.
    //

    HANDLE MappingHandle;

    //
    // Base address of the memory map.
    //

    union {
        PVOID BaseAddress;
        PULONG Keys;
    };

    //
    // Fully-qualified, NULL-terminated path of the source keys file.
    //

    UNICODE_STRING Path;

} PERFECT_HASH_TABLE_KEYS;
typedef PERFECT_HASH_TABLE_KEYS *PPERFECT_HASH_TABLE_KEYS;

//
// Algorithms are required to register a callback routine with the perfect hash
// table context that matches the following signature.  This routine will be
// called for each work item it pushes to the context's main threadpool, with
// a pointer to the SLIST_ENTRY that was popped off the list.
//

typedef
VOID
(CALLBACK PERFECT_HASH_TABLE_MAIN_WORK_CALLBACK)(
    _In_ PTP_CALLBACK_INSTANCE Instance,
    _In_ PPERFECT_HASH_TABLE_CONTEXT Context,
    _In_ PSLIST_ENTRY ListEntry
    );
typedef PERFECT_HASH_TABLE_MAIN_WORK_CALLBACK
      *PPERFECT_HASH_TABLE_MAIN_WORK_CALLBACK;

//
// Additionally, algorithms can register a callback routine for performing
// file-oriented operations in the main threadpool (not directly related to
// graph solving).
//

typedef
VOID
(CALLBACK PERFECT_HASH_TABLE_FILE_WORK_CALLBACK)(
    _In_ PTP_CALLBACK_INSTANCE Instance,
    _In_ PPERFECT_HASH_TABLE_CONTEXT Context,
    _In_ PSLIST_ENTRY ListEntry
    );
typedef PERFECT_HASH_TABLE_FILE_WORK_CALLBACK
      *PPERFECT_HASH_TABLE_FILE_WORK_CALLBACK;


//
// Define a runtime context to encapsulate threadpool resources.  This is
// passed to CreatePerfectHashTable() and allows for algorithms to search for
// perfect hash solutions in parallel.
//

typedef union _PERFECT_HASH_TABLE_CONTEXT_FLAGS {
    struct {
        ULONG Unused:32;
    };
    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_TABLE_CONTEXT_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_TABLE_CONTEXT_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_TABLE_CONTEXT_FLAGS *PPERFECT_HASH_TABLE_CONTEXT_FLAGS;

typedef struct _Struct_size_bytes_(SizeOfStruct) _PERFECT_HASH_TABLE_CONTEXT {

    //
    // Size of the structure, in bytes.
    //

    _Field_range_(==, sizeof(struct _PERFECT_HASH_TABLE_CONTEXT))
        ULONG SizeOfStruct;

    //
    // Flags.
    //

    PERFECT_HASH_TABLE_CONTEXT_FLAGS Flags;

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

    //
    // Pointer to an initialized RTL structure.
    //

    PRTL Rtl;

    //
    // Pointer to an initialized allocator.
    //

    PALLOCATOR Allocator;

    //
    // Pointer to the API structure in use.
    //

    PPERFECT_HASH_TABLE_ANY_API AnyApi;

    //
    // Pointer to the active perfect hash table.
    //

    struct _PERFECT_HASH_TABLE *Table;

    //
    // The highest number of deleted edges count encountered by a worker thread.
    // This is useful when debugging a poorly performing hash/mask combo that is
    // failing to find a solution.
    //

    volatile ULONG HighestDeletedEdgesCount;

    //
    // The number of attempts we'll make at trying to solve the graph before
    // giving up and resizing with a larger underlying table.
    //

    ULONG ResizeTableThreshold;

    //
    // Limit on how many times a resize will be kicked off.
    //

    ULONG ResizeLimit;

    //
    // Define the events used to communicate various internal state changes
    // between the CreatePerfectHashTable() function and the algorithm-specific
    // creation routine.
    //
    // N.B. All of these events are created with the manual reset flag set to
    //      TRUE, such that they stay signalled even after they have satisfied
    //      a wait.
    //

    //
    // A global "shutdown" event handle that threads can query to determine
    // whether or not they should continue processing at various internal
    // checkpoints.
    //

    union {
        HANDLE ShutdownEvent;
        PVOID FirstEvent;
    };

    //
    // This event will be set if an algorithm was successful in finding a
    // perfect hash.  Either it or the FailedEvent will be set; never both.
    //

    HANDLE SucceededEvent;

    //
    // This event will be set if an algorithm failed to find a perfect hash
    // solution.  This may be due to the algorithm exhausting all possible
    // options, hitting a time limit, or potentially as a result of being
    // forcibly terminated or some other internal error.  It will never be
    // set if SucceededEvent is also set.
    //

    HANDLE FailedEvent;

    //
    // The following event is required to be set by an algorithm's creation
    // routine upon completion (regardless of success or failure).  This event
    // is waited upon by the CreatePerfectHashTable() function, and thus, is
    // critical in synchronizing the execution of parallel perfect hash solution
    // finding.
    //

    HANDLE CompletedEvent;

    //
    // The following event is set when a worker thread detects that the number
    // of attempts has exceeded a specified threshold, and that the main thread
    // should cancel the current attempts and try again with a larger vertex
    // table size.
    //

    HANDLE TryLargerTableSizeEvent;

    //
    // The following event is set when a worker thread has completed preparing
    // the underlying backing file in order for the solved graph to be persisted
    // to disk.
    //

    HANDLE PreparedFileEvent;

    //
    // The following event is set by the main thread when it has completed
    // verification of the solved graph.  It is used to signal to the save
    // file worker that verification has finished such that cycle counts can
    // be captured in order to calculate the number of cycles and microseconds
    // it took to verify the graph.
    //

    HANDLE VerifiedEvent;

    //
    // The following event is set when a worker thread has completed saving the
    // solved graph to disk.
    //

    union {
        HANDLE SavedFileEvent;
        PVOID LastEvent;
    };

    //
    // N.B. All events are created as named events, using the random object
    //      name generation helper Rtl->CreateRandomObjectNames().  This will
    //      fill out an array of PUNICODE_STRING pointers.  The next field
    //      points to the first element of that array.  Subsequent fields
    //      capture various book-keeping items about the random object names
    //      allocation (provided by the Rtl routine).
    //

    PUNICODE_STRING ObjectNames;
    PPUNICODE_STRING ObjectNamesPointerArray;
    PWSTR ObjectNamesWideBuffer;
    ULONG SizeOfObjectNamesWideBuffer;
    ULONG NumberOfObjects;

    //
    // Number of attempts made by the algorithm to find a solution.
    //

    volatile ULONGLONG Attempts;

    //
    // Counters used for capturing performance information.  We capture both a
    // cycle count, using __rdtsc(), plus a "performance counter" count, via
    // QueryPerformanceCounter().  The former provides a finer resolution, but
    // can't be used to calculate elapsed microseconds due to turbo boost and
    // variable frequencies.  The latter provides a coarser resolution, but
    // can be used to convert into elapsed microseconds (via the frequency,
    // also captured below).
    //

    LARGE_INTEGER Frequency;

    //
    // Capture the time required to solve the perfect hash table.  This is not
    // a sum of all cycles consumed by all worker threads; it is the cycles
    // consumed between the "main" thread (i.e. the CreatePerfectHashTable()
    // impl routine (CreatePerfectHashTableImplChm01())) dispatching parallel
    // work to the threadpool, and a solution being found.
    //

    ULARGE_INTEGER SolveStartCycles;
    LARGE_INTEGER SolveStartCounter;

    ULARGE_INTEGER SolveEndCycles;
    LARGE_INTEGER SolveEndCounter;

    ULARGE_INTEGER SolveElapsedCycles;
    ULARGE_INTEGER SolveElapsedMicroseconds;

    //
    // Capture the time required to verify the solution.  This involves walking
    // the entire key set, applying the perfect hash function to derive an index
    // into the Assigned array, and verifying that we only saw each index value
    // at most once.
    //
    // This is a reasonably good measure of the combined performance of the
    // chosen hash and mask algorithm, with lower cycles and counter values
    // indicating better performance.
    //

    ULARGE_INTEGER VerifyStartCycles;
    LARGE_INTEGER VerifyStartCounter;

    ULARGE_INTEGER VerifyEndCycles;
    LARGE_INTEGER VerifyEndCounter;

    ULARGE_INTEGER VerifyElapsedCycles;
    ULARGE_INTEGER VerifyElapsedMicroseconds;

    //
    // Capture the time required to prepare the backing .pht1 file in the file
    // work threadpool.
    //

    ULARGE_INTEGER PrepareFileStartCycles;
    LARGE_INTEGER PrepareFileStartCounter;

    ULARGE_INTEGER PrepareFileEndCycles;
    LARGE_INTEGER PrepareFileEndCounter;

    ULARGE_INTEGER PrepareFileElapsedCycles;
    ULARGE_INTEGER PrepareFileElapsedMicroseconds;

    //
    // Capture the time required to save the final Assigned array to the backing
    // file prepared in an earlier step.  This is also dispatched to the file
    // work thread pool, and consists of a memory copy from the assigned array
    // of the graph to the base address of the backing file's memory map, then
    // flushing the map, unmapping it, closing the section, and closing the
    // file.
    //

    ULARGE_INTEGER SaveFileStartCycles;
    LARGE_INTEGER SaveFileStartCounter;

    ULARGE_INTEGER SaveFileEndCycles;
    LARGE_INTEGER SaveFileEndCounter;

    ULARGE_INTEGER SaveFileElapsedCycles;
    ULARGE_INTEGER SaveFileElapsedMicroseconds;

    //
    // Number of failed attempts at solving the graph across all threads.
    //

    volatile ULONGLONG FailedAttempts;

    //
    // The main threadpool callback environment, used for solving perfect hash
    // solutions in parallel.
    //

    TP_CALLBACK_ENVIRON MainCallbackEnv;
    PTP_CLEANUP_GROUP MainCleanupGroup;
    PTP_POOL MainThreadpool;
    PTP_WORK MainWork;
    SLIST_HEADER MainWorkListHead;
    ULONG MinimumConcurrency;
    ULONG MaximumConcurrency;

    //
    // The algorithm is responsible for registering an appropriate callback
    // for main thread work items in this next field.
    //

    PPERFECT_HASH_TABLE_MAIN_WORK_CALLBACK MainWorkCallback;

    //
    // A threadpool for offloading file operations.
    //

    TP_CALLBACK_ENVIRON FileCallbackEnv;
    PTP_CLEANUP_GROUP FileCleanupGroup;
    PTP_POOL FileThreadpool;
    PTP_WORK FileWork;
    SLIST_HEADER FileWorkListHead;

    //
    // Provide a means for file work callbacks to indicate an error back to
    // the creation routine by incrementing the following counter.
    //

    volatile ULONG FileWorkErrors;
    volatile ULONG FileWorkLastError;

    //
    // The algorithm is responsible for registering an appropriate callback
    // for file work threadpool work items in this next field.
    //

    PPERFECT_HASH_TABLE_FILE_WORK_CALLBACK FileWorkCallback;

    //
    // If a threadpool worker thread finds a perfect hash solution, it will
    // enqueue a "Finished!"-type work item to a separate threadpool, captured
    // by the following callback environment.  This allows for a separate
    // threadpool worker to schedule the cancellation of other in-progress
    // and outstanding perfect hash solution attempts without deadlocking.
    //
    // This threadpool environment is serviced by a single thread.
    //
    // N.B. This cleanup only refers to the main graph solving thread pool.
    //      The file threadpool is managed by the implicit lifetime of the
    //      algorithm's creation routine (e.g. CreatePerfectHashTableImplChm01).
    //

    TP_CALLBACK_ENVIRON FinishedCallbackEnv;
    PTP_POOL FinishedThreadpool;
    PTP_WORK FinishedWork;
    SLIST_HEADER FinishedWorkListHead;

    //
    // If a worker thread successfully finds a perfect hash solution, it will
    // push its solution to the FinishedListHead above, then submit a finished
    // work item via SubmitThreadpoolWork(Context->FinishedWork).
    //
    // This callback will be processed by the finished group above, and provides
    // a means for that thread to set the ShutdownEvent and cancel outstanding
    // main work callbacks.
    //
    // N.B. Although we only need one solution, we don't prevent multiple
    //      successful solutions from being pushed to the FinishedListHead.
    //      Whatever the first solution is that the finished callback pops
    //      off that list is the solution that wins.
    //

    volatile ULONGLONG FinishedCount;

    //
    // Similar to the Finished group above, provide an Error group that also
    // consists of a single thread.  If a main threadpool worker thread runs
    // into a fatal error that requires termination of all in-progress and
    // outstanding threadpool work items, it can just dispatch a work item
    // to this particular pool (e.g. SubmitThreadpoolWork(Context->ErrorWork)).
    //
    // There is no ErrorListHead as no error information is captured that needs
    // communicating back to a central location.
    //

    TP_CALLBACK_ENVIRON ErrorCallbackEnv;
    PTP_POOL ErrorThreadpool;
    PTP_WORK ErrorWork;

    //
    // An opaque pointer that can be used by the algorithm to stash additional
    // context.
    //

    PVOID AlgorithmContext;

    //
    // An opaque pointer that can be used by the hash function to stash
    // additional context.
    //

    PVOID HashFunctionContext;

    //
    // An opaque pointer to the winning solution (i.e. the solved graph).
    //

    PVOID SolvedContext;

} PERFECT_HASH_TABLE_CONTEXT;
typedef PERFECT_HASH_TABLE_CONTEXT *PPERFECT_HASH_TABLE_CONTEXT;

//
// Define helper macros for marking start/end points for the context's
// cycle/counter fields.  When starting, we put __rdtsc() last, and when
// stopping we put it first, as its resolution is more sensitive than the
// QueryPerformanceCounter() routine.
//

#define CONTEXT_START_TIMERS(Name)                           \
    QueryPerformanceCounter(&Context->##Name##StartCounter); \
    Context->##Name##StartCycles.QuadPart = __rdtsc()

#define CONTEXT_END_TIMERS(Name)                              \
    Context->##Name##EndCycles.QuadPart = __rdtsc();          \
    QueryPerformanceCounter(&Context->##Name##EndCounter);    \
    Context->##Name##ElapsedCycles.QuadPart = (               \
        Context->##Name##EndCycles.QuadPart -                 \
        Context->##Name##StartCycles.QuadPart                 \
    );                                                        \
    Context->##Name##ElapsedMicroseconds.QuadPart = (         \
        Context->##Name##EndCounter.QuadPart -                \
        Context->##Name##StartCounter.QuadPart                \
    );                                                        \
    Context->##Name##ElapsedMicroseconds.QuadPart *= 1000000; \
    Context->##Name##ElapsedMicroseconds.QuadPart /= (        \
        Context->Frequency.QuadPart                           \
    )

#define CONTEXT_SAVE_TIMERS_TO_HEADER(Name)                                    \
    Header->##Name##Cycles.QuadPart = Context->##Name##ElapsedCycles.QuadPart; \
    Header->##Name##Microseconds.QuadPart = (                                  \
        Context->##Name##ElapsedMicroseconds.QuadPart                          \
    )

//
// Forward definition of the hash table context destructor.
//

DESTROY_PERFECT_HASH_TABLE_CONTEXT DestroyPerfectHashTableContext;

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

    //
    // Our extended vtbl slot comes first, COM-style.
    //

    struct _PERFECT_HASH_TABLE_VTBL_EX *Vtbl;

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
    // Size of the structure, in bytes.
    //

    _Field_range_(==, sizeof(struct _PERFECT_HASH_TABLE)) ULONG SizeOfStruct;

    //
    // Flags.
    //

    PERFECT_HASH_TABLE_FLAGS Flags;

    //
    // Generic singly-linked list entry.
    //

    SLIST_ENTRY ListEntry;

    //
    // Pointer to an initialized RTL structure.
    //

    PRTL Rtl;

    //
    // Pointer to an initialized ALLOCATOR structure.
    //

    PALLOCATOR Allocator;

    //
    // Reference count.  Affected by AddRef() and Release().
    //

    volatile ULONG ReferenceCount;

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
    // If a caller provided the number of table elements as a parameter to the
    // CreatePerfectHashTable() function, that value will be captured here.  It
    // overrides the default sizing heuristics.  (If non-zero, it will be at
    // least equal to or greater than the number of keys.)
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
    // During creation, a large bitmap is created to cover the entire range of
    // possible ULONG keys.  This is used to ensure no duplicate keys appear in
    // the input key set, and also assists in debugging.
    //

    RTL_BITMAP KeysBitmap;

} PERFECT_HASH_TABLE;
typedef PERFECT_HASH_TABLE *PPERFECT_HASH_TABLE;

//
// Declare the AddRef and Release functions for the hash table.
//

PERFECT_HASH_TABLE_ADD_REF PerfectHashTableAddRef;
PERFECT_HASH_TABLE_RELEASE PerfectHashTableRelease;

//
// Define the seeded hash routine, which explicitly takes an array of seeds.
// This is used by the solving routines when attempting to create a perfect
// hash table, and seed values are being generated randomly.  It is not used
// for loaded tables, as those have hash values that can be accessed easily
// via the Table->Header->Seed fields.
//

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

//
// Extend the public vtable with internal methods we need for the hash table.
//

typedef struct _PERFECT_HASH_TABLE_VTBL_EX {

    //
    // Inline PERFECT_HASH_TABLE_VTBL.
    //

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

    //
    // Begin extended functions.
    //

    PPERFECT_HASH_TABLE_SEEDED_HASH SeededHash;
    PPERFECT_HASH_TABLE_INDEX FastIndex;
    PPERFECT_HASH_TABLE_INDEX SlowIndex;

} PERFECT_HASH_TABLE_VTBL_EX;
typedef PERFECT_HASH_TABLE_VTBL_EX *PPERFECT_HASH_TABLE_VTBL_EX;

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
// Forward definitions of our generic (i.e. non-algorithm specific) routines
// for insert, index, lookup, hashing and masking.
//

PERFECT_HASH_TABLE_INSERT PerfectHashTableInsert;
PERFECT_HASH_TABLE_LOOKUP PerfectHashTableLookup;
PERFECT_HASH_TABLE_DELETE PerfectHashTableDelete;
PERFECT_HASH_TABLE_INDEX PerfectHashTableIndex;

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

    //
    // Pad out to an 8 byte boundary.
    //

    ULONG Unused;

} FILE_WORK_ITEM;
typedef FILE_WORK_ITEM *PFILE_WORK_ITEM;

//
// Private function definition for destroying a hash table.  We don't make
// this a public function as the CreatePerfectHashTable() does not return
// a table to the caller, and LoadPerfectHashTable() returns a vtbl pointer
// that we expect the caller to use AddRef()/Release() on correctly in order
// to manage lifetime.
//

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
// TLS-related structures and functions.
//

typedef struct _PERFECT_HASH_TABLE_TLS_CONTEXT {
    PVOID Unused;
} PERFECT_HASH_TABLE_TLS_CONTEXT;
typedef PERFECT_HASH_TABLE_TLS_CONTEXT *PPERFECT_HASH_TABLE_TLS_CONTEXT;

extern ULONG PerfectHashTableTlsIndex;

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
// Each algorithm implements a routine that returns the required size of the
// extended vtbl.
//

typedef
_Check_return_
USHORT
(NTAPI GET_VTBL_EX_SIZE)(
    VOID
    );
typedef GET_VTBL_EX_SIZE *PGET_VTBL_EX_SIZE;

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
    PPERFECT_HASH_TABLE_INDEX FastIndex;
} PERFECT_HASH_TABLE_FAST_INDEX_TUPLE;
typedef PERFECT_HASH_TABLE_FAST_INDEX_TUPLE
      *PPERFECT_HASH_TABLE_FAST_INDEX_TUPLE;
typedef const PERFECT_HASH_TABLE_FAST_INDEX_TUPLE
           *PCPERFECT_HASH_TABLE_FAST_INDEX_TUPLE;

//
// For each algorithm, declare a routine that returns the size of the vtbl used
// by that algorithm.
//

GET_VTBL_EX_SIZE GetVtblExSizeChm01;

//
// The PROCESS_ATTACH and PROCESS_ATTACH functions share the same signature.
//

typedef
_Check_return_
_Success_(return != 0)
(PERFECT_HASH_TABLE_TLS_FUNCTION)(
    _In_    HMODULE     Module,
    _In_    DWORD       Reason,
    _In_    LPVOID      Reserved
    );
typedef PERFECT_HASH_TABLE_TLS_FUNCTION *PPERFECT_HASH_TABLE_TLS_FUNCTION;

PERFECT_HASH_TABLE_TLS_FUNCTION PerfectHashTableTlsProcessAttach;
PERFECT_HASH_TABLE_TLS_FUNCTION PerfectHashTableTlsProcessDetach;

//
// Define TLS Get/Set context functions.
//

typedef
_Check_return_
_Success_(return != 0)
BOOLEAN
(PERFECT_HASH_TABLE_TLS_SET_CONTEXT)(
    _In_ struct _PERFECT_HASH_TABLE_CONTEXT *Context
    );
typedef PERFECT_HASH_TABLE_TLS_SET_CONTEXT *PPERFECT_HASH_TABLE_TLS_SET_CONTEXT;

typedef
_Check_return_
_Success_(return != 0)
struct _PERFECT_HASH_TABLE_CONTEXT *
(PERFECT_HASH_TABLE_TLS_GET_CONTEXT)(
    VOID
    );
typedef PERFECT_HASH_TABLE_TLS_GET_CONTEXT *PPERFECT_HASH_TABLE_TLS_GET_CONTEXT;

extern PERFECT_HASH_TABLE_TLS_SET_CONTEXT PerfectHashTableTlsSetContext;
extern PERFECT_HASH_TABLE_TLS_GET_CONTEXT PerfectHashTableTlsGetContext;

//
// Inline helper functions.
//

#define MAX_RDRAND_RETRY_COUNT 10

FORCEINLINE
BOOLEAN
GetRandomSeeds(
    _Out_ PULARGE_INTEGER Output,
    _Out_opt_ PULARGE_INTEGER Cycles,
    _Out_opt_ PULONG Attempts
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
    _Out_ PULARGE_INTEGER Output
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
