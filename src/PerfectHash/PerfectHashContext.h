/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashContext.h

Abstract:

    This is the private header file for the PERFECT_HASH_CONTEXT
    component of the perfect hash table library.  It defines the structure,
    and function pointer typedefs for the initialize and rundown functions.

--*/

#pragma once

#include "stdafx.h"

//
// Algorithms are required to register a callback routine with the perfect hash
// table context that matches the following signature.  This routine will be
// called for each work item it pushes to the context's main threadpool, with
// a pointer to the SLIST_ENTRY that was popped off the list.
//

typedef
VOID
(CALLBACK PERFECT_HASH_MAIN_WORK_CALLBACK)(
    _In_ PTP_CALLBACK_INSTANCE Instance,
    _In_ struct _PERFECT_HASH_CONTEXT *Context,
    _In_ PSLIST_ENTRY ListEntry
    );
typedef PERFECT_HASH_MAIN_WORK_CALLBACK
      *PPERFECT_HASH_MAIN_WORK_CALLBACK;

//
// Additionally, algorithms can register a callback routine for performing
// file-oriented operations in the main threadpool (not directly related to
// graph solving).
//

typedef
VOID
(CALLBACK PERFECT_HASH_FILE_WORK_CALLBACK)(
    _In_ PTP_CALLBACK_INSTANCE Instance,
    _In_ struct _PERFECT_HASH_CONTEXT *Context,
    _In_ PSLIST_ENTRY ListEntry
    );
typedef PERFECT_HASH_FILE_WORK_CALLBACK
      *PPERFECT_HASH_FILE_WORK_CALLBACK;


//
// Define a runtime context to encapsulate threadpool resources.  This is
// passed to CreatePerfectHash() and allows for algorithms to search for
// perfect hash solutions in parallel.
//

typedef union _PERFECT_HASH_CONTEXT_STATE {
    struct {

        //
        // When set, indicates that the context needs to be reset before
        // another CreateTable() is serviced.
        //

        ULONG NeedsReset:1;

        //
        // Unused bits.
        //

        ULONG Unused:31;
    };
    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_CONTEXT_STATE;
C_ASSERT(sizeof(PERFECT_HASH_CONTEXT_STATE) == sizeof(ULONG));
typedef PERFECT_HASH_CONTEXT_STATE *PPERFECT_HASH_CONTEXT_STATE;

DEFINE_UNUSED_FLAGS(PERFECT_HASH_CONTEXT);
typedef struct _Struct_size_bytes_(SizeOfStruct) _PERFECT_HASH_CONTEXT {

    COMMON_COMPONENT_HEADER(PERFECT_HASH_CONTEXT);

    //
    // The algorithm in use.
    //

    PERFECT_HASH_ALGORITHM_ID AlgorithmId;

    //
    // The masking type in use.
    //

    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId;

    //
    // The hash function in use.
    //

    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId;

    //
    // Pad out to an 8-byte boundary.
    //

    ULONG Padding;

    //
    // Fully-qualified path of the output directory.  Manipulated via the
    // SetOutputDirectory() and GetOutputDirectory() routines.
    //

    UNICODE_STRING OutputDirectory;

    //
    // Pointer to the active perfect hash table.
    //

    struct _PERFECT_HASH_TABLE *Table;

    //
    // An in-memory union of all possible on-disk table info representations.
    // This is used to capture table info prior to the :Info stream being
    // available.  The backing memory is stack-allocated in the algorithm's
    // create table routine.
    //

    union {
        struct _TABLE_INFO_ON_DISK *TableInfoOnDisk;
        struct _GRAPH_INFO_ON_DISK *GraphInfoOnDisk;
    };

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

    ULONG Padding2;

    //
    // Define the events used to communicate various internal state changes
    // between the CreatePerfectHash() function and the algorithm-specific
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
    // is waited upon by the CreatePerfectHash() function, and thus, is
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
    // The following event is set by the main thread when it has completed
    // verification of the solved graph.  It is used to signal to the save
    // file worker that verification has finished such that cycle counts can
    // be captured in order to calculate the number of cycles and microseconds
    // it took to verify the graph.
    //

    HANDLE VerifiedTableEvent;

    //
    // The following event is set when a worker thread has completed preparing
    // the underlying backing table file in order for the solved graph to be
    // persisted to disk.
    //

    HANDLE PreparedTableFileEvent;

    //
    // The following event is set when a worker thread has completed saving the
    // solved graph to disk.
    //

    HANDLE SavedTableFileEvent;

    //
    // The following event is set when a worker thread has completed preparing
    // the C source file containing table data.
    //

    HANDLE PreparedCSourceTableDataFileEvent;

    //
    // The following event is set when a worker thread has completed saving
    // the C source file containing table data.
    //

    HANDLE SavedCSourceTableDataFileEvent;

    //
    // The following event is set when a worker thread has completed preparing
    // the table's :Info stream.
    //

    HANDLE PreparedTableInfoStreamEvent;

    //
    // The following event is set when a worker thread has completed saving the
    // table's :Info stream to disk.
    //

    HANDLE SavedTableInfoStreamEvent;

    //
    // The following event is set when a worker thread has completed preparing
    // the C header file for the perfect hash table.
    //

    HANDLE PreparedCHeaderFileEvent;

    //
    // The following events are set when a worker thread has completed
    // preparing the relevant C source file for the perfect hash table.
    //

    HANDLE PreparedCSourceFileEvent;
    HANDLE PreparedCSourceKeysFileEvent;

    //
    // The following event is set when a worker thread has completed writing
    // the contents of the C header file once a perfect hash table solution has
    // been found.
    //

    HANDLE SavedCHeaderFileEvent;

    //
    // The following events are set when a worker thread has completed writing
    // the contents of the C source files once a perfect hash table solution
    // has been found.
    //

    HANDLE SavedCSourceFileEvent;

    union {
        HANDLE SavedCSourceKeysFileEvent;
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

    volatile LONGLONG Attempts;

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
    // N.B. These stats should probably be relocated into a separate structure.
    //

    //
    // Capture the time required to solve the perfect hash table.  This is not
    // a sum of all cycles consumed by all worker threads; it is the cycles
    // consumed between the "main" thread (i.e. the CreatePerfectHash()
    // impl routine (CreatePerfectHashImplChm01())) dispatching parallel
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

    ULARGE_INTEGER PrepareTableFileStartCycles;
    LARGE_INTEGER PrepareTableFileStartCounter;

    ULARGE_INTEGER PrepareTableFileEndCycles;
    LARGE_INTEGER PrepareTableFileEndCounter;

    ULARGE_INTEGER PrepareTableFileElapsedCycles;
    ULARGE_INTEGER PrepareTableFileElapsedMicroseconds;

    //
    // Capture the time required to save the final Assigned array to the backing
    // file prepared in an earlier step.  This is also dispatched to the file
    // work thread pool, and consists of a memory copy from the assigned array
    // of the graph to the base address of the backing file's memory map, then
    // flushing the map, unmapping it, closing the section, and closing the
    // file.
    //

    ULARGE_INTEGER SaveTableFileStartCycles;
    LARGE_INTEGER SaveTableFileStartCounter;

    ULARGE_INTEGER SaveTableFileEndCycles;
    LARGE_INTEGER SaveTableFileEndCounter;

    ULARGE_INTEGER SaveTableFileElapsedCycles;
    ULARGE_INTEGER SaveTableFileElapsedMicroseconds;

    //
    // Number of failed attempts at solving the graph across all threads.
    //

    volatile LONGLONG FailedAttempts;

    //
    // The main threadpool callback environment, used for solving perfect hash
    // solutions in parallel.
    //

    TP_CALLBACK_ENVIRON MainCallbackEnv;
    SLIST_HEADER MainWorkListHead;
    PTP_CLEANUP_GROUP MainCleanupGroup;
    PTP_POOL MainThreadpool;
    PTP_WORK MainWork;
    ULONG MinimumConcurrency;
    ULONG MaximumConcurrency;

    //
    // The algorithm is responsible for registering an appropriate callback
    // for main thread work items in this next field.
    //

    PPERFECT_HASH_MAIN_WORK_CALLBACK MainWorkCallback;

    //
    // A threadpool for offloading file operations.
    //

    TP_CALLBACK_ENVIRON FileCallbackEnv;
    SLIST_HEADER FileWorkListHead;
    PTP_CLEANUP_GROUP FileCleanupGroup;
    PTP_POOL FileThreadpool;
    PTP_WORK FileWork;

    //
    // The algorithm is responsible for registering an appropriate callback
    // for file work threadpool work items in this next field.
    //

    PPERFECT_HASH_FILE_WORK_CALLBACK FileWorkCallback;

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
    //      algorithm's creation routine (e.g. CreatePerfectHashImplChm01).
    //

    TP_CALLBACK_ENVIRON FinishedCallbackEnv;
    PTP_POOL FinishedThreadpool;
    SLIST_HEADER FinishedWorkListHead;
    PTP_WORK FinishedWork;

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

    volatile LONGLONG FinishedCount;

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

    //
    // N.B. These fields are commented in the definition of the structure
    //      TABLE_INFO_ON_DISK.
    //

    ULONGLONG InitialTableSize;
    ULONGLONG NumberOfTableResizeEvents;
    ULONGLONG TotalNumberOfAttemptsWithSmallerTableSizes;
    ULONGLONG ClosestWeCameToSolvingGraphWithSmallerTableSizes;

    //
    // Backing vtbl.
    //

    PERFECT_HASH_CONTEXT_VTBL Interface;

    //
    // N.B. As additional table functions are added to the context vtbl, you'll
    //      need to comment and un-comment the following padding field in order
    //      to avoid "warning: additional 8 bytes padding added after ..."-type
    //      warnings.
    //

    //PVOID Padding3;

} PERFECT_HASH_CONTEXT;
typedef PERFECT_HASH_CONTEXT *PPERFECT_HASH_CONTEXT;

#define TryAcquirePerfectHashContextLockExclusive(Context) \
    TryAcquireSRWLockExclusive(&Context->Lock)

#define ReleasePerfectHashContextLockExclusive(Context) \
    ReleaseSRWLockExclusive(&Context->Lock)

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

#define CONTEXT_SAVE_TIMERS_TO_TABLE_INFO_ON_DISK(Name) \
    TableInfoOnDisk->##Name##Cycles.QuadPart = (        \
        Context->##Name##ElapsedCycles.QuadPart         \
    );                                                  \
    TableInfoOnDisk->##Name##Microseconds.QuadPart = (  \
        Context->##Name##ElapsedMicroseconds.QuadPart   \
    )

//
// Internal method typedefs.
//

typedef
HRESULT
(NTAPI PERFECT_HASH_CONTEXT_INITIALIZE)(
    _In_ PPERFECT_HASH_CONTEXT Context
    );
typedef PERFECT_HASH_CONTEXT_INITIALIZE
      *PPERFECT_HASH_CONTEXT_INITIALIZE;

typedef
VOID
(NTAPI PERFECT_HASH_CONTEXT_RUNDOWN)(
    _In_ _Post_ptr_invalid_ PPERFECT_HASH_CONTEXT Context
    );
typedef PERFECT_HASH_CONTEXT_RUNDOWN
      *PPERFECT_HASH_CONTEXT_RUNDOWN;

typedef
HRESULT
(NTAPI PERFECT_HASH_CONTEXT_RESET)(
    _In_ PPERFECT_HASH_CONTEXT Context
    );
typedef PERFECT_HASH_CONTEXT_RESET *PPERFECT_HASH_CONTEXT_RESET;

//
// Public vtbl function decls.
//

extern PERFECT_HASH_CONTEXT_INITIALIZE PerfectHashContextInitialize;
extern PERFECT_HASH_CONTEXT_RUNDOWN PerfectHashContextRundown;
extern PERFECT_HASH_CONTEXT_RESET PerfectHashContextReset;
extern PERFECT_HASH_CONTEXT_SET_MAXIMUM_CONCURRENCY
    PerfectHashContextSetMaximumConcurrency;
extern PERFECT_HASH_CONTEXT_GET_MAXIMUM_CONCURRENCY
    PerfectHashContextGetMaximumConcurrency;
extern PERFECT_HASH_CONTEXT_SELF_TEST PerfectHashContextSelfTest;
extern PERFECT_HASH_CONTEXT_SELF_TEST_ARGVW
    PerfectHashContextSelfTestArgvW;
extern PERFECT_HASH_CONTEXT_EXTRACT_SELF_TEST_ARGS_FROM_ARGVW
    PerfectHashContextExtractSelfTestArgsFromArgvW;

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
