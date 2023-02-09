/*++

Copyright (c) 2018-2023 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashContext.h

Abstract:

    This is the private header file for the PERFECT_HASH_CONTEXT component of
    the perfect hash library.  It defines the structure, and function pointer
    typedefs for private non-vtbl members.

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
    _In_ PLIST_ENTRY ListEntry
    );
typedef PERFECT_HASH_MAIN_WORK_CALLBACK
      *PPERFECT_HASH_MAIN_WORK_CALLBACK;

//
// As above, but for the console thread.
//

typedef
VOID
(CALLBACK PERFECT_HASH_CONSOLE_WORK_CALLBACK)(
    _In_ struct _PERFECT_HASH_CONTEXT *Context
    );
typedef PERFECT_HASH_CONSOLE_WORK_CALLBACK
      *PPERFECT_HASH_CONSOLE_WORK_CALLBACK;

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
    _In_ PLIST_ENTRY ListEntry
    );
typedef PERFECT_HASH_FILE_WORK_CALLBACK
      *PPERFECT_HASH_FILE_WORK_CALLBACK;

typedef
VOID
(CALLBACK PERFECT_HASH_FILE_WORK_ITEM_CALLBACK)(
    _In_ PFILE_WORK_ITEM Item
    );
typedef PERFECT_HASH_FILE_WORK_ITEM_CALLBACK
      *PPERFECT_HASH_FILE_WORK_ITEM_CALLBACK;


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
        // When set, indicates that the first graph to be solved, "wins".  If
        // a graph is solved but another thread has already beaten us to the
        // solution, we skip the graph assign step if this bit is set.
        //

        ULONG FirstSolvedGraphWins:1;

        //
        // When set, indicates multiple graph solutions will be found, and the
        // one with the best memory coverage will win.
        //

        ULONG FindBestMemoryCoverage:1;

        //
        // In conjunction with the bit above being set, indicates that the
        // best memory coverage mode we're looking for uses a subset of keys.
        //

        ULONG BestMemoryCoverageForKeysSubset:1;

        //
        // When set, indicates all solving activities should be stopped.
        //

        ULONG StopSolving:1;

        //
        // When set, indicates a bulk create operation is active.  This can be
        // used to alter console output and logging operations.
        //

        ULONG IsBulkCreate:1;

        //
        // When set, indicates the graph solving failed because every worker
        // thread failed to allocate sufficient memory to even attempt solving
        // the graph (specifically, the every graph's LoadInfo() call returned
        // E_OUTOFMEMORY).
        //

        ULONG AllGraphsFailedMemoryAllocation:1;

        //
        // When set, indicates a table create operation is active.  This can be
        // used to alter console output and logging operations.
        //

        ULONG IsTableCreate:1;

        //
        // When set, indicates a solve timeout expired.
        //

        ULONG SolveTimeoutExpired:1;

        //
        // When set, indicates the fixed number of attempts were reached.
        //

        ULONG FixedAttemptsReached:1;

        //
        // When set, indicates the max number of attempts were reached.
        //

        ULONG MaxAttemptsReached:1;

        //
        // When set, indicates the current resize request was manually signaled
        // via the console.
        //

        ULONG UserRequestedResize:1;

        //
        // When set, indicates function hook infrastructure is active.
        //

        ULONG HasFunctionHooking:1;

        //
        // When set, indicates a best coverage target value is present.
        //

        ULONG HasBestCoverageTargetValue:1;

        //
        // When set, indicates the best coverage target value is a double.
        // When not set, indicates the value is an integer.
        //
        // N.B. HasBestCoverageTargetValue must be set in order for this
        //      bitfield to be considered valid.
        //

        ULONG BestCoverageTargetValueIsDouble:1;

        //
        // When set, indicates a solution was found that met the best coverage
        // target value criteria.
        //

        ULONG BestCoverageTargetValueFound:1;

        //
        // Unused bits.
        //

        ULONG Unused:16;
    };
    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_CONTEXT_STATE;
C_ASSERT(sizeof(PERFECT_HASH_CONTEXT_STATE) == sizeof(ULONG));
typedef PERFECT_HASH_CONTEXT_STATE *PPERFECT_HASH_CONTEXT_STATE;

#define FirstSolvedGraphWins(Context) Context->State.FirstSolvedGraphWins
#define FindBestMemoryCoverage(Context) \
    (((Context)->State.FindBestMemoryCoverage) != FALSE)
#define BestMemoryCoverageForKeysSubset(Context) \
    ((Context)->State.BestMemoryCoverageForKeysSubset == TRUE)

#define IsLookingForBestCoverageTargetValue(Context) \
    ((Context)->State.HasBestCoverageTargetValue != FALSE)

#define BestCoverageTargetValueIsDouble(Context)           \
    ((Context)->State.HasBestCoverageTargetValue != FALSE)

#define FirstSolvedGraphWinsAndSkipMemoryCoverage(Context) (                  \
    (Context)->State.FirstSolvedGraphWins == TRUE &&                          \
    (Context)->Table->TableCreateFlags.SkipMemoryCoverageInFirstGraphWinsMode \
        == TRUE                                                               \
)

#define SetFirstSolvedGraphWins(Context)          \
    ASSERT(Context->MinAttempts == 0);            \
    Context->State.FirstSolvedGraphWins = TRUE;   \
    Context->State.FindBestMemoryCoverage = FALSE

#define SetFindBestMemoryCoverage(Context)       \
    Context->State.FirstSolvedGraphWins = FALSE; \
    Context->State.FindBestMemoryCoverage = TRUE

#define StopSolving(Context) (Context->State.StopSolving != FALSE)

#define SetStopSolving(Context) (Context->State.StopSolving = TRUE)
#define ClearStopSolving(Context) (Context->State.StopSolving = FALSE)

#define IsContextBulkCreate(Context) ((Context)->State.IsBulkCreate == TRUE)
#define SetContextBulkCreate(Context) ((Context)->State.IsBulkCreate = TRUE)
#define ClearContextBulkCreate(Context) ((Context)->State.IsBulkCreate = FALSE)

#define IsContextTableCreate(Context) ((Context)->State.IsTableCreate == TRUE)
#define SetContextTableCreate(Context) ((Context)->State.IsTableCreate = TRUE)
#define ClearContextTableCreate(Context) \
    ((Context)->State.IsTableCreate = FALSE)

#define IsSystemRngId(Id) (Id == PerfectHashRngSystemId)
#define IsSystemRng(Context) ((Context)->RngId == PerfectHashRngSystemId)

DEFINE_UNUSED_FLAGS(PERFECT_HASH_CONTEXT);

typedef struct _BEST_GRAPH_INFO {

    //
    // The attempt that found this best graph.
    //

    LONGLONG Attempt;

    //
    // Number of elapsed milliseconds for this best graph to be found.
    //

    ULONGLONG ElapsedMilliseconds;

    //
    // The value used in the predicate comparison to determine if this was
    // the best graph (e.g. HighestMaxGraphTraversalDepth).
    //

    ULONGLONG Value;

    //
    // Number of times a best graph was found that was equal to this existing
    // best graph.  (This is useful for depicting how many best graphs were
    // found for a given value before the next best level was found.)
    //

    ULONG EqualCount;

    //
    // Pad out to an 8-byte boundary.
    //

    ULONG Padding;

    //
    // Coverage value as a double, if applicable.
    //

    DOUBLE ValueAsDouble;

    //
    // Seed values used for this best graph.
    //

    ULONG Seeds[MAX_NUMBER_OF_SEEDS];

    //
    // Inline the entire coverage structure.
    //

    union {
        ASSIGNED_MEMORY_COVERAGE Coverage;
        ASSIGNED16_MEMORY_COVERAGE Coverage16;
    };

} BEST_GRAPH_INFO, *PBEST_GRAPH_INFO;
#define MAX_BEST_GRAPH_INFO 32

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
    // Capture the system allocation granularity.
    //

    ULONG SystemAllocationGranularity;

    //
    // Pointer to the path instance of the base output directory set via
    // SetBaseOutputDirectory().
    //

    PPERFECT_HASH_DIRECTORY BaseOutputDirectory;

    //
    // If a bulk create operation is in progress, a pointer to a file instance
    // for the <BaseOutputDir>\PerfectHashBulkCreate_<HeaderHash>.csv file.
    //

    struct _PERFECT_HASH_FILE *BulkCreateCsvFile;

    //
    // If a table create operation is in progress, a pointer to a file instance
    // for the <BaseOutputDir>\PerfectHashTableCreate_<HeaderHash>.csv file.
    //

    struct _PERFECT_HASH_FILE *TableCreateCsvFile;

    //
    // Pointer to a base buffer and current buffer and size for .csv rows, if
    // applicable (i.e. if in bulk create mode).
    //

    PCHAR BaseRowBuffer;
    PCHAR RowBuffer;
    ULONGLONG RowBufferSize;

    //
    // This counter is incremented every time a low-memory event is observed.
    //

    volatile LONG LowMemoryObserved;

    //
    // Count of active graph solving loops (worker threads).  This is
    // incremented on solving loop entry and decremented on solving loop exit.
    //

    volatile LONG ActiveSolvingLoops;

    //
    // Count of remaining solving loops.  This is initialized to the maximum
    // concurrency value prior to starting solver threads and decremented every
    // time one finishes.
    //

    volatile LONG RemainingSolverLoops;

    //
    // Prior to submitting graph solving work, the following field is
    // initialized to the number of threads that will be participating in the
    // solving attempt.  It is decremented each time the initial graph's
    // LoadInfo() call fails due to an out-of-memory condition.  If it hits
    // zero, it indicates no threads were able to allocate sufficient memory
    // to attempt solving, and the FailedEvent is set, which unwaits the main
    // thread.
    //
    // N.B. Although the name of the field implies a count of graph memory
    //      failures, it is actually used in the reverse direction, i.e. it
    //      gets decremented for each failure and then tested against 0.  If
    //      we incremented each failure, we'd need to check against an expected
    //      failure count to determine if all graphs failed, which involves more
    //      moving parts.
    //

    volatile LONG GraphMemoryFailures;

    //
    // Pointer to the active perfect hash table.
    //

    struct _PERFECT_HASH_TABLE *Table;

#ifdef PH_WINDOWS
    //
    // Pointer to a CU instance, if applicable.
    //

    struct _CU *Cu;

    //
    // CUDA devices.
    //

    PH_CU_DEVICES CuDevices;

    //
    // Pointer to the array of CUDA device ordinals from the command line.
    //

    PVALUE_ARRAY CuDeviceOrdinals;

    //
    // CUDA device ordinal from the command line.
    //

    LONG CuDeviceOrdinal;
#endif

    //
    // Minimum number of keys that need to be present in order to honor find
    // best graph mode.
    //

    ULONG MinNumberOfKeysForFindBestGraph;

    //
    // Pointer to a buffer that is used to construct console output before
    // issuing WriteFile().
    //

    _Writable_elements_(ConsoleBufferSizeInBytes)
    PCHAR ConsoleBuffer;
    ULONGLONG ConsoleBufferSizeInBytes;

    //
    // Handles to stdin and stdout, and current process.
    //

    HANDLE InputHandle;
    HANDLE OutputHandle;
    HANDLE ProcessHandle;

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
    // Guarded list for capturing GRAPH instances.
    //

    PGUARDED_LIST GraphList;

    //
    // Command line used (for information purposes only).
    //

    LPWSTR CommandLineW;

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
    // Initial number of table resizes to simulate before graph solving.  This
    // is used to reduce the keys-to-vertices ratio (which improves the graph
    // solving probability).
    //

    ULONG InitialResizes;

    //
    // The following members control the minimum and maximum graph solving
    // attempts, and are particularly useful for ensuring a fixed set of
    // attempts are made at solving a graph, which is useful during
    // benchmarking.  Attempt refers to literally that: an attempt.  So, it
    // includes both failed and successfully-solved graphs.  If you want to
    // generate a fixed number of solutions, use TargetNumberOfSolutions.
    //
    // FixedAttempts, if non-zero, trumps everything else.
    //

    ULONG MinAttempts;
    ULONG MaxAttempts;
    ULONG TargetNumberOfSolutions;
    ULONG FixedAttempts;

    //
    // Maximum number of equal best graphs.  When this number is hit, graph
    // solving will stop, even if the target number of attempts supplied by
    // --BestCoverageAttempts hasn't been hit.
    //

    ULONG MaxNumberOfEqualBestGraphs;

    //
    // If we're attempting to find the best memory coverage, the following
    // fields capture the type of "best" coverage we're looking for, and the
    // number of attempts to make.
    //

    PERFECT_HASH_TABLE_BEST_COVERAGE_TYPE_ID BestCoverageType;
    ULONGLONG BestCoverageAttempts;

    //
    // If a best coverage target value has been supplied, it will be captured
    // here.
    //

    union {
        DOUBLE AsDouble;
        ULONG AsULong;
    } BestCoverageTargetValue;

    //
    // Pointer to a keys subset structure, if applicable (e.g. if the best
    // coverage type is one that uses a keys subset, such as lowest number of
    // used cache lines by subset of keys.
    //

    PKEYS_SUBSET KeysSubset;

    //
    // Pointer to user-provided seed data, if applicable.
    //

    PVALUE_ARRAY UserSeeds;

    //
    // Pointer to seed masks, if applicable.
    //

    PCSEED_MASKS SeedMasks;

    //
    // Pointer to relevant seed mask byte counts, if applicable.
    //

    PCSEED_MASK_COUNTS Seed3Byte1MaskCounts;
    PCSEED_MASK_COUNTS Seed3Byte2MaskCounts;

    //
    // Captures the number of failures due to the generation of two
    // identical vertices.
    //

    volatile LONGLONG VertexCollisionFailures;

    //
    // Computer name.
    //

    STRING ComputerName;
    CHAR ComputerNameBuffer[MAX_COMPUTERNAME_LENGTH+1];

    //
    // Hex string representation of the CSV header hash.
    //

    STRING HexHeaderHash;
    CHAR HexHeaderHashBuffer[16];

    //
    // Best and spare graphs.
    //

    CRITICAL_SECTION BestGraphCriticalSection;

    _Guarded_by_(BestGraphCriticalSection)
    struct _GRAPH *BestGraph;

    _Guarded_by_(BestGraphCriticalSection)
    struct _GRAPH *SpareGraph;

    //
    // Captures the attempt number of the first solved graph.
    //

    _Guarded_by_(BestGraphCriticalSection)
    ULONGLONG FirstAttemptSolved;

    //
    // Captures the attempt number of the most recent solved graph (regardless
    // of whether or not it was the best graph, if applicable).  Any graph can
    // write to this value at any time; it is not synchronized in anyway.  It
    // is used for informational purposes only.
    //

    volatile ULONGLONG MostRecentSolvedAttempt;

    //
    // The following counter is incremented every time a new "best graph" is
    // registered.  It can be helpful during debugging.
    //

    _Guarded_by_(BestGraphCriticalSection)
    volatile LONG NewBestGraphCount;

    //
    // The following counter is incremented every time a graph is found whose
    // coverage matches the existing best graph's coverage (for the given
    // predicate when in "find best graph" mode).
    //

    _Guarded_by_(BestGraphCriticalSection)
    volatile LONG EqualBestGraphCount;

    //
    // Each time a graph is solved, the running value of solutions found ratio
    // is updated (number of solved attempts divided by number of total
    // attempts).
    //

    _Guarded_by_(BestGraphCriticalSection)
    DOUBLE RunningSolutionsFoundRatio;

    //
    // Milliseconds returned by GetTickCount64() when solving starts; this is
    // use to derive the value for ElapsedMilliseconds in the following array
    // of BEST_GRAPH_INFO elements.
    //

    ULONGLONG StartMilliseconds;

    //
    // Array of best graph info structs that capture the attempt and time when
    // a new best graph was found (when in FindBestGraph solving mode).
    //

    _Guarded_by_(BestGraphCriticalSection)
    BEST_GRAPH_INFO BestGraphInfo[MAX_BEST_GRAPH_INFO];

    //
    // Handle to a low-memory resource notification event.
    //
    // N.B. This event handle differs from the ones below in that it is obtained
    //      from CreateMemoryResourceNotification(), and thus, doesn't have a
    //      random object name generated for it.
    //

    HANDLE LowMemoryEvent;

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
    // The following event is set if we're in find best graph mode and a new
    // best graph has been found.
    //

    HANDLE NewBestGraphFoundEvent;

    //
    // The following events are related to file work.  Each file work ID has
    // both a prepared and saved event, which is used to synchronize with the
    // main solving thread.
    //

#define EXPAND_AS_EVENT(            \
    Verb, VUpper, Name, Upper,      \
    EofType, EofValue,              \
    Suffix, Extension, Stream, Base \
)                                   \
    HANDLE Verb##d##Name##Event;

#define EXPAND_AS_FIRST_EVENT(        \
    Verb, VUpper, Name, Upper,        \
    EofType, EofValue,                \
    Suffix, Extension, Stream, Base   \
)                                     \
    union {                           \
        HANDLE Verb##d##Name##Event;  \
        HANDLE First##Verb##d##Event; \
    };

#define EXPAND_AS_LAST_EVENT(        \
    Verb, VUpper, Name, Upper,       \
    EofType, EofValue,               \
    Suffix, Extension, Stream, Base  \
)                                    \
    union {                          \
        HANDLE Verb##d##Name##Event; \
        HANDLE Last##Verb##d##Event; \
    };

    PREPARE_FILE_WORK_TABLE(EXPAND_AS_FIRST_EVENT,
                            EXPAND_AS_EVENT,
                            EXPAND_AS_LAST_EVENT)

    SAVE_FILE_WORK_TABLE(EXPAND_AS_FIRST_EVENT,
                         EXPAND_AS_EVENT,
                         EXPAND_AS_LAST_EVENT)


    volatile LONG GraphRegisterSolvedTsxStarted;
    ULONG Padding3;

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
    // Used to capture current vs total attempts per second in console output.
    //

    LONGLONG BaselineAttempts;
    FILETIME64 BaselineFileTime;

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

    PGUARDED_LIST MainWorkList;
    TP_CALLBACK_ENVIRON MainCallbackEnv;
    PTP_CLEANUP_GROUP MainCleanupGroup;
    PTP_POOL MainThreadpool;
    PTP_WORK MainWork;
    PTP_WORK ConsoleWork;
    PTP_TIMER SolveTimeout;
    ULONG MinimumConcurrency;
    ULONG MaximumConcurrency;

    //
    // RNG details.
    //

    PCUNICODE_STRING RngName;
    PERFECT_HASH_RNG_ID RngId;
    RNG_FLAGS RngFlags;
    ULONGLONG RngSeed;
    ULONGLONG RngSubsequence;
    ULONGLONG RngOffset;

    //
    // The algorithm is responsible for registering an appropriate callback
    // for main thread work items in this next field.
    //

    PPERFECT_HASH_MAIN_WORK_CALLBACK MainWorkCallback;

    //
    // Likewise for console interaction.
    //

    PPERFECT_HASH_CONSOLE_WORK_CALLBACK ConsoleWorkCallback;

    //
    // A threadpool for offloading file operations.
    //

    PGUARDED_LIST FileWorkList;
    TP_CALLBACK_ENVIRON FileCallbackEnv;
    PTP_CLEANUP_GROUP FileCleanupGroup;
    PTP_POOL FileThreadpool;
    PTP_WORK FileWork;

    volatile LONG GraphRegisterSolvedTsxSuccess;
    ULONG Padding4;

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

    PGUARDED_LIST FinishedWorkList;
    TP_CALLBACK_ENVIRON FinishedCallbackEnv;
    PTP_POOL FinishedThreadpool;
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

    volatile LONG GraphRegisterSolvedTsxRetry;
    ULONG Padding5;

    //
    // Timestamp of instance creation.  The STRING structure's Buffer is wired
    // up to the address of the TimestampBuffer variable.
    //

    FILETIME64 FileTime;
    SYSTEMTIME SystemTime;
    CHAR TimestampBuffer[RTL_TIMESTAMP_FORMAT_LENGTH];
    STRING TimestampString;

    //
    // N.B. These fields are commented in the definition of the structure
    //      TABLE_INFO_ON_DISK.
    //

    ULONGLONG InitialTableSize;
    ULONGLONG NumberOfTableResizeEvents;
    ULONGLONG TotalNumberOfAttemptsWithSmallerTableSizes;
    ULONGLONG ClosestWeCameToSolvingGraphWithSmallerTableSizes;

    //
    // Captures the number of failures due to the graph being cyclic.
    //

    volatile LONGLONG CyclicGraphFailures;

    //
    // Pointers to the context file instances.
    //

#define EXPAND_AS_FIRST_CONTEXT_FILE(Verb, VUpper, Name, Upper) \
    union {                                                     \
        PPERFECT_HASH_FILE Name;                                \
        PPERFECT_HASH_FILE FirstFile;                           \
    };

#define EXPAND_AS_LAST_CONTEXT_FILE(Verb, VUpper, Name, Upper) \
    union {                                                    \
        PPERFECT_HASH_FILE Name;                               \
        PPERFECT_HASH_FILE LastFile;                           \
    };

#define EXPAND_AS_CONTEXT_FILE(Verb, VUpper, Name, Upper) \
    PPERFECT_HASH_FILE Name;

    CONTEXT_FILE_WORK_TABLE(EXPAND_AS_FIRST_CONTEXT_FILE,
                            EXPAND_AS_CONTEXT_FILE,
                            EXPAND_AS_LAST_CONTEXT_FILE)

    volatile LONG GraphRegisterSolvedTsxFailed;
    ULONG Padding6;

    //
    // Pointers to function hook related infrastructure.
    //

#ifdef PH_WINDOWS
    HMODULE CallbackModule;
    HMODULE FunctionHookModule;
    PCUNICODE_STRING CallbackDllPath;
    PFUNCTION_ENTRY_CALLBACK FunctionEntryCallback;
    PSET_FUNCTION_ENTRY_CALLBACK SetFunctionEntryCallback;
    PGET_FUNCTION_ENTRY_CALLBACK GetFunctionEntryCallback;
    PCLEAR_FUNCTION_ENTRY_CALLBACK ClearFunctionEntryCallback;
    PIS_FUNCTION_ENTRY_CALLBACK_ENABLED IsFunctionEntryCallbackEnabled;
    PVOID CallbackModuleTableValues;
    SIZE_T CallbackModuleTableValueSizeInBytes;
    SIZE_T CallbackModuleNumberOfTableValues;
    LARGE_INTEGER CallbackModuleTableValuesEndOfFile;

    //
    // If function hooking is enabled, and the callback DLL is a compiled
    // perfect hash table, this file will save a copy of the table values
    // during context rundown.
    //

    struct _PERFECT_HASH_FILE *CallbackModuleTableValuesFile;

    //
    // We allow toggling of the callback function on and off during runtime
    // via the console.  The following variables capture the necessary state
    // required to effectuate this.
    //

    PVOID CallbackContext;
    PVOID CallbackModuleBaseAddress;
    PFUNCTION_ENTRY_CALLBACK CallbackFunction;
    ULONG CallbackModuleSizeInBytes;
    ULONG CallbackModuleIgnoreRip;
#endif

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

    PVOID Padding8;

} PERFECT_HASH_CONTEXT;
typedef PERFECT_HASH_CONTEXT *PPERFECT_HASH_CONTEXT;

#define TryAcquirePerfectHashContextLockExclusive(Context) \
    TryAcquireSRWLockExclusive(&Context->Lock)

#define ReleasePerfectHashContextLockExclusive(Context) \
    ReleaseSRWLockExclusive(&Context->Lock)

#define ActiveCuDevice(Context) \
    &((Context)->CuDevices.Devices[(Context)->CuDeviceOrdinal])

//
// Define helper macros for appending items to the tail of a guarded list.
//

//
// Main work.
//

#define InsertTailMainWork(Context, ListEntry) \
    Context->MainWorkList->Vtbl->InsertTail(   \
        Context->MainWorkList,                 \
        ListEntry                              \
    )

#define RemoveHeadMainWork(Context, ListEntry) \
    Context->MainWorkList->Vtbl->RemoveHeadEx( \
        Context->MainWorkList,                 \
        ListEntry                              \
    )

#define ResetMainWorkList(Context) \
    Context->MainWorkList->Vtbl->Reset(Context->MainWorkList)

//
// File work.
//

#define InsertTailFileWork(Context, ListEntry) \
    Context->FileWorkList->Vtbl->InsertTail(   \
        Context->FileWorkList,                 \
        ListEntry                              \
    )

#define RemoveHeadFileWork(Context, ListEntry) \
    Context->FileWorkList->Vtbl->RemoveHeadEx( \
        Context->FileWorkList,                 \
        ListEntry                              \
    )

#define ResetFileWorkList(Context) \
    Context->FileWorkList->Vtbl->Reset(Context->FileWorkList)

//
// Finished work.
//

#define InsertHeadFinishedWork(Context, ListEntry) \
    Context->FinishedWorkList->Vtbl->InsertHead(   \
        Context->FinishedWorkList,                 \
        ListEntry                                  \
    )

#define InsertTailFinishedWork(Context, ListEntry) \
    Context->FinishedWorkList->Vtbl->InsertTail(   \
        Context->FinishedWorkList,                 \
        ListEntry                                  \
    )

#define RemoveHeadFinishedWork(Context, ListEntry) \
    Context->FinishedWorkList->Vtbl->RemoveHeadEx( \
        Context->FinishedWorkList,                 \
        ListEntry                                  \
    )

#define ResetFinishedWorkList(Context) \
    Context->FinishedWorkList->Vtbl->Reset(Context->FinishedWorkList)

//
// Define helper macros for marking start/end points for the context's
// cycle/counter fields.  When starting, we put __rdtsc() last, and when
// stopping we put it first, as its resolution is more sensitive than the
// QueryPerformanceCounter() routine.
//

#define CONTEXT_START_TIMERS(Name)                         \
    QueryPerformanceCounter(&Context->Name##StartCounter); \
    Context->Name##StartCycles.QuadPart = __rdtsc()

#define CONTEXT_END_TIMERS(Name)                            \
    Context->Name##EndCycles.QuadPart = __rdtsc();          \
    QueryPerformanceCounter(&Context->Name##EndCounter);    \
    Context->Name##ElapsedCycles.QuadPart = (               \
        Context->Name##EndCycles.QuadPart -                 \
        Context->Name##StartCycles.QuadPart                 \
    );                                                      \
    Context->Name##ElapsedMicroseconds.QuadPart = (         \
        Context->Name##EndCounter.QuadPart -                \
        Context->Name##StartCounter.QuadPart                \
    );                                                      \
    Context->Name##ElapsedMicroseconds.QuadPart *= 1000000; \
    Context->Name##ElapsedMicroseconds.QuadPart /= (        \
        Context->Frequency.QuadPart                         \
    )

#define CONTEXT_SAVE_TIMERS_TO_TABLE_INFO_ON_DISK(Name) \
    TableInfoOnDisk->Name##Cycles.QuadPart = (          \
        Context->Name##ElapsedCycles.QuadPart           \
    );                                                  \
    TableInfoOnDisk->Name##Microseconds.QuadPart = (    \
        Context->Name##ElapsedMicroseconds.QuadPart     \
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

typedef
VOID
(NTAPI PERFECT_HASH_CONTEXT_APPLY_THREADPOOL_PRIORITIES)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    );
typedef PERFECT_HASH_CONTEXT_APPLY_THREADPOOL_PRIORITIES
      *PPERFECT_HASH_CONTEXT_APPLY_THREADPOOL_PRIORITIES;

typedef
HRESULT
(NTAPI PERFECT_HASH_CONTEXT_INITIALIZE_RNG)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags,
    _In_ PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    );
typedef PERFECT_HASH_CONTEXT_INITIALIZE_RNG
      *PPERFECT_HASH_CONTEXT_INITIALIZE_RNG;

typedef
HRESULT
(NTAPI PERFECT_HASH_CONTEXT_INITIALIZE_CUDA)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    );
typedef PERFECT_HASH_CONTEXT_INITIALIZE_CUDA
      *PPERFECT_HASH_CONTEXT_INITIALIZE_CUDA;

typedef
HRESULT
(NTAPI PERFECT_HASH_CONTEXT_INITIALIZE_FUNCTION_HOOK_CALLBACK_DLL)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags,
    _In_ PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    );
typedef PERFECT_HASH_CONTEXT_INITIALIZE_FUNCTION_HOOK_CALLBACK_DLL
      *PPERFECT_HASH_CONTEXT_INITIALIZE_FUNCTION_HOOK_CALLBACK_DLL;

typedef
_Must_inspect_result_
HRESULT
(NTAPI PERFECT_HASH_CONTEXT_INITIALIZE_KEY_SIZE)(
    _In_ PPERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags,
    _In_ PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters,
    _Out_ PULONG KeySizeInBytes
    );
typedef PERFECT_HASH_CONTEXT_INITIALIZE_KEY_SIZE
      *PPERFECT_HASH_CONTEXT_INITIALIZE_KEY_SIZE;

typedef
_Must_inspect_result_
HRESULT
(NTAPI PERFECT_HASH_CONTEXT_TRY_PREPARE_CALLBACK_TABLE_VALUES_FILE)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags
    );
typedef PERFECT_HASH_CONTEXT_TRY_PREPARE_CALLBACK_TABLE_VALUES_FILE
      *PPERFECT_HASH_CONTEXT_TRY_PREPARE_CALLBACK_TABLE_VALUES_FILE;

typedef
_Must_inspect_result_
HRESULT
(NTAPI PERFECT_HASH_CONTEXT_TRY_RUNDOWN_CALLBACK_TABLE_VALUES_FILE)(
    _In_ PPERFECT_HASH_CONTEXT Context
    );
typedef PERFECT_HASH_CONTEXT_TRY_RUNDOWN_CALLBACK_TABLE_VALUES_FILE
      *PPERFECT_HASH_CONTEXT_TRY_RUNDOWN_CALLBACK_TABLE_VALUES_FILE;

//
// Function decls.
//

#ifndef __INTELLISENSE__
extern PERFECT_HASH_CONTEXT_INITIALIZE PerfectHashContextInitialize;
extern PERFECT_HASH_CONTEXT_RUNDOWN PerfectHashContextRundown;
extern PERFECT_HASH_CONTEXT_RESET PerfectHashContextReset;
extern PERFECT_HASH_CONTEXT_INITIALIZE_KEY_SIZE
    PerfectHashContextInitializeKeySize;
extern PERFECT_HASH_CONTEXT_SET_MAXIMUM_CONCURRENCY
    PerfectHashContextSetMaximumConcurrency;
extern PERFECT_HASH_CONTEXT_GET_MAXIMUM_CONCURRENCY
    PerfectHashContextGetMaximumConcurrency;
extern PERFECT_HASH_CONTEXT_SET_BASE_OUTPUT_DIRECTORY
    PerfectHashContextSetBaseOutputDirectory;
extern PERFECT_HASH_CONTEXT_GET_BASE_OUTPUT_DIRECTORY
    PerfectHashContextGetBaseOutputDirectory;
extern PERFECT_HASH_CONTEXT_BULK_CREATE PerfectHashContextBulkCreate;
extern PERFECT_HASH_CONTEXT_BULK_CREATE_ARGVW
    PerfectHashContextBulkCreateArgvW;
extern PERFECT_HASH_CONTEXT_EXTRACT_BULK_CREATE_ARGS_FROM_ARGVW
    PerfectHashContextExtractBulkCreateArgsFromArgvW;
extern PERFECT_HASH_CONTEXT_TABLE_CREATE PerfectHashContextTableCreate;
extern PERFECT_HASH_CONTEXT_TABLE_CREATE_ARGVW
    PerfectHashContextTableCreateArgvW;
#ifndef PH_WINDOWS
extern PERFECT_HASH_CONTEXT_TABLE_CREATE_ARGVA
    PerfectHashContextTableCreateArgvA;
extern PERFECT_HASH_CONTEXT_BULK_CREATE_ARGVA
    PerfectHashContextBulkCreateArgvA;
#endif
extern PERFECT_HASH_CONTEXT_EXTRACT_TABLE_CREATE_ARGS_FROM_ARGVW
    PerfectHashContextExtractTableCreateArgsFromArgvW;
extern PERFECT_HASH_CONTEXT_APPLY_THREADPOOL_PRIORITIES
    PerfectHashContextApplyThreadpoolPriorities;
extern PERFECT_HASH_CONTEXT_INITIALIZE_RNG
    PerfectHashContextInitializeRng;
extern PERFECT_HASH_CONTEXT_INITIALIZE_CUDA
    PerfectHashContextInitializeCuda;
extern PERFECT_HASH_CONTEXT_INITIALIZE_FUNCTION_HOOK_CALLBACK_DLL
    PerfectHashContextInitializeFunctionHookCallbackDll;
extern PERFECT_HASH_CONTEXT_TRY_PREPARE_CALLBACK_TABLE_VALUES_FILE
    PerfectHashContextTryPrepareCallbackTableValuesFile;
extern PERFECT_HASH_CONTEXT_TRY_RUNDOWN_CALLBACK_TABLE_VALUES_FILE
    PerfectHashContextTryRundownCallbackTableValuesFile;
#endif

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
