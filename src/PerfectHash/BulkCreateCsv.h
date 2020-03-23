/*++

Copyright (c) 2018-2020 Trent Nelson <trent@trent.me>

Module Name:

    BulkCreateCsv.h

Abstract:

    Private header file for bulk creation CSV glue.

--*/

//
// For VER_PRODUCTVERSION_STR.
//

#include <PerfectHashVersion.rc>

//
// Define the number of pages that should be allocated for the row buffer when
// constructing the CSV file.
//

#define BULK_CREATE_CSV_ROW_BUFFER_NUMBER_OF_PAGES 2

//
// Define an "X-Macro"-style macro for capturing the ordered definition of
// columns in a row of bulk create .csv output.
//
// The ENTRY macros receive (Name, Value, OutputMacro) as their parameters.
//

#define BULK_CREATE_CSV_ROW_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY)          \
    FIRST_ENTRY(ContextTimestamp,                                          \
                &Context->TimestampString,                                 \
                OUTPUT_STRING)                                             \
                                                                           \
    ENTRY(TableTimestamp,                                                  \
          &Table->TimestampString,                                         \
          OUTPUT_STRING)                                                   \
                                                                           \
    ENTRY(HeaderHash,                                                      \
          &Context->HexHeaderHash,                                         \
          OUTPUT_STRING)                                                   \
                                                                           \
    ENTRY(KeysName,                                                        \
          &Keys->File->Path->BaseNameA,                                    \
          OUTPUT_STRING)                                                   \
                                                                           \
    ENTRY(NumberOfKeys,                                                    \
          Keys->NumberOfElements.QuadPart,                                 \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(NumberOfEdges,                                                   \
          Table->IndexSize,                                                \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(NumberOfVertices,                                                \
          Table->HashSize,                                                 \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(Algorithm,                                                       \
          AlgorithmNames[Context->AlgorithmId],                            \
          OUTPUT_UNICODE_STRING_FAST)                                      \
                                                                           \
    ENTRY(HashFunction,                                                    \
          HashFunctionNames[Context->HashFunctionId],                      \
          OUTPUT_UNICODE_STRING_FAST)                                      \
                                                                           \
    ENTRY(MaskFunction,                                                    \
          MaskFunctionNames[Context->MaskFunctionId],                      \
          OUTPUT_UNICODE_STRING_FAST)                                      \
                                                                           \
    ENTRY(BuildType,                                                       \
          PerfectHashBuildConfigString,                                    \
          OUTPUT_CSTR)                                                     \
                                                                           \
    ENTRY(Version,                                                         \
          VER_PRODUCTVERSION_STR,                                          \
          OUTPUT_CSTR)                                                     \
                                                                           \
    ENTRY(MaximumConcurrency,                                              \
          Context->MaximumConcurrency,                                     \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(SolutionFound,                                                   \
          (TableCreateResult == S_OK ? 'Y' : 'N'),                         \
          OUTPUT_CHR)                                                      \
                                                                           \
    ENTRY(LowMemory,                                                       \
          (TableCreateResult == PH_I_LOW_MEMORY ? 'Y' : 'N'),              \
          OUTPUT_CHR)                                                      \
                                                                           \
    ENTRY(OutOfMemory,                                                     \
          (TableCreateResult == PH_I_OUT_OF_MEMORY ? 'Y' : 'N'),           \
          OUTPUT_CHR)                                                      \
                                                                           \
    ENTRY(OtherMemoryIssue,                                                \
          (TableCreateResult ==                                            \
           PH_I_FAILED_TO_ALLOCATE_MEMORY_FOR_ALL_GRAPHS ? 'Y' : 'N'),     \
          OUTPUT_CHR)                                                      \
                                                                           \
    ENTRY(TableCreateResult,                                               \
          TableCreateResult,                                               \
          OUTPUT_ERROR_CODE_STRING)                                        \
                                                                           \
    ENTRY(NumberOfSolutionsFound,                                          \
          Context->FinishedCount,                                          \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(Attempts,                                                        \
          Context->Attempts,                                               \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(FailedAttempts,                                                  \
          Context->FailedAttempts,                                         \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(TableDataUsesLargePages,                                         \
          (Table->Flags.TableDataUsesLargePages != FALSE ? 'Y' : 'N'),     \
          OUTPUT_CHR)                                                      \
                                                                           \
    ENTRY(ValuesArrayUsesLargePages,                                       \
          (Table->Flags.ValuesArrayUsesLargePages != FALSE ? 'Y' : 'N'),   \
          OUTPUT_CHR)                                                      \
                                                                           \
    ENTRY(KeysDataUsesLargePages,                                          \
          (Keys->Flags.KeysDataUsesLargePages != FALSE ? 'Y' : 'N'),       \
          OUTPUT_CHR)                                                      \
                                                                           \
    ENTRY(GraphRegisterSolvedTsxSuccessCount,                              \
          Context->GraphRegisterSolvedTsxSuccess,                          \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(GraphRegisterSolvedTsxStartedCount,                              \
          Context->GraphRegisterSolvedTsxStarted,                          \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(GraphRegisterSolvedTsxRetryCount,                                \
          Context->GraphRegisterSolvedTsxRetry,                            \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(GraphRegisterSolvedTsxFailedCount,                               \
          Context->GraphRegisterSolvedTsxFailed,                           \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(UsePreviousTableSize,                                            \
          (TableCreateFlags.UsePreviousTableSize == TRUE ?                 \
           'Y' : 'N'),                                                     \
          OUTPUT_CHR)                                                      \
                                                                           \
    ENTRY(InitialNumberOfTableResizes,                                     \
          Context->InitialResizes,                                         \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(NumberOfTableResizeEvents,                                       \
          Context->NumberOfTableResizeEvents,                              \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(MaximumGraphTraversalDepth,                                      \
          Table->MaximumGraphTraversalDepth,                               \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(SolveMicroseconds,                                               \
          Context->SolveElapsedMicroseconds.QuadPart,                      \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(VerifyMicroseconds,                                              \
          Context->VerifyElapsedMicroseconds.QuadPart,                     \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(BenchmarkWarmups,                                                \
          Table->BenchmarkWarmups,                                         \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(BenchmarkAttempts,                                               \
          Table->BenchmarkAttempts,                                        \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(BenchmarkIterationsPerAttempt,                                   \
          Table->BenchmarkIterations,                                      \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(SeededHashMinimumCyclesPerAttempt,                               \
          Table->SeededHashTimestamp.MinimumCycles.QuadPart,               \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(SeededHashMinimumNanosecondsPerAttempt,                          \
          Table->SeededHashTimestamp.MinimumNanoseconds.QuadPart,          \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(NullSeededHashMinimumCyclesPerAttempt,                           \
          Table->NullSeededHashTimestamp.MinimumCycles.QuadPart,           \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(NullSeededHashMinimumNanosecondsPerAttempt,                      \
          Table->NullSeededHashTimestamp.MinimumNanoseconds.QuadPart,      \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(DeltaHashMinimumCycles,                                          \
          Table->SeededHashTimestamp.MinimumCycles.QuadPart -              \
          Table->NullSeededHashTimestamp.MinimumCycles.QuadPart,           \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(DeltaHashMinimumNanoseconds,                                     \
          Table->SeededHashTimestamp.MinimumNanoseconds.QuadPart -         \
          Table->NullSeededHashTimestamp.MinimumNanoseconds.QuadPart,      \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(DeltaHashMinimumNanosecondsPerIteration,                         \
          Table->BenchmarkIterations == 0 ? 0 : (                          \
            (Table->SeededHashTimestamp.MinimumNanoseconds.QuadPart -      \
             Table->NullSeededHashTimestamp.MinimumNanoseconds.QuadPart) / \
             Table->BenchmarkIterations                                    \
          ),                                                               \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(SlowIndexMinimumCycles,                                          \
          Table->SlowIndexTimestamp.MinimumCycles.QuadPart,                \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(SlowIndexMinimumNanoseconds,                                     \
          Table->SlowIndexTimestamp.MinimumNanoseconds.QuadPart,           \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(SlowIndexMinimumNanosecondsPerIteration,                         \
          Table->BenchmarkIterations == 0 ? 0 : (                          \
            (Table->SlowIndexTimestamp.MinimumNanoseconds.QuadPart -       \
             Table->NullSeededHashTimestamp.MinimumNanoseconds.QuadPart) / \
             Table->BenchmarkIterations                                    \
          ),                                                               \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(NumberOfSeeds,                                                   \
          HashRoutineNumberOfSeeds[Context->HashFunctionId],               \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(Seed1,                                                           \
          (TableCreateResult == S_OK &&                                    \
           Table->TableInfoOnDisk->NumberOfSeeds >= 1 ?                    \
           Table->TableInfoOnDisk->Seed1 : 0),                             \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(Seed2,                                                           \
          (TableCreateResult == S_OK &&                                    \
           Table->TableInfoOnDisk->NumberOfSeeds >= 2 ?                    \
           Table->TableInfoOnDisk->Seed2 : 0),                             \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(Seed3,                                                           \
          (TableCreateResult == S_OK &&                                    \
           Table->TableInfoOnDisk->NumberOfSeeds >= 3 ?                    \
           Table->TableInfoOnDisk->Seed3 : 0),                             \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(Seed4,                                                           \
          (TableCreateResult == S_OK &&                                    \
           Table->TableInfoOnDisk->NumberOfSeeds >= 4 ?                    \
           Table->TableInfoOnDisk->Seed4 : 0),                             \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(Seed5,                                                           \
          (TableCreateResult == S_OK &&                                    \
           Table->TableInfoOnDisk->NumberOfSeeds >= 5 ?                    \
           Table->TableInfoOnDisk->Seed5 : 0),                             \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(Seed6,                                                           \
          (TableCreateResult == S_OK &&                                    \
           Table->TableInfoOnDisk->NumberOfSeeds >= 6 ?                    \
           Table->TableInfoOnDisk->Seed6 : 0),                             \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(Seed7,                                                           \
          (TableCreateResult == S_OK &&                                    \
           Table->TableInfoOnDisk->NumberOfSeeds >= 7 ?                    \
           Table->TableInfoOnDisk->Seed7 : 0),                             \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(Seed8,                                                           \
          (TableCreateResult == S_OK &&                                    \
           Table->TableInfoOnDisk->NumberOfSeeds >= 8 ?                    \
           Table->TableInfoOnDisk->Seed8 : 0),                             \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(NumberOfUserSeeds,                                               \
          (Context->UserSeeds != NULL ?                                    \
           Context->UserSeeds->NumberOfValues : 0),                        \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(UserSeed1,                                                       \
          (Context->UserSeeds != NULL &&                                   \
           Context->UserSeeds->NumberOfValues >= 1 ?                       \
           Context->UserSeeds->Values[0] : 0),                             \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(UserSeed2,                                                       \
          (Context->UserSeeds != NULL &&                                   \
           Context->UserSeeds->NumberOfValues >= 2 ?                       \
           Context->UserSeeds->Values[1] : 0),                             \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(UserSeed3,                                                       \
          (Context->UserSeeds != NULL &&                                   \
           Context->UserSeeds->NumberOfValues >= 3 ?                       \
           Context->UserSeeds->Values[2] : 0),                             \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(UserSeed4,                                                       \
          (Context->UserSeeds != NULL &&                                   \
           Context->UserSeeds->NumberOfValues >= 4 ?                       \
           Context->UserSeeds->Values[3] : 0),                             \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(UserSeed5,                                                       \
          (Context->UserSeeds != NULL &&                                   \
           Context->UserSeeds->NumberOfValues >= 5 ?                       \
           Context->UserSeeds->Values[4] : 0),                             \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(UserSeed6,                                                       \
          (Context->UserSeeds != NULL &&                                   \
           Context->UserSeeds->NumberOfValues >= 6 ?                       \
           Context->UserSeeds->Values[5] : 0),                             \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(UserSeed7,                                                       \
          (Context->UserSeeds != NULL &&                                   \
           Context->UserSeeds->NumberOfValues >= 7 ?                       \
           Context->UserSeeds->Values[6] : 0),                             \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(UserSeed8,                                                       \
          (Context->UserSeeds != NULL &&                                   \
           Context->UserSeeds->NumberOfValues >= 8 ?                       \
           Context->UserSeeds->Values[7] : 0),                             \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(FirstGraphWins,                                                  \
          (FirstSolvedGraphWins(Context) ? 'Y' : 'N'),                     \
          OUTPUT_CHR)                                                      \
                                                                           \
    ENTRY(FindBestMemoryCoverage,                                          \
          ((FindBestMemoryCoverage(Context) &&                             \
           !BestMemoryCoverageForKeysSubset(Context)) ? 'Y' : 'N'),        \
          OUTPUT_CHR)                                                      \
                                                                           \
    ENTRY(FindBestMemoryCoverageForKeysSubset,                             \
          (BestMemoryCoverageForKeysSubset(Context) ? 'Y' : 'N'),          \
          OUTPUT_CHR)                                                      \
                                                                           \
    ENTRY(BestCoverageType,                                                \
          &BestCoverageTypeNamesA[Context->BestCoverageType],              \
          OUTPUT_STRING)                                                   \
                                                                           \
    ENTRY(AttemptThatFoundBestGraph,                                       \
          Coverage->Attempt,                                               \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(NewBestGraphCount,                                               \
          Context->NewBestGraphCount,                                      \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(EqualBestGraphCount,                                             \
          Context->EqualBestGraphCount,                                    \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(NumberOfKeysInSubset,                                            \
          (Context->KeysSubset ? Context->KeysSubset->NumberOfValues : 0), \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(TotalNumberOfPages,                                              \
          Coverage->TotalNumberOfPages,                                    \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(TotalNumberOfLargePages,                                         \
          Coverage->TotalNumberOfLargePages,                               \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(TotalNumberOfCacheLines,                                         \
          Coverage->TotalNumberOfCacheLines,                               \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(NumberOfUsedPages,                                               \
          Coverage->NumberOfUsedPages,                                     \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(NumberOfUsedLargePages,                                          \
          Coverage->NumberOfUsedLargePages,                                \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(NumberOfUsedCacheLines,                                          \
          Coverage->NumberOfUsedCacheLines,                                \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(NumberOfEmptyPages,                                              \
          Coverage->NumberOfEmptyPages,                                    \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(NumberOfEmptyLargePages,                                         \
          Coverage->NumberOfEmptyLargePages,                               \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(NumberOfEmptyCacheLines,                                         \
          Coverage->NumberOfEmptyCacheLines,                               \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(FirstPageUsed,                                                   \
          Coverage->FirstPageUsed,                                         \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(FirstLargePageUsed,                                              \
          Coverage->FirstLargePageUsed,                                    \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(FirstCacheLineUsed,                                              \
          Coverage->FirstCacheLineUsed,                                    \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(LastPageUsed,                                                    \
          Coverage->LastPageUsed,                                          \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(LastLargePageUsed,                                               \
          Coverage->LastLargePageUsed,                                     \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(LastCacheLineUsed,                                               \
          Coverage->LastCacheLineUsed,                                     \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(TotalNumberOfAssigned,                                           \
          Coverage->TotalNumberOfAssigned,                                 \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_0,                         \
          Coverage->NumberOfAssignedPerCacheLineCounts[0],                 \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_1,                         \
          Coverage->NumberOfAssignedPerCacheLineCounts[1],                 \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_2,                         \
          Coverage->NumberOfAssignedPerCacheLineCounts[2],                 \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_3,                         \
          Coverage->NumberOfAssignedPerCacheLineCounts[3],                 \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_4,                         \
          Coverage->NumberOfAssignedPerCacheLineCounts[4],                 \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_5,                         \
          Coverage->NumberOfAssignedPerCacheLineCounts[5],                 \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_6,                         \
          Coverage->NumberOfAssignedPerCacheLineCounts[6],                 \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_7,                         \
          Coverage->NumberOfAssignedPerCacheLineCounts[7],                 \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_8,                         \
          Coverage->NumberOfAssignedPerCacheLineCounts[8],                 \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_9,                         \
          Coverage->NumberOfAssignedPerCacheLineCounts[9],                 \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_10,                        \
          Coverage->NumberOfAssignedPerCacheLineCounts[10],                \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_11,                        \
          Coverage->NumberOfAssignedPerCacheLineCounts[11],                \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_12,                        \
          Coverage->NumberOfAssignedPerCacheLineCounts[12],                \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_13,                        \
          Coverage->NumberOfAssignedPerCacheLineCounts[13],                \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_14,                        \
          Coverage->NumberOfAssignedPerCacheLineCounts[14],                \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_15,                        \
          Coverage->NumberOfAssignedPerCacheLineCounts[15],                \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_16,                        \
          Coverage->NumberOfAssignedPerCacheLineCounts[16],                \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(KeysMinValue,                                                    \
          Keys->Stats.MinValue,                                            \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(KeysMaxValue,                                                    \
          Keys->Stats.MaxValue,                                            \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(KeysFullPath,                                                    \
          &Keys->File->Path->FullPath,                                     \
          OUTPUT_UNICODE_STRING_FAST)                                      \
                                                                           \
    LAST_ENTRY(KeysBitmapString,                                           \
               Keys->Stats.KeysBitmap.String,                              \
               OUTPUT_RAW)

//
// Sometimes, whilst playing around with output, you may be on the fence about
// whether or not a particular field adds value and should be included.  For
// those that don't make the cut, you can append them to this excluded macro
// instead of deleting them entirely (this macro isn't used or referenced
// anywhere).
//

#define BULK_CREATE_CSV_ROW_TABLE_EXCLUDED(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    ENTRY(HighestDeletedEdgesCount,                                        \
          Context->HighestDeletedEdgesCount,                               \
          OUTPUT_INT)                                                      \
                                                                           \
    ENTRY(ClosestWeCameToSolvingGraphWithSmallerTableSizes,                \
          Context->ClosestWeCameToSolvingGraphWithSmallerTableSizes,       \
          OUTPUT_INT)                                                      \
                                                                           \

//
// Define a macro for initializing the local variables prior to writing a row.
//

#define BULK_CREATE_CSV_PRE_ROW()                                             \
    PCHAR Base;                                                               \
    PCHAR Output;                                                             \
                                                                              \
    Base = (PCHAR)CsvFile->BaseAddress;                                       \
    Output = RtlOffsetToPointer(Base, CsvFile->NumberOfBytesWritten.QuadPart)

//
// And one for post-row writing.
//

#define BULK_CREATE_CSV_POST_ROW() \
    CsvFile->NumberOfBytesWritten.QuadPart = RtlPointerToOffset(Base, Output)

#define EXPAND_AS_WRITE_BULK_CREATE_ROW_NOT_LAST_COLUMN(Name,        \
                                                        Value,       \
                                                        OutputMacro) \
    OutputMacro(Value);                                              \
    OUTPUT_CHR(',');

#define EXPAND_AS_WRITE_BULK_CREATE_ROW_LAST_COLUMN(Name,        \
                                                    Value,       \
                                                    OutputMacro) \
    OutputMacro(Value);                                          \
    OUTPUT_CHR('\n');


#define WRITE_BULK_CREATE_CSV_ROW() do {                 \
    BULK_CREATE_CSV_PRE_ROW();                           \
    BULK_CREATE_CSV_ROW_TABLE(                           \
        EXPAND_AS_WRITE_BULK_CREATE_ROW_NOT_LAST_COLUMN, \
        EXPAND_AS_WRITE_BULK_CREATE_ROW_NOT_LAST_COLUMN, \
        EXPAND_AS_WRITE_BULK_CREATE_ROW_LAST_COLUMN      \
    );                                                   \
    BULK_CREATE_CSV_POST_ROW();                          \
} while (0)

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
