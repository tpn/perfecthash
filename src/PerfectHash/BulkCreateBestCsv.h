/*++

Copyright (c) 2018-2020 Trent Nelson <trent@trent.me>

Module Name:

    BulkCreateBestCsv.h

Abstract:

    Private header file for bulk creation CSV glue when find best graph mode is
    active.

--*/

//
// For VER_PRODUCTVERSION_STR.
//

#include <PerfectHashVersion.rc>

//
// Define the number of pages that should be allocated for the row buffer when
// constructing the CSV file.
//

#define BULK_CREATE_BEST_CSV_ROW_BUFFER_NUMBER_OF_PAGES 14

//
// Define an "X-Macro"-style macro for capturing the ordered definition of
// columns in a row of bulk create .csv output.
//
// The ENTRY macros receive (Name, Value, OutputMacro) as their parameters.
//

#define BULK_CREATE_BEST_CSV_ROW_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY)                       \
    FIRST_ENTRY(ContextTimestamp,                                                            \
                &Context->TimestampString,                                                   \
                OUTPUT_STRING)                                                               \
                                                                                             \
    ENTRY(TableTimestamp,                                                                    \
          &Table->TimestampString,                                                           \
          OUTPUT_STRING)                                                                     \
                                                                                             \
    ENTRY(HeaderHash,                                                                        \
          &Context->HexHeaderHash,                                                           \
          OUTPUT_STRING)                                                                     \
                                                                                             \
    ENTRY(ComputerName,                                                                      \
          &Context->ComputerName,                                                            \
          OUTPUT_STRING)                                                                     \
                                                                                             \
    ENTRY(CpuBrand,                                                                          \
          &Context->Rtl->CpuFeatures.Brand,                                                  \
          OUTPUT_STRING)                                                                     \
                                                                                             \
    ENTRY(KeysName,                                                                          \
          &Keys->File->Path->BaseNameA,                                                      \
          OUTPUT_STRING)                                                                     \
                                                                                             \
    ENTRY(NumberOfKeys,                                                                      \
          Keys->NumberOfElements.QuadPart,                                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(NumberOfEdges,                                                                     \
          Table->IndexSize,                                                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(NumberOfVertices,                                                                  \
          Table->HashSize,                                                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(Algorithm,                                                                         \
          AlgorithmNames[Context->AlgorithmId],                                              \
          OUTPUT_UNICODE_STRING_FAST)                                                        \
                                                                                             \
    ENTRY(HashFunction,                                                                      \
          HashFunctionNames[Context->HashFunctionId],                                        \
          OUTPUT_UNICODE_STRING_FAST)                                                        \
                                                                                             \
    ENTRY(MaskFunction,                                                                      \
          MaskFunctionNames[Context->MaskFunctionId],                                        \
          OUTPUT_UNICODE_STRING_FAST)                                                        \
                                                                                             \
    ENTRY(BuildType,                                                                         \
          PerfectHashBuildConfigString,                                                      \
          OUTPUT_CSTR)                                                                       \
                                                                                             \
    ENTRY(Version,                                                                           \
          VER_PRODUCTVERSION_STR,                                                            \
          OUTPUT_CSTR)                                                                       \
                                                                                             \
    ENTRY(MaximumConcurrency,                                                                \
          Context->MaximumConcurrency,                                                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(SolutionFound,                                                                     \
          (TableCreateResult == S_OK ? 'Y' : 'N'),                                           \
          OUTPUT_CHR)                                                                        \
                                                                                             \
    ENTRY(LowMemory,                                                                         \
          (TableCreateResult == PH_I_LOW_MEMORY ? 'Y' : 'N'),                                \
          OUTPUT_CHR)                                                                        \
                                                                                             \
    ENTRY(OutOfMemory,                                                                       \
          (TableCreateResult == PH_I_OUT_OF_MEMORY ? 'Y' : 'N'),                             \
          OUTPUT_CHR)                                                                        \
                                                                                             \
    ENTRY(OtherMemoryIssue,                                                                  \
          (TableCreateResult ==                                                              \
           PH_I_FAILED_TO_ALLOCATE_MEMORY_FOR_ALL_GRAPHS ? 'Y' : 'N'),                       \
          OUTPUT_CHR)                                                                        \
                                                                                             \
    ENTRY(TableCreateResult,                                                                 \
          TableCreateResult,                                                                 \
          OUTPUT_ERROR_CODE_STRING)                                                          \
                                                                                             \
    ENTRY(NumberOfSolutionsFound,                                                            \
          Context->FinishedCount,                                                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(Attempts,                                                                          \
          Context->Attempts,                                                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(FailedAttempts,                                                                    \
          Context->FailedAttempts,                                                           \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(VertexCollisionFailures,                                                           \
          Context->VertexCollisionFailures,                                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(CyclicGraphFailures,                                                               \
          Context->CyclicGraphFailures,                                                      \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestCoverageAttempts,                                                              \
          Context->BestCoverageAttempts,                                                     \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(TableDataUsesLargePages,                                                           \
          (Table->Flags.TableDataUsesLargePages != FALSE ? 'Y' : 'N'),                       \
          OUTPUT_CHR)                                                                        \
                                                                                             \
    ENTRY(ValuesArrayUsesLargePages,                                                         \
          (Table->Flags.ValuesArrayUsesLargePages != FALSE ? 'Y' : 'N'),                     \
          OUTPUT_CHR)                                                                        \
                                                                                             \
    ENTRY(KeysDataUsesLargePages,                                                            \
          (Keys->Flags.KeysDataUsesLargePages != FALSE ? 'Y' : 'N'),                         \
          OUTPUT_CHR)                                                                        \
                                                                                             \
    ENTRY(UseOriginalSeededHashRoutines,                                                     \
          (TableCreateFlags.UseOriginalSeededHashRoutines != FALSE ? 'Y' : 'N'),             \
          OUTPUT_CHR)                                                                        \
                                                                                             \
    ENTRY(HashAllKeysFirst,                                                                  \
          (TableCreateFlags.HashAllKeysFirst != FALSE ? 'Y' : 'N'),                          \
          OUTPUT_CHR)                                                                        \
                                                                                             \
    ENTRY(EnableWriteCombineForVertexPairs,                                                  \
          (TableCreateFlags.EnableWriteCombineForVertexPairs != FALSE ? 'Y' : 'N'),          \
          OUTPUT_CHR)                                                                        \
                                                                                             \
    ENTRY(RemoveWriteCombineAfterSuccessfulHashKeys,                                         \
          (TableCreateFlags.RemoveWriteCombineAfterSuccessfulHashKeys != FALSE ? 'Y' : 'N'), \
          OUTPUT_CHR)                                                                        \
                                                                                             \
    ENTRY(VertexPairsArrayUsesLargePages,                                                    \
          (Table->Flags.VertexPairsArrayUsesLargePages != FALSE ? 'Y' : 'N'),                \
          OUTPUT_CHR)                                                                        \
                                                                                             \
    ENTRY(GraphRegisterSolvedTsxSuccessCount,                                                \
          Context->GraphRegisterSolvedTsxSuccess,                                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(GraphRegisterSolvedTsxStartedCount,                                                \
          Context->GraphRegisterSolvedTsxStarted,                                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(GraphRegisterSolvedTsxRetryCount,                                                  \
          Context->GraphRegisterSolvedTsxRetry,                                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(GraphRegisterSolvedTsxFailedCount,                                                 \
          Context->GraphRegisterSolvedTsxFailed,                                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(UsePreviousTableSize,                                                              \
          (TableCreateFlags.UsePreviousTableSize == TRUE ?                                   \
           'Y' : 'N'),                                                                       \
          OUTPUT_CHR)                                                                        \
                                                                                             \
    ENTRY(ClampNumberOfEdges,                                                                \
          (TableCreateFlags.ClampNumberOfEdges == TRUE ?                                     \
           'Y' : 'N'),                                                                       \
          OUTPUT_CHR)                                                                        \
                                                                                             \
    ENTRY(HasSeedMaskCounts,                                                                 \
          (((Table->TableCreateParameters != NULL) &&                                        \
            (Table->TableCreateParameters->Flags.HasSeedMaskCounts                           \
             != FALSE)) ?                                                                    \
           'Y' : 'N'),                                                                       \
          OUTPUT_CHR)                                                                        \
                                                                                             \
    ENTRY(InitialNumberOfTableResizes,                                                       \
          Context->InitialResizes,                                                           \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(NumberOfTableResizeEvents,                                                         \
          Context->NumberOfTableResizeEvents,                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(MaximumGraphTraversalDepth,                                                        \
          Table->MaximumGraphTraversalDepth,                                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(AddKeysElapsedCycles,                                                              \
          Table->AddKeysElapsedCycles.QuadPart,                                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(HashKeysElapsedCycles,                                                             \
          Table->HashKeysElapsedCycles.QuadPart,                                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(AddHashedKeysElapsedCycles,                                                        \
          Table->AddHashedKeysElapsedCycles.QuadPart,                                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(AddKeysElapsedMicroseconds,                                                        \
          Table->AddKeysElapsedMicroseconds.QuadPart,                                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(HashKeysElapsedMicroseconds,                                                       \
          Table->HashKeysElapsedMicroseconds.QuadPart,                                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(AddHashedKeysElapsedMicroseconds,                                                  \
          Table->AddHashedKeysElapsedMicroseconds.QuadPart,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(SolveMicroseconds,                                                                 \
          Context->SolveElapsedMicroseconds.QuadPart,                                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(VerifyMicroseconds,                                                                \
          Context->VerifyElapsedMicroseconds.QuadPart,                                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BenchmarkWarmups,                                                                  \
          Table->BenchmarkWarmups,                                                           \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BenchmarkAttempts,                                                                 \
          Table->BenchmarkAttempts,                                                          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BenchmarkIterationsPerAttempt,                                                     \
          Table->BenchmarkIterations,                                                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(SeededHashMinimumCyclesPerAttempt,                                                 \
          Table->SeededHashTimestamp.MinimumCycles.QuadPart,                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(SeededHashMinimumNanosecondsPerAttempt,                                            \
          Table->SeededHashTimestamp.MinimumNanoseconds.QuadPart,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(NullSeededHashMinimumCyclesPerAttempt,                                             \
          Table->NullSeededHashTimestamp.MinimumCycles.QuadPart,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(NullSeededHashMinimumNanosecondsPerAttempt,                                        \
          Table->NullSeededHashTimestamp.MinimumNanoseconds.QuadPart,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(DeltaHashMinimumCycles,                                                            \
          Table->SeededHashTimestamp.MinimumCycles.QuadPart -                                \
          Table->NullSeededHashTimestamp.MinimumCycles.QuadPart,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(DeltaHashMinimumNanoseconds,                                                       \
          Table->SeededHashTimestamp.MinimumNanoseconds.QuadPart -                           \
          Table->NullSeededHashTimestamp.MinimumNanoseconds.QuadPart,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(DeltaHashMinimumNanosecondsPerIteration,                                           \
          Table->BenchmarkIterations == 0 ? 0 : (                                            \
            (Table->SeededHashTimestamp.MinimumNanoseconds.QuadPart -                        \
             Table->NullSeededHashTimestamp.MinimumNanoseconds.QuadPart) /                   \
             Table->BenchmarkIterations                                                      \
          ),                                                                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(SlowIndexMinimumCycles,                                                            \
          Table->SlowIndexTimestamp.MinimumCycles.QuadPart,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(SlowIndexMinimumNanoseconds,                                                       \
          Table->SlowIndexTimestamp.MinimumNanoseconds.QuadPart,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(SlowIndexMinimumNanosecondsPerIteration,                                           \
          Table->BenchmarkIterations == 0 ? 0 : (                                            \
            (Table->SlowIndexTimestamp.MinimumNanoseconds.QuadPart -                         \
             Table->NullSeededHashTimestamp.MinimumNanoseconds.QuadPart) /                   \
             Table->BenchmarkIterations                                                      \
          ),                                                                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(NumberOfSeeds,                                                                     \
          HashRoutineNumberOfSeeds[Context->HashFunctionId],                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(Seed1,                                                                             \
          (TableCreateResult == S_OK &&                                                      \
           Table->TableInfoOnDisk->NumberOfSeeds >= 1 ?                                      \
           Table->TableInfoOnDisk->Seed1 : 0),                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(Seed2,                                                                             \
          (TableCreateResult == S_OK &&                                                      \
           Table->TableInfoOnDisk->NumberOfSeeds >= 2 ?                                      \
           Table->TableInfoOnDisk->Seed2 : 0),                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(Seed3,                                                                             \
          (TableCreateResult == S_OK &&                                                      \
           Table->TableInfoOnDisk->NumberOfSeeds >= 3 ?                                      \
           Table->TableInfoOnDisk->Seed3 : 0),                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(Seed4,                                                                             \
          (TableCreateResult == S_OK &&                                                      \
           Table->TableInfoOnDisk->NumberOfSeeds >= 4 ?                                      \
           Table->TableInfoOnDisk->Seed4 : 0),                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(Seed5,                                                                             \
          (TableCreateResult == S_OK &&                                                      \
           Table->TableInfoOnDisk->NumberOfSeeds >= 5 ?                                      \
           Table->TableInfoOnDisk->Seed5 : 0),                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(Seed6,                                                                             \
          (TableCreateResult == S_OK &&                                                      \
           Table->TableInfoOnDisk->NumberOfSeeds >= 6 ?                                      \
           Table->TableInfoOnDisk->Seed6 : 0),                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(Seed7,                                                                             \
          (TableCreateResult == S_OK &&                                                      \
           Table->TableInfoOnDisk->NumberOfSeeds >= 7 ?                                      \
           Table->TableInfoOnDisk->Seed7 : 0),                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(Seed8,                                                                             \
          (TableCreateResult == S_OK &&                                                      \
           Table->TableInfoOnDisk->NumberOfSeeds >= 8 ?                                      \
           Table->TableInfoOnDisk->Seed8 : 0),                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(NumberOfUserSeeds,                                                                 \
          (Context->UserSeeds != NULL ?                                                      \
           Context->UserSeeds->NumberOfValues : 0),                                          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(UserSeed1,                                                                         \
          (Context->UserSeeds != NULL &&                                                     \
           Context->UserSeeds->NumberOfValues >= 1 ?                                         \
           Context->UserSeeds->Values[0] : 0),                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(UserSeed2,                                                                         \
          (Context->UserSeeds != NULL &&                                                     \
           Context->UserSeeds->NumberOfValues >= 2 ?                                         \
           Context->UserSeeds->Values[1] : 0),                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(UserSeed3,                                                                         \
          (Context->UserSeeds != NULL &&                                                     \
           Context->UserSeeds->NumberOfValues >= 3 ?                                         \
           Context->UserSeeds->Values[2] : 0),                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(UserSeed4,                                                                         \
          (Context->UserSeeds != NULL &&                                                     \
           Context->UserSeeds->NumberOfValues >= 4 ?                                         \
           Context->UserSeeds->Values[3] : 0),                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(UserSeed5,                                                                         \
          (Context->UserSeeds != NULL &&                                                     \
           Context->UserSeeds->NumberOfValues >= 5 ?                                         \
           Context->UserSeeds->Values[4] : 0),                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(UserSeed6,                                                                         \
          (Context->UserSeeds != NULL &&                                                     \
           Context->UserSeeds->NumberOfValues >= 6 ?                                         \
           Context->UserSeeds->Values[5] : 0),                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(UserSeed7,                                                                         \
          (Context->UserSeeds != NULL &&                                                     \
           Context->UserSeeds->NumberOfValues >= 7 ?                                         \
           Context->UserSeeds->Values[6] : 0),                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(UserSeed8,                                                                         \
          (Context->UserSeeds != NULL &&                                                     \
           Context->UserSeeds->NumberOfValues >= 8 ?                                         \
           Context->UserSeeds->Values[7] : 0),                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(SeedMask1,                                                                         \
          (Context->SeedMasks != NULL ?                                                      \
           Context->SeedMasks->Mask1 : 0),                                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(SeedMask2,                                                                         \
          (Context->SeedMasks != NULL ?                                                      \
           Context->SeedMasks->Mask2 : 0),                                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(SeedMask3,                                                                         \
          (Context->SeedMasks != NULL ?                                                      \
           Context->SeedMasks->Mask3 : 0),                                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(SeedMask4,                                                                         \
          (Context->SeedMasks != NULL ?                                                      \
           Context->SeedMasks->Mask4 : 0),                                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(SeedMask5,                                                                         \
          (Context->SeedMasks != NULL ?                                                      \
           Context->SeedMasks->Mask5 : 0),                                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(SeedMask6,                                                                         \
          (Context->SeedMasks != NULL ?                                                      \
           Context->SeedMasks->Mask6 : 0),                                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(SeedMask7,                                                                         \
          (Context->SeedMasks != NULL ?                                                      \
           Context->SeedMasks->Mask7 : 0),                                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(SeedMask8,                                                                         \
          (Context->SeedMasks != NULL ?                                                      \
           Context->SeedMasks->Mask8 : 0),                                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(Seed3Byte1MaskCounts,                                                              \
          (Context->Seed3Byte1MaskCounts != NULL ?                                           \
           &Context->Seed3Byte1MaskCounts->CountsString : 0),                                \
          OUTPUT_UNICODE_STRING_FAST)                                                        \
                                                                                             \
    ENTRY(Seed3Byte2MaskCount,                                                               \
          (Context->Seed3Byte2MaskCounts != NULL ?                                           \
           &Context->Seed3Byte2MaskCounts->CountsString : 0),                                \
          OUTPUT_UNICODE_STRING_FAST)                                                        \
                                                                                             \
    ENTRY(FirstGraphWins,                                                                    \
          (FirstSolvedGraphWins(Context) ? 'Y' : 'N'),                                       \
          OUTPUT_CHR)                                                                        \
                                                                                             \
    ENTRY(FindBestMemoryCoverage,                                                            \
          ((FindBestMemoryCoverage(Context) &&                                               \
           !BestMemoryCoverageForKeysSubset(Context)) ? 'Y' : 'N'),                          \
          OUTPUT_CHR)                                                                        \
                                                                                             \
    ENTRY(FindBestMemoryCoverageForKeysSubset,                                               \
          (BestMemoryCoverageForKeysSubset(Context) ? 'Y' : 'N'),                            \
          OUTPUT_CHR)                                                                        \
                                                                                             \
    ENTRY(BestCoverageType,                                                                  \
          &BestCoverageTypeNamesA[Context->BestCoverageType],                                \
          OUTPUT_STRING)                                                                     \
                                                                                             \
    ENTRY(AttemptThatFoundBestGraph,                                                         \
          Coverage->Attempt,                                                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(NewBestGraphCount,                                                                 \
          Context->NewBestGraphCount,                                                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(EqualBestGraphCount,                                                               \
          Context->EqualBestGraphCount,                                                      \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(NumberOfKeysInSubset,                                                              \
          (Context->KeysSubset ? Context->KeysSubset->NumberOfValues : 0),                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(TotalNumberOfPages,                                                                \
          Coverage->TotalNumberOfPages,                                                      \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(TotalNumberOfLargePages,                                                           \
          Coverage->TotalNumberOfLargePages,                                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(TotalNumberOfCacheLines,                                                           \
          Coverage->TotalNumberOfCacheLines,                                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(NumberOfUsedPages,                                                                 \
          Coverage->NumberOfUsedPages,                                                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(NumberOfUsedLargePages,                                                            \
          Coverage->NumberOfUsedLargePages,                                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(NumberOfUsedCacheLines,                                                            \
          Coverage->NumberOfUsedCacheLines,                                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(NumberOfEmptyPages,                                                                \
          Coverage->NumberOfEmptyPages,                                                      \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(NumberOfEmptyLargePages,                                                           \
          Coverage->NumberOfEmptyLargePages,                                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(NumberOfEmptyCacheLines,                                                           \
          Coverage->NumberOfEmptyCacheLines,                                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(FirstPageUsed,                                                                     \
          Coverage->FirstPageUsed,                                                           \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(FirstLargePageUsed,                                                                \
          Coverage->FirstLargePageUsed,                                                      \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(FirstCacheLineUsed,                                                                \
          Coverage->FirstCacheLineUsed,                                                      \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(LastPageUsed,                                                                      \
          Coverage->LastPageUsed,                                                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(LastLargePageUsed,                                                                 \
          Coverage->LastLargePageUsed,                                                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(LastCacheLineUsed,                                                                 \
          Coverage->LastCacheLineUsed,                                                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(TotalNumberOfAssigned,                                                             \
          Coverage->TotalNumberOfAssigned,                                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_0,                                           \
          Coverage->NumberOfAssignedPerCacheLineCounts[0],                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_1,                                           \
          Coverage->NumberOfAssignedPerCacheLineCounts[1],                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_2,                                           \
          Coverage->NumberOfAssignedPerCacheLineCounts[2],                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_3,                                           \
          Coverage->NumberOfAssignedPerCacheLineCounts[3],                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_4,                                           \
          Coverage->NumberOfAssignedPerCacheLineCounts[4],                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_5,                                           \
          Coverage->NumberOfAssignedPerCacheLineCounts[5],                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_6,                                           \
          Coverage->NumberOfAssignedPerCacheLineCounts[6],                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_7,                                           \
          Coverage->NumberOfAssignedPerCacheLineCounts[7],                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_8,                                           \
          Coverage->NumberOfAssignedPerCacheLineCounts[8],                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_9,                                           \
          Coverage->NumberOfAssignedPerCacheLineCounts[9],                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_10,                                          \
          Coverage->NumberOfAssignedPerCacheLineCounts[10],                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_11,                                          \
          Coverage->NumberOfAssignedPerCacheLineCounts[11],                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_12,                                          \
          Coverage->NumberOfAssignedPerCacheLineCounts[12],                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_13,                                          \
          Coverage->NumberOfAssignedPerCacheLineCounts[13],                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_14,                                          \
          Coverage->NumberOfAssignedPerCacheLineCounts[14],                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_15,                                          \
          Coverage->NumberOfAssignedPerCacheLineCounts[15],                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(CountOfCacheLinesWithNumberOfAssigned_16,                                          \
          Coverage->NumberOfAssignedPerCacheLineCounts[16],                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_Attempt,                                                                \
          Context->BestGraphInfo[0].Attempt,                                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_ElapsedMilliseconds,                                                    \
          Context->BestGraphInfo[0].ElapsedMilliseconds,                                     \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_Value,                                                                  \
          Context->BestGraphInfo[0].Value,                                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_EqualCount,                                                             \
          Context->BestGraphInfo[0].EqualCount,                                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_Attempt,                                                                \
          Context->BestGraphInfo[1].Attempt,                                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_ElapsedMilliseconds,                                                    \
          Context->BestGraphInfo[1].ElapsedMilliseconds,                                     \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_Value,                                                                  \
          Context->BestGraphInfo[1].Value,                                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_EqualCount,                                                             \
          Context->BestGraphInfo[1].EqualCount,                                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_Attempt,                                                                \
          Context->BestGraphInfo[2].Attempt,                                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_ElapsedMilliseconds,                                                    \
          Context->BestGraphInfo[2].ElapsedMilliseconds,                                     \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_Value,                                                                  \
          Context->BestGraphInfo[2].Value,                                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_EqualCount,                                                             \
          Context->BestGraphInfo[2].EqualCount,                                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_Attempt,                                                                \
          Context->BestGraphInfo[3].Attempt,                                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_ElapsedMilliseconds,                                                    \
          Context->BestGraphInfo[3].ElapsedMilliseconds,                                     \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_Value,                                                                  \
          Context->BestGraphInfo[3].Value,                                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_EqualCount,                                                             \
          Context->BestGraphInfo[3].EqualCount,                                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_Attempt,                                                                \
          Context->BestGraphInfo[4].Attempt,                                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_ElapsedMilliseconds,                                                    \
          Context->BestGraphInfo[4].ElapsedMilliseconds,                                     \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_Value,                                                                  \
          Context->BestGraphInfo[4].Value,                                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_EqualCount,                                                             \
          Context->BestGraphInfo[4].EqualCount,                                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_Attempt,                                                                \
          Context->BestGraphInfo[5].Attempt,                                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_ElapsedMilliseconds,                                                    \
          Context->BestGraphInfo[5].ElapsedMilliseconds,                                     \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_Value,                                                                  \
          Context->BestGraphInfo[5].Value,                                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_EqualCount,                                                             \
          Context->BestGraphInfo[5].EqualCount,                                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_Attempt,                                                                \
          Context->BestGraphInfo[6].Attempt,                                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_ElapsedMilliseconds,                                                    \
          Context->BestGraphInfo[6].ElapsedMilliseconds,                                     \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_Value,                                                                  \
          Context->BestGraphInfo[6].Value,                                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_EqualCount,                                                             \
          Context->BestGraphInfo[6].EqualCount,                                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_Attempt,                                                                \
          Context->BestGraphInfo[7].Attempt,                                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_ElapsedMilliseconds,                                                    \
          Context->BestGraphInfo[7].ElapsedMilliseconds,                                     \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_Value,                                                                  \
          Context->BestGraphInfo[7].Value,                                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_EqualCount,                                                             \
          Context->BestGraphInfo[7].EqualCount,                                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_Attempt,                                                                \
          Context->BestGraphInfo[8].Attempt,                                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_ElapsedMilliseconds,                                                    \
          Context->BestGraphInfo[8].ElapsedMilliseconds,                                     \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_Value,                                                                  \
          Context->BestGraphInfo[8].Value,                                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_EqualCount,                                                             \
          Context->BestGraphInfo[8].EqualCount,                                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_Attempt,                                                               \
          Context->BestGraphInfo[9].Attempt,                                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_ElapsedMilliseconds,                                                   \
          Context->BestGraphInfo[9].ElapsedMilliseconds,                                     \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_Value,                                                                 \
          Context->BestGraphInfo[9].Value,                                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_EqualCount,                                                            \
          Context->BestGraphInfo[9].EqualCount,                                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_Attempt,                                                               \
          Context->BestGraphInfo[10].Attempt,                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_ElapsedMilliseconds,                                                   \
          Context->BestGraphInfo[10].ElapsedMilliseconds,                                    \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_Value,                                                                 \
          Context->BestGraphInfo[10].Value,                                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_EqualCount,                                                            \
          Context->BestGraphInfo[10].EqualCount,                                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_Attempt,                                                               \
          Context->BestGraphInfo[11].Attempt,                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_ElapsedMilliseconds,                                                   \
          Context->BestGraphInfo[11].ElapsedMilliseconds,                                    \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_Value,                                                                 \
          Context->BestGraphInfo[11].Value,                                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_EqualCount,                                                            \
          Context->BestGraphInfo[11].EqualCount,                                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_Attempt,                                                               \
          Context->BestGraphInfo[12].Attempt,                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_ElapsedMilliseconds,                                                   \
          Context->BestGraphInfo[12].ElapsedMilliseconds,                                    \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_Value,                                                                 \
          Context->BestGraphInfo[12].Value,                                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_EqualCount,                                                            \
          Context->BestGraphInfo[12].EqualCount,                                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_Attempt,                                                               \
          Context->BestGraphInfo[13].Attempt,                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_ElapsedMilliseconds,                                                   \
          Context->BestGraphInfo[13].ElapsedMilliseconds,                                    \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_Value,                                                                 \
          Context->BestGraphInfo[13].Value,                                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_EqualCount,                                                            \
          Context->BestGraphInfo[13].EqualCount,                                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_Attempt,                                                               \
          Context->BestGraphInfo[14].Attempt,                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_ElapsedMilliseconds,                                                   \
          Context->BestGraphInfo[14].ElapsedMilliseconds,                                    \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_Value,                                                                 \
          Context->BestGraphInfo[14].Value,                                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_EqualCount,                                                            \
          Context->BestGraphInfo[14].EqualCount,                                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_Attempt,                                                               \
          Context->BestGraphInfo[15].Attempt,                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_ElapsedMilliseconds,                                                   \
          Context->BestGraphInfo[15].ElapsedMilliseconds,                                    \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_Value,                                                                 \
          Context->BestGraphInfo[15].Value,                                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_EqualCount,                                                            \
          Context->BestGraphInfo[15].EqualCount,                                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_Attempt,                                                               \
          Context->BestGraphInfo[16].Attempt,                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_ElapsedMilliseconds,                                                   \
          Context->BestGraphInfo[16].ElapsedMilliseconds,                                    \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_Value,                                                                 \
          Context->BestGraphInfo[16].Value,                                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_EqualCount,                                                            \
          Context->BestGraphInfo[16].EqualCount,                                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_Attempt,                                                               \
          Context->BestGraphInfo[17].Attempt,                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_ElapsedMilliseconds,                                                   \
          Context->BestGraphInfo[17].ElapsedMilliseconds,                                    \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_Value,                                                                 \
          Context->BestGraphInfo[17].Value,                                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_EqualCount,                                                            \
          Context->BestGraphInfo[17].EqualCount,                                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_Attempt,                                                               \
          Context->BestGraphInfo[18].Attempt,                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_ElapsedMilliseconds,                                                   \
          Context->BestGraphInfo[18].ElapsedMilliseconds,                                    \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_Value,                                                                 \
          Context->BestGraphInfo[18].Value,                                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_EqualCount,                                                            \
          Context->BestGraphInfo[18].EqualCount,                                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_Attempt,                                                               \
          Context->BestGraphInfo[19].Attempt,                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_ElapsedMilliseconds,                                                   \
          Context->BestGraphInfo[19].ElapsedMilliseconds,                                    \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_Value,                                                                 \
          Context->BestGraphInfo[19].Value,                                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_EqualCount,                                                            \
          Context->BestGraphInfo[19].EqualCount,                                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_Attempt,                                                               \
          Context->BestGraphInfo[20].Attempt,                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_ElapsedMilliseconds,                                                   \
          Context->BestGraphInfo[20].ElapsedMilliseconds,                                    \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_Value,                                                                 \
          Context->BestGraphInfo[20].Value,                                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_EqualCount,                                                            \
          Context->BestGraphInfo[20].EqualCount,                                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_Attempt,                                                               \
          Context->BestGraphInfo[21].Attempt,                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_ElapsedMilliseconds,                                                   \
          Context->BestGraphInfo[21].ElapsedMilliseconds,                                    \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_Value,                                                                 \
          Context->BestGraphInfo[21].Value,                                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_EqualCount,                                                            \
          Context->BestGraphInfo[21].EqualCount,                                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_Attempt,                                                               \
          Context->BestGraphInfo[22].Attempt,                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_ElapsedMilliseconds,                                                   \
          Context->BestGraphInfo[22].ElapsedMilliseconds,                                    \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_Value,                                                                 \
          Context->BestGraphInfo[22].Value,                                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_EqualCount,                                                            \
          Context->BestGraphInfo[22].EqualCount,                                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_Attempt,                                                               \
          Context->BestGraphInfo[23].Attempt,                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_ElapsedMilliseconds,                                                   \
          Context->BestGraphInfo[23].ElapsedMilliseconds,                                    \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_Value,                                                                 \
          Context->BestGraphInfo[23].Value,                                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_EqualCount,                                                            \
          Context->BestGraphInfo[23].EqualCount,                                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_Attempt,                                                               \
          Context->BestGraphInfo[24].Attempt,                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_ElapsedMilliseconds,                                                   \
          Context->BestGraphInfo[24].ElapsedMilliseconds,                                    \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_Value,                                                                 \
          Context->BestGraphInfo[24].Value,                                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_EqualCount,                                                            \
          Context->BestGraphInfo[24].EqualCount,                                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_Attempt,                                                               \
          Context->BestGraphInfo[25].Attempt,                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_ElapsedMilliseconds,                                                   \
          Context->BestGraphInfo[25].ElapsedMilliseconds,                                    \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_Value,                                                                 \
          Context->BestGraphInfo[25].Value,                                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_EqualCount,                                                            \
          Context->BestGraphInfo[25].EqualCount,                                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_Attempt,                                                               \
          Context->BestGraphInfo[26].Attempt,                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_ElapsedMilliseconds,                                                   \
          Context->BestGraphInfo[26].ElapsedMilliseconds,                                    \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_Value,                                                                 \
          Context->BestGraphInfo[26].Value,                                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_EqualCount,                                                            \
          Context->BestGraphInfo[26].EqualCount,                                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_Attempt,                                                               \
          Context->BestGraphInfo[27].Attempt,                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_ElapsedMilliseconds,                                                   \
          Context->BestGraphInfo[27].ElapsedMilliseconds,                                    \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_Value,                                                                 \
          Context->BestGraphInfo[27].Value,                                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_EqualCount,                                                            \
          Context->BestGraphInfo[27].EqualCount,                                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_Attempt,                                                               \
          Context->BestGraphInfo[28].Attempt,                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_ElapsedMilliseconds,                                                   \
          Context->BestGraphInfo[28].ElapsedMilliseconds,                                    \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_Value,                                                                 \
          Context->BestGraphInfo[28].Value,                                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_EqualCount,                                                            \
          Context->BestGraphInfo[28].EqualCount,                                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_Attempt,                                                               \
          Context->BestGraphInfo[29].Attempt,                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_ElapsedMilliseconds,                                                   \
          Context->BestGraphInfo[29].ElapsedMilliseconds,                                    \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_Value,                                                                 \
          Context->BestGraphInfo[29].Value,                                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_EqualCount,                                                            \
          Context->BestGraphInfo[29].EqualCount,                                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_Attempt,                                                               \
          Context->BestGraphInfo[30].Attempt,                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_ElapsedMilliseconds,                                                   \
          Context->BestGraphInfo[30].ElapsedMilliseconds,                                    \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_Value,                                                                 \
          Context->BestGraphInfo[30].Value,                                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_EqualCount,                                                            \
          Context->BestGraphInfo[30].EqualCount,                                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_Attempt,                                                               \
          Context->BestGraphInfo[31].Attempt,                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_ElapsedMilliseconds,                                                   \
          Context->BestGraphInfo[31].ElapsedMilliseconds,                                    \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_Value,                                                                 \
          Context->BestGraphInfo[31].Value,                                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_EqualCount,                                                            \
          Context->BestGraphInfo[31].EqualCount,                                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_TotalNumberOfPages,                                                     \
          Context->BestGraphInfo[0].Coverage.TotalNumberOfPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_TotalNumberOfLargePages,                                                \
          Context->BestGraphInfo[0].Coverage.TotalNumberOfLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_TotalNumberOfCacheLines,                                                \
          Context->BestGraphInfo[0].Coverage.TotalNumberOfCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_NumberOfUsedPages,                                                      \
          Context->BestGraphInfo[0].Coverage.NumberOfUsedPages,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_NumberOfUsedLargePages,                                                 \
          Context->BestGraphInfo[0].Coverage.NumberOfUsedLargePages,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_NumberOfUsedCacheLines,                                                 \
          Context->BestGraphInfo[0].Coverage.NumberOfUsedCacheLines,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_NumberOfEmptyPages,                                                     \
          Context->BestGraphInfo[0].Coverage.NumberOfEmptyPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_NumberOfEmptyLargePages,                                                \
          Context->BestGraphInfo[0].Coverage.NumberOfEmptyLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_NumberOfEmptyCacheLines,                                                \
          Context->BestGraphInfo[0].Coverage.NumberOfEmptyCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_FirstPageUsed,                                                          \
          Context->BestGraphInfo[0].Coverage.FirstPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_FirstLargePageUsed,                                                     \
          Context->BestGraphInfo[0].Coverage.FirstLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_FirstCacheLineUsed,                                                     \
          Context->BestGraphInfo[0].Coverage.FirstCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_LastPageUsed,                                                           \
          Context->BestGraphInfo[0].Coverage.LastPageUsed,                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_LastLargePageUsed,                                                      \
          Context->BestGraphInfo[0].Coverage.LastLargePageUsed,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_LastCacheLineUsed,                                                      \
          Context->BestGraphInfo[0].Coverage.LastCacheLineUsed,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_TotalNumberOfAssigned,                                                  \
          Context->BestGraphInfo[0].Coverage.TotalNumberOfAssigned,                          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_CountOfCacheLinesWithNumberOfAssigned_0,                                \
          Context->BestGraphInfo[0].Coverage.NumberOfAssignedPerCacheLineCounts[0],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_CountOfCacheLinesWithNumberOfAssigned_1,                                \
          Context->BestGraphInfo[0].Coverage.NumberOfAssignedPerCacheLineCounts[1],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_CountOfCacheLinesWithNumberOfAssigned_2,                                \
          Context->BestGraphInfo[0].Coverage.NumberOfAssignedPerCacheLineCounts[2],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_CountOfCacheLinesWithNumberOfAssigned_3,                                \
          Context->BestGraphInfo[0].Coverage.NumberOfAssignedPerCacheLineCounts[3],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_CountOfCacheLinesWithNumberOfAssigned_4,                                \
          Context->BestGraphInfo[0].Coverage.NumberOfAssignedPerCacheLineCounts[4],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_CountOfCacheLinesWithNumberOfAssigned_5,                                \
          Context->BestGraphInfo[0].Coverage.NumberOfAssignedPerCacheLineCounts[5],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_CountOfCacheLinesWithNumberOfAssigned_6,                                \
          Context->BestGraphInfo[0].Coverage.NumberOfAssignedPerCacheLineCounts[6],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_CountOfCacheLinesWithNumberOfAssigned_7,                                \
          Context->BestGraphInfo[0].Coverage.NumberOfAssignedPerCacheLineCounts[7],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_CountOfCacheLinesWithNumberOfAssigned_8,                                \
          Context->BestGraphInfo[0].Coverage.NumberOfAssignedPerCacheLineCounts[8],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_CountOfCacheLinesWithNumberOfAssigned_9,                                \
          Context->BestGraphInfo[0].Coverage.NumberOfAssignedPerCacheLineCounts[9],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_CountOfCacheLinesWithNumberOfAssigned_10,                               \
          Context->BestGraphInfo[0].Coverage.NumberOfAssignedPerCacheLineCounts[10],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_CountOfCacheLinesWithNumberOfAssigned_11,                               \
          Context->BestGraphInfo[0].Coverage.NumberOfAssignedPerCacheLineCounts[11],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_CountOfCacheLinesWithNumberOfAssigned_12,                               \
          Context->BestGraphInfo[0].Coverage.NumberOfAssignedPerCacheLineCounts[12],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_CountOfCacheLinesWithNumberOfAssigned_13,                               \
          Context->BestGraphInfo[0].Coverage.NumberOfAssignedPerCacheLineCounts[13],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_CountOfCacheLinesWithNumberOfAssigned_14,                               \
          Context->BestGraphInfo[0].Coverage.NumberOfAssignedPerCacheLineCounts[14],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_CountOfCacheLinesWithNumberOfAssigned_15,                               \
          Context->BestGraphInfo[0].Coverage.NumberOfAssignedPerCacheLineCounts[15],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_CountOfCacheLinesWithNumberOfAssigned_16,                               \
          Context->BestGraphInfo[0].Coverage.NumberOfAssignedPerCacheLineCounts[16],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_TotalNumberOfPages,                                                     \
          Context->BestGraphInfo[1].Coverage.TotalNumberOfPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_TotalNumberOfLargePages,                                                \
          Context->BestGraphInfo[1].Coverage.TotalNumberOfLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_TotalNumberOfCacheLines,                                                \
          Context->BestGraphInfo[1].Coverage.TotalNumberOfCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_NumberOfUsedPages,                                                      \
          Context->BestGraphInfo[1].Coverage.NumberOfUsedPages,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_NumberOfUsedLargePages,                                                 \
          Context->BestGraphInfo[1].Coverage.NumberOfUsedLargePages,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_NumberOfUsedCacheLines,                                                 \
          Context->BestGraphInfo[1].Coverage.NumberOfUsedCacheLines,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_NumberOfEmptyPages,                                                     \
          Context->BestGraphInfo[1].Coverage.NumberOfEmptyPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_NumberOfEmptyLargePages,                                                \
          Context->BestGraphInfo[1].Coverage.NumberOfEmptyLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_NumberOfEmptyCacheLines,                                                \
          Context->BestGraphInfo[1].Coverage.NumberOfEmptyCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_FirstPageUsed,                                                          \
          Context->BestGraphInfo[1].Coverage.FirstPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_FirstLargePageUsed,                                                     \
          Context->BestGraphInfo[1].Coverage.FirstLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_FirstCacheLineUsed,                                                     \
          Context->BestGraphInfo[1].Coverage.FirstCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_LastPageUsed,                                                           \
          Context->BestGraphInfo[1].Coverage.LastPageUsed,                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_LastLargePageUsed,                                                      \
          Context->BestGraphInfo[1].Coverage.LastLargePageUsed,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_LastCacheLineUsed,                                                      \
          Context->BestGraphInfo[1].Coverage.LastCacheLineUsed,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_TotalNumberOfAssigned,                                                  \
          Context->BestGraphInfo[1].Coverage.TotalNumberOfAssigned,                          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_CountOfCacheLinesWithNumberOfAssigned_0,                                \
          Context->BestGraphInfo[1].Coverage.NumberOfAssignedPerCacheLineCounts[0],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_CountOfCacheLinesWithNumberOfAssigned_1,                                \
          Context->BestGraphInfo[1].Coverage.NumberOfAssignedPerCacheLineCounts[1],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_CountOfCacheLinesWithNumberOfAssigned_2,                                \
          Context->BestGraphInfo[1].Coverage.NumberOfAssignedPerCacheLineCounts[2],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_CountOfCacheLinesWithNumberOfAssigned_3,                                \
          Context->BestGraphInfo[1].Coverage.NumberOfAssignedPerCacheLineCounts[3],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_CountOfCacheLinesWithNumberOfAssigned_4,                                \
          Context->BestGraphInfo[1].Coverage.NumberOfAssignedPerCacheLineCounts[4],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_CountOfCacheLinesWithNumberOfAssigned_5,                                \
          Context->BestGraphInfo[1].Coverage.NumberOfAssignedPerCacheLineCounts[5],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_CountOfCacheLinesWithNumberOfAssigned_6,                                \
          Context->BestGraphInfo[1].Coverage.NumberOfAssignedPerCacheLineCounts[6],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_CountOfCacheLinesWithNumberOfAssigned_7,                                \
          Context->BestGraphInfo[1].Coverage.NumberOfAssignedPerCacheLineCounts[7],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_CountOfCacheLinesWithNumberOfAssigned_8,                                \
          Context->BestGraphInfo[1].Coverage.NumberOfAssignedPerCacheLineCounts[8],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_CountOfCacheLinesWithNumberOfAssigned_9,                                \
          Context->BestGraphInfo[1].Coverage.NumberOfAssignedPerCacheLineCounts[9],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_CountOfCacheLinesWithNumberOfAssigned_10,                               \
          Context->BestGraphInfo[1].Coverage.NumberOfAssignedPerCacheLineCounts[10],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_CountOfCacheLinesWithNumberOfAssigned_11,                               \
          Context->BestGraphInfo[1].Coverage.NumberOfAssignedPerCacheLineCounts[11],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_CountOfCacheLinesWithNumberOfAssigned_12,                               \
          Context->BestGraphInfo[1].Coverage.NumberOfAssignedPerCacheLineCounts[12],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_CountOfCacheLinesWithNumberOfAssigned_13,                               \
          Context->BestGraphInfo[1].Coverage.NumberOfAssignedPerCacheLineCounts[13],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_CountOfCacheLinesWithNumberOfAssigned_14,                               \
          Context->BestGraphInfo[1].Coverage.NumberOfAssignedPerCacheLineCounts[14],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_CountOfCacheLinesWithNumberOfAssigned_15,                               \
          Context->BestGraphInfo[1].Coverage.NumberOfAssignedPerCacheLineCounts[15],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_CountOfCacheLinesWithNumberOfAssigned_16,                               \
          Context->BestGraphInfo[1].Coverage.NumberOfAssignedPerCacheLineCounts[16],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_TotalNumberOfPages,                                                     \
          Context->BestGraphInfo[2].Coverage.TotalNumberOfPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_TotalNumberOfLargePages,                                                \
          Context->BestGraphInfo[2].Coverage.TotalNumberOfLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_TotalNumberOfCacheLines,                                                \
          Context->BestGraphInfo[2].Coverage.TotalNumberOfCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_NumberOfUsedPages,                                                      \
          Context->BestGraphInfo[2].Coverage.NumberOfUsedPages,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_NumberOfUsedLargePages,                                                 \
          Context->BestGraphInfo[2].Coverage.NumberOfUsedLargePages,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_NumberOfUsedCacheLines,                                                 \
          Context->BestGraphInfo[2].Coverage.NumberOfUsedCacheLines,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_NumberOfEmptyPages,                                                     \
          Context->BestGraphInfo[2].Coverage.NumberOfEmptyPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_NumberOfEmptyLargePages,                                                \
          Context->BestGraphInfo[2].Coverage.NumberOfEmptyLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_NumberOfEmptyCacheLines,                                                \
          Context->BestGraphInfo[2].Coverage.NumberOfEmptyCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_FirstPageUsed,                                                          \
          Context->BestGraphInfo[2].Coverage.FirstPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_FirstLargePageUsed,                                                     \
          Context->BestGraphInfo[2].Coverage.FirstLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_FirstCacheLineUsed,                                                     \
          Context->BestGraphInfo[2].Coverage.FirstCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_LastPageUsed,                                                           \
          Context->BestGraphInfo[2].Coverage.LastPageUsed,                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_LastLargePageUsed,                                                      \
          Context->BestGraphInfo[2].Coverage.LastLargePageUsed,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_LastCacheLineUsed,                                                      \
          Context->BestGraphInfo[2].Coverage.LastCacheLineUsed,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_TotalNumberOfAssigned,                                                  \
          Context->BestGraphInfo[2].Coverage.TotalNumberOfAssigned,                          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_CountOfCacheLinesWithNumberOfAssigned_0,                                \
          Context->BestGraphInfo[2].Coverage.NumberOfAssignedPerCacheLineCounts[0],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_CountOfCacheLinesWithNumberOfAssigned_1,                                \
          Context->BestGraphInfo[2].Coverage.NumberOfAssignedPerCacheLineCounts[1],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_CountOfCacheLinesWithNumberOfAssigned_2,                                \
          Context->BestGraphInfo[2].Coverage.NumberOfAssignedPerCacheLineCounts[2],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_CountOfCacheLinesWithNumberOfAssigned_3,                                \
          Context->BestGraphInfo[2].Coverage.NumberOfAssignedPerCacheLineCounts[3],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_CountOfCacheLinesWithNumberOfAssigned_4,                                \
          Context->BestGraphInfo[2].Coverage.NumberOfAssignedPerCacheLineCounts[4],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_CountOfCacheLinesWithNumberOfAssigned_5,                                \
          Context->BestGraphInfo[2].Coverage.NumberOfAssignedPerCacheLineCounts[5],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_CountOfCacheLinesWithNumberOfAssigned_6,                                \
          Context->BestGraphInfo[2].Coverage.NumberOfAssignedPerCacheLineCounts[6],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_CountOfCacheLinesWithNumberOfAssigned_7,                                \
          Context->BestGraphInfo[2].Coverage.NumberOfAssignedPerCacheLineCounts[7],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_CountOfCacheLinesWithNumberOfAssigned_8,                                \
          Context->BestGraphInfo[2].Coverage.NumberOfAssignedPerCacheLineCounts[8],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_CountOfCacheLinesWithNumberOfAssigned_9,                                \
          Context->BestGraphInfo[2].Coverage.NumberOfAssignedPerCacheLineCounts[9],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_CountOfCacheLinesWithNumberOfAssigned_10,                               \
          Context->BestGraphInfo[2].Coverage.NumberOfAssignedPerCacheLineCounts[10],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_CountOfCacheLinesWithNumberOfAssigned_11,                               \
          Context->BestGraphInfo[2].Coverage.NumberOfAssignedPerCacheLineCounts[11],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_CountOfCacheLinesWithNumberOfAssigned_12,                               \
          Context->BestGraphInfo[2].Coverage.NumberOfAssignedPerCacheLineCounts[12],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_CountOfCacheLinesWithNumberOfAssigned_13,                               \
          Context->BestGraphInfo[2].Coverage.NumberOfAssignedPerCacheLineCounts[13],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_CountOfCacheLinesWithNumberOfAssigned_14,                               \
          Context->BestGraphInfo[2].Coverage.NumberOfAssignedPerCacheLineCounts[14],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_CountOfCacheLinesWithNumberOfAssigned_15,                               \
          Context->BestGraphInfo[2].Coverage.NumberOfAssignedPerCacheLineCounts[15],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_CountOfCacheLinesWithNumberOfAssigned_16,                               \
          Context->BestGraphInfo[2].Coverage.NumberOfAssignedPerCacheLineCounts[16],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_TotalNumberOfPages,                                                     \
          Context->BestGraphInfo[3].Coverage.TotalNumberOfPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_TotalNumberOfLargePages,                                                \
          Context->BestGraphInfo[3].Coverage.TotalNumberOfLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_TotalNumberOfCacheLines,                                                \
          Context->BestGraphInfo[3].Coverage.TotalNumberOfCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_NumberOfUsedPages,                                                      \
          Context->BestGraphInfo[3].Coverage.NumberOfUsedPages,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_NumberOfUsedLargePages,                                                 \
          Context->BestGraphInfo[3].Coverage.NumberOfUsedLargePages,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_NumberOfUsedCacheLines,                                                 \
          Context->BestGraphInfo[3].Coverage.NumberOfUsedCacheLines,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_NumberOfEmptyPages,                                                     \
          Context->BestGraphInfo[3].Coverage.NumberOfEmptyPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_NumberOfEmptyLargePages,                                                \
          Context->BestGraphInfo[3].Coverage.NumberOfEmptyLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_NumberOfEmptyCacheLines,                                                \
          Context->BestGraphInfo[3].Coverage.NumberOfEmptyCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_FirstPageUsed,                                                          \
          Context->BestGraphInfo[3].Coverage.FirstPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_FirstLargePageUsed,                                                     \
          Context->BestGraphInfo[3].Coverage.FirstLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_FirstCacheLineUsed,                                                     \
          Context->BestGraphInfo[3].Coverage.FirstCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_LastPageUsed,                                                           \
          Context->BestGraphInfo[3].Coverage.LastPageUsed,                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_LastLargePageUsed,                                                      \
          Context->BestGraphInfo[3].Coverage.LastLargePageUsed,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_LastCacheLineUsed,                                                      \
          Context->BestGraphInfo[3].Coverage.LastCacheLineUsed,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_TotalNumberOfAssigned,                                                  \
          Context->BestGraphInfo[3].Coverage.TotalNumberOfAssigned,                          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_CountOfCacheLinesWithNumberOfAssigned_0,                                \
          Context->BestGraphInfo[3].Coverage.NumberOfAssignedPerCacheLineCounts[0],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_CountOfCacheLinesWithNumberOfAssigned_1,                                \
          Context->BestGraphInfo[3].Coverage.NumberOfAssignedPerCacheLineCounts[1],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_CountOfCacheLinesWithNumberOfAssigned_2,                                \
          Context->BestGraphInfo[3].Coverage.NumberOfAssignedPerCacheLineCounts[2],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_CountOfCacheLinesWithNumberOfAssigned_3,                                \
          Context->BestGraphInfo[3].Coverage.NumberOfAssignedPerCacheLineCounts[3],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_CountOfCacheLinesWithNumberOfAssigned_4,                                \
          Context->BestGraphInfo[3].Coverage.NumberOfAssignedPerCacheLineCounts[4],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_CountOfCacheLinesWithNumberOfAssigned_5,                                \
          Context->BestGraphInfo[3].Coverage.NumberOfAssignedPerCacheLineCounts[5],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_CountOfCacheLinesWithNumberOfAssigned_6,                                \
          Context->BestGraphInfo[3].Coverage.NumberOfAssignedPerCacheLineCounts[6],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_CountOfCacheLinesWithNumberOfAssigned_7,                                \
          Context->BestGraphInfo[3].Coverage.NumberOfAssignedPerCacheLineCounts[7],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_CountOfCacheLinesWithNumberOfAssigned_8,                                \
          Context->BestGraphInfo[3].Coverage.NumberOfAssignedPerCacheLineCounts[8],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_CountOfCacheLinesWithNumberOfAssigned_9,                                \
          Context->BestGraphInfo[3].Coverage.NumberOfAssignedPerCacheLineCounts[9],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_CountOfCacheLinesWithNumberOfAssigned_10,                               \
          Context->BestGraphInfo[3].Coverage.NumberOfAssignedPerCacheLineCounts[10],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_CountOfCacheLinesWithNumberOfAssigned_11,                               \
          Context->BestGraphInfo[3].Coverage.NumberOfAssignedPerCacheLineCounts[11],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_CountOfCacheLinesWithNumberOfAssigned_12,                               \
          Context->BestGraphInfo[3].Coverage.NumberOfAssignedPerCacheLineCounts[12],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_CountOfCacheLinesWithNumberOfAssigned_13,                               \
          Context->BestGraphInfo[3].Coverage.NumberOfAssignedPerCacheLineCounts[13],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_CountOfCacheLinesWithNumberOfAssigned_14,                               \
          Context->BestGraphInfo[3].Coverage.NumberOfAssignedPerCacheLineCounts[14],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_CountOfCacheLinesWithNumberOfAssigned_15,                               \
          Context->BestGraphInfo[3].Coverage.NumberOfAssignedPerCacheLineCounts[15],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_CountOfCacheLinesWithNumberOfAssigned_16,                               \
          Context->BestGraphInfo[3].Coverage.NumberOfAssignedPerCacheLineCounts[16],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_TotalNumberOfPages,                                                     \
          Context->BestGraphInfo[4].Coverage.TotalNumberOfPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_TotalNumberOfLargePages,                                                \
          Context->BestGraphInfo[4].Coverage.TotalNumberOfLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_TotalNumberOfCacheLines,                                                \
          Context->BestGraphInfo[4].Coverage.TotalNumberOfCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_NumberOfUsedPages,                                                      \
          Context->BestGraphInfo[4].Coverage.NumberOfUsedPages,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_NumberOfUsedLargePages,                                                 \
          Context->BestGraphInfo[4].Coverage.NumberOfUsedLargePages,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_NumberOfUsedCacheLines,                                                 \
          Context->BestGraphInfo[4].Coverage.NumberOfUsedCacheLines,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_NumberOfEmptyPages,                                                     \
          Context->BestGraphInfo[4].Coverage.NumberOfEmptyPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_NumberOfEmptyLargePages,                                                \
          Context->BestGraphInfo[4].Coverage.NumberOfEmptyLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_NumberOfEmptyCacheLines,                                                \
          Context->BestGraphInfo[4].Coverage.NumberOfEmptyCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_FirstPageUsed,                                                          \
          Context->BestGraphInfo[4].Coverage.FirstPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_FirstLargePageUsed,                                                     \
          Context->BestGraphInfo[4].Coverage.FirstLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_FirstCacheLineUsed,                                                     \
          Context->BestGraphInfo[4].Coverage.FirstCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_LastPageUsed,                                                           \
          Context->BestGraphInfo[4].Coverage.LastPageUsed,                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_LastLargePageUsed,                                                      \
          Context->BestGraphInfo[4].Coverage.LastLargePageUsed,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_LastCacheLineUsed,                                                      \
          Context->BestGraphInfo[4].Coverage.LastCacheLineUsed,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_TotalNumberOfAssigned,                                                  \
          Context->BestGraphInfo[4].Coverage.TotalNumberOfAssigned,                          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_CountOfCacheLinesWithNumberOfAssigned_0,                                \
          Context->BestGraphInfo[4].Coverage.NumberOfAssignedPerCacheLineCounts[0],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_CountOfCacheLinesWithNumberOfAssigned_1,                                \
          Context->BestGraphInfo[4].Coverage.NumberOfAssignedPerCacheLineCounts[1],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_CountOfCacheLinesWithNumberOfAssigned_2,                                \
          Context->BestGraphInfo[4].Coverage.NumberOfAssignedPerCacheLineCounts[2],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_CountOfCacheLinesWithNumberOfAssigned_3,                                \
          Context->BestGraphInfo[4].Coverage.NumberOfAssignedPerCacheLineCounts[3],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_CountOfCacheLinesWithNumberOfAssigned_4,                                \
          Context->BestGraphInfo[4].Coverage.NumberOfAssignedPerCacheLineCounts[4],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_CountOfCacheLinesWithNumberOfAssigned_5,                                \
          Context->BestGraphInfo[4].Coverage.NumberOfAssignedPerCacheLineCounts[5],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_CountOfCacheLinesWithNumberOfAssigned_6,                                \
          Context->BestGraphInfo[4].Coverage.NumberOfAssignedPerCacheLineCounts[6],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_CountOfCacheLinesWithNumberOfAssigned_7,                                \
          Context->BestGraphInfo[4].Coverage.NumberOfAssignedPerCacheLineCounts[7],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_CountOfCacheLinesWithNumberOfAssigned_8,                                \
          Context->BestGraphInfo[4].Coverage.NumberOfAssignedPerCacheLineCounts[8],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_CountOfCacheLinesWithNumberOfAssigned_9,                                \
          Context->BestGraphInfo[4].Coverage.NumberOfAssignedPerCacheLineCounts[9],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_CountOfCacheLinesWithNumberOfAssigned_10,                               \
          Context->BestGraphInfo[4].Coverage.NumberOfAssignedPerCacheLineCounts[10],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_CountOfCacheLinesWithNumberOfAssigned_11,                               \
          Context->BestGraphInfo[4].Coverage.NumberOfAssignedPerCacheLineCounts[11],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_CountOfCacheLinesWithNumberOfAssigned_12,                               \
          Context->BestGraphInfo[4].Coverage.NumberOfAssignedPerCacheLineCounts[12],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_CountOfCacheLinesWithNumberOfAssigned_13,                               \
          Context->BestGraphInfo[4].Coverage.NumberOfAssignedPerCacheLineCounts[13],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_CountOfCacheLinesWithNumberOfAssigned_14,                               \
          Context->BestGraphInfo[4].Coverage.NumberOfAssignedPerCacheLineCounts[14],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_CountOfCacheLinesWithNumberOfAssigned_15,                               \
          Context->BestGraphInfo[4].Coverage.NumberOfAssignedPerCacheLineCounts[15],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_CountOfCacheLinesWithNumberOfAssigned_16,                               \
          Context->BestGraphInfo[4].Coverage.NumberOfAssignedPerCacheLineCounts[16],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_TotalNumberOfPages,                                                     \
          Context->BestGraphInfo[5].Coverage.TotalNumberOfPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_TotalNumberOfLargePages,                                                \
          Context->BestGraphInfo[5].Coverage.TotalNumberOfLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_TotalNumberOfCacheLines,                                                \
          Context->BestGraphInfo[5].Coverage.TotalNumberOfCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_NumberOfUsedPages,                                                      \
          Context->BestGraphInfo[5].Coverage.NumberOfUsedPages,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_NumberOfUsedLargePages,                                                 \
          Context->BestGraphInfo[5].Coverage.NumberOfUsedLargePages,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_NumberOfUsedCacheLines,                                                 \
          Context->BestGraphInfo[5].Coverage.NumberOfUsedCacheLines,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_NumberOfEmptyPages,                                                     \
          Context->BestGraphInfo[5].Coverage.NumberOfEmptyPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_NumberOfEmptyLargePages,                                                \
          Context->BestGraphInfo[5].Coverage.NumberOfEmptyLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_NumberOfEmptyCacheLines,                                                \
          Context->BestGraphInfo[5].Coverage.NumberOfEmptyCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_FirstPageUsed,                                                          \
          Context->BestGraphInfo[5].Coverage.FirstPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_FirstLargePageUsed,                                                     \
          Context->BestGraphInfo[5].Coverage.FirstLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_FirstCacheLineUsed,                                                     \
          Context->BestGraphInfo[5].Coverage.FirstCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_LastPageUsed,                                                           \
          Context->BestGraphInfo[5].Coverage.LastPageUsed,                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_LastLargePageUsed,                                                      \
          Context->BestGraphInfo[5].Coverage.LastLargePageUsed,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_LastCacheLineUsed,                                                      \
          Context->BestGraphInfo[5].Coverage.LastCacheLineUsed,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_TotalNumberOfAssigned,                                                  \
          Context->BestGraphInfo[5].Coverage.TotalNumberOfAssigned,                          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_CountOfCacheLinesWithNumberOfAssigned_0,                                \
          Context->BestGraphInfo[5].Coverage.NumberOfAssignedPerCacheLineCounts[0],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_CountOfCacheLinesWithNumberOfAssigned_1,                                \
          Context->BestGraphInfo[5].Coverage.NumberOfAssignedPerCacheLineCounts[1],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_CountOfCacheLinesWithNumberOfAssigned_2,                                \
          Context->BestGraphInfo[5].Coverage.NumberOfAssignedPerCacheLineCounts[2],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_CountOfCacheLinesWithNumberOfAssigned_3,                                \
          Context->BestGraphInfo[5].Coverage.NumberOfAssignedPerCacheLineCounts[3],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_CountOfCacheLinesWithNumberOfAssigned_4,                                \
          Context->BestGraphInfo[5].Coverage.NumberOfAssignedPerCacheLineCounts[4],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_CountOfCacheLinesWithNumberOfAssigned_5,                                \
          Context->BestGraphInfo[5].Coverage.NumberOfAssignedPerCacheLineCounts[5],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_CountOfCacheLinesWithNumberOfAssigned_6,                                \
          Context->BestGraphInfo[5].Coverage.NumberOfAssignedPerCacheLineCounts[6],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_CountOfCacheLinesWithNumberOfAssigned_7,                                \
          Context->BestGraphInfo[5].Coverage.NumberOfAssignedPerCacheLineCounts[7],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_CountOfCacheLinesWithNumberOfAssigned_8,                                \
          Context->BestGraphInfo[5].Coverage.NumberOfAssignedPerCacheLineCounts[8],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_CountOfCacheLinesWithNumberOfAssigned_9,                                \
          Context->BestGraphInfo[5].Coverage.NumberOfAssignedPerCacheLineCounts[9],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_CountOfCacheLinesWithNumberOfAssigned_10,                               \
          Context->BestGraphInfo[5].Coverage.NumberOfAssignedPerCacheLineCounts[10],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_CountOfCacheLinesWithNumberOfAssigned_11,                               \
          Context->BestGraphInfo[5].Coverage.NumberOfAssignedPerCacheLineCounts[11],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_CountOfCacheLinesWithNumberOfAssigned_12,                               \
          Context->BestGraphInfo[5].Coverage.NumberOfAssignedPerCacheLineCounts[12],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_CountOfCacheLinesWithNumberOfAssigned_13,                               \
          Context->BestGraphInfo[5].Coverage.NumberOfAssignedPerCacheLineCounts[13],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_CountOfCacheLinesWithNumberOfAssigned_14,                               \
          Context->BestGraphInfo[5].Coverage.NumberOfAssignedPerCacheLineCounts[14],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_CountOfCacheLinesWithNumberOfAssigned_15,                               \
          Context->BestGraphInfo[5].Coverage.NumberOfAssignedPerCacheLineCounts[15],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_CountOfCacheLinesWithNumberOfAssigned_16,                               \
          Context->BestGraphInfo[5].Coverage.NumberOfAssignedPerCacheLineCounts[16],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_TotalNumberOfPages,                                                     \
          Context->BestGraphInfo[6].Coverage.TotalNumberOfPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_TotalNumberOfLargePages,                                                \
          Context->BestGraphInfo[6].Coverage.TotalNumberOfLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_TotalNumberOfCacheLines,                                                \
          Context->BestGraphInfo[6].Coverage.TotalNumberOfCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_NumberOfUsedPages,                                                      \
          Context->BestGraphInfo[6].Coverage.NumberOfUsedPages,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_NumberOfUsedLargePages,                                                 \
          Context->BestGraphInfo[6].Coverage.NumberOfUsedLargePages,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_NumberOfUsedCacheLines,                                                 \
          Context->BestGraphInfo[6].Coverage.NumberOfUsedCacheLines,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_NumberOfEmptyPages,                                                     \
          Context->BestGraphInfo[6].Coverage.NumberOfEmptyPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_NumberOfEmptyLargePages,                                                \
          Context->BestGraphInfo[6].Coverage.NumberOfEmptyLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_NumberOfEmptyCacheLines,                                                \
          Context->BestGraphInfo[6].Coverage.NumberOfEmptyCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_FirstPageUsed,                                                          \
          Context->BestGraphInfo[6].Coverage.FirstPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_FirstLargePageUsed,                                                     \
          Context->BestGraphInfo[6].Coverage.FirstLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_FirstCacheLineUsed,                                                     \
          Context->BestGraphInfo[6].Coverage.FirstCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_LastPageUsed,                                                           \
          Context->BestGraphInfo[6].Coverage.LastPageUsed,                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_LastLargePageUsed,                                                      \
          Context->BestGraphInfo[6].Coverage.LastLargePageUsed,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_LastCacheLineUsed,                                                      \
          Context->BestGraphInfo[6].Coverage.LastCacheLineUsed,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_TotalNumberOfAssigned,                                                  \
          Context->BestGraphInfo[6].Coverage.TotalNumberOfAssigned,                          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_CountOfCacheLinesWithNumberOfAssigned_0,                                \
          Context->BestGraphInfo[6].Coverage.NumberOfAssignedPerCacheLineCounts[0],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_CountOfCacheLinesWithNumberOfAssigned_1,                                \
          Context->BestGraphInfo[6].Coverage.NumberOfAssignedPerCacheLineCounts[1],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_CountOfCacheLinesWithNumberOfAssigned_2,                                \
          Context->BestGraphInfo[6].Coverage.NumberOfAssignedPerCacheLineCounts[2],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_CountOfCacheLinesWithNumberOfAssigned_3,                                \
          Context->BestGraphInfo[6].Coverage.NumberOfAssignedPerCacheLineCounts[3],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_CountOfCacheLinesWithNumberOfAssigned_4,                                \
          Context->BestGraphInfo[6].Coverage.NumberOfAssignedPerCacheLineCounts[4],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_CountOfCacheLinesWithNumberOfAssigned_5,                                \
          Context->BestGraphInfo[6].Coverage.NumberOfAssignedPerCacheLineCounts[5],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_CountOfCacheLinesWithNumberOfAssigned_6,                                \
          Context->BestGraphInfo[6].Coverage.NumberOfAssignedPerCacheLineCounts[6],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_CountOfCacheLinesWithNumberOfAssigned_7,                                \
          Context->BestGraphInfo[6].Coverage.NumberOfAssignedPerCacheLineCounts[7],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_CountOfCacheLinesWithNumberOfAssigned_8,                                \
          Context->BestGraphInfo[6].Coverage.NumberOfAssignedPerCacheLineCounts[8],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_CountOfCacheLinesWithNumberOfAssigned_9,                                \
          Context->BestGraphInfo[6].Coverage.NumberOfAssignedPerCacheLineCounts[9],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_CountOfCacheLinesWithNumberOfAssigned_10,                               \
          Context->BestGraphInfo[6].Coverage.NumberOfAssignedPerCacheLineCounts[10],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_CountOfCacheLinesWithNumberOfAssigned_11,                               \
          Context->BestGraphInfo[6].Coverage.NumberOfAssignedPerCacheLineCounts[11],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_CountOfCacheLinesWithNumberOfAssigned_12,                               \
          Context->BestGraphInfo[6].Coverage.NumberOfAssignedPerCacheLineCounts[12],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_CountOfCacheLinesWithNumberOfAssigned_13,                               \
          Context->BestGraphInfo[6].Coverage.NumberOfAssignedPerCacheLineCounts[13],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_CountOfCacheLinesWithNumberOfAssigned_14,                               \
          Context->BestGraphInfo[6].Coverage.NumberOfAssignedPerCacheLineCounts[14],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_CountOfCacheLinesWithNumberOfAssigned_15,                               \
          Context->BestGraphInfo[6].Coverage.NumberOfAssignedPerCacheLineCounts[15],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_CountOfCacheLinesWithNumberOfAssigned_16,                               \
          Context->BestGraphInfo[6].Coverage.NumberOfAssignedPerCacheLineCounts[16],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_TotalNumberOfPages,                                                     \
          Context->BestGraphInfo[7].Coverage.TotalNumberOfPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_TotalNumberOfLargePages,                                                \
          Context->BestGraphInfo[7].Coverage.TotalNumberOfLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_TotalNumberOfCacheLines,                                                \
          Context->BestGraphInfo[7].Coverage.TotalNumberOfCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_NumberOfUsedPages,                                                      \
          Context->BestGraphInfo[7].Coverage.NumberOfUsedPages,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_NumberOfUsedLargePages,                                                 \
          Context->BestGraphInfo[7].Coverage.NumberOfUsedLargePages,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_NumberOfUsedCacheLines,                                                 \
          Context->BestGraphInfo[7].Coverage.NumberOfUsedCacheLines,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_NumberOfEmptyPages,                                                     \
          Context->BestGraphInfo[7].Coverage.NumberOfEmptyPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_NumberOfEmptyLargePages,                                                \
          Context->BestGraphInfo[7].Coverage.NumberOfEmptyLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_NumberOfEmptyCacheLines,                                                \
          Context->BestGraphInfo[7].Coverage.NumberOfEmptyCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_FirstPageUsed,                                                          \
          Context->BestGraphInfo[7].Coverage.FirstPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_FirstLargePageUsed,                                                     \
          Context->BestGraphInfo[7].Coverage.FirstLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_FirstCacheLineUsed,                                                     \
          Context->BestGraphInfo[7].Coverage.FirstCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_LastPageUsed,                                                           \
          Context->BestGraphInfo[7].Coverage.LastPageUsed,                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_LastLargePageUsed,                                                      \
          Context->BestGraphInfo[7].Coverage.LastLargePageUsed,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_LastCacheLineUsed,                                                      \
          Context->BestGraphInfo[7].Coverage.LastCacheLineUsed,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_TotalNumberOfAssigned,                                                  \
          Context->BestGraphInfo[7].Coverage.TotalNumberOfAssigned,                          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_CountOfCacheLinesWithNumberOfAssigned_0,                                \
          Context->BestGraphInfo[7].Coverage.NumberOfAssignedPerCacheLineCounts[0],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_CountOfCacheLinesWithNumberOfAssigned_1,                                \
          Context->BestGraphInfo[7].Coverage.NumberOfAssignedPerCacheLineCounts[1],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_CountOfCacheLinesWithNumberOfAssigned_2,                                \
          Context->BestGraphInfo[7].Coverage.NumberOfAssignedPerCacheLineCounts[2],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_CountOfCacheLinesWithNumberOfAssigned_3,                                \
          Context->BestGraphInfo[7].Coverage.NumberOfAssignedPerCacheLineCounts[3],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_CountOfCacheLinesWithNumberOfAssigned_4,                                \
          Context->BestGraphInfo[7].Coverage.NumberOfAssignedPerCacheLineCounts[4],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_CountOfCacheLinesWithNumberOfAssigned_5,                                \
          Context->BestGraphInfo[7].Coverage.NumberOfAssignedPerCacheLineCounts[5],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_CountOfCacheLinesWithNumberOfAssigned_6,                                \
          Context->BestGraphInfo[7].Coverage.NumberOfAssignedPerCacheLineCounts[6],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_CountOfCacheLinesWithNumberOfAssigned_7,                                \
          Context->BestGraphInfo[7].Coverage.NumberOfAssignedPerCacheLineCounts[7],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_CountOfCacheLinesWithNumberOfAssigned_8,                                \
          Context->BestGraphInfo[7].Coverage.NumberOfAssignedPerCacheLineCounts[8],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_CountOfCacheLinesWithNumberOfAssigned_9,                                \
          Context->BestGraphInfo[7].Coverage.NumberOfAssignedPerCacheLineCounts[9],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_CountOfCacheLinesWithNumberOfAssigned_10,                               \
          Context->BestGraphInfo[7].Coverage.NumberOfAssignedPerCacheLineCounts[10],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_CountOfCacheLinesWithNumberOfAssigned_11,                               \
          Context->BestGraphInfo[7].Coverage.NumberOfAssignedPerCacheLineCounts[11],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_CountOfCacheLinesWithNumberOfAssigned_12,                               \
          Context->BestGraphInfo[7].Coverage.NumberOfAssignedPerCacheLineCounts[12],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_CountOfCacheLinesWithNumberOfAssigned_13,                               \
          Context->BestGraphInfo[7].Coverage.NumberOfAssignedPerCacheLineCounts[13],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_CountOfCacheLinesWithNumberOfAssigned_14,                               \
          Context->BestGraphInfo[7].Coverage.NumberOfAssignedPerCacheLineCounts[14],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_CountOfCacheLinesWithNumberOfAssigned_15,                               \
          Context->BestGraphInfo[7].Coverage.NumberOfAssignedPerCacheLineCounts[15],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_CountOfCacheLinesWithNumberOfAssigned_16,                               \
          Context->BestGraphInfo[7].Coverage.NumberOfAssignedPerCacheLineCounts[16],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_TotalNumberOfPages,                                                     \
          Context->BestGraphInfo[8].Coverage.TotalNumberOfPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_TotalNumberOfLargePages,                                                \
          Context->BestGraphInfo[8].Coverage.TotalNumberOfLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_TotalNumberOfCacheLines,                                                \
          Context->BestGraphInfo[8].Coverage.TotalNumberOfCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_NumberOfUsedPages,                                                      \
          Context->BestGraphInfo[8].Coverage.NumberOfUsedPages,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_NumberOfUsedLargePages,                                                 \
          Context->BestGraphInfo[8].Coverage.NumberOfUsedLargePages,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_NumberOfUsedCacheLines,                                                 \
          Context->BestGraphInfo[8].Coverage.NumberOfUsedCacheLines,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_NumberOfEmptyPages,                                                     \
          Context->BestGraphInfo[8].Coverage.NumberOfEmptyPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_NumberOfEmptyLargePages,                                                \
          Context->BestGraphInfo[8].Coverage.NumberOfEmptyLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_NumberOfEmptyCacheLines,                                                \
          Context->BestGraphInfo[8].Coverage.NumberOfEmptyCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_FirstPageUsed,                                                          \
          Context->BestGraphInfo[8].Coverage.FirstPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_FirstLargePageUsed,                                                     \
          Context->BestGraphInfo[8].Coverage.FirstLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_FirstCacheLineUsed,                                                     \
          Context->BestGraphInfo[8].Coverage.FirstCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_LastPageUsed,                                                           \
          Context->BestGraphInfo[8].Coverage.LastPageUsed,                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_LastLargePageUsed,                                                      \
          Context->BestGraphInfo[8].Coverage.LastLargePageUsed,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_LastCacheLineUsed,                                                      \
          Context->BestGraphInfo[8].Coverage.LastCacheLineUsed,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_TotalNumberOfAssigned,                                                  \
          Context->BestGraphInfo[8].Coverage.TotalNumberOfAssigned,                          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_CountOfCacheLinesWithNumberOfAssigned_0,                                \
          Context->BestGraphInfo[8].Coverage.NumberOfAssignedPerCacheLineCounts[0],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_CountOfCacheLinesWithNumberOfAssigned_1,                                \
          Context->BestGraphInfo[8].Coverage.NumberOfAssignedPerCacheLineCounts[1],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_CountOfCacheLinesWithNumberOfAssigned_2,                                \
          Context->BestGraphInfo[8].Coverage.NumberOfAssignedPerCacheLineCounts[2],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_CountOfCacheLinesWithNumberOfAssigned_3,                                \
          Context->BestGraphInfo[8].Coverage.NumberOfAssignedPerCacheLineCounts[3],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_CountOfCacheLinesWithNumberOfAssigned_4,                                \
          Context->BestGraphInfo[8].Coverage.NumberOfAssignedPerCacheLineCounts[4],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_CountOfCacheLinesWithNumberOfAssigned_5,                                \
          Context->BestGraphInfo[8].Coverage.NumberOfAssignedPerCacheLineCounts[5],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_CountOfCacheLinesWithNumberOfAssigned_6,                                \
          Context->BestGraphInfo[8].Coverage.NumberOfAssignedPerCacheLineCounts[6],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_CountOfCacheLinesWithNumberOfAssigned_7,                                \
          Context->BestGraphInfo[8].Coverage.NumberOfAssignedPerCacheLineCounts[7],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_CountOfCacheLinesWithNumberOfAssigned_8,                                \
          Context->BestGraphInfo[8].Coverage.NumberOfAssignedPerCacheLineCounts[8],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_CountOfCacheLinesWithNumberOfAssigned_9,                                \
          Context->BestGraphInfo[8].Coverage.NumberOfAssignedPerCacheLineCounts[9],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_CountOfCacheLinesWithNumberOfAssigned_10,                               \
          Context->BestGraphInfo[8].Coverage.NumberOfAssignedPerCacheLineCounts[10],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_CountOfCacheLinesWithNumberOfAssigned_11,                               \
          Context->BestGraphInfo[8].Coverage.NumberOfAssignedPerCacheLineCounts[11],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_CountOfCacheLinesWithNumberOfAssigned_12,                               \
          Context->BestGraphInfo[8].Coverage.NumberOfAssignedPerCacheLineCounts[12],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_CountOfCacheLinesWithNumberOfAssigned_13,                               \
          Context->BestGraphInfo[8].Coverage.NumberOfAssignedPerCacheLineCounts[13],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_CountOfCacheLinesWithNumberOfAssigned_14,                               \
          Context->BestGraphInfo[8].Coverage.NumberOfAssignedPerCacheLineCounts[14],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_CountOfCacheLinesWithNumberOfAssigned_15,                               \
          Context->BestGraphInfo[8].Coverage.NumberOfAssignedPerCacheLineCounts[15],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_CountOfCacheLinesWithNumberOfAssigned_16,                               \
          Context->BestGraphInfo[8].Coverage.NumberOfAssignedPerCacheLineCounts[16],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_TotalNumberOfPages,                                                    \
          Context->BestGraphInfo[9].Coverage.TotalNumberOfPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_TotalNumberOfLargePages,                                               \
          Context->BestGraphInfo[9].Coverage.TotalNumberOfLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_TotalNumberOfCacheLines,                                               \
          Context->BestGraphInfo[9].Coverage.TotalNumberOfCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_NumberOfUsedPages,                                                     \
          Context->BestGraphInfo[9].Coverage.NumberOfUsedPages,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_NumberOfUsedLargePages,                                                \
          Context->BestGraphInfo[9].Coverage.NumberOfUsedLargePages,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_NumberOfUsedCacheLines,                                                \
          Context->BestGraphInfo[9].Coverage.NumberOfUsedCacheLines,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_NumberOfEmptyPages,                                                    \
          Context->BestGraphInfo[9].Coverage.NumberOfEmptyPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_NumberOfEmptyLargePages,                                               \
          Context->BestGraphInfo[9].Coverage.NumberOfEmptyLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_NumberOfEmptyCacheLines,                                               \
          Context->BestGraphInfo[9].Coverage.NumberOfEmptyCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_FirstPageUsed,                                                         \
          Context->BestGraphInfo[9].Coverage.FirstPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_FirstLargePageUsed,                                                    \
          Context->BestGraphInfo[9].Coverage.FirstLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_FirstCacheLineUsed,                                                    \
          Context->BestGraphInfo[9].Coverage.FirstCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_LastPageUsed,                                                          \
          Context->BestGraphInfo[9].Coverage.LastPageUsed,                                   \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_LastLargePageUsed,                                                     \
          Context->BestGraphInfo[9].Coverage.LastLargePageUsed,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_LastCacheLineUsed,                                                     \
          Context->BestGraphInfo[9].Coverage.LastCacheLineUsed,                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_TotalNumberOfAssigned,                                                 \
          Context->BestGraphInfo[9].Coverage.TotalNumberOfAssigned,                          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_CountOfCacheLinesWithNumberOfAssigned_0,                               \
          Context->BestGraphInfo[9].Coverage.NumberOfAssignedPerCacheLineCounts[0],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_CountOfCacheLinesWithNumberOfAssigned_1,                               \
          Context->BestGraphInfo[9].Coverage.NumberOfAssignedPerCacheLineCounts[1],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_CountOfCacheLinesWithNumberOfAssigned_2,                               \
          Context->BestGraphInfo[9].Coverage.NumberOfAssignedPerCacheLineCounts[2],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_CountOfCacheLinesWithNumberOfAssigned_3,                               \
          Context->BestGraphInfo[9].Coverage.NumberOfAssignedPerCacheLineCounts[3],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_CountOfCacheLinesWithNumberOfAssigned_4,                               \
          Context->BestGraphInfo[9].Coverage.NumberOfAssignedPerCacheLineCounts[4],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_CountOfCacheLinesWithNumberOfAssigned_5,                               \
          Context->BestGraphInfo[9].Coverage.NumberOfAssignedPerCacheLineCounts[5],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_CountOfCacheLinesWithNumberOfAssigned_6,                               \
          Context->BestGraphInfo[9].Coverage.NumberOfAssignedPerCacheLineCounts[6],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_CountOfCacheLinesWithNumberOfAssigned_7,                               \
          Context->BestGraphInfo[9].Coverage.NumberOfAssignedPerCacheLineCounts[7],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_CountOfCacheLinesWithNumberOfAssigned_8,                               \
          Context->BestGraphInfo[9].Coverage.NumberOfAssignedPerCacheLineCounts[8],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_CountOfCacheLinesWithNumberOfAssigned_9,                               \
          Context->BestGraphInfo[9].Coverage.NumberOfAssignedPerCacheLineCounts[9],          \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_CountOfCacheLinesWithNumberOfAssigned_10,                              \
          Context->BestGraphInfo[9].Coverage.NumberOfAssignedPerCacheLineCounts[10],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_CountOfCacheLinesWithNumberOfAssigned_11,                              \
          Context->BestGraphInfo[9].Coverage.NumberOfAssignedPerCacheLineCounts[11],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_CountOfCacheLinesWithNumberOfAssigned_12,                              \
          Context->BestGraphInfo[9].Coverage.NumberOfAssignedPerCacheLineCounts[12],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_CountOfCacheLinesWithNumberOfAssigned_13,                              \
          Context->BestGraphInfo[9].Coverage.NumberOfAssignedPerCacheLineCounts[13],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_CountOfCacheLinesWithNumberOfAssigned_14,                              \
          Context->BestGraphInfo[9].Coverage.NumberOfAssignedPerCacheLineCounts[14],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_CountOfCacheLinesWithNumberOfAssigned_15,                              \
          Context->BestGraphInfo[9].Coverage.NumberOfAssignedPerCacheLineCounts[15],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_CountOfCacheLinesWithNumberOfAssigned_16,                              \
          Context->BestGraphInfo[9].Coverage.NumberOfAssignedPerCacheLineCounts[16],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_TotalNumberOfPages,                                                    \
          Context->BestGraphInfo[10].Coverage.TotalNumberOfPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_TotalNumberOfLargePages,                                               \
          Context->BestGraphInfo[10].Coverage.TotalNumberOfLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_TotalNumberOfCacheLines,                                               \
          Context->BestGraphInfo[10].Coverage.TotalNumberOfCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_NumberOfUsedPages,                                                     \
          Context->BestGraphInfo[10].Coverage.NumberOfUsedPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_NumberOfUsedLargePages,                                                \
          Context->BestGraphInfo[10].Coverage.NumberOfUsedLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_NumberOfUsedCacheLines,                                                \
          Context->BestGraphInfo[10].Coverage.NumberOfUsedCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_NumberOfEmptyPages,                                                    \
          Context->BestGraphInfo[10].Coverage.NumberOfEmptyPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_NumberOfEmptyLargePages,                                               \
          Context->BestGraphInfo[10].Coverage.NumberOfEmptyLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_NumberOfEmptyCacheLines,                                               \
          Context->BestGraphInfo[10].Coverage.NumberOfEmptyCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_FirstPageUsed,                                                         \
          Context->BestGraphInfo[10].Coverage.FirstPageUsed,                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_FirstLargePageUsed,                                                    \
          Context->BestGraphInfo[10].Coverage.FirstLargePageUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_FirstCacheLineUsed,                                                    \
          Context->BestGraphInfo[10].Coverage.FirstCacheLineUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_LastPageUsed,                                                          \
          Context->BestGraphInfo[10].Coverage.LastPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_LastLargePageUsed,                                                     \
          Context->BestGraphInfo[10].Coverage.LastLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_LastCacheLineUsed,                                                     \
          Context->BestGraphInfo[10].Coverage.LastCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_TotalNumberOfAssigned,                                                 \
          Context->BestGraphInfo[10].Coverage.TotalNumberOfAssigned,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_CountOfCacheLinesWithNumberOfAssigned_0,                               \
          Context->BestGraphInfo[10].Coverage.NumberOfAssignedPerCacheLineCounts[0],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_CountOfCacheLinesWithNumberOfAssigned_1,                               \
          Context->BestGraphInfo[10].Coverage.NumberOfAssignedPerCacheLineCounts[1],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_CountOfCacheLinesWithNumberOfAssigned_2,                               \
          Context->BestGraphInfo[10].Coverage.NumberOfAssignedPerCacheLineCounts[2],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_CountOfCacheLinesWithNumberOfAssigned_3,                               \
          Context->BestGraphInfo[10].Coverage.NumberOfAssignedPerCacheLineCounts[3],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_CountOfCacheLinesWithNumberOfAssigned_4,                               \
          Context->BestGraphInfo[10].Coverage.NumberOfAssignedPerCacheLineCounts[4],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_CountOfCacheLinesWithNumberOfAssigned_5,                               \
          Context->BestGraphInfo[10].Coverage.NumberOfAssignedPerCacheLineCounts[5],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_CountOfCacheLinesWithNumberOfAssigned_6,                               \
          Context->BestGraphInfo[10].Coverage.NumberOfAssignedPerCacheLineCounts[6],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_CountOfCacheLinesWithNumberOfAssigned_7,                               \
          Context->BestGraphInfo[10].Coverage.NumberOfAssignedPerCacheLineCounts[7],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_CountOfCacheLinesWithNumberOfAssigned_8,                               \
          Context->BestGraphInfo[10].Coverage.NumberOfAssignedPerCacheLineCounts[8],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_CountOfCacheLinesWithNumberOfAssigned_9,                               \
          Context->BestGraphInfo[10].Coverage.NumberOfAssignedPerCacheLineCounts[9],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_CountOfCacheLinesWithNumberOfAssigned_10,                              \
          Context->BestGraphInfo[10].Coverage.NumberOfAssignedPerCacheLineCounts[10],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_CountOfCacheLinesWithNumberOfAssigned_11,                              \
          Context->BestGraphInfo[10].Coverage.NumberOfAssignedPerCacheLineCounts[11],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_CountOfCacheLinesWithNumberOfAssigned_12,                              \
          Context->BestGraphInfo[10].Coverage.NumberOfAssignedPerCacheLineCounts[12],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_CountOfCacheLinesWithNumberOfAssigned_13,                              \
          Context->BestGraphInfo[10].Coverage.NumberOfAssignedPerCacheLineCounts[13],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_CountOfCacheLinesWithNumberOfAssigned_14,                              \
          Context->BestGraphInfo[10].Coverage.NumberOfAssignedPerCacheLineCounts[14],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_CountOfCacheLinesWithNumberOfAssigned_15,                              \
          Context->BestGraphInfo[10].Coverage.NumberOfAssignedPerCacheLineCounts[15],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_CountOfCacheLinesWithNumberOfAssigned_16,                              \
          Context->BestGraphInfo[10].Coverage.NumberOfAssignedPerCacheLineCounts[16],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_TotalNumberOfPages,                                                    \
          Context->BestGraphInfo[11].Coverage.TotalNumberOfPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_TotalNumberOfLargePages,                                               \
          Context->BestGraphInfo[11].Coverage.TotalNumberOfLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_TotalNumberOfCacheLines,                                               \
          Context->BestGraphInfo[11].Coverage.TotalNumberOfCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_NumberOfUsedPages,                                                     \
          Context->BestGraphInfo[11].Coverage.NumberOfUsedPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_NumberOfUsedLargePages,                                                \
          Context->BestGraphInfo[11].Coverage.NumberOfUsedLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_NumberOfUsedCacheLines,                                                \
          Context->BestGraphInfo[11].Coverage.NumberOfUsedCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_NumberOfEmptyPages,                                                    \
          Context->BestGraphInfo[11].Coverage.NumberOfEmptyPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_NumberOfEmptyLargePages,                                               \
          Context->BestGraphInfo[11].Coverage.NumberOfEmptyLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_NumberOfEmptyCacheLines,                                               \
          Context->BestGraphInfo[11].Coverage.NumberOfEmptyCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_FirstPageUsed,                                                         \
          Context->BestGraphInfo[11].Coverage.FirstPageUsed,                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_FirstLargePageUsed,                                                    \
          Context->BestGraphInfo[11].Coverage.FirstLargePageUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_FirstCacheLineUsed,                                                    \
          Context->BestGraphInfo[11].Coverage.FirstCacheLineUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_LastPageUsed,                                                          \
          Context->BestGraphInfo[11].Coverage.LastPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_LastLargePageUsed,                                                     \
          Context->BestGraphInfo[11].Coverage.LastLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_LastCacheLineUsed,                                                     \
          Context->BestGraphInfo[11].Coverage.LastCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_TotalNumberOfAssigned,                                                 \
          Context->BestGraphInfo[11].Coverage.TotalNumberOfAssigned,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_CountOfCacheLinesWithNumberOfAssigned_0,                               \
          Context->BestGraphInfo[11].Coverage.NumberOfAssignedPerCacheLineCounts[0],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_CountOfCacheLinesWithNumberOfAssigned_1,                               \
          Context->BestGraphInfo[11].Coverage.NumberOfAssignedPerCacheLineCounts[1],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_CountOfCacheLinesWithNumberOfAssigned_2,                               \
          Context->BestGraphInfo[11].Coverage.NumberOfAssignedPerCacheLineCounts[2],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_CountOfCacheLinesWithNumberOfAssigned_3,                               \
          Context->BestGraphInfo[11].Coverage.NumberOfAssignedPerCacheLineCounts[3],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_CountOfCacheLinesWithNumberOfAssigned_4,                               \
          Context->BestGraphInfo[11].Coverage.NumberOfAssignedPerCacheLineCounts[4],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_CountOfCacheLinesWithNumberOfAssigned_5,                               \
          Context->BestGraphInfo[11].Coverage.NumberOfAssignedPerCacheLineCounts[5],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_CountOfCacheLinesWithNumberOfAssigned_6,                               \
          Context->BestGraphInfo[11].Coverage.NumberOfAssignedPerCacheLineCounts[6],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_CountOfCacheLinesWithNumberOfAssigned_7,                               \
          Context->BestGraphInfo[11].Coverage.NumberOfAssignedPerCacheLineCounts[7],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_CountOfCacheLinesWithNumberOfAssigned_8,                               \
          Context->BestGraphInfo[11].Coverage.NumberOfAssignedPerCacheLineCounts[8],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_CountOfCacheLinesWithNumberOfAssigned_9,                               \
          Context->BestGraphInfo[11].Coverage.NumberOfAssignedPerCacheLineCounts[9],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_CountOfCacheLinesWithNumberOfAssigned_10,                              \
          Context->BestGraphInfo[11].Coverage.NumberOfAssignedPerCacheLineCounts[10],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_CountOfCacheLinesWithNumberOfAssigned_11,                              \
          Context->BestGraphInfo[11].Coverage.NumberOfAssignedPerCacheLineCounts[11],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_CountOfCacheLinesWithNumberOfAssigned_12,                              \
          Context->BestGraphInfo[11].Coverage.NumberOfAssignedPerCacheLineCounts[12],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_CountOfCacheLinesWithNumberOfAssigned_13,                              \
          Context->BestGraphInfo[11].Coverage.NumberOfAssignedPerCacheLineCounts[13],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_CountOfCacheLinesWithNumberOfAssigned_14,                              \
          Context->BestGraphInfo[11].Coverage.NumberOfAssignedPerCacheLineCounts[14],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_CountOfCacheLinesWithNumberOfAssigned_15,                              \
          Context->BestGraphInfo[11].Coverage.NumberOfAssignedPerCacheLineCounts[15],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_CountOfCacheLinesWithNumberOfAssigned_16,                              \
          Context->BestGraphInfo[11].Coverage.NumberOfAssignedPerCacheLineCounts[16],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_TotalNumberOfPages,                                                    \
          Context->BestGraphInfo[12].Coverage.TotalNumberOfPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_TotalNumberOfLargePages,                                               \
          Context->BestGraphInfo[12].Coverage.TotalNumberOfLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_TotalNumberOfCacheLines,                                               \
          Context->BestGraphInfo[12].Coverage.TotalNumberOfCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_NumberOfUsedPages,                                                     \
          Context->BestGraphInfo[12].Coverage.NumberOfUsedPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_NumberOfUsedLargePages,                                                \
          Context->BestGraphInfo[12].Coverage.NumberOfUsedLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_NumberOfUsedCacheLines,                                                \
          Context->BestGraphInfo[12].Coverage.NumberOfUsedCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_NumberOfEmptyPages,                                                    \
          Context->BestGraphInfo[12].Coverage.NumberOfEmptyPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_NumberOfEmptyLargePages,                                               \
          Context->BestGraphInfo[12].Coverage.NumberOfEmptyLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_NumberOfEmptyCacheLines,                                               \
          Context->BestGraphInfo[12].Coverage.NumberOfEmptyCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_FirstPageUsed,                                                         \
          Context->BestGraphInfo[12].Coverage.FirstPageUsed,                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_FirstLargePageUsed,                                                    \
          Context->BestGraphInfo[12].Coverage.FirstLargePageUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_FirstCacheLineUsed,                                                    \
          Context->BestGraphInfo[12].Coverage.FirstCacheLineUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_LastPageUsed,                                                          \
          Context->BestGraphInfo[12].Coverage.LastPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_LastLargePageUsed,                                                     \
          Context->BestGraphInfo[12].Coverage.LastLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_LastCacheLineUsed,                                                     \
          Context->BestGraphInfo[12].Coverage.LastCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_TotalNumberOfAssigned,                                                 \
          Context->BestGraphInfo[12].Coverage.TotalNumberOfAssigned,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_CountOfCacheLinesWithNumberOfAssigned_0,                               \
          Context->BestGraphInfo[12].Coverage.NumberOfAssignedPerCacheLineCounts[0],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_CountOfCacheLinesWithNumberOfAssigned_1,                               \
          Context->BestGraphInfo[12].Coverage.NumberOfAssignedPerCacheLineCounts[1],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_CountOfCacheLinesWithNumberOfAssigned_2,                               \
          Context->BestGraphInfo[12].Coverage.NumberOfAssignedPerCacheLineCounts[2],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_CountOfCacheLinesWithNumberOfAssigned_3,                               \
          Context->BestGraphInfo[12].Coverage.NumberOfAssignedPerCacheLineCounts[3],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_CountOfCacheLinesWithNumberOfAssigned_4,                               \
          Context->BestGraphInfo[12].Coverage.NumberOfAssignedPerCacheLineCounts[4],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_CountOfCacheLinesWithNumberOfAssigned_5,                               \
          Context->BestGraphInfo[12].Coverage.NumberOfAssignedPerCacheLineCounts[5],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_CountOfCacheLinesWithNumberOfAssigned_6,                               \
          Context->BestGraphInfo[12].Coverage.NumberOfAssignedPerCacheLineCounts[6],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_CountOfCacheLinesWithNumberOfAssigned_7,                               \
          Context->BestGraphInfo[12].Coverage.NumberOfAssignedPerCacheLineCounts[7],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_CountOfCacheLinesWithNumberOfAssigned_8,                               \
          Context->BestGraphInfo[12].Coverage.NumberOfAssignedPerCacheLineCounts[8],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_CountOfCacheLinesWithNumberOfAssigned_9,                               \
          Context->BestGraphInfo[12].Coverage.NumberOfAssignedPerCacheLineCounts[9],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_CountOfCacheLinesWithNumberOfAssigned_10,                              \
          Context->BestGraphInfo[12].Coverage.NumberOfAssignedPerCacheLineCounts[10],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_CountOfCacheLinesWithNumberOfAssigned_11,                              \
          Context->BestGraphInfo[12].Coverage.NumberOfAssignedPerCacheLineCounts[11],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_CountOfCacheLinesWithNumberOfAssigned_12,                              \
          Context->BestGraphInfo[12].Coverage.NumberOfAssignedPerCacheLineCounts[12],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_CountOfCacheLinesWithNumberOfAssigned_13,                              \
          Context->BestGraphInfo[12].Coverage.NumberOfAssignedPerCacheLineCounts[13],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_CountOfCacheLinesWithNumberOfAssigned_14,                              \
          Context->BestGraphInfo[12].Coverage.NumberOfAssignedPerCacheLineCounts[14],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_CountOfCacheLinesWithNumberOfAssigned_15,                              \
          Context->BestGraphInfo[12].Coverage.NumberOfAssignedPerCacheLineCounts[15],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_CountOfCacheLinesWithNumberOfAssigned_16,                              \
          Context->BestGraphInfo[12].Coverage.NumberOfAssignedPerCacheLineCounts[16],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_TotalNumberOfPages,                                                    \
          Context->BestGraphInfo[13].Coverage.TotalNumberOfPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_TotalNumberOfLargePages,                                               \
          Context->BestGraphInfo[13].Coverage.TotalNumberOfLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_TotalNumberOfCacheLines,                                               \
          Context->BestGraphInfo[13].Coverage.TotalNumberOfCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_NumberOfUsedPages,                                                     \
          Context->BestGraphInfo[13].Coverage.NumberOfUsedPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_NumberOfUsedLargePages,                                                \
          Context->BestGraphInfo[13].Coverage.NumberOfUsedLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_NumberOfUsedCacheLines,                                                \
          Context->BestGraphInfo[13].Coverage.NumberOfUsedCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_NumberOfEmptyPages,                                                    \
          Context->BestGraphInfo[13].Coverage.NumberOfEmptyPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_NumberOfEmptyLargePages,                                               \
          Context->BestGraphInfo[13].Coverage.NumberOfEmptyLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_NumberOfEmptyCacheLines,                                               \
          Context->BestGraphInfo[13].Coverage.NumberOfEmptyCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_FirstPageUsed,                                                         \
          Context->BestGraphInfo[13].Coverage.FirstPageUsed,                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_FirstLargePageUsed,                                                    \
          Context->BestGraphInfo[13].Coverage.FirstLargePageUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_FirstCacheLineUsed,                                                    \
          Context->BestGraphInfo[13].Coverage.FirstCacheLineUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_LastPageUsed,                                                          \
          Context->BestGraphInfo[13].Coverage.LastPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_LastLargePageUsed,                                                     \
          Context->BestGraphInfo[13].Coverage.LastLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_LastCacheLineUsed,                                                     \
          Context->BestGraphInfo[13].Coverage.LastCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_TotalNumberOfAssigned,                                                 \
          Context->BestGraphInfo[13].Coverage.TotalNumberOfAssigned,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_CountOfCacheLinesWithNumberOfAssigned_0,                               \
          Context->BestGraphInfo[13].Coverage.NumberOfAssignedPerCacheLineCounts[0],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_CountOfCacheLinesWithNumberOfAssigned_1,                               \
          Context->BestGraphInfo[13].Coverage.NumberOfAssignedPerCacheLineCounts[1],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_CountOfCacheLinesWithNumberOfAssigned_2,                               \
          Context->BestGraphInfo[13].Coverage.NumberOfAssignedPerCacheLineCounts[2],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_CountOfCacheLinesWithNumberOfAssigned_3,                               \
          Context->BestGraphInfo[13].Coverage.NumberOfAssignedPerCacheLineCounts[3],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_CountOfCacheLinesWithNumberOfAssigned_4,                               \
          Context->BestGraphInfo[13].Coverage.NumberOfAssignedPerCacheLineCounts[4],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_CountOfCacheLinesWithNumberOfAssigned_5,                               \
          Context->BestGraphInfo[13].Coverage.NumberOfAssignedPerCacheLineCounts[5],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_CountOfCacheLinesWithNumberOfAssigned_6,                               \
          Context->BestGraphInfo[13].Coverage.NumberOfAssignedPerCacheLineCounts[6],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_CountOfCacheLinesWithNumberOfAssigned_7,                               \
          Context->BestGraphInfo[13].Coverage.NumberOfAssignedPerCacheLineCounts[7],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_CountOfCacheLinesWithNumberOfAssigned_8,                               \
          Context->BestGraphInfo[13].Coverage.NumberOfAssignedPerCacheLineCounts[8],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_CountOfCacheLinesWithNumberOfAssigned_9,                               \
          Context->BestGraphInfo[13].Coverage.NumberOfAssignedPerCacheLineCounts[9],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_CountOfCacheLinesWithNumberOfAssigned_10,                              \
          Context->BestGraphInfo[13].Coverage.NumberOfAssignedPerCacheLineCounts[10],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_CountOfCacheLinesWithNumberOfAssigned_11,                              \
          Context->BestGraphInfo[13].Coverage.NumberOfAssignedPerCacheLineCounts[11],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_CountOfCacheLinesWithNumberOfAssigned_12,                              \
          Context->BestGraphInfo[13].Coverage.NumberOfAssignedPerCacheLineCounts[12],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_CountOfCacheLinesWithNumberOfAssigned_13,                              \
          Context->BestGraphInfo[13].Coverage.NumberOfAssignedPerCacheLineCounts[13],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_CountOfCacheLinesWithNumberOfAssigned_14,                              \
          Context->BestGraphInfo[13].Coverage.NumberOfAssignedPerCacheLineCounts[14],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_CountOfCacheLinesWithNumberOfAssigned_15,                              \
          Context->BestGraphInfo[13].Coverage.NumberOfAssignedPerCacheLineCounts[15],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_CountOfCacheLinesWithNumberOfAssigned_16,                              \
          Context->BestGraphInfo[13].Coverage.NumberOfAssignedPerCacheLineCounts[16],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_TotalNumberOfPages,                                                    \
          Context->BestGraphInfo[14].Coverage.TotalNumberOfPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_TotalNumberOfLargePages,                                               \
          Context->BestGraphInfo[14].Coverage.TotalNumberOfLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_TotalNumberOfCacheLines,                                               \
          Context->BestGraphInfo[14].Coverage.TotalNumberOfCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_NumberOfUsedPages,                                                     \
          Context->BestGraphInfo[14].Coverage.NumberOfUsedPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_NumberOfUsedLargePages,                                                \
          Context->BestGraphInfo[14].Coverage.NumberOfUsedLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_NumberOfUsedCacheLines,                                                \
          Context->BestGraphInfo[14].Coverage.NumberOfUsedCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_NumberOfEmptyPages,                                                    \
          Context->BestGraphInfo[14].Coverage.NumberOfEmptyPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_NumberOfEmptyLargePages,                                               \
          Context->BestGraphInfo[14].Coverage.NumberOfEmptyLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_NumberOfEmptyCacheLines,                                               \
          Context->BestGraphInfo[14].Coverage.NumberOfEmptyCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_FirstPageUsed,                                                         \
          Context->BestGraphInfo[14].Coverage.FirstPageUsed,                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_FirstLargePageUsed,                                                    \
          Context->BestGraphInfo[14].Coverage.FirstLargePageUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_FirstCacheLineUsed,                                                    \
          Context->BestGraphInfo[14].Coverage.FirstCacheLineUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_LastPageUsed,                                                          \
          Context->BestGraphInfo[14].Coverage.LastPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_LastLargePageUsed,                                                     \
          Context->BestGraphInfo[14].Coverage.LastLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_LastCacheLineUsed,                                                     \
          Context->BestGraphInfo[14].Coverage.LastCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_TotalNumberOfAssigned,                                                 \
          Context->BestGraphInfo[14].Coverage.TotalNumberOfAssigned,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_CountOfCacheLinesWithNumberOfAssigned_0,                               \
          Context->BestGraphInfo[14].Coverage.NumberOfAssignedPerCacheLineCounts[0],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_CountOfCacheLinesWithNumberOfAssigned_1,                               \
          Context->BestGraphInfo[14].Coverage.NumberOfAssignedPerCacheLineCounts[1],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_CountOfCacheLinesWithNumberOfAssigned_2,                               \
          Context->BestGraphInfo[14].Coverage.NumberOfAssignedPerCacheLineCounts[2],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_CountOfCacheLinesWithNumberOfAssigned_3,                               \
          Context->BestGraphInfo[14].Coverage.NumberOfAssignedPerCacheLineCounts[3],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_CountOfCacheLinesWithNumberOfAssigned_4,                               \
          Context->BestGraphInfo[14].Coverage.NumberOfAssignedPerCacheLineCounts[4],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_CountOfCacheLinesWithNumberOfAssigned_5,                               \
          Context->BestGraphInfo[14].Coverage.NumberOfAssignedPerCacheLineCounts[5],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_CountOfCacheLinesWithNumberOfAssigned_6,                               \
          Context->BestGraphInfo[14].Coverage.NumberOfAssignedPerCacheLineCounts[6],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_CountOfCacheLinesWithNumberOfAssigned_7,                               \
          Context->BestGraphInfo[14].Coverage.NumberOfAssignedPerCacheLineCounts[7],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_CountOfCacheLinesWithNumberOfAssigned_8,                               \
          Context->BestGraphInfo[14].Coverage.NumberOfAssignedPerCacheLineCounts[8],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_CountOfCacheLinesWithNumberOfAssigned_9,                               \
          Context->BestGraphInfo[14].Coverage.NumberOfAssignedPerCacheLineCounts[9],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_CountOfCacheLinesWithNumberOfAssigned_10,                              \
          Context->BestGraphInfo[14].Coverage.NumberOfAssignedPerCacheLineCounts[10],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_CountOfCacheLinesWithNumberOfAssigned_11,                              \
          Context->BestGraphInfo[14].Coverage.NumberOfAssignedPerCacheLineCounts[11],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_CountOfCacheLinesWithNumberOfAssigned_12,                              \
          Context->BestGraphInfo[14].Coverage.NumberOfAssignedPerCacheLineCounts[12],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_CountOfCacheLinesWithNumberOfAssigned_13,                              \
          Context->BestGraphInfo[14].Coverage.NumberOfAssignedPerCacheLineCounts[13],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_CountOfCacheLinesWithNumberOfAssigned_14,                              \
          Context->BestGraphInfo[14].Coverage.NumberOfAssignedPerCacheLineCounts[14],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_CountOfCacheLinesWithNumberOfAssigned_15,                              \
          Context->BestGraphInfo[14].Coverage.NumberOfAssignedPerCacheLineCounts[15],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_CountOfCacheLinesWithNumberOfAssigned_16,                              \
          Context->BestGraphInfo[14].Coverage.NumberOfAssignedPerCacheLineCounts[16],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_TotalNumberOfPages,                                                    \
          Context->BestGraphInfo[15].Coverage.TotalNumberOfPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_TotalNumberOfLargePages,                                               \
          Context->BestGraphInfo[15].Coverage.TotalNumberOfLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_TotalNumberOfCacheLines,                                               \
          Context->BestGraphInfo[15].Coverage.TotalNumberOfCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_NumberOfUsedPages,                                                     \
          Context->BestGraphInfo[15].Coverage.NumberOfUsedPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_NumberOfUsedLargePages,                                                \
          Context->BestGraphInfo[15].Coverage.NumberOfUsedLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_NumberOfUsedCacheLines,                                                \
          Context->BestGraphInfo[15].Coverage.NumberOfUsedCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_NumberOfEmptyPages,                                                    \
          Context->BestGraphInfo[15].Coverage.NumberOfEmptyPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_NumberOfEmptyLargePages,                                               \
          Context->BestGraphInfo[15].Coverage.NumberOfEmptyLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_NumberOfEmptyCacheLines,                                               \
          Context->BestGraphInfo[15].Coverage.NumberOfEmptyCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_FirstPageUsed,                                                         \
          Context->BestGraphInfo[15].Coverage.FirstPageUsed,                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_FirstLargePageUsed,                                                    \
          Context->BestGraphInfo[15].Coverage.FirstLargePageUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_FirstCacheLineUsed,                                                    \
          Context->BestGraphInfo[15].Coverage.FirstCacheLineUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_LastPageUsed,                                                          \
          Context->BestGraphInfo[15].Coverage.LastPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_LastLargePageUsed,                                                     \
          Context->BestGraphInfo[15].Coverage.LastLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_LastCacheLineUsed,                                                     \
          Context->BestGraphInfo[15].Coverage.LastCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_TotalNumberOfAssigned,                                                 \
          Context->BestGraphInfo[15].Coverage.TotalNumberOfAssigned,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_CountOfCacheLinesWithNumberOfAssigned_0,                               \
          Context->BestGraphInfo[15].Coverage.NumberOfAssignedPerCacheLineCounts[0],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_CountOfCacheLinesWithNumberOfAssigned_1,                               \
          Context->BestGraphInfo[15].Coverage.NumberOfAssignedPerCacheLineCounts[1],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_CountOfCacheLinesWithNumberOfAssigned_2,                               \
          Context->BestGraphInfo[15].Coverage.NumberOfAssignedPerCacheLineCounts[2],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_CountOfCacheLinesWithNumberOfAssigned_3,                               \
          Context->BestGraphInfo[15].Coverage.NumberOfAssignedPerCacheLineCounts[3],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_CountOfCacheLinesWithNumberOfAssigned_4,                               \
          Context->BestGraphInfo[15].Coverage.NumberOfAssignedPerCacheLineCounts[4],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_CountOfCacheLinesWithNumberOfAssigned_5,                               \
          Context->BestGraphInfo[15].Coverage.NumberOfAssignedPerCacheLineCounts[5],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_CountOfCacheLinesWithNumberOfAssigned_6,                               \
          Context->BestGraphInfo[15].Coverage.NumberOfAssignedPerCacheLineCounts[6],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_CountOfCacheLinesWithNumberOfAssigned_7,                               \
          Context->BestGraphInfo[15].Coverage.NumberOfAssignedPerCacheLineCounts[7],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_CountOfCacheLinesWithNumberOfAssigned_8,                               \
          Context->BestGraphInfo[15].Coverage.NumberOfAssignedPerCacheLineCounts[8],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_CountOfCacheLinesWithNumberOfAssigned_9,                               \
          Context->BestGraphInfo[15].Coverage.NumberOfAssignedPerCacheLineCounts[9],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_CountOfCacheLinesWithNumberOfAssigned_10,                              \
          Context->BestGraphInfo[15].Coverage.NumberOfAssignedPerCacheLineCounts[10],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_CountOfCacheLinesWithNumberOfAssigned_11,                              \
          Context->BestGraphInfo[15].Coverage.NumberOfAssignedPerCacheLineCounts[11],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_CountOfCacheLinesWithNumberOfAssigned_12,                              \
          Context->BestGraphInfo[15].Coverage.NumberOfAssignedPerCacheLineCounts[12],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_CountOfCacheLinesWithNumberOfAssigned_13,                              \
          Context->BestGraphInfo[15].Coverage.NumberOfAssignedPerCacheLineCounts[13],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_CountOfCacheLinesWithNumberOfAssigned_14,                              \
          Context->BestGraphInfo[15].Coverage.NumberOfAssignedPerCacheLineCounts[14],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_CountOfCacheLinesWithNumberOfAssigned_15,                              \
          Context->BestGraphInfo[15].Coverage.NumberOfAssignedPerCacheLineCounts[15],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_CountOfCacheLinesWithNumberOfAssigned_16,                              \
          Context->BestGraphInfo[15].Coverage.NumberOfAssignedPerCacheLineCounts[16],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_TotalNumberOfPages,                                                    \
          Context->BestGraphInfo[16].Coverage.TotalNumberOfPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_TotalNumberOfLargePages,                                               \
          Context->BestGraphInfo[16].Coverage.TotalNumberOfLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_TotalNumberOfCacheLines,                                               \
          Context->BestGraphInfo[16].Coverage.TotalNumberOfCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_NumberOfUsedPages,                                                     \
          Context->BestGraphInfo[16].Coverage.NumberOfUsedPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_NumberOfUsedLargePages,                                                \
          Context->BestGraphInfo[16].Coverage.NumberOfUsedLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_NumberOfUsedCacheLines,                                                \
          Context->BestGraphInfo[16].Coverage.NumberOfUsedCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_NumberOfEmptyPages,                                                    \
          Context->BestGraphInfo[16].Coverage.NumberOfEmptyPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_NumberOfEmptyLargePages,                                               \
          Context->BestGraphInfo[16].Coverage.NumberOfEmptyLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_NumberOfEmptyCacheLines,                                               \
          Context->BestGraphInfo[16].Coverage.NumberOfEmptyCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_FirstPageUsed,                                                         \
          Context->BestGraphInfo[16].Coverage.FirstPageUsed,                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_FirstLargePageUsed,                                                    \
          Context->BestGraphInfo[16].Coverage.FirstLargePageUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_FirstCacheLineUsed,                                                    \
          Context->BestGraphInfo[16].Coverage.FirstCacheLineUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_LastPageUsed,                                                          \
          Context->BestGraphInfo[16].Coverage.LastPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_LastLargePageUsed,                                                     \
          Context->BestGraphInfo[16].Coverage.LastLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_LastCacheLineUsed,                                                     \
          Context->BestGraphInfo[16].Coverage.LastCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_TotalNumberOfAssigned,                                                 \
          Context->BestGraphInfo[16].Coverage.TotalNumberOfAssigned,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_CountOfCacheLinesWithNumberOfAssigned_0,                               \
          Context->BestGraphInfo[16].Coverage.NumberOfAssignedPerCacheLineCounts[0],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_CountOfCacheLinesWithNumberOfAssigned_1,                               \
          Context->BestGraphInfo[16].Coverage.NumberOfAssignedPerCacheLineCounts[1],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_CountOfCacheLinesWithNumberOfAssigned_2,                               \
          Context->BestGraphInfo[16].Coverage.NumberOfAssignedPerCacheLineCounts[2],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_CountOfCacheLinesWithNumberOfAssigned_3,                               \
          Context->BestGraphInfo[16].Coverage.NumberOfAssignedPerCacheLineCounts[3],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_CountOfCacheLinesWithNumberOfAssigned_4,                               \
          Context->BestGraphInfo[16].Coverage.NumberOfAssignedPerCacheLineCounts[4],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_CountOfCacheLinesWithNumberOfAssigned_5,                               \
          Context->BestGraphInfo[16].Coverage.NumberOfAssignedPerCacheLineCounts[5],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_CountOfCacheLinesWithNumberOfAssigned_6,                               \
          Context->BestGraphInfo[16].Coverage.NumberOfAssignedPerCacheLineCounts[6],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_CountOfCacheLinesWithNumberOfAssigned_7,                               \
          Context->BestGraphInfo[16].Coverage.NumberOfAssignedPerCacheLineCounts[7],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_CountOfCacheLinesWithNumberOfAssigned_8,                               \
          Context->BestGraphInfo[16].Coverage.NumberOfAssignedPerCacheLineCounts[8],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_CountOfCacheLinesWithNumberOfAssigned_9,                               \
          Context->BestGraphInfo[16].Coverage.NumberOfAssignedPerCacheLineCounts[9],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_CountOfCacheLinesWithNumberOfAssigned_10,                              \
          Context->BestGraphInfo[16].Coverage.NumberOfAssignedPerCacheLineCounts[10],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_CountOfCacheLinesWithNumberOfAssigned_11,                              \
          Context->BestGraphInfo[16].Coverage.NumberOfAssignedPerCacheLineCounts[11],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_CountOfCacheLinesWithNumberOfAssigned_12,                              \
          Context->BestGraphInfo[16].Coverage.NumberOfAssignedPerCacheLineCounts[12],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_CountOfCacheLinesWithNumberOfAssigned_13,                              \
          Context->BestGraphInfo[16].Coverage.NumberOfAssignedPerCacheLineCounts[13],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_CountOfCacheLinesWithNumberOfAssigned_14,                              \
          Context->BestGraphInfo[16].Coverage.NumberOfAssignedPerCacheLineCounts[14],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_CountOfCacheLinesWithNumberOfAssigned_15,                              \
          Context->BestGraphInfo[16].Coverage.NumberOfAssignedPerCacheLineCounts[15],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_CountOfCacheLinesWithNumberOfAssigned_16,                              \
          Context->BestGraphInfo[16].Coverage.NumberOfAssignedPerCacheLineCounts[16],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_TotalNumberOfPages,                                                    \
          Context->BestGraphInfo[17].Coverage.TotalNumberOfPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_TotalNumberOfLargePages,                                               \
          Context->BestGraphInfo[17].Coverage.TotalNumberOfLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_TotalNumberOfCacheLines,                                               \
          Context->BestGraphInfo[17].Coverage.TotalNumberOfCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_NumberOfUsedPages,                                                     \
          Context->BestGraphInfo[17].Coverage.NumberOfUsedPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_NumberOfUsedLargePages,                                                \
          Context->BestGraphInfo[17].Coverage.NumberOfUsedLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_NumberOfUsedCacheLines,                                                \
          Context->BestGraphInfo[17].Coverage.NumberOfUsedCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_NumberOfEmptyPages,                                                    \
          Context->BestGraphInfo[17].Coverage.NumberOfEmptyPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_NumberOfEmptyLargePages,                                               \
          Context->BestGraphInfo[17].Coverage.NumberOfEmptyLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_NumberOfEmptyCacheLines,                                               \
          Context->BestGraphInfo[17].Coverage.NumberOfEmptyCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_FirstPageUsed,                                                         \
          Context->BestGraphInfo[17].Coverage.FirstPageUsed,                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_FirstLargePageUsed,                                                    \
          Context->BestGraphInfo[17].Coverage.FirstLargePageUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_FirstCacheLineUsed,                                                    \
          Context->BestGraphInfo[17].Coverage.FirstCacheLineUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_LastPageUsed,                                                          \
          Context->BestGraphInfo[17].Coverage.LastPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_LastLargePageUsed,                                                     \
          Context->BestGraphInfo[17].Coverage.LastLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_LastCacheLineUsed,                                                     \
          Context->BestGraphInfo[17].Coverage.LastCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_TotalNumberOfAssigned,                                                 \
          Context->BestGraphInfo[17].Coverage.TotalNumberOfAssigned,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_CountOfCacheLinesWithNumberOfAssigned_0,                               \
          Context->BestGraphInfo[17].Coverage.NumberOfAssignedPerCacheLineCounts[0],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_CountOfCacheLinesWithNumberOfAssigned_1,                               \
          Context->BestGraphInfo[17].Coverage.NumberOfAssignedPerCacheLineCounts[1],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_CountOfCacheLinesWithNumberOfAssigned_2,                               \
          Context->BestGraphInfo[17].Coverage.NumberOfAssignedPerCacheLineCounts[2],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_CountOfCacheLinesWithNumberOfAssigned_3,                               \
          Context->BestGraphInfo[17].Coverage.NumberOfAssignedPerCacheLineCounts[3],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_CountOfCacheLinesWithNumberOfAssigned_4,                               \
          Context->BestGraphInfo[17].Coverage.NumberOfAssignedPerCacheLineCounts[4],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_CountOfCacheLinesWithNumberOfAssigned_5,                               \
          Context->BestGraphInfo[17].Coverage.NumberOfAssignedPerCacheLineCounts[5],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_CountOfCacheLinesWithNumberOfAssigned_6,                               \
          Context->BestGraphInfo[17].Coverage.NumberOfAssignedPerCacheLineCounts[6],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_CountOfCacheLinesWithNumberOfAssigned_7,                               \
          Context->BestGraphInfo[17].Coverage.NumberOfAssignedPerCacheLineCounts[7],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_CountOfCacheLinesWithNumberOfAssigned_8,                               \
          Context->BestGraphInfo[17].Coverage.NumberOfAssignedPerCacheLineCounts[8],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_CountOfCacheLinesWithNumberOfAssigned_9,                               \
          Context->BestGraphInfo[17].Coverage.NumberOfAssignedPerCacheLineCounts[9],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_CountOfCacheLinesWithNumberOfAssigned_10,                              \
          Context->BestGraphInfo[17].Coverage.NumberOfAssignedPerCacheLineCounts[10],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_CountOfCacheLinesWithNumberOfAssigned_11,                              \
          Context->BestGraphInfo[17].Coverage.NumberOfAssignedPerCacheLineCounts[11],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_CountOfCacheLinesWithNumberOfAssigned_12,                              \
          Context->BestGraphInfo[17].Coverage.NumberOfAssignedPerCacheLineCounts[12],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_CountOfCacheLinesWithNumberOfAssigned_13,                              \
          Context->BestGraphInfo[17].Coverage.NumberOfAssignedPerCacheLineCounts[13],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_CountOfCacheLinesWithNumberOfAssigned_14,                              \
          Context->BestGraphInfo[17].Coverage.NumberOfAssignedPerCacheLineCounts[14],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_CountOfCacheLinesWithNumberOfAssigned_15,                              \
          Context->BestGraphInfo[17].Coverage.NumberOfAssignedPerCacheLineCounts[15],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_CountOfCacheLinesWithNumberOfAssigned_16,                              \
          Context->BestGraphInfo[17].Coverage.NumberOfAssignedPerCacheLineCounts[16],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_TotalNumberOfPages,                                                    \
          Context->BestGraphInfo[18].Coverage.TotalNumberOfPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_TotalNumberOfLargePages,                                               \
          Context->BestGraphInfo[18].Coverage.TotalNumberOfLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_TotalNumberOfCacheLines,                                               \
          Context->BestGraphInfo[18].Coverage.TotalNumberOfCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_NumberOfUsedPages,                                                     \
          Context->BestGraphInfo[18].Coverage.NumberOfUsedPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_NumberOfUsedLargePages,                                                \
          Context->BestGraphInfo[18].Coverage.NumberOfUsedLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_NumberOfUsedCacheLines,                                                \
          Context->BestGraphInfo[18].Coverage.NumberOfUsedCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_NumberOfEmptyPages,                                                    \
          Context->BestGraphInfo[18].Coverage.NumberOfEmptyPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_NumberOfEmptyLargePages,                                               \
          Context->BestGraphInfo[18].Coverage.NumberOfEmptyLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_NumberOfEmptyCacheLines,                                               \
          Context->BestGraphInfo[18].Coverage.NumberOfEmptyCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_FirstPageUsed,                                                         \
          Context->BestGraphInfo[18].Coverage.FirstPageUsed,                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_FirstLargePageUsed,                                                    \
          Context->BestGraphInfo[18].Coverage.FirstLargePageUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_FirstCacheLineUsed,                                                    \
          Context->BestGraphInfo[18].Coverage.FirstCacheLineUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_LastPageUsed,                                                          \
          Context->BestGraphInfo[18].Coverage.LastPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_LastLargePageUsed,                                                     \
          Context->BestGraphInfo[18].Coverage.LastLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_LastCacheLineUsed,                                                     \
          Context->BestGraphInfo[18].Coverage.LastCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_TotalNumberOfAssigned,                                                 \
          Context->BestGraphInfo[18].Coverage.TotalNumberOfAssigned,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_CountOfCacheLinesWithNumberOfAssigned_0,                               \
          Context->BestGraphInfo[18].Coverage.NumberOfAssignedPerCacheLineCounts[0],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_CountOfCacheLinesWithNumberOfAssigned_1,                               \
          Context->BestGraphInfo[18].Coverage.NumberOfAssignedPerCacheLineCounts[1],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_CountOfCacheLinesWithNumberOfAssigned_2,                               \
          Context->BestGraphInfo[18].Coverage.NumberOfAssignedPerCacheLineCounts[2],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_CountOfCacheLinesWithNumberOfAssigned_3,                               \
          Context->BestGraphInfo[18].Coverage.NumberOfAssignedPerCacheLineCounts[3],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_CountOfCacheLinesWithNumberOfAssigned_4,                               \
          Context->BestGraphInfo[18].Coverage.NumberOfAssignedPerCacheLineCounts[4],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_CountOfCacheLinesWithNumberOfAssigned_5,                               \
          Context->BestGraphInfo[18].Coverage.NumberOfAssignedPerCacheLineCounts[5],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_CountOfCacheLinesWithNumberOfAssigned_6,                               \
          Context->BestGraphInfo[18].Coverage.NumberOfAssignedPerCacheLineCounts[6],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_CountOfCacheLinesWithNumberOfAssigned_7,                               \
          Context->BestGraphInfo[18].Coverage.NumberOfAssignedPerCacheLineCounts[7],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_CountOfCacheLinesWithNumberOfAssigned_8,                               \
          Context->BestGraphInfo[18].Coverage.NumberOfAssignedPerCacheLineCounts[8],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_CountOfCacheLinesWithNumberOfAssigned_9,                               \
          Context->BestGraphInfo[18].Coverage.NumberOfAssignedPerCacheLineCounts[9],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_CountOfCacheLinesWithNumberOfAssigned_10,                              \
          Context->BestGraphInfo[18].Coverage.NumberOfAssignedPerCacheLineCounts[10],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_CountOfCacheLinesWithNumberOfAssigned_11,                              \
          Context->BestGraphInfo[18].Coverage.NumberOfAssignedPerCacheLineCounts[11],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_CountOfCacheLinesWithNumberOfAssigned_12,                              \
          Context->BestGraphInfo[18].Coverage.NumberOfAssignedPerCacheLineCounts[12],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_CountOfCacheLinesWithNumberOfAssigned_13,                              \
          Context->BestGraphInfo[18].Coverage.NumberOfAssignedPerCacheLineCounts[13],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_CountOfCacheLinesWithNumberOfAssigned_14,                              \
          Context->BestGraphInfo[18].Coverage.NumberOfAssignedPerCacheLineCounts[14],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_CountOfCacheLinesWithNumberOfAssigned_15,                              \
          Context->BestGraphInfo[18].Coverage.NumberOfAssignedPerCacheLineCounts[15],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_CountOfCacheLinesWithNumberOfAssigned_16,                              \
          Context->BestGraphInfo[18].Coverage.NumberOfAssignedPerCacheLineCounts[16],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_TotalNumberOfPages,                                                    \
          Context->BestGraphInfo[19].Coverage.TotalNumberOfPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_TotalNumberOfLargePages,                                               \
          Context->BestGraphInfo[19].Coverage.TotalNumberOfLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_TotalNumberOfCacheLines,                                               \
          Context->BestGraphInfo[19].Coverage.TotalNumberOfCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_NumberOfUsedPages,                                                     \
          Context->BestGraphInfo[19].Coverage.NumberOfUsedPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_NumberOfUsedLargePages,                                                \
          Context->BestGraphInfo[19].Coverage.NumberOfUsedLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_NumberOfUsedCacheLines,                                                \
          Context->BestGraphInfo[19].Coverage.NumberOfUsedCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_NumberOfEmptyPages,                                                    \
          Context->BestGraphInfo[19].Coverage.NumberOfEmptyPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_NumberOfEmptyLargePages,                                               \
          Context->BestGraphInfo[19].Coverage.NumberOfEmptyLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_NumberOfEmptyCacheLines,                                               \
          Context->BestGraphInfo[19].Coverage.NumberOfEmptyCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_FirstPageUsed,                                                         \
          Context->BestGraphInfo[19].Coverage.FirstPageUsed,                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_FirstLargePageUsed,                                                    \
          Context->BestGraphInfo[19].Coverage.FirstLargePageUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_FirstCacheLineUsed,                                                    \
          Context->BestGraphInfo[19].Coverage.FirstCacheLineUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_LastPageUsed,                                                          \
          Context->BestGraphInfo[19].Coverage.LastPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_LastLargePageUsed,                                                     \
          Context->BestGraphInfo[19].Coverage.LastLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_LastCacheLineUsed,                                                     \
          Context->BestGraphInfo[19].Coverage.LastCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_TotalNumberOfAssigned,                                                 \
          Context->BestGraphInfo[19].Coverage.TotalNumberOfAssigned,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_CountOfCacheLinesWithNumberOfAssigned_0,                               \
          Context->BestGraphInfo[19].Coverage.NumberOfAssignedPerCacheLineCounts[0],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_CountOfCacheLinesWithNumberOfAssigned_1,                               \
          Context->BestGraphInfo[19].Coverage.NumberOfAssignedPerCacheLineCounts[1],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_CountOfCacheLinesWithNumberOfAssigned_2,                               \
          Context->BestGraphInfo[19].Coverage.NumberOfAssignedPerCacheLineCounts[2],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_CountOfCacheLinesWithNumberOfAssigned_3,                               \
          Context->BestGraphInfo[19].Coverage.NumberOfAssignedPerCacheLineCounts[3],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_CountOfCacheLinesWithNumberOfAssigned_4,                               \
          Context->BestGraphInfo[19].Coverage.NumberOfAssignedPerCacheLineCounts[4],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_CountOfCacheLinesWithNumberOfAssigned_5,                               \
          Context->BestGraphInfo[19].Coverage.NumberOfAssignedPerCacheLineCounts[5],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_CountOfCacheLinesWithNumberOfAssigned_6,                               \
          Context->BestGraphInfo[19].Coverage.NumberOfAssignedPerCacheLineCounts[6],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_CountOfCacheLinesWithNumberOfAssigned_7,                               \
          Context->BestGraphInfo[19].Coverage.NumberOfAssignedPerCacheLineCounts[7],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_CountOfCacheLinesWithNumberOfAssigned_8,                               \
          Context->BestGraphInfo[19].Coverage.NumberOfAssignedPerCacheLineCounts[8],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_CountOfCacheLinesWithNumberOfAssigned_9,                               \
          Context->BestGraphInfo[19].Coverage.NumberOfAssignedPerCacheLineCounts[9],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_CountOfCacheLinesWithNumberOfAssigned_10,                              \
          Context->BestGraphInfo[19].Coverage.NumberOfAssignedPerCacheLineCounts[10],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_CountOfCacheLinesWithNumberOfAssigned_11,                              \
          Context->BestGraphInfo[19].Coverage.NumberOfAssignedPerCacheLineCounts[11],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_CountOfCacheLinesWithNumberOfAssigned_12,                              \
          Context->BestGraphInfo[19].Coverage.NumberOfAssignedPerCacheLineCounts[12],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_CountOfCacheLinesWithNumberOfAssigned_13,                              \
          Context->BestGraphInfo[19].Coverage.NumberOfAssignedPerCacheLineCounts[13],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_CountOfCacheLinesWithNumberOfAssigned_14,                              \
          Context->BestGraphInfo[19].Coverage.NumberOfAssignedPerCacheLineCounts[14],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_CountOfCacheLinesWithNumberOfAssigned_15,                              \
          Context->BestGraphInfo[19].Coverage.NumberOfAssignedPerCacheLineCounts[15],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_CountOfCacheLinesWithNumberOfAssigned_16,                              \
          Context->BestGraphInfo[19].Coverage.NumberOfAssignedPerCacheLineCounts[16],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_TotalNumberOfPages,                                                    \
          Context->BestGraphInfo[20].Coverage.TotalNumberOfPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_TotalNumberOfLargePages,                                               \
          Context->BestGraphInfo[20].Coverage.TotalNumberOfLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_TotalNumberOfCacheLines,                                               \
          Context->BestGraphInfo[20].Coverage.TotalNumberOfCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_NumberOfUsedPages,                                                     \
          Context->BestGraphInfo[20].Coverage.NumberOfUsedPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_NumberOfUsedLargePages,                                                \
          Context->BestGraphInfo[20].Coverage.NumberOfUsedLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_NumberOfUsedCacheLines,                                                \
          Context->BestGraphInfo[20].Coverage.NumberOfUsedCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_NumberOfEmptyPages,                                                    \
          Context->BestGraphInfo[20].Coverage.NumberOfEmptyPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_NumberOfEmptyLargePages,                                               \
          Context->BestGraphInfo[20].Coverage.NumberOfEmptyLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_NumberOfEmptyCacheLines,                                               \
          Context->BestGraphInfo[20].Coverage.NumberOfEmptyCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_FirstPageUsed,                                                         \
          Context->BestGraphInfo[20].Coverage.FirstPageUsed,                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_FirstLargePageUsed,                                                    \
          Context->BestGraphInfo[20].Coverage.FirstLargePageUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_FirstCacheLineUsed,                                                    \
          Context->BestGraphInfo[20].Coverage.FirstCacheLineUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_LastPageUsed,                                                          \
          Context->BestGraphInfo[20].Coverage.LastPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_LastLargePageUsed,                                                     \
          Context->BestGraphInfo[20].Coverage.LastLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_LastCacheLineUsed,                                                     \
          Context->BestGraphInfo[20].Coverage.LastCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_TotalNumberOfAssigned,                                                 \
          Context->BestGraphInfo[20].Coverage.TotalNumberOfAssigned,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_CountOfCacheLinesWithNumberOfAssigned_0,                               \
          Context->BestGraphInfo[20].Coverage.NumberOfAssignedPerCacheLineCounts[0],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_CountOfCacheLinesWithNumberOfAssigned_1,                               \
          Context->BestGraphInfo[20].Coverage.NumberOfAssignedPerCacheLineCounts[1],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_CountOfCacheLinesWithNumberOfAssigned_2,                               \
          Context->BestGraphInfo[20].Coverage.NumberOfAssignedPerCacheLineCounts[2],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_CountOfCacheLinesWithNumberOfAssigned_3,                               \
          Context->BestGraphInfo[20].Coverage.NumberOfAssignedPerCacheLineCounts[3],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_CountOfCacheLinesWithNumberOfAssigned_4,                               \
          Context->BestGraphInfo[20].Coverage.NumberOfAssignedPerCacheLineCounts[4],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_CountOfCacheLinesWithNumberOfAssigned_5,                               \
          Context->BestGraphInfo[20].Coverage.NumberOfAssignedPerCacheLineCounts[5],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_CountOfCacheLinesWithNumberOfAssigned_6,                               \
          Context->BestGraphInfo[20].Coverage.NumberOfAssignedPerCacheLineCounts[6],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_CountOfCacheLinesWithNumberOfAssigned_7,                               \
          Context->BestGraphInfo[20].Coverage.NumberOfAssignedPerCacheLineCounts[7],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_CountOfCacheLinesWithNumberOfAssigned_8,                               \
          Context->BestGraphInfo[20].Coverage.NumberOfAssignedPerCacheLineCounts[8],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_CountOfCacheLinesWithNumberOfAssigned_9,                               \
          Context->BestGraphInfo[20].Coverage.NumberOfAssignedPerCacheLineCounts[9],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_CountOfCacheLinesWithNumberOfAssigned_10,                              \
          Context->BestGraphInfo[20].Coverage.NumberOfAssignedPerCacheLineCounts[10],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_CountOfCacheLinesWithNumberOfAssigned_11,                              \
          Context->BestGraphInfo[20].Coverage.NumberOfAssignedPerCacheLineCounts[11],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_CountOfCacheLinesWithNumberOfAssigned_12,                              \
          Context->BestGraphInfo[20].Coverage.NumberOfAssignedPerCacheLineCounts[12],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_CountOfCacheLinesWithNumberOfAssigned_13,                              \
          Context->BestGraphInfo[20].Coverage.NumberOfAssignedPerCacheLineCounts[13],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_CountOfCacheLinesWithNumberOfAssigned_14,                              \
          Context->BestGraphInfo[20].Coverage.NumberOfAssignedPerCacheLineCounts[14],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_CountOfCacheLinesWithNumberOfAssigned_15,                              \
          Context->BestGraphInfo[20].Coverage.NumberOfAssignedPerCacheLineCounts[15],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_CountOfCacheLinesWithNumberOfAssigned_16,                              \
          Context->BestGraphInfo[20].Coverage.NumberOfAssignedPerCacheLineCounts[16],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_TotalNumberOfPages,                                                    \
          Context->BestGraphInfo[21].Coverage.TotalNumberOfPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_TotalNumberOfLargePages,                                               \
          Context->BestGraphInfo[21].Coverage.TotalNumberOfLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_TotalNumberOfCacheLines,                                               \
          Context->BestGraphInfo[21].Coverage.TotalNumberOfCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_NumberOfUsedPages,                                                     \
          Context->BestGraphInfo[21].Coverage.NumberOfUsedPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_NumberOfUsedLargePages,                                                \
          Context->BestGraphInfo[21].Coverage.NumberOfUsedLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_NumberOfUsedCacheLines,                                                \
          Context->BestGraphInfo[21].Coverage.NumberOfUsedCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_NumberOfEmptyPages,                                                    \
          Context->BestGraphInfo[21].Coverage.NumberOfEmptyPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_NumberOfEmptyLargePages,                                               \
          Context->BestGraphInfo[21].Coverage.NumberOfEmptyLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_NumberOfEmptyCacheLines,                                               \
          Context->BestGraphInfo[21].Coverage.NumberOfEmptyCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_FirstPageUsed,                                                         \
          Context->BestGraphInfo[21].Coverage.FirstPageUsed,                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_FirstLargePageUsed,                                                    \
          Context->BestGraphInfo[21].Coverage.FirstLargePageUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_FirstCacheLineUsed,                                                    \
          Context->BestGraphInfo[21].Coverage.FirstCacheLineUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_LastPageUsed,                                                          \
          Context->BestGraphInfo[21].Coverage.LastPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_LastLargePageUsed,                                                     \
          Context->BestGraphInfo[21].Coverage.LastLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_LastCacheLineUsed,                                                     \
          Context->BestGraphInfo[21].Coverage.LastCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_TotalNumberOfAssigned,                                                 \
          Context->BestGraphInfo[21].Coverage.TotalNumberOfAssigned,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_CountOfCacheLinesWithNumberOfAssigned_0,                               \
          Context->BestGraphInfo[21].Coverage.NumberOfAssignedPerCacheLineCounts[0],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_CountOfCacheLinesWithNumberOfAssigned_1,                               \
          Context->BestGraphInfo[21].Coverage.NumberOfAssignedPerCacheLineCounts[1],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_CountOfCacheLinesWithNumberOfAssigned_2,                               \
          Context->BestGraphInfo[21].Coverage.NumberOfAssignedPerCacheLineCounts[2],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_CountOfCacheLinesWithNumberOfAssigned_3,                               \
          Context->BestGraphInfo[21].Coverage.NumberOfAssignedPerCacheLineCounts[3],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_CountOfCacheLinesWithNumberOfAssigned_4,                               \
          Context->BestGraphInfo[21].Coverage.NumberOfAssignedPerCacheLineCounts[4],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_CountOfCacheLinesWithNumberOfAssigned_5,                               \
          Context->BestGraphInfo[21].Coverage.NumberOfAssignedPerCacheLineCounts[5],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_CountOfCacheLinesWithNumberOfAssigned_6,                               \
          Context->BestGraphInfo[21].Coverage.NumberOfAssignedPerCacheLineCounts[6],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_CountOfCacheLinesWithNumberOfAssigned_7,                               \
          Context->BestGraphInfo[21].Coverage.NumberOfAssignedPerCacheLineCounts[7],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_CountOfCacheLinesWithNumberOfAssigned_8,                               \
          Context->BestGraphInfo[21].Coverage.NumberOfAssignedPerCacheLineCounts[8],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_CountOfCacheLinesWithNumberOfAssigned_9,                               \
          Context->BestGraphInfo[21].Coverage.NumberOfAssignedPerCacheLineCounts[9],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_CountOfCacheLinesWithNumberOfAssigned_10,                              \
          Context->BestGraphInfo[21].Coverage.NumberOfAssignedPerCacheLineCounts[10],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_CountOfCacheLinesWithNumberOfAssigned_11,                              \
          Context->BestGraphInfo[21].Coverage.NumberOfAssignedPerCacheLineCounts[11],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_CountOfCacheLinesWithNumberOfAssigned_12,                              \
          Context->BestGraphInfo[21].Coverage.NumberOfAssignedPerCacheLineCounts[12],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_CountOfCacheLinesWithNumberOfAssigned_13,                              \
          Context->BestGraphInfo[21].Coverage.NumberOfAssignedPerCacheLineCounts[13],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_CountOfCacheLinesWithNumberOfAssigned_14,                              \
          Context->BestGraphInfo[21].Coverage.NumberOfAssignedPerCacheLineCounts[14],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_CountOfCacheLinesWithNumberOfAssigned_15,                              \
          Context->BestGraphInfo[21].Coverage.NumberOfAssignedPerCacheLineCounts[15],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_CountOfCacheLinesWithNumberOfAssigned_16,                              \
          Context->BestGraphInfo[21].Coverage.NumberOfAssignedPerCacheLineCounts[16],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_TotalNumberOfPages,                                                    \
          Context->BestGraphInfo[22].Coverage.TotalNumberOfPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_TotalNumberOfLargePages,                                               \
          Context->BestGraphInfo[22].Coverage.TotalNumberOfLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_TotalNumberOfCacheLines,                                               \
          Context->BestGraphInfo[22].Coverage.TotalNumberOfCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_NumberOfUsedPages,                                                     \
          Context->BestGraphInfo[22].Coverage.NumberOfUsedPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_NumberOfUsedLargePages,                                                \
          Context->BestGraphInfo[22].Coverage.NumberOfUsedLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_NumberOfUsedCacheLines,                                                \
          Context->BestGraphInfo[22].Coverage.NumberOfUsedCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_NumberOfEmptyPages,                                                    \
          Context->BestGraphInfo[22].Coverage.NumberOfEmptyPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_NumberOfEmptyLargePages,                                               \
          Context->BestGraphInfo[22].Coverage.NumberOfEmptyLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_NumberOfEmptyCacheLines,                                               \
          Context->BestGraphInfo[22].Coverage.NumberOfEmptyCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_FirstPageUsed,                                                         \
          Context->BestGraphInfo[22].Coverage.FirstPageUsed,                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_FirstLargePageUsed,                                                    \
          Context->BestGraphInfo[22].Coverage.FirstLargePageUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_FirstCacheLineUsed,                                                    \
          Context->BestGraphInfo[22].Coverage.FirstCacheLineUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_LastPageUsed,                                                          \
          Context->BestGraphInfo[22].Coverage.LastPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_LastLargePageUsed,                                                     \
          Context->BestGraphInfo[22].Coverage.LastLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_LastCacheLineUsed,                                                     \
          Context->BestGraphInfo[22].Coverage.LastCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_TotalNumberOfAssigned,                                                 \
          Context->BestGraphInfo[22].Coverage.TotalNumberOfAssigned,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_CountOfCacheLinesWithNumberOfAssigned_0,                               \
          Context->BestGraphInfo[22].Coverage.NumberOfAssignedPerCacheLineCounts[0],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_CountOfCacheLinesWithNumberOfAssigned_1,                               \
          Context->BestGraphInfo[22].Coverage.NumberOfAssignedPerCacheLineCounts[1],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_CountOfCacheLinesWithNumberOfAssigned_2,                               \
          Context->BestGraphInfo[22].Coverage.NumberOfAssignedPerCacheLineCounts[2],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_CountOfCacheLinesWithNumberOfAssigned_3,                               \
          Context->BestGraphInfo[22].Coverage.NumberOfAssignedPerCacheLineCounts[3],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_CountOfCacheLinesWithNumberOfAssigned_4,                               \
          Context->BestGraphInfo[22].Coverage.NumberOfAssignedPerCacheLineCounts[4],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_CountOfCacheLinesWithNumberOfAssigned_5,                               \
          Context->BestGraphInfo[22].Coverage.NumberOfAssignedPerCacheLineCounts[5],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_CountOfCacheLinesWithNumberOfAssigned_6,                               \
          Context->BestGraphInfo[22].Coverage.NumberOfAssignedPerCacheLineCounts[6],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_CountOfCacheLinesWithNumberOfAssigned_7,                               \
          Context->BestGraphInfo[22].Coverage.NumberOfAssignedPerCacheLineCounts[7],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_CountOfCacheLinesWithNumberOfAssigned_8,                               \
          Context->BestGraphInfo[22].Coverage.NumberOfAssignedPerCacheLineCounts[8],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_CountOfCacheLinesWithNumberOfAssigned_9,                               \
          Context->BestGraphInfo[22].Coverage.NumberOfAssignedPerCacheLineCounts[9],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_CountOfCacheLinesWithNumberOfAssigned_10,                              \
          Context->BestGraphInfo[22].Coverage.NumberOfAssignedPerCacheLineCounts[10],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_CountOfCacheLinesWithNumberOfAssigned_11,                              \
          Context->BestGraphInfo[22].Coverage.NumberOfAssignedPerCacheLineCounts[11],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_CountOfCacheLinesWithNumberOfAssigned_12,                              \
          Context->BestGraphInfo[22].Coverage.NumberOfAssignedPerCacheLineCounts[12],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_CountOfCacheLinesWithNumberOfAssigned_13,                              \
          Context->BestGraphInfo[22].Coverage.NumberOfAssignedPerCacheLineCounts[13],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_CountOfCacheLinesWithNumberOfAssigned_14,                              \
          Context->BestGraphInfo[22].Coverage.NumberOfAssignedPerCacheLineCounts[14],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_CountOfCacheLinesWithNumberOfAssigned_15,                              \
          Context->BestGraphInfo[22].Coverage.NumberOfAssignedPerCacheLineCounts[15],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_CountOfCacheLinesWithNumberOfAssigned_16,                              \
          Context->BestGraphInfo[22].Coverage.NumberOfAssignedPerCacheLineCounts[16],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_TotalNumberOfPages,                                                    \
          Context->BestGraphInfo[23].Coverage.TotalNumberOfPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_TotalNumberOfLargePages,                                               \
          Context->BestGraphInfo[23].Coverage.TotalNumberOfLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_TotalNumberOfCacheLines,                                               \
          Context->BestGraphInfo[23].Coverage.TotalNumberOfCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_NumberOfUsedPages,                                                     \
          Context->BestGraphInfo[23].Coverage.NumberOfUsedPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_NumberOfUsedLargePages,                                                \
          Context->BestGraphInfo[23].Coverage.NumberOfUsedLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_NumberOfUsedCacheLines,                                                \
          Context->BestGraphInfo[23].Coverage.NumberOfUsedCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_NumberOfEmptyPages,                                                    \
          Context->BestGraphInfo[23].Coverage.NumberOfEmptyPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_NumberOfEmptyLargePages,                                               \
          Context->BestGraphInfo[23].Coverage.NumberOfEmptyLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_NumberOfEmptyCacheLines,                                               \
          Context->BestGraphInfo[23].Coverage.NumberOfEmptyCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_FirstPageUsed,                                                         \
          Context->BestGraphInfo[23].Coverage.FirstPageUsed,                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_FirstLargePageUsed,                                                    \
          Context->BestGraphInfo[23].Coverage.FirstLargePageUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_FirstCacheLineUsed,                                                    \
          Context->BestGraphInfo[23].Coverage.FirstCacheLineUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_LastPageUsed,                                                          \
          Context->BestGraphInfo[23].Coverage.LastPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_LastLargePageUsed,                                                     \
          Context->BestGraphInfo[23].Coverage.LastLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_LastCacheLineUsed,                                                     \
          Context->BestGraphInfo[23].Coverage.LastCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_TotalNumberOfAssigned,                                                 \
          Context->BestGraphInfo[23].Coverage.TotalNumberOfAssigned,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_CountOfCacheLinesWithNumberOfAssigned_0,                               \
          Context->BestGraphInfo[23].Coverage.NumberOfAssignedPerCacheLineCounts[0],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_CountOfCacheLinesWithNumberOfAssigned_1,                               \
          Context->BestGraphInfo[23].Coverage.NumberOfAssignedPerCacheLineCounts[1],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_CountOfCacheLinesWithNumberOfAssigned_2,                               \
          Context->BestGraphInfo[23].Coverage.NumberOfAssignedPerCacheLineCounts[2],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_CountOfCacheLinesWithNumberOfAssigned_3,                               \
          Context->BestGraphInfo[23].Coverage.NumberOfAssignedPerCacheLineCounts[3],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_CountOfCacheLinesWithNumberOfAssigned_4,                               \
          Context->BestGraphInfo[23].Coverage.NumberOfAssignedPerCacheLineCounts[4],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_CountOfCacheLinesWithNumberOfAssigned_5,                               \
          Context->BestGraphInfo[23].Coverage.NumberOfAssignedPerCacheLineCounts[5],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_CountOfCacheLinesWithNumberOfAssigned_6,                               \
          Context->BestGraphInfo[23].Coverage.NumberOfAssignedPerCacheLineCounts[6],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_CountOfCacheLinesWithNumberOfAssigned_7,                               \
          Context->BestGraphInfo[23].Coverage.NumberOfAssignedPerCacheLineCounts[7],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_CountOfCacheLinesWithNumberOfAssigned_8,                               \
          Context->BestGraphInfo[23].Coverage.NumberOfAssignedPerCacheLineCounts[8],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_CountOfCacheLinesWithNumberOfAssigned_9,                               \
          Context->BestGraphInfo[23].Coverage.NumberOfAssignedPerCacheLineCounts[9],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_CountOfCacheLinesWithNumberOfAssigned_10,                              \
          Context->BestGraphInfo[23].Coverage.NumberOfAssignedPerCacheLineCounts[10],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_CountOfCacheLinesWithNumberOfAssigned_11,                              \
          Context->BestGraphInfo[23].Coverage.NumberOfAssignedPerCacheLineCounts[11],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_CountOfCacheLinesWithNumberOfAssigned_12,                              \
          Context->BestGraphInfo[23].Coverage.NumberOfAssignedPerCacheLineCounts[12],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_CountOfCacheLinesWithNumberOfAssigned_13,                              \
          Context->BestGraphInfo[23].Coverage.NumberOfAssignedPerCacheLineCounts[13],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_CountOfCacheLinesWithNumberOfAssigned_14,                              \
          Context->BestGraphInfo[23].Coverage.NumberOfAssignedPerCacheLineCounts[14],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_CountOfCacheLinesWithNumberOfAssigned_15,                              \
          Context->BestGraphInfo[23].Coverage.NumberOfAssignedPerCacheLineCounts[15],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_CountOfCacheLinesWithNumberOfAssigned_16,                              \
          Context->BestGraphInfo[23].Coverage.NumberOfAssignedPerCacheLineCounts[16],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_TotalNumberOfPages,                                                    \
          Context->BestGraphInfo[24].Coverage.TotalNumberOfPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_TotalNumberOfLargePages,                                               \
          Context->BestGraphInfo[24].Coverage.TotalNumberOfLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_TotalNumberOfCacheLines,                                               \
          Context->BestGraphInfo[24].Coverage.TotalNumberOfCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_NumberOfUsedPages,                                                     \
          Context->BestGraphInfo[24].Coverage.NumberOfUsedPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_NumberOfUsedLargePages,                                                \
          Context->BestGraphInfo[24].Coverage.NumberOfUsedLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_NumberOfUsedCacheLines,                                                \
          Context->BestGraphInfo[24].Coverage.NumberOfUsedCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_NumberOfEmptyPages,                                                    \
          Context->BestGraphInfo[24].Coverage.NumberOfEmptyPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_NumberOfEmptyLargePages,                                               \
          Context->BestGraphInfo[24].Coverage.NumberOfEmptyLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_NumberOfEmptyCacheLines,                                               \
          Context->BestGraphInfo[24].Coverage.NumberOfEmptyCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_FirstPageUsed,                                                         \
          Context->BestGraphInfo[24].Coverage.FirstPageUsed,                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_FirstLargePageUsed,                                                    \
          Context->BestGraphInfo[24].Coverage.FirstLargePageUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_FirstCacheLineUsed,                                                    \
          Context->BestGraphInfo[24].Coverage.FirstCacheLineUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_LastPageUsed,                                                          \
          Context->BestGraphInfo[24].Coverage.LastPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_LastLargePageUsed,                                                     \
          Context->BestGraphInfo[24].Coverage.LastLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_LastCacheLineUsed,                                                     \
          Context->BestGraphInfo[24].Coverage.LastCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_TotalNumberOfAssigned,                                                 \
          Context->BestGraphInfo[24].Coverage.TotalNumberOfAssigned,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_CountOfCacheLinesWithNumberOfAssigned_0,                               \
          Context->BestGraphInfo[24].Coverage.NumberOfAssignedPerCacheLineCounts[0],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_CountOfCacheLinesWithNumberOfAssigned_1,                               \
          Context->BestGraphInfo[24].Coverage.NumberOfAssignedPerCacheLineCounts[1],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_CountOfCacheLinesWithNumberOfAssigned_2,                               \
          Context->BestGraphInfo[24].Coverage.NumberOfAssignedPerCacheLineCounts[2],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_CountOfCacheLinesWithNumberOfAssigned_3,                               \
          Context->BestGraphInfo[24].Coverage.NumberOfAssignedPerCacheLineCounts[3],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_CountOfCacheLinesWithNumberOfAssigned_4,                               \
          Context->BestGraphInfo[24].Coverage.NumberOfAssignedPerCacheLineCounts[4],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_CountOfCacheLinesWithNumberOfAssigned_5,                               \
          Context->BestGraphInfo[24].Coverage.NumberOfAssignedPerCacheLineCounts[5],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_CountOfCacheLinesWithNumberOfAssigned_6,                               \
          Context->BestGraphInfo[24].Coverage.NumberOfAssignedPerCacheLineCounts[6],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_CountOfCacheLinesWithNumberOfAssigned_7,                               \
          Context->BestGraphInfo[24].Coverage.NumberOfAssignedPerCacheLineCounts[7],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_CountOfCacheLinesWithNumberOfAssigned_8,                               \
          Context->BestGraphInfo[24].Coverage.NumberOfAssignedPerCacheLineCounts[8],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_CountOfCacheLinesWithNumberOfAssigned_9,                               \
          Context->BestGraphInfo[24].Coverage.NumberOfAssignedPerCacheLineCounts[9],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_CountOfCacheLinesWithNumberOfAssigned_10,                              \
          Context->BestGraphInfo[24].Coverage.NumberOfAssignedPerCacheLineCounts[10],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_CountOfCacheLinesWithNumberOfAssigned_11,                              \
          Context->BestGraphInfo[24].Coverage.NumberOfAssignedPerCacheLineCounts[11],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_CountOfCacheLinesWithNumberOfAssigned_12,                              \
          Context->BestGraphInfo[24].Coverage.NumberOfAssignedPerCacheLineCounts[12],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_CountOfCacheLinesWithNumberOfAssigned_13,                              \
          Context->BestGraphInfo[24].Coverage.NumberOfAssignedPerCacheLineCounts[13],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_CountOfCacheLinesWithNumberOfAssigned_14,                              \
          Context->BestGraphInfo[24].Coverage.NumberOfAssignedPerCacheLineCounts[14],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_CountOfCacheLinesWithNumberOfAssigned_15,                              \
          Context->BestGraphInfo[24].Coverage.NumberOfAssignedPerCacheLineCounts[15],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_CountOfCacheLinesWithNumberOfAssigned_16,                              \
          Context->BestGraphInfo[24].Coverage.NumberOfAssignedPerCacheLineCounts[16],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_TotalNumberOfPages,                                                    \
          Context->BestGraphInfo[25].Coverage.TotalNumberOfPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_TotalNumberOfLargePages,                                               \
          Context->BestGraphInfo[25].Coverage.TotalNumberOfLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_TotalNumberOfCacheLines,                                               \
          Context->BestGraphInfo[25].Coverage.TotalNumberOfCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_NumberOfUsedPages,                                                     \
          Context->BestGraphInfo[25].Coverage.NumberOfUsedPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_NumberOfUsedLargePages,                                                \
          Context->BestGraphInfo[25].Coverage.NumberOfUsedLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_NumberOfUsedCacheLines,                                                \
          Context->BestGraphInfo[25].Coverage.NumberOfUsedCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_NumberOfEmptyPages,                                                    \
          Context->BestGraphInfo[25].Coverage.NumberOfEmptyPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_NumberOfEmptyLargePages,                                               \
          Context->BestGraphInfo[25].Coverage.NumberOfEmptyLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_NumberOfEmptyCacheLines,                                               \
          Context->BestGraphInfo[25].Coverage.NumberOfEmptyCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_FirstPageUsed,                                                         \
          Context->BestGraphInfo[25].Coverage.FirstPageUsed,                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_FirstLargePageUsed,                                                    \
          Context->BestGraphInfo[25].Coverage.FirstLargePageUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_FirstCacheLineUsed,                                                    \
          Context->BestGraphInfo[25].Coverage.FirstCacheLineUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_LastPageUsed,                                                          \
          Context->BestGraphInfo[25].Coverage.LastPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_LastLargePageUsed,                                                     \
          Context->BestGraphInfo[25].Coverage.LastLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_LastCacheLineUsed,                                                     \
          Context->BestGraphInfo[25].Coverage.LastCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_TotalNumberOfAssigned,                                                 \
          Context->BestGraphInfo[25].Coverage.TotalNumberOfAssigned,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_CountOfCacheLinesWithNumberOfAssigned_0,                               \
          Context->BestGraphInfo[25].Coverage.NumberOfAssignedPerCacheLineCounts[0],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_CountOfCacheLinesWithNumberOfAssigned_1,                               \
          Context->BestGraphInfo[25].Coverage.NumberOfAssignedPerCacheLineCounts[1],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_CountOfCacheLinesWithNumberOfAssigned_2,                               \
          Context->BestGraphInfo[25].Coverage.NumberOfAssignedPerCacheLineCounts[2],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_CountOfCacheLinesWithNumberOfAssigned_3,                               \
          Context->BestGraphInfo[25].Coverage.NumberOfAssignedPerCacheLineCounts[3],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_CountOfCacheLinesWithNumberOfAssigned_4,                               \
          Context->BestGraphInfo[25].Coverage.NumberOfAssignedPerCacheLineCounts[4],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_CountOfCacheLinesWithNumberOfAssigned_5,                               \
          Context->BestGraphInfo[25].Coverage.NumberOfAssignedPerCacheLineCounts[5],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_CountOfCacheLinesWithNumberOfAssigned_6,                               \
          Context->BestGraphInfo[25].Coverage.NumberOfAssignedPerCacheLineCounts[6],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_CountOfCacheLinesWithNumberOfAssigned_7,                               \
          Context->BestGraphInfo[25].Coverage.NumberOfAssignedPerCacheLineCounts[7],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_CountOfCacheLinesWithNumberOfAssigned_8,                               \
          Context->BestGraphInfo[25].Coverage.NumberOfAssignedPerCacheLineCounts[8],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_CountOfCacheLinesWithNumberOfAssigned_9,                               \
          Context->BestGraphInfo[25].Coverage.NumberOfAssignedPerCacheLineCounts[9],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_CountOfCacheLinesWithNumberOfAssigned_10,                              \
          Context->BestGraphInfo[25].Coverage.NumberOfAssignedPerCacheLineCounts[10],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_CountOfCacheLinesWithNumberOfAssigned_11,                              \
          Context->BestGraphInfo[25].Coverage.NumberOfAssignedPerCacheLineCounts[11],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_CountOfCacheLinesWithNumberOfAssigned_12,                              \
          Context->BestGraphInfo[25].Coverage.NumberOfAssignedPerCacheLineCounts[12],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_CountOfCacheLinesWithNumberOfAssigned_13,                              \
          Context->BestGraphInfo[25].Coverage.NumberOfAssignedPerCacheLineCounts[13],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_CountOfCacheLinesWithNumberOfAssigned_14,                              \
          Context->BestGraphInfo[25].Coverage.NumberOfAssignedPerCacheLineCounts[14],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_CountOfCacheLinesWithNumberOfAssigned_15,                              \
          Context->BestGraphInfo[25].Coverage.NumberOfAssignedPerCacheLineCounts[15],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_CountOfCacheLinesWithNumberOfAssigned_16,                              \
          Context->BestGraphInfo[25].Coverage.NumberOfAssignedPerCacheLineCounts[16],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_TotalNumberOfPages,                                                    \
          Context->BestGraphInfo[26].Coverage.TotalNumberOfPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_TotalNumberOfLargePages,                                               \
          Context->BestGraphInfo[26].Coverage.TotalNumberOfLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_TotalNumberOfCacheLines,                                               \
          Context->BestGraphInfo[26].Coverage.TotalNumberOfCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_NumberOfUsedPages,                                                     \
          Context->BestGraphInfo[26].Coverage.NumberOfUsedPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_NumberOfUsedLargePages,                                                \
          Context->BestGraphInfo[26].Coverage.NumberOfUsedLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_NumberOfUsedCacheLines,                                                \
          Context->BestGraphInfo[26].Coverage.NumberOfUsedCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_NumberOfEmptyPages,                                                    \
          Context->BestGraphInfo[26].Coverage.NumberOfEmptyPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_NumberOfEmptyLargePages,                                               \
          Context->BestGraphInfo[26].Coverage.NumberOfEmptyLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_NumberOfEmptyCacheLines,                                               \
          Context->BestGraphInfo[26].Coverage.NumberOfEmptyCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_FirstPageUsed,                                                         \
          Context->BestGraphInfo[26].Coverage.FirstPageUsed,                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_FirstLargePageUsed,                                                    \
          Context->BestGraphInfo[26].Coverage.FirstLargePageUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_FirstCacheLineUsed,                                                    \
          Context->BestGraphInfo[26].Coverage.FirstCacheLineUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_LastPageUsed,                                                          \
          Context->BestGraphInfo[26].Coverage.LastPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_LastLargePageUsed,                                                     \
          Context->BestGraphInfo[26].Coverage.LastLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_LastCacheLineUsed,                                                     \
          Context->BestGraphInfo[26].Coverage.LastCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_TotalNumberOfAssigned,                                                 \
          Context->BestGraphInfo[26].Coverage.TotalNumberOfAssigned,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_CountOfCacheLinesWithNumberOfAssigned_0,                               \
          Context->BestGraphInfo[26].Coverage.NumberOfAssignedPerCacheLineCounts[0],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_CountOfCacheLinesWithNumberOfAssigned_1,                               \
          Context->BestGraphInfo[26].Coverage.NumberOfAssignedPerCacheLineCounts[1],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_CountOfCacheLinesWithNumberOfAssigned_2,                               \
          Context->BestGraphInfo[26].Coverage.NumberOfAssignedPerCacheLineCounts[2],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_CountOfCacheLinesWithNumberOfAssigned_3,                               \
          Context->BestGraphInfo[26].Coverage.NumberOfAssignedPerCacheLineCounts[3],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_CountOfCacheLinesWithNumberOfAssigned_4,                               \
          Context->BestGraphInfo[26].Coverage.NumberOfAssignedPerCacheLineCounts[4],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_CountOfCacheLinesWithNumberOfAssigned_5,                               \
          Context->BestGraphInfo[26].Coverage.NumberOfAssignedPerCacheLineCounts[5],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_CountOfCacheLinesWithNumberOfAssigned_6,                               \
          Context->BestGraphInfo[26].Coverage.NumberOfAssignedPerCacheLineCounts[6],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_CountOfCacheLinesWithNumberOfAssigned_7,                               \
          Context->BestGraphInfo[26].Coverage.NumberOfAssignedPerCacheLineCounts[7],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_CountOfCacheLinesWithNumberOfAssigned_8,                               \
          Context->BestGraphInfo[26].Coverage.NumberOfAssignedPerCacheLineCounts[8],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_CountOfCacheLinesWithNumberOfAssigned_9,                               \
          Context->BestGraphInfo[26].Coverage.NumberOfAssignedPerCacheLineCounts[9],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_CountOfCacheLinesWithNumberOfAssigned_10,                              \
          Context->BestGraphInfo[26].Coverage.NumberOfAssignedPerCacheLineCounts[10],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_CountOfCacheLinesWithNumberOfAssigned_11,                              \
          Context->BestGraphInfo[26].Coverage.NumberOfAssignedPerCacheLineCounts[11],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_CountOfCacheLinesWithNumberOfAssigned_12,                              \
          Context->BestGraphInfo[26].Coverage.NumberOfAssignedPerCacheLineCounts[12],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_CountOfCacheLinesWithNumberOfAssigned_13,                              \
          Context->BestGraphInfo[26].Coverage.NumberOfAssignedPerCacheLineCounts[13],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_CountOfCacheLinesWithNumberOfAssigned_14,                              \
          Context->BestGraphInfo[26].Coverage.NumberOfAssignedPerCacheLineCounts[14],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_CountOfCacheLinesWithNumberOfAssigned_15,                              \
          Context->BestGraphInfo[26].Coverage.NumberOfAssignedPerCacheLineCounts[15],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_CountOfCacheLinesWithNumberOfAssigned_16,                              \
          Context->BestGraphInfo[26].Coverage.NumberOfAssignedPerCacheLineCounts[16],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_TotalNumberOfPages,                                                    \
          Context->BestGraphInfo[27].Coverage.TotalNumberOfPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_TotalNumberOfLargePages,                                               \
          Context->BestGraphInfo[27].Coverage.TotalNumberOfLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_TotalNumberOfCacheLines,                                               \
          Context->BestGraphInfo[27].Coverage.TotalNumberOfCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_NumberOfUsedPages,                                                     \
          Context->BestGraphInfo[27].Coverage.NumberOfUsedPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_NumberOfUsedLargePages,                                                \
          Context->BestGraphInfo[27].Coverage.NumberOfUsedLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_NumberOfUsedCacheLines,                                                \
          Context->BestGraphInfo[27].Coverage.NumberOfUsedCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_NumberOfEmptyPages,                                                    \
          Context->BestGraphInfo[27].Coverage.NumberOfEmptyPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_NumberOfEmptyLargePages,                                               \
          Context->BestGraphInfo[27].Coverage.NumberOfEmptyLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_NumberOfEmptyCacheLines,                                               \
          Context->BestGraphInfo[27].Coverage.NumberOfEmptyCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_FirstPageUsed,                                                         \
          Context->BestGraphInfo[27].Coverage.FirstPageUsed,                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_FirstLargePageUsed,                                                    \
          Context->BestGraphInfo[27].Coverage.FirstLargePageUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_FirstCacheLineUsed,                                                    \
          Context->BestGraphInfo[27].Coverage.FirstCacheLineUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_LastPageUsed,                                                          \
          Context->BestGraphInfo[27].Coverage.LastPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_LastLargePageUsed,                                                     \
          Context->BestGraphInfo[27].Coverage.LastLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_LastCacheLineUsed,                                                     \
          Context->BestGraphInfo[27].Coverage.LastCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_TotalNumberOfAssigned,                                                 \
          Context->BestGraphInfo[27].Coverage.TotalNumberOfAssigned,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_CountOfCacheLinesWithNumberOfAssigned_0,                               \
          Context->BestGraphInfo[27].Coverage.NumberOfAssignedPerCacheLineCounts[0],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_CountOfCacheLinesWithNumberOfAssigned_1,                               \
          Context->BestGraphInfo[27].Coverage.NumberOfAssignedPerCacheLineCounts[1],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_CountOfCacheLinesWithNumberOfAssigned_2,                               \
          Context->BestGraphInfo[27].Coverage.NumberOfAssignedPerCacheLineCounts[2],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_CountOfCacheLinesWithNumberOfAssigned_3,                               \
          Context->BestGraphInfo[27].Coverage.NumberOfAssignedPerCacheLineCounts[3],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_CountOfCacheLinesWithNumberOfAssigned_4,                               \
          Context->BestGraphInfo[27].Coverage.NumberOfAssignedPerCacheLineCounts[4],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_CountOfCacheLinesWithNumberOfAssigned_5,                               \
          Context->BestGraphInfo[27].Coverage.NumberOfAssignedPerCacheLineCounts[5],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_CountOfCacheLinesWithNumberOfAssigned_6,                               \
          Context->BestGraphInfo[27].Coverage.NumberOfAssignedPerCacheLineCounts[6],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_CountOfCacheLinesWithNumberOfAssigned_7,                               \
          Context->BestGraphInfo[27].Coverage.NumberOfAssignedPerCacheLineCounts[7],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_CountOfCacheLinesWithNumberOfAssigned_8,                               \
          Context->BestGraphInfo[27].Coverage.NumberOfAssignedPerCacheLineCounts[8],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_CountOfCacheLinesWithNumberOfAssigned_9,                               \
          Context->BestGraphInfo[27].Coverage.NumberOfAssignedPerCacheLineCounts[9],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_CountOfCacheLinesWithNumberOfAssigned_10,                              \
          Context->BestGraphInfo[27].Coverage.NumberOfAssignedPerCacheLineCounts[10],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_CountOfCacheLinesWithNumberOfAssigned_11,                              \
          Context->BestGraphInfo[27].Coverage.NumberOfAssignedPerCacheLineCounts[11],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_CountOfCacheLinesWithNumberOfAssigned_12,                              \
          Context->BestGraphInfo[27].Coverage.NumberOfAssignedPerCacheLineCounts[12],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_CountOfCacheLinesWithNumberOfAssigned_13,                              \
          Context->BestGraphInfo[27].Coverage.NumberOfAssignedPerCacheLineCounts[13],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_CountOfCacheLinesWithNumberOfAssigned_14,                              \
          Context->BestGraphInfo[27].Coverage.NumberOfAssignedPerCacheLineCounts[14],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_CountOfCacheLinesWithNumberOfAssigned_15,                              \
          Context->BestGraphInfo[27].Coverage.NumberOfAssignedPerCacheLineCounts[15],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_CountOfCacheLinesWithNumberOfAssigned_16,                              \
          Context->BestGraphInfo[27].Coverage.NumberOfAssignedPerCacheLineCounts[16],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_TotalNumberOfPages,                                                    \
          Context->BestGraphInfo[28].Coverage.TotalNumberOfPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_TotalNumberOfLargePages,                                               \
          Context->BestGraphInfo[28].Coverage.TotalNumberOfLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_TotalNumberOfCacheLines,                                               \
          Context->BestGraphInfo[28].Coverage.TotalNumberOfCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_NumberOfUsedPages,                                                     \
          Context->BestGraphInfo[28].Coverage.NumberOfUsedPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_NumberOfUsedLargePages,                                                \
          Context->BestGraphInfo[28].Coverage.NumberOfUsedLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_NumberOfUsedCacheLines,                                                \
          Context->BestGraphInfo[28].Coverage.NumberOfUsedCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_NumberOfEmptyPages,                                                    \
          Context->BestGraphInfo[28].Coverage.NumberOfEmptyPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_NumberOfEmptyLargePages,                                               \
          Context->BestGraphInfo[28].Coverage.NumberOfEmptyLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_NumberOfEmptyCacheLines,                                               \
          Context->BestGraphInfo[28].Coverage.NumberOfEmptyCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_FirstPageUsed,                                                         \
          Context->BestGraphInfo[28].Coverage.FirstPageUsed,                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_FirstLargePageUsed,                                                    \
          Context->BestGraphInfo[28].Coverage.FirstLargePageUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_FirstCacheLineUsed,                                                    \
          Context->BestGraphInfo[28].Coverage.FirstCacheLineUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_LastPageUsed,                                                          \
          Context->BestGraphInfo[28].Coverage.LastPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_LastLargePageUsed,                                                     \
          Context->BestGraphInfo[28].Coverage.LastLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_LastCacheLineUsed,                                                     \
          Context->BestGraphInfo[28].Coverage.LastCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_TotalNumberOfAssigned,                                                 \
          Context->BestGraphInfo[28].Coverage.TotalNumberOfAssigned,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_CountOfCacheLinesWithNumberOfAssigned_0,                               \
          Context->BestGraphInfo[28].Coverage.NumberOfAssignedPerCacheLineCounts[0],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_CountOfCacheLinesWithNumberOfAssigned_1,                               \
          Context->BestGraphInfo[28].Coverage.NumberOfAssignedPerCacheLineCounts[1],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_CountOfCacheLinesWithNumberOfAssigned_2,                               \
          Context->BestGraphInfo[28].Coverage.NumberOfAssignedPerCacheLineCounts[2],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_CountOfCacheLinesWithNumberOfAssigned_3,                               \
          Context->BestGraphInfo[28].Coverage.NumberOfAssignedPerCacheLineCounts[3],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_CountOfCacheLinesWithNumberOfAssigned_4,                               \
          Context->BestGraphInfo[28].Coverage.NumberOfAssignedPerCacheLineCounts[4],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_CountOfCacheLinesWithNumberOfAssigned_5,                               \
          Context->BestGraphInfo[28].Coverage.NumberOfAssignedPerCacheLineCounts[5],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_CountOfCacheLinesWithNumberOfAssigned_6,                               \
          Context->BestGraphInfo[28].Coverage.NumberOfAssignedPerCacheLineCounts[6],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_CountOfCacheLinesWithNumberOfAssigned_7,                               \
          Context->BestGraphInfo[28].Coverage.NumberOfAssignedPerCacheLineCounts[7],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_CountOfCacheLinesWithNumberOfAssigned_8,                               \
          Context->BestGraphInfo[28].Coverage.NumberOfAssignedPerCacheLineCounts[8],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_CountOfCacheLinesWithNumberOfAssigned_9,                               \
          Context->BestGraphInfo[28].Coverage.NumberOfAssignedPerCacheLineCounts[9],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_CountOfCacheLinesWithNumberOfAssigned_10,                              \
          Context->BestGraphInfo[28].Coverage.NumberOfAssignedPerCacheLineCounts[10],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_CountOfCacheLinesWithNumberOfAssigned_11,                              \
          Context->BestGraphInfo[28].Coverage.NumberOfAssignedPerCacheLineCounts[11],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_CountOfCacheLinesWithNumberOfAssigned_12,                              \
          Context->BestGraphInfo[28].Coverage.NumberOfAssignedPerCacheLineCounts[12],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_CountOfCacheLinesWithNumberOfAssigned_13,                              \
          Context->BestGraphInfo[28].Coverage.NumberOfAssignedPerCacheLineCounts[13],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_CountOfCacheLinesWithNumberOfAssigned_14,                              \
          Context->BestGraphInfo[28].Coverage.NumberOfAssignedPerCacheLineCounts[14],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_CountOfCacheLinesWithNumberOfAssigned_15,                              \
          Context->BestGraphInfo[28].Coverage.NumberOfAssignedPerCacheLineCounts[15],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_CountOfCacheLinesWithNumberOfAssigned_16,                              \
          Context->BestGraphInfo[28].Coverage.NumberOfAssignedPerCacheLineCounts[16],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_TotalNumberOfPages,                                                    \
          Context->BestGraphInfo[29].Coverage.TotalNumberOfPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_TotalNumberOfLargePages,                                               \
          Context->BestGraphInfo[29].Coverage.TotalNumberOfLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_TotalNumberOfCacheLines,                                               \
          Context->BestGraphInfo[29].Coverage.TotalNumberOfCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_NumberOfUsedPages,                                                     \
          Context->BestGraphInfo[29].Coverage.NumberOfUsedPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_NumberOfUsedLargePages,                                                \
          Context->BestGraphInfo[29].Coverage.NumberOfUsedLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_NumberOfUsedCacheLines,                                                \
          Context->BestGraphInfo[29].Coverage.NumberOfUsedCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_NumberOfEmptyPages,                                                    \
          Context->BestGraphInfo[29].Coverage.NumberOfEmptyPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_NumberOfEmptyLargePages,                                               \
          Context->BestGraphInfo[29].Coverage.NumberOfEmptyLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_NumberOfEmptyCacheLines,                                               \
          Context->BestGraphInfo[29].Coverage.NumberOfEmptyCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_FirstPageUsed,                                                         \
          Context->BestGraphInfo[29].Coverage.FirstPageUsed,                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_FirstLargePageUsed,                                                    \
          Context->BestGraphInfo[29].Coverage.FirstLargePageUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_FirstCacheLineUsed,                                                    \
          Context->BestGraphInfo[29].Coverage.FirstCacheLineUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_LastPageUsed,                                                          \
          Context->BestGraphInfo[29].Coverage.LastPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_LastLargePageUsed,                                                     \
          Context->BestGraphInfo[29].Coverage.LastLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_LastCacheLineUsed,                                                     \
          Context->BestGraphInfo[29].Coverage.LastCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_TotalNumberOfAssigned,                                                 \
          Context->BestGraphInfo[29].Coverage.TotalNumberOfAssigned,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_CountOfCacheLinesWithNumberOfAssigned_0,                               \
          Context->BestGraphInfo[29].Coverage.NumberOfAssignedPerCacheLineCounts[0],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_CountOfCacheLinesWithNumberOfAssigned_1,                               \
          Context->BestGraphInfo[29].Coverage.NumberOfAssignedPerCacheLineCounts[1],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_CountOfCacheLinesWithNumberOfAssigned_2,                               \
          Context->BestGraphInfo[29].Coverage.NumberOfAssignedPerCacheLineCounts[2],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_CountOfCacheLinesWithNumberOfAssigned_3,                               \
          Context->BestGraphInfo[29].Coverage.NumberOfAssignedPerCacheLineCounts[3],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_CountOfCacheLinesWithNumberOfAssigned_4,                               \
          Context->BestGraphInfo[29].Coverage.NumberOfAssignedPerCacheLineCounts[4],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_CountOfCacheLinesWithNumberOfAssigned_5,                               \
          Context->BestGraphInfo[29].Coverage.NumberOfAssignedPerCacheLineCounts[5],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_CountOfCacheLinesWithNumberOfAssigned_6,                               \
          Context->BestGraphInfo[29].Coverage.NumberOfAssignedPerCacheLineCounts[6],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_CountOfCacheLinesWithNumberOfAssigned_7,                               \
          Context->BestGraphInfo[29].Coverage.NumberOfAssignedPerCacheLineCounts[7],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_CountOfCacheLinesWithNumberOfAssigned_8,                               \
          Context->BestGraphInfo[29].Coverage.NumberOfAssignedPerCacheLineCounts[8],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_CountOfCacheLinesWithNumberOfAssigned_9,                               \
          Context->BestGraphInfo[29].Coverage.NumberOfAssignedPerCacheLineCounts[9],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_CountOfCacheLinesWithNumberOfAssigned_10,                              \
          Context->BestGraphInfo[29].Coverage.NumberOfAssignedPerCacheLineCounts[10],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_CountOfCacheLinesWithNumberOfAssigned_11,                              \
          Context->BestGraphInfo[29].Coverage.NumberOfAssignedPerCacheLineCounts[11],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_CountOfCacheLinesWithNumberOfAssigned_12,                              \
          Context->BestGraphInfo[29].Coverage.NumberOfAssignedPerCacheLineCounts[12],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_CountOfCacheLinesWithNumberOfAssigned_13,                              \
          Context->BestGraphInfo[29].Coverage.NumberOfAssignedPerCacheLineCounts[13],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_CountOfCacheLinesWithNumberOfAssigned_14,                              \
          Context->BestGraphInfo[29].Coverage.NumberOfAssignedPerCacheLineCounts[14],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_CountOfCacheLinesWithNumberOfAssigned_15,                              \
          Context->BestGraphInfo[29].Coverage.NumberOfAssignedPerCacheLineCounts[15],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_CountOfCacheLinesWithNumberOfAssigned_16,                              \
          Context->BestGraphInfo[29].Coverage.NumberOfAssignedPerCacheLineCounts[16],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_TotalNumberOfPages,                                                    \
          Context->BestGraphInfo[30].Coverage.TotalNumberOfPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_TotalNumberOfLargePages,                                               \
          Context->BestGraphInfo[30].Coverage.TotalNumberOfLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_TotalNumberOfCacheLines,                                               \
          Context->BestGraphInfo[30].Coverage.TotalNumberOfCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_NumberOfUsedPages,                                                     \
          Context->BestGraphInfo[30].Coverage.NumberOfUsedPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_NumberOfUsedLargePages,                                                \
          Context->BestGraphInfo[30].Coverage.NumberOfUsedLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_NumberOfUsedCacheLines,                                                \
          Context->BestGraphInfo[30].Coverage.NumberOfUsedCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_NumberOfEmptyPages,                                                    \
          Context->BestGraphInfo[30].Coverage.NumberOfEmptyPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_NumberOfEmptyLargePages,                                               \
          Context->BestGraphInfo[30].Coverage.NumberOfEmptyLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_NumberOfEmptyCacheLines,                                               \
          Context->BestGraphInfo[30].Coverage.NumberOfEmptyCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_FirstPageUsed,                                                         \
          Context->BestGraphInfo[30].Coverage.FirstPageUsed,                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_FirstLargePageUsed,                                                    \
          Context->BestGraphInfo[30].Coverage.FirstLargePageUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_FirstCacheLineUsed,                                                    \
          Context->BestGraphInfo[30].Coverage.FirstCacheLineUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_LastPageUsed,                                                          \
          Context->BestGraphInfo[30].Coverage.LastPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_LastLargePageUsed,                                                     \
          Context->BestGraphInfo[30].Coverage.LastLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_LastCacheLineUsed,                                                     \
          Context->BestGraphInfo[30].Coverage.LastCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_TotalNumberOfAssigned,                                                 \
          Context->BestGraphInfo[30].Coverage.TotalNumberOfAssigned,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_CountOfCacheLinesWithNumberOfAssigned_0,                               \
          Context->BestGraphInfo[30].Coverage.NumberOfAssignedPerCacheLineCounts[0],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_CountOfCacheLinesWithNumberOfAssigned_1,                               \
          Context->BestGraphInfo[30].Coverage.NumberOfAssignedPerCacheLineCounts[1],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_CountOfCacheLinesWithNumberOfAssigned_2,                               \
          Context->BestGraphInfo[30].Coverage.NumberOfAssignedPerCacheLineCounts[2],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_CountOfCacheLinesWithNumberOfAssigned_3,                               \
          Context->BestGraphInfo[30].Coverage.NumberOfAssignedPerCacheLineCounts[3],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_CountOfCacheLinesWithNumberOfAssigned_4,                               \
          Context->BestGraphInfo[30].Coverage.NumberOfAssignedPerCacheLineCounts[4],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_CountOfCacheLinesWithNumberOfAssigned_5,                               \
          Context->BestGraphInfo[30].Coverage.NumberOfAssignedPerCacheLineCounts[5],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_CountOfCacheLinesWithNumberOfAssigned_6,                               \
          Context->BestGraphInfo[30].Coverage.NumberOfAssignedPerCacheLineCounts[6],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_CountOfCacheLinesWithNumberOfAssigned_7,                               \
          Context->BestGraphInfo[30].Coverage.NumberOfAssignedPerCacheLineCounts[7],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_CountOfCacheLinesWithNumberOfAssigned_8,                               \
          Context->BestGraphInfo[30].Coverage.NumberOfAssignedPerCacheLineCounts[8],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_CountOfCacheLinesWithNumberOfAssigned_9,                               \
          Context->BestGraphInfo[30].Coverage.NumberOfAssignedPerCacheLineCounts[9],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_CountOfCacheLinesWithNumberOfAssigned_10,                              \
          Context->BestGraphInfo[30].Coverage.NumberOfAssignedPerCacheLineCounts[10],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_CountOfCacheLinesWithNumberOfAssigned_11,                              \
          Context->BestGraphInfo[30].Coverage.NumberOfAssignedPerCacheLineCounts[11],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_CountOfCacheLinesWithNumberOfAssigned_12,                              \
          Context->BestGraphInfo[30].Coverage.NumberOfAssignedPerCacheLineCounts[12],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_CountOfCacheLinesWithNumberOfAssigned_13,                              \
          Context->BestGraphInfo[30].Coverage.NumberOfAssignedPerCacheLineCounts[13],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_CountOfCacheLinesWithNumberOfAssigned_14,                              \
          Context->BestGraphInfo[30].Coverage.NumberOfAssignedPerCacheLineCounts[14],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_CountOfCacheLinesWithNumberOfAssigned_15,                              \
          Context->BestGraphInfo[30].Coverage.NumberOfAssignedPerCacheLineCounts[15],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_CountOfCacheLinesWithNumberOfAssigned_16,                              \
          Context->BestGraphInfo[30].Coverage.NumberOfAssignedPerCacheLineCounts[16],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_TotalNumberOfPages,                                                    \
          Context->BestGraphInfo[31].Coverage.TotalNumberOfPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_TotalNumberOfLargePages,                                               \
          Context->BestGraphInfo[31].Coverage.TotalNumberOfLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_TotalNumberOfCacheLines,                                               \
          Context->BestGraphInfo[31].Coverage.TotalNumberOfCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_NumberOfUsedPages,                                                     \
          Context->BestGraphInfo[31].Coverage.NumberOfUsedPages,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_NumberOfUsedLargePages,                                                \
          Context->BestGraphInfo[31].Coverage.NumberOfUsedLargePages,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_NumberOfUsedCacheLines,                                                \
          Context->BestGraphInfo[31].Coverage.NumberOfUsedCacheLines,                        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_NumberOfEmptyPages,                                                    \
          Context->BestGraphInfo[31].Coverage.NumberOfEmptyPages,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_NumberOfEmptyLargePages,                                               \
          Context->BestGraphInfo[31].Coverage.NumberOfEmptyLargePages,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_NumberOfEmptyCacheLines,                                               \
          Context->BestGraphInfo[31].Coverage.NumberOfEmptyCacheLines,                       \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_FirstPageUsed,                                                         \
          Context->BestGraphInfo[31].Coverage.FirstPageUsed,                                 \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_FirstLargePageUsed,                                                    \
          Context->BestGraphInfo[31].Coverage.FirstLargePageUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_FirstCacheLineUsed,                                                    \
          Context->BestGraphInfo[31].Coverage.FirstCacheLineUsed,                            \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_LastPageUsed,                                                          \
          Context->BestGraphInfo[31].Coverage.LastPageUsed,                                  \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_LastLargePageUsed,                                                     \
          Context->BestGraphInfo[31].Coverage.LastLargePageUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_LastCacheLineUsed,                                                     \
          Context->BestGraphInfo[31].Coverage.LastCacheLineUsed,                             \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_TotalNumberOfAssigned,                                                 \
          Context->BestGraphInfo[31].Coverage.TotalNumberOfAssigned,                         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_CountOfCacheLinesWithNumberOfAssigned_0,                               \
          Context->BestGraphInfo[31].Coverage.NumberOfAssignedPerCacheLineCounts[0],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_CountOfCacheLinesWithNumberOfAssigned_1,                               \
          Context->BestGraphInfo[31].Coverage.NumberOfAssignedPerCacheLineCounts[1],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_CountOfCacheLinesWithNumberOfAssigned_2,                               \
          Context->BestGraphInfo[31].Coverage.NumberOfAssignedPerCacheLineCounts[2],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_CountOfCacheLinesWithNumberOfAssigned_3,                               \
          Context->BestGraphInfo[31].Coverage.NumberOfAssignedPerCacheLineCounts[3],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_CountOfCacheLinesWithNumberOfAssigned_4,                               \
          Context->BestGraphInfo[31].Coverage.NumberOfAssignedPerCacheLineCounts[4],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_CountOfCacheLinesWithNumberOfAssigned_5,                               \
          Context->BestGraphInfo[31].Coverage.NumberOfAssignedPerCacheLineCounts[5],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_CountOfCacheLinesWithNumberOfAssigned_6,                               \
          Context->BestGraphInfo[31].Coverage.NumberOfAssignedPerCacheLineCounts[6],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_CountOfCacheLinesWithNumberOfAssigned_7,                               \
          Context->BestGraphInfo[31].Coverage.NumberOfAssignedPerCacheLineCounts[7],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_CountOfCacheLinesWithNumberOfAssigned_8,                               \
          Context->BestGraphInfo[31].Coverage.NumberOfAssignedPerCacheLineCounts[8],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_CountOfCacheLinesWithNumberOfAssigned_9,                               \
          Context->BestGraphInfo[31].Coverage.NumberOfAssignedPerCacheLineCounts[9],         \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_CountOfCacheLinesWithNumberOfAssigned_10,                              \
          Context->BestGraphInfo[31].Coverage.NumberOfAssignedPerCacheLineCounts[10],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_CountOfCacheLinesWithNumberOfAssigned_11,                              \
          Context->BestGraphInfo[31].Coverage.NumberOfAssignedPerCacheLineCounts[11],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_CountOfCacheLinesWithNumberOfAssigned_12,                              \
          Context->BestGraphInfo[31].Coverage.NumberOfAssignedPerCacheLineCounts[12],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_CountOfCacheLinesWithNumberOfAssigned_13,                              \
          Context->BestGraphInfo[31].Coverage.NumberOfAssignedPerCacheLineCounts[13],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_CountOfCacheLinesWithNumberOfAssigned_14,                              \
          Context->BestGraphInfo[31].Coverage.NumberOfAssignedPerCacheLineCounts[14],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_CountOfCacheLinesWithNumberOfAssigned_15,                              \
          Context->BestGraphInfo[31].Coverage.NumberOfAssignedPerCacheLineCounts[15],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_CountOfCacheLinesWithNumberOfAssigned_16,                              \
          Context->BestGraphInfo[31].Coverage.NumberOfAssignedPerCacheLineCounts[16],        \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_Seed1,                                                                  \
          Context->BestGraphInfo[0].Seeds[0],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_Seed2,                                                                  \
          Context->BestGraphInfo[0].Seeds[1],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_Seed3,                                                                  \
          Context->BestGraphInfo[0].Seeds[2],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_Seed4,                                                                  \
          Context->BestGraphInfo[0].Seeds[3],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_Seed5,                                                                  \
          Context->BestGraphInfo[0].Seeds[4],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_Seed6,                                                                  \
          Context->BestGraphInfo[0].Seeds[5],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_Seed7,                                                                  \
          Context->BestGraphInfo[0].Seeds[6],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph1_Seed8,                                                                  \
          Context->BestGraphInfo[0].Seeds[7],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_Seed1,                                                                  \
          Context->BestGraphInfo[1].Seeds[0],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_Seed2,                                                                  \
          Context->BestGraphInfo[1].Seeds[1],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_Seed3,                                                                  \
          Context->BestGraphInfo[1].Seeds[2],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_Seed4,                                                                  \
          Context->BestGraphInfo[1].Seeds[3],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_Seed5,                                                                  \
          Context->BestGraphInfo[1].Seeds[4],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_Seed6,                                                                  \
          Context->BestGraphInfo[1].Seeds[5],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_Seed7,                                                                  \
          Context->BestGraphInfo[1].Seeds[6],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph2_Seed8,                                                                  \
          Context->BestGraphInfo[1].Seeds[7],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_Seed1,                                                                  \
          Context->BestGraphInfo[2].Seeds[0],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_Seed2,                                                                  \
          Context->BestGraphInfo[2].Seeds[1],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_Seed3,                                                                  \
          Context->BestGraphInfo[2].Seeds[2],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_Seed4,                                                                  \
          Context->BestGraphInfo[2].Seeds[3],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_Seed5,                                                                  \
          Context->BestGraphInfo[2].Seeds[4],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_Seed6,                                                                  \
          Context->BestGraphInfo[2].Seeds[5],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_Seed7,                                                                  \
          Context->BestGraphInfo[2].Seeds[6],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph3_Seed8,                                                                  \
          Context->BestGraphInfo[2].Seeds[7],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_Seed1,                                                                  \
          Context->BestGraphInfo[3].Seeds[0],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_Seed2,                                                                  \
          Context->BestGraphInfo[3].Seeds[1],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_Seed3,                                                                  \
          Context->BestGraphInfo[3].Seeds[2],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_Seed4,                                                                  \
          Context->BestGraphInfo[3].Seeds[3],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_Seed5,                                                                  \
          Context->BestGraphInfo[3].Seeds[4],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_Seed6,                                                                  \
          Context->BestGraphInfo[3].Seeds[5],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_Seed7,                                                                  \
          Context->BestGraphInfo[3].Seeds[6],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph4_Seed8,                                                                  \
          Context->BestGraphInfo[3].Seeds[7],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_Seed1,                                                                  \
          Context->BestGraphInfo[4].Seeds[0],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_Seed2,                                                                  \
          Context->BestGraphInfo[4].Seeds[1],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_Seed3,                                                                  \
          Context->BestGraphInfo[4].Seeds[2],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_Seed4,                                                                  \
          Context->BestGraphInfo[4].Seeds[3],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_Seed5,                                                                  \
          Context->BestGraphInfo[4].Seeds[4],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_Seed6,                                                                  \
          Context->BestGraphInfo[4].Seeds[5],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_Seed7,                                                                  \
          Context->BestGraphInfo[4].Seeds[6],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph5_Seed8,                                                                  \
          Context->BestGraphInfo[4].Seeds[7],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_Seed1,                                                                  \
          Context->BestGraphInfo[5].Seeds[0],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_Seed2,                                                                  \
          Context->BestGraphInfo[5].Seeds[1],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_Seed3,                                                                  \
          Context->BestGraphInfo[5].Seeds[2],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_Seed4,                                                                  \
          Context->BestGraphInfo[5].Seeds[3],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_Seed5,                                                                  \
          Context->BestGraphInfo[5].Seeds[4],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_Seed6,                                                                  \
          Context->BestGraphInfo[5].Seeds[5],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_Seed7,                                                                  \
          Context->BestGraphInfo[5].Seeds[6],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph6_Seed8,                                                                  \
          Context->BestGraphInfo[5].Seeds[7],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_Seed1,                                                                  \
          Context->BestGraphInfo[6].Seeds[0],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_Seed2,                                                                  \
          Context->BestGraphInfo[6].Seeds[1],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_Seed3,                                                                  \
          Context->BestGraphInfo[6].Seeds[2],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_Seed4,                                                                  \
          Context->BestGraphInfo[6].Seeds[3],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_Seed5,                                                                  \
          Context->BestGraphInfo[6].Seeds[4],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_Seed6,                                                                  \
          Context->BestGraphInfo[6].Seeds[5],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_Seed7,                                                                  \
          Context->BestGraphInfo[6].Seeds[6],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph7_Seed8,                                                                  \
          Context->BestGraphInfo[6].Seeds[7],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_Seed1,                                                                  \
          Context->BestGraphInfo[7].Seeds[0],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_Seed2,                                                                  \
          Context->BestGraphInfo[7].Seeds[1],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_Seed3,                                                                  \
          Context->BestGraphInfo[7].Seeds[2],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_Seed4,                                                                  \
          Context->BestGraphInfo[7].Seeds[3],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_Seed5,                                                                  \
          Context->BestGraphInfo[7].Seeds[4],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_Seed6,                                                                  \
          Context->BestGraphInfo[7].Seeds[5],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_Seed7,                                                                  \
          Context->BestGraphInfo[7].Seeds[6],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph8_Seed8,                                                                  \
          Context->BestGraphInfo[7].Seeds[7],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_Seed1,                                                                  \
          Context->BestGraphInfo[8].Seeds[0],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_Seed2,                                                                  \
          Context->BestGraphInfo[8].Seeds[1],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_Seed3,                                                                  \
          Context->BestGraphInfo[8].Seeds[2],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_Seed4,                                                                  \
          Context->BestGraphInfo[8].Seeds[3],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_Seed5,                                                                  \
          Context->BestGraphInfo[8].Seeds[4],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_Seed6,                                                                  \
          Context->BestGraphInfo[8].Seeds[5],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_Seed7,                                                                  \
          Context->BestGraphInfo[8].Seeds[6],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph9_Seed8,                                                                  \
          Context->BestGraphInfo[8].Seeds[7],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_Seed1,                                                                 \
          Context->BestGraphInfo[9].Seeds[0],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_Seed2,                                                                 \
          Context->BestGraphInfo[9].Seeds[1],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_Seed3,                                                                 \
          Context->BestGraphInfo[9].Seeds[2],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_Seed4,                                                                 \
          Context->BestGraphInfo[9].Seeds[3],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_Seed5,                                                                 \
          Context->BestGraphInfo[9].Seeds[4],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_Seed6,                                                                 \
          Context->BestGraphInfo[9].Seeds[5],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_Seed7,                                                                 \
          Context->BestGraphInfo[9].Seeds[6],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph10_Seed8,                                                                 \
          Context->BestGraphInfo[9].Seeds[7],                                                \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_Seed1,                                                                 \
          Context->BestGraphInfo[10].Seeds[0],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_Seed2,                                                                 \
          Context->BestGraphInfo[10].Seeds[1],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_Seed3,                                                                 \
          Context->BestGraphInfo[10].Seeds[2],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_Seed4,                                                                 \
          Context->BestGraphInfo[10].Seeds[3],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_Seed5,                                                                 \
          Context->BestGraphInfo[10].Seeds[4],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_Seed6,                                                                 \
          Context->BestGraphInfo[10].Seeds[5],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_Seed7,                                                                 \
          Context->BestGraphInfo[10].Seeds[6],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph11_Seed8,                                                                 \
          Context->BestGraphInfo[10].Seeds[7],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_Seed1,                                                                 \
          Context->BestGraphInfo[11].Seeds[0],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_Seed2,                                                                 \
          Context->BestGraphInfo[11].Seeds[1],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_Seed3,                                                                 \
          Context->BestGraphInfo[11].Seeds[2],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_Seed4,                                                                 \
          Context->BestGraphInfo[11].Seeds[3],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_Seed5,                                                                 \
          Context->BestGraphInfo[11].Seeds[4],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_Seed6,                                                                 \
          Context->BestGraphInfo[11].Seeds[5],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_Seed7,                                                                 \
          Context->BestGraphInfo[11].Seeds[6],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph12_Seed8,                                                                 \
          Context->BestGraphInfo[11].Seeds[7],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_Seed1,                                                                 \
          Context->BestGraphInfo[12].Seeds[0],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_Seed2,                                                                 \
          Context->BestGraphInfo[12].Seeds[1],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_Seed3,                                                                 \
          Context->BestGraphInfo[12].Seeds[2],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_Seed4,                                                                 \
          Context->BestGraphInfo[12].Seeds[3],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_Seed5,                                                                 \
          Context->BestGraphInfo[12].Seeds[4],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_Seed6,                                                                 \
          Context->BestGraphInfo[12].Seeds[5],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_Seed7,                                                                 \
          Context->BestGraphInfo[12].Seeds[6],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph13_Seed8,                                                                 \
          Context->BestGraphInfo[12].Seeds[7],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_Seed1,                                                                 \
          Context->BestGraphInfo[13].Seeds[0],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_Seed2,                                                                 \
          Context->BestGraphInfo[13].Seeds[1],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_Seed3,                                                                 \
          Context->BestGraphInfo[13].Seeds[2],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_Seed4,                                                                 \
          Context->BestGraphInfo[13].Seeds[3],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_Seed5,                                                                 \
          Context->BestGraphInfo[13].Seeds[4],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_Seed6,                                                                 \
          Context->BestGraphInfo[13].Seeds[5],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_Seed7,                                                                 \
          Context->BestGraphInfo[13].Seeds[6],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph14_Seed8,                                                                 \
          Context->BestGraphInfo[13].Seeds[7],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_Seed1,                                                                 \
          Context->BestGraphInfo[14].Seeds[0],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_Seed2,                                                                 \
          Context->BestGraphInfo[14].Seeds[1],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_Seed3,                                                                 \
          Context->BestGraphInfo[14].Seeds[2],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_Seed4,                                                                 \
          Context->BestGraphInfo[14].Seeds[3],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_Seed5,                                                                 \
          Context->BestGraphInfo[14].Seeds[4],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_Seed6,                                                                 \
          Context->BestGraphInfo[14].Seeds[5],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_Seed7,                                                                 \
          Context->BestGraphInfo[14].Seeds[6],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph15_Seed8,                                                                 \
          Context->BestGraphInfo[14].Seeds[7],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_Seed1,                                                                 \
          Context->BestGraphInfo[15].Seeds[0],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_Seed2,                                                                 \
          Context->BestGraphInfo[15].Seeds[1],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_Seed3,                                                                 \
          Context->BestGraphInfo[15].Seeds[2],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_Seed4,                                                                 \
          Context->BestGraphInfo[15].Seeds[3],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_Seed5,                                                                 \
          Context->BestGraphInfo[15].Seeds[4],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_Seed6,                                                                 \
          Context->BestGraphInfo[15].Seeds[5],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_Seed7,                                                                 \
          Context->BestGraphInfo[15].Seeds[6],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph16_Seed8,                                                                 \
          Context->BestGraphInfo[15].Seeds[7],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_Seed1,                                                                 \
          Context->BestGraphInfo[16].Seeds[0],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_Seed2,                                                                 \
          Context->BestGraphInfo[16].Seeds[1],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_Seed3,                                                                 \
          Context->BestGraphInfo[16].Seeds[2],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_Seed4,                                                                 \
          Context->BestGraphInfo[16].Seeds[3],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_Seed5,                                                                 \
          Context->BestGraphInfo[16].Seeds[4],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_Seed6,                                                                 \
          Context->BestGraphInfo[16].Seeds[5],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_Seed7,                                                                 \
          Context->BestGraphInfo[16].Seeds[6],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph17_Seed8,                                                                 \
          Context->BestGraphInfo[16].Seeds[7],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_Seed1,                                                                 \
          Context->BestGraphInfo[17].Seeds[0],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_Seed2,                                                                 \
          Context->BestGraphInfo[17].Seeds[1],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_Seed3,                                                                 \
          Context->BestGraphInfo[17].Seeds[2],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_Seed4,                                                                 \
          Context->BestGraphInfo[17].Seeds[3],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_Seed5,                                                                 \
          Context->BestGraphInfo[17].Seeds[4],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_Seed6,                                                                 \
          Context->BestGraphInfo[17].Seeds[5],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_Seed7,                                                                 \
          Context->BestGraphInfo[17].Seeds[6],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph18_Seed8,                                                                 \
          Context->BestGraphInfo[17].Seeds[7],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_Seed1,                                                                 \
          Context->BestGraphInfo[18].Seeds[0],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_Seed2,                                                                 \
          Context->BestGraphInfo[18].Seeds[1],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_Seed3,                                                                 \
          Context->BestGraphInfo[18].Seeds[2],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_Seed4,                                                                 \
          Context->BestGraphInfo[18].Seeds[3],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_Seed5,                                                                 \
          Context->BestGraphInfo[18].Seeds[4],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_Seed6,                                                                 \
          Context->BestGraphInfo[18].Seeds[5],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_Seed7,                                                                 \
          Context->BestGraphInfo[18].Seeds[6],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph19_Seed8,                                                                 \
          Context->BestGraphInfo[18].Seeds[7],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_Seed1,                                                                 \
          Context->BestGraphInfo[19].Seeds[0],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_Seed2,                                                                 \
          Context->BestGraphInfo[19].Seeds[1],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_Seed3,                                                                 \
          Context->BestGraphInfo[19].Seeds[2],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_Seed4,                                                                 \
          Context->BestGraphInfo[19].Seeds[3],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_Seed5,                                                                 \
          Context->BestGraphInfo[19].Seeds[4],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_Seed6,                                                                 \
          Context->BestGraphInfo[19].Seeds[5],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_Seed7,                                                                 \
          Context->BestGraphInfo[19].Seeds[6],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph20_Seed8,                                                                 \
          Context->BestGraphInfo[19].Seeds[7],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_Seed1,                                                                 \
          Context->BestGraphInfo[20].Seeds[0],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_Seed2,                                                                 \
          Context->BestGraphInfo[20].Seeds[1],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_Seed3,                                                                 \
          Context->BestGraphInfo[20].Seeds[2],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_Seed4,                                                                 \
          Context->BestGraphInfo[20].Seeds[3],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_Seed5,                                                                 \
          Context->BestGraphInfo[20].Seeds[4],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_Seed6,                                                                 \
          Context->BestGraphInfo[20].Seeds[5],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_Seed7,                                                                 \
          Context->BestGraphInfo[20].Seeds[6],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph21_Seed8,                                                                 \
          Context->BestGraphInfo[20].Seeds[7],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_Seed1,                                                                 \
          Context->BestGraphInfo[21].Seeds[0],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_Seed2,                                                                 \
          Context->BestGraphInfo[21].Seeds[1],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_Seed3,                                                                 \
          Context->BestGraphInfo[21].Seeds[2],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_Seed4,                                                                 \
          Context->BestGraphInfo[21].Seeds[3],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_Seed5,                                                                 \
          Context->BestGraphInfo[21].Seeds[4],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_Seed6,                                                                 \
          Context->BestGraphInfo[21].Seeds[5],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_Seed7,                                                                 \
          Context->BestGraphInfo[21].Seeds[6],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph22_Seed8,                                                                 \
          Context->BestGraphInfo[21].Seeds[7],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_Seed1,                                                                 \
          Context->BestGraphInfo[22].Seeds[0],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_Seed2,                                                                 \
          Context->BestGraphInfo[22].Seeds[1],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_Seed3,                                                                 \
          Context->BestGraphInfo[22].Seeds[2],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_Seed4,                                                                 \
          Context->BestGraphInfo[22].Seeds[3],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_Seed5,                                                                 \
          Context->BestGraphInfo[22].Seeds[4],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_Seed6,                                                                 \
          Context->BestGraphInfo[22].Seeds[5],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_Seed7,                                                                 \
          Context->BestGraphInfo[22].Seeds[6],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph23_Seed8,                                                                 \
          Context->BestGraphInfo[22].Seeds[7],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_Seed1,                                                                 \
          Context->BestGraphInfo[23].Seeds[0],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_Seed2,                                                                 \
          Context->BestGraphInfo[23].Seeds[1],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_Seed3,                                                                 \
          Context->BestGraphInfo[23].Seeds[2],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_Seed4,                                                                 \
          Context->BestGraphInfo[23].Seeds[3],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_Seed5,                                                                 \
          Context->BestGraphInfo[23].Seeds[4],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_Seed6,                                                                 \
          Context->BestGraphInfo[23].Seeds[5],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_Seed7,                                                                 \
          Context->BestGraphInfo[23].Seeds[6],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph24_Seed8,                                                                 \
          Context->BestGraphInfo[23].Seeds[7],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_Seed1,                                                                 \
          Context->BestGraphInfo[24].Seeds[0],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_Seed2,                                                                 \
          Context->BestGraphInfo[24].Seeds[1],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_Seed3,                                                                 \
          Context->BestGraphInfo[24].Seeds[2],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_Seed4,                                                                 \
          Context->BestGraphInfo[24].Seeds[3],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_Seed5,                                                                 \
          Context->BestGraphInfo[24].Seeds[4],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_Seed6,                                                                 \
          Context->BestGraphInfo[24].Seeds[5],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_Seed7,                                                                 \
          Context->BestGraphInfo[24].Seeds[6],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph25_Seed8,                                                                 \
          Context->BestGraphInfo[24].Seeds[7],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_Seed1,                                                                 \
          Context->BestGraphInfo[25].Seeds[0],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_Seed2,                                                                 \
          Context->BestGraphInfo[25].Seeds[1],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_Seed3,                                                                 \
          Context->BestGraphInfo[25].Seeds[2],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_Seed4,                                                                 \
          Context->BestGraphInfo[25].Seeds[3],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_Seed5,                                                                 \
          Context->BestGraphInfo[25].Seeds[4],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_Seed6,                                                                 \
          Context->BestGraphInfo[25].Seeds[5],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_Seed7,                                                                 \
          Context->BestGraphInfo[25].Seeds[6],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph26_Seed8,                                                                 \
          Context->BestGraphInfo[25].Seeds[7],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_Seed1,                                                                 \
          Context->BestGraphInfo[26].Seeds[0],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_Seed2,                                                                 \
          Context->BestGraphInfo[26].Seeds[1],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_Seed3,                                                                 \
          Context->BestGraphInfo[26].Seeds[2],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_Seed4,                                                                 \
          Context->BestGraphInfo[26].Seeds[3],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_Seed5,                                                                 \
          Context->BestGraphInfo[26].Seeds[4],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_Seed6,                                                                 \
          Context->BestGraphInfo[26].Seeds[5],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_Seed7,                                                                 \
          Context->BestGraphInfo[26].Seeds[6],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph27_Seed8,                                                                 \
          Context->BestGraphInfo[26].Seeds[7],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_Seed1,                                                                 \
          Context->BestGraphInfo[27].Seeds[0],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_Seed2,                                                                 \
          Context->BestGraphInfo[27].Seeds[1],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_Seed3,                                                                 \
          Context->BestGraphInfo[27].Seeds[2],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_Seed4,                                                                 \
          Context->BestGraphInfo[27].Seeds[3],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_Seed5,                                                                 \
          Context->BestGraphInfo[27].Seeds[4],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_Seed6,                                                                 \
          Context->BestGraphInfo[27].Seeds[5],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_Seed7,                                                                 \
          Context->BestGraphInfo[27].Seeds[6],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph28_Seed8,                                                                 \
          Context->BestGraphInfo[27].Seeds[7],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_Seed1,                                                                 \
          Context->BestGraphInfo[28].Seeds[0],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_Seed2,                                                                 \
          Context->BestGraphInfo[28].Seeds[1],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_Seed3,                                                                 \
          Context->BestGraphInfo[28].Seeds[2],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_Seed4,                                                                 \
          Context->BestGraphInfo[28].Seeds[3],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_Seed5,                                                                 \
          Context->BestGraphInfo[28].Seeds[4],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_Seed6,                                                                 \
          Context->BestGraphInfo[28].Seeds[5],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_Seed7,                                                                 \
          Context->BestGraphInfo[28].Seeds[6],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph29_Seed8,                                                                 \
          Context->BestGraphInfo[28].Seeds[7],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_Seed1,                                                                 \
          Context->BestGraphInfo[29].Seeds[0],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_Seed2,                                                                 \
          Context->BestGraphInfo[29].Seeds[1],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_Seed3,                                                                 \
          Context->BestGraphInfo[29].Seeds[2],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_Seed4,                                                                 \
          Context->BestGraphInfo[29].Seeds[3],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_Seed5,                                                                 \
          Context->BestGraphInfo[29].Seeds[4],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_Seed6,                                                                 \
          Context->BestGraphInfo[29].Seeds[5],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_Seed7,                                                                 \
          Context->BestGraphInfo[29].Seeds[6],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph30_Seed8,                                                                 \
          Context->BestGraphInfo[29].Seeds[7],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_Seed1,                                                                 \
          Context->BestGraphInfo[30].Seeds[0],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_Seed2,                                                                 \
          Context->BestGraphInfo[30].Seeds[1],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_Seed3,                                                                 \
          Context->BestGraphInfo[30].Seeds[2],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_Seed4,                                                                 \
          Context->BestGraphInfo[30].Seeds[3],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_Seed5,                                                                 \
          Context->BestGraphInfo[30].Seeds[4],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_Seed6,                                                                 \
          Context->BestGraphInfo[30].Seeds[5],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_Seed7,                                                                 \
          Context->BestGraphInfo[30].Seeds[6],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph31_Seed8,                                                                 \
          Context->BestGraphInfo[30].Seeds[7],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_Seed1,                                                                 \
          Context->BestGraphInfo[31].Seeds[0],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_Seed2,                                                                 \
          Context->BestGraphInfo[31].Seeds[1],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_Seed3,                                                                 \
          Context->BestGraphInfo[31].Seeds[2],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_Seed4,                                                                 \
          Context->BestGraphInfo[31].Seeds[3],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_Seed5,                                                                 \
          Context->BestGraphInfo[31].Seeds[4],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_Seed6,                                                                 \
          Context->BestGraphInfo[31].Seeds[5],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_Seed7,                                                                 \
          Context->BestGraphInfo[31].Seeds[6],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(BestGraph32_Seed8,                                                                 \
          Context->BestGraphInfo[31].Seeds[7],                                               \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(KeysMinValue,                                                                      \
          Keys->Stats.MinValue,                                                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(KeysMaxValue,                                                                      \
          Keys->Stats.MaxValue,                                                              \
          OUTPUT_INT)                                                                        \
                                                                                             \
    ENTRY(KeysFullPath,                                                                      \
          &Keys->File->Path->FullPath,                                                       \
          OUTPUT_UNICODE_STRING_FAST)                                                        \
                                                                                             \
    LAST_ENTRY(KeysBitmapString,                                                             \
               Keys->Stats.KeysBitmap.String,                                                \
               OUTPUT_RAW)

//
// Define a macro for initializing the local variables prior to writing a row.
//

#define BULK_CREATE_BEST_CSV_PRE_ROW()                                        \
    PCHAR Base;                                                               \
    PCHAR Output;                                                             \
                                                                              \
    Base = (PCHAR)CsvFile->BaseAddress;                                       \
    Output = RtlOffsetToPointer(Base, CsvFile->NumberOfBytesWritten.QuadPart)

//
// And one for post-row writing.
//

#define BULK_CREATE_BEST_CSV_POST_ROW() \
    CsvFile->NumberOfBytesWritten.QuadPart = RtlPointerToOffset(Base, Output)

#define EXPAND_AS_WRITE_BULK_CREATE_BEST_ROW_NOT_LAST_COLUMN(Name,        \
                                                             Value,       \
                                                             OutputMacro) \
    OutputMacro(Value);                                                   \
    OUTPUT_CHR(',');

#define EXPAND_AS_WRITE_BULK_CREATE_BEST_ROW_LAST_COLUMN(Name,   \
                                                    Value,       \
                                                    OutputMacro) \
    OutputMacro(Value);                                          \
    OUTPUT_CHR('\n');


#define WRITE_BULK_CREATE_BEST_CSV_ROW() do {                 \
    BULK_CREATE_BEST_CSV_PRE_ROW();                           \
    BULK_CREATE_BEST_CSV_ROW_TABLE(                           \
        EXPAND_AS_WRITE_BULK_CREATE_BEST_ROW_NOT_LAST_COLUMN, \
        EXPAND_AS_WRITE_BULK_CREATE_BEST_ROW_NOT_LAST_COLUMN, \
        EXPAND_AS_WRITE_BULK_CREATE_BEST_ROW_LAST_COLUMN      \
    );                                                        \
    BULK_CREATE_BEST_CSV_POST_ROW();                          \
} while (0)

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
