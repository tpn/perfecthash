/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    TableCreateCsv.h

Abstract:

    Private header file for table creation CSV glue.  Shared by Chm01.c and
    PerfectHashContextBulkCreate.c.

--*/

//
// Define an "X-Macro"-style macro for capturing the ordered definition of
// columns in a row of bulk create .csv output.
//
// The ENTRY macros receive (Name, Value, OutputMacro) as their parameters.
//

#define BULK_CREATE_CSV_ROW_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY)    \
    FIRST_ENTRY(KeysName,                                            \
                &Table->Keys->File->Path->BaseNameA,                 \
                OUTPUT_STRING)                                       \
                                                                     \
    ENTRY(NumberOfKeys,                                              \
          Table->Keys->NumberOfElements.QuadPart,                    \
          OUTPUT_INT)                                                \
                                                                     \
    ENTRY(NumberOfEdges,                                             \
          Table->IndexSize,                                          \
          OUTPUT_INT)                                                \
                                                                     \
    ENTRY(NumberOfVertices,                                          \
          Table->HashSize,                                           \
          OUTPUT_INT)                                                \
                                                                     \
    ENTRY(Algorithm,                                                 \
          AlgorithmNames[Context->AlgorithmId],                      \
          OUTPUT_UNICODE_STRING_FAST)                                \
                                                                     \
    ENTRY(HashFunction,                                              \
          HashFunctionNames[Context->HashFunctionId],                \
          OUTPUT_UNICODE_STRING_FAST)                                \
                                                                     \
    ENTRY(MaskFunction,                                              \
          MaskFunctionNames[Context->MaskFunctionId],                \
          OUTPUT_UNICODE_STRING_FAST)                                \
                                                                     \
    ENTRY(SolutionFound,                                             \
          (Result == S_OK ? 'Y' : 'N'),                              \
          OUTPUT_CHR)                                                \
                                                                     \
    ENTRY(LowMemory,                                                 \
          (Result == PH_I_LOW_MEMORY ? 'Y' : 'N'),                   \
          OUTPUT_CHR)                                                \
                                                                     \
    ENTRY(OutOfMemory,                                               \
          (Result == PH_I_OUT_OF_MEMORY ? 'Y' : 'N'),                \
          OUTPUT_CHR)                                                \
                                                                     \
    ENTRY(OtherMemoryIssue,                                          \
          (Result == PH_I_FAILED_TO_ALLOCATE_MEMORY_FOR_ALL_GRAPHS   \
            ? 'Y' : 'N'),                                            \
          OUTPUT_CHR)                                                \
                                                                     \
    ENTRY(TableCreateResult, Result, OUTPUT_HEX_RAW_0x)              \
                                                                     \
    ENTRY(NumberOfSolutionsFound,                                    \
          Context->FinishedCount,                                    \
          OUTPUT_INT)                                                \
                                                                     \
    ENTRY(Attempts,                                                  \
          Context->Attempts,                                         \
          OUTPUT_INT)                                                \
                                                                     \
    ENTRY(FailedAttempts,                                            \
          Context->FailedAttempts,                                   \
          OUTPUT_INT)                                                \
                                                                     \
    ENTRY(NumberOfTableResizeEvents,                                 \
          Context->NumberOfTableResizeEvents,                        \
          OUTPUT_INT)                                                \
                                                                     \
    ENTRY(HighestDeletedEdgesCount,                                  \
          Context->HighestDeletedEdgesCount,                         \
          OUTPUT_INT)                                                \
                                                                     \
    ENTRY(ClosestWeCameToSolvingGraphWithSmallerTableSizes,          \
          Context->ClosestWeCameToSolvingGraphWithSmallerTableSizes, \
          OUTPUT_INT)                                                \
                                                                     \
    ENTRY(SolveMicroseconds,                                         \
          Context->SolveElapsedMicroseconds.QuadPart,                \
          OUTPUT_INT)                                                \
                                                                     \
    ENTRY(VerifyMicroseconds,                                        \
          Context->VerifyElapsedMicroseconds.QuadPart,               \
          OUTPUT_INT)                                                \
                                                                     \
    ENTRY(KeysMinValue,                                              \
          Table->Keys->Stats.MinValue,                               \
          OUTPUT_INT)                                                \
                                                                     \
    ENTRY(KeysMaxValue,                                              \
          Table->Keys->Stats.MaxValue,                               \
          OUTPUT_INT)                                                \
                                                                     \
    LAST_ENTRY(KeysBitmapString,                                     \
               Table->Keys->Stats.KeysBitmap.String,                 \
               OUTPUT_RAW)


//
// Define a macro for initializing the Base/Output local variables prior to
// writing a row.
//

#define BULK_CREATE_CSV_PRE_ROW()                                              \
        PCHAR Base;                                                            \
        PCHAR Output;                                                          \
        PPERFECT_HASH_FILE File;                                               \
                                                                               \
        File = Context->BulkCreateCsvFile;                                     \
        Base = (PCHAR)File->BaseAddress;                                       \
        Output = RtlOffsetToPointer(Base, File->NumberOfBytesWritten.QuadPart)

//
// And one for post-row writing.
//

#define BULK_CREATE_CSV_POST_ROW()                                             \
        File->NumberOfBytesWritten.QuadPart = RtlPointerToOffset(Base, Output)

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
