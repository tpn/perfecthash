/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    TableCreateCsv.h

Abstract:

    Private header file for table creation CSV glue.

--*/

//
// Define an "X-Macro"-style macro for capturing the ordered definition of
// columns in a row of bulk create .csv output.
//
// The ENTRY macros receive (Name, Value, OutputMacro) as their parameters.
//

#define TABLE_CREATE_CSV_ROW_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY)     \
    FIRST_ENTRY(KeysName,                                              \
                &Keys->File->Path->BaseNameA,                          \
                OUTPUT_STRING)                                         \
                                                                       \
    ENTRY(NumberOfKeys,                                                \
          Keys->NumberOfElements.QuadPart,                             \
          OUTPUT_INT)                                                  \
                                                                       \
    ENTRY(NumberOfEdges,                                               \
          Table->IndexSize,                                            \
          OUTPUT_INT)                                                  \
                                                                       \
    ENTRY(NumberOfVertices,                                            \
          Table->HashSize,                                             \
          OUTPUT_INT)                                                  \
                                                                       \
    ENTRY(Algorithm,                                                   \
          AlgorithmNames[Context->AlgorithmId],                        \
          OUTPUT_UNICODE_STRING_FAST)                                  \
                                                                       \
    ENTRY(HashFunction,                                                \
          HashFunctionNames[Context->HashFunctionId],                  \
          OUTPUT_UNICODE_STRING_FAST)                                  \
                                                                       \
    ENTRY(MaskFunction,                                                \
          MaskFunctionNames[Context->MaskFunctionId],                  \
          OUTPUT_UNICODE_STRING_FAST)                                  \
                                                                       \
    ENTRY(SolutionFound,                                               \
          (TableCreateResult == S_OK ? 'Y' : 'N'),                     \
          OUTPUT_CHR)                                                  \
                                                                       \
    ENTRY(LowMemory,                                                   \
          (TableCreateResult == PH_I_LOW_MEMORY ? 'Y' : 'N'),          \
          OUTPUT_CHR)                                                  \
                                                                       \
    ENTRY(OutOfMemory,                                                 \
          (TableCreateResult == PH_I_OUT_OF_MEMORY ? 'Y' : 'N'),       \
          OUTPUT_CHR)                                                  \
                                                                       \
    ENTRY(OtherMemoryIssue,                                            \
          (TableCreateResult ==                                        \
           PH_I_FAILED_TO_ALLOCATE_MEMORY_FOR_ALL_GRAPHS ? 'Y' : 'N'), \
          OUTPUT_CHR)                                                  \
                                                                       \
    ENTRY(TableCreateResult,                                           \
          TableCreateResult,                                           \
          OUTPUT_ERROR_CODE_STRING)                                    \
                                                                       \
    ENTRY(NumberOfSolutionsFound,                                      \
          Context->FinishedCount,                                      \
          OUTPUT_INT)                                                  \
                                                                       \
    ENTRY(Attempts,                                                    \
          Context->Attempts,                                           \
          OUTPUT_INT)                                                  \
                                                                       \
    ENTRY(FailedAttempts,                                              \
          Context->FailedAttempts,                                     \
          OUTPUT_INT)                                                  \
                                                                       \
    ENTRY(NumberOfTableResizeEvents,                                   \
          Context->NumberOfTableResizeEvents,                          \
          OUTPUT_INT)                                                  \
                                                                       \
    ENTRY(HighestDeletedEdgesCount,                                    \
          Context->HighestDeletedEdgesCount,                           \
          OUTPUT_INT)                                                  \
                                                                       \
    ENTRY(ClosestWeCameToSolvingGraphWithSmallerTableSizes,            \
          Context->ClosestWeCameToSolvingGraphWithSmallerTableSizes,   \
          OUTPUT_INT)                                                  \
                                                                       \
    ENTRY(SolveMicroseconds,                                           \
          Context->SolveElapsedMicroseconds.QuadPart,                  \
          OUTPUT_INT)                                                  \
                                                                       \
    ENTRY(VerifyMicroseconds,                                          \
          Context->VerifyElapsedMicroseconds.QuadPart,                 \
          OUTPUT_INT)                                                  \
                                                                       \
    ENTRY(KeysMinValue,                                                \
          Keys->Stats.MinValue,                                        \
          OUTPUT_INT)                                                  \
                                                                       \
    ENTRY(KeysMaxValue,                                                \
          Keys->Stats.MaxValue,                                        \
          OUTPUT_INT)                                                  \
                                                                       \
    LAST_ENTRY(KeysBitmapString,                                       \
               Keys->Stats.KeysBitmap.String,                          \
               OUTPUT_RAW)


//
// Define a macro for initializing the local variables prior to writing a row.
//

#define TABLE_CREATE_CSV_PRE_ROW()                                            \
    PCHAR Base;                                                               \
    PCHAR Output;                                                             \
                                                                              \
    Base = (PCHAR)CsvFile->BaseAddress;                                       \
    Output = RtlOffsetToPointer(Base, CsvFile->NumberOfBytesWritten.QuadPart)

//
// And one for post-row writing.
//

#define TABLE_CREATE_CSV_POST_ROW() \
    CsvFile->NumberOfBytesWritten.QuadPart = RtlPointerToOffset(Base, Output)

#define EXPAND_AS_WRITE_TABLE_CREATE_ROW_NOT_LAST_COLUMN(Name,        \
                                                         Value,       \
                                                         OutputMacro) \
    OutputMacro(Value);                                               \
    OUTPUT_CHR(',');

#define EXPAND_AS_WRITE_TABLE_CREATE_ROW_LAST_COLUMN(Name,        \
                                                     Value,       \
                                                     OutputMacro) \
    OutputMacro(Value);                                           \
    OUTPUT_CHR('\n');

#define WRITE_TABLE_CREATE_CSV_ROW() do {                 \
    TABLE_CREATE_CSV_PRE_ROW();                           \
    TABLE_CREATE_CSV_ROW_TABLE(                           \
        EXPAND_AS_WRITE_TABLE_CREATE_ROW_NOT_LAST_COLUMN, \
        EXPAND_AS_WRITE_TABLE_CREATE_ROW_NOT_LAST_COLUMN, \
        EXPAND_AS_WRITE_TABLE_CREATE_ROW_LAST_COLUMN      \
    );                                                    \
    TABLE_CREATE_CSV_POST_ROW();                          \
} while (0)

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
