/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashFileWork.h

Abstract:

    This is the private header file for the file work component of the perfect
    hash library.  It defines the FILE_WORK_ID enumeration and various X-macro
    helper macro definitions.

--*/

#pragma once

#include "stdafx.h"

//
// Define an "X-Macro"-style macro for capturing the ordered definition of file
// work items.
//

#define VERB_FILE_WORK_TABLE(Verb, VUpper, FIRST_ENTRY, ENTRY, LAST_ENTRY)                   \
    FIRST_ENTRY(Verb, VUpper, TableFile,                TABLE_FILE)                          \
    ENTRY(Verb, VUpper, TableInfoStream,                TABLE_INFO_STREAM)                   \
    ENTRY(Verb, VUpper, CHeaderFile,                    C_HEADER_FILE)                       \
    ENTRY(Verb, VUpper, CSourceFile,                    C_SOURCE_FILE)                       \
    ENTRY(Verb, VUpper, CHeaderStdAfxFile,              C_HEADER_STDAFX_FILE)                       \
    ENTRY(Verb, VUpper, CSourceStdAfxFile,              C_SOURCE_STDAFX_FILE)                \
    ENTRY(Verb, VUpper, CSourceKeysFile,                C_SOURCE_KEYS_FILE)                  \
    ENTRY(Verb, VUpper, CSourceTableDataFile,           C_SOURCE_TABLE_DATA_FILE)            \
    ENTRY(Verb, VUpper, VCProjectDllFile,               VCPROJECT_DLL_FILE)                  \
    ENTRY(Verb, VUpper, CSourceSupportFile,             C_SOURCE_SUPPORT_FILE)               \
    ENTRY(Verb, VUpper, CSourceTestFile,                C_SOURCE_TEST_FILE)                  \
    ENTRY(Verb, VUpper, CSourceTestExeFile,             C_SOURCE_TEST_EXE_FILE)              \
    ENTRY(Verb, VUpper, VCProjectTestExeFile,           VCPROJECT_TEST_EXE_FILE)             \
    ENTRY(Verb, VUpper, CSourceBenchmarkFullFile,       C_SOURCE_BENCHMARK_FULL_FILE)        \
    ENTRY(Verb, VUpper, CSourceBenchmarkFullExeFile,    C_SOURCE_BENCHMARK_FULL_EXE_FILE)    \
    ENTRY(Verb, VUpper, VCProjectBenchmarkFullExeFile,  VCPROJECT_BENCHMARK_FULL_EXE_FILE)   \
    ENTRY(Verb, VUpper, CSourceBenchmarkIndexFile,      C_SOURCE_BENCHMARK_INDEX_FILE)       \
    ENTRY(Verb, VUpper, CSourceBenchmarkIndexExeFile,   C_SOURCE_BENCHMARK_INDEX_EXE_FILE)   \
    ENTRY(Verb, VUpper, VCProjectBenchmarkIndexExeFile, VCPROJECT_BENCHMARK_INDEX_EXE_FILE)  \
    LAST_ENTRY(Verb, VUpper, TableStatsTextFile,        TABLE_STATS_TEXT_FILE)

#define PREPARE_FILE_WORK_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    VERB_FILE_WORK_TABLE(Prepare, PREPARE, FIRST_ENTRY, ENTRY, LAST_ENTRY)

#define SAVE_FILE_WORK_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    VERB_FILE_WORK_TABLE(Save, SAVE, FIRST_ENTRY, ENTRY, LAST_ENTRY)

#define PREPARE_FILE_WORK_TABLE_ENTRY(ENTRY) \
    VERB_FILE_WORK_TABLE(Prepare, PREPARE, ENTRY, ENTRY, ENTRY)

#define SAVE_FILE_WORK_TABLE_ENTRY(ENTRY) \
    VERB_FILE_WORK_TABLE(Save, SAVE, ENTRY, ENTRY, ENTRY)

#define FILE_WORK_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    VERB_FILE_WORK_TABLE(Nothing, NOTHING, FIRST_ENTRY, ENTRY, LAST_ENTRY)

#define FILE_WORK_TABLE_ENTRY(ENTRY) FILE_WORK_TABLE(ENTRY, ENTRY, ENTRY)

//
// Define an enum for individual file IDs.
//

typedef enum _FILE_ID {

    //
    // Null ID.
    //

    FileNullId = 0,

#define EXPAND_AS_ENUMS(Verb, VUpper, Name, Upper) File##Name##Id,

#define EXPAND_AS_FIRST_ENUM(Verb, VUpper, Name, Upper)   \
    File##Name##Id,                           \
    FileFirstId = File##Name##Id,

#define EXPAND_AS_LAST_ENUM(Verb, VUpper, Name, Upper)   \
    File##Name##Id,                          \
    FileLastId = File##Name##Id,

    FILE_WORK_TABLE(EXPAND_AS_FIRST_ENUM,
                    EXPAND_AS_ENUMS,
                    EXPAND_AS_LAST_ENUM)

#undef EXPAND_AS_ENUMS
#undef EXPAND_AS_FIRST_ENUM
#undef EXPAND_AS_LAST_ENUM

    //
    // Invalid ID, this must come last.
    //

    FileInvalidId,

} FILE_ID;
typedef FILE_ID *PFILE_ID;

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
    // Initial file preparation work.
    //

#define EXPAND_AS_ENUMS(Verb, VUpper, Name, Upper) FileWork##Verb####Name##Id,

#define EXPAND_AS_FIRST_ENUM(Verb, VUpper, Name, Upper)   \
    FileWork##Verb####Name##Id,                           \
    FileWork##Verb##FirstId = FileWork##Verb####Name##Id,

#define EXPAND_AS_LAST_ENUM(Verb, VUpper, Name, Upper)   \
    FileWork##Verb####Name##Id,                          \
    FileWork##Verb##LastId = FileWork##Verb####Name##Id,

    PREPARE_FILE_WORK_TABLE(EXPAND_AS_FIRST_ENUM,
                            EXPAND_AS_ENUMS,
                            EXPAND_AS_LAST_ENUM)

    SAVE_FILE_WORK_TABLE(EXPAND_AS_FIRST_ENUM,
                         EXPAND_AS_ENUMS,
                         EXPAND_AS_LAST_ENUM)

    //
    // Invalid ID, this must come last.
    //

    FileWorkInvalidId,

} FILE_WORK_ID;
typedef FILE_WORK_ID *PFILE_WORK_ID;

#define NUMBER_OF_PREPARE_FILE_EVENTS (                  \
    (FileWorkPrepareLastId - FileWorkPrepareFirstId) + 1 \
)

#define NUMBER_OF_SAVE_FILE_EVENTS (               \
    (FileWorkSaveLastId - FileWorkSaveFirstId) + 1 \
)

C_ASSERT(NUMBER_OF_PREPARE_FILE_EVENTS == NUMBER_OF_SAVE_FILE_EVENTS);

#define NUMBER_OF_FILES NUMBER_OF_PREPARE_FILE_EVENTS

#define TOTAL_NUMBER_OF_FILE_EVENTS ( \
    NUMBER_OF_PREPARE_FILE_EVENTS +   \
    NUMBER_OF_SAVE_FILE_EVENTS        \
)

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

FORCEINLINE
BOOLEAN
IsPrepareFileWorkId(
    _In_ FILE_WORK_ID FileWorkId
    )
{
    return (
        FileWorkId >= FileWorkPrepareFirstId &&
        FileWorkId <= FileWorkPrepareLastId
    );
}

FORCEINLINE
BOOLEAN
IsSaveFileWorkId(
    _In_ FILE_WORK_ID FileWorkId
    )
{
    return (
        FileWorkId >= FileWorkSaveFirstId &&
        FileWorkId <= FileWorkSaveLastId
    );
}

FORCEINLINE
ULONG
FileWorkIdToFileIndex(
    _In_ FILE_WORK_ID FileWorkId
    )
{
    LONG Index;

    Index = FileWorkId - 1;

    if (IsSaveFileWorkId(FileWorkId)) {
        Index -= (FileWorkSaveFirstId - 1);
    }

    ASSERT(Index >= 0);

    return Index;
}

FORCEINLINE
ULONG
FileWorkIdToEventIndex(
    _In_ FILE_WORK_ID FileWorkId
    )
{
    return FileWorkId - 1;
}

FORCEINLINE
FILE_WORK_ID
FileWorkIdToDependentId(
    _In_ FILE_WORK_ID FileWorkId
    )
{
    LONG Id;

    if (IsPrepareFileWorkId(FileWorkId)) {
        Id = FileWorkId + FileWorkSaveFirstId;
    } else {
        Id = FileWorkId - (FileWorkSaveFirstId - 1);
    }

    ASSERT(Id >= FileWorkPrepareFirstId && Id <= FileWorkSaveLastId);

    return Id;
}

FORCEINLINE
ULONG
FileWorkIdToDependentEventIndex(
    _In_ FILE_WORK_ID FileWorkId
    )
{
    return FileWorkIdToEventIndex(FileWorkIdToDependentId(FileWorkId));
}

typedef enum _EOF_INIT_TYPE {
    EofInitTypeNull = 0,
    EofInitTypeDefault,
    EofInitTypeZero,
    EofInitTypeAssignedSize,
    EofInitTypeFixed,
    EofInitTypeNumberOfPages,
    EofInitTypeNumberOfKeysMultiplier,
    EofInitTypeNumberOfTableElementsMultiplier,
    EofInitTypeInvalid
} EOF_INIT_TYPE;

FORCEINLINE
BOOLEAN
IsValidEofInitType(
    _In_ EOF_INIT_TYPE EofInitType
    )
{
    return EofInitType < EofInitTypeInvalid;
}

typedef struct _EOF_INIT {
    EOF_INIT_TYPE Type;
    union {
        ULONG FixedValue;
        ULONG Multiplier;
        ULONG NumberOfPages;
    };
} EOF_INIT;
typedef EOF_INIT *PEOF_INIT;
typedef const EOF_INIT *PCEOF_INIT;

//
// Define a file work item structure that will be pushed to the context's
// file work list head.
//

typedef struct _FILE_WORK_ITEM {

    //
    // Entry used to add this structure onto a guarded list.
    //

    LIST_ENTRY ListEntry;

    //
    // Type of work requested.
    //

    FILE_WORK_ID FileWorkId;

    volatile LONG NumberOfErrors;
    volatile LONG LastError;

    volatile HRESULT LastResult;

    struct _PERFECT_HASH_FILE **FilePointer;

} FILE_WORK_ITEM;
typedef FILE_WORK_ITEM *PFILE_WORK_ITEM;

//
// Define specific file work functions.
//

typedef
_Check_return_
_Success_(return >= 0)
HRESULT
(NTAPI FILE_WORK_CALLBACK_IMPL)(
    _In_ struct _PERFECT_HASH_CONTEXT *Context,
    _In_ PFILE_WORK_ITEM Item
    );
typedef FILE_WORK_CALLBACK_IMPL *PFILE_WORK_CALLBACK_IMPL;


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
