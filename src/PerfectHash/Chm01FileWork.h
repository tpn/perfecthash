/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWork.h

Abstract:

    This is the private header file for the CHM v1 algorithm implementation's
    file work functionality.

--*/

#include "stdafx.h"

#define EXPAND_AS_CALLBACK_DECL(Verb, VUpper, Name, Upper) \
    extern FILE_WORK_CALLBACK_IMPL Verb####Name##Chm01;

PREPARE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_CALLBACK_DECL);
SAVE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_CALLBACK_DECL);

#define PrepareTableFileChm01 NULL
#define PrepareTableInfoStreamChm01 NULL
//#define PrepareCHeaderFileChm01 NULL
//#define PrepareCSourceFileChm01 NULL
//#define PrepareCSourceKeysFileChm01 NULL
#define PrepareCSourceTableDataFileChm01 NULL
#define PrepareVCProjectDllFileChm01 NULL
#define PrepareCSourceSupportFileChm01 NULL
#define PrepareCSourceTestFileChm01 NULL
#define PrepareCSourceTestExeFileChm01 NULL
#define PrepareVCProjectTestExeFileChm01 NULL
#define PrepareCSourceBenchmarkFullFileChm01 NULL
#define PrepareCSourceBenchmarkFullExeFileChm01 NULL
#define PrepareVCProjectBenchmarkFullExeFileChm01 NULL
#define PrepareCSourceBenchmarkIndexFileChm01 NULL
#define PrepareCSourceBenchmarkIndexExeFileChm01 NULL
#define PrepareVCProjectBenchmarkIndexExeFileChm01 NULL
#define PrepareCHeaderCompiledPerfectHashFileChm01 NULL
#define PrepareVCPropsCompiledPerfectHashFileChm01 NULL
#define PrepareTableStatsTextFileChm01 NULL

//#define SaveTableFileChm01 NULL
//#define SaveTableInfoStreamChm01 NULL
//#define SaveCHeaderFileChm01 NULL
#define SaveCSourceFileChm01 NULL
#define SaveCSourceKeysFileChm01 NULL
//#define SaveCSourceTableDataFileChm01 NULL
#define SaveVCProjectDllFileChm01 NULL
#define SaveCSourceSupportFileChm01 NULL
#define SaveCSourceTestFileChm01 NULL
#define SaveCSourceTestExeFileChm01 NULL
#define SaveVCProjectTestExeFileChm01 NULL
#define SaveCSourceBenchmarkFullFileChm01 NULL
#define SaveCSourceBenchmarkFullExeFileChm01 NULL
#define SaveVCProjectBenchmarkFullExeFileChm01 NULL
#define SaveCSourceBenchmarkIndexFileChm01 NULL
#define SaveCSourceBenchmarkIndexExeFileChm01 NULL
#define SaveVCProjectBenchmarkIndexExeFileChm01 NULL
#define SaveCHeaderCompiledPerfectHashFileChm01 NULL
#define SaveVCPropsCompiledPerfectHashFileChm01 NULL
#define SaveTableStatsTextFileChm01 NULL

extern FILE_WORK_CALLBACK_IMPL *FileCallbacks[];

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
