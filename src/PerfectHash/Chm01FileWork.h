/*++

Copyright (c) 2018-2023 Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWork.h

Abstract:

    This is the private header file for the CHM v1 algorithm implementation's
    file work functionality.

--*/

#pragma once

#include "stdafx.h"

#define EXPAND_AS_CALLBACK_DECL(                      \
    Verb, VUpper, Name, Upper,                        \
    EofType, EofValue,                                \
    Suffix, Extension, Stream, Base                   \
)                                                     \
    extern FILE_WORK_CALLBACK_IMPL Verb##Name##Chm01;

PREPARE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_CALLBACK_DECL);
SAVE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_CALLBACK_DECL);

//
// Add defines for files that don't have a prepare callback.  Corresponds to
// NO_PREPARE_CALLBACK for the 'Prepare' X-macro parameter in the file work
// macro.  Absense of a prepare callback indicates the file can not be written
// with any useful content until a solved graph is available.
//

#define PrepareTableFileChm01 NULL
#define PrepareTableInfoStreamChm01 NULL
#define PrepareCSourceTableDataFileChm01 NULL
#define PreparePythonFileChm01 NULL
#define PreparePythonTestFileChm01 NULL
#define PrepareCppHeaderOnlyTestFileChm01 NULL
#define PrepareRustCargoTomlFileChm01 NULL
#define PrepareRustLibFileChm01 NULL
#define PrepareRustTestFileChm01 NULL
#define PrepareRustBenchFileChm01 NULL
#define PrepareGraphVertexPairsFileChm01 NULL
#define PrepareGraphVertices3FileChm01 NULL

//
// Add defines for files that don't have a save callback.  Corresponds to
// NO_SAVE_CALLBACK for the 'Save' X-macro parameter in the file work macro.
// Absense of a save callback means that the file does not need to do any more
// processing once the solve graph is available (i.e. it wrote everything it
// needed to in the prepare stage).
//

#define SaveCSourceFileChm01 NULL
#define SaveCHeaderTypesFileChm01 NULL
#define SaveCHeaderStdAfxFileChm01 NULL
#define SaveCSourceStdAfxFileChm01 NULL
#define SaveCSourceKeysFileChm01 NULL
#define SaveCSourceDownsizedKeysFileChm01 NULL
#define SaveCSourceTableValuesFileChm01 NULL
#define SaveVCProjectDllFileChm01 NULL
#define SaveCHeaderSupportFileChm01 NULL
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
#define SaveVSSolutionFileChm01 NULL
#define SaveCHeaderCompiledPerfectHashFileChm01 NULL
#define SaveCHeaderCompiledPerfectHashMacroGlueFileChm01 NULL
#define SaveVCPropsCompiledPerfectHashFileChm01 NULL
#define SaveBatchBuildSolutionFileChm01 NULL
#define SaveMakefileFileChm01 NULL
#define SaveMakefileMainMkFileChm01 NULL
#define SaveMakefileLibMkFileChm01 NULL
#define SaveMakefileSoMkFileChm01 NULL
#define SaveMakefileTestMkFileChm01 NULL
#define SaveMakefileBenchmarkIndexMkFileChm01 NULL
#define SaveMakefileBenchmarkFullMkFileChm01 NULL
#define SaveCHeaderNoSal2FileChm01 NULL
#define SaveModuleDefFileChm01 NULL
#define SaveCMakeListsTextFileChm01 NULL

extern FILE_WORK_CALLBACK_IMPL *FileCallbacks[];

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
