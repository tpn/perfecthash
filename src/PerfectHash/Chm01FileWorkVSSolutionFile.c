/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWorkVSSolutionFile.c

Abstract:

    This module implements the prepare and save file work callback routines
    for a compiled perfect hash table's Visual Studio solution file (.sln),
    which is responsible for encapuslating all .vcxproj files generated (e.g.
    Dll.vcxproj, TestExe.vcxproj etc).

--*/

#include "stdafx.h"

#define VS_SOLUTION_FILE_FORMAT_VERSION "12.0"
#define VS_VERSION "15.0.27004.2009"
#define MIN_VS_VERSION "10.0.40219.1"
#define VCPP_GUID "8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942"

#ifndef RCS
#define RCS RTL_CONSTANT_STRING
#endif

const STRING Configurations[] = {
    RCS("Release|x64"),
    RCS("Debug|x64"),
    RCS("PGInstrument|x64"),
    RCS("PGUpdate|x64"),
    RCS("PGOptimize|x64"),
    RCS("Release|Win32"),
    RCS("Debug|Win32"),
    RCS("PGInstrument|Win32"),
    RCS("PGUpdate|Win32"),
    RCS("PGOptimize|Win32"),
#if 0
    RCS("Release|ARM"),
    RCS("Debug|ARM"),
    RCS("PGInstrument|ARM"),
    RCS("PGUpdate|ARM"),
    RCS("PGOptimize|ARM"),
    RCS("Release|ARM64"),
    RCS("Debug|ARM64"),
    RCS("PGInstrument|ARM64"),
    RCS("PGUpdate|ARM64"),
    RCS("PGOptimize|ARM64"),
#endif
};

const BYTE NumberOfConfigurations = ARRAYSIZE(Configurations);

_Use_decl_annotations_
HRESULT
PrepareVSSolutionFileChm01(
    PPERFECT_HASH_CONTEXT Context,
    PFILE_WORK_ITEM Item
    )
{
    PRTL Rtl;
    PCHAR Base;
    PCHAR Output;
    ULONG WaitResult;
    BYTE Index = 0;
    BYTE ConfigIndex;
    PCSTRING Config;
    PCSTRING BaseName;
    PCSTRING TableName;
    PCSTRING ProjectGuid;
    PCSTRING SolutionGuid;
    PCSTRING DllProjectGuid;
    PCSTRING TestExeProjectGuid;
    HRESULT Result = S_OK;
    PPERFECT_HASH_PATH Path;
    PPERFECT_HASH_FILE File;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_PATH ProjectPath;
    PPERFECT_HASH_PATH DllProjectPath;
    PPERFECT_HASH_PATH TestExeProjectPath;
    const BOOL WaitForAllEvents = TRUE;
    const BYTE NumberOfEvents = NUMBER_OF_VCPROJECT_FILES;
    const BYTE NumberOfVCProjects = NUMBER_OF_VCPROJECT_FILES;
    PPERFECT_HASH_FILE VCProjects[NUMBER_OF_VCPROJECT_FILES];
    PPERFECT_HASH_FILE ProjectFile;
    PPERFECT_HASH_FILE DllProjectFile;
    PPERFECT_HASH_FILE TestExeProjectFile;
    HANDLE PrepareEvents[NUMBER_OF_VCPROJECT_FILES];
    PHANDLE PrepareEvent = PrepareEvents;

    //
    // Fill out the array of handles for prepare events.
    //

#define EXPAND_AS_ASSIGN_EVENT(Verb, VUpper, Name, Upper) \
    *Verb##Event++ = Context->Verb##d##Name##Event;

    PREPARE_VCPROJECT_FILE_WORK_TABLE_ENTRY(EXPAND_AS_ASSIGN_EVENT);

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Table = Context->Table;
    File = *Item->FilePointer;
    Path = GetActivePath(File);
    BaseName = &Path->BaseNameA;
    TableName = &Path->TableNameA;
    SolutionGuid = &File->Uuid;

    ASSERT(IsValidUuidString(SolutionGuid));

    Base = (PCHAR)File->BaseAddress;
    Output = Base;

    //
    // Wait on all VC Project file prepare events to be signaled before we
    // attempt to reference the underlying file instance and UUID string.
    //

    ASSERT(ARRAYSIZE(PrepareEvents) == (SIZE_T)NumberOfEvents);

    WaitResult = WaitForMultipleObjects(NumberOfEvents,
                                        PrepareEvents,
                                        WaitForAllEvents,
                                        INFINITE);

    if (WaitResult != WAIT_OBJECT_0) {
        SYS_ERROR(WaitForMultipleObjects);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // If all the prepare work routines were successful, all the corresponding
    // file instance pointers should be non-NULL and have valid UUID strings.
    //

#define EXPAND_AS_ASSIGN_FILE_POINTER(Verb, VUpper, Name, Upper) \
    if (!Table->Name) {                                          \
        goto Error;                                              \
    }                                                            \
    if (!IsValidUuidString(&Table->Name->Uuid)) {                \
        goto Error;                                              \
    }                                                            \
    VCProjects[Index++] = Table->Name;

    VCPROJECT_FILE_WORK_TABLE_ENTRY(EXPAND_AS_ASSIGN_FILE_POINTER);

    //
    // The Dll project should always come first, and the TestExe project second.
    // Verify this now.
    //

    DllProjectFile = VCProjects[0];
    ASSERT(DllProjectFile->FileId == FileVCProjectDllFileId);

    DllProjectPath = GetActivePath(DllProjectFile);
    DllProjectGuid = &DllProjectFile->Uuid;

    TestExeProjectFile = VCProjects[1];
    ASSERT(TestExeProjectFile->FileId == FileVCProjectTestExeFileId);

    TestExeProjectPath = GetActivePath(TestExeProjectFile);
    TestExeProjectGuid = &TestExeProjectFile->Uuid;

    //
    // Switch the pointers around so that TestExe comes first and the Dll
    // project comes second.  This will result in the TestExe becoming the
    // default startup project.
    //

    VCProjects[0] = TestExeProjectFile;
    VCProjects[1] = DllProjectFile;

    //
    // Write the solution header.
    //

    OUTPUT_RAW("Microsoft Visual Studio Solution File, "
               "Format Version " VS_SOLUTION_FILE_FORMAT_VERSION "\r\n"
               "# Visual Studio 15\r\n"
               "VisualStudioVersion = " VS_VERSION "\r\n"
               "MinimumVisualStudioVersion = " MIN_VS_VERSION "\r\n");

    //
    // If we're in index-only mode, the only projects we write are the Dll and
    // BenchmarkIndex.  (The others require TableValues[] and Insert, Lookup,
    // and Delete routines.)  Define a helper macro to detect this condition.
    //

#define MAYBE_SKIP_PROJECT_FILE()                                          \
    if (IsIndexOnly(Table)) {                                              \
        if (ProjectFile->FileId != FileVCProjectDllFileId &&               \
            ProjectFile->FileId != FileVCProjectBenchmarkIndexExeFileId) { \
            continue;                                                      \
        }                                                                  \
    }

    //
    // Write the project definitions.
    //

    for (Index = 0; Index < NumberOfVCProjects; Index++) {

        ProjectFile = VCProjects[Index];
        ProjectGuid = &ProjectFile->Uuid;
        ProjectPath = GetActivePath(ProjectFile);

        //
        // Sanity check the index/id ordering invariant.
        //

        if (Index == 0) {
            ASSERT(ProjectFile->FileId == FileVCProjectTestExeFileId);
        } else if (Index == 1) {
            ASSERT(ProjectFile->FileId == FileVCProjectDllFileId);
        }

        MAYBE_SKIP_PROJECT_FILE();

        OUTPUT_RAW("Project(\"{" VCPP_GUID "}\") = \"");
        OUTPUT_STRING(&ProjectPath->BaseNameA);
        OUTPUT_RAW("\", \"");
        OUTPUT_STRING(&ProjectPath->BaseNameA);
        OUTPUT_RAW(".vcxproj\", \"{");
        OUTPUT_STRING(ProjectGuid);
        OUTPUT_RAW("}\"\r\n");

        if (ProjectFile->FileId != FileVCProjectDllFileId) {

            //
            // All other projects have a dependency on the Dll project.  That
            // is, they require the .dll's .lib import library to be available
            // at link time.  We add a project dependency section to tell VS
            // to effectively build the Dll project first before kicking off
            // builds for other components.
            //

            OUTPUT_RAW("\tProjectSection(ProjectDependencies) = "
                       "postProject\r\n\t\t{");
            OUTPUT_STRING(DllProjectGuid);
            OUTPUT_RAW("} = {");
            OUTPUT_STRING(DllProjectGuid);
            OUTPUT_RAW("}\r\n\tEndProjectSection\r\n");

        }

        OUTPUT_RAW("EndProject\r\n");

    }

    OUTPUT_RAW("Global\r\n");

    //
    // Write the solution configurations.
    //

    OUTPUT_RAW("\tGlobalSection(SolutionConfigurationPlatforms) "
               "= preSolution\r\n");

#define FOR_EACH_CONFIGURATION                     \
    for (ConfigIndex = 0, Config = Configurations; \
         ConfigIndex < NumberOfConfigurations;     \
         ConfigIndex++, Config++)

    FOR_EACH_CONFIGURATION {

        OUTPUT_RAW("\t\t");
        OUTPUT_STRING(Config);
        OUTPUT_RAW(" = ");
        OUTPUT_STRING(Config);
        OUTPUT_RAW("\r\n");

    }

    OUTPUT_RAW(
        "\tEndGlobalSection\r\n"
        "\tGlobalSection(ProjectConfigurationPlatforms) = postSolution\r\n"
    );

    //
    // Write the project configurations.
    //

    for (Index = 0; Index < NumberOfVCProjects; Index++) {

        ProjectFile = VCProjects[Index];
        ProjectGuid = &ProjectFile->Uuid;

        MAYBE_SKIP_PROJECT_FILE();

        FOR_EACH_CONFIGURATION {

            OUTPUT_RAW("\t\t{");
            OUTPUT_STRING(ProjectGuid);
            OUTPUT_RAW("}.");
            OUTPUT_STRING(Config);
            OUTPUT_RAW(".ActiveCfg = ");
            OUTPUT_STRING(Config);
            OUTPUT_RAW("\r\n");

            OUTPUT_RAW("\t\t{");
            OUTPUT_STRING(ProjectGuid);
            OUTPUT_RAW("}.");
            OUTPUT_STRING(Config);
            OUTPUT_RAW(".Build.0 = ");
            OUTPUT_STRING(Config);
            OUTPUT_RAW("\r\n");

        }
    }

    //
    // Write the final solution sections.
    //

    OUTPUT_RAW(
        "\tEndGlobalSection\r\n"
        "\tGlobalSection(SolutionProperties) = preSolution\r\n"
            "\t\tHideSolutionNode = FALSE\r\n"
        "\tEndGlobalSection\r\n"
        "\tGlobalSection(ExtensibilityGlobals) = postSolution\r\n"
            "\t\tSolutionGuid = {"
    );
    OUTPUT_STRING(SolutionGuid);
    OUTPUT_RAW("}\r\n\tEndGlobalSection\r\nEndGlobal\r\n");

    //
    // We're done; capture the number of bytes written and finish up.
    //

    File->NumberOfBytesWritten.QuadPart = RtlPointerToOffset(Base, Output);

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
