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
    RCS("Debug|x86"),
    RCS("Release|x86"),
    RCS("PGInstrument|x86"),
    RCS("PGUpdate|x86"),
    RCS("PGOptimize|x86"),
    RCS("Debug|x64"),
    RCS("Release|x64"),
    RCS("PGInstrument|x64"),
    RCS("PGUpdate|x64"),
    RCS("PGOptimize|x64"),
    RCS("Debug|ARM"),
    RCS("Release|ARM"),
    RCS("PGInstrument|ARM"),
    RCS("PGUpdate|ARM"),
    RCS("PGOptimize|ARM"),
    RCS("Debug|ARM64"),
    RCS("Release|ARM64"),
    RCS("PGInstrument|ARM64"),
    RCS("PGUpdate|ARM64"),
    RCS("PGOptimize|ARM64"),
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
    BYTE Index;
    BYTE ConfigIndex;
    PCSTRING Config;
    PCSTRING BaseName;
    PCSTRING TableName;
    PCSTRING ProjectGuid;
    PCSTRING SolutionGuid;
    HRESULT Result = S_OK;
    PPERFECT_HASH_PATH Path;
    PPERFECT_HASH_FILE File;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_PATH ProjectPath;
    const BOOL WaitForAllEvents = TRUE;
    const BYTE NumberOfEvents = NUMBER_OF_VCPROJECT_FILES;
    const BYTE NumberOfVCProjects = NUMBER_OF_VCPROJECT_FILES;
    PPERFECT_HASH_FILE VCProjects[NUMBER_OF_VCPROJECT_FILES];
    PPERFECT_HASH_FILE *VCProject = VCProjects;
    HANDLE PrepareEvents[NUMBER_OF_VCPROJECT_FILES];
    PHANDLE PrepareEvent = PrepareEvents;

    //
    // Fill out the array of handles for prepare events.
    //

#define EXPAND_AS_ASSIGN_EVENT(Verb, VUpper, Name, Upper) \
    *##Verb##Event++ = Context->##Verb##d##Name##Event;

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
    if (!Table->##Name) {                                        \
        goto Error;                                              \
    }                                                            \
    if (!IsValidUuidString(&Table->##Name##->Uuid)) {            \
        goto Error;                                              \
    }                                                            \
    *VCProject++ = Table->##Name##;

    VCPROJECT_FILE_WORK_TABLE_ENTRY(EXPAND_AS_ASSIGN_FILE_POINTER);

    //
    // Write the solution header.
    //

    OUTPUT_RAW("Microsoft Visual Studio Solution File, "
               "Format Version " VS_SOLUTION_FILE_FORMAT_VERSION "\r\n"
               "# Visual Studio 15\r\n"
               "VisualStudioVersion = " VS_VERSION "\r\n"
               "MinimumVisualStudioVersion = " MIN_VS_VERSION "\r\n");

    //
    // Write the project references.
    //

#define FOR_EACH_PROJECT                          \
    for (Index = 0,                               \
         VCProject = VCProjects,                  \
         ProjectGuid = &(*VCProject)->Uuid,       \
         ProjectPath = GetActivePath(*VCProject); \
                                                  \
         Index < NumberOfVCProjects;              \
                                                  \
         Index++,                                 \
         VCProject++,                             \
         ProjectGuid = &(*VCProject)->Uuid,       \
         ProjectPath = GetActivePath(*VCProject))


    FOR_EACH_PROJECT {

        OUTPUT_RAW("Project(\"{" VCPP_GUID "}\") = \"");
        OUTPUT_STRING(&ProjectPath->BaseNameA);
        OUTPUT_RAW("\", \"");
        OUTPUT_STRING(&ProjectPath->BaseNameA);
        OUTPUT_RAW(".vcxproj\", \"{");
        OUTPUT_STRING(ProjectGuid);
        OUTPUT_RAW("}\"\r\nEndProject\r\n");

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

    FOR_EACH_PROJECT {

        FOR_EACH_CONFIGURATION {

            OUTPUT_RAW("\t\t\t{");
            OUTPUT_STRING(ProjectGuid);
            OUTPUT_RAW("}.");
            OUTPUT_STRING(Config);
            OUTPUT_RAW(".ActiveCfg = ");
            OUTPUT_STRING(Config);
            OUTPUT_RAW("\r\n");

            OUTPUT_RAW("\t\t\t{");
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

//
// Define various string constants.
//

DECLSPEC_ALIGN(16)
const CHAR VSSolutionGlobalSectionCStr[] =
    "\\tGlobalSection(SolutionConfigurationPlatforms) = preSolution\r\n"
        "\t\tDebug|x86 = Debug|x86\r\n"
        "\t\tRelease|x86 = Release|x86\r\n"
        "\t\tPGInstrument|x86 = PGInstrument|x86\r\n"
        "\t\tPGUpdate|x86 = PGUpdate|x86\r\n"
        "\t\tPGOptimize|x86 = PGOptimize|x86\r\n"
        "\t\tDebug|x64 = Debug|x64\r\n"
        "\t\tRelease|x64 = Release|x64\r\n"
        "\t\tPGInstrument|x64 = PGInstrument|x64\r\n"
        "\t\tPGUpdate|x64 = PGUpdate|x64\r\n"
        "\t\tPGOptimize|x64 = PGOptimize|x64\r\n"
        "\t\tDebug|ARM = Debug|ARM\r\n"
        "\t\tRelease|ARM = Release|ARM\r\n"
        "\t\tPGInstrument|ARM = PGInstrument|ARM\r\n"
        "\t\tPGUpdate|ARM = PGUpdate|ARM\r\n"
        "\t\tPGOptimize|ARM = PGOptimize|ARM\r\n"
        "\t\tDebug|ARM64 = Debug|ARM64\r\n"
        "\t\tRelease|ARM64 = Release|ARM64\r\n"
        "\t\tPGInstrument|ARM64 = PGInstrument|ARM64\r\n"
        "\t\tPGUpdate|ARM64 = PGUpdate|ARM64\r\n"
        "\t\tPGOptimize|ARM64 = PGOptimize|ARM64\r\n"
    "\\tEndGlobalSection\r\n";

const STRING VSSolutionGlobalSection = {
    sizeof(VSSolutionGlobalSectionCStr) - sizeof(CHAR),
    sizeof(VSSolutionGlobalSectionCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&VSSolutionGlobalSectionCStr,
};

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
