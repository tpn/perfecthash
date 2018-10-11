/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWorkVCProjectTestExeFile.c

Abstract:

    This module implements the prepare and save file work callback routines for
    a compiled perfect hash table's TestExe.vcxproj file.

--*/

#include "stdafx.h"

#include <rpc.h>

extern const ULONG VCProjectTestExeFileNumberOfChunks;
extern CHUNK VCProjectTestExeFileChunks[];
#define UUID_STRING_LENGTH 36

_Use_decl_annotations_
HRESULT
PrepareVCProjectTestExeFileChm01(
    PPERFECT_HASH_CONTEXT Context,
    PFILE_WORK_ITEM Item
    )
{
    PRTL Rtl;
    PCHAR Base;
    PCHAR Output;
    GUID Guid;
    RPC_CSTR GuidCStr = NULL;
    STRING GuidString;
    PCSTRING BaseName;
    PCSTRING TableName;
    CHUNK_VALUES Values;
    HRESULT Result = S_OK;
    PPERFECT_HASH_PATH Path;
    PPERFECT_HASH_FILE File;
    PPERFECT_HASH_TABLE Table;
    PCCHUNK Chunks = VCProjectTestExeFileChunks;
    const ULONG NumberOfChunks = VCProjectTestExeFileNumberOfChunks;
    STRING TargetPrefix = RTL_CONSTANT_STRING("Test_");
    STRING TestExeFileSuffix = RTL_CONSTANT_STRING("TestExe");

    //
    // Create a new UUID and then convert it into a string representation.  We
    // will use this for the ProjectGuid field.
    //

    Result = UuidCreate(&Guid);
    if (FAILED(Result)) {
        SYS_ERROR(UuidCreate);
        goto Error;
    }

    Result = UuidToStringA(&Guid, &GuidCStr);
    if (FAILED(Result)) {
        SYS_ERROR(UuidCreate);
        goto Error;
    }

    GuidString.Buffer = (PCHAR)GuidCStr;
    GuidString.Length = (USHORT)strlen(GuidString.Buffer);
    GuidString.MaximumLength = GuidString.Length + 1;
    ASSERT(GuidString.Length == UUID_STRING_LENGTH);
    ASSERT(GuidString.Buffer[GuidString.Length] == '\0');

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Table = Context->Table;
    File = *Item->FilePointer;
    Path = GetActivePath(File);
    BaseName = &Path->BaseNameA;
    TableName = &Path->TableNameA;

    Base = (PCHAR)File->BaseAddress;
    Output = Base;

    //
    // Fill out the chunk values structure.
    //

    ZeroStruct(Values);

    Values.ProjectGuid = &GuidString;
    Values.RootNamespace = TableName;
    Values.ProjectName = BaseName;
    Values.BaseName = TableName;
    Values.TableName = TableName;
    Values.TargetName = BaseName;
    Values.TargetExt = &DotExeSuffixA;
    Values.FileSuffix = &TestExeFileSuffix;
    Values.TargetPrefix = &TargetPrefix;
    Values.ConfigurationType = &ApplicationConfigurationTypeA;

    Result = ProcessChunks(Rtl, Chunks, NumberOfChunks, &Values, &Output);

    if (FAILED(Result)) {
        PH_ERROR(ProcessChunks, Result);
        goto Error;
    }

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

    if (GuidCStr) {
        Result = RpcStringFreeA(&GuidCStr);
        if (FAILED(Result)) {
            SYS_ERROR(RpcStringFreeA);
        }
    }

    return Result;
}

#ifndef RCS
#define RCS RTL_CONSTANT_STRING
#endif

CHUNK VCProjectTestExeFileChunks[] = {

    { ChunkOpStringPointer, .StringPointer = &VCProjectFileHeaderChunk, },

    { ChunkOpInsertProjectGuid, },

    { ChunkOpRaw, RCS("}</ProjectGuid>\r\n    <RootNamespace>") },

    { ChunkOpInsertRootNamespace, },

    { ChunkOpRaw, RCS("</RootNamespace>\r\n    <ProjectName>") },

    { ChunkOpInsertProjectName, },

    {
        ChunkOpRaw,
        RCS(
            "</ProjectName>\r\n"
            "    <IntermediateOutputPath>$(Platform)\\$(Configuration)\\"
        ),
    },

    { ChunkOpInsertFileSuffix, },

    {
        ChunkOpRaw,
        RCS(
            "\\</IntermediateOutputPath>\r\n"
            "  </PropertyGroup>\r\n"
            "  <PropertyGroup>\r\n"
            "    <TargetName>"
        ),
    },

    { ChunkOpInsertTargetPrefix, },

    { ChunkOpInsertTableName, },

    {
        ChunkOpRaw,
        RCS(
                "</TargetName>\r\n"
            "    <TargetExt>"
        ),
    },

    { ChunkOpInsertTargetExt, },

    {
        ChunkOpRaw,
        RCS(
                "</TargetExt>\r\n"
            "  </PropertyGroup>\r\n"
            "  <Import Project=\"$(VCTargetsPath)\\Microsoft.Cpp.Default.props\" />\r\n"
            "  <PropertyGroup Label=\"Configuration\">\r\n"
            "    <ConfigurationType>"
        ),
    },

    { ChunkOpInsertConfigurationType, },

    {
        ChunkOpRaw,
        RCS(
            "</ConfigurationType>\r\n"
            "  </PropertyGroup>\r\n"
            "  <Import Project=\"..\\CompiledPerfectHash.props\" />\r\n"
            "  <Import Project=\"$(VCTargetsPath)\\Microsoft.Cpp.props\" />\r\n"
            "  <ImportGroup Label=\"ExtensionSettings\">\r\n"
            "  </ImportGroup>\r\n"
            "  <ImportGroup Label=\"Shared\">\r\n"
            "  </ImportGroup>\r\n"
            "  <ImportGroup Label=\"PropertySheets\" Condition=\"'$(Platform)'=='Win32'\">\r\n"
            "    <Import Project=\"$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props\" Condition=\"exists('$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props')\" Label=\"LocalAppDataPlatform\" />\r\n"
            "  </ImportGroup>\r\n"
            "  <ImportGroup Label=\"PropertySheets\" Condition=\"'$(Platform)'=='x64'\">\r\n"
            "    <Import Project=\"$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props\" Condition=\"exists('$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props')\" Label=\"LocalAppDataPlatform\" />\r\n"
            "  </ImportGroup>\r\n"
            "  <PropertyGroup Label=\"UserMacros\" />\r\n"
            "  <ItemDefinitionGroup>\r\n"
            "    <ClCompile>\r\n"
            "      <AdditionalIncludeDirectories>$(ProjectDir)\\..</AdditionalIncludeDirectories>\r\n"
            "      <PrecompiledHeaderFile>"
        ),
    },

    { ChunkOpInsertTableName, },

    {
        ChunkOpRaw,
        RCS(
                   "_StdAfx.h</PrecompiledHeaderFile>\r\n"
            "    </ClCompile>\r\n"
            "  </ItemDefinitionGroup>\r\n"
            "  <ItemGroup>\r\n"
            "    <ClInclude Include=\""
        ),
    },

    { ChunkOpInsertTableName, },

    {
        ChunkOpRaw,
        RCS(
            "_StdAfx.h\" />\r\n"
            "    <ClInclude Include=\""
        ),
    },

    { ChunkOpInsertTableName, },

    {
        ChunkOpRaw,
        RCS(
            ".h\" />\r\n"
            "  </ItemGroup>\r\n"
            "  <ItemGroup>\r\n"
            "    <ClCompile Include=\""
        ),
    },

    { ChunkOpInsertTableName, },

    {
        ChunkOpRaw,
        RCS(
            "_StdAfx.c\">\r\n"
            "      <PrecompiledHeader>Create</PrecompiledHeader>\r\n"
            "    </ClCompile>\r\n"
            "    <ClCompile Include=\""
        ),
    },

    { ChunkOpInsertTableName, },

    {
        ChunkOpRaw,
        RCS(
            "_Test.c\" />\r\n"
            "    <ClCompile Include=\""
        ),
    },

    { ChunkOpInsertTableName, },

    {
        ChunkOpRaw,
        RCS(
            "_TestExe.c\" />\r\n"
            "    <ClCompile Include=\""
        ),
    },

    { ChunkOpInsertTableName, },

    {
        ChunkOpRaw,
        RCS(
            "_Keys.c\" />\r\n"
            "  </ItemGroup>\r\n"
            "  <Import Project=\"$(VCTargetsPath)\\Microsoft.Cpp.targets\" />\r\n"
            "</Project>\r\n"
        ),
    },
};

const ULONG VCProjectTestExeFileNumberOfChunks =
    ARRAYSIZE(VCProjectTestExeFileChunks);

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
