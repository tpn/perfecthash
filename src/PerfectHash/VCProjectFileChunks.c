/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    VCProjectFileChunks.c

Abstract:

    This module contains common chunk definitions used for .vcxproj files.

--*/

#include "stdafx.h"

DECLSPEC_ALIGN(16)
const CHAR VCProjectFileHeaderChunkCStr[] =
    "<?xml version=\"1.0\" encoding=\"utf-8\"?>\r\n"
    "<Project DefaultTargets=\"Build\" ToolsVersion=\"15.0\" xmlns=\"http://schemas.microsoft.com/developer/msbuild/2003\">\r\n"
    "  <ItemGroup Label=\"ProjectConfigurations\">\r\n"
    "    <ProjectConfiguration Include=\"Debug|Win32\">\r\n"
    "      <Configuration>Debug</Configuration>\r\n"
    "      <Platform>Win32</Platform>\r\n"
    "    </ProjectConfiguration>\r\n"
    "    <ProjectConfiguration Include=\"Release|Win32\">\r\n"
    "      <Configuration>Release</Configuration>\r\n"
    "      <Platform>Win32</Platform>\r\n"
    "    </ProjectConfiguration>\r\n"
    "    <ProjectConfiguration Include=\"PGInstrument|Win32\">\r\n"
    "      <Configuration>PGInstrument</Configuration>\r\n"
    "      <Platform>Win32</Platform>\r\n"
    "    </ProjectConfiguration>\r\n"
    "    <ProjectConfiguration Include=\"PGUpdate|Win32\">\r\n"
    "      <Configuration>PGUpdate</Configuration>\r\n"
    "      <Platform>Win32</Platform>\r\n"
    "    </ProjectConfiguration>\r\n"
    "    <ProjectConfiguration Include=\"PGOptimize|Win32\">\r\n"
    "      <Configuration>PGOptimize</Configuration>\r\n"
    "      <Platform>Win32</Platform>\r\n"
    "    </ProjectConfiguration>\r\n"
    "    <ProjectConfiguration Include=\"Debug|x64\">\r\n"
    "      <Configuration>Debug</Configuration>\r\n"
    "      <Platform>x64</Platform>\r\n"
    "    </ProjectConfiguration>\r\n"
    "    <ProjectConfiguration Include=\"Release|x64\">\r\n"
    "      <Configuration>Release</Configuration>\r\n"
    "      <Platform>x64</Platform>\r\n"
    "    </ProjectConfiguration>\r\n"
    "    <ProjectConfiguration Include=\"PGInstrument|x64\">\r\n"
    "      <Configuration>PGInstrument</Configuration>\r\n"
    "      <Platform>x64</Platform>\r\n"
    "    </ProjectConfiguration>\r\n"
    "    <ProjectConfiguration Include=\"PGUpdate|x64\">\r\n"
    "      <Configuration>PGUpdate</Configuration>\r\n"
    "      <Platform>x64</Platform>\r\n"
    "    </ProjectConfiguration>\r\n"
    "    <ProjectConfiguration Include=\"PGOptimize|x64\">\r\n"
    "      <Configuration>PGOptimize</Configuration>\r\n"
    "      <Platform>x64</Platform>\r\n"
    "    </ProjectConfiguration>\r\n"
    "    <ProjectConfiguration Include=\"Release|ARM\">\r\n"
    "      <Configuration>Release</Configuration>\r\n"
    "      <Platform>ARM</Platform>\r\n"
    "    </ProjectConfiguration>\r\n"
    "    <ProjectConfiguration Include=\"PGInstrument|ARM\">\r\n"
    "      <Configuration>PGInstrument</Configuration>\r\n"
    "      <Platform>ARM</Platform>\r\n"
    "    </ProjectConfiguration>\r\n"
    "    <ProjectConfiguration Include=\"PGUpdate|ARM\">\r\n"
    "      <Configuration>PGUpdate</Configuration>\r\n"
    "      <Platform>ARM</Platform>\r\n"
    "    </ProjectConfiguration>\r\n"
    "    <ProjectConfiguration Include=\"PGOptimize|ARM\">\r\n"
    "      <Configuration>PGOptimize</Configuration>\r\n"
    "      <Platform>ARM</Platform>\r\n"
    "    </ProjectConfiguration>\r\n"
    "    <ProjectConfiguration Include=\"Release|ARM64\">\r\n"
    "      <Configuration>Release</Configuration>\r\n"
    "      <Platform>ARM64</Platform>\r\n"
    "    </ProjectConfiguration>\r\n"
    "    <ProjectConfiguration Include=\"PGInstrument|ARM64\">\r\n"
    "      <Configuration>PGInstrument</Configuration>\r\n"
    "      <Platform>ARM64</Platform>\r\n"
    "    </ProjectConfiguration>\r\n"
    "    <ProjectConfiguration Include=\"PGUpdate|ARM64\">\r\n"
    "      <Configuration>PGUpdate</Configuration>\r\n"
    "      <Platform>ARM64</Platform>\r\n"
    "    </ProjectConfiguration>\r\n"
    "    <ProjectConfiguration Include=\"PGOptimize|ARM64\">\r\n"
    "      <Configuration>PGOptimize</Configuration>\r\n"
    "      <Platform>ARM64</Platform>\r\n"
    "    </ProjectConfiguration>\r\n"
    "  </ItemGroup>\r\n"
    "  <PropertyGroup Label=\"Globals\">\r\n"
    "    <OutDir>..\\$(Platform)\\$(Configuration)\\</OutDir>\r\n"
    "    <WindowsTargetPlatformVersion>" WINDOWS_TARGET_PLATFORM_VERSION "</WindowsTargetPlatformVersion>\r\n"
    "    <PlatformToolset>" PLATFORM_TOOLSET "</PlatformToolset>\r\n"
    "    <ProjectGuid>{";

const STRING VCProjectFileHeaderChunk = {
    sizeof(VCProjectFileHeaderChunkCStr) - sizeof(CHAR),
    sizeof(VCProjectFileHeaderChunkCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&VCProjectFileHeaderChunkCStr,
};

// vim:set ts=8 sw=4 sts=4 tw=0 expandtab                                     :
