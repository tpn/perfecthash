/*++

Copyright (c) 2018-2022 Trent Nelson <trent@trent.me>

Module Name:

    VCProjectFileChunks.h

Abstract:

    This is the private header file for the common chunk definitions used
    for .vcxproj files.

--*/

#pragma once

#include "stdafx.h"

#define WINDOWS_TARGET_PLATFORM_VERSION "10.0.22000.0"
#define PLATFORM_TOOLSET "v143"

extern const CHAR VCProjectFileHeaderChunkCStr[];
extern const STRING VCProjectFileHeaderChunk;

// vim:set ts=8 sw=4 sts=4 tw=0 expandtab                                     :
