/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    VCProjectFileChunks.h

Abstract:

    This is the private header file for the common chunk definitions used
    for .vcxproj files.

--*/

#pragma once

#include "stdafx.h"

#define WINDOWS_TARGET_PLATFORM_VERSION "8.1"
#define PLATFORM_TOOLSET "v141"

extern const CHAR VCProjectFileHeaderChunkCStr[];
extern const STRING VCProjectFileHeaderChunk;

// vim:set ts=8 sw=4 sts=4 tw=0 expandtab                                     :
