/*++

Copyright (c) 2020 Trent Nelson <trent@trent.me>

Module Name:

    memset.c

Abstract:

    This module implements memset().  It is required because the mc-generated
    <PerfectHashEvents.h>  has calls to RtlZeroMemory, which, on a release
    build, is a macro that expands to memset.  We don't link against the CRT,
    so we don't have a memset available, which causes the linker to fail.  So,
    we need to provide one, hence this module.

--*/

#include "stdafx.h"

void *
_memset(void *dest, int c, size_t count)
{
    __stosb(dest, (unsigned char)c, count);
    return dest;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
