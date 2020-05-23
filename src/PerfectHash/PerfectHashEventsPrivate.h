/*++

Copyright (c) 2020 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashEventsPrivate.h

Abstract:

    This is the private header file for the PerfectHashEvents component.  It is
    intended to be included by modules needing ETW facilities (instead of
    including the <PerfectHashEvents.h> header that ships in the ../../include
    directory).

    It's sole job at the moment is to provide glue for the memset hack (see
    comment in memset.c.)

--*/

#pragma once

void *_memset(void* dest, int c, size_t count);

//
// 4514: unreferenced inline function removed
//
// 4710: function not inlined
//
// 4820: padding added after member
//
// 5045: spectre mitigation warning
//
// 26451: arithmetic overflow warning
//

#pragma warning(push)
#pragma warning(disable: 4514 4710 4820 5045 26451)
#define RtlZeroMemory(Destination,Length) _memset((Destination),0,(Length))
#define memset _memset
#include <PerfectHashEvents.h>
#undef RtlZeroMemory
#undef memset
#pragma warning(pop)

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
