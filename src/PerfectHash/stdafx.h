/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    stdafx.h

Abstract:

    This is the precompiled header file for the PerfectHashTable component.

--*/

#ifndef _PERFECT_HASH_TABLE_INTERNAL_BUILD
#error PerfectHashTable's stdafx.h being included but _PERFECT_HASH_TABLE_INTERNAL_BUILD not set.
#endif

#pragma once

#include "targetver.h"

//
// N.B. The warning disable glue is necessary to get the system headers to
//      include with all errors enabled (/Wall).
//

//
// 4255:
//      winuser.h(6502): warning C4255: 'EnableMouseInPointerForThread':
//          no function prototype given: converting '()' to '(void)'
//
// 4668:
//      winioctl.h(8910): warning C4668: '_WIN32_WINNT_WIN10_TH2'
//          is not defined as a preprocessor macro, replacing with
//          '0' for '#if/#elif'
//
//

#pragma warning(push)
#pragma warning(disable: 4255)
#pragma warning(disable: 4668)
#include <Windows.h>
#pragma warning(pop)

//
// <concurrencysal.h> appears to need _PREFAST_ defined.
//

#ifndef _PREFAST_
#define _PREFAST_
#endif

#include <sal.h>
#include <concurrencysal.h>

#include <Windows.h>
#include <Strsafe.h>

#include <PerfectHashTable.h>
#include <PerfectHashTableErrors.h>

#include "Component.h"
#include "Rtl.h"
#include "RtlOutput.h"
#include "PerfectHashTableErrorHandling.h"
#include "PerfectHashTableTls.h"
#include "PerfectHashTableKeys.h"
#include "PerfectHashTablePrime.h"
#include "PerfectHashTableAllocator.h"
#include "PerfectHashTableContext.h"
#include "PerfectHashTablePrivate.h"
#include "PerfectHashTableConstants.h"

//
// warning C4820: '<unnamed-tag>': '4' bytes padding added after
//      data member 'MessageId'
//

#pragma warning(push)
#pragma warning(disable: 4820)
#include "PerfectHashTableErrors.dbg"
#pragma warning(pop)

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
