/*++

Copyright (c) 2026 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashOnlineJitPrivate.h

Abstract:

    The objects behind the opaque handles of PerfectHashOnlineJit.h. Tests
    include this to stand in for a context's Online interface.

--*/

#pragma once

struct PH_ONLINE_JIT_CONTEXT {
    PICLASSFACTORY ClassFactory;
    PPERFECT_HASH_ONLINE Online;
};

struct PH_ONLINE_JIT_TABLE {
    PPERFECT_HASH_TABLE Table;
};
