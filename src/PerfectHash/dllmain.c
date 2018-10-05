/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    dllmain.c

Abstract:

    This is the DLL main entry point for the PerfectHash component.  It hooks
    into process and thread attach and detach messages in order to provide TLS
    glue (see PerfectHashTls.c for more information).

--*/

#include "stdafx.h"

HMODULE PerfectHashModule;

BOOLEAN IsTsxAvailable;

#ifdef _M_AMD64
#pragma optimize("", off)

extern const VOID *ComponentInterfaces[];
extern PVOID GuardedListTsxInterface;

PVOID TsxScratch;
static
NOINLINE
BOOLEAN
CanWeUseTsx(VOID)
{
    ULONG Status;
    BOOLEAN UseTsx = TRUE;

    TRY_TSX {
Retry:
        Status = _xbegin();
        if (Status & _XABORT_RETRY) {
            goto Retry;
        } else if (Status != _XBEGIN_STARTED) {
            goto End;
        }

        TsxScratch = _AddressOfReturnAddress();
        _xend();
    } CATCH_EXCEPTION_ILLEGAL_INSTRUCTION {
        UseTsx = FALSE;
    }

End:
    TsxScratch = NULL;
    return UseTsx;
}
#pragma optimize("", on)
#endif

BOOL
APIENTRY
_DllMainCRTStartup(
    _In_    HMODULE     Module,
    _In_    DWORD       Reason,
    _In_    LPVOID      Reserved
    )
{

    switch (Reason) {
        case DLL_PROCESS_ATTACH:
            __security_init_cookie();
            PerfectHashModule = Module;
            IsTsxAvailable = FALSE;
#ifdef _M_AMD64
            if (CanWeUseTsx()) {
                PERFECT_HASH_INTERFACE_ID Id;
                IsTsxAvailable = TRUE;
                Id = PerfectHashGuardedListInterfaceId;
                ComponentInterfaces[Id] = &GuardedListTsxInterface;
            }
#endif
            if (!PerfectHashTlsProcessAttach(Module, Reason, Reserved)) {
                return FALSE;
            }
            break;
        case DLL_THREAD_ATTACH:
            break;
        case DLL_THREAD_DETACH:
            break;
        case DLL_PROCESS_DETACH:
            if (!PerfectHashTlsProcessDetach(Module, Reason, Reserved)) {
                NOTHING;
            }
            break;
    }

    return TRUE;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
