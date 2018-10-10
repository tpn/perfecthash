/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    dllmain.c

Abstract:

    This is the DLL main entry point for the PerfectHash component.  It hooks
    into process and thread attach and detach messages in order to provide TLS
    glue (see PerfectHashTls.c for more information).

    It also attempts a TSX transaction on process attach (if we're x64), and,
    if that succeeds, replaces the guarded list component interface with the
    TSX-enlightened version of the same interface.

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
    BOOLEAN UseTsx = FALSE;

    TsxScratch = NULL;

Retry:
    Status = _xbegin();
    if (Status & _XABORT_RETRY) {
        goto Retry;
    } else if (Status != _XBEGIN_STARTED) {
        goto End;
    }

    TsxScratch = _AddressOfReturnAddress();
    _xend();

End:

    if (TsxScratch == _AddressOfReturnAddress()) {
        UseTsx = TRUE;
    }

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

            IsTsxAvailable = TRUE;

#ifdef _M_AMD64

            TRY_TSX {

                if (CanWeUseTsx()) {
                    NOTHING;
                }

            } CATCH_EXCEPTION_ILLEGAL_INSTRUCTION {
                IsTsxAvailable = FALSE;
            }

            if (IsTsxAvailable) {
                PERFECT_HASH_INTERFACE_ID Id;
                Id = PerfectHashGuardedListInterfaceId;
                ComponentInterfaces[Id] = &GuardedListTsxInterface;
            }

#endif

#if 0
            IsTsxAvailable = FALSE;

#ifdef _M_AMD64

            TRY_TSX {

                if (CanWeUseTsx()) {
                    PERFECT_HASH_INTERFACE_ID Id;
                    IsTsxAvailable = TRUE;
                    Id = PerfectHashGuardedListInterfaceId;
                    ComponentInterfaces[Id] = &GuardedListTsxInterface;
                }

            } CATCH_EXCEPTION_ILLEGAL_INSTRUCTION {
                IsTsxAvailable = FALSE;
            }

#endif
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
