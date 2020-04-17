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
    TSX-enlightened version of the same interface.  And the RegisterSolved()
    vtbl entry of the GRAPH component.

    And finally, we've also got some Ctrl-C interception glue stashed here.

--*/

#include "stdafx.h"

HMODULE PerfectHashModule;

//
// We need to define a _fltused ULONG symbol as we're working with floats and
// doubles but not linking to the CRT.  Without this, we'll get a missing
// symbol error during linking.  (It is used by the kernel to know if a given
// routine is using floating point state during trap handling.)
//

ULONG _fltused;

//
// Ctrl-C glue.
//

volatile ULONG CtrlCPressed = 0;

BOOL
RunSingleFunctionCtrlCHandler(
    ULONG ControlType
    )
{
    //
    // N.B. We consider all the signals as if Ctrl-C was pressed as it
    //      simplifies the downstream detection logic (and the behavior is
    //      the same for all signals; shutdown cleanly and earliest possible
    //      opportunity).
    //

    BOOLEAN IsCtrlC = (
        ControlType == CTRL_C_EVENT         ||
        ControlType == CTRL_BREAK_EVENT     ||
        ControlType == CTRL_CLOSE_EVENT     ||
        ControlType == CTRL_LOGOFF_EVENT    ||
        ControlType == CTRL_SHUTDOWN_EVENT
    );

    if (IsCtrlC) {
        CtrlCPressed = 1;
        return TRUE;
    }
    return FALSE;
}

#if 0

//
// TSX detection glue.
//

BOOLEAN IsTsxAvailable;

#ifdef _M_AMD64
#pragma optimize("", off)

extern const VOID *ComponentInterfaces[];
extern PVOID GuardedListTsxInterface;

extern GRAPH_VTBL GraphInterface;

PVOID TsxScratch;
static
NOINLINE
BOOLEAN
CanWeUseTsx(VOID)
{
    ULONG Started = 0;
    ULONG Retried = 0;
    ULONG Status;
    BOOLEAN UseTsx = FALSE;

    TsxScratch = NULL;

Retry:

    Status = _xbegin();

    if (Status != _XBEGIN_STARTED) {
        if (Status & _XABORT_RETRY) {
            Retried++;
            goto Retry;
        }
    } else {
        Started++;
        TsxScratch = _AddressOfReturnAddress();
        _xend();
    }

    if (TsxScratch == _AddressOfReturnAddress()) {
        UseTsx = TRUE;
    }

    TsxScratch = NULL;
    return UseTsx;
}
#pragma optimize("", on)
#endif
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

#if 0
            IsTsxAvailable = FALSE;

#ifdef _M_AMD64

            TRY_TSX {

                if (CanWeUseTsx()) {
                    IsTsxAvailable = TRUE;
                }

            } CATCH_EXCEPTION_ILLEGAL_INSTRUCTION {
                NOTHING;
            }

            if (IsTsxAvailable) {

                //
                // Use the TSX-enlightened version of the guarded list interface
                // and graph RegisterSolved() method.
                //

                PERFECT_HASH_INTERFACE_ID Id;
                Id = PerfectHashGuardedListInterfaceId;
                ComponentInterfaces[Id] = &GuardedListTsxInterface;

                GraphInterface.RegisterSolved = GraphRegisterSolvedTsx;
            }

#endif

#endif

            if (!PerfectHashTlsProcessAttach(Module, Reason, Reserved)) {
                return FALSE;
            }

            if (!SetConsoleCtrlHandler(RunSingleFunctionCtrlCHandler, TRUE)) {
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
