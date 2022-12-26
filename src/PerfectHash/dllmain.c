/*++

Copyright (c) 2018-2022 Trent Nelson <trent@trent.me>

Module Name:

    dllmain.c

Abstract:

    This is the DLL main entry point for the PerfectHash component.  It hooks
    into process and thread attach and detach messages in order to provide TLS
    glue (see PerfectHashTls.c for more information).  It also registers a
    console control handler function to intercept Ctrl-C et al.

--*/

#include "stdafx.h"
#include "PerfectHashEventsPrivate.h"

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

            if (!PerfectHashTlsProcessAttach(Module, Reason, Reserved)) {
                return FALSE;
            }

            if (!SetConsoleCtrlHandler(RunSingleFunctionCtrlCHandler, TRUE)) {
                return FALSE;
            }

            if (EventRegisterPerfectHash() != ERROR_SUCCESS) {
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

            if (EventUnregisterPerfectHash() != ERROR_SUCCESS) {
                NOTHING;
            }
            break;
    }

    return TRUE;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
