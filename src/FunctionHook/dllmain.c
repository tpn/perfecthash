/*++

Copyright (c) 2022-2023 Trent Nelson <trent@trent.me>

Module Name:

    dllmain.c

Abstract:

    This is the DLL main entry point for the FunctionHook component.

--*/

#include "stdafx.h"

HMODULE FunctionHookModule = NULL;

BOOL
APIENTRY
_DllMainCRTStartup(
    _In_    HMODULE     Module,
    _In_    DWORD       Reason,
    _In_    LPVOID      Reserved
    )
{
    UNREFERENCED_PARAMETER(Reserved);

    switch (Reason) {
        case DLL_PROCESS_ATTACH:
            FunctionHookModule = Module;
            __security_init_cookie();
            break;
        case DLL_THREAD_ATTACH:
            break;
        case DLL_THREAD_DETACH:
            break;
        case DLL_PROCESS_DETACH:
            break;
    }

    return TRUE;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
