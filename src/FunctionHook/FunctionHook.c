/*++

Copyright (c) 2022-2023 Trent Nelson <trent@trent.me>

Module Name:

    FunctionHook.c

Abstract:

    This module implements function hooking glue.

--*/

#include "stdafx.h"

//
// Globals
//

volatile PFUNCTION_ENTRY_CALLBACK FunctionEntryCallback = NULL;
volatile PVOID FunctionEntryCallbackContext = NULL;
volatile PVOID HookedModuleBaseAddress = NULL;
volatile ULONG HookedModuleSizeInBytes = 0;
volatile ULONG HookedModuleIgnoreRip = 0;

//
// Functions
//

VOID
SetFunctionEntryCallback (
    _In_ PFUNCTION_ENTRY_CALLBACK Callback,
    _In_ PVOID Context,
    _In_ PVOID ModuleBaseAddress,
    _In_ ULONG ModuleSizeInBytes,
    _In_ ULONG IgnoreRip
    )
{
    FunctionEntryCallbackContext = Context;
    HookedModuleSizeInBytes = ModuleSizeInBytes;
    HookedModuleBaseAddress = ModuleBaseAddress;
    HookedModuleIgnoreRip = IgnoreRip;
    FunctionEntryCallback = Callback;
}

VOID
GetFunctionEntryCallback (
    _Out_ PFUNCTION_ENTRY_CALLBACK *Callback,
    _Out_ PVOID *Context,
    _Out_ PVOID *ModuleBaseAddress,
    _Out_ ULONG *ModuleSizeInBytes,
    _Out_ ULONG *IgnoreRip
    )
{
    *Context = FunctionEntryCallbackContext;
    *ModuleSizeInBytes = HookedModuleSizeInBytes;
    *ModuleBaseAddress = HookedModuleBaseAddress;
    *IgnoreRip = HookedModuleIgnoreRip;
    *Callback = FunctionEntryCallback;
}

BOOLEAN
IsFunctionEntryCallbackEnabled (
    VOID
    )
{
    return (
        FunctionEntryCallback != NULL &&
        HookedModuleBaseAddress != NULL &&
        HookedModuleSizeInBytes > 0
    );
}

VOID
ClearFunctionEntryCallback (
    _Out_opt_ PFUNCTION_ENTRY_CALLBACK *Callback,
    _Out_opt_ PVOID *Context,
    _Out_opt_ PVOID *ModuleBaseAddress,
    _Out_opt_ ULONG *ModuleSizeInBytes,
    _Out_opt_ ULONG *IgnoreRip
    )
{
    if (ARGUMENT_PRESENT(Context)) {
        *Context = FunctionEntryCallbackContext;
    }

    if (ARGUMENT_PRESENT(ModuleSizeInBytes)) {
        *ModuleSizeInBytes = HookedModuleSizeInBytes;
    }

    if (ARGUMENT_PRESENT(ModuleBaseAddress)) {
        *ModuleBaseAddress = HookedModuleBaseAddress;
    }

    if (ARGUMENT_PRESENT(IgnoreRip)) {
        *IgnoreRip = HookedModuleIgnoreRip;
    }

    if (ARGUMENT_PRESENT(Callback)) {
        *Callback = FunctionEntryCallback;
    }

    FunctionEntryCallback = NULL;
    HookedModuleBaseAddress = NULL;
    HookedModuleSizeInBytes = 0;
    HookedModuleIgnoreRip = 0;
    FunctionEntryCallbackContext = NULL;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
