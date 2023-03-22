/*++

Copyright (c) 2023 Trent Nelson <trent@trent.me>

Module Name:

    Debugger.h

Abstract:

    This is the header file for the debugger module.

--*/

#include "stdafx.h"

typedef enum _DEBUGGER_TYPE {
    DebuggerTypeNone = 0,
    DebuggerTypeGdb,
    DebuggerTypeCudaGdb,
    DebuggerTypeUnknown,
} DEBUGGER_TYPE;
typedef DEBUGGER_TYPE *PDEBUGGER_TYPE;

typedef enum _DEBUGGER_DISPOSITION {
    DebuggerDispositionNone = 0,
    DebuggerMaybeWaitForGdbAttachDisposition,
    DebuggerMaybeSwitchToCudaGdbDisposition,
    DebuggerMaybeSwitchBackToGdbDisposition,
} DEBUGGER_DISPOSITION;
typedef DEBUGGER_DISPOSITION *PDEBUGGER_DISPOSITION;

typedef union _DEBUGGER_CONTEXT_FLAGS {
    struct {
        ULONG WaitForDebugger:1;
        ULONG SwitchToCudaGdbBeforeLaunchKernel:1;
        ULONG Unused:30;
    };
    ULONG AsULong;
} DEBUGGER_CONTEXT_FLAGS;
C_ASSERT(sizeof(DEBUGGER_CONTEXT_FLAGS) == sizeof(ULONG));
typedef DEBUGGER_CONTEXT_FLAGS *PDEBUGGER_CONTEXT_FLAGS;

typedef union _DEBUGGER_CONTEXT_STATE {
    struct {
        ULONG Unused:32;
    };
    ULONG AsULong;
} DEBUGGER_CONTEXT_STATE;
C_ASSERT(sizeof(DEBUGGER_CONTEXT_STATE) == sizeof(ULONG));
typedef DEBUGGER_CONTEXT_STATE *PDEBUGGER_CONTEXT_STATE;

typedef struct _DEBUGGER_CONTEXT {
    DEBUGGER_CONTEXT_FLAGS Flags;
    DEBUGGER_CONTEXT_STATE State;
} DEBUGGER_CONTEXT;
typedef DEBUGGER_CONTEXT *PDEBUGGER_CONTEXT;

#ifdef PH_WINDOWS

#define WaitForDebuggerAttach(Debugger, Disposition)

#else

//
// Public functions.
//

VOID
InitializeDebuggerContext (
    _Out_ PDEBUGGER_CONTEXT Context,
    _In_ BOOLEAN WaitForDebugger,
    _In_ BOOLEAN SwitchToCudaGdbBeforeLaunchKernel
    );

HRESULT
MaybeWaitForGdbAttach (
    _Inout_ PDEBUGGER_CONTEXT Context
    );

HRESULT
MaybeSwitchToCudaGdb (
    _Inout_ PDEBUGGER_CONTEXT Context
    );

HRESULT
MaybeSwitchBackToGdb (
    _Inout_ PDEBUGGER_CONTEXT Context
    );

#endif // !PH_WINDOWS

/*// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :*/
