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
    DebuggerMaybeWaitForAttachDisposition,
    DebuggerMaybeSwitchToCudaGdbDisposition,
    DebuggerMaybeSwitchBackToGdbDisposition,
} DEBUGGER_DISPOSITION;
typedef DEBUGGER_DISPOSITION *PDEBUGGER_DISPOSITION;

typedef union _DEBUGGER_CONTEXT_FLAGS {
    struct {
        ULONG WaitForGdb:1;
        ULONG WaitForCudaGdb:1;
        ULONG UseGdbForHostDebugging:1;
        ULONG Unused:29;
    };
    ULONG AsULong;
} DEBUGGER_CONTEXT_FLAGS;
C_ASSERT(sizeof(DEBUGGER_CONTEXT_FLAGS) == sizeof(ULONG));
typedef DEBUGGER_CONTEXT_FLAGS *PDEBUGGER_CONTEXT_FLAGS;

#define WantsDebuggerSwitching(Context) (                  \
    (                                                      \
        ((Context)->Flags.WaitForGdb != FALSE) &&          \
        ((Context)->Flags.WaitForCudaGdb == FALSE)         \
    ) || (                                                 \
        ((Context)->Flags.WaitForCudaGdb != FALSE) &&      \
        ((Context)->Flags.UseGdbForHostDebugging != FALSE) \
    )                                                      \
)

typedef union _DEBUGGER_CONTEXT_STATE {
    struct {
        ULONG FirstDebuggerAttach:1;
        ULONG Unused:31;
    };
    ULONG AsULong;
} DEBUGGER_CONTEXT_STATE;
C_ASSERT(sizeof(DEBUGGER_CONTEXT_STATE) == sizeof(ULONG));
typedef DEBUGGER_CONTEXT_STATE *PDEBUGGER_CONTEXT_STATE;

typedef struct _DEBUGGER_CONTEXT {
    DEBUGGER_CONTEXT_FLAGS Flags;
    DEBUGGER_CONTEXT_STATE State;
    DEBUGGER_TYPE FirstDebuggerType;
} DEBUGGER_CONTEXT;
typedef DEBUGGER_CONTEXT *PDEBUGGER_CONTEXT;

//
// Public functions.
//

HRESULT
InitializeDebuggerContext (
    _Out_ PDEBUGGER_CONTEXT Context,
    _In_ PDEBUGGER_CONTEXT_FLAGS Flags
    );

#ifdef PH_WINDOWS

#define MaybeWaitForDebuggerAttach(Context) (S_FALSE)
#define MaybeSwitchToCudaGdb(Context) (S_FALSE)
#define MaybeSwitchBackToCudaGdb(Context) (S_FALSE)
#define MaybeSwitchBackToGdb(Context) (S_FALSE)

#else

HRESULT
MaybeWaitForDebuggerAttach (
    _Inout_ PDEBUGGER_CONTEXT Context
    );

HRESULT
MaybeSwitchToCudaGdb (
    _Inout_ PDEBUGGER_CONTEXT Context
    );

HRESULT
MaybeSwitchBackToCudaGdb (
    _Inout_ PDEBUGGER_CONTEXT Context
    );

HRESULT
MaybeSwitchBackToGdb (
    _Inout_ PDEBUGGER_CONTEXT Context
    );

#endif // !PH_WINDOWS

/*// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :*/
