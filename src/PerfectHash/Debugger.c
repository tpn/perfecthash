/*++

Copyright (c) 2023 Trent Nelson <trent@trent.me>

Module Name:

    Debugger.c

Abstract:

    This module provides helper routines for attaching and detaching gdb and
    cuda-gdb instances to an instance of the perfect hash executable.  It is
    intended for Linux.

--*/

#include "stdafx.h"

#ifdef PH_WINDOWS
#error This file is not for Windows.
#endif

#ifndef PH_COMPAT
#error PH_COMPAT not defined.
#endif

const STRING GetTracerPidCommand = RTL_CONSTANT_STRING(
    "cat /proc/%d/status | "
        "grep -E \"^TracerPid:[[:space:]]*[1-9][0-9]*$\" | "
        "awk '{print $2}'"
);

const STRING GetNameCommand = RTL_CONSTANT_STRING(
    "cat /proc/%d/status | grep '^Name:' | awk '{ print $2 }'"
);

const STRING CopyToXClipCommand = RTL_CONSTANT_STRING(
    "printf \"%s\" | xclip -selection clipboard"
);

#if 1
const STRING DebuggerAttachCommand = RTL_CONSTANT_STRING(
    "%s --pid=%d %s"
);
#else
const STRING DebuggerAttachCommand = RTL_CONSTANT_STRING(
    "%s --pid=%d %s --ex '!kill -SIGUSR1 %d'"
);
#endif

const STRING GdbExtraCommand = RTL_CONSTANT_STRING(
    "--iex 'set pagination off' "
    "--iex 'set non-stop on' "
    "--iex 'set debug infrun 1'"
    //"--ex 'set scheduler-locking off' "
    //"--ex 'handle SIGUSR1 nostop print pass' "
);

#if 1
const STRING CudaGdbExtraCommand = RTL_CONSTANT_STRING(
    "--iex 'set pagination off' "
    "--iex 'set non-stop on' "
    "--iex 'set debug infrun 1'"
    //"--ex 'set scheduler-locking off' "
    //"--ex 'handle SIGUSR1 nostop print pass' "
);
#else
const STRING CudaGdbExtraCommand = RTL_CONSTANT_STRING("");
#endif

volatile LONG ContinueExecution = 0;

VOID
SignalHandlerUsr1(
    INT Signal
    )
/*++

Routine Description:

    This routine is the signal handler for SIGUSR1.  It is registered by
    WaitForDebugger() and is used to signal to the process that the user
    has attached a debugger and that the process should continue.  (The
    user needs to explicitly send the SIGUSR1 signal to the process from
    an external shell in order to trigger this handler.)

Arguments:

    Signal - Supplies the signal number.

Return Value:

    None.

--*/
{
    printf("Received signal (%d), continuing.\n", Signal);
    ContinueExecution = 1;
}


VOID
InitializeDebuggerContext (
    _Out_ PDEBUGGER_CONTEXT Context,
    _In_ BOOLEAN WaitForDebugger,
    _In_ BOOLEAN SwitchToCudaGdbBeforeLaunchKernel
    )
/*++

Routine Description:

    This routine initializes a DEBUGGER_CONTEXT structure.

Arguments:

    Context - Supplies a pointer to a DEBUGGER_CONTEXT structure to
        initialize.

    WaitForDebugger - Supplies a boolean value indicating whether the
        process should wait for a debugger to attach before continuing.

    SwitchToCudaGdbBeforeLaunchKernel - Supplies a boolean value indicating
        whether the process should switch to cuda-gdb before launching the
        kernel.

Return Value:

    None.

--*/
{
    ZeroStructPointer(Context);
    Context->Flags.WaitForDebugger = WaitForDebugger;
    Context->Flags.SwitchToCudaGdbBeforeLaunchKernel =
        SwitchToCudaGdbBeforeLaunchKernel;
}


HRESULT
CopyStringToClipboard (
    _In_ PSTR String
    )
/*++

Routine Description:

    This routine copies the supplied string to the clipboard using xclip.

Arguments:

    String - Supplies a pointer to a NULL-terminated string to copy to the
        clipboard.

Return Value:

    S_OK - The string was copied to the clipboard.

    Or, an appropriate error code.

--*/
{
    HRESULT Result;
    CHAR Command[MAX_PATH] = { 0, };

    sprintf(Command, CopyToXClipCommand.Buffer, String);

    if (system(Command) != 0) {
        Result = PH_E_SYSTEM_CALL_FAILED;
        PH_ERROR(CopyStringToClipboard_CopyToXClipCommandFailed, Result);
    } else {
        Result = S_OK;
    }

    return Result;
}


HRESULT
IsDebuggerAttached (
    _Out_ PDEBUGGER_TYPE DebuggerType
    )
/*++

Routine Description:

    This routine determines whether a debugger is attached to the current
    process.

Arguments:

    DebuggerType - Supplies a pointer to a variable that receives the debugger
        type, if a debugger is attached.  If no debugger is attached, this
        value will be set to DebuggerTypeNone.

Return Value:

    S_OK - A debugger is attached.

    S_FALSE - No debugger is attached.

    PH_E_SYSTEM_CALL_FAILED - A system call failed.

    PH_E_UNKNOWN_DEBUGGER - An unknown debugger was detected.

--*/
{
    HRESULT Result;
    FILE *Stdout = NULL;
    LONG TracingPid = 0;
    CHAR Command[MAX_PATH] = { 0, };
    CHAR Buffer[MAX_PATH] = { 0, };

    sprintf(Command, GetTracerPidCommand.Buffer, getpid());

    Stdout = popen(Command, "r");
    if (!Stdout) {
        Result = PH_E_SYSTEM_CALL_FAILED;
        PH_ERROR(IsDebuggerAttached_GetTracerPidCommandFailed, Result);
        goto Error;
    }

    if (fgets(Buffer, sizeof(Buffer), Stdout) != NULL) {
        TracingPid = atoi(Buffer);
    }
    pclose(Stdout);
    Stdout = NULL;

    if (TracingPid == 0) {

        //
        // No debugger is attached.
        //

        *DebuggerType = DebuggerTypeNone;
        Result = S_FALSE;
        goto End;
    }

    //
    // Get the name of the process that's tracing us.
    //

    sprintf(Command, GetNameCommand.Buffer, TracingPid);

    Stdout = popen(Command, "r");
    while (fgets(Buffer, sizeof(Buffer), Stdout)) {

        //
        // N.B. It's important we check for cuda-gdb before gdb, as the latter
        //      is contained within the former, so `strstr(Buffer, "gdb")` will
        //      return true for both gdb and cuda-gdb.
        //

        if (strstr(Buffer, "cuda-gdb")) {
            *DebuggerType = DebuggerTypeCudaGdb;
            Result = S_OK;
        } else if (strstr(Buffer, "gdb")) {
            *DebuggerType = DebuggerTypeGdb;
            Result = S_OK;
        } else {
            *DebuggerType = DebuggerTypeUnknown;
            Result = PH_E_UNKNOWN_DEBUGGER;
        }
        break;
    }
    pclose(Stdout);
    Stdout = NULL;

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

End:

    if (Stdout) {
        pclose(Stdout);
    }

    return Result;
}


HRESULT
WaitForDebuggerAttach (
    VOID
    )
/*++

Routine Description:

    This routine waits for a debugger to attach to the current process.

Arguments:

    None.

Return Value:

    S_OK - A debugger is attached to the current process.

    Or, an appropriate error code.

--*/
{
    BOOL First = TRUE;
    HRESULT Result;
    DEBUGGER_TYPE DebuggerType;

    while (TRUE) {
        if (CtrlCPressed) {
            Result = PH_E_CTRL_C_PRESSED;
            break;
        }
        Result = IsDebuggerAttached(&DebuggerType);
        if (FAILED(Result)) {
            PH_ERROR(WaitForDebuggerAttach_IsDebuggerAttachedFailed, Result);
        }
        if (DebuggerType != DebuggerTypeNone) {
            Result = S_OK;
            break;
        } else if (First) {
            printf("Waiting for a debugger to attach.");
            fflush(stdout);
            First = FALSE;
        }
        sleep(1);
        printf(".");
        fflush(stdout);
    }

    if (!First) {
        printf("\n");
    }

    return Result;
}


HRESULT
WaitForDebuggerDetach (
    VOID
    )
/*++

Routine Description:

    This routine waits for a debugger to detach from the current process.

Arguments:

    None.

Return Value:

    S_OK - No debugger is attached to the current process.

    Or, an appropriate error code.

--*/
{
    BOOL First = TRUE;
    HRESULT Result;
    DEBUGGER_TYPE DebuggerType;

    while (TRUE) {
        if (CtrlCPressed) {
            Result = PH_E_CTRL_C_PRESSED;
            break;
        }
        Result = IsDebuggerAttached(&DebuggerType);
        if (FAILED(Result)) {
            PH_ERROR(WaitForDebuggerDetach_IsDebuggerAttachedFailed, Result);
        }
        if (DebuggerType == DebuggerTypeNone) {
            Result = S_OK;
            break;
        } else if (First) {
            printf("Waiting for debugger to detach.");
            fflush(stdout);
            First = FALSE;
        }
        sleep(1);
        printf(".");
        fflush(stdout);
    }

    if (!First) {
        printf("\n");
    }

    return Result;
}


HRESULT
DoDebuggerAction (
    _Inout_ PDEBUGGER_CONTEXT Context,
    _In_ DEBUGGER_DISPOSITION Disposition
    )
/*++

Routine Description:

    This routine handles the various debugger actions that can be taken.

Arguments:

    Context - Supplies a pointer to an initialized DEBUGGER_CONTEXT structure.
        The context flags WaitForDebugger and SwitchToCudaGdbBeforeLaunchKernel
        are consulted to determine how this routine should behave.

    Disposition - Supplies the desired disposition of the debugger attach.

Return Value:

    S_OK - The debugger attach was successful.

    S_FALSE - The debugger attach was not performed.

    PH_E_SYSTEM_CALL_FAILED - A system call failed.

    PH_E_UNKNOWN_DEBUGGER - An unknown debugger was detected.

    Or, an appropriate error code.

--*/
{
    INT Pid;
    ULONG Count;
    HRESULT Result;
    CHAR Buffer[MAX_PATH] = { 0, };
    PSTR DebuggerName = "gdb";
    PSTR DebuggerExtra = GdbExtraCommand.Buffer;
    DEBUGGER_TYPE CurrentDebuggerType;
    DEBUGGER_TYPE DesiredDebuggerType = DebuggerTypeNone;

    //
    // If the debugger context flags don't indicate that we should wait for a
    // debugger, return S_FALSE immediately.
    //

    if (!Context->Flags.WaitForDebugger) {
        return S_FALSE;
    }

    //
    // Always reset the volatile global sentinel ContinueExecution to 0.
    //

    ContinueExecution = 0;

    //
    // We explicitly, unconditionally, *always* register our signal handler.
    // When we don't do this, cycling between gdb and cuda-gdb appears to lose
    // the signal handler registration.
    //

    signal(SIGUSR1, SignalHandlerUsr1);

    //
    // Determine if a debugger is already attached.
    //

    Result = IsDebuggerAttached(&CurrentDebuggerType);
    if (FAILED(Result)) {
        PH_ERROR(DoDebuggerAction_IsDebuggerAttachedFailed, Result);
        return Result;
    }

    //
    // The only permissible return value here is S_OK or S_FALSE.
    //

    ASSERT(Result == S_OK || Result == S_FALSE);

    if (Result == S_FALSE) {

        //
        // No debugger was attached, this is probably the first time the
        // function was called.
        //

        ASSERT(CurrentDebuggerType == DebuggerTypeNone);

        if (Disposition != DebuggerMaybeWaitForGdbAttachDisposition) {

            //
            // The disposition doesn't match the current debugger state.
            //

            return S_FALSE;

        } else {
            DesiredDebuggerType = DebuggerTypeGdb;
        }

    } else {

        //
        // A debugger is currently attached.  If the context flag indicates no
        // CUDA GDB switch is desired, there's nothing more to do.
        //

        if (!Context->Flags.SwitchToCudaGdbBeforeLaunchKernel) {
            return S_FALSE;
        }

        if (CurrentDebuggerType == DebuggerTypeGdb) {

            if (Disposition == DebuggerMaybeSwitchToCudaGdbDisposition) {
                DesiredDebuggerType = DebuggerTypeCudaGdb;
            }

        } else {

            if (Disposition == DebuggerMaybeSwitchBackToGdbDisposition) {
                DesiredDebuggerType = DebuggerTypeGdb;
            }

        }

        if (DesiredDebuggerType == DebuggerTypeNone) {

            //
            // The disposition doesn't match the current debugger state.
            //

            return S_FALSE;
        }

        printf("Detach existing debugger now (e.g. via `detach`).\n");
        Result = WaitForDebuggerDetach();
        if (FAILED(Result)) {
            return Result;
        }
        printf("Existing debugger detached, continuing.\n");
    }

    if (DesiredDebuggerType == DebuggerTypeCudaGdb) {
        DebuggerName = "cuda-gdb";
        DebuggerExtra = CudaGdbExtraCommand.Buffer;
    }

    //
    // Construct the desired debugger attach command and print it.
    //

    Pid = getpid();
    sprintf(Buffer,
            DebuggerAttachCommand.Buffer,
            DebuggerName,
            Pid,
            DebuggerExtra,
            Pid);

    printf("Attach %s debugger now: `%s`\n", DebuggerName, Buffer);

    if (SUCCEEDED(CopyStringToClipboard(Buffer))) {
        printf("N.B. Debugger attach command copied to clipboard.\n");
    }

    Result = WaitForDebuggerAttach();
    if (FAILED(Result)) {

        //
        // We can't do anything more here, and an error will have already been
        // printed.
        //

        NOTHING;
    } else {
        printf("Debugger attached, continuing.\n");
    }

}


//
// Glue for switching between gdb and cuda-gdb.
//

HRESULT
MaybeWaitForGdbAttach (
    _Inout_ PDEBUGGER_CONTEXT Context
    )
{
    DoDebuggerAction(Context, DebuggerMaybeWaitForGdbAttachDisposition);
}

HRESULT
MaybeSwitchToCudaGdb (
    _Inout_ PDEBUGGER_CONTEXT Context
    )
{
    DoDebuggerAction(Context, DebuggerMaybeSwitchToCudaGdbDisposition);
}

HRESULT
MaybeSwitchBackToGdb (
    _Inout_ PDEBUGGER_CONTEXT Context
    )
{
    DoDebuggerAction(Context, DebuggerMaybeSwitchBackToGdbDisposition);
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
