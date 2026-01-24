/*++

Copyright (c) 2018-2023 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashCreateExe.c

Abstract:

    This module implements the main entry point for the perfect hash library's
    table create functionality.  It loads the perfect hash library, obtains a
    class factory, creates a context, then calls the table create function with
    the current executable's command line parameters.

--*/

#include "stdafx.h"

#include <stdio.h>
#include <string.h>

#ifdef PH_WINDOWS
#pragma warning(push)
#pragma warning(disable: 4820 4255)
#include <DbgHelp.h>
#pragma warning(pop)
#endif

#ifdef PH_WINDOWS
#define PH_CREATE_MAX_STACK_FRAMES 64

static
VOID
PerfectHashCreateWriteString(
    _In_z_ PCSTR String
    )
/*++

Routine Description:

    Writes a null-terminated string to STDERR.

Arguments:

    String - Supplies a pointer to a NULL-terminated string.

Return Value:

    None.

--*/
{
    DWORD BytesWritten;
    size_t Length;
    HANDLE ErrorHandle;

    ErrorHandle = GetStdHandle(STD_ERROR_HANDLE);
    if (ErrorHandle == INVALID_HANDLE_VALUE || ErrorHandle == NULL) {
        return;
    }

    Length = strlen(String);
    if (Length == 0 || Length > MAXDWORD) {
        return;
    }

    WriteFile(ErrorHandle,
              String,
              (DWORD)Length,
              &BytesWritten,
              NULL);
}

static
VOID
PerfectHashCreateDumpStack(
    _In_ PCONTEXT Context
    )
/*++

Routine Description:

    Dumps a best-effort stack trace to STDERR using DbgHelp.  Intended for
    diagnosing unhandled exceptions during table creation in debug builds.

Arguments:

    Context - Supplies a pointer to a CONTEXT record.

Return Value:

    None.

--*/
{
    HANDLE Process;
    HANDLE Thread;
    DWORD MachineType;
    DWORD64 Displacement;
    DWORD LineDisplacement;
    ULONG FrameIndex;
    STACKFRAME64 Frame = { 0 };
    PCSTR SymbolPath = NULL;
    CHAR ModulePath[MAX_PATH];
    CHAR *LastSlash;
    CHAR Buffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME] = { 0 };
    PSYMBOL_INFO Symbol = (PSYMBOL_INFO)Buffer;
    IMAGEHLP_LINE64 Line = { 0 };
    CHAR LineBuffer[512];

    if (!ARGUMENT_PRESENT(Context)) {
        return;
    }

    Process = GetCurrentProcess();
    Thread = GetCurrentThread();

    SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME | SYMOPT_LOAD_LINES);

    ModulePath[0] = '\0';
    if (GetModuleFileNameA(NULL, ModulePath, ARRAYSIZE(ModulePath)) != 0) {
        LastSlash = strrchr(ModulePath, '\\');
        if (LastSlash) {
            *LastSlash = '\0';
            SymbolPath = ModulePath;
        }
    }

    if (!SymInitialize(Process, SymbolPath, TRUE)) {
        return;
    }

#if defined(_M_AMD64) || defined(_M_X64)
    MachineType = IMAGE_FILE_MACHINE_AMD64;
    Frame.AddrPC.Offset = Context->Rip;
    Frame.AddrFrame.Offset = Context->Rbp;
    Frame.AddrStack.Offset = Context->Rsp;
#elif defined(_M_IX86)
    MachineType = IMAGE_FILE_MACHINE_I386;
    Frame.AddrPC.Offset = Context->Eip;
    Frame.AddrFrame.Offset = Context->Ebp;
    Frame.AddrStack.Offset = Context->Esp;
#else
    MachineType = IMAGE_FILE_MACHINE_UNKNOWN;
#endif

    Frame.AddrPC.Mode = AddrModeFlat;
    Frame.AddrFrame.Mode = AddrModeFlat;
    Frame.AddrStack.Mode = AddrModeFlat;

    PerfectHashCreateWriteString("Unhandled exception stack trace:\n");

    for (FrameIndex = 0; FrameIndex < PH_CREATE_MAX_STACK_FRAMES; FrameIndex++) {

        if (!StackWalk64(MachineType,
                         Process,
                         Thread,
                         &Frame,
                         Context,
                         NULL,
                         SymFunctionTableAccess64,
                         SymGetModuleBase64,
                         NULL)) {
            break;
        }

        if (Frame.AddrPC.Offset == 0) {
            break;
        }

        Symbol->SizeOfStruct = sizeof(*Symbol);
        Symbol->MaxNameLen = MAX_SYM_NAME;
        Displacement = 0;

        Line.SizeOfStruct = sizeof(Line);
        LineDisplacement = 0;

        if (SymFromAddr(Process,
                        Frame.AddrPC.Offset,
                        &Displacement,
                        Symbol)) {

            if (SymGetLineFromAddr64(Process,
                                     Frame.AddrPC.Offset,
                                     &LineDisplacement,
                                     &Line)) {
                sprintf_s(LineBuffer,
                          sizeof(LineBuffer),
                          "  %02lu: %s + 0x%llx (%s:%lu)\n",
                          FrameIndex,
                          Symbol->Name,
                          (unsigned long long)Displacement,
                          Line.FileName,
                          Line.LineNumber);
            } else {
                sprintf_s(LineBuffer,
                          sizeof(LineBuffer),
                          "  %02lu: %s + 0x%llx\n",
                          FrameIndex,
                          Symbol->Name,
                          (unsigned long long)Displacement);
            }

        } else {
            sprintf_s(LineBuffer,
                      sizeof(LineBuffer),
                      "  %02lu: 0x%llx\n",
                      FrameIndex,
                      (unsigned long long)Frame.AddrPC.Offset);
        }

        PerfectHashCreateWriteString(LineBuffer);
    }

    SymCleanup(Process);
}

static
LONG
WINAPI
PerfectHashCreateUnhandledExceptionFilter(
    _In_ PEXCEPTION_POINTERS ExceptionPointers
    )
/*++

Routine Description:

    Unhandled exception filter that logs the exception code and stack trace.

Arguments:

    ExceptionPointers - Supplies a pointer to the exception information.

Return Value:

    EXCEPTION_EXECUTE_HANDLER.

--*/
{
    CHAR Buffer[128];
    ULONG Code;

    if (!ARGUMENT_PRESENT(ExceptionPointers) ||
        !ARGUMENT_PRESENT(ExceptionPointers->ExceptionRecord)) {
        return EXCEPTION_EXECUTE_HANDLER;
    }

    Code = ExceptionPointers->ExceptionRecord->ExceptionCode;

    sprintf_s(Buffer,
              sizeof(Buffer),
              "Unhandled exception: 0x%08lX\n",
              Code);
    PerfectHashCreateWriteString(Buffer);

    if (ARGUMENT_PRESENT(ExceptionPointers->ContextRecord)) {
        PerfectHashCreateDumpStack(ExceptionPointers->ContextRecord);
    }

    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

//
// Main entry point.
//

#ifdef PH_WINDOWS
DECLSPEC_NORETURN
VOID
WINAPI
mainCRTStartup(
    VOID
    )
{
    HMODULE Module = NULL;
    HRESULT Result = S_OK;
    LPWSTR *ArgvW;
    LPWSTR CommandLineW;
    PICLASSFACTORY ClassFactory;
    PPERFECT_HASH_CONTEXT Context;
    PPERFECT_HASH_PRINT_ERROR PerfectHashPrintError = NULL;
    PPERFECT_HASH_PRINT_MESSAGE PerfectHashPrintMessage;
    PICLASSFACTORY_CREATE_INSTANCE CreateInstance;
    INT NumberOfArguments = 0;

    CommandLineW = GetCommandLineW();
    ArgvW = CommandLineToArgvW(CommandLineW, &NumberOfArguments);

#ifdef PH_WINDOWS
    SetUnhandledExceptionFilter(PerfectHashCreateUnhandledExceptionFilter);
#endif

    Result = PerfectHashBootstrap(&ClassFactory,
                                  &PerfectHashPrintError,
                                  &PerfectHashPrintMessage,
                                  &Module);

    if (FAILED(Result)) {

        //
        // We can only use PH_ERROR() if PerfectHashPrintError is available.
        //

        if (PerfectHashPrintError != NULL) {
            PH_ERROR(PerfectHashBootstrap, Result);
        }

        goto Error;
    }

    CreateInstance = ClassFactory->Vtbl->CreateInstance;

    Result = CreateInstance(ClassFactory,
                            NULL,
                            &IID_PERFECT_HASH_CONTEXT,
                            &Context);

    if (FAILED(Result)) {
        if (PerfectHashPrintError != NULL) {
            PH_ERROR(PerfectHashContextCreateInstance, Result);
        }
        goto Error;
    }

    Result = Context->Vtbl->TableCreateArgvW(Context,
                                             NumberOfArguments,
                                             ArgvW,
                                             CommandLineW);

    //
    // Print the usage string if the create routine failed due to invalid number
    // of arguments.  Otherwise, as long as we were able to resolve the error
    // routine, print an error message.
    //

    if (FAILED(Result)) {
        if (Result == PH_E_CONTEXT_TABLE_CREATE_INVALID_NUM_ARGS) {
            PH_USAGE();
        } else if (PerfectHashPrintError != NULL) {
            PH_ERROR(TableCreate, Result);
        }
    }

    Context->Vtbl->Release(Context);

    ClassFactory->Vtbl->Release(ClassFactory);

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    if (Module) {
        FreeLibrary(Module);
    }

    ExitProcess((ULONG)Result);
}

#else // PH_WINDOWS

int
main(
    int NumberOfArguments,
    char **ArgvA
    )
{
    HMODULE Module = NULL;
    HRESULT Result = S_OK;
    LPWSTR *ArgvW;
    LPWSTR CommandLineW;
    PICLASSFACTORY ClassFactory;
    PPERFECT_HASH_CONTEXT Context;
    PPERFECT_HASH_PRINT_ERROR PerfectHashPrintError = NULL;
    PPERFECT_HASH_PRINT_MESSAGE PerfectHashPrintMessage;
    PICLASSFACTORY_CREATE_INSTANCE CreateInstance;

    Result = PerfectHashBootstrap(&ClassFactory,
                                  &PerfectHashPrintError,
                                  &PerfectHashPrintMessage,
                                  &Module);

    if (FAILED(Result)) {

        //
        // We can only use PH_ERROR() if PerfectHashPrintError is available.
        //

        if (PerfectHashPrintError != NULL) {
            PH_ERROR(PerfectHashBootstrap, Result);
        }

        goto Error;
    }

    CreateInstance = ClassFactory->Vtbl->CreateInstance;

    Result = CreateInstance(ClassFactory,
                            NULL,
                            &IID_PERFECT_HASH_CONTEXT,
                            &Context);

    if (FAILED(Result)) {
        if (PerfectHashPrintError != NULL) {
            PH_ERROR(PerfectHashContextCreateInstance, Result);
        }
        goto Error;
    }

    Result = Context->Vtbl->TableCreateArgvA(Context,
                                             NumberOfArguments,
                                             ArgvA);

    //
    // Print the usage string if the create routine failed due to invalid number
    // of arguments.  Otherwise, as long as we were able to resolve the error
    // routine, print an error message.
    //

    if (FAILED(Result)) {
        if (Result == PH_E_CONTEXT_TABLE_CREATE_INVALID_NUM_ARGS) {
            PH_USAGE();
        } else if (PerfectHashPrintError != NULL) {
            PH_ERROR(TableCreate, Result);
        }
    }

    Context->Vtbl->Release(Context);

    ClassFactory->Vtbl->Release(ClassFactory);

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    ExitProcess((ULONG)Result);
}
#endif // PH_WINDOWS

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
