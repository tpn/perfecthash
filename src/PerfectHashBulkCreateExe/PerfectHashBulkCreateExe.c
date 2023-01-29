/*++

Copyright (c) 2018-2023 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashBulkCreateExe.c

Abstract:

    This module implements the main entry point for the perfect hash library's
    bulk-create functionality.  It loads the perfect hash library, obtains a
    class factory, creates a context, then calls the bulk-create function
    against the current executable's command line parameters.

--*/

#include "stdafx.h"

//
// Main entry point.
//

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

    Result = Context->Vtbl->BulkCreateArgvW(Context,
                                            NumberOfArguments,
                                            ArgvW,
                                            CommandLineW);

    //
    // Print the usage string if the create routine failed due to invalid number
    // of arguments.  Otherwise, as long as we were able to resolve the error
    // routine, print an error message.
    //

    if (FAILED(Result)) {
        if (Result == PH_E_CONTEXT_BULK_CREATE_INVALID_NUM_ARGS) {
            PH_USAGE();
        } else if (PerfectHashPrintError != NULL) {
            PH_ERROR(BulkCreate, Result);
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

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
