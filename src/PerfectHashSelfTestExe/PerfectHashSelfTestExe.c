/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashSelfTestExe.c

Abstract:

    This module implements the main entry point for the perfect hash library's
    self-test functionality.  It loads the perfect hash library, obtains a
    class factory, creates a context, then calls the self-test function against
    the current executable's command line parameters.

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
    HRESULT Result;
    LPWSTR *ArgvW;
    LPWSTR CommandLineW;
    PICLASSFACTORY ClassFactory;
    PPERFECT_HASH_CONTEXT Context;
    PICLASSFACTORY_CREATE_INSTANCE CreateInstance;
    INT NumberOfArguments = 0;

    CommandLineW = GetCommandLineW();
    ArgvW = CommandLineToArgvW(CommandLineW, &NumberOfArguments);

    Result = PerfectHashLoadLibraryAndGetClassFactory(&ClassFactory);
    if (FAILED(Result)) {
        ExitProcess(1);
    }

    CreateInstance = ClassFactory->Vtbl->CreateInstance;

    Result = CreateInstance(ClassFactory,
                            NULL,
                            &IID_PERFECT_HASH_CONTEXT,
                            &Context);

    if (FAILED(Result)) {
        ExitProcess(2);
    }

    Result = Context->Vtbl->SelfTestArgvW(Context,
                                          NumberOfArguments,
                                          ArgvW);

    if (FAILED(Result)) {
        ExitProcess((ULONG)Result);
    }

    Context->Vtbl->Release(Context);

    ExitProcess(0);
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
