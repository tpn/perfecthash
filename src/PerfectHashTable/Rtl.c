/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Rtl.c

Abstract:

    TBD.

--*/

#include "stdafx.h"

//
// Forward definitions.
//

extern LOAD_SYMBOLS_FROM_MULTIPLE_MODULES LoadSymbolsFromMultipleModules;

const PCSTR RtlFunctionNames[] = {
    _RTL_FUNCTION_NAMES_HEAD
};
#define Names RtlFunctionNames

const PCZPCWSTR RtlModuleNames[] = {
    _RTL_MODULE_NAMES_HEAD
};
#define ModuleNames RtlModuleNames

RTL_INITIALIZE RtlInitialize;

_Use_decl_annotations_
HRESULT
RtlInitialize(
    PRTL Rtl
    )
{
    BOOL Success;
    ULONG Index;
    HRESULT Result;
    HMODULE *Module;
    PWSTR *Name;
    ULONG NumberOfModules;
    ULONG NumberOfSymbols;
    ULONG NumberOfResolvedSymbols;

    //
    // Define an appropriately sized bitmap we can passed to LoadSymbols().
    //

    ULONG BitmapBuffer[(ALIGN_UP(ARRAYSIZE(Names),
                        sizeof(ULONG) << 3) >> 5)+1] = { 0 };
    RTL_BITMAP FailedBitmap;

    FailedBitmap.SizeOfBitMap = ARRAYSIZE(Names) + 1;
    FailedBitmap.Buffer = (PULONG)&BitmapBuffer;

    if (!ARGUMENT_PRESENT(Rtl)) {
        return E_POINTER;
    }

    //
    // Attempt to initialize the error handling scaffolding such that SYS_ERROR
    // is available as early as possible.
    //

    Rtl->SysErrorOutputHandle = GetStdHandle(STD_ERROR_HANDLE);

    //
    // Create an error message buffer.
    //

    InitializeSRWLock(&Rtl->SysErrorMessageBufferLock);
    AcquireRtlSysErrorMessageBufferLock(Rtl);

    Rtl->SizeOfSysErrorMessageBufferInBytes = PAGE_SIZE;

    Rtl->SysErrorMessageBuffer = (PCHAR)(
        HeapAlloc(GetProcessHeap(),
                  HEAP_ZERO_MEMORY,
                  Rtl->SizeOfSysErrorMessageBufferInBytes)
    );

    if (!Rtl->SysErrorMessageBuffer) {
        ReleaseRtlSysErrorMessageBufferLock(Rtl);
        return E_OUTOFMEMORY;
    }

    ReleaseRtlSysErrorMessageBufferLock(Rtl);

    //
    // Load the modules.
    //

    Name = (PWSTR *)ModuleNames;
    Module = Rtl->Modules;
    NumberOfModules = GetNumberOfRtlModules(Rtl);

    for (Index = 0; Index < NumberOfModules; Index++, Module++, Name++) {
        *Module = LoadLibraryW(*Name);
        if (!*Module) {
            SYS_ERROR(LoadLibraryW);
            goto Error;
        }
        Rtl->NumberOfModules++;
    }
    ASSERT(Rtl->NumberOfModules == NumberOfModules);

    //
    // Calculate the number of symbols in the function pointer array.
    //

    NumberOfSymbols = sizeof(RTL_FUNCTIONS) / sizeof(ULONG_PTR);
    ASSERT(NumberOfSymbols == ARRAYSIZE(Names));

    //
    // Load the symbols.
    //

    Success = LoadSymbolsFromMultipleModules(Names,
                                             NumberOfSymbols,
                                             (PULONG_PTR)&Rtl->RtlFunctions,
                                             NumberOfSymbols,
                                             Rtl->Modules,
                                             (USHORT)Rtl->NumberOfModules,
                                             &FailedBitmap,
                                             &NumberOfResolvedSymbols);

    if (!Success) {
        goto Error;
    }

    ASSERT(NumberOfSymbols == NumberOfResolvedSymbols);

    Success = CryptAcquireContextW(&Rtl->CryptProv,
                                   NULL,
                                   NULL,
                                   PROV_RSA_FULL,
                                   CRYPT_VERIFYCONTEXT);

    if (!Success) {
        SYS_ERROR(CryptAcquireContextW);
        goto Error;
    }

    //
    // We're done, indicate success and finish up.
    //

    Result = S_OK;
    goto End;

Error:

    Result = E_FAIL;

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}

RTL_RUNDOWN RtlRundown;

_Use_decl_annotations_
VOID
RtlRundown(
    PRTL Rtl
    )
{
    ULONG Index;
    HMODULE *Module;

    if (!ARGUMENT_PRESENT(Rtl)) {
        return;
    }

    if (Rtl->CryptProv) {
        if (!CryptReleaseContext(Rtl->CryptProv, 0)) {
            SYS_ERROR(CryptReleaseContext);
        }
        Rtl->CryptProv = 0;
    }

    //
    // Free any loaded modules.
    //

    Module = Rtl->Modules;

    for (Index = 0; Index < Rtl->NumberOfModules; Index++) {
        if (!FreeLibrary(*Module)) {
            SYS_ERROR(FreeLibrary);
        }
        *Module++ = 0;
    }

    //
    // Free the sys error buffer.
    //

    if (Rtl->SysErrorMessageBuffer) {
        if (!HeapFree(GetProcessHeap(), 0, Rtl->SysErrorMessageBuffer)) {

            //
            // We can't reliably report this with SYS_ERROR() as the buffer is
            // in an unknown state.
            //

            NOTHING;
        }

        Rtl->SysErrorMessageBuffer = NULL;
    }
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
