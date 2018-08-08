/*++

Copyright (c) 2016 Trent Nelson <trent@trent.me>

Module Name:

    RtlErrorHandling.c

Abstract:

    This module is responsible for generic error handling routines.

--*/

#include "stdafx.h"

PRINT_SYS_ERROR RtlPrintSysError;

_Use_decl_annotations_
BOOLEAN
RtlPrintSysError(
    PRTL Rtl,
    PCSZ FunctionName,
    PCSZ FileName,
    ULONG LineNumber
    )
{
    BOOL Success;
    LONG Result1;
    ULONG Result2;
    ULONG Flags;
    PCHAR Buffer;
    PCHAR BaseBuffer;
    PCHAR EndBuffer;
    ULONG LastError;
    ULONG LanguageId;
    ULONG BytesWritten;
    ULONG_PTR BytesToWrite;
    LONG_PTR SizeOfBufferInChars;

    LastError = GetLastError();

    AcquireRtlErrorMessageBufferLock(Rtl);

    Buffer = BaseBuffer = Rtl->ErrorMessageBuffer;
    EndBuffer = Buffer + Rtl->SizeOfErrorMessageBufferInBytes;

    //
    // The following is unnecessary when dealing with bytes, but will allow
    // easy conversion into a WCHAR version at a later date.
    //

    SizeOfBufferInChars = (LONG_PTR)(
        (LONG_PTR)Rtl->SizeOfErrorMessageBufferInBytes *
        (LONG_PTR)sizeof(*Buffer)
    );

    Result1 = Rtl->sprintf_s(Buffer,
                             (ULONG)SizeOfBufferInChars,
                             "%s: %lu: %s failed with error: %lu.  ",
                             FileName,
                             LineNumber,
                             FunctionName,
                             LastError);

    if (Result1 <= 0) {
        OutputDebugStringA("RtlPrintSysError: Rtl->sprintf_s() failed.\n");
        goto Error;
    }

    Buffer += Result1;
    SizeOfBufferInChars -= Result1;

    Flags = (FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS);

    LanguageId = MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),

    Result2 = FormatMessageA(Flags,
                             NULL,
                             LastError,
                             LanguageId,
                             (PSTR)Buffer,
                             (ULONG)SizeOfBufferInChars,
                             NULL);

    if (!Result2) {
        OutputDebugStringA("RtlPrintSysError: FormatMessageA() failed.\n");
        goto Error;
    }

    Buffer += Result2;
    SizeOfBufferInChars -= Result2;

    //
    // We want at least two characters left in the buffer for the \n and
    // trailing NULL.
    //

    ASSERT(SizeOfBufferInChars >= 2);
    ASSERT((ULONG_PTR)Buffer <= (ULONG_PTR)(EndBuffer - 2));

    *Buffer += '\n';
    *Buffer += '\0';

    ASSERT((ULONG_PTR)Buffer <= (ULONG_PTR)EndBuffer);

    BytesToWrite = RtlPointerToOffset(BaseBuffer, Buffer);
    ASSERT(BytesToWrite <= Rtl->SizeOfErrorMessageBufferInBytes);

    Success = WriteFile(Rtl->ErrorOutputHandle,
                        BaseBuffer,
                        (ULONG)BytesToWrite,
                        &BytesWritten,
                        NULL);

    if (!Success) {
        OutputDebugStringA("RtlPrintSysError: WriteFile() failed.\n");
        goto Error;
    }

    //
    // We're done, finish up and return.
    //

    Success = TRUE;
    goto End;

Error:

    Success = FALSE;

    //
    // Intentional follow-on to End.
    //

End:

    SecureZeroMemory(Rtl->ErrorMessageBuffer,
                     Rtl->SizeOfErrorMessageBufferInBytes);

    ReleaseRtlErrorMessageBufferLock(Rtl);

    return Success;
}

INITIALIZE_CRT InitializeCrt;

BOOL
InitializeCrt(
    PRTL Rtl
    )
{
    if (Rtl->Flags.CrtInitialized) {
        return TRUE;
    }

    Rtl->CrtModule = LoadLibraryA("msvcrt.dll");
    if (!Rtl->CrtModule) {
        return FALSE;
    }

    Rtl->Flags.CrtInitialized = TRUE;
    return TRUE;
}


_Use_decl_annotations_
BOOL
InitializeRtl(
    PRTL   Rtl,
    PULONG SizeOfRtl
    )
{
    BOOL Success;
    HANDLE HeapHandle;
    PRTL_LDR_NOTIFICATION_TABLE Table;

    if (!Rtl) {
        if (SizeOfRtl) {
            *SizeOfRtl = sizeof(*Rtl);
        }
        return FALSE;
    }

    if (!SizeOfRtl) {
        return FALSE;
    }

    if (*SizeOfRtl < sizeof(*Rtl)) {
        *SizeOfRtl = sizeof(*Rtl);
        return FALSE;
    } else {
        *SizeOfRtl = sizeof(*Rtl);
    }

    HeapHandle = GetProcessHeap();
    if (!HeapHandle) {
        return FALSE;
    }

    SecureZeroMemory(Rtl, sizeof(*Rtl));

    if (!LoadRtlSymbols(Rtl)) {
        return FALSE;
    }

    Rtl->SizeOfStruct = sizeof(*Rtl);

    SetCSpecificHandler(Rtl->NtdllModule);
    Rtl->__C_specific_handler = __C_specific_handler_impl;
    if (!Rtl->__C_specific_handler) {
        return FALSE;
    }

    Rtl->HeapHandle = HeapHandle;

    if (!LoadRtlExSymbols(NULL, Rtl)) {
        return FALSE;
    }

    if (!InitializeWindowsDirectories(Rtl)) {
        return FALSE;
    }

    if (!InitializeTsx(Rtl)) {
        return FALSE;
    }

    if (!InitializeLargePages(Rtl)) {
        return FALSE;
    }

    Rtl->atexit = atexit_impl;
    Rtl->AtExitEx = AtExitExImpl;
    Rtl->RundownGlobalAtExitFunctions = RundownGlobalAtExitFunctions;

    Rtl->InitializeInjection = InitializeInjection;
    Rtl->Inject = Inject;

    Rtl->InitializeCrt = InitializeCrt;

    Rtl->GetCu = GetCu;

    //
    // Windows 8 onward.
    //

    Rtl->MapViewOfFileExNuma = (PMAP_VIEW_OF_FILE_EX_NUMA)(
        GetProcAddress(
            Rtl->Kernel32Module,
            "MapViewOfFileExNuma"
        )
    );

    //
    // Windows 10 1703 onward.
    //

    Rtl->MapViewOfFileNuma2 = (PMAP_VIEW_OF_FILE_NUMA2)(
        GetProcAddress(
            Rtl->KernelBaseModule,
            "MapViewOfFileNuma2"
        )
    );

    Rtl->TryMapViewOfFileNuma2 = TryMapViewOfFileNuma2;

    Rtl->OutputDebugStringA = OutputDebugStringA;
    Rtl->OutputDebugStringW = OutputDebugStringW;

    Rtl->MaximumFileSectionSize = Rtl->MmGetMaximumFileSectionSize();

    Table = Rtl->LoaderNotificationTable = (PRTL_LDR_NOTIFICATION_TABLE)(
        HeapAlloc(
            HeapHandle,
            HEAP_ZERO_MEMORY,
            sizeof(*Rtl->LoaderNotificationTable)
        )
    );

    if (!Table) {
        return FALSE;
    }

    Success = InitializeRtlLdrNotificationTable(Rtl, Table);
    if (!Success) {
        HeapFree(HeapHandle, 0, Table);
        Rtl->LoaderNotificationTable = NULL;
    }

    Rtl->Multiplicand.QuadPart = TIMESTAMP_TO_SECONDS;
    QueryPerformanceFrequency(&Rtl->Frequency);

    Success = CryptAcquireContextW(&Rtl->CryptProv,
                                   NULL,
                                   NULL,
                                   PROV_RSA_FULL,
                                   CRYPT_VERIFYCONTEXT);

    if (!Success) {
        Rtl->LastError = GetLastError();
        return FALSE;
    }

    Rtl->CryptGenRandom = RtlCryptGenRandom;
    Rtl->CreateEventA = CreateEventA;
    Rtl->CreateEventW = CreateEventW;

    Rtl->InitializeCom = InitializeCom;
    Rtl->LoadDbgEng = LoadDbgEng;
    Rtl->CopyPages = CopyPagesNonTemporalAvx2_v4;
    Rtl->FillPages = FillPagesNonTemporalAvx2_v1;
    Rtl->ProbeForRead = ProbeForRead;

    Rtl->FindAndReplaceByte = FindAndReplaceByte;
    Rtl->MakeRandomString = MakeRandomString;
    Rtl->CreateBuffer = CreateBuffer;
    Rtl->CreateMultipleBuffers = CreateMultipleBuffers;
    Rtl->DestroyBuffer = DestroyBuffer;

    Rtl->SetDllPath = RtlpSetDllPath;
    Rtl->SetInjectionThunkDllPath = RtlpSetInjectionThunkDllPath;
    Rtl->CopyFunction = CopyFunction;

    Rtl->CreateNamedEvent = RtlpCreateNamedEvent;
    Rtl->CreateRandomObjectNames = CreateRandomObjectNames;
    Rtl->CreateSingleRandomObjectName = CreateSingleRandomObjectName;

    Rtl->LoadSymbols = LoadSymbols;
    Rtl->LoadSymbolsFromMultipleModules = LoadSymbolsFromMultipleModules;

    Rtl->InitializeHeapAllocator = RtlHeapAllocatorInitialize;
    Rtl->DestroyHeapAllocator = RtlHeapAllocatorDestroy;

#ifdef _RTL_TEST
    Rtl->TestLoadSymbols = TestLoadSymbols;
    Rtl->TestLoadSymbolsFromMultipleModules = (
        TestLoadSymbolsFromMultipleModules
    );
    Rtl->TestCreateAndDestroyBuffer = TestCreateAndDestroyBuffer;
#endif

    //
    // Error handling.
    //

    Rtl->ErrorOutputHandle = GetStdHandle(STD_ERROR_HANDLE);
    Rtl->PrintSysError = RtlPrintSysError;

    //
    // Create an error message buffer.
    //

    InitializeSRWLock(&Rtl->ErrorMessageBufferLock);

    AcquireRtlErrorMessageBufferLock(Rtl);

    Rtl->SizeOfErrorMessageBufferInBytes = PAGE_SIZE;

    Rtl->ErrorMessageBuffer = (PCHAR)(
        HeapAlloc(HeapHandle,
                  HEAP_ZERO_MEMORY,
                  Rtl->SizeOfErrorMessageBufferInBytes)
    );

    if (!Rtl->ErrorMessageBuffer) {
        Success = FALSE;
    }

    ReleaseRtlErrorMessageBufferLock(Rtl);

    return Success;
}

RTL_API
BOOL
InitializeRtlManually(PRTL Rtl, PULONG SizeOfRtl)
{
    return InitializeRtlManuallyInline(Rtl, SizeOfRtl);
}

_Use_decl_annotations_
VOID
DestroyRtl(
    PPRTL RtlPointer
    )
{
    PRTL Rtl;

    if (!ARGUMENT_PRESENT(RtlPointer)) {
        return;
    }

    Rtl = *RtlPointer;

    if (!ARGUMENT_PRESENT(Rtl)) {
        return;
    }

    //
    // Clear the caller's pointer straight away.
    //

    *RtlPointer = NULL;

    if (Rtl->NtdllModule) {
        FreeLibrary(Rtl->NtdllModule);
        Rtl->NtdllModule = NULL;
    }

    if (Rtl->Kernel32Module) {
        FreeLibrary(Rtl->Kernel32Module);
        Rtl->Kernel32Module = NULL;
    }

    if (Rtl->KernelBaseModule) {
        FreeLibrary(Rtl->KernelBaseModule);
        Rtl->KernelBaseModule = NULL;
    }

    if (Rtl->NtosKrnlModule) {
        FreeLibrary(Rtl->NtosKrnlModule);
        Rtl->NtosKrnlModule = NULL;
    }

    return;
}

VOID
Debugbreak()
{
    __debugbreak();
}

_Use_decl_annotations_
PLIST_ENTRY
RemoveHeadGuardedListTsx(
    PGUARDED_LIST GuardedList
    )
{
    return RemoveHeadGuardedListTsxInline(GuardedList);
}

_Use_decl_annotations_
PLIST_ENTRY
RemoveTailGuardedListTsx(
    PGUARDED_LIST GuardedList
    )
{
    return RemoveTailGuardedListTsxInline(GuardedList);
}

_Use_decl_annotations_
VOID
InsertTailGuardedListTsx(
    PGUARDED_LIST GuardedList,
    PLIST_ENTRY Entry
    )
{
    InsertTailGuardedListTsxInline(GuardedList, Entry);
}

_Use_decl_annotations_
VOID
AppendTailGuardedListTsx(
    PGUARDED_LIST GuardedList,
    PLIST_ENTRY Entry
    )
{
    AppendTailGuardedListTsxInline(GuardedList, Entry);
}

#ifndef VECTORCALL
#define VECTORCALL __vectorcall
#endif

RTL_API
XMMWORD
VECTORCALL
DummyVectorCall1(
    _In_ XMMWORD Xmm0,
    _In_ XMMWORD Xmm1,
    _In_ XMMWORD Xmm2,
    _In_ XMMWORD Xmm3
    )
{
    XMMWORD Temp1;
    XMMWORD Temp2;
    Temp1 = _mm_xor_si128(Xmm0, Xmm1);
    Temp2 = _mm_xor_si128(Xmm2, Xmm3);
    return _mm_xor_si128(Temp1, Temp2);
}

typedef struct _TEST_HVA3 {
    XMMWORD X;
    XMMWORD Y;
    XMMWORD Z;
} TEST_HVA3;

RTL_API
TEST_HVA3
VECTORCALL
DummyHvaCall1(
    _In_ TEST_HVA3 Hva3
    )
{
    Hva3.X = _mm_xor_si128(Hva3.Y, Hva3.Z);
    return Hva3;
}

#if 0
typedef struct _TEST_HFA3 {
    DOUBLE X;
    DOUBLE Y;
    DOUBLE Z;
} TEST_HFA3;

RTL_API
TEST_HFA3
VECTORCALL
DummyHfaCall1(
    _In_ TEST_HFA3 Hfa3
)
{
    __m128d Double;
    Double = _mm_setr_pd(Hfa3.Y, Hfa3.Z);
    Hfa3.X = Double.m128d_f64[0];
    return Hfa3;
}
#endif


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
