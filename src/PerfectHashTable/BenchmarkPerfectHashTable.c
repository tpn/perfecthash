/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    BenchmarkPerfectHashTable.c

Abstract:

    This module implements the benchmark scaffolding for the PerfectHashTable
    component.  Two types of benchmarking activities are supported: "single"
    and "all".  Single is intended to be used when profiling with tools like
    VTune or xperf; it takes a target perfect hash table file, loads it, then
    launches into a loop calling a single benchmark function until a) a number
    of iterations have completed, b) a number of seconds have passed, or c)
    until the user presses Ctrl-C.  No timings are recorded, the sole purpose
    is to get the library to repetitively execute a single routine such that
    performance counters can be captured for the area of interest.

    Benchmarking "all", on the other hand, is geared toward loading either
    a single perfect hash table file, or all files in a directory, running all
    benchmark functions, then saving timings to a CSV file for offline analysis.

--*/

#include "stdafx.h"

//
// We need to define a _fltused ULONG symbol as we're working with floats and
// doubles but not linking to the CRT.  Without this, we'll get a missing
// symbol error during linking.  (It is used by the kernel to know if a given
// routine is using floating point state during trap handling.)
//

ULONG _fltused;

//
// Global variable that provides a convenient way to detect if Ctrl-C is
// pressed.  This is used by the single benchmark functionality.
//

volatile ULONG CtrlCPressed;

//
// Define helper macro for printing error messages to stderr.
//

#define PRINT_ERROR(Name)                    \
    Success = WriteFile(Ctx->StdErrorHandle, \
                        Name##.Buffer,       \
                        Name##.Length,       \
                        NULL,                \
                        NULL);               \
    ASSERT(Success)

#define PRINT_WIDE_ERROR(Name)               \
    Success = WriteFile(Ctx->StdErrorHandle, \
                        Name##.Buffer,       \
                        Name##.Length >> 1,  \
                        NULL,                \
                        NULL);               \
    ASSERT(Success)


//
// Generic benchmark run function.
//

typedef
ULONG
(NTAPI RUN_BENCHMARK)(
    _In_ struct _BENCHMARK_CONTEXT *Ctx
    );
typedef RUN_BENCHMARK *PRUN_BENCHMARK;

RUN_BENCHMARK RunAllBenchmarks;
RUN_BENCHMARK RunSingleBenchmark;

//
// Context initializers.
//

typedef
_Success_(return != 0)
BOOLEAN
(NTAPI INITIALIZE_BENCHMARK_CONTEXT)(
    _Inout_ struct _BENCHMARK_CONTEXT *Ctx
    );
typedef INITIALIZE_BENCHMARK_CONTEXT *PINITIALIZE_BENCHMARK_CONTEXT;

INITIALIZE_BENCHMARK_CONTEXT InitializeBenchmarkContextSingle;
INITIALIZE_BENCHMARK_CONTEXT InitializeBenchmarkContextAll;

//
// Forward definitions of functions that facilitate the Ctrl-C detection and
// time-limit based benchmark running.
//

typedef
BOOL
(NTAPI CTRL_C_HANDLER)(
    _In_opt_ ULONG ControlType
    );
typedef CTRL_C_HANDLER *PCTRL_C_HANDLER;

CTRL_C_HANDLER CtrlCHandler;

typedef
VOID
(CALLBACK CANCEL_SINGLE_FUNCTION_THREADPOOL_CALLBACK)(
    _Inout_ PTP_CALLBACK_INSTANCE Instance,
    _Inout_opt_ PVOID Context
    );
typedef CANCEL_SINGLE_FUNCTION_THREADPOOL_CALLBACK
      *PCANCEL_SINGLE_FUNCTION_THREADPOOL_CALLBACK;

CANCEL_SINGLE_FUNCTION_THREADPOOL_CALLBACK
    CancelSingleFunctionThreadpoolCallback;

//
// Define a benchmark context structure.  This encapsulates all the state
// that would normally live in a standalone executable's main.c file.
//

typedef struct _BENCHMARK_CONTEXT {

    PRTL Rtl;
    PRTL_BOOTSTRAP RtlBootstrap;
    PALLOCATOR Allocator;
    PPERFECT_HASH_TABLE_ANY_API AnyApi;

    PPERFECT_HASH_TABLE_API_EX Api;

    HMODULE RtlModule;
    HMODULE PerfectHashTableModule;

    LARGE_INTEGER Frequency;

    ULONG SizeOfRtl;
    ULONG BytesWritten;
    ULONG CharsWritten;
    ULONG OldCodePage;
    ULONG RunCount;

    PERFECT_HASH_TABLE_BENCHMARK_TYPE BenchmarkType;
    PERFECT_HASH_TABLE_BENCHMARK_FUNCTION_ID BenchmarkFunctionId;

    PCHAR Output;
    PCHAR OutputBuffer;
    HANDLE OutputHandle;
    ULONGLONG OutputBufferSize;

    PWCHAR WideOutput;
    PWCHAR WideOutputBuffer;
    ULONGLONG WideOutputBufferSize;

    ULARGE_INTEGER BytesToWrite;
    LARGE_INTEGER Delay;

    HANDLE StdInputHandle;
    HANDLE StdErrorHandle;
    HANDLE StdOutputHandle;

    //
    // Generic wide string buffer we can use for path construction.
    //

    PWCHAR PathBuffer;
    PWCHAR PathBufferBase;
    ULONGLONG PathBufferSize;

    UNICODE_STRING KeysPath;
    UNICODE_STRING TablePath;
    UNICODE_STRING TestDirectoryPath;

    PSTRING TableName;
    UNICODE_STRING TableNameWide;

    PSTRING KeysName;
    UNICODE_STRING KeysNameWide;

    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_KEYS Keys;

    //
    // Command line glue.
    //

    HMODULE Shell32Module;
    PCOMMAND_LINE_TO_ARGVW CommandLineToArgvW;
    PSTR CommandLineA;
    PWSTR CommandLineW;
    LONG NumberOfArguments;

    PSTR Seconds;
    PPSTR ArgvA;
    PPWSTR ArgvW;

    PRUN_BENCHMARK Run;

    //
    // Stash our huge structures at the end.
    //

    RTL GlobalRtl;
    RTL_BOOTSTRAP GlobalRtlBootstrap;
    ALLOCATOR GlobalAllocator;
    PERFECT_HASH_TABLE_API_EX GlobalApi;

} BENCHMARK_CONTEXT;
typedef BENCHMARK_CONTEXT *PBENCHMARK_CONTEXT;


//
// Main benchmark initialization routine.
//

BOOLEAN
InitializeBenchmarkContext(
    _Inout_ PBENCHMARK_CONTEXT Ctx,
    _In_ PERFECT_HASH_TABLE_BENCHMARK_TYPE BenchmarkType
    )
{
    PRTL Rtl;
    BOOLEAN Success;
    ULONG HeapFlags;
    ULONG NumberOfPages;
    PALLOCATOR Allocator;
    HMODULE Shell32Module = 0;
    PPSTR ArgvA;
    PPWSTR ArgvW;
    PSTR CommandLineA;
    PWSTR CommandLineW;
    LONG NumberOfArguments;
    PCOMMAND_LINE_TO_ARGVW CommandLineToArgvW;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Ctx)) {
        return FALSE;
    }

    if (!IsValidPerfectHashTableBenchmarkType(BenchmarkType)) {
        return FALSE;
    }

    //
    // Clear the entire structure.
    //

    ZeroStructPointer(Ctx);

    //
    // Initialize internal pointers.
    //

    Ctx->SizeOfRtl = sizeof(*Ctx->Rtl);
    Ctx->Rtl = &Ctx->GlobalRtl;
    Ctx->Allocator = &Ctx->GlobalAllocator;
    Ctx->RtlBootstrap = &Ctx->GlobalRtlBootstrap;

    if (!BootstrapRtl(&Ctx->RtlModule, Ctx->RtlBootstrap)) {
        goto Error;
    }

    HeapFlags = HEAP_GENERATE_EXCEPTIONS;
    Success = Ctx->RtlBootstrap->InitializeHeapAllocatorEx(Ctx->Allocator,
                                                           HeapFlags,
                                                           0,
                                                           0);

    if (!Success) {
        goto Error;
    }

    CHECKED_MSG(
        Ctx->RtlBootstrap->InitializeRtl(Ctx->Rtl, &Ctx->SizeOfRtl),
        "InitializeRtl()"
    );

    Rtl = Ctx->Rtl;
    Allocator = Ctx->Allocator;

    ASSERT(Rtl->InitializeCrt(Rtl));
    ASSERT(Rtl->LoadShlwapi(Rtl));

    Ctx->AnyApi = (PPERFECT_HASH_TABLE_ANY_API)&Ctx->GlobalApi;
    Ctx->Api = (PPERFECT_HASH_TABLE_API_EX)&Ctx->GlobalApi;

    ASSERT(LoadPerfectHashTableApi(Rtl,
                                   &Ctx->PerfectHashTableModule,
                                   NULL,
                                   sizeof(Ctx->GlobalApi),
                                   Ctx->AnyApi));


    //
    // Initialize command line glue.
    //

    LOAD_LIBRARY_A(Shell32Module, Shell32);

    RESOLVE_FUNCTION(CommandLineToArgvW,
                     Shell32Module,
                     PCOMMAND_LINE_TO_ARGVW,
                     CommandLineToArgvW);

    CHECKED_MSG(CommandLineW = GetCommandLineW(), "GetCommandLineW()");
    CHECKED_MSG(CommandLineA = GetCommandLineA(), "GetCommandLineA()");

    ArgvW = CommandLineToArgvW(CommandLineW, &NumberOfArguments);

    CHECKED_MSG(ArgvW, "Shell32!CommandLineToArgvW()");

    CHECKED_MSG(
        Rtl->ArgvWToArgvA(
            ArgvW,
            NumberOfArguments,
            &ArgvA,
            NULL,
            Allocator
        ),
        "Rtl!ArgvWToArgA"
    );

    //
    // Copy variables over to context structure.
    //

    Ctx->ArgvA = ArgvA;
    Ctx->ArgvW = ArgvW;
    Ctx->NumberOfArguments = NumberOfArguments;
    Ctx->CommandLineA = CommandLineA;
    Ctx->CommandLineW = CommandLineW;

    Ctx->OldCodePage = GetConsoleCP();
    ASSERT(SetConsoleCP(20127));

    Ctx->StdInputHandle = GetStdHandle(STD_INPUT_HANDLE);
    Ctx->StdErrorHandle = GetStdHandle(STD_ERROR_HANDLE);
    Ctx->StdOutputHandle = GetStdHandle(STD_OUTPUT_HANDLE);

    //
    // Initialize buffers.
    //

    Success = Rtl->CreateBuffer(Rtl,
                                NULL,
                                10,
                                0,
                                &Ctx->OutputBufferSize,
                                &Ctx->OutputBuffer);

    if (!Success) {
        __debugbreak();
        goto End;
    }

    Ctx->Output = Ctx->OutputBuffer;

    //
    // Wide char buffer.
    //

    Success = Rtl->CreateBuffer(Rtl,
                                NULL,
                                10,
                                0,
                                &Ctx->WideOutputBufferSize,
                                &Ctx->WideOutputBuffer);

    if (!Success) {
        __debugbreak();
        goto End;
    }

    Ctx->WideOutput = Ctx->WideOutputBuffer;

    //
    // Create a buffer we can use for temporary path construction.  We want it
    // to be MAX_USHORT in size, so (1 << 16) >> PAGE_SHIFT converts this into
    // the number of pages we need.
    //

    NumberOfPages = (1 << 16) >> PAGE_SHIFT;
    Success = Rtl->CreateBuffer(Rtl,
                                NULL,
                                NumberOfPages,
                                NULL,
                                &Ctx->PathBufferSize,
                                &Ctx->PathBufferBase);

    if (!Success) {
        return FALSE;
    }

    Ctx->PathBuffer = Ctx->PathBufferBase;

    Ctx->BenchmarkType = BenchmarkType;

    switch (BenchmarkType) {

        case PerfectHashTableSingleBenchmarkType:

            Success = InitializeBenchmarkContextSingle(Ctx);

            break;

        case PerfectHashTableAllBenchmarkType:

            Success = InitializeBenchmarkContextAll(Ctx);

            break;

        default:

            //
            // Should be unreachable as we've validated the benchmark type.
            //

            ASSERT(FALSE);
            goto Error;
    }

    if (!Success) {
        goto Error;
    }

    goto End;

Error:

    Success = FALSE;

    //
    // Intentional follow-on to End.
    //

End:

    return Success;
}


ULONG
PerfectHashTableBenchmarkMainCommon(
    PERFECT_HASH_TABLE_BENCHMARK_TYPE BenchmarkType
    )
{
    ULONG ExitCode;
    BENCHMARK_CONTEXT Context;
    PBENCHMARK_CONTEXT Ctx;

    Ctx = &Context;

    if (!InitializeBenchmarkContext(Ctx, BenchmarkType)) {
        goto Error;
    }

    if (!Ctx->Run(Ctx)) {
        goto Error;
    }

    ExitCode = 0;

    goto End;

Error:

    ExitCode = 1;

    //
    // Intentional follow-on to End.
    //

End:

    if (GetConsoleCP() != Ctx->OldCodePage) {
        SetConsoleCP(Ctx->OldCodePage);
        Ctx->OldCodePage = 0;
    }

    return ExitCode;
}

////////////////////////////////////////////////////////////////////////////////
// Benchmark Single Functionality
////////////////////////////////////////////////////////////////////////////////

ULONG
PerfectHashTableBenchmarkSingleMain(
    VOID
    )
{
    PERFECT_HASH_TABLE_BENCHMARK_TYPE BenchmarkType;

    BenchmarkType = PerfectHashTableSingleBenchmarkType;

    return PerfectHashTableBenchmarkMainCommon(BenchmarkType);
}

_Use_decl_annotations_
BOOLEAN
InitializeBenchmarkContextSingle(
    PBENCHMARK_CONTEXT Ctx
    )
{
    PRTL Rtl;
    PALLOCATOR Allocator;
    BOOL Result;
    ULONG RunCount;
    NTSTATUS Status;
    BOOLEAN Success;
    PWCHAR WideOutput;
    PWCHAR WideOutputBuffer;
    ULONG BytesWritten;
    ULONG WideCharsWritten;
    HANDLE WideOutputHandle;
    ULARGE_INTEGER BytesToWrite;
    ULARGE_INTEGER WideCharsToWrite;
    PERFECT_HASH_TABLE_BENCHMARK_FUNCTION_ID Id;
    PPERFECT_HASH_TABLE_API_EX Api;

    const UNICODE_STRING InvalidBenchmarkFunctionId = RTL_CONSTANT_STRING(
        L"Invalid benchmark function ID.\n"
    );

    const UNICODE_STRING Usage = RTL_CONSTANT_STRING(
        L"Usage: PerfectHashTableBenchmarkSingle.exe "
        L"<OriginalKeysFile (must be fully-qualified)> "
        L"<SinglePerfectHashTableFile (must be fully-qualified)> "
        L"<BenchmarkFunctionId> "
        L"[<RunCount>|<Seconds>s] "
        L"E.g.: PerfectHashTableBenchmarkSingle.exe "
        L"S:\\perfecthash\\data\\mshtml-37209.pht1 "
        L"1 10s\n\n"
    );

    Rtl = Ctx->Rtl;
    Allocator = Ctx->Allocator;
    WideOutput = Ctx->WideOutput;
    WideOutputBuffer = Ctx->WideOutputBuffer;
    WideOutputHandle = Ctx->StdErrorHandle;
    Api = Ctx->Api;

    switch (Ctx->NumberOfArguments) {

        case 4:
        case 5:

            //
            // Extract benchmark function ID.
            //

            CHECKED_NTSTATUS_MSG(
                Rtl->RtlCharToInteger(
                    Ctx->ArgvA[3],
                    10,
                    (PULONG)&Id
                ),
                "Rtl->RtlCharToInteger(ArgvA[3])"
            );

            if (!IsValidPerfectHashTableBenchmarkFunctionId(Id)) {
                PRINT_ERROR(InvalidBenchmarkFunctionId);
                goto PrintUsage;
            }

            Ctx->BenchmarkFunctionId = Id;

            //
            // Extract keys path.
            //

            Ctx->KeysPath.Buffer = Ctx->ArgvW[1];
            Ctx->KeysPath.Length = (USHORT)wcslen(Ctx->KeysPath.Buffer) << 1;
            Ctx->KeysPath.MaximumLength = (
                Ctx->KeysPath.Length +
                sizeof(Ctx->KeysPath.Buffer[0])
            );

            Success = Api->LoadPerfectHashTableKeys(Rtl,
                                                    Allocator,
                                                    &Ctx->KeysPath,
                                                    &Ctx->Keys);

            if (!Success) {

                WIDE_OUTPUT_RAW(WideOutput, L"Failed to load keys for ");
                WIDE_OUTPUT_UNICODE_STRING(WideOutput, &Ctx->KeysPath);
                WIDE_OUTPUT_RAW(WideOutput, L".\n");
                WIDE_OUTPUT_FLUSH();
                goto Error;

            }

            InitializeUnicodeStringFromUnicodeString(&Ctx->KeysNameWide,
                                                     &Ctx->KeysPath);

            Rtl->PathStripPathW(Ctx->KeysNameWide.Buffer);
            Ctx->KeysNameWide.Length = (USHORT)(
                wcslen(Ctx->KeysNameWide.Buffer) << 1
            );

            ASSERT(
                Ctx->KeysNameWide.Buffer[Ctx->KeysNameWide.Length] == L'\0'
            );
            Ctx->KeysNameWide.MaximumLength = (
                sizeof(WCHAR) +
                Ctx->KeysNameWide.Length
            );

            Success = ConvertUtf16StringToUtf8String(&Ctx->KeysNameWide,
                                                     &Ctx->KeysName,
                                                     Allocator);
            ASSERT(Success);

            //
            // Extract table path.
            //

            Ctx->TablePath.Buffer = Ctx->ArgvW[2];
            Ctx->TablePath.Length = (USHORT)(
                wcslen(Ctx->TablePath.Buffer) << 1
            );
            Ctx->TablePath.MaximumLength = (
                Ctx->TablePath.Length +
                sizeof(Ctx->TablePath.Buffer[0])
            );

            //
            // Load the table with the keys.
            //

            Success = Api->LoadPerfectHashTable(Rtl,
                                                Allocator,
                                                Ctx->Keys,
                                                &Ctx->TablePath,
                                                &Ctx->Table);


            if (!Success) {

                WIDE_OUTPUT_RAW(WideOutput, L"Failed to load perfect "
                                            L"hash table: ");
                WIDE_OUTPUT_UNICODE_STRING(WideOutput, &Ctx->TablePath);
                WIDE_OUTPUT_RAW(WideOutput, L".\n");
                WIDE_OUTPUT_FLUSH();

                goto Error;
            }

            //
            // Perform the same name extraction on the table path as we did
            // for the keys path.
            //

            InitializeUnicodeStringFromUnicodeString(&Ctx->TableNameWide,
                                                     &Ctx->TablePath);

            Rtl->PathStripPathW(Ctx->TableNameWide.Buffer);
            Ctx->TableNameWide.Length = (USHORT)(
                wcslen(Ctx->TableNameWide.Buffer) << 1
            );

            ASSERT(
                Ctx->TableNameWide.Buffer[Ctx->TableNameWide.Length] == L'\0'
            );
            Ctx->TableNameWide.MaximumLength = (
                sizeof(WCHAR) +
                Ctx->TableNameWide.Length
            );

            Success = ConvertUtf16StringToUtf8String(&Ctx->TableNameWide,
                                                     &Ctx->TableName,
                                                     Allocator);
            ASSERT(Success);

            //
            // Test the table before proceeding.
            //

            Success = Api->TestPerfectHashTable(Ctx->Table, TRUE);

            if (!Success) {

                WIDE_OUTPUT_RAW(WideOutput, L"Test failed for perfect "
                                            L"hash table loaded from disk: ");
                WIDE_OUTPUT_UNICODE_STRING(WideOutput, &Ctx->TablePath);
                WIDE_OUTPUT_RAW(WideOutput, L".\n");
                WIDE_OUTPUT_FLUSH();
                goto Error;
            }

            //
            // Extract optional run count or seconds.
            //

            if (Ctx->NumberOfArguments != 5) {

                RunCount = 0;

            } else {

                ULONG Number;
                BOOLEAN UseSeconds = FALSE;
                PCHAR Char = Ctx->ArgvA[4];

                //
                // Advance to the NULL terminator, and then back one.
                //

                while (*++Char);
                Char--;

                //
                // Check to see if the last character is 's', implying seconds.
                //

                if (*Char == 's') {

                    //
                    // Make a note that seconds have been requested, then clear
                    // the character with a NULL, such that RtlCharToInteger
                    // will work.
                    //

                    UseSeconds = TRUE;
                    *Char = '\0';
                }

                CHECKED_NTSTATUS_MSG(
                    Rtl->RtlCharToInteger(
                        Ctx->ArgvA[3],
                        10,
                        &Number
                    ),
                    "Rtl->RtlCharToInteger(ArgvA[3])"
                );

                if (UseSeconds) {
                    PTP_SIMPLE_CALLBACK Callback;

                    Ctx->Seconds = Ctx->ArgvA[3];
                    *Char = 's';
                    Callback = CancelSingleFunctionThreadpoolCallback;

                    Result = TrySubmitThreadpoolCallback(Callback,
                                                         &Number,
                                                         NULL);
                    ASSERT(Result);

                    RunCount = 0;
                }

            }

            Ctx->RunCount = RunCount;

            break;

PrintUsage:

        default:
            PRINT_ERROR(Usage);
            return FALSE;
    }

    Ctx->Run = RunSingleBenchmark;
    return TRUE;

Error:

    return FALSE;
}

_Use_decl_annotations_
ULONG
RunSingleBenchmark(
    PBENCHMARK_CONTEXT Ctx
    )
{
    ULONG Key;
    PULONG Keys;
    PCHAR Seconds;
    ULONG RunCount;
    PPERFECT_HASH_TABLE Table;

    RunCount = Ctx->RunCount;
    Seconds = Ctx->Seconds;

    CtrlCPressed = 0;

    Table = Ctx->Table;
    Keys = Table->Keys->Keys;
    Key = Keys[0];

    return 0;
}

_Use_decl_annotations_
BOOL
CtrlCHandler(
    ULONG ControlType
    )
{
    if (ControlType == CTRL_C_EVENT) {
        CtrlCPressed = 1;
        return TRUE;
    }
    return FALSE;
}

_Use_decl_annotations_
VOID
CALLBACK
CancelSingleFunctionThreadpoolCallback(
    PTP_CALLBACK_INSTANCE Instance,
    PVOID Context
    )
{
    ULONG Seconds = *((PULONG)Context);
    ULONG Milliseconds = Seconds * 1000;

    //
    // Detach from the threadpool.
    //

    DisassociateCurrentThreadFromCallback(Instance);

    //
    // Simulate pressing Ctrl-C after an elapsed time.
    //

    SleepEx(Milliseconds, TRUE);
    CtrlCPressed = 1;
}

////////////////////////////////////////////////////////////////////////////////
// Benchmark All Functionality
////////////////////////////////////////////////////////////////////////////////

ULONG
PerfectHashTableBenchmarkAllMain(
    VOID
    )
{
    PERFECT_HASH_TABLE_BENCHMARK_TYPE BenchmarkType;

    BenchmarkType = PerfectHashTableAllBenchmarkType;

    return PerfectHashTableBenchmarkMainCommon(BenchmarkType);
}

BOOLEAN
InitializeBenchmarkContextAll(
    _Inout_ PBENCHMARK_CONTEXT Ctx
    )
{
    Ctx->Run = RunAllBenchmarks;
    return TRUE;
}


_Use_decl_annotations_
ULONG
RunAllBenchmarks(
    PBENCHMARK_CONTEXT Ctx
    )
{
    return 0;
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
