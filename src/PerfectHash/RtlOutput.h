/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    RtlOutput.h

Abstract:

    Helper methods for various string writing functionality.

--*/

#pragma once

#include "stdafx.h"

static CONST CHAR IntegerToCharTable[] = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F'
};

static CONST WCHAR IntegerToWCharTable[] = {
    L'0', L'1', L'2', L'3', L'4', L'5', L'6', L'7', L'8', L'9',
    L'A', L'B', L'C', L'D', L'E', L'F'
};


FORCEINLINE
BYTE
CountNumberOfDigitsInline(_In_ ULONG Value)
{
    BYTE Count = 0;

    do {
        Count++;
        Value = Value / 10;
    } while (Value != 0);

    return Count;
}

FORCEINLINE
BYTE
CountNumberOfLongLongDigitsInline(_In_ ULONGLONG Value)
{
    BYTE Count = 0;

    do {
        Count++;
        Value = Value / 10;
    } while (Value != 0);

    return Count;
}

//
// Helper string routines for buffer manipulation.
//

typedef
BOOLEAN
(NTAPI APPEND_INTEGER_TO_UNICODE_STRING)(
    _In_ PUNICODE_STRING String,
    _In_ ULONG Integer,
    _In_ USHORT NumberOfDigits,
    _In_opt_ WCHAR Trailer
    );
typedef APPEND_INTEGER_TO_UNICODE_STRING *PAPPEND_INTEGER_TO_UNICODE_STRING;

typedef
BOOLEAN
(NTAPI APPEND_LONGLONG_INTEGER_TO_UNICODE_STRING)(
    _In_ PUNICODE_STRING String,
    _In_ ULONGLONG Integer,
    _In_ USHORT NumberOfDigits,
    _In_opt_ WCHAR Trailer
    );
typedef APPEND_LONGLONG_INTEGER_TO_UNICODE_STRING
      *PAPPEND_LONGLONG_INTEGER_TO_UNICODE_STRING;

typedef
VOID
(NTAPI APPEND_INTEGER_TO_CHAR_BUFFER)(
    _Inout_ PCHAR *BufferPointer,
    _In_ ULONGLONG Integer
    );
typedef APPEND_INTEGER_TO_CHAR_BUFFER *PAPPEND_INTEGER_TO_CHAR_BUFFER;

typedef
VOID
(NTAPI APPEND_INTEGER_TO_CHAR_BUFFER_AS_HEX)(
    _Inout_ PCHAR *BufferPointer,
    _In_opt_ ULONG Integer
    );
typedef APPEND_INTEGER_TO_CHAR_BUFFER_AS_HEX
      *PAPPEND_INTEGER_TO_CHAR_BUFFER_AS_HEX;

typedef
VOID
(NTAPI APPEND_INTEGER_TO_CHAR_BUFFER_EX)(
    _Inout_ PCHAR *BufferPointer,
    _In_ ULONGLONG Integer,
    _In_ BYTE NumberOfDigits,
    _In_ CHAR Pad,
    _In_opt_ CHAR Trailer
    );
typedef APPEND_INTEGER_TO_CHAR_BUFFER_EX *PAPPEND_INTEGER_TO_CHAR_BUFFER_EX;

typedef
VOID
(NTAPI APPEND_STRING_TO_CHAR_BUFFER)(
    _Inout_ PCHAR *BufferPointer,
    _In_ PCSTRING String
    );
typedef APPEND_STRING_TO_CHAR_BUFFER *PAPPEND_STRING_TO_CHAR_BUFFER;

typedef
VOID
(NTAPI APPEND_CHAR_BUFFER_TO_CHAR_BUFFER)(
    _Inout_ PCHAR *BufferPointer,
    _In_ PCCHAR String,
    _In_ ULONG SizeInBytes
    );
typedef APPEND_CHAR_BUFFER_TO_CHAR_BUFFER *PAPPEND_CHAR_BUFFER_TO_CHAR_BUFFER;

typedef
VOID
(NTAPI APPEND_CHAR_TO_CHAR_BUFFER)(
    _Inout_ PCHAR *BufferPointer,
    _In_ CHAR Char
    );
typedef APPEND_CHAR_TO_CHAR_BUFFER *PAPPEND_CHAR_TO_CHAR_BUFFER;

typedef
VOID
(NTAPI APPEND_CSTR_TO_CHAR_BUFFER)(
    _Inout_ PCHAR *BufferPointer,
    _In_ PCSZ String
    );
typedef APPEND_CSTR_TO_CHAR_BUFFER *PAPPEND_CSTR_TO_CHAR_BUFFER;

//
// Wide character versions.
//

typedef
VOID
(NTAPI APPEND_INTEGER_TO_WIDE_CHAR_BUFFER)(
    _Inout_ PWCHAR *BufferPointer,
    _In_ ULONGLONG Integer
    );
typedef APPEND_INTEGER_TO_WIDE_CHAR_BUFFER *PAPPEND_INTEGER_TO_WIDE_CHAR_BUFFER;

typedef
VOID
(NTAPI APPEND_INTEGER_TO_WIDE_CHAR_BUFFER_EX)(
    _Inout_ PWCHAR *BufferPointer,
    _In_ ULONGLONG Integer,
    _In_ BYTE NumberOfDigits,
    _In_ WCHAR Pad,
    _In_opt_ WCHAR Trailer
    );
typedef APPEND_INTEGER_TO_WIDE_CHAR_BUFFER_EX
      *PAPPEND_INTEGER_TO_WIDE_CHAR_BUFFER_EX;

typedef
VOID
(NTAPI APPEND_UNICODE_STRING_TO_WIDE_CHAR_BUFFER)(
    _Inout_ PWCHAR *BufferPointer,
    _In_ PCUNICODE_STRING UnicodeString
    );
typedef APPEND_UNICODE_STRING_TO_WIDE_CHAR_BUFFER
      *PAPPEND_UNICODE_STRING_TO_WIDE_CHAR_BUFFER;

typedef
VOID
(NTAPI APPEND_WIDE_CHAR_BUFFER_TO_WIDE_CHAR_BUFFER)(
    _Inout_ PWCHAR *BufferPointer,
    _In_ PCWCHAR String,
    _In_ ULONG SizeInBytes
    );
typedef APPEND_WIDE_CHAR_BUFFER_TO_WIDE_CHAR_BUFFER
      *PAPPEND_WIDE_CHAR_BUFFER_TO_WIDE_CHAR_BUFFER;

typedef
VOID
(NTAPI APPEND_WIDE_CHAR_TO_WIDE_CHAR_BUFFER)(
    _Inout_ PWCHAR *BufferPointer,
    _In_ WCHAR Char
    );
typedef APPEND_WIDE_CHAR_TO_WIDE_CHAR_BUFFER
      *PAPPEND_WIDE_CHAR_TO_WIDE_CHAR_BUFFER;

typedef
VOID
(NTAPI APPEND_WIDE_CSTR_TO_WIDE_CHAR_BUFFER)(
    _Inout_ PWCHAR *BufferPointer,
    _In_ PCWSZ String
    );
typedef APPEND_WIDE_CSTR_TO_WIDE_CHAR_BUFFER
      *PAPPEND_WIDE_CSTR_TO_WIDE_CHAR_BUFFER;

extern APPEND_INTEGER_TO_UNICODE_STRING AppendIntegerToUnicodeString;
extern APPEND_LONGLONG_INTEGER_TO_UNICODE_STRING
    AppendLongLongIntegerToUnicodeString;
extern APPEND_INTEGER_TO_CHAR_BUFFER AppendIntegerToCharBuffer;
extern APPEND_INTEGER_TO_CHAR_BUFFER_AS_HEX AppendIntegerToCharBufferAsHex;
extern APPEND_INTEGER_TO_CHAR_BUFFER_EX AppendIntegerToCharBufferEx;
extern APPEND_STRING_TO_CHAR_BUFFER AppendStringToCharBuffer;
extern APPEND_CHAR_BUFFER_TO_CHAR_BUFFER AppendCharBufferToCharBuffer;
extern APPEND_CSTR_TO_CHAR_BUFFER AppendCStrToCharBuffer;
extern APPEND_CHAR_TO_CHAR_BUFFER AppendCharToCharBuffer;
extern APPEND_UNICODE_STRING_TO_WIDE_CHAR_BUFFER
    AppendUnicodeStringToWideCharBuffer;
extern APPEND_WIDE_CSTR_TO_WIDE_CHAR_BUFFER AppendWideCStrToWideCharBuffer;
extern APPEND_WIDE_CHAR_BUFFER_TO_WIDE_CHAR_BUFFER
    AppendWideCharBufferToWideCharBuffer;
extern APPEND_WIDE_CHAR_TO_WIDE_CHAR_BUFFER AppendWideCharToWideCharBuffer;
extern APPEND_INTEGER_TO_WIDE_CHAR_BUFFER AppendIntegerToWideCharBuffer;

//
// Output helpers.
//

static PCSZ Dot = ".";
static PCSZ Dash = "-";
static PCSZ Cross = "x";
static PCSZ Newline = "\n";

#define DO_OUTPUT(Buffer, Size)                                        \
    if (!WriteFile(OutputHandle, Buffer, Size, &BytesWritten, NULL)) { \
        Result = PH_E_SYSTEM_CALL_FAILED;                              \
        SYS_ERROR(WriteFile);                                          \
        goto Error;                                                    \
    } else if (BytesWritten != Size) {                                 \
        Result = PH_E_NOT_ALL_BYTES_WRITTEN;                           \
        PH_ERROR(__FUNCTION__, Result);                                \
        goto Error;                                                    \
    }

#define MAYBE_OUTPUT(Buffer, Size) \
    if (OutputHandle) {            \
        DO_OUTPUT(Buffer, Size);   \
    }

#define DOT() DO_OUTPUT(Dot, 1)
#define DASH() DO_OUTPUT(Dash, 1)
#define CROSS() DO_OUTPUT(Cross, 1)
#define NEWLINE() DO_OUTPUT(Newline, 1)

#define MAYBE_DOT() MAYBE_OUTPUT(Dot, 1)
#define MAYBE_DASH() MAYBE_OUTPUT(Dash, 1)
#define MAYBE_CROSS() MAYBE_OUTPUT(Cross, 1)
#define MAYBE_NEWLINE() MAYBE_OUTPUT(Newline, 1)

#define OUTPUT_RAW(String)                                          \
    AppendCharBufferToCharBuffer(&Output, String, sizeof(String)-1)

#define OUTPUT_HEX(Integer) AppendIntegerToCharBufferAsHex(&Output, Integer)

#define OUTPUT_STRING(String) AppendStringToCharBuffer(&Output, String)

#define OUTPUT_CSTR(Str) AppendCStrToCharBuffer(&Output, Str)
#define OUTPUT_CHR(Char) AppendCharToCharBuffer(&Output, Char)
#define OUTPUT_SEP() AppendCharToCharBuffer(&Output, ',')
#define OUTPUT_LF() AppendCharToCharBuffer(&Output, '\n')

#define OUTPUT_INT(Value)                      \
    AppendIntegerToCharBuffer(&Output, Value);

#define OUTPUT_FLUSH_CONSOLE()                                               \
    BytesToWrite.QuadPart = ((ULONG_PTR)Output) - ((ULONG_PTR)OutputBuffer); \
    Success = WriteConsoleA(OutputHandle,                                    \
                            OutputBuffer,                                    \
                            BytesToWrite.LowPart,                            \
                            &CharsWritten,                                   \
                            NULL);                                           \
    ASSERT(Success);                                                         \
    Output = OutputBuffer

#define OUTPUT_FLUSH_FILE()                                                    \
    BytesToWrite.QuadPart = ((ULONG_PTR)Output) - ((ULONG_PTR)OutputBuffer)-1; \
    Success = WriteFile(OutputHandle,                                          \
                        OutputBuffer,                                          \
                        BytesToWrite.LowPart,                                  \
                        &BytesWritten,                                         \
                        NULL);                                                 \
    ASSERT(Success);                                                           \
    Output = OutputBuffer

#define OUTPUT_FLUSH()                                                       \
    BytesToWrite.QuadPart = ((ULONG_PTR)Output) - ((ULONG_PTR)OutputBuffer); \
    Success = WriteConsoleA(OutputHandle,                                    \
                            OutputBuffer,                                    \
                            BytesToWrite.LowPart,                            \
                            &CharsWritten,                                   \
                            NULL);                                           \
    if (!Success) {                                                          \
        Success = WriteFile(OutputHandle,                                    \
                            OutputBuffer,                                    \
                            BytesToWrite.LowPart,                            \
                            &BytesWritten,                                   \
                            NULL);                                           \
        ASSERT(Success);                                                     \
    }                                                                        \
    Output = OutputBuffer

//
// Wide output helpers.
//

#define WIDE_OUTPUT_RAW(WideOutput, WideString)            \
    AppendWideCharBufferToWideCharBuffer(                  \
        &WideOutput,                                       \
        WideString,                                        \
        (USHORT)(sizeof(WideString)-sizeof(WideString[0])) \
    )

#define WIDE_OUTPUT_UNICODE_STRING(WideOutput, UnicodeString)       \
    AppendUnicodeStringToWideCharBuffer(&WideOutput, UnicodeString)

#define WIDE_OUTPUT_WCSTR(WideOutput, WideCStr)           \
    AppendWideCStrToWideCharBuffer(&WideOutput, WideCStr)

#define WIDE_OUTPUT_WCHR(WideOutput, WideChar)            \
    AppendWideCharToWideCharBuffer(&WideOutput, WideChar)

#define WIDE_OUTPUT_SEP(WideOutput)                   \
    AppendWideCharToWideCharBuffer(&WideOutput, L',')

#define WIDE_OUTPUT_LF(WideOutput)                     \
    AppendWideCharToWideCharBuffer(&WideOutput, L'\n')

#define WIDE_OUTPUT_INT(WideOutput, Value)             \
    AppendIntegerToWideCharBuffer(&WideOutput, Value);

#define WIDE_OUTPUT_FLUSH_CONSOLE()                         \
    BytesToWrite.QuadPart = (                               \
        ((ULONG_PTR)WideOutput) -                           \
        ((ULONG_PTR)WideOutputBuffer)                       \
    );                                                      \
    WideCharsToWrite.QuadPart = BytesToWrite.QuadPart >> 1; \
    Success = WriteConsoleW(WideOutputHandle,               \
                            WideOutputBuffer,               \
                            WideCharsToWrite.LowPart,       \
                            &WideCharsWritten,              \
                            NULL);                          \
    ASSERT(Success);                                        \
    WideOutput = WideOutputBuffer

#define WIDE_OUTPUT_FLUSH_FILE()              \
    BytesToWrite.QuadPart = (                 \
        ((ULONG_PTR)WideOutput) -             \
        ((ULONG_PTR)WideOutputBuffer)         \
    ) - 1;                                    \
    Success = WriteFile(WideOutputHandle,     \
                        WideOutputBuffer,     \
                        BytesToWrite.LowPart, \
                        &BytesWritten,        \
                        NULL);                \
    ASSERT(Success);                          \
    WideOutput = WideOutputBuffer

#define WIDE_OUTPUT_FLUSH()                                 \
    BytesToWrite.QuadPart = (                               \
        ((ULONG_PTR)WideOutput) -                           \
        ((ULONG_PTR)WideOutputBuffer)                       \
    );                                                      \
    WideCharsToWrite.QuadPart = BytesToWrite.QuadPart >> 1; \
    Success = WriteConsoleW(WideOutputHandle,               \
                            WideOutputBuffer,               \
                            WideCharsToWrite.LowPart,       \
                            &WideCharsWritten,              \
                            NULL);                          \
    if (!Success) {                                         \
        Success = WriteFile(WideOutputHandle,               \
                            WideOutputBuffer,               \
                            BytesToWrite.LowPart,           \
                            &BytesWritten,                  \
                            NULL);                          \
        ASSERT(Success);                                    \
    }                                                       \
    WideOutput = WideOutputBuffer

//
// Timestamp glue for benchmarking.
//

#define TIMESTAMP_TO_MICROSECONDS 1000000ULL
#define TIMESTAMP_TO_NANOSECONDS  1000000000ULL

typedef struct _TIMESTAMP {
    ULONGLONG Id;
    ULONGLONG Count;
    STRING Name;
    union {
        ULONG Aux;
        ULONG CpuId[4];
    };
    LARGE_INTEGER Start;
    LARGE_INTEGER End;
    ULARGE_INTEGER StartTsc;
    ULARGE_INTEGER EndTsc;
    ULARGE_INTEGER Tsc;
    ULARGE_INTEGER TotalTsc;
    ULARGE_INTEGER MinimumTsc;
    ULARGE_INTEGER MaximumTsc;
    ULARGE_INTEGER Cycles;
    ULARGE_INTEGER MinimumCycles;
    ULARGE_INTEGER MaximumCycles;
    ULARGE_INTEGER TotalCycles;
    ULARGE_INTEGER Nanoseconds;
    ULARGE_INTEGER TotalNanoseconds;
    ULARGE_INTEGER MinimumNanoseconds;
    ULARGE_INTEGER MaximumNanoseconds;
} TIMESTAMP;
typedef TIMESTAMP *PTIMESTAMP;

#define INIT_TIMESTAMP(Idx, Namex)                         \
    ZeroStructPointer(Timestamp);                          \
    Timestamp->Id = Idx;                                   \
    Timestamp->Name.Length = sizeof(Namex)-1;              \
    Timestamp->Name.MaximumLength = sizeof(Namex);         \
    Timestamp->Name.Buffer = Namex;                        \
    Timestamp->TotalTsc.QuadPart = 0;                      \
    Timestamp->TotalCycles.QuadPart = 0;                   \
    Timestamp->TotalNanoseconds.QuadPart = 0;              \
    Timestamp->MinimumTsc.QuadPart = (ULONGLONG)-1;        \
    Timestamp->MinimumCycles.QuadPart = (ULONGLONG)-1;     \
    Timestamp->MinimumNanoseconds.QuadPart = (ULONGLONG)-1

#define INIT_TIMESTAMP_FROM_STRING(Idx, String)            \
    ZeroStructPointer(Timestamp);                          \
    Timestamp->Id = Idx;                                   \
    Timestamp->Name.Length = String->Length;               \
    Timestamp->Name.MaximumLength = String->MaximumLength; \
    Timestamp->Name.Buffer = String->Buffer;               \
    Timestamp->TotalTsc.QuadPart = 0;                      \
    Timestamp->TotalCycles.QuadPart = 0;                   \
    Timestamp->TotalNanoseconds.QuadPart = 0;              \
    Timestamp->MinimumTsc.QuadPart = (ULONGLONG)-1;        \
    Timestamp->MinimumCycles.QuadPart = (ULONGLONG)-1;     \
    Timestamp->MinimumNanoseconds.QuadPart = (ULONGLONG)-1

#define RESET_TIMESTAMP()                                   \
    Timestamp->Count = 0;                                   \
    Timestamp->TotalTsc.QuadPart = 0;                       \
    Timestamp->TotalCycles.QuadPart = 0;                    \
    Timestamp->TotalNanoseconds.QuadPart = 0;               \
    Timestamp->MinimumTsc.QuadPart = (ULONGLONG)-1;         \
    Timestamp->MaximumTsc.QuadPart = 0;                     \
    Timestamp->MinimumCycles.QuadPart = (ULONGLONG)-1;      \
    Timestamp->MaximumCycles.QuadPart = 0;                  \
    Timestamp->MinimumNanoseconds.QuadPart = (ULONGLONG)-1; \
    Timestamp->MaximumNanoseconds.QuadPart = 0

#define START_TIMESTAMP_CPUID()                 \
    ++Timestamp->Count;                         \
    QueryPerformanceCounter(&Timestamp->Start); \
    __cpuid((PULONG)&Timestamp->CpuId, 0);      \
    Timestamp->StartTsc.QuadPart = __rdtsc()

#define START_TIMESTAMP_RDTSCP()                             \
    ++Timestamp->Count;                                      \
    QueryPerformanceCounter(&Timestamp->Start);              \
    Timestamp->StartTsc.QuadPart = __rdtscp(&Timestamp->Aux)

#define START_TIMESTAMP_RDTSC()                 \
    ++Timestamp->Count;                         \
    QueryPerformanceCounter(&Timestamp->Start); \
    Timestamp->StartTsc.QuadPart = __rdtsc()

#define END_TIMESTAMP_COMMON()                             \
    Timestamp->Tsc.QuadPart = (                            \
        Timestamp->EndTsc.QuadPart -                       \
        Timestamp->StartTsc.QuadPart                       \
    );                                                     \
    Timestamp->Cycles.QuadPart = (                         \
        Timestamp->End.QuadPart -                          \
        Timestamp->Start.QuadPart                          \
    );                                                     \
    Timestamp->TotalTsc.QuadPart += (                      \
        Timestamp->Tsc.QuadPart                            \
    );                                                     \
    Timestamp->TotalCycles.QuadPart += (                   \
        Timestamp->Cycles.QuadPart                         \
    );                                                     \
    Timestamp->Nanoseconds.QuadPart = (                    \
        Timestamp->Cycles.QuadPart *                       \
        TIMESTAMP_TO_NANOSECONDS                           \
    );                                                     \
    Timestamp->Nanoseconds.QuadPart /= Frequency.QuadPart; \
    Timestamp->TotalNanoseconds.QuadPart += (              \
        Timestamp->Nanoseconds.QuadPart                    \
    );                                                     \
    if (Timestamp->MinimumNanoseconds.QuadPart >           \
        Timestamp->Nanoseconds.QuadPart) {                 \
            Timestamp->MinimumNanoseconds.QuadPart = (     \
                Timestamp->Nanoseconds.QuadPart            \
            );                                             \
    }                                                      \
    if (Timestamp->MaximumNanoseconds.QuadPart <           \
        Timestamp->Nanoseconds.QuadPart) {                 \
            Timestamp->MaximumNanoseconds.QuadPart = (     \
                Timestamp->Nanoseconds.QuadPart            \
            );                                             \
    }                                                      \
    if (Timestamp->MinimumTsc.QuadPart >                   \
        Timestamp->Tsc.QuadPart) {                         \
            Timestamp->MinimumTsc.QuadPart = (             \
                Timestamp->Tsc.QuadPart                    \
            );                                             \
    }                                                      \
    if (Timestamp->MaximumTsc.QuadPart <                   \
        Timestamp->Tsc.QuadPart) {                         \
            Timestamp->MaximumTsc.QuadPart = (             \
                Timestamp->Tsc.QuadPart                    \
            );                                             \
    }                                                      \
    if (Timestamp->MinimumCycles.QuadPart >                \
        Timestamp->Cycles.QuadPart) {                      \
            Timestamp->MinimumCycles.QuadPart = (          \
                Timestamp->Cycles.QuadPart                 \
            );                                             \
    }                                                      \
    if (Timestamp->MaximumCycles.QuadPart <                \
        Timestamp->Cycles.QuadPart) {                      \
            Timestamp->MaximumCycles.QuadPart = (          \
                Timestamp->Cycles.QuadPart                 \
            );                                             \
    }

#define END_TIMESTAMP_CPUID()                 \
    __cpuid((PULONG)&Timestamp->CpuId, 0);    \
    Timestamp->EndTsc.QuadPart = __rdtsc();   \
    QueryPerformanceCounter(&Timestamp->End); \
    END_TIMESTAMP_COMMON(Id)

#define END_TIMESTAMP_RDTSCP()                              \
    Timestamp->EndTsc.QuadPart = __rdtscp(&Timestamp->Aux); \
    QueryPerformanceCounter(&Timestamp->End);               \
    END_TIMESTAMP_COMMON()

#define END_TIMESTAMP_RDTSC()                 \
    Timestamp->EndTsc.QuadPart = __rdtsc();   \
    QueryPerformanceCounter(&Timestamp->End); \
    END_TIMESTAMP_COMMON()

#define FINISH_TIMESTAMP_EXAMPLE(Id, Length, Iterations) \
    OUTPUT_STRING(&Timestamp->Name);                     \
    OUTPUT_SEP();                                        \
    OUTPUT_INT(*Length);                                 \
    OUTPUT_SEP();                                        \
    OUTPUT_INT(Iterations);                              \
    OUTPUT_SEP();                                        \
    OUTPUT_INT(Timestamp->MinimumTsc.QuadPart);          \
    OUTPUT_SEP();                                        \
    OUTPUT_INT(Timestamp->MaximumTsc.QuadPart);          \
    OUTPUT_SEP();                                        \
    OUTPUT_INT(Timestamp->TotalTsc.QuadPart);            \
    OUTPUT_SEP();                                        \
    OUTPUT_INT(Timestamp->MinimumCycles.QuadPart);       \
    OUTPUT_SEP();                                        \
    OUTPUT_INT(Timestamp->MaximumCycles.QuadPart);       \
    OUTPUT_SEP();                                        \
    OUTPUT_INT(Timestamp->TotalCycles.QuadPart);         \
    OUTPUT_SEP();                                        \
    OUTPUT_INT(Timestamp->MinimumNanoseconds.QuadPart);  \
    OUTPUT_SEP();                                        \
    OUTPUT_INT(Timestamp->MaximumNanoseconds.QuadPart);  \
    OUTPUT_SEP();                                        \
    OUTPUT_INT(Timestamp->TotalNanoseconds.QuadPart);    \
    OUTPUT_LF()

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
