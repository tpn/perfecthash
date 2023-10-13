/*++

Copyright (c) 2018-2023 Trent Nelson <trent@trent.me>

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

FORCEINLINE
BYTE
CountNumberOfHexCharsInline(_In_ ULONG Value)
{
    BYTE Count = 0;

    do {
        Count++;
        Value >>= 4;
    } while (Value != 0);

    return Count;
}

FORCEINLINE
BYTE
CountNumberOfHex64CharsInline(_In_ ULONGLONG Value)
{
    BYTE Count = 0;

    do {
        Count++;
        Value >>= 4;
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
    _In_ WCHAR Trailer
    );
typedef APPEND_INTEGER_TO_UNICODE_STRING *PAPPEND_INTEGER_TO_UNICODE_STRING;

typedef
BOOLEAN
(NTAPI APPEND_LONGLONG_INTEGER_TO_UNICODE_STRING)(
    _In_ PUNICODE_STRING String,
    _In_ ULONGLONG Integer,
    _In_ USHORT NumberOfDigits,
    _In_ WCHAR Trailer
    );
typedef APPEND_LONGLONG_INTEGER_TO_UNICODE_STRING
      *PAPPEND_LONGLONG_INTEGER_TO_UNICODE_STRING;

typedef
BOOLEAN
(NTAPI APPEND_INTEGER_TO_STRING)(
    _In_ PSTRING String,
    _In_ ULONG Integer,
    _In_ USHORT NumberOfDigits,
    _In_ CHAR Trailer
    );
typedef APPEND_INTEGER_TO_STRING *PAPPEND_INTEGER_TO_STRING;

typedef
BOOLEAN
(NTAPI APPEND_LONGLONG_INTEGER_TO_STRING)(
    _In_ PSTRING String,
    _In_ ULONGLONG Integer,
    _In_ USHORT NumberOfDigits,
    _In_ CHAR Trailer
    );
typedef APPEND_LONGLONG_INTEGER_TO_STRING
      *PAPPEND_LONGLONG_INTEGER_TO_STRING;

typedef
VOID
(NTAPI APPEND_INTEGER_TO_CHAR_BUFFER)(
    _Inout_ PCHAR *BufferPointer,
    _In_ ULONGLONG Integer
    );
typedef APPEND_INTEGER_TO_CHAR_BUFFER *PAPPEND_INTEGER_TO_CHAR_BUFFER;

typedef
VOID
(NTAPI APPEND_SIGNED_INTEGER_TO_CHAR_BUFFER)(
    _Inout_ PCHAR *BufferPointer,
    _In_ LONGLONG Integer
    );
typedef APPEND_SIGNED_INTEGER_TO_CHAR_BUFFER
      *PAPPEND_SIGNED_INTEGER_TO_CHAR_BUFFER;

typedef
VOID
(NTAPI APPEND_DOUBLE_TO_CHAR_BUFFER)(
    _Inout_ PCHAR *BufferPointer,
    _In_ DOUBLE Double
    );
typedef APPEND_DOUBLE_TO_CHAR_BUFFER *PAPPEND_DOUBLE_TO_CHAR_BUFFER;

typedef
VOID
(NTAPI APPEND_INTEGER_TO_CHAR_BUFFER_AS_HEX)(
    _Inout_ PCHAR *BufferPointer,
    _In_ ULONG Integer
    );
typedef APPEND_INTEGER_TO_CHAR_BUFFER_AS_HEX
      *PAPPEND_INTEGER_TO_CHAR_BUFFER_AS_HEX;

typedef
VOID
(NTAPI APPEND_LONGLONG_INTEGER_TO_CHAR_BUFFER_AS_HEX)(
    _Inout_ PCHAR *BufferPointer,
    _In_ ULONGLONG Integer
    );
typedef APPEND_LONGLONG_INTEGER_TO_CHAR_BUFFER_AS_HEX
      *PAPPEND_LONGLONG_INTEGER_TO_CHAR_BUFFER_AS_HEX;

//
// As above, but no leading spaces or 0x padding.
//

typedef
VOID
(NTAPI APPEND_INTEGER_TO_CHAR_BUFFER_AS_HEX_RAW)(
    _Inout_ PCHAR *BufferPointer,
    _In_ ULONG Integer
    );
typedef APPEND_INTEGER_TO_CHAR_BUFFER_AS_HEX_RAW
      *PAPPEND_INTEGER_TO_CHAR_BUFFER_AS_HEX_RAW;

typedef
VOID
(NTAPI APPEND_LONGLONG_INTEGER_TO_CHAR_BUFFER_AS_HEX_RAW)(
    _Inout_ PCHAR *BufferPointer,
    _In_ ULONGLONG Integer
    );
typedef APPEND_LONGLONG_INTEGER_TO_CHAR_BUFFER_AS_HEX_RAW
      *PAPPEND_LONGLONG_INTEGER_TO_CHAR_BUFFER_AS_HEX_RAW;

typedef
VOID
(NTAPI APPEND_INTEGER_TO_CHAR_BUFFER_EX)(
    _Inout_ PCHAR *BufferPointer,
    _In_ ULONGLONG Integer,
    _In_ BYTE NumberOfDigits,
    _In_ CHAR Pad,
    _In_ CHAR Trailer
    );
typedef APPEND_INTEGER_TO_CHAR_BUFFER_EX *PAPPEND_INTEGER_TO_CHAR_BUFFER_EX;

typedef
VOID
(NTAPI APPEND_LONGLONG_INTEGER_TO_CHAR_BUFFER_AS_HEX_RAW)(
    _Inout_ PCHAR *BufferPointer,
    _In_ ULONGLONG Integer
    );
typedef APPEND_LONGLONG_INTEGER_TO_CHAR_BUFFER_AS_HEX_RAW
      *PAPPEND_LONGLONG_INTEGER_TO_CHAR_BUFFER_AS_HEX_RAW;

typedef
VOID
(NTAPI APPEND_STRING_TO_CHAR_BUFFER)(
    _Inout_ PCHAR *BufferPointer,
    _In_ PCSTRING String
    );
typedef APPEND_STRING_TO_CHAR_BUFFER *PAPPEND_STRING_TO_CHAR_BUFFER;

//
// N.B. "FAST" in this context means that we don't do a wide character to
//      multibyte conversion; we just cast each word as a byte, and use that.
//      It should only be used when the input unicode string is guaranteed to
//      be ASCII.
//

typedef
VOID
(NTAPI APPEND_UNICODE_STRING_TO_CHAR_BUFFER_FAST)(
    _Inout_ PCHAR *BufferPointer,
    _In_opt_ PCUNICODE_STRING String
    );
typedef APPEND_UNICODE_STRING_TO_CHAR_BUFFER_FAST
      *PAPPEND_UNICODE_STRING_TO_CHAR_BUFFER_FAST;

//
// N.B. As above, but vice versa.  Cast each byte into a word.
//

typedef
VOID
(NTAPI APPEND_STRING_TO_WIDE_CHAR_BUFFER_FAST)(
    _Inout_ PWCHAR *BufferPointer,
    _In_ PCSTRING String
    );
typedef APPEND_STRING_TO_WIDE_CHAR_BUFFER_FAST
      *PAPPEND_STRING_TO_WIDE_CHAR_BUFFER_FAST;

typedef
HRESULT
(NTAPI APPEND_STRING_TO_UNICODE_STRING_FAST)(
    _In_ PCSTRING String,
    _Inout_ PUNICODE_STRING UnicodeString
    );
typedef APPEND_STRING_TO_UNICODE_STRING_FAST
      *PAPPEND_STRING_TO_UNICODE_STRING_FAST;

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

typedef
VOID
(NTAPI APPEND_WSTR_TO_CHAR_BUFFER_FAST)(
    _Inout_ PCHAR *BufferPointer,
    _In_ PWSTR String
    );
typedef APPEND_WSTR_TO_CHAR_BUFFER_FAST *PAPPEND_WSTR_TO_CHAR_BUFFER_FAST;

typedef
VOID
(NTAPI APPEND_ERROR_CODE_CSTR_TO_CHAR_BUFFER)(
    _In_ PRTL Rtl,
    _Inout_ PCHAR *BufferPointer,
    _In_ HRESULT Code
    );
typedef APPEND_ERROR_CODE_CSTR_TO_CHAR_BUFFER
      *PAPPEND_ERROR_CODE_CSTR_TO_CHAR_BUFFER;

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
    _In_ WCHAR Trailer
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

//
// Hash glue.
//

typedef
VOID
(NTAPI HASH_STRING)(
    _Inout_ PSTRING String
    );
typedef HASH_STRING *PHASH_STRING;
extern HASH_STRING Crc32HashString;

typedef
VOID
(NTAPI HASH_UNICODE_STRING)(
    _Inout_ PUNICODE_STRING String
    );
typedef HASH_UNICODE_STRING *PHASH_UNICODE_STRING;
extern HASH_UNICODE_STRING Crc32HashUnicodeString;

#define HashString Crc32HashString
#define HashUnicodeString Crc32HashUnicodeString

//
// Decls.
//

#ifndef __INTELLISENSE__
extern APPEND_INTEGER_TO_UNICODE_STRING AppendIntegerToUnicodeString;
extern APPEND_LONGLONG_INTEGER_TO_UNICODE_STRING
    AppendLongLongIntegerToUnicodeString;
extern APPEND_INTEGER_TO_STRING AppendIntegerToString;
extern APPEND_LONGLONG_INTEGER_TO_STRING AppendLongLongIntegerToString;
extern APPEND_INTEGER_TO_CHAR_BUFFER AppendIntegerToCharBuffer;
extern APPEND_SIGNED_INTEGER_TO_CHAR_BUFFER AppendSignedIntegerToCharBuffer;
extern APPEND_DOUBLE_TO_CHAR_BUFFER AppendDoubleToCharBuffer;
extern APPEND_INTEGER_TO_CHAR_BUFFER_AS_HEX AppendIntegerToCharBufferAsHex;
extern APPEND_LONGLONG_INTEGER_TO_CHAR_BUFFER_AS_HEX
    AppendLongLongIntegerToCharBufferAsHex;
extern APPEND_INTEGER_TO_CHAR_BUFFER_AS_HEX_RAW
    AppendIntegerToCharBufferAsHexRaw;
extern APPEND_LONGLONG_INTEGER_TO_CHAR_BUFFER_AS_HEX_RAW
    AppendLongLongIntegerToCharBufferAsHexRaw;
extern APPEND_INTEGER_TO_CHAR_BUFFER_EX AppendIntegerToCharBufferEx;
extern APPEND_STRING_TO_CHAR_BUFFER AppendStringToCharBuffer;
extern APPEND_UNICODE_STRING_TO_CHAR_BUFFER_FAST
    AppendUnicodeStringToCharBufferFast;
extern APPEND_STRING_TO_WIDE_CHAR_BUFFER_FAST AppendStringToWideCharBufferFast;
extern APPEND_STRING_TO_UNICODE_STRING_FAST AppendStringToUnicodeStringFast;
extern APPEND_CHAR_BUFFER_TO_CHAR_BUFFER AppendCharBufferToCharBuffer;
extern APPEND_CSTR_TO_CHAR_BUFFER AppendCStrToCharBuffer;
extern APPEND_WSTR_TO_CHAR_BUFFER_FAST AppendWStrToCharBufferFast;
extern APPEND_ERROR_CODE_CSTR_TO_CHAR_BUFFER AppendErrorCodeCStrToCharBuffer;
extern APPEND_CHAR_TO_CHAR_BUFFER AppendCharToCharBuffer;
extern APPEND_UNICODE_STRING_TO_WIDE_CHAR_BUFFER
    AppendUnicodeStringToWideCharBuffer;
extern APPEND_WIDE_CSTR_TO_WIDE_CHAR_BUFFER AppendWideCStrToWideCharBuffer;
extern APPEND_WIDE_CHAR_BUFFER_TO_WIDE_CHAR_BUFFER
    AppendWideCharBufferToWideCharBuffer;
extern APPEND_WIDE_CHAR_TO_WIDE_CHAR_BUFFER AppendWideCharToWideCharBuffer;
extern APPEND_INTEGER_TO_WIDE_CHAR_BUFFER AppendIntegerToWideCharBuffer;
#endif

//
// Some timestamp helpers.
//

#define RTL_TIMESTAMP_FORMAT "yyyy-MM-dd HH:mm:ss.000"
#define RTL_TIMESTAMP_FORMAT_LENGTH 24
C_ASSERT(sizeof(RTL_TIMESTAMP_FORMAT) == RTL_TIMESTAMP_FORMAT_LENGTH);

FORCEINLINE
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
InitializeTimestampString(
    _Inout_ PCHAR Buffer,
    _In_ ULONG SizeOfBufferInBytes,
    _Inout_ PSTRING String,
    _Out_ PFILETIME FileTime,
    _Out_ PSYSTEMTIME SystemTime
    )
{
    ULONG Value;
    HRESULT Result = S_OK;

    //
    // Validate arguments.
    //

    if (SizeOfBufferInBytes != RTL_TIMESTAMP_FORMAT_LENGTH) {
        return E_INVALIDARG;
    }

    String->Buffer = Buffer;
    String->Length = 0;
    String->MaximumLength = (USHORT)RTL_TIMESTAMP_FORMAT_LENGTH;

    GetLocalTime(SystemTime);

    if (!SystemTimeToFileTime(SystemTime, FileTime)) {
        return PH_E_SYSTEM_CALL_FAILED;
    }

#define RTL_APPEND_TIME_FIELD(Field, Digits, Trailer)             \
    Value = SystemTime->Field;                                    \
    if (!AppendIntegerToString(String, Value, Digits, Trailer)) { \
        Result = PH_E_STRING_BUFFER_OVERFLOW;                     \
        goto End;                                                 \
    }

    RTL_APPEND_TIME_FIELD(wYear,          4, '-');
    RTL_APPEND_TIME_FIELD(wMonth,         2, '-');
    RTL_APPEND_TIME_FIELD(wDay,           2, ' ');
    RTL_APPEND_TIME_FIELD(wHour,          2, ':');
    RTL_APPEND_TIME_FIELD(wMinute,        2, ':');
    RTL_APPEND_TIME_FIELD(wSecond,        2, '.');
    RTL_APPEND_TIME_FIELD(wMilliseconds,  3,   0);

End:

    return Result;
}

#define RTL_TIMESTAMP_FORMAT_FILE_SUFFIX "yyyy-MM-dd_HH-mm-ss.000"
#define RTL_TIMESTAMP_FORMAT_FILE_SUFFIX_LENGTH 24
C_ASSERT(sizeof(RTL_TIMESTAMP_FORMAT_FILE_SUFFIX) ==
         RTL_TIMESTAMP_FORMAT_FILE_SUFFIX_LENGTH);

FORCEINLINE
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
InitializeTimestampStringForFileSuffix (
    _Inout_ PCHAR Buffer,
    _In_ ULONG SizeOfBufferInBytes,
    _Inout_ PSTRING String,
    _In_ PSYSTEMTIME SystemTime
    )
{
    ULONG Value;
    HRESULT Result = S_OK;

    //
    // Validate arguments.
    //

    if (SizeOfBufferInBytes != RTL_TIMESTAMP_FORMAT_FILE_SUFFIX_LENGTH) {
        return E_INVALIDARG;
    }

    String->Buffer = Buffer;
    String->Length = 0;
    String->MaximumLength = (USHORT)RTL_TIMESTAMP_FORMAT_FILE_SUFFIX_LENGTH;

#define RTL_APPEND_TIME_FIELD(Field, Digits, Trailer)             \
    Value = SystemTime->Field;                                    \
    if (!AppendIntegerToString(String, Value, Digits, Trailer)) { \
        Result = PH_E_STRING_BUFFER_OVERFLOW;                     \
        goto End;                                                 \
    }

    RTL_APPEND_TIME_FIELD(wYear,          4, '-');
    RTL_APPEND_TIME_FIELD(wMonth,         2, '-');
    RTL_APPEND_TIME_FIELD(wDay,           2, '_');
    RTL_APPEND_TIME_FIELD(wHour,          2, '-');
    RTL_APPEND_TIME_FIELD(wMinute,        2, '-');
    RTL_APPEND_TIME_FIELD(wSecond,        2, '.');
    RTL_APPEND_TIME_FIELD(wMilliseconds,  3,   0);

End:

    return Result;
}

//
// Output helpers.
//

static PCSZ Dot = ".";
static PCSZ BigF = "F";
static PCSZ BigL = "L";
static PCSZ BigS = "S";
static PCSZ BigT = "T";
static PCSZ BigV = "V";
static PCSZ Dash = "-";
static PCSZ Plus = "+";
static PCSZ Caret = "^";
static PCSZ Cross = "x";
static PCSZ LittleT = "t";
static PCSZ Percent = "%";
static PCSZ Newline = "\n";
static PCSZ Question = "?";
static PCSZ Asterisk = "*";
static PCSZ Exclamation = "!";

#define DO_OUTPUT(Buffer, Size)                                            \
    if (!Silent) {                                                         \
        if (!WriteFile(OutputHandle, Buffer, Size, &BytesWritten, NULL)) { \
            Result = PH_E_SYSTEM_CALL_FAILED;                              \
            SYS_ERROR(WriteFile);                                          \
            goto Error;                                                    \
        } else if (BytesWritten != Size) {                                 \
            Result = PH_E_NOT_ALL_BYTES_WRITTEN;                           \
            PH_ERROR(__FUNCTION__, Result);                                \
            goto Error;                                                    \
        }                                                                  \
    }

#define MAYBE_OUTPUT(Buffer, Size) \
    if (OutputHandle) {            \
        DO_OUTPUT(Buffer, Size);   \
    }

#define DOT() DO_OUTPUT(Dot, 1)
#define BIGF() DO_OUTPUT(BigF, 1)
#define BIGL() DO_OUTPUT(BigL, 1)
#define BIGS() DO_OUTPUT(BigS, 1)
#define BIGT() DO_OUTPUT(BigT, 1)
#define BIGV() DO_OUTPUT(BigV, 1)
#define DASH() DO_OUTPUT(Dash, 1)
#define CARET() DO_OUTPUT(Caret, 1)
#define CROSS() DO_OUTPUT(Cross, 1)
#define LITTLET() DO_OUTPUT(LittleT, 1)
#define PERCENT() DO_OUTPUT(Percent, 1)
#define NEWLINE() DO_OUTPUT(Newline, 1)
#define QUESTION() DO_OUTPUT(Question, 1)
#define ASTERISK() DO_OUTPUT(Asterisk, 1)
#define EXCLAMATION() DO_OUTPUT(Exclamation, 1)

#define PRINT_CSTR(Buf) do {                \
    DO_OUTPUT((Buf), (DWORD)strlen((Buf))); \
    NEWLINE();                              \
} while (0)

#define PRINT_WSTR(Buf) do {                       \
    DO_OUTPUT((Buf), ((DWORD)wcslen((Buf)) << 1)); \
    NEWLINE();                                     \
} while (0)

#define MAYBE_DOT() MAYBE_OUTPUT(Dot, 1)
#define MAYBE_DASH() MAYBE_OUTPUT(Dash, 1)
#define MAYBE_PLUS() MAYBE_OUTPUT(Plus, 1)
#define MAYBE_CARET() MAYBE_OUTPUT(Caret, 1)
#define MAYBE_CROSS() MAYBE_OUTPUT(Cross, 1)
#define MAYBE_PERCENT() MAYBE_OUTPUT(Percent, 1)
#define MAYBE_NEWLINE() MAYBE_OUTPUT(Newline, 1)
#define MAYBE_ASTERISK() MAYBE_OUTPUT(Asterisk, 1)
#define MAYBE_EXCLAMATION() MAYBE_OUTPUT(Exclamation, 1)

#define OUTPUT_RAW(String)                                          \
    AppendCharBufferToCharBuffer(&Output, String, sizeof(String)-1)

#define OUTPUT_BITMAP32_RAW(String)                      \
    *Output++ = '0';                                     \
    *Output++ = 'b';                                     \
    AppendCharBufferToCharBuffer(&Output, String+32, 32)

#define OUTPUT_BITMAP64_RAW(String)                   \
    *Output++ = '0';                                  \
    *Output++ = 'b';                                  \
    AppendCharBufferToCharBuffer(&Output, String, 64)

#define OUTPUT_HEX(Integer) AppendIntegerToCharBufferAsHex(&Output, Integer)
#define OUTPUT_HEX64(Integer) \
    AppendLongLongIntegerToCharBufferAsHex(&Output, Integer)

#define OUTPUT_HEX_RAW(Integer)                         \
    AppendIntegerToCharBufferAsHexRaw(&Output, Integer)

#define OUTPUT_HEX64_RAW(LongLongInteger)                      \
    AppendLongLongIntegerToCharBufferAsHexRaw(&Output,         \
                                              LongLongInteger)

#define OUTPUT_HEX_RAW_0x(Integer)                      \
    *Output++ = 'x';                                    \
    *Output++ = '0';                                    \
    AppendIntegerToCharBufferAsHexRaw(&Output, Integer)

#define OUTPUT_STRING(String) AppendStringToCharBuffer(&Output, String)

#define OUTPUT_UNICODE_STRING_FAST(String) \
    AppendUnicodeStringToCharBufferFast(&Output, String)

#define OUTPUT_ERROR_CODE_STRING(Code) \
    AppendErrorCodeCStrToCharBuffer(Rtl, &Output, Code)

#define OUTPUT_CSTR(Str) AppendCStrToCharBuffer(&Output, Str)
#define OUTPUT_WSTR_FAST(Str) AppendWStrToCharBufferFast(&Output, Str)
#define OUTPUT_CHR(Char) AppendCharToCharBuffer(&Output, Char)
#define OUTPUT_SEP() AppendCharToCharBuffer(&Output, ',')
#define OUTPUT_LF() AppendCharToCharBuffer(&Output, '\n')

#define OUTPUT_INT(Value)                     \
    AppendIntegerToCharBuffer(&Output, Value)

#define OUTPUT_SIGNED_INT(Value)                     \
    AppendSignedIntegerToCharBuffer(&Output, Value)

#define OUTPUT_DOUBLE(Value)                 \
    AppendDoubleToCharBuffer(&Output, Value)

#ifdef PH_WINDOWS
#define OUTPUT_FLUSH_CONSOLE()                                               \
    BytesToWrite.QuadPart = ((ULONG_PTR)Output) - ((ULONG_PTR)OutputBuffer); \
    Success = WriteConsoleA(OutputHandle,                                    \
                            OutputBuffer,                                    \
                            BytesToWrite.LowPart,                            \
                            &CharsWritten,                                   \
                            NULL);                                           \
    ASSERT(Success);                                                         \
    Output = OutputBuffer
#else
#define OUTPUT_FLUSH_CONSOLE()                                               \
    BytesToWrite.QuadPart = ((ULONG_PTR)Output) - ((ULONG_PTR)OutputBuffer); \
    write(1, OutputBuffer, BytesToWrite.LowPart);                            \
    Output = OutputBuffer
#endif

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

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
