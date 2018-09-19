/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    RtlOutput.c

Abstract:

    Temporary stop-gap measure for console output.

--*/

#include "stdafx.h"

_Use_decl_annotations_
VOID
AppendIntegerToCharBuffer(
    PCHAR *BufferPointer,
    ULONGLONG Integer
    )
{
    PCHAR Buffer;
    USHORT Offset;
    USHORT NumberOfDigits;
    ULONGLONG Digit;
    ULONGLONG Value;
    ULONGLONG Count;
    ULONGLONG Bytes;
    CHAR Char;
    PCHAR Dest;

    Buffer = *BufferPointer;

    //
    // Count the number of digits required to represent the integer in base 10.
    //

    NumberOfDigits = CountNumberOfLongLongDigitsInline(Integer);

    //
    // Initialize our destination pointer to the last digit.  (We write
    // back-to-front.)
    //

    Offset = (NumberOfDigits - 1) * sizeof(Char);
    Dest = (PCHAR)RtlOffsetToPointer(Buffer, Offset);

    Count = 0;
    Bytes = 0;

    //
    // Convert each digit into the corresponding character and copy to the
    // string buffer, retreating the pointer as we go.
    //

    Value = Integer;

    do {
        Count++;
        Bytes += sizeof(Char);
        Digit = Value % 10;
        Value = Value / 10;
        Char = ((CHAR)Digit + '0');
        *Dest-- = Char;
    } while (Value != 0);

    *BufferPointer = RtlOffsetToPointer(Buffer, Bytes);

    return;
}

APPEND_INTEGER_TO_CHAR_BUFFER_AS_HEX AppendIntegerToCharBufferAsHex;

_Use_decl_annotations_
VOID
AppendIntegerToCharBufferAsHex(
    PCHAR *BufferPointer,
    ULONG Integer
    )
/*++

Routine Description:

    This is a helper routine that appends an integer to a character buffer
    in the hexadecimal format 0x00000000 (i.e. zeros are added as padding where
    necessary and the 0x prefix is included).

Arguments:

    BufferPointer - Supplies a pointer to a variable that contains the address
        of a character buffer to which the hex string representation of the
        integer will be written.  The pointer is adjusted to point after the
        length of the written bytes prior to returning.  This will always be
        10 characters/bytes.

    Integer - Supplies the integer value to be appended to the string.

Return Value:

    None.

--*/
{
    CHAR Char;
    ULONG Pad;
    ULONG Count;
    ULONG Digit;
    ULONG Value;
    PCHAR End;
    PCHAR Dest;
    PCHAR Buffer;

    Buffer = *BufferPointer;

    End = Dest = RtlOffsetToPointer(Buffer, 9);

    Count = 0;
    Value = Integer;

    do {
        Count++;
        Digit = Value & 0xf;
        Value >>= 4;
        Char = IntegerToCharTable[Digit];
        *Dest-- = Char;
    } while (Value != 0);

    Pad = 8 - Count;
    while (Pad) {
        *Dest-- = '0';
        Pad--;
    }

    *Dest-- = 'x';
    *Dest-- = '0';

    *BufferPointer = End + 1;

    return;
}

_Use_decl_annotations_
VOID
AppendIntegerToCharBufferEx(
    PCHAR *BufferPointer,
    ULONGLONG Integer,
    BYTE NumberOfDigits,
    CHAR Pad,
    CHAR Trailer
    )
/*++

Routine Description:

    This is a helper routine that appends an integer to a character buffer,
    with optional support for padding and trailer characters.

Arguments:

    BufferPointer - Supplies a pointer to a variable that contains the address
        of a character buffer to which the string representation of the integer
        will be written.  The pointer is adjusted to point after the length of
        the written bytes prior to returning.

    Integer - Supplies the long long integer value to be appended to the string.

    NumberOfDigits - The expected number of digits for the value.  If Integer
        has less digits than this number, it will be left-padded with the char
        indicated by the Pad parameter.

    Pad - A character to use for padding, if applicable.

    Trailer - An optional trailing wide character to append.

Return Value:

    None.

--*/
{
    BYTE Offset;
    BYTE NumberOfCharsToPad;
    BYTE ActualNumberOfDigits;
    ULONGLONG Digit;
    ULONGLONG Value;
    ULONGLONG Count;
    ULONGLONG Bytes;
    CHAR Char;
    PCHAR End;
    PCHAR Dest;
    PCHAR Start;
    PCHAR Expected;

    Start = *BufferPointer;

    //
    // Make sure the integer value doesn't have more digits than specified.
    //

    ActualNumberOfDigits = CountNumberOfLongLongDigitsInline(Integer);
    ASSERT(ActualNumberOfDigits <= NumberOfDigits);

    //
    // Initialize our destination pointer to the last digit.  (We write
    // back-to-front.)
    //

    Offset = (NumberOfDigits - 1) * sizeof(Char);
    Dest = (PCHAR)RtlOffsetToPointer(Start, Offset);
    End = Dest + 1;

    Count = 0;
    Bytes = 0;

    //
    // Convert each digit into the corresponding character and copy to the
    // string buffer, retreating the pointer as we go.
    //

    Value = Integer;

    do {
        Count++;
        Bytes += sizeof(Char);
        Digit = Value % 10;
        Value = Value / 10;
        Char = ((CHAR)Digit + '0');
        *Dest-- = Char;
    } while (Value != 0);

    //
    // Pad the string with zeros if necessary.
    //

    NumberOfCharsToPad = NumberOfDigits - ActualNumberOfDigits;

    if (NumberOfCharsToPad && Pad) {
        do {
            Count++;
            Bytes += sizeof(Char);
            *Dest-- = Pad;
        } while (--NumberOfCharsToPad);
    }

    //
    // Add the trailer if applicable.
    //

    if (Trailer) {
        Bytes += sizeof(Char);
        *End++ = Trailer;
    }

    Expected = (PCHAR)RtlOffsetToPointer(Start, Bytes);
    ASSERT(Expected == End);

    *BufferPointer = End;

    return;
}

_Use_decl_annotations_
VOID
AppendStringToCharBuffer(
    PCHAR *BufferPointer,
    PCSTRING String
    )
{
    PVOID Buffer;

    Buffer = *BufferPointer;
    CopyMemoryInline(Buffer, String->Buffer, String->Length);
    *BufferPointer = RtlOffsetToPointer(Buffer, String->Length);

    return;
}

_Use_decl_annotations_
VOID
AppendCharBufferToCharBuffer(
    PCHAR *BufferPointer,
    PCCHAR String,
    ULONG SizeInBytes
    )
{
    PVOID Buffer;

    Buffer = *BufferPointer;
    CopyMemoryInline(Buffer, String, SizeInBytes);
    *BufferPointer = RtlOffsetToPointer(Buffer, SizeInBytes);

    return;
}

_Use_decl_annotations_
VOID
AppendCharToCharBuffer(
    PCHAR *BufferPointer,
    CHAR Char
    )
{
    PCHAR Buffer;

    Buffer = *BufferPointer;
    *Buffer = Char;
    *BufferPointer = Buffer + 1;
}

_Use_decl_annotations_
VOID
AppendCStrToCharBuffer(
    PCHAR *BufferPointer,
    PCSZ String
    )
{
    PCHAR Dest = *BufferPointer;
    PCHAR Source = (PCHAR)String;

    while (*Source) {
        *Dest++ = *Source++;
    }

    *BufferPointer = Dest;

    return;
}

_Use_decl_annotations_
VOID
AppendIntegerToWideCharBuffer(
    PWCHAR *BufferPointer,
    ULONGLONG Integer
    )
{
    PWCHAR Buffer;
    USHORT Offset;
    USHORT NumberOfDigits;
    ULONGLONG Digit;
    ULONGLONG Value;
    ULONGLONG Count;
    ULONGLONG Bytes;
    WCHAR WideChar;
    PWCHAR Dest;

    Buffer = *BufferPointer;

    //
    // Count the number of digits required to represent the integer in base 10.
    //

    NumberOfDigits = CountNumberOfLongLongDigitsInline(Integer);

    //
    // Initialize our destination pointer to the last digit.  (We write
    // back-to-front.)
    //

    Offset = (NumberOfDigits - 1) * sizeof(WideChar);
    Dest = (PWCHAR)RtlOffsetToPointer(Buffer, Offset);

    Count = 0;
    Bytes = 0;

    //
    // Convert each digit into the corresponding character and copy to the
    // string buffer, retreating the pointer as we go.
    //

    Value = Integer;

    do {
        Count++;
        Bytes += sizeof(WideChar);
        Digit = Value % 10;
        Value = Value / 10;
        WideChar = ((WCHAR)Digit + '0');
        *Dest-- = WideChar;
    } while (Value != 0);

    *BufferPointer = (PWCHAR)RtlOffsetToPointer(Buffer, Bytes);

    return;
}

APPEND_INTEGER_TO_WIDE_CHAR_BUFFER_EX AppendIntegerToWideCharBufferEx;

_Use_decl_annotations_
VOID
AppendIntegerToWideCharBufferEx(
    PWCHAR *BufferPointer,
    ULONGLONG Integer,
    BYTE NumberOfDigits,
    WCHAR Pad,
    WCHAR Trailer
    )
/*++

Routine Description:

    This is a helper routine that appends an integer to a character buffer,
    with optional support for padding and trailer characters.

Arguments:

    BufferPointer - Supplies a pointer to a variable that contains the address
        of a character buffer to which the string representation of the integer
        will be written.  The pointer is adjusted to point after the length of
        the written bytes prior to returning.

    Integer - Supplies the long long integer value to be appended to the string.

    NumberOfDigits - The expected number of digits for the value.  If Integer
        has less digits than this number, it will be left-padded with the char
        indicated by the Pad parameter.

    Pad - A character to use for padding, if applicable.

    Trailer - An optional trailing wide character to append.

Return Value:

    None.

--*/
{
    BYTE Offset;
    BYTE NumberOfWideCharsToPad;
    BYTE ActualNumberOfDigits;
    ULONGLONG Digit;
    ULONGLONG Value;
    ULONGLONG Count;
    ULONGLONG Bytes;
    WCHAR WideChar;
    PWCHAR End;
    PWCHAR Dest;
    PWCHAR Start;
    PWCHAR Expected;

    Start = *BufferPointer;

    //
    // Make sure the integer value doesn't have more digits than specified.
    //

    ActualNumberOfDigits = CountNumberOfLongLongDigitsInline(Integer);
    ASSERT(ActualNumberOfDigits <= NumberOfDigits);

    //
    // Initialize our destination pointer to the last digit.  (We write
    // back-to-front.)
    //

    Offset = (NumberOfDigits - 1) * sizeof(WideChar);
    Dest = (PWCHAR)RtlOffsetToPointer(Start, Offset);
    End = Dest + 1;

    Count = 0;
    Bytes = 0;

    //
    // Convert each digit into the corresponding character and copy to the
    // string buffer, retreating the pointer as we go.
    //

    Value = Integer;

    do {
        Count++;
        Bytes += sizeof(WideChar);
        Digit = Value % 10;
        Value = Value / 10;
        WideChar = ((WCHAR)Digit + '0');
        *Dest-- = WideChar;
    } while (Value != 0);

    //
    // Pad the string with zeros if necessary.
    //

    NumberOfWideCharsToPad = NumberOfDigits - ActualNumberOfDigits;

    if (NumberOfWideCharsToPad && Pad) {
        do {
            Count++;
            Bytes += sizeof(WideChar);
            *Dest-- = Pad;
        } while (--NumberOfWideCharsToPad);
    }

    //
    // Add the trailer if applicable.
    //

    if (Trailer) {
        Bytes += sizeof(WideChar);
        *End++ = Trailer;
    }

    Expected = (PWCHAR)RtlOffsetToPointer(Start, Bytes);
    ASSERT(Expected == End);

    *BufferPointer = End;

    return;
}

_Use_decl_annotations_
VOID
AppendUnicodeStringToWideCharBuffer(
    PWCHAR *BufferPointer,
    PCUNICODE_STRING UnicodeString
    )
{
    PVOID Buffer;

    Buffer = *BufferPointer;
    CopyMemoryInline(Buffer, UnicodeString->Buffer, UnicodeString->Length);
    *BufferPointer = (PWCHAR)RtlOffsetToPointer(Buffer, UnicodeString->Length);

    return;
}

_Use_decl_annotations_
VOID
AppendWideCharBufferToWideCharBuffer(
    PWCHAR *BufferPointer,
    PCWCHAR String,
    ULONG SizeInBytes
    )
{
    PVOID Buffer;

    Buffer = *BufferPointer;
    CopyMemoryInline(Buffer, String, SizeInBytes);
    *BufferPointer = (PWCHAR)RtlOffsetToPointer(Buffer, SizeInBytes);

    return;
}

_Use_decl_annotations_
VOID
AppendWideCharToWideCharBuffer(
    PWCHAR *BufferPointer,
    WCHAR WideChar
    )
{
    PWCHAR Buffer;

    Buffer = *BufferPointer;
    *Buffer = WideChar;
    *BufferPointer = Buffer + 1;
}

_Use_decl_annotations_
VOID
AppendWideCStrToWideCharBuffer(
    PWCHAR *BufferPointer,
    PCWSZ String
    )
{
    PWCHAR Dest = *BufferPointer;
    PWCHAR Source = (PWCHAR)String;

    while (*Source) {
        *Dest++ = *Source++;
    }

    *BufferPointer = Dest;

    return;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
