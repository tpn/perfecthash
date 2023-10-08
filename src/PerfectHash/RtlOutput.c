/*++

Copyright (c) 2018-2023 Trent Nelson <trent@trent.me>

Module Name:

    RtlOutput.c

Abstract:

    Implementation of various string manipulation routines.

--*/

#include "stdafx.h"

APPEND_INTEGER_TO_UNICODE_STRING AppendIntegerToUnicodeString;

_Use_decl_annotations_
BOOLEAN
AppendIntegerToUnicodeString(
    PUNICODE_STRING String,
    ULONG Integer,
    USHORT NumberOfDigits,
    WCHAR Trailer
    )
/*++

Routine Description:

    This is a helper routine that allows construction of unicode strings out
    of integer values.

Arguments:

    String - Supplies a pointer to a UNICODE_STRING that will be appended to.
        Sufficient buffer space must exist for the entire string to be written.

    Integer - The integer value to be appended to the string.

    NumberOfDigits - The expected number of digits for the value.  If Integer
        has less digits than this number, it will be left-padded with zeros.

    Trailer - An optional trailing wide character to append.

Return Value:

    TRUE on success, FALSE on failure.

--*/
{
    USHORT ActualNumberOfDigits;
    USHORT BytesRequired;
    USHORT BytesRemaining;
    USHORT NumberOfZerosToPad;
    ULONG Digit;
    ULONG Value;
    ULONG Count;
    ULONG Bytes;
    WCHAR Char;
    PWCHAR Dest;

    //
    // Verify the unicode string has sufficient space.
    //

    BytesRequired = NumberOfDigits * sizeof(WCHAR);

    if (Trailer) {
        BytesRequired += (1 * sizeof(Trailer));
    }

    BytesRemaining = (
        String->MaximumLength -
        String->Length
    );

    if (BytesRemaining < BytesRequired) {
        return FALSE;
    }

    //
    // Make sure the integer value doesn't have more digits than
    // specified.
    //

    ActualNumberOfDigits = CountNumberOfDigitsInline(Integer);

    if (ActualNumberOfDigits > NumberOfDigits) {
        return FALSE;
    }

    //
    // Initialize our destination pointer to the last digit.  (We write
    // back-to-front.)
    //

    Dest = (PWCHAR)(
        RtlOffsetToPointer(
            String->Buffer,
            String->Length + (
                (NumberOfDigits - 1) *
                sizeof(WCHAR)
            )
        )
    );
    Count = 0;
    Bytes = 0;

    //
    // Convert each digit into the corresponding character and copy to the
    // string buffer, retreating the pointer as we go.
    //

    Value = Integer;

    do {
        Count++;
        Bytes += 2;
        Digit = Value % 10;
        Value = Value / 10;
        Char = IntegerToWCharTable[Digit];
        *Dest-- = Char;
    } while (Value != 0);

    //
    // Pad the string with zeros if necessary.
    //

    NumberOfZerosToPad = NumberOfDigits - ActualNumberOfDigits;

    if (NumberOfZerosToPad) {
        do {
            Count++;
            Bytes += 2;
            *Dest-- = L'0';
        } while (--NumberOfZerosToPad);
    }

    //
    // Update the string with the new length.
    //

    String->Length += (USHORT)Bytes;

    //
    // Add the trailer if applicable.
    //

    if (Trailer) {
        String->Length += sizeof(WCHAR);
        String->Buffer[(String->Length - 1) / sizeof(WCHAR)] = Trailer;
    }

    return TRUE;
}

APPEND_LONGLONG_INTEGER_TO_UNICODE_STRING AppendLongLongIntegerToUnicodeString;

_Use_decl_annotations_
BOOLEAN
AppendLongLongIntegerToUnicodeString(
    PUNICODE_STRING String,
    ULONGLONG Integer,
    USHORT NumberOfDigits,
    WCHAR Trailer
    )
/*++

Routine Description:

    This is a helper routine that allows construction of unicode strings out
    of integer values.

Arguments:

    String - Supplies a pointer to a UNICODE_STRING that will be appended to.
        Sufficient buffer space must exist for the entire string to be written.

    Integer - Supplies the long long integer value to be appended to the string.

    NumberOfDigits - The expected number of digits for the value.  If Integer
        has less digits than this number, it will be left-padded with zeros.

    Trailer - An optional trailing wide character to append.

Return Value:

    TRUE on success, FALSE on failure.

--*/
{
    USHORT ActualNumberOfDigits;
    USHORT BytesRequired;
    USHORT BytesRemaining;
    USHORT NumberOfZerosToPad;
    ULONGLONG Digit;
    ULONGLONG Value;
    ULONGLONG Count;
    ULONGLONG Bytes;
    WCHAR Char;
    PWCHAR Dest;

    //
    // Verify the unicode string has sufficient space.
    //

    BytesRequired = NumberOfDigits * sizeof(WCHAR);

    if (Trailer) {
        BytesRequired += (1 * sizeof(Trailer));
    }

    BytesRemaining = (
        String->MaximumLength -
        String->Length
    );

    if (BytesRemaining < BytesRequired) {
        return FALSE;
    }

    //
    // Make sure the integer value doesn't have more digits than
    // specified.
    //

    ActualNumberOfDigits = CountNumberOfLongLongDigitsInline(Integer);

    if (ActualNumberOfDigits > NumberOfDigits) {
        return FALSE;
    }

    //
    // Initialize our destination pointer to the last digit.  (We write
    // back-to-front.)
    //

    Dest = (PWCHAR)(
        RtlOffsetToPointer(
            String->Buffer,
            String->Length + (
                (NumberOfDigits - 1) *
                sizeof(WCHAR)
            )
        )
    );
    Count = 0;
    Bytes = 0;

    //
    // Convert each digit into the corresponding character and copy to the
    // string buffer, retreating the pointer as we go.
    //

    Value = Integer;

    do {
        Count++;
        Bytes += 2;
        Digit = Value % 10;
        Value = Value / 10;
        Char = IntegerToWCharTable[Digit];
        *Dest-- = Char;
    } while (Value != 0);

    //
    // Pad the string with zeros if necessary.
    //

    NumberOfZerosToPad = NumberOfDigits - ActualNumberOfDigits;

    if (NumberOfZerosToPad) {
        do {
            Count++;
            Bytes += 2;
            *Dest-- = L'0';
        } while (--NumberOfZerosToPad);
    }

    //
    // Update the string with the new length.
    //

    String->Length += (USHORT)Bytes;

    //
    // Add the trailer if applicable.
    //

    if (Trailer) {
        String->Length += sizeof(WCHAR);
        String->Buffer[(String->Length - 1) / sizeof(WCHAR)] = Trailer;
    }

    return TRUE;
}


APPEND_INTEGER_TO_STRING AppendIntegerToString;

_Use_decl_annotations_
BOOLEAN
AppendIntegerToString(
    PSTRING String,
    ULONG Integer,
    USHORT NumberOfDigits,
    CHAR Trailer
    )
/*++

Routine Description:

    This is a helper routine that allows construction of STRING structures out
    of integer values.

Arguments:

    String - Supplies a pointer to a STRING that will be appended to.
        Sufficient buffer space must exist for the entire string to
        be written.

    Integer - The integer value to be appended to the string.

    NumberOfDigits - The expected number of digits for the value.  If Integer
        has less digits than this number, it will be left-padded with zeros.

    Trailer - An optional trailing character to append.

Return Value:

    TRUE on success, FALSE on failure.

--*/
{
    USHORT ActualNumberOfDigits;
    USHORT BytesRequired;
    USHORT BytesRemaining;
    USHORT NumberOfZerosToPad;
    ULONG Digit;
    ULONG Value;
    ULONG Bytes;
    CHAR Char;
    PCHAR Dest;

    //
    // Verify the unicode string has sufficient space.
    //

    BytesRequired = NumberOfDigits * sizeof(CHAR);

    if (Trailer) {
        BytesRequired += (1 * sizeof(Trailer));
    }

    BytesRemaining = (
        String->MaximumLength -
        String->Length
    );

    if (BytesRemaining < BytesRequired) {
        return FALSE;
    }

    //
    // Make sure the integer value doesn't have more digits than
    // specified.
    //

    ActualNumberOfDigits = CountNumberOfDigitsInline(Integer);

    if (ActualNumberOfDigits > NumberOfDigits) {
        return FALSE;
    }

    //
    // Initialize our destination pointer to the last digit.  (We write
    // back-to-front.)
    //

    Dest = (PCHAR)(
        RtlOffsetToPointer(
            String->Buffer,
            String->Length + (
                (NumberOfDigits - 1) *
                sizeof(CHAR)
            )
        )
    );
    Bytes = 0;

    //
    // Convert each digit into the corresponding character and copy to the
    // string buffer, retreating the pointer as we go.
    //

    Value = Integer;

    do {
        Bytes++;
        Digit = Value % 10;
        Value = Value / 10;
        Char = IntegerToCharTable[Digit];
        *Dest-- = Char;
    } while (Value != 0);

    //
    // Pad the string with zeros if necessary.
    //

    NumberOfZerosToPad = NumberOfDigits - ActualNumberOfDigits;

    if (NumberOfZerosToPad) {
        do {
            Bytes++;
            *Dest-- = '0';
        } while (--NumberOfZerosToPad);
    }

    //
    // Update the string with the new length.
    //

    String->Length += (USHORT)Bytes;

    //
    // Add the trailer if applicable.
    //

    if (Trailer) {
        String->Length += sizeof(CHAR);
        String->Buffer[String->Length - 1] = Trailer;
    }

    return TRUE;
}


APPEND_LONGLONG_INTEGER_TO_STRING AppendLongLongIntegerToString;

_Use_decl_annotations_
BOOLEAN
AppendLongLongIntegerToString(
    PSTRING String,
    ULONGLONG Integer,
    USHORT NumberOfDigits,
    CHAR Trailer
    )
/*++

Routine Description:

    This is a helper routine that allows construction of STRING structures out
    of integer values.

Arguments:

    String - Supplies a pointer to a STRING that will be appended to.
        Sufficient buffer space must exist for the entire string to be
        written.

    Integer - Supplies the long long integer value to be appended to the string.

    NumberOfDigits - The expected number of digits for the value.  If Integer
        has less digits than this number, it will be left-padded with zeros.

    Trailer - An optional trailing character to append.

Return Value:

    TRUE on success, FALSE on failure.

--*/
{
    USHORT ActualNumberOfDigits;
    USHORT BytesRequired;
    USHORT BytesRemaining;
    USHORT NumberOfZerosToPad;
    ULONGLONG Digit;
    ULONGLONG Value;
    ULONGLONG Bytes;
    CHAR Char;
    PCHAR Dest;

    //
    // Verify the unicode string has sufficient space.
    //

    BytesRequired = NumberOfDigits * sizeof(CHAR);

    if (Trailer) {
        BytesRequired += (1 * sizeof(Trailer));
    }

    BytesRemaining = (
        String->MaximumLength -
        String->Length
    );

    if (BytesRemaining < BytesRequired) {
        return FALSE;
    }

    //
    // Make sure the integer value doesn't have more digits than
    // specified.
    //

    ActualNumberOfDigits = CountNumberOfLongLongDigitsInline(Integer);

    if (ActualNumberOfDigits > NumberOfDigits) {
        return FALSE;
    }

    //
    // Initialize our destination pointer to the last digit.  (We write
    // back-to-front.)
    //

    Dest = (PCHAR)(
        RtlOffsetToPointer(
            String->Buffer,
            String->Length + (
                (NumberOfDigits - 1) *
                sizeof(CHAR)
            )
        )
    );
    Bytes = 0;

    //
    // Convert each digit into the corresponding character and copy to the
    // string buffer, retreating the pointer as we go.
    //

    Value = Integer;

    do {
        Bytes++;
        Digit = Value % 10;
        Value = Value / 10;
        Char = IntegerToCharTable[Digit];
        *Dest-- = Char;
    } while (Value != 0);

    //
    // Pad the string with zeros if necessary.
    //

    NumberOfZerosToPad = NumberOfDigits - ActualNumberOfDigits;

    if (NumberOfZerosToPad) {
        do {
            Bytes++;
            *Dest-- = '0';
        } while (--NumberOfZerosToPad);
    }

    //
    // Update the string with the new length.
    //

    String->Length += (USHORT)Bytes;

    //
    // Add the trailer if applicable.
    //

    if (Trailer) {
        String->Length += sizeof(CHAR);
        String->Buffer[String->Length - 1] = Trailer;
    }

    return TRUE;
}

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

_Use_decl_annotations_
VOID
AppendSignedIntegerToCharBuffer(
    PCHAR *BufferPointer,
    LONGLONG Integer
    )
{
    PCHAR Buffer;
    ULONGLONG Positive;

    if (Integer < 0) {
        Buffer = *BufferPointer;
        *Buffer++ = '-';
        *BufferPointer += 1;
        Positive = (ULONGLONG)(Integer * -1LL);
    } else {
        Positive = (ULONGLONG)Integer;
    }

    AppendIntegerToCharBuffer(BufferPointer, Positive);
}

//
// Due to historical reluctance to use the CRT on Windows (vestige of the tracer
// project), we have our own double-to-string conversion routine.  This has
// worked fine on Windows, but on Linux, the _dtoa() routine sometimes gets
// stuck in an infinite loop.  So, on Linux (PH_COMPAT), just use snprintf().
//

#ifdef PH_WINDOWS

APPEND_DOUBLE_TO_CHAR_BUFFER AppendDoubleToCharBuffer;

_Use_decl_annotations_
VOID
AppendDoubleToCharBuffer(
    PCHAR *BufferPointer,
    DOUBLE Double
    )
{
    PCHAR Buffer;
    PCHAR End;
    PCHAR Result;
    PCHAR AfterDecimal;
    INT Count;
    INT Mode;
    INT Sign;
    INT Length;
    INT DecimalPoint;
    INT NumberOfDigits;
    INT VirtualDigitsStart;
    INT VirtualDigitsEnd;
    SIZE_T ResultStrlen;
    ULONG_PTR ResultLength;

    End = NULL;
    Sign = 0;
    DecimalPoint = 0;
    NumberOfDigits = 9;

    Buffer = *BufferPointer;

    //
    // Fast path for 0 and 1.
    //

    if (Double == 0.0) {
        AppendCStrToCharBuffer(BufferPointer, "0.0");
        return;
    } else if (Double == 1.0) {
        AppendCStrToCharBuffer(BufferPointer, "1.0");
        return;
    }

    //
    // Mode = 0 -> Shortest string that yields the double when read in and
    // rounded to the nearest value.
    //

    Mode = 0;

    Result = _dtoa(
        Double,
        Mode,
        NumberOfDigits,
        &DecimalPoint,
        &Sign,
        &End
    );

    if (Result == NULL) {
        return;
    }

    ResultLength = RtlPointerToOffset(Result, End);
    if (ResultLength > 0x7fff) {
        goto End;
    }

    if (DecimalPoint == 9999) {
        *Buffer++ = 'i';
        *Buffer++ = 'n';
        *Buffer++ = 'f';
        goto End;
    }

    ResultStrlen = strlen(Result);
    ASSERT((ULONG_PTR)ResultStrlen == ResultLength);

    Length = (INT)ResultLength;
    VirtualDigitsEnd = DecimalPoint + NumberOfDigits;
    if (VirtualDigitsEnd <= DecimalPoint) {
        VirtualDigitsEnd = DecimalPoint + 1;
    }
    VirtualDigitsStart = DecimalPoint <= 0 ? DecimalPoint - 1 : 0;

    ASSERT(VirtualDigitsStart <= 0);
    ASSERT(0 <= Length);

    if (Sign == 1) {
        *Buffer++ = '-';
    }

    //
    // Zero padding on the left.
    //

    if (DecimalPoint <= 0) {
        Count = DecimalPoint - VirtualDigitsStart;
        if (Count > 0) {
            while (Count--) {
                *Buffer++ = '0';
            }
        }
        *Buffer++ = '.';
        Count = 0 - DecimalPoint;
        if (Count > 0) {
            while (Count--) {
                *Buffer++ = '0';
            }
        }
    } else {
        Count = 0 - VirtualDigitsStart;
        if (Count > 0) {
            while (Count--) {
                *Buffer++ = '0';
            }
        }
    }

    //
    // Digits, with included decimal point.
    //

    if (DecimalPoint > 0 && DecimalPoint < Length) {
        AppendCharBufferToCharBuffer(&Buffer, Result, DecimalPoint);
        *Buffer++ = '.';
        AfterDecimal = Result + DecimalPoint;
        AppendCharBufferToCharBuffer(&Buffer,
                                     AfterDecimal,
                                     Length-DecimalPoint);
    } else {
        AppendCharBufferToCharBuffer(&Buffer, Result, Length);
    }

    //
    // Add zeros on the right.
    //

    if (Length < DecimalPoint) {
        Count = DecimalPoint - Length;
        if (Count > 0) {
            while (Count--) {
                *Buffer++ = '0';
            }
        }
        *Buffer += '.';
        Count = VirtualDigitsEnd - DecimalPoint;
        if (Count > 0) {
            while (Count--) {
                *Buffer++ = '0';
            }
        }
    } else {
        Count = VirtualDigitsEnd - Length;
        if (Count > 0) {
            while (Count--) {
                *Buffer++ = '0';
            }
        }
    }

End:
    ASSERT(Result != NULL);
    _freedtoa(Result);

    *BufferPointer = Buffer;

    return;
}

#else // PH_WINDOWS

APPEND_DOUBLE_TO_CHAR_BUFFER AppendDoubleToCharBuffer;

_Use_decl_annotations_
VOID
AppendDoubleToCharBuffer(
    PCHAR *BufferPointer,
    DOUBLE Double
    )
{
    INT Count;
    PCHAR Buffer;

    Buffer = *BufferPointer;

    Count = sprintf(Buffer, "%f", Double);

    //
    // Update the buffer pointer based on the number of characters written,
    // excluding the trailing NUL.
    //

    ASSERT(Buffer[Count] == 0);
    *BufferPointer = Buffer + (Count - 1);
}

#endif // !PH_WINDOWS


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
    in hexadecimal format, left-padding the string with spaces as necessary
    such that 10 bytes are always consumed, e.g.:

        "    0x1207"
        "  0x251081"
        "0x20180823"

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

    *Dest-- = 'x';
    *Dest-- = '0';

    Count += 2;

    Pad = 10 - Count;
    while (Pad) {
        *Dest-- = ' ';
        Pad--;
    }

    *BufferPointer = End + 1;

    return;
}


APPEND_LONGLONG_INTEGER_TO_CHAR_BUFFER_AS_HEX
    AppendLongLongIntegerToCharBufferAsHex;

_Use_decl_annotations_
VOID
AppendLongLongIntegerToCharBufferAsHex(
    PCHAR *BufferPointer,
    ULONGLONG Integer
    )
/*++

Routine Description:

    A 64-bit version of AppendIntegerToCharBufferAsHex().

Arguments:

    BufferPointer - Supplies a pointer to a variable that contains the address
        of a character buffer to which the hex string representation of the
        integer will be written.  The pointer is adjusted to point after the
        length of the written bytes prior to returning.  This will always be
        20 characters/bytes.

    Integer - Supplies the integer value to be appended to the string.

Return Value:

    None.

--*/
{
    CHAR Char;
    ULONG Pad;
    ULONG Count;
    ULONG Digit;
    ULONGLONG Value;
    PCHAR End;
    PCHAR Dest;
    PCHAR Buffer;

    Buffer = *BufferPointer;

    End = Dest = RtlOffsetToPointer(Buffer, 19);

    Count = 0;
    Value = Integer;

    do {
        Count++;
        Digit = (ULONG)(Value & 0xf);
        Value >>= 4;
        Char = IntegerToCharTable[Digit];
        *Dest-- = Char;
    } while (Value != 0);

    *Dest-- = 'x';
    *Dest-- = '0';

    Count += 2;

    Pad = 20 - Count;
    while (Pad) {
        *Dest-- = ' ';
        Pad--;
    }

    *BufferPointer = End + 1;

    return;
}


APPEND_INTEGER_TO_CHAR_BUFFER_AS_HEX_RAW AppendIntegerToCharBufferAsHexRaw;

_Use_decl_annotations_
VOID
AppendIntegerToCharBufferAsHexRaw(
    PCHAR *BufferPointer,
    ULONG Integer
    )
/*++

Routine Description:

    This is a helper routine that appends an integer to a character buffer
    in hexadecimal format.  No 0x or additional padding is added to the string.

Arguments:

    BufferPointer - Supplies a pointer to a variable that contains the address
        of a character buffer to which the hex string representation of the
        integer will be written.

    Integer - Supplies the integer value to be appended to the string.

Return Value:

    None.

--*/
{
    CHAR Char;
    ULONG Count;
    ULONG Digit;
    ULONG Value;
    PCHAR End;
    PCHAR Dest;
    PCHAR Buffer;
    BYTE NumberOfHexChars;

    Buffer = *BufferPointer;

    NumberOfHexChars = CountNumberOfHexCharsInline(Integer);

    End = Dest = RtlOffsetToPointer(Buffer, NumberOfHexChars - 1);

    Count = 0;
    Value = Integer;

    do {
        Count++;
        Digit = Value & 0xf;
        Value >>= 4;
        Char = IntegerToCharTable[Digit];
        *Dest-- = Char;
    } while (Value != 0);

    *BufferPointer = End + 1;

    return;
}


APPEND_LONGLONG_INTEGER_TO_CHAR_BUFFER_AS_HEX_RAW
    AppendLongLongIntegerToCharBufferAsHexRaw;

_Use_decl_annotations_
VOID
AppendLongLongIntegerToCharBufferAsHexRaw(
    PCHAR *BufferPointer,
    ULONGLONG Integer
    )
/*++

Routine Description:

    This is a helper routine that appends 64-bit integer to a character buffer
    in hexadecimal format.  No 0x or additional padding is added to the string.

Arguments:

    BufferPointer - Supplies a pointer to a variable that contains the address
        of a character buffer to which the hex string representation of the
        integer will be written.

    Integer - Supplies the 64-bit integer value to be appended to the string.

Return Value:

    None.

--*/
{
    CHAR Char;
    ULONG Count;
    ULONG Digit;
    ULONGLONG Value;
    PCHAR End;
    PCHAR Dest;
    PCHAR Buffer;
    BYTE NumberOfHexChars;

    Buffer = *BufferPointer;

    NumberOfHexChars = CountNumberOfHex64CharsInline(Integer);

    End = Dest = RtlOffsetToPointer(Buffer, NumberOfHexChars - 1);

    Count = 0;
    Value = Integer;

    do {
        Count++;
        Digit = (ULONG)(Value & 0xf);
        Value >>= 4;
        Char = IntegerToCharTable[Digit];
        *Dest-- = Char;
    } while (Value != 0);

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
AppendUnicodeStringToCharBufferFast(
    PCHAR *BufferPointer,
    PCUNICODE_STRING String
    )
{
    CHAR Char;
    WCHAR Wide;
    PWSTR Source;
    PSTR Dest;
    USHORT Count;
    USHORT Index;

    if (String == NULL || String->Buffer == NULL || String->Length == 0) {
        return;
    }

    Count = String->Length / sizeof(WCHAR);
    Source = String->Buffer;
    Dest = *BufferPointer;

    for (Index = 0; Index < Count; Index++) {
        Wide = Source[Index];
        Char = (CHAR)Wide;
        Dest[Index] = Char;
    }

    *BufferPointer = RtlOffsetToPointer(Dest, Count);

    return;
}

_Use_decl_annotations_
VOID
AppendStringToWideCharBufferFast(
    PWCHAR *BufferPointer,
    PCSTRING String
    )
{
    CHAR Char;
    WCHAR Wide;
    PSTR Source;
    PWSTR Dest;
    USHORT Count;
    USHORT Index;

    Count = String->Length;
    Source = String->Buffer;
    Dest = *BufferPointer;

    for (Index = 0; Index < Count; Index++) {
        Char = Source[Index];
        Wide = (WCHAR)Char;
        Dest[Index] = Wide;
    }

    *BufferPointer = (PWCHAR)(
        RtlOffsetToPointer(
            Dest,
            (ULONG_PTR)Count * sizeof(WCHAR)
        )
    );

    return;
}

_Use_decl_annotations_
HRESULT
AppendStringToUnicodeStringFast (
    PCSTRING String,
    PUNICODE_STRING UnicodeString
    )
{
    CHAR Char;
    WCHAR Wide;
    PSTR Source;
    PWSTR Dest;
    USHORT Count;
    USHORT Index;

    if (String->Length > (UnicodeString->MaximumLength / sizeof(WCHAR))) {
        return PH_E_STRING_BUFFER_TOO_SMALL;
    }

    Count = String->Length;
    Source = String->Buffer;
    Dest = UnicodeString->Buffer;

    for (Index = 0; Index < Count; Index++) {
        Char = Source[Index];
        Wide = (WCHAR)Char;
        Dest[Index] = Wide;
    }

    UnicodeString->Length = (USHORT)(Count * sizeof(WCHAR));

    return S_OK;
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
AppendWStrToCharBufferFast(
    PCHAR *BufferPointer,
    PWSTR String
    )
{
    CHAR Char;
    WCHAR Wide;
    PCHAR Dest = *BufferPointer;
    PWSTR Source = (PWSTR)String;

    while (*Source) {
        Wide = *Source++;
        Char = (CHAR)Wide;
        *Dest++ = Char;
    }

    *BufferPointer = Dest;

    return;
}

_Use_decl_annotations_
VOID
AppendErrorCodeCStrToCharBuffer(
    PRTL Rtl,
    PCHAR *BufferPointer,
    HRESULT Code
    )
{
    PCHAR Dest = *BufferPointer;
    PCHAR Source;
    HRESULT Result;

    Result = PerfectHashGetErrorCodeString(Rtl, Code, &Source);
    if (FAILED(Result)) {
        PH_ERROR(PerfectHashGetErrorCodeString, Result);
        PH_RAISE(Result);
    }

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

_Use_decl_annotations_
VOID
Crc32HashString(
    PSTRING String
    )
{
    BYTE TrailingBytes;
    ULONG Hash;
    ULONG Index;
    PULONG DoubleWord;
    ULONG NumberOfDoubleWords;

    //
    // Calculate the string hash.
    //

    Hash = String->Length;
    DoubleWord = (PULONG)String->Buffer;
    NumberOfDoubleWords = String->Length >> 2;
    TrailingBytes = (BYTE)(String->Length - (NumberOfDoubleWords << 2));

    if (NumberOfDoubleWords) {

        //
        // Process as many 4 byte chunks as we can.
        //

        for (Index = 0; Index < NumberOfDoubleWords; Index++) {
            Hash = _mm_crc32_u32(Hash, *DoubleWord++);
        }
    }

    if (TrailingBytes) {

#ifdef PERFECTHASH_CPU_BMI2

        //
        // There are between 1 and 3 bytes remaining at the end of the string.
        // We can't use _mm_crc32_u32() here directly on the last ULONG as we
        // will include the bytes past the end of the string, which will be
        // random and will affect our hash value.  So, we load the last ULONG
        // then zero out the high bits that we want to ignore via _bzhi_u32().
        // This ensures that only the bytes that are part of the input string
        // participate in the hash value calculation.
        //

        ULONG Last = 0;
        ULONG HighBits;

        //
        // (Sanity check we can math.)
        //

        ASSERT(TrailingBytes >= 1 && TrailingBytes <= 3);

        //
        // Initialize our HighBits to the number of bits in a ULONG (32),
        // then subtract the number of bits represented by TrailingBytes.
        //

        HighBits = sizeof(ULONG) << 3;
        HighBits -= (TrailingBytes << 3);

        //
        // Load the last ULONG, zero out the high bits, then hash.
        //

        Last = _bzhi_u32(*DoubleWord, HighBits);
        Hash = _mm_crc32_u32(Hash, Last);

#else
        PBYTE Byte;

        Byte = (PBYTE)DoubleWord;
        while (TrailingBytes) {
            Hash = _mm_crc32_u8(Hash, *Byte);
            Byte++;
            TrailingBytes--;
        }

#endif
    }

    //
    // Save the hash and return.
    //

    String->Hash = Hash;

    return;
}

_Use_decl_annotations_
VOID
Crc32HashUnicodeString(
    PUNICODE_STRING String
    )
{
    Crc32HashString((PSTRING)String);
}

#ifdef PH_WINDOWS
size_t __cdecl __imp_wcslen(_In_z_ wchar_t const *s)
{
    const wchar_t *p = s;
    while (*p)
        ++p;
    return p - s;
}
#endif

size_t __cdecl strlen(_In_z_ char const *s)
{
    const char *p = s;
    while (*p)
        ++p;
    return p - s;
}

#if 0
size_t __cdecl wcslen(_In_z_ wchar_t const *s)
{
    const wchar_t *p = s;
    while (*p)
        ++p;
    return p - s;
}
#endif

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
