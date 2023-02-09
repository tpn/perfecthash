/*
 * COPYRIGHT:         See COPYING in the top level directory
 * PROJECT:           ReactOS system libraries
 * PURPOSE:           Unicode Conversion Routines
 * FILE:              lib/rtl/unicode.c
 * PROGRAMMER:        Alex Ionescu (alex@relsoft.net)
 *                    Emanuele Aliberti
 *                    Gunnar Dalsnes
 */

/* INCLUDES *****************************************************************/

#include "stdafx.h"

/* GLOBALS *******************************************************************/


/* FUNCTIONS *****************************************************************/


NTSTATUS
NTAPI
RtlUnicodeStringToInteger(
    const UNICODE_STRING *str, /* [I] Unicode string to be converted */
    ULONG base,                /* [I] Number base for conversion (allowed 0, 2, 8, 10 or 16) */
    ULONG *value)              /* [O] Destination for the converted value */
{
    LPWSTR lpwstr = str->Buffer;
    USHORT CharsRemaining = str->Length / sizeof(WCHAR);
    WCHAR wchCurrent;
    int digit;
    ULONG RunningTotal = 0;
    char bMinus = 0;

    while (CharsRemaining >= 1 && *lpwstr <= ' ')
    {
        lpwstr++;
        CharsRemaining--;
    }

    if (CharsRemaining >= 1)
    {
        if (*lpwstr == '+')
        {
            lpwstr++;
            CharsRemaining--;
        }
        else if (*lpwstr == '-')
        {
            bMinus = 1;
            lpwstr++;
            CharsRemaining--;
        }
    }

    if (base == 0)
    {
        base = 10;

        if (CharsRemaining >= 2 && lpwstr[0] == '0')
        {
            if (lpwstr[1] == 'b')
            {
                lpwstr += 2;
                CharsRemaining -= 2;
                base = 2;
            }
            else if (lpwstr[1] == 'o')
            {
                lpwstr += 2;
                CharsRemaining -= 2;
                base = 8;
            }
            else if (lpwstr[1] == 'x')
            {
                lpwstr += 2;
                CharsRemaining -= 2;
                base = 16;
            }
        }
    }
    else if (base != 2 && base != 8 && base != 10 && base != 16)
    {
        return STATUS_INVALID_PARAMETER;
    }

    if (value == NULL)
    {
        return STATUS_ACCESS_VIOLATION;
    }

    while (CharsRemaining >= 1)
    {
        wchCurrent = *lpwstr;

        if (wchCurrent >= '0' && wchCurrent <= '9')
        {
            digit = wchCurrent - '0';
        }
        else if (wchCurrent >= 'A' && wchCurrent <= 'Z')
        {
            digit = wchCurrent - 'A' + 10;
        }
        else if (wchCurrent >= 'a' && wchCurrent <= 'z')
        {
            digit = wchCurrent - 'a' + 10;
        }
        else
        {
            digit = -1;
        }

        if (digit < 0 || (ULONG)digit >= base) break;

        RunningTotal = RunningTotal * base + digit;
        lpwstr++;
        CharsRemaining--;
    }

    *value = bMinus ? (0 - RunningTotal) : RunningTotal;
    return STATUS_SUCCESS;
}


NTSTATUS
NTAPI
RtlUnicodeStringToInt64(
    _In_ PCUNICODE_STRING String,
    _In_ ULONG Base,
    _Out_ PLONG64 Number,
    _Out_opt_ PWSTR *EndPointer
    )
{
    LPWSTR lpwstr = String->Buffer;
    USHORT CharsRemaining = String->Length / sizeof(WCHAR);
    WCHAR wchCurrent;
    int digit;
    ULONG64 RunningTotal = 0;
    char bMinus = 0;

    //
    // We don't support endPointer.
    //

    ASSERT(EndPointer == NULL);

    while (CharsRemaining >= 1 && *lpwstr <= ' ')
    {
        lpwstr++;
        CharsRemaining--;
    }

    if (CharsRemaining >= 1)
    {
        if (*lpwstr == '+')
        {
            lpwstr++;
            CharsRemaining--;
        }
        else if (*lpwstr == '-')
        {
            bMinus = 1;
            lpwstr++;
            CharsRemaining--;
        }
    }

    if (Base == 0)
    {
        Base = 10;

        if (CharsRemaining >= 2 && lpwstr[0] == '0')
        {
            if (lpwstr[1] == 'b')
            {
                lpwstr += 2;
                CharsRemaining -= 2;
                Base = 2;
            }
            else if (lpwstr[1] == 'o')
            {
                lpwstr += 2;
                CharsRemaining -= 2;
                Base = 8;
            }
            else if (lpwstr[1] == 'x')
            {
                lpwstr += 2;
                CharsRemaining -= 2;
                Base = 16;
            }
        }
    }
    else if (Base != 2 && Base != 8 && Base != 10 && Base != 16)
    {
        return STATUS_INVALID_PARAMETER;
    }

    if (Number == NULL)
    {
        return STATUS_ACCESS_VIOLATION;
    }

    while (CharsRemaining >= 1)
    {
        wchCurrent = *lpwstr;

        if (wchCurrent >= '0' && wchCurrent <= '9')
        {
            digit = wchCurrent - '0';
        }
        else if (wchCurrent >= 'A' && wchCurrent <= 'Z')
        {
            digit = wchCurrent - 'A' + 10;
        }
        else if (wchCurrent >= 'a' && wchCurrent <= 'z')
        {
            digit = wchCurrent - 'a' + 10;
        }
        else
        {
            digit = -1;
        }

        if (digit < 0 || (ULONG)digit >= Base) break;

        RunningTotal = RunningTotal * Base + digit;
        lpwstr++;
        CharsRemaining--;
    }

    *Number = bMinus ? (0 - RunningTotal) : RunningTotal;
    return STATUS_SUCCESS;
}

/*
 * @implemented
 *
 * NOTES
 *  If src->length is zero dest is unchanged.
 *  Dest is never nullterminated.
 */
NTSTATUS
NTAPI
RtlAppendStringToString(IN PSTRING Destination,
                        IN const STRING *Source)
{
    USHORT SourceLength = Source->Length;

    if (SourceLength)
    {
        if (Destination->Length + SourceLength > Destination->MaximumLength)
        {
            return STATUS_BUFFER_TOO_SMALL;
        }

        RtlMoveMemory(&Destination->Buffer[Destination->Length],
                      Source->Buffer,
                      SourceLength);

        Destination->Length += SourceLength;
    }

    return STATUS_SUCCESS;
}

/*
 * @implemented
 *
 * NOTES
 *  If src->length is zero dest is unchanged.
 *  Dest is nullterminated when the MaximumLength allowes it.
 *  When dest fits exactly in MaximumLength characters the nullterm is ommitted.
 */
NTSTATUS
NTAPI
RtlAppendUnicodeStringToString(
    IN OUT PUNICODE_STRING Destination,
    IN PCUNICODE_STRING Source)
{
    USHORT SourceLength = Source->Length;
    PWCHAR Buffer = &Destination->Buffer[Destination->Length / sizeof(WCHAR)];

    if (SourceLength)
    {
        if ((SourceLength + Destination->Length) > Destination->MaximumLength)
        {
            return STATUS_BUFFER_TOO_SMALL;
        }

        RtlMoveMemory(Buffer, Source->Buffer, SourceLength);
        Destination->Length += SourceLength;

        /* append terminating '\0' if enough space */
        if (Destination->MaximumLength > Destination->Length)
        {
            Buffer[SourceLength / sizeof(WCHAR)] = UNICODE_NULL;
        }
    }

    return STATUS_SUCCESS;
}

LONG
NTAPI
RtlCompareUnicodeString(
    IN PCUNICODE_STRING s1,
    IN PCUNICODE_STRING s2,
    IN BOOLEAN  CaseInsensitive)
{
    unsigned int len;
    LONG ret = 0;
    LPCWSTR p1, p2;

    len = min(s1->Length, s2->Length) / sizeof(WCHAR);
    p1 = s1->Buffer;
    p2 = s2->Buffer;

    ASSERT(CaseInsensitive == FALSE);

    while (!ret && len--) ret = *p1++ - *p2++;

    if (!ret) ret = s1->Length - s2->Length;

    return ret;
}

/*
 * @implemented
 *
 * RETURNS
 *  TRUE if strings are equal.
 */
BOOLEAN
NTAPI
RtlEqualUnicodeString(
    IN CONST UNICODE_STRING *s1,
    IN CONST UNICODE_STRING *s2,
    IN BOOLEAN  CaseInsensitive)
{
    if (s1->Length != s2->Length) return FALSE;
    return !RtlCompareUnicodeString(s1, s2, CaseInsensitive );
}

