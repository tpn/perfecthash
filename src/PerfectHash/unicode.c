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

