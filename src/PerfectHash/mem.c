/*
 * COPYRIGHT:       See COPYING in the top level directory
 * PROJECT:         ReactOS system libraries
 * FILE:            lib/rtl/mem.c
 * PURPOSE:         Memory functions
 * PROGRAMMER:      David Welch (welch@mcmail.com)
 */

/* INCLUDES *****************************************************************/

#include "stdafx.h"

/* FUNCTIONS *****************************************************************/

/******************************************************************************
 *  RtlCompareMemory   [NTDLL.@]
 *
 * Compare one block of memory with another
 *
 * PARAMS
 *  Source1 [I] Source block
 *  Source2 [I] Block to compare to Source1
 *  Length  [I] Number of bytes to fill
 *
 * RETURNS
 *  The length of the first byte at which Source1 and Source2 differ, or Length
 *  if they are the same.
 *
 * @implemented
 */
SIZE_T
RtlCompareMemory(_In_ const VOID *Source1,
                 _In_ const VOID *Source2,
                 _In_ SIZE_T Length)
{
    PBYTE Left = (PBYTE)Source1;
    PBYTE Right = (PBYTE)Source2;
    SIZE_T Index;
    for (Index = 0;
         (Index < Length) && (Left[Index] == Right[Index]);
         Index++);

    return Index;
}

/*
 * FUNCTION: Compares a block of ULONGs with an ULONG and returns the number of equal bytes
 * ARGUMENTS:
 *      Source = Block to compare
 *      Length = Number of bytes to compare
 *      Value = Value to compare
 * RETURNS: Number of equal bytes
 *
 * @implemented
 */
SIZE_T
NTAPI
RtlCompareMemoryUlong(IN PVOID Source,
                      IN SIZE_T Length,
                      IN ULONG Value)
{
    PULONG ptr = (PULONG)Source;
    ULONG_PTR len = Length / sizeof(ULONG);
    ULONG_PTR i;

    for (i = 0; i < len; i++)
    {
        if (*ptr != Value)
            break;

        ptr++;
    }

    return (SIZE_T)((PCHAR)ptr - (PCHAR)Source);
}


#undef RtlFillMemory
/*
 * @implemented
 */
VOID
NTAPI
RtlFillMemory(PVOID Destination,
              SIZE_T Length,
              UCHAR Fill)
{
    memset(Destination, Fill, Length);
}


/*
 * @implemented
 */
VOID
NTAPI
RtlFillMemoryUlong(PVOID Destination,
                   SIZE_T Length,
                   ULONG Fill)
{
    PULONG Dest  = Destination;
    SIZE_T Count = Length / sizeof(ULONG);

    while (Count > 0)
    {
        *Dest = Fill;
        Dest++;
        Count--;
    }
}

VOID
NTAPI
RtlFillMemoryUlonglong(
    PVOID Destination,
    SIZE_T Length,
    ULONGLONG Fill)
{
    PULONGLONG Dest  = Destination;
    SIZE_T Count = Length / sizeof(ULONGLONG);

    while (Count > 0)
    {
        *Dest = Fill;
        Dest++;
        Count--;
    }
}

#undef RtlMoveMemory
/*
 * @implemented
 */
VOID
NTAPI
RtlMoveMemory(
    _Out_writes_bytes_all_(Length) PVOID Destination,
    _In_ const PVOID Source,
    _In_ ULONG_PTR Length
    )
{
    memmove(Destination, Source, Length);
}

#undef RtlZeroMemory
/*
 * @implemented
 */
VOID
NTAPI
RtlZeroMemory(PVOID Destination,
              SIZE_T Length)
{
    RtlFillMemory(Destination, Length, 0);
}

/* EOF */
