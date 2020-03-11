/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashPrimes.h

Abstract:

    This is the header file for the prime number module of the perfect hash
    library.  It declares structures and functions related to the handling of
    prime numbers.

--*/

#pragma once

//
// Declare array of prime numbers.
//

extern const ULONGLONG Primes[];
extern const SHORT NumberOfPrimes;

//
// Helper for returning an index into the primes array.
//

FORCEINLINE
SHORT
FindIndexForFirstPrimeGreaterThanOrEqual(
    _In_ ULONGLONG Value
    )
{
    SHORT Index;

    for (Index = 0; Index < NumberOfPrimes; Index++) {
        if (Value < Primes[Index]) {
            return Index;
        }
    }

    return -1;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
