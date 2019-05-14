/*++

Copyright (c) 2019 Trent Nelson <trent@trent.me>

Module Name:

    BitManipulation.h

Abstract:

    This header file defines function pointer types related to bit manipulation
    instructions that have hardware implementations in newer CPU variants.  E.g.
    counting leading zeros, trailing zeros, etc.

--*/

#pragma once

#include "stdafx.h"

//
// Function pointer typedefs.
//

typedef
ULONG
(LEADING_ZEROS_32)(
    _In_ ULONG Input
    );
typedef LEADING_ZEROS_32 *PLEADING_ZEROS_32;

typedef
ULONGLONG
(LEADING_ZEROS_64)(
    _In_ ULONGLONG Input
    );
typedef LEADING_ZEROS_64 *PLEADING_ZEROS_64;

typedef
ULONG_PTR
(LEADING_ZEROS_POINTER)(
    _In_ ULONG_PTR Input
    );
typedef LEADING_ZEROS_POINTER *PLEADING_ZEROS_POINTER;

typedef
ULONG
(TRAILING_ZEROS_32)(
    _In_ ULONG Input
    );
typedef TRAILING_ZEROS_32 *PTRAILING_ZEROS_32;

typedef
ULONGLONG
(TRAILING_ZEROS_64)(
    _In_ ULONGLONG Input
    );
typedef TRAILING_ZEROS_64 *PTRAILING_ZEROS_64;

typedef
ULONG_PTR
(TRAILING_ZEROS_POINTER)(
    _In_ ULONG_PTR Input
    );
typedef TRAILING_ZEROS_POINTER *PTRAILING_ZEROS_POINTER;

typedef
ULONG
(POPULATION_COUNT_32)(
    _In_ ULONG Input
    );
typedef POPULATION_COUNT_32 *PPOPULATION_COUNT_32;

typedef
ULONGLONG
(POPULATION_COUNT_64)(
    _In_ ULONGLONG Input
    );
typedef POPULATION_COUNT_64 *PPOPULATION_COUNT_64;

typedef
ULONG_PTR
(POPULATION_COUNT_POINTER)(
    _In_ ULONG_PTR Input
    );
typedef POPULATION_COUNT_POINTER *PPOPULATION_COUNT_POINTER;

typedef
ULONGLONG
(ROUND_UP_POWER_OF_TWO_32)(
    _In_ ULONG Input
    );
typedef ROUND_UP_POWER_OF_TWO_32 *PROUND_UP_POWER_OF_TWO_32;

typedef
ULONGLONG
(ROUND_UP_NEXT_POWER_OF_TWO_32)(
    _In_ ULONG Input
    );
typedef ROUND_UP_NEXT_POWER_OF_TWO_32 *PROUND_UP_NEXT_POWER_OF_TWO_32;

typedef
ULONGLONG
(ROUND_UP_POWER_OF_TWO_64)(
    _In_ ULONGLONG Input
    );
typedef ROUND_UP_POWER_OF_TWO_64 *PROUND_UP_POWER_OF_TWO_64;

typedef
ULONGLONG
(ROUND_UP_NEXT_POWER_OF_TWO_64)(
    _In_ ULONGLONG Input
    );
typedef ROUND_UP_NEXT_POWER_OF_TWO_64 *PROUND_UP_NEXT_POWER_OF_TWO_64;

typedef
ULONG_PTR
(ROUND_UP_NEXT_POWER_OF_TWO_POINTER)(
    _In_ ULONG_PTR Input
    );
typedef ROUND_UP_NEXT_POWER_OF_TWO_POINTER *PROUND_UP_NEXT_POWER_OF_TWO_POINTER;

//
// Forward decls.
//

#ifndef __INTELLISENSE__
extern LEADING_ZEROS_32 LeadingZeros32_C;
extern TRAILING_ZEROS_32 TrailingZeros32_C;
extern POPULATION_COUNT_32 PopulationCount32_C;
extern LEADING_ZEROS_64 LeadingZeros64_C;
extern TRAILING_ZEROS_64 TrailingZeros64_C;
extern POPULATION_COUNT_64 PopulationCount64_C;
extern LEADING_ZEROS_POINTER LeadingZerosPointer_C;;
extern TRAILING_ZEROS_POINTER TrailingZerosPointer_C;;
extern POPULATION_COUNT_POINTER PopulationCountPointer_C;
extern ROUND_UP_POWER_OF_TWO_32 RoundUpPowerOfTwo32_C;
extern ROUND_UP_NEXT_POWER_OF_TWO_32 RoundUpNextPowerOfTwo32_C;

extern LEADING_ZEROS_32 LeadingZeros32_LZCNT;
extern TRAILING_ZEROS_32 TrailingZeros32_BMI1;
extern POPULATION_COUNT_32 PopulationCount32_POPCNT;
extern LEADING_ZEROS_64 LeadingZeros64_LZCNT;
extern TRAILING_ZEROS_64 TrailingZeros64_BMI1;
extern POPULATION_COUNT_64 PopulationCount64_POPCNT;
extern LEADING_ZEROS_POINTER LeadingZerosPointer_POPCNT;
extern TRAILING_ZEROS_POINTER TrailingZerosPointer_POPCNT;
extern POPULATION_COUNT_POINTER PopulationCountPointer_POPCNT;
extern ROUND_UP_POWER_OF_TWO_32 RoundUpPowerOfTwo32_LZCNT;
extern ROUND_UP_NEXT_POWER_OF_TWO_32 RoundUpNextPowerOfTwo32_LZCNT;
#endif


#ifdef PERFECTHASH_CPU_LZCNT
#define LeadingZeros32 LeadingZeros32_LZCNT
#define LeadingZeros64 LeadingZeros64_LZCNT
#define LeadingZerosPointer LeadingZerosPointer_LZCNT
#define RoundUpPowerOfTwo32 RoundUpPowerOfTwo32_LZCNT
#define RoundUpNextPowerOfTwo32 RoundUpNextPowerOfTwo32_LZCNT
#else
#define LeadingZeros32 LeadingZeros32_C
#define LeadingZeros64 LeadingZeros64_C
#define LeadingZerosPointer LeadingZerosPointer_C
#define RoundUpPowerOfTwo32 RoundUpPowerOfTwo32_C
#define RoundUpNextPowerOfTwo32 RoundUpNextPowerOfTwo32_C
#endif

#ifdef PERFECTHASH_CPU_BMI1
#define TrailingZeros32 TrailingZeros32_BMI1
#define TrailingZeros64 TrailingZeros64_BMI1
#define TrailingZerosPointer TrailingZerosPointer_BMI1
#else
#define TrailingZeros32 TrailingZeros32_C
#define TrailingZeros64 TrailingZeros64_C
#define TrailingZerosPointer TrailingZerosPointer_C
#endif


//
// PopulationCount64 is defined as a macro in winnt.h.  Undefine it now.
//

#undef PopulationCount64

#ifdef PERFECTHASH_CPU_POPCNT
#define PopulationCount32 PopulationCount32_POPCNT
#define PopulationCount64 PopulationCount64_POPCNT
#define PopulationCountPointer PopulationCountPointer_POPCNT
#else
#define PopulationCount32 PopulationCount32_C
#define PopulationCount64 PopulationCount64_C
#define PopulationCountPointer PopulationCountPointer_C
#endif

//
// Helper functions for power-of-2 alignment.
//


//
// Inline helper for determing if a value is a power of 2.
//

FORCEINLINE
BOOLEAN
IsPowerOfTwo(
    _In_ ULONGLONG Value
    )
{
    if (Value <= 1) {
        return FALSE;
    }

    return ((Value & (Value - 1)) == 0);
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
