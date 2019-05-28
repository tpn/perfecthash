/*++

Copyright (c) 2019 Trent Nelson <trent@trent.me>

Module Name:

    BitManipulation.c

Abstract:

    Implementation of bit manipulation instructions in plain C and hardware
    intrinsics.

--*/

#include "stdafx.h"

//
// Define C implementations.
//

//
// LeadingZeros.
//

LEADING_ZEROS_32 LeadingZeros32_C;

_Use_decl_annotations_
ULONG
LeadingZeros32_C(
    ULONG Input
    )
{
    ULONG Count;

    if (Input == 0) {
        return 32;
    }

    Count = 0;

    if (Input <= 0x0000FFFF) {
        Count = Count + 16;
        Input = Input << 16;
    }

    if (Input <= 0x00FFFFFF) {
        Count = Count + 8;
        Input = Input << 8;
    }

    if (Input <= 0x0FFFFFFF) {
        Count = Count + 4;
        Input = Input << 4;
    }

    if (Input <= 0x3FFFFFFF) {
        Count = Count + 2;
        Input = Input << 2;
    }

    if (Input <= 0x7FFFFFFF) {
        Count = Count + 1;
    }

    return Count;
}


LEADING_ZEROS_64 LeadingZeros64_C;

_Use_decl_annotations_
ULONGLONG
LeadingZeros64_C(
    ULONGLONG Input
    )
{
    ULONG Count;
    ULARGE_INTEGER Large;

    if (Input == 0) {
        return 64;
    }

    Large.QuadPart = Input;

    if (Large.HighPart > 0) {
        Count = LeadingZeros32_C(Large.HighPart);
    } else {
        Count = LeadingZeros32_C(Large.LowPart);
        Count += 32;
    }

    return Count;
}

LEADING_ZEROS_POINTER LeadingZerosPointer_C;

_Use_decl_annotations_
ULONG_PTR
LeadingZerosPointer_C(
    ULONG_PTR Integer
    )
{
#ifdef _WIN64
    return LeadingZeros64_C((ULONGLONG)Integer);
#else
    return LeadingZeros32_C((ULONG)Integer);
#endif
}

//
// TrailingZeros.
//

TRAILING_ZEROS_32 TrailingZeros32_C;

ULONG
TrailingZeros32_C(
    ULONG Input
    )
{
    ULONG Count;
    ULONG Value;

    Value = Input;
    Value = ~Value & (Value - 1);
    Count = 0;

    while (Value != 0) {
        Count++;
        Value >>= 1;
    }

    return Count;
}

TRAILING_ZEROS_64 TrailingZeros64_C;

_Use_decl_annotations_
ULONGLONG
TrailingZeros64_C(
    ULONGLONG Input
    )
{
    ULONG Count;
    ULONGLONG Value;

    Value = Input;
    Value = ~Value & (Value - 1);
    Count = 0;

    while (Value != 0) {
        Count++;
        Value >>= 1;
    }

    return Count;
}

TRAILING_ZEROS_POINTER TrailingZerosPointer_C;

_Use_decl_annotations_
ULONG_PTR
TrailingZerosPointer_C(
    ULONG_PTR Integer
    )
{
#ifdef _WIN64
    return TrailingZeros64_C((ULONGLONG)Integer);
#else
    return TrailingZeros32_C((ULONG)Integer);
#endif
}

//
// PopulationCount.
//

POPULATION_COUNT_32 PopulationCount32_C;

_Use_decl_annotations_
ULONG
PopulationCount32_C(
    ULONG Input
    )
{
    ULONG Value;

    Value = Input;

    Value -= ((Value >> 1) & 0x55555555);
    Value = (((Value >> 2) & 0x33333333) + (Value & 0x33333333));
    Value = (((Value >> 4) + Value) & 0x0f0f0f0f);
    Value += (Value >> 8);
    Value += (Value >> 16);
    return (Value & 0x0000003f);
}


POPULATION_COUNT_64 PopulationCount64_C;

_Use_decl_annotations_
ULONGLONG
PopulationCount64_C(
    ULONGLONG Input
    )
{
    ULONG LowCount;
    ULONG HighCount;
    ULONGLONG Count;
    ULARGE_INTEGER Value;

    Value.QuadPart = Input;

    LowCount = PopulationCount32_C(Value.LowPart);
    HighCount = PopulationCount32_C(Value.HighPart);

    Count = (ULONGLONG)LowCount + (ULONGLONG)HighCount;

    return Count;
}


POPULATION_COUNT_POINTER PopulationCountPointer_C;

_Use_decl_annotations_
ULONG_PTR
PopulationCountPointer_C(
    ULONG_PTR Integer
    )
{
#ifdef _WIN64
    return PopulationCount64_C((ULONGLONG)Integer);
#else
    return PopulationCount32_C((ULONG)Integer);
#endif
}

//
// RoundUpPowerOfTwo and RoundUpNextPowerOfTwo; 32, 64 and pointer-sized
// variants of C implementations.
//

ROUND_UP_POWER_OF_TWO_32 RoundUpPowerOfTwo32_C;

_Use_decl_annotations_
ULONGLONG
RoundUpPowerOfTwo32_C(
    ULONG Input
    )
{
    if (Input <= 1) {
        return 2;
    }

    if (IsPowerOfTwo(Input)) {
        return Input;
    }

    return 1ULL << (32 - LeadingZeros32_C(Input - 1));
}


ROUND_UP_NEXT_POWER_OF_TWO_32 RoundUpNextPowerOfTwo32_C;

_Use_decl_annotations_
ULONGLONG
RoundUpNextPowerOfTwo32_C(
    ULONG Input
    )
{
    if (Input <= 1) {
        return 2;
    }

    if (IsPowerOfTwo(Input)) {
        Input += 1;
    }

    return 1ULL << (32 - LeadingZeros32_C(Input - 1));
}


ROUND_UP_POWER_OF_TWO_64 RoundUpPowerOfTwo64_C;

_Use_decl_annotations_
ULONGLONG
RoundUpPowerOfTwo64_C(
    ULONGLONG Input
    )
{
    if (Input <= 1) {
        return 2;
    }

    if (IsPowerOfTwo(Input)) {
        return Input;
    }

    return 1ULL << (64 - LeadingZeros64_C(Input - 1));
}


ROUND_UP_NEXT_POWER_OF_TWO_64 RoundUpNextPowerOfTwo64_C;

_Use_decl_annotations_
ULONGLONG
RoundUpNextPowerOfTwo64_C(
    ULONGLONG Input
    )
{
    if (Input <= 1) {
        return 2;
    }

    if (IsPowerOfTwo(Input)) {
        Input += 1;
    }

    return 1ULL << (64 - LeadingZeros64_C(Input - 1));
}


ROUND_UP_POWER_OF_TWO_POINTER RoundUpPowerOfTwoPointer_C;

_Use_decl_annotations_
ULONGLONG
RoundUpPowerOfTwoPointer_C(
    ULONG_PTR Input
    )
{
    const ULONG_PTR Bits = sizeof(ULONG_PTR) << 3;

    if (Input <= 1) {
        return 2;
    }

    if (IsPowerOfTwo(Input)) {
        return Input;
    }

    return 1ULL << (Bits - LeadingZerosPointer_C(Input - 1));
}


ROUND_UP_NEXT_POWER_OF_TWO_POINTER RoundUpNextPowerOfTwoPointer_C;

_Use_decl_annotations_
ULONGLONG
RoundUpNextPowerOfTwoPointer_C(
    ULONG_PTR Input
    )
{
    const ULONG_PTR Bits = sizeof(ULONG_PTR) << 3;

    if (Input <= 1) {
        return 2;
    }

    if (IsPowerOfTwo(Input)) {
        Input += 1;
    }

    return 1ULL << (Bits - LeadingZerosPointer_C(Input - 1));
}

//
// Define intrinsic versions.
//

//
// LeadingZeros.
//

LEADING_ZEROS_32 LeadingZeros32_LZCNT;

_Use_decl_annotations_
ULONG
LeadingZeros32_LZCNT(
    ULONG Integer
    )
{
    return _lzcnt_u32(Integer);
}


#ifdef _WIN64
LEADING_ZEROS_64 LeadingZeros64_LZCNT;

_Use_decl_annotations_
ULONGLONG
LeadingZeros64_LZCNT(
    ULONGLONG Integer
    )
{
    return _lzcnt_u64(Integer);
}
#endif


LEADING_ZEROS_POINTER LeadingZerosPointer_LZCNT;

_Use_decl_annotations_
ULONG_PTR
LeadingZerosPointer_LZCNT(
    ULONG_PTR Integer
    )
{
#ifdef _WIN64
    return LeadingZeros64_LZCNT((ULONGLONG)Integer);
#else
    return LeadingZeros32_LZCNT((ULONG)Integer);
#endif
}

//
// TrailingZeros.
//

TRAILING_ZEROS_32 TrailingZeros32_BMI1;

_Use_decl_annotations_
ULONG
TrailingZeros32_BMI1(
    ULONG Integer
    )
{
    return _tzcnt_u32(Integer);
}


#ifdef _WIN64
TRAILING_ZEROS_64 TrailingZeros64_BMI1;

_Use_decl_annotations_
ULONGLONG
TrailingZeros64_BMI1(
    ULONGLONG Integer
    )
{
    return _tzcnt_u64(Integer);
}
#endif


TRAILING_ZEROS_POINTER TrailingZerosPointer_BMI1;

_Use_decl_annotations_
ULONG_PTR
TrailingZerosPointer_BMI1(
    ULONG_PTR Integer
    )
{
#ifdef _WIN64
    return TrailingZeros64_BMI1((ULONGLONG)Integer);
#else
    return TrailingZeros32_BMI1((ULONG)Integer);
#endif
}

//
// PopulationCount.
//

POPULATION_COUNT_32 PopulationCount32_POPCNT;

_Use_decl_annotations_
ULONG
PopulationCount32_POPCNT(
    ULONG Integer
    )
{
    return __popcnt(Integer);
}

#ifdef _WIN64
POPULATION_COUNT_64 PopulationCount64_POPCNT;

_Use_decl_annotations_
ULONGLONG
PopulationCount64_POPCNT(
    ULONGLONG Integer
    )
{
    return __popcnt64(Integer);
}
#endif

POPULATION_COUNT_POINTER PopulationCountPointer_POPCNT;

_Use_decl_annotations_
ULONG_PTR
PopulationCountPointer_POPCNT(
    ULONG_PTR Integer
    )
{
#ifdef _WIN64
    return PopulationCount64_POPCNT((ULONGLONG)Integer);
#else
    return PopulationCount32_POPCNT((ULONG)Integer);
#endif
}

//
// RoundUpPowerOfTwo32 and RoundUpNextPowerOfTwo32.
//

ROUND_UP_POWER_OF_TWO_32 RoundUpPowerOfTwo32_LZCNT;

_Use_decl_annotations_
ULONGLONG
RoundUpPowerOfTwo32_LZCNT(
    ULONG Input
    )
{
    if (Input <= 1) {
        return 2;
    }

    if (IsPowerOfTwo(Input)) {
        return Input;
    }

    return 1ULL << (32 - LeadingZeros32_LZCNT(Input - 1));
}


ROUND_UP_NEXT_POWER_OF_TWO_32 RoundUpNextPowerOfTwo32_LZCNT;

_Use_decl_annotations_
ULONGLONG
RoundUpNextPowerOfTwo32_LZCNT(
    ULONG Input
    )
{
    if (Input <= 1) {
        return 2;
    }

    if (IsPowerOfTwo(Input)) {
        Input += 1;
    }

    return 1ULL << (32 - LeadingZeros32_LZCNT(Input - 1));
}

//
// RoundUpPowerOfTwo64 and RoundUpNextPowerOfTwo64.
//

ROUND_UP_POWER_OF_TWO_64 RoundUpPowerOfTwo64_LZCNT;

_Use_decl_annotations_
ULONGLONG
RoundUpPowerOfTwo64_LZCNT(
    ULONGLONG Input
    )
{
    if (Input <= 1) {
        return 2;
    }

    if (IsPowerOfTwo(Input)) {
        return Input;
    }

    return 1ULL << (64 - LeadingZeros64_LZCNT(Input - 1));
}


ROUND_UP_NEXT_POWER_OF_TWO_64 RoundUpNextPowerOfTwo64_LZCNT;

_Use_decl_annotations_
ULONGLONG
RoundUpNextPowerOfTwo64_LZCNT(
    ULONGLONG Input
    )
{
    if (Input <= 1) {
        return 2;
    }

    if (IsPowerOfTwo(Input)) {
        Input += 1;
    }

    return 1ULL << (64 - LeadingZeros64_LZCNT(Input - 1));
}

//
// RoundUpPowerOfTwoPointer and RoundUpNextPowerOfTwoPointer.
//

ROUND_UP_POWER_OF_TWO_POINTER RoundUpPowerOfTwoPointer_LZCNT;

_Use_decl_annotations_
ULONGLONG
RoundUpPowerOfTwoPointer_LZCNT(
    ULONG_PTR Input
    )
{
    const ULONG_PTR Bits = sizeof(ULONG_PTR) << 3;

    if (Input <= 1) {
        return 2;
    }

    if (IsPowerOfTwo(Input)) {
        return Input;
    }

    return 1ULL << (Bits - LeadingZerosPointer_LZCNT(Input - 1));
}


ROUND_UP_NEXT_POWER_OF_TWO_POINTER RoundUpNextPowerOfTwoPointer_LZCNT;

_Use_decl_annotations_
ULONGLONG
RoundUpNextPowerOfTwoPointer_LZCNT(
    ULONG_PTR Input
    )
{
    const ULONG_PTR Bits = sizeof(ULONG_PTR) << 3;

    if (Input <= 1) {
        return 2;
    }

    if (IsPowerOfTwo(Input)) {
        Input += 1;
    }

    return 1ULL << (Bits - LeadingZerosPointer_LZCNT(Input - 1));
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
