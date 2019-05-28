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
// PopulationCount64 is defined as a macro in winnt.h.  Undefine it now.
//

#undef PopulationCount64

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
(ROUND_UP_POWER_OF_TWO_POINTER)(
    _In_ ULONG_PTR Input
    );
typedef ROUND_UP_POWER_OF_TWO_POINTER *PROUND_UP_POWER_OF_TWO_POINTER;

typedef
ULONG_PTR
(ROUND_UP_NEXT_POWER_OF_TWO_POINTER)(
    _In_ ULONG_PTR Input
    );
typedef ROUND_UP_NEXT_POWER_OF_TWO_POINTER *PROUND_UP_NEXT_POWER_OF_TWO_POINTER;

//
// Structure of function pointers.
//


//
// Define the X-macro.
//

#define RTL_BIT_MANIPULATION_FUNCTION_TABLE(FIRST_ENTRY, \
                                            ENTRY,       \
                                            LAST_ENTRY)  \
                                                         \
    FIRST_ENTRY(                                         \
        LEADING_ZEROS_32,                                \
        LeadingZeros32,                                  \
        LZCNT,                                           \
        ABM                                              \
    )                                                    \
                                                         \
    ENTRY(                                               \
        TRAILING_ZEROS_32,                               \
        TrailingZeros32,                                 \
        BMI1,                                            \
        ABM                                              \
    )                                                    \
                                                         \
    ENTRY(                                               \
        POPULATION_COUNT_32,                             \
        PopulationCount32,                               \
        POPCNT,                                          \
        ABM                                              \
    )                                                    \
                                                         \
    ENTRY(                                               \
        LEADING_ZEROS_64,                                \
        LeadingZeros64,                                  \
        LZCNT,                                           \
        ABM                                              \
    )                                                    \
                                                         \
    ENTRY(                                               \
        TRAILING_ZEROS_64,                               \
        TrailingZeros64,                                 \
        BMI1,                                            \
        ABM                                              \
    )                                                    \
                                                         \
    ENTRY(                                               \
        POPULATION_COUNT_64,                             \
        PopulationCount64,                               \
        POPCNT,                                          \
        ABM                                              \
    )                                                    \
                                                         \
    ENTRY(                                               \
        LEADING_ZEROS_POINTER,                           \
        LeadingZerosPointer,                             \
        LZCNT,                                           \
        ABM                                              \
    )                                                    \
                                                         \
    ENTRY(                                               \
        TRAILING_ZEROS_POINTER,                          \
        TrailingZerosPointer,                            \
        BMI1,                                            \
        ABM                                              \
    )                                                    \
                                                         \
    ENTRY(                                               \
        POPULATION_COUNT_POINTER,                        \
        PopulationCountPointer,                          \
        POPCNT,                                          \
        ABM                                              \
    )                                                    \
                                                         \
    ENTRY(                                               \
        ROUND_UP_POWER_OF_TWO_32,                        \
        RoundUpPowerOfTwo32,                             \
        LZCNT,                                           \
        ABM                                              \
    )                                                    \
                                                         \
    ENTRY(                                               \
        ROUND_UP_NEXT_POWER_OF_TWO_32,                   \
        RoundUpNextPowerOfTwo32,                         \
        LZCNT,                                           \
        ABM                                              \
    )                                                    \
                                                         \
    ENTRY(                                               \
        ROUND_UP_POWER_OF_TWO_64,                        \
        RoundUpPowerOfTwo64,                             \
        LZCNT,                                           \
        ABM                                              \
    )                                                    \
                                                         \
    ENTRY(                                               \
        ROUND_UP_NEXT_POWER_OF_TWO_64,                   \
        RoundUpNextPowerOfTwo64,                         \
        LZCNT,                                           \
        ABM                                              \
    )                                                    \
                                                         \
    ENTRY(                                               \
        ROUND_UP_POWER_OF_TWO_POINTER,                   \
        RoundUpPowerOfTwoPointer,                        \
        LZCNT,                                           \
        ABM                                              \
    )                                                    \
                                                         \
    LAST_ENTRY(                                          \
        ROUND_UP_NEXT_POWER_OF_TWO_POINTER,              \
        RoundUpNextPowerOfTwoPointer,                    \
        LZCNT,                                           \
        ABM                                              \
    )

#define RTL_BIT_MANIPULATION_FUNCTION_TABLE_ENTRY(ENTRY) \
    RTL_BIT_MANIPULATION_FUNCTION_TABLE(ENTRY,           \
                                        ENTRY,           \
                                        ENTRY)

#define EXPAND_AS_FUNCTION_POINTER(Upper, Name, Unused3, Unused4) \
    P##Upper Name;
typedef struct _RTL_BIT_MANIPULATION_FUNCTIONS {
    RTL_BIT_MANIPULATION_FUNCTION_TABLE_ENTRY(EXPAND_AS_FUNCTION_POINTER)
} RTL_BIT_MANIPULATION_FUNCTIONS;
typedef RTL_BIT_MANIPULATION_FUNCTIONS *PRTL_BIT_MANIPULATION_FUNCTIONS;
#undef EXPAND_AS_FUNCTION_POINTER

#define EXPAND_AS_FUNCTION_POINTER_C(Upper, Name, Unused3, Unused4) \
    P##Upper Name##_C;
typedef struct _RTL_BIT_MANIPULATION_FUNCTIONS_C {
    RTL_BIT_MANIPULATION_FUNCTION_TABLE_ENTRY(EXPAND_AS_FUNCTION_POINTER_C)
} RTL_BIT_MANIPULATION_FUNCTIONS_C;
typedef RTL_BIT_MANIPULATION_FUNCTIONS_C *PRTL_BIT_MANIPULATION_FUNCTIONS_C;
#undef EXPAND_AS_FUNCTION_POINTER_C

#define EXPAND_AS_FUNCTION_POINTER_INTEL(Upper, Name, IntelFeature, Unused4) \
    P##Upper Name##_##IntelFeature;
typedef struct _RTL_BIT_MANIPULATION_FUNCTIONS_INTEL {
    RTL_BIT_MANIPULATION_FUNCTION_TABLE_ENTRY(EXPAND_AS_FUNCTION_POINTER_INTEL)
} RTL_BIT_MANIPULATION_FUNCTIONS_INTEL;
typedef RTL_BIT_MANIPULATION_FUNCTIONS_INTEL
      *PRTL_BIT_MANIPULATION_FUNCTIONS_INTEL;
#undef EXPAND_AS_FUNCTION_POINTER_INTEL

#define EXPAND_AS_FUNCTION_POINTER_AMD(Upper, Name, Unused3, AmdFeature) \
    P##Upper Name##_##AmdFeature;
typedef struct _RTL_BIT_MANIPULATION_FUNCTIONS_AMD {
    RTL_BIT_MANIPULATION_FUNCTION_TABLE_ENTRY(EXPAND_AS_FUNCTION_POINTER_AMD)
} RTL_BIT_MANIPULATION_FUNCTIONS_AMD;
typedef RTL_BIT_MANIPULATION_FUNCTIONS_AMD
      *PRTL_BIT_MANIPULATION_FUNCTIONS_AMD;
#undef EXPAND_AS_FUNCTION_POINTER_AMD

#define EXPAND_AS_FUNCTION_DECL_C(Upper, Name, Unused3, Unused4) \
    Upper Name##_C;
RTL_BIT_MANIPULATION_FUNCTION_TABLE_ENTRY(EXPAND_AS_FUNCTION_DECL_C)
#undef EXPAND_AS_FUNCTION_DECL_C

#define EXPAND_AS_FUNCTION_DECL_INTEL(Upper, Name, IntelFeature, Unused4) \
    Upper Name##_##IntelFeature;
RTL_BIT_MANIPULATION_FUNCTION_TABLE_ENTRY(EXPAND_AS_FUNCTION_DECL_INTEL)
#undef EXPAND_AS_FUNCTION_DECL_INTEL

#define EXPAND_AS_FUNCTION_DECL_AMD(Upper, Name, Unused3, AmdFeature) \
    Upper Name##_##AmdFeature;
RTL_BIT_MANIPULATION_FUNCTION_TABLE_ENTRY(EXPAND_AS_FUNCTION_DECL_AMD)
#undef EXPAND_AS_FUNCTION_DECL_AMD

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
