/*++

Copyright (c) 2021 Trent Nelson <trent@trent.me>

Module Name:

    Rng.h

Abstract:

    This is the private header file for the random number generation component
    of the perfect hash library.  This component is inspired by the CUDA RNG
    component (cuRAND).

--*/

#pragma once

#include "stdafx.h"

//
// Default RNG values.
//

#define RNG_DEFAULT_ID PerfectHashRngPhilox43210Id
#define RNG_DEFAULT_SEED 0x2019090319811025
#define RNG_DEFAULT_SUBSEQUENCE 0x0
#define RNG_DEFAULT_OFFSET 0x0

//
// Helper CUDA-like packed int "vector" typedefs.
//

typedef union DECLSPEC_ALIGN(8) _UINT2 {
    struct {
        ULONG X;
        ULONG Y;
    };
    ULONGLONG AsULongLong;
} UINT2, *PUINT2;

typedef union DECLSPEC_ALIGN(16) _UINT4 {
    struct {
        ULONG X;
        ULONG Y;
        ULONG Z;
        ULONG W;
    };
    struct {
        ULONGLONG XY;
        ULONGLONG ZW;
    };
    XMMWORD AsXmmWord;
} UINT4, *PUINT4;

//
// Define the random state structs for each RNG type.
//

typedef struct DECLSPEC_ALIGN(16) _RNG_STATE_PHILOX43210 {
    UINT4 Counter;
    UINT4 Output;
    UINT2 Key;
    ULONG CurrentCount;
    ULONG NumberOfZeroLongsEncountered;
    ULONGLONG TotalCount;
    ULONGLONG TotalBytes;
} RNG_STATE_PHILOX43210, *PRNG_STATE_PHILOX43210;

//
// Define the RNG component.
//

typedef union _RNG_STATE {
    struct {

        //
        // RNG instance has been initialized.
        //

        ULONG Initialized:1;

        //
        // Unused bits.
        //

        ULONG Unused:31;

    };
    LONG AsLong;
    ULONG AsULong;
} RNG_STATE;
C_ASSERT(sizeof(RNG_STATE) == sizeof(ULONG));
typedef RNG_STATE *PRNG_STATE;

//
// Helper state macros.
//

#define IsRngInitialized(Rng) (Rng->State.Initialized != FALSE)

//
// Helper flag macros.
//

#define WantsRandomStartSeed(Rng) (Rng->Flags.UseRandomStartSeed != FALSE)
#define WantsDefaultStartSeed(Rng) (Rng->Flags.UseDefaultStartSeed != FALSE)

DECLARE_COMPONENT(Rng, RNG);

typedef struct _Struct_size_bytes_(SizeOfStruct) _RNG {
    COMMON_COMPONENT_HEADER(RNG);

    PERFECT_HASH_RNG_ID RngId;
    ULONG Padding1;

    ULONGLONG Seed;
    ULONGLONG Subsequence;
    ULONGLONG Offset;

    union {
        RNG_STATE_PHILOX43210 Philox43210;
        PVOID StateAddress;
    };

    //
    // Backing vtbl;
    //

    RNG_VTBL Interface;

    //
    // N.B. As additional table functions are added to the context vtbl, you'll
    //      need to comment and un-comment the following padding field in order
    //      to avoid "warning: additional 8 bytes padding added after ..."-type
    //      warnings.
    //

    //PVOID Padding3;

} RNG;
typedef RNG *PRNG;

#define STATE_TO_RNG(State) CONTAINING_RECORD(State, RNG, StateAddress)

//
// Private non-vtbl methods.
//

typedef
HRESULT
(NTAPI RNG_INITIALIZE)(
    _In_ PRNG Rng
    );
typedef RNG_INITIALIZE *PRNG_INITIALIZE;

typedef
VOID
(NTAPI RNG_RUNDOWN)(
    _In_ _Post_ptr_invalid_ PRNG Rng
    );
typedef RNG_RUNDOWN *PRNG_RUNDOWN;

//
// Function decls.
//

RNG_INITIALIZE RngInitialize;
RNG_RUNDOWN RngRundown;
RNG_INITIALIZE_PSEUDO RngInitializePseudo;
RNG_GENERATE_RANDOM_BYTES RngGenerateRandomBytes;
RNG_GET_CURRENT_OFFSET RngGetCurrentOffset;

//
// Private impl function decls.
//

VOID
RngPhilox43210Init(
    PRNG_STATE_PHILOX43210 State
    );

HRESULT
RngPhilox43210GenerateRandomBytes(
    PRNG_STATE_PHILOX43210 State,
    SIZE_T SizeOfBufferInBytes,
    PBYTE Buffer
    );

//
// Inline helper routines.
//

FORCEINLINE
HRESULT
RngGenerateRandomBytesFromSystem(
    _In_ PRNG Rng,
    _In_ SIZE_T SizeOfBufferInBytes,
    _Out_writes_(SizeOfBufferInBytes) PBYTE Buffer
    )
{
    HRESULT Result;

    Result = Rng->Rtl->Vtbl->GenerateRandomBytes(
        Rng->Rtl,
        (ULONG)SizeOfBufferInBytes,
        Buffer
    );

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
