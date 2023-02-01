/*++

Copyright (c) 2021 Trent Nelson <trent@trent.me>

Module Name:

    Rng.c

Abstract:

    This module implements the random number generation component for the
    perfect hash library.  Routines are provided to initialize and rundown
    the backing COM components, initialize pseudo-random parameters, and get
    random bytes from an initialized generator.

--*/

#include "stdafx.h"

//
// COM scaffolding routines for initialization and rundown.
//

RNG_INITIALIZE RngInitialize;

_Use_decl_annotations_
HRESULT
RngInitialize(
    PRNG Rng
    )
/*++

Routine Description:

    Initializes an RNG structure.  This is a relatively simple method that
    just primes the COM scaffolding.

Arguments:

    Rng - Supplies a pointer to a RNG structure for which initialization
        is to be performed.

Return Value:

    S_OK - Success.

    E_POINTER - Rng is NULL.

    E_UNEXPECTED - All other errors.

--*/
{
    HRESULT Result = S_OK;

    if (!ARGUMENT_PRESENT(Rng)) {
        return E_POINTER;
    }

    Rng->SizeOfStruct = sizeof(*Rng);

    //
    // Create an Rtl component.
    //

    Result = Rng->Vtbl->CreateInstance(Rng,
                                         NULL,
                                         &IID_PERFECT_HASH_RTL,
                                         PPV(&Rng->Rtl));

    if (FAILED(Result)) {
        goto Error;
    }

    //
    // We're done!  Indicate success and finish up.
    //

    Result = S_OK;
    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}


RNG_RUNDOWN RngRundown;

_Use_decl_annotations_
VOID
RngRundown(
    PRNG Rng
    )
/*++

Routine Description:

    Release all resources associated with an RNG.

Arguments:

    Rng - Supplies a pointer to the RNG instance to rundown.

Return Value:

    None.

--*/
{
    //
    // Sanity check structure size.
    //

    ASSERT(Rng->SizeOfStruct == sizeof(*Rng));

    //
    // Release applicable COM references.
    //

    RELEASE(Rng->Rtl);

    return;
}

RNG_INITIALIZE_PSEUDO RngInitializePseudo;

_Use_decl_annotations_
HRESULT
RngInitializePseudo(
    PRNG Rng,
    PERFECT_HASH_RNG_ID RngId,
    PRNG_FLAGS Flags,
    ULONGLONG Seed,
    ULONGLONG Subsequence,
    ULONGLONG Offset
    )
/*++

Routine Description:

    Initializes a pseudo-RNG instance.

Arguments:

    Rng - Supplies a pointer to the RNG instance.

    RngId - Supplies the ID of the RNG to use.

    Flags - Supplies a pointer to RNG flags that can be used to customize
        initialization behavior.

    Seed - Supplies a default seed that will be used if neither of the following
        two flags are set: UseRandomStartSeed, UseDefaultStartSeed.  Ignored
        otherwise.

    Subsequence - Supplies the subsequence to use.

    Offset - Supplies the offset to use.

Return Value:

    S_OK - Offset successfully obtained.

    E_POINTER - Rng was NULL.

    PH_E_INVALID_RNG_ID - Invalid RNG ID.

    PH_E_INVALID_RNG_FLAGS - Invalid RNG flags.

--*/
{
    HRESULT Result;

    if (Rng == NULL) {
        Result = E_POINTER;
        goto Error;
    }

    if (!IsValidPerfectHashRngId(RngId)) {
        Result = PH_E_INVALID_RNG_ID;
        goto Error;
    }

    if (FAILED(IsValidRngFlags(Flags))) {
        Result = PH_E_INVALID_RNG_FLAGS;
        goto Error;
    }

    //
    // Copy the flags over.
    //

    Rng->Flags.AsULong = Flags->AsULong;

    //
    // Determine if we need to override the seed value.  UseRandomStartSeed
    // takes precedence, then UseDefaultStartSeed.
    //

    if (WantsRandomStartSeed(Rng)) {
        Result = RngGenerateRandomBytesFromSystem(Rng,
                                                  sizeof(Rng->Seed),
                                                  (PBYTE)&Rng->Seed);
        if (FAILED(Result)) {
            PH_ERROR(RngInitializePseudo_GenRandomBytesFromSystem, Result);
            goto Error;
        }
    } else if (WantsDefaultStartSeed(Rng)) {
        Rng->Seed = RNG_DEFAULT_SEED;
    } else {

        //
        // Use the seed provided by the user.
        //

        Rng->Seed = Seed;
    }

    Rng->RngId = RngId;
    Rng->Subsequence = Subsequence;
    Rng->Offset = Offset;

    //
    // Disable "enum not handled in switch statement" warning.
    //

#pragma warning(push)
#pragma warning(disable: 4062)

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wswitch"


    switch (Rng->RngId) {
        case PerfectHashRngSystemId:

            //
            // No initialization needed for the system RNG.
            //

            NOTHING;
            break;

        case PerfectHashRngPhilox43210Id:
            RngPhilox43210Init(&Rng->Philox43210);
            break;
    }

#pragma clang diagnostic pop
#pragma warning(pop)

    //
    // We're done, finish up and indicate success.
    //

    Rng->State.Initialized = TRUE;
    Result = S_OK;
    goto End;

Error:
    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:
    return Result;
}

RNG_GENERATE_RANDOM_BYTES RngGenerateRandomBytes;

_Use_decl_annotations_
HRESULT
RngGenerateRandomBytes(
    PRNG Rng,
    SIZE_T SizeOfBufferInBytes,
    PBYTE Buffer
    )
/*++

Routine Description:

    Obtains random bytes from an initialized RNG.

Arguments:

    Rng - Supplies a pointer to the RNG instance.

    SizeOfBufferInBytes - Supplies the size of the Buffer parameter in bytes.
        Must be a multiple of 4 (i.e. sizeof(ULONG)).

    Buffer - Supplies a pointer to a buffer that will receive the number of
        random bytes indicated by the SizeOfBufferInBytes parameter.

Return Value:

    S_OK - Random bytes successfully obtained.

    E_POINTER - Rng or Buffer was NULL.

    PH_E_RNG_GENERATE_RANDOM_BYTES_INVALID_BUFFER_SIZE - An invalid buffer size
        was supplied (i.e. not a multiple of 4).

    PH_E_RNG_NOT_INITIALIZED - The RNG has not been initialized yet.

--*/
{
    PRTL Rtl;
    HRESULT Result;

    //
    // Validate arguments.
    //

    if (Rng == NULL || Buffer == NULL) {
        return E_POINTER;
    }

    if (!IsRngInitialized(Rng)) {
        return PH_E_RNG_NOT_INITIALIZED;
    }

    if ((SizeOfBufferInBytes <= 0) ||
        (SizeOfBufferInBytes >= ULONG_MAX) ||
        ((SizeOfBufferInBytes % 4) != 0)) {
        return PH_E_RNG_GENERATE_RANDOM_BYTES_INVALID_BUFFER_SIZE;
    }

    ASSERT(IsValidPerfectHashRngId(Rng->RngId));

    //
    // Argument validation complete, continue.
    //

    //
    // Zero all incoming buffer memory.
    //

    Rtl = Rng->Rtl;
    ZeroMemory(Buffer, SizeOfBufferInBytes);

    Result = E_UNEXPECTED;

    //
    // Dispatch to the applicable RNG method for obtaining random data.
    //

    switch (Rng->RngId) {
        case PerfectHashRngSystemId:
            Result = Rtl->Vtbl->GenerateRandomBytes(
                Rtl,
                (ULONG)SizeOfBufferInBytes,
                Buffer
            );
            break;

        case PerfectHashRngPhilox43210Id:
            Result = RngPhilox43210GenerateRandomBytes(
                &Rng->Philox43210,
                SizeOfBufferInBytes,
                Buffer
            );
            break;

        case PerfectHashNullRngId:
        case PerfectHashInvalidRngId:
        default:
            PH_RAISE(PH_E_UNREACHABLE_CODE);
            break;
    }

    return Result;
}

RNG_GET_CURRENT_OFFSET RngGetCurrentOffset;

_Use_decl_annotations_
HRESULT
RngGetCurrentOffset(
    PRNG Rng,
    PULONGLONG Offset
    )
/*++

Routine Description:

    Gets the current offset of the RNG instance.  Only applies to pseudo-RNGs.

Arguments:

    Rng - Supplies a pointer to the RNG instance.

    Offset - Supplies a pointer to a variable that receives the current offset
        of the RNG instance.

Return Value:

    S_OK - Offset successfully obtained.

--*/
{
    switch (Rng->RngId) {
        case PerfectHashRngPhilox43210Id:
            *Offset = Rng->Philox43210.TotalCount;
            break;

        case PerfectHashNullRngId:
        case PerfectHashInvalidRngId:
        case PerfectHashRngSystemId:
        default:
            *Offset = 0;
            break;
    }

    return S_OK;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
