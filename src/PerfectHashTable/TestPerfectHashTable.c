/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    TestPerfectHashTable.c

Abstract:

    This module implements the test routine for an individual instance of a
    PERFECT_HASH_TABLE structure.  It is primarily used by the module's
    self-test functionality.

--*/

#include "stdafx.h"

//
// Disable global optimizations, even in release builds.  Without this, the
// compiler does clever things with regards to optimizing our __debugbreak()
// logic below, such that it's impossible to tell which ASSERT() triggered it.
//

#pragma optimize("", off)

//
// Use a custom ASSERT() macro that optionally issues a __debugbreak() based
// on the caller's DebugBreakOnFailure boolean flag.
//

#undef ASSERT
#define ASSERT(Condition)          \
    if (!(Condition)) {            \
        if (DebugBreakOnFailure) { \
            __debugbreak();        \
        };                         \
        goto Error;                \
    }

TEST_PERFECT_HASH_TABLE TestPerfectHashTable;

_Use_decl_annotations_
BOOLEAN
TestPerfectHashTable(
    PPERFECT_HASH_TABLE Table,
    BOOLEAN DebugBreakOnFailure
    )
/*++

Routine Description:

    Tests an instance of a PERFECT_HASH_TABLE for correctness.

Arguments:

    Table - Supplies a pointer to an initialized PERFECT_HASH_TABLE structure
        for which the testing will be undertaken.  This structure is obtained
        via either CreatePerfectHashTable() or LoadPerfectHashTable().

    DebugBreakOnFailure - Supplies a boolean flag that indicates whether or
        not a __debugbreak() should be issued if a test fails.

Return Value:

    TRUE if all internal tests pass, FALSE if not.

--*/
{
    PRTL Rtl;
    ULONG Key;
    ULONG Index;
    ULONG Value;
    ULONG Rotated;
    ULONG Previous;
    ULONG NumberOfKeys;
    ULONG ValueIndex;
    ULONG Bit;
    ULONG NumberOfBitsSet;
    PULONG Keys;
    BOOLEAN Success;
    HRESULT Result;
    PALLOCATOR Allocator;
    RTL_BITMAP Indices;
    PLONG BitmapBuffer = NULL;
    ULARGE_INTEGER NumberOfBits;
    ULARGE_INTEGER BitmapBufferSize;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Table)) {
        return FALSE;
    }

    //
    // Initialize aliases.
    //

    Rtl = Table->Rtl;
    Allocator = Table->Allocator;

    //
    // Sanity check the perfect hash structure size matches what we expect.
    //

    ASSERT(Table->SizeOfStruct == sizeof(*Table));

    //
    // Verify the table has a valid keys structure wired up, and that the number
    // of keys is within MAX_ULONG.
    //

    ASSERT(Table->Keys && Table->Keys->Keys);
    ASSERT(!Table->Keys->NumberOfElements.HighPart);

    NumberOfKeys = Table->Keys->NumberOfElements.LowPart;

    //
    // Calculate the space required for a bitmap buffer that allows us to set
    // a bit for each unique key index.  We add 1 to the size to account for
    // the fact that we have to add 1 to every index value returned by the
    // table's Index() method in order to account for the fact that 0 may be
    // returned as an index.
    //

    NumberOfBits.QuadPart = (ULONGLONG)NumberOfKeys + 1ULL;
    ASSERT(!NumberOfBits.HighPart);

    Indices.SizeOfBitMap = NumberOfBits.LowPart;

    //
    // Divide by 8 (>> 3) to convert number of bits to bytes required, then
    // round up to a 16 byte boundary.
    //

    BitmapBufferSize.QuadPart = (
        ALIGN_UP(NumberOfBits.QuadPart >> 3ULL, 16)
    );

    //
    // Overflow check.
    //

    ASSERT(!BitmapBufferSize.HighPart);

    //
    // Allocate the memory.
    //

    BitmapBuffer = (PLONG)(
        Allocator->Calloc(
            Allocator->Context,
            1,
            BitmapBufferSize.LowPart
        )
    );

    ASSERT(BitmapBuffer);

    //
    // N.B. We use PLONG for the BitmapBuffer type instead of PULONG (which is
    //      the type of the RTL_BITMAP Buffer field), because we use the bit
    //      test and set intrinsic below, and that wants PLONG.
    //

    Indices.Buffer = (PLONG)BitmapBuffer;

    //
    // Grab the first key from the underlying keys table such that we've got
    // a valid key that was present in the input set.
    //


    Keys = Table->Keys->Keys;
    Key = Keys[0];

    //
    // Rotate the key such that it differs from the original value, but in a
    // way that can be easily reversed.
    //

    Rotated = _rotl(Key, 15);
    ASSERT(Key == _rotr(Rotated, 15));

    //
    // Verify looking up a key that hasn't been inserted returns 0 as the value.
    //

    Result = Table->Vtbl->Lookup(Table, Key, &Value);
    ASSERT(!FAILED(Result));
    ASSERT(Value == 0);

    //
    // Verify insertion.
    //

    Result = Table->Vtbl->Insert(Table, Key, Rotated, &Previous);
    ASSERT(!FAILED(Result));
    ASSERT(Previous == 0);

    //
    // Lookup the inserted key.
    //

    Result = Table->Vtbl->Lookup(Table, Key, &Value);
    ASSERT(!FAILED(Result));
    ASSERT(Value == Rotated);

    //
    // Delete the inserted key.  Returned value should be Rotated.
    //

    Value = 0;
    Result = Table->Vtbl->Delete(Table, Key, &Value);
    ASSERT(!FAILED(Result));
    ASSERT(Value == Rotated);

    //
    // Verify a subsequent lookup returns 0.
    //

    Value = (ULONG)-1;
    Result = Table->Vtbl->Lookup(Table, Key, &Value);
    ASSERT(!FAILED(Result));
    ASSERT(Value == 0);

    //
    // Loop through the entire key set and obtain the values array index for
    // that given key.  Verify that the corresponding bit in our bitmap isn't
    // set -- if it is, it indicates we've been given the same index for two
    // different keys, which is a pretty severe error.
    //

    for (Index = 0; Index < NumberOfKeys; Index++) {

        Key = Keys[Index];

        Result = Table->Vtbl->Index(Table, Key, &ValueIndex);
        ASSERT(!FAILED(Result));

        //
        // Account for the fact that the value index may be 0 by adding 1.
        //

        Bit = ValueIndex + 1;

        //
        // Verify bit isn't already set and set it.
        //

        ASSERT(!BitTestAndSet(BitmapBuffer, Bit));

    }

    //
    // Count the number of set bits and verify it matches the number of keys.
    //

    NumberOfBitsSet = Rtl->RtlNumberOfSetBits(&Indices);
    ASSERT(NumberOfBitsSet == NumberOfKeys);

    //
    // Loop through the entire key set and insert a rotated version of the key.
    //

    for (Index = 0; Index < NumberOfKeys; Index++) {

        Key = Keys[Index];
        Rotated = _rotl(Key, 15);

        Result = Table->Vtbl->Insert(Table, Key, Rotated, &Previous);
        ASSERT(!FAILED(Result));
        ASSERT(Previous == 0);

    }

    //
    // Loop through the entire set again and ensure that lookup returns the
    // rotated version.
    //

    for (Index = 0; Index < NumberOfKeys; Index++) {

        Key = Keys[Index];
        Rotated = _rotl(Key, 15);

        Result = Table->Vtbl->Lookup(Table, Key, &Value);
        ASSERT(!FAILED(Result));
        ASSERT(Value == Rotated);

    }

    //
    // Loop through again and delete everything.
    //

    for (Index = 0; Index < NumberOfKeys; Index++) {

        Key = Keys[Index];
        Rotated = _rotl(Key, 15);

        Result = Table->Vtbl->Delete(Table, Key, &Previous);
        ASSERT(!FAILED(Result));
        ASSERT(Previous == Rotated);

    }

    //
    // And a final loop through to confirm all lookups now return 0.
    //

    for (Index = 0; Index < NumberOfKeys; Index++) {

        Key = Keys[Index];

        Result = Table->Vtbl->Lookup(Table, Key, &Value);
        ASSERT(!FAILED(Result));
        ASSERT(Value == 0);

    }

    //
    // We're finished!  Indicate success and finish up.
    //

    Success = TRUE;

    goto End;

Error:

    Success = FALSE;

    //
    // Intentional follow-on to End.
    //

End:

    return Success;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
