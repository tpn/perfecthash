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
        NumberOfErrors++;          \
    }

DECLARE_TEST_COMPILED_PERFECT_HASH_TABLE_ROUTINE_HEADER()
/*++

Routine Description:

    Tests a compiled perfect hash table for correctness.

Arguments:

    DebugBreakOnFailure - Supplies a boolean flag that indicates whether or
        not a __debugbreak() should be issued as soon as a test fails.

Return Value:

    The number of failed tests.  If 0, all tests passed.

--*/
{
    ULONG Key;
    ULONG Index;
    ULONG Value;
    ULONG Rotated;
    ULONG Previous;
    ULONG NumberOfKeys;
    ULONG NumberOfErrors = 0;
    const ULONG *Source;

    NumberOfKeys = CphTableNumberOfKeys;

    Key = *CphTableKeys;

    //
    // Rotate the key such that it differs from the original value, but in a
    // way that can be easily reversed.
    //

    Rotated = _rotl(Key, 15);
    ASSERT(Key == _rotr(Rotated, 15));

    //
    // Verify looking up a key that hasn't been inserted returns 0 as the value.
    //

    Value = CphTableLookup(Key);
    ASSERT(Value == 0);

    //
    // Verify insertion.
    //

    Previous = CphTableInsert(Key, Rotated);
    ASSERT(Previous == 0);

    //
    // Lookup the inserted key.
    //

    Value = CphTableLookup(Key);
    ASSERT(Value == Rotated);

    //
    // Delete the inserted key.  Returned value should be Rotated.
    //

    Value = CphTableDelete(Key);
    ASSERT(Value == Rotated);

    //
    // Verify a subsequent lookup returns 0.
    //

    Value = CphTableLookup(Key);
    ASSERT(Value == 0);

    //
    // Loop through the entire key set and insert a rotated version of the key.
    //

    for (Index = 0, Source = CphTableKeys; Index < NumberOfKeys; Index++) {

        Key = *Source++;
        Rotated = _rotl(Key, 15);

        Previous = CphTableInsert(Key, Rotated);
        ASSERT(Previous == 0);

    }

    //
    // Loop through the entire set again and ensure that lookup returns the
    // rotated version.
    //

    for (Index = 0, Source = CphTableKeys; Index < NumberOfKeys; Index++) {

        Key = *Source++;
        Rotated = _rotl(Key, 15);

        Value = CphTableLookup(Key);
        ASSERT(Value == Rotated);

    }

    //
    // Loop through again and delete everything.
    //

    for (Index = 0, Source = CphTableKeys; Index < NumberOfKeys; Index++) {

        Key = *Source++;
        Rotated = _rotl(Key, 15);

        Previous = CphTableDelete(Key);
        ASSERT(Previous == Rotated);

    }

    //
    // And a final loop through to confirm all lookups now return 0.
    //

    for (Index = 0, Source = CphTableKeys; Index < NumberOfKeys; Index++) {

        Key = *Source++;

        Value = CphTableLookup(Key);
        ASSERT(Value == 0);

    }

    //
    // We're finished!  Return the number of errors.
    //

    return NumberOfErrors;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
