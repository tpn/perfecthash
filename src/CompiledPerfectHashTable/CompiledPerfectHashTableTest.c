
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
            DEBUGBREAK();          \
        };                         \
        NumberOfErrors++;          \
    }

DECLARE_TEST_CPH_ROUTINE()
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
    CPHINDEX Index;
    CPHKEY Key;
    CPHKEY Rotated;
    CPHVALUE Value;
    CPHVALUE Previous;
    ULONG NumberOfErrors = 0;
    const CPHKEY *Source;

    Key = *KEYS;

    //
    // Rotate the key such that it differs from the original value, but in a
    // way that can be easily reversed.
    //

    Rotated = ROTATE_KEY_LEFT(Key, 15);
    ASSERT(Key == ROTATE_KEY_RIGHT(Rotated, 15));

    //
    // Verify looking up a key that hasn't been inserted returns 0 as the value.
    //

    Value = LOOKUP_ROUTINE(Key);
    ASSERT(Value == 0);

    //
    // Verify insertion.
    //

    Previous = INSERT_ROUTINE(Key, (CPHVALUE)Rotated);
    ASSERT(Previous == 0);

    //
    // Lookup the inserted key.
    //

    Value = LOOKUP_ROUTINE(Key);
    ASSERT(Value == (CPHVALUE)Rotated);

    //
    // Delete the inserted key.  Returned value should be Rotated.
    //

    Value = DELETE_ROUTINE(Key);
    ASSERT(Value == (CPHVALUE)Rotated);

    //
    // Verify a subsequent lookup returns 0.
    //

    Value = LOOKUP_ROUTINE(Key);
    ASSERT(Value == 0);

    //
    // Loop through the entire key set and insert a rotated version of the key.
    //

    FOR_EACH_KEY {

        Key = *Source++;
        Rotated = ROTATE_KEY_LEFT(Key, 15);

        Previous = INSERT_ROUTINE(Key, (CPHVALUE)Rotated);
        ASSERT(Previous == 0);

    }

    //
    // Loop through the entire set again and ensure that lookup returns the
    // rotated version.
    //

    FOR_EACH_KEY {

        Key = *Source++;
        Rotated = ROTATE_KEY_LEFT(Key, 15);

        Value = LOOKUP_ROUTINE(Key);
        ASSERT(Value == (CPHVALUE)Rotated);

    }

    //
    // Loop through again and delete everything.
    //

    FOR_EACH_KEY {

        Key = *Source++;
        Rotated = ROTATE_KEY_LEFT(Key, 15);

        Previous = DELETE_ROUTINE(Key);
        ASSERT(Previous == (CPHVALUE)Rotated);

    }

    //
    // And a final loop through to confirm all lookups now return 0.
    //

    FOR_EACH_KEY {

        Key = *Source++;

        Value = LOOKUP_ROUTINE(Key);
        ASSERT(Value == 0);

    }

    //
    // We're finished!  Return the number of errors.
    //

    return NumberOfErrors;
}

