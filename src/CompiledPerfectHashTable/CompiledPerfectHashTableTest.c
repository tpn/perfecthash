
#ifdef _WIN32

//
// Disable global optimizations, even in release builds.  Without this, the
// compiler does clever things with regards to optimizing our __debugbreak()
// logic below, such that it's impossible to tell which ASSERT() triggered it.
//

#pragma optimize("", off)

//
// Disable Spectre warnings.
//

#pragma warning(disable: 5045)

#endif

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

#ifdef CPH_HAS_INDEX32X8_ROUTINE
extern
VOID
CPHCALLTYPE
INDEX32X8_ROUTINE(
    _In_ CPHKEY Key1,
    _In_ CPHKEY Key2,
    _In_ CPHKEY Key3,
    _In_ CPHKEY Key4,
    _In_ CPHKEY Key5,
    _In_ CPHKEY Key6,
    _In_ CPHKEY Key7,
    _In_ CPHKEY Key8,
    _Out_ PCPHINDEX Index1,
    _Out_ PCPHINDEX Index2,
    _Out_ PCPHINDEX Index3,
    _Out_ PCPHINDEX Index4,
    _Out_ PCPHINDEX Index5,
    _Out_ PCPHINDEX Index6,
    _Out_ PCPHINDEX Index7,
    _Out_ PCPHINDEX Index8
    );
#endif

#ifdef CPH_HAS_INDEX32X16_ROUTINE
extern
VOID
CPHCALLTYPE
INDEX32X16_ROUTINE(
    _In_ CPHKEY Key1,
    _In_ CPHKEY Key2,
    _In_ CPHKEY Key3,
    _In_ CPHKEY Key4,
    _In_ CPHKEY Key5,
    _In_ CPHKEY Key6,
    _In_ CPHKEY Key7,
    _In_ CPHKEY Key8,
    _In_ CPHKEY Key9,
    _In_ CPHKEY Key10,
    _In_ CPHKEY Key11,
    _In_ CPHKEY Key12,
    _In_ CPHKEY Key13,
    _In_ CPHKEY Key14,
    _In_ CPHKEY Key15,
    _In_ CPHKEY Key16,
    _Out_ PCPHINDEX Index1,
    _Out_ PCPHINDEX Index2,
    _Out_ PCPHINDEX Index3,
    _Out_ PCPHINDEX Index4,
    _Out_ PCPHINDEX Index5,
    _Out_ PCPHINDEX Index6,
    _Out_ PCPHINDEX Index7,
    _Out_ PCPHINDEX Index8,
    _Out_ PCPHINDEX Index9,
    _Out_ PCPHINDEX Index10,
    _Out_ PCPHINDEX Index11,
    _Out_ PCPHINDEX Index12,
    _Out_ PCPHINDEX Index13,
    _Out_ PCPHINDEX Index14,
    _Out_ PCPHINDEX Index15,
    _Out_ PCPHINDEX Index16
    );
#endif

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

#ifdef CPH_HAS_INDEX32X8_ROUTINE
    if (NUMBER_OF_KEYS >= 8) {
        ULONG Lane;
        CPHINDEX Indices[8];
        const CPHKEY *VectorKeys = KEYS;

        INDEX32X8_ROUTINE(
            VectorKeys[0],
            VectorKeys[1],
            VectorKeys[2],
            VectorKeys[3],
            VectorKeys[4],
            VectorKeys[5],
            VectorKeys[6],
            VectorKeys[7],
            &Indices[0],
            &Indices[1],
            &Indices[2],
            &Indices[3],
            &Indices[4],
            &Indices[5],
            &Indices[6],
            &Indices[7]
        );

        for (Lane = 0; Lane < 8; Lane++) {
            Index = INDEX_ROUTINE(VectorKeys[Lane]);
            ASSERT(Indices[Lane] == Index);
        }
    }
#endif

#ifdef CPH_HAS_INDEX32X16_ROUTINE
    if (NUMBER_OF_KEYS >= 16) {
        ULONG Lane;
        CPHINDEX Indices[16];
        const CPHKEY *VectorKeys = KEYS;

        INDEX32X16_ROUTINE(
            VectorKeys[0],
            VectorKeys[1],
            VectorKeys[2],
            VectorKeys[3],
            VectorKeys[4],
            VectorKeys[5],
            VectorKeys[6],
            VectorKeys[7],
            VectorKeys[8],
            VectorKeys[9],
            VectorKeys[10],
            VectorKeys[11],
            VectorKeys[12],
            VectorKeys[13],
            VectorKeys[14],
            VectorKeys[15],
            &Indices[0],
            &Indices[1],
            &Indices[2],
            &Indices[3],
            &Indices[4],
            &Indices[5],
            &Indices[6],
            &Indices[7],
            &Indices[8],
            &Indices[9],
            &Indices[10],
            &Indices[11],
            &Indices[12],
            &Indices[13],
            &Indices[14],
            &Indices[15]
        );

        for (Lane = 0; Lane < 16; Lane++) {
            Index = INDEX_ROUTINE(VectorKeys[Lane]);
            ASSERT(Indices[Lane] == Index);
        }
    }
#endif

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
