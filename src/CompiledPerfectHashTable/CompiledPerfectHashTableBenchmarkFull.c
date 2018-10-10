extern
BOOLEAN
QueryPerformanceCounter(
    _Out_ PLARGE_INTEGER Count
    );

extern volatile ULONG CtrlCPressed;

DECLARE_BENCHMARK_FULL_COMPILED_PERFECT_HASH_TABLE_ROUTINE_HEADER()
{
    ULONG Key;
    ULONG Count;
    ULONG Index;
    ULONG Value = 0;
    ULONG Rotated;
    ULONG Previous;
    ULONG NumberOfKeys;
    ULONG Best = (ULONG)-1;
    ULONG Attempts = 100;
    const ULONG Iterations = 1000;
    const ULONG *Source;
    LARGE_INTEGER Start;
    LARGE_INTEGER End;
    LARGE_INTEGER Delta;

    NumberOfKeys = CphTableNumberOfKeys;

#define FOR_EACH_KEY                      \
    for (Index = 0, Source = CphTableKeys; \
         Index < NumberOfKeys;            \
         Index++)

    if (Seconds) {

        while (!CtrlCPressed) {

            QueryPerformanceCounter(&Start);

            for (Count = Iterations; Count != 0; Count--) {

                //
                // Loop through the entire key set and insert a rotated version of the key.
                //

                FOR_EACH_KEY {

                    Key = *Source++;
                    Rotated = _rotl(Key, 15);

                    Previous = CphTableInsert(Key, Rotated);

                }

                //
                // Loop through the entire set again and ensure that lookup returns the
                // rotated version.
                //

                FOR_EACH_KEY {

                    Key = *Source++;
                    Rotated = _rotl(Key, 15);

                    Value = CphTableLookup(Key);

                }

                //
                // Loop through again and delete everything.
                //

                FOR_EACH_KEY {

                    Key = *Source++;
                    Rotated = _rotl(Key, 15);

                    Previous = CphTableDelete(Key);

                }

                //
                // And a final loop through to confirm all lookups now return 0.
                //

                FOR_EACH_KEY {

                    Key = *Source++;

                    Value = CphTableLookup(Key);

                }

            }

            QueryPerformanceCounter(&End);

            Delta.QuadPart = End.QuadPart - Start.QuadPart;

            if (Delta.LowPart < Best) {
                Best = Delta.LowPart;
            }
        }

    } else {

        while (Attempts--) {

            QueryPerformanceCounter(&Start);

            for (Count = Iterations; Count != 0; Count--) {

                //
                // Loop through the entire key set and insert a rotated version of the key.
                //

                FOR_EACH_KEY {

                    Key = *Source++;
                    Rotated = _rotl(Key, 15);

                    Previous = CphTableInsert(Key, Rotated);

                }

                //
                // Loop through the entire set again and ensure that lookup returns the
                // rotated version.
                //

                FOR_EACH_KEY {

                    Key = *Source++;
                    Rotated = _rotl(Key, 15);

                    Value = CphTableLookup(Key);

                }

                //
                // Loop through again and delete everything.
                //

                FOR_EACH_KEY {

                    Key = *Source++;
                    Rotated = _rotl(Key, 15);

                    Previous = CphTableDelete(Key);

                }

                //
                // And a final loop through to confirm all lookups now return 0.
                //

                FOR_EACH_KEY {

                    Key = *Source++;

                    Value = CphTableLookup(Key);

                }

            }

            QueryPerformanceCounter(&End);

            Delta.QuadPart = End.QuadPart - Start.QuadPart;

            if (Delta.LowPart < Best) {
                Best = Delta.LowPart;
            }

        }

    }

    return Best;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
