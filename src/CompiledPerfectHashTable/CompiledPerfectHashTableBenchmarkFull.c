
DECLARE_BENCHMARK_FULL_CPH_ROUTINE()
{
    ULONG Key;
    ULONG Count;
    ULONG Index;
    ULONG Value = 0;
    ULONG Rotated;
    ULONG Previous;
    ULONG Best = (ULONG)-1;
    ULONG Attempts = 100;
    const ULONG Iterations = 1000;
    const ULONG *Source;
    LARGE_INTEGER Start;
    LARGE_INTEGER End;
    LARGE_INTEGER Delta;

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

                    Previous = INSERT_ROUTINE(Key, Rotated);

                }

                //
                // Loop through the entire set again and ensure that lookup returns the
                // rotated version.
                //

                FOR_EACH_KEY {

                    Key = *Source++;
                    Rotated = _rotl(Key, 15);

                    Value = LOOKUP_ROUTINE(Key);

                }

                //
                // Loop through again and delete everything.
                //

                FOR_EACH_KEY {

                    Key = *Source++;
                    Rotated = _rotl(Key, 15);

                    Previous = DELETE_ROUTINE(Key);

                }

                //
                // And a final loop through to confirm all lookups now return 0.
                //

                FOR_EACH_KEY {

                    Key = *Source++;

                    Value = LOOKUP_ROUTINE(Key);

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

                    Previous = INSERT_ROUTINE(Key, Rotated);

                }

                //
                // Loop through the entire set again and ensure that lookup returns the
                // rotated version.
                //

                FOR_EACH_KEY {

                    Key = *Source++;
                    Rotated = _rotl(Key, 15);

                    Value = LOOKUP_ROUTINE(Key);

                }

                //
                // Loop through again and delete everything.
                //

                FOR_EACH_KEY {

                    Key = *Source++;
                    Rotated = _rotl(Key, 15);

                    Previous = DELETE_ROUTINE(Key);

                }

                //
                // And a final loop through to confirm all lookups now return 0.
                //

                FOR_EACH_KEY {

                    Key = *Source++;

                    Value = LOOKUP_ROUTINE(Key);

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

