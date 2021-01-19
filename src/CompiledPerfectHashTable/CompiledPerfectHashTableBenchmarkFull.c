
DECLARE_BENCHMARK_FULL_CPH_ROUTINE()
{
    ULONG Count;
    CPHKEY Key;
    CPHKEY Rotated;
    CPHINDEX Index;
    CPHVALUE Value = 0;
    CPHVALUE Previous;
    ULONG Best = (ULONG)-1;
    ULONG Attempts = 100;
    const ULONG Iterations = 10;
    const CPHKEY *Source;
    LARGE_INTEGER Start;
    LARGE_INTEGER End;
    LARGE_INTEGER Delta;

    if (Seconds) {

        while (!CtrlCPressed) {

            CphQueryPerformanceCounter(&Start);

            for (Count = Iterations; Count != 0; Count--) {

                //
                // Loop through the entire key set and insert a rotated version
                // of the key.
                //

                FOR_EACH_KEY {
                    Key = *Source++;
                    Rotated = ROTATE_KEY_LEFT(Key, 15);
                    Previous = INSERT_ROUTINE(Key, (CPHVALUE)Rotated);
                }

                //
                // Loop through the entire set again and ensure that lookup
                // returns the rotated version.
                //

                FOR_EACH_KEY {
                    Key = *Source++;
                    Value = LOOKUP_ROUTINE(Key);
                }

                //
                // Loop through again and delete everything.
                //

                FOR_EACH_KEY {
                    Key = *Source++;
                    Previous = DELETE_ROUTINE(Key);
                }

            }

            CphQueryPerformanceCounter(&End);

            Delta.QuadPart = End.QuadPart - Start.QuadPart;

            if (Delta.LowPart < Best) {
                Best = Delta.LowPart;
            }
        }

    } else {

        while (Attempts--) {

            CphQueryPerformanceCounter(&Start);

            for (Count = Iterations; Count != 0; Count--) {

                //
                // Loop through the entire key set and insert a rotated version
                // of the key.
                //

                FOR_EACH_KEY {
                    Key = *Source++;
                    Rotated = ROTATE_KEY_LEFT(Key, 15);
                    Previous = INSERT_ROUTINE(Key, (CPHVALUE)Rotated);
                }

                //
                // Loop through the entire set again and ensure that lookup
                // returns the rotated version.
                //

                FOR_EACH_KEY {
                    Key = *Source++;
                    Value = LOOKUP_ROUTINE(Key);
                }

                //
                // Loop through again and delete everything.
                //

                FOR_EACH_KEY {
                    Key = *Source++;
                    Previous = DELETE_ROUTINE(Key);
                }
            }

            CphQueryPerformanceCounter(&End);

            Delta.QuadPart = End.QuadPart - Start.QuadPart;

            if (Delta.LowPart < Best) {
                Best = Delta.LowPart;
            }

        }

    }

    return Best;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
