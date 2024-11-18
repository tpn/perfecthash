
DECLARE_BENCHMARK_INDEX_CPH_ROUTINE()
{
    CPHKEY Key;
    CPHINDEX Index;
    ULONG Count;
    ULONG Attempt = 1000;
    const ULONG Iterations = 1000;
    LARGE_INTEGER Start;
    LARGE_INTEGER End;
    LARGE_INTEGER Delta;
    ULONG Best = (ULONG)-1;

    Key = *KEYS;

    if (Seconds) {

        while (!CtrlCPressed) {

            CphQueryPerformanceCounter(&Start);

            for (Count = Iterations; Count != 0; Count--) {
                Index = INDEX_INLINE_ROUTINE(Key);
            }

            CphQueryPerformanceCounter(&End);

            Delta.QuadPart = End.QuadPart - Start.QuadPart;

            if (Delta.LowPart < Best) {
                Best = Delta.LowPart;
            }

        }

    } else {

        while (Attempt--) {

            CphQueryPerformanceCounter(&Start);

            for (Count = Iterations; Count != 0; Count--) {
                Index = INDEX_INLINE_ROUTINE(Key);
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

