
DECLARE_BENCHMARK_INDEX_CPH_ROUTINE()
{
    ULONG Key;
    ULONG Index;
    ULONG Count;
    ULONG Attempt = 1000;
    const ULONG Iterations = 100000;
    LARGE_INTEGER Start;
    LARGE_INTEGER End;
    LARGE_INTEGER Delta;
    ULONG Best = (ULONG)-1;

    Key = *KEYS;

    if (Seconds) {

        while (!CtrlCPressed) {

            QueryPerformanceCounter(&Start);

            for (Count = Iterations; Count != 0; Count--) {
                Index = INDEX_ROUTINE(Key);
            }

            QueryPerformanceCounter(&End);

            Delta.QuadPart = End.QuadPart - Start.QuadPart;

            if (Delta.LowPart < Best) {
                Best = Delta.LowPart;
            }

        }

    } else {

        while (Attempt--) {

            QueryPerformanceCounter(&Start);

            for (Count = Iterations; Count != 0; Count--) {
                Index = INDEX_ROUTINE(Key);
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

