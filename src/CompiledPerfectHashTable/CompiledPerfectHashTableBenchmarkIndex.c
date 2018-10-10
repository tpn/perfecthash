extern
BOOLEAN
QueryPerformanceCounter(
    _Out_ PLARGE_INTEGER Count
    );

extern volatile ULONG CtrlCPressed;

DECLARE_BENCHMARK_INDEX_COMPILED_PERFECT_HASH_TABLE_ROUTINE_HEADER()
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

    Key = *CphTableKeys;

    if (Seconds) {

        while (!CtrlCPressed) {

            QueryPerformanceCounter(&Start);

            for (Count = Iterations; Count != 0; Count--) {
                Index = CphTableIndex(Key);
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
                Index = CphTableIndex(Key);
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
