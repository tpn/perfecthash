
#pragma optimize("", off)

void
__stdcall
mainCRTStartup(
    void
    )
{
    ULONG Cycles;
    ULONG Seconds = 0;

    Cycles = BENCHMARK_FULL_CPH_ROUTINE(Seconds);

    ExitProcess(Cycles);
}

