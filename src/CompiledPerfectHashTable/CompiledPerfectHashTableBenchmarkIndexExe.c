
//
// Begin CompiledPerfectHashTableBenchmarkIndexExe.c.
//

#pragma optimize("", off)

void
__stdcall
mainCRTStartup(
    void
    )
{
    ULONG Cycles;
    ULONG Seconds = 0;

    Cycles = BENCHMARK_INDEX_CPH_ROUTINE(Seconds);

    ExitProcess(Cycles);
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
