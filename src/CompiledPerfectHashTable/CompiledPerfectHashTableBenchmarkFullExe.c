#pragma optimize("", off)

extern void ExitProcess(ULONG);

volatile ULONG CtrlCPressed = 0;

void
__stdcall
mainCRTStartup(
    void
    )
{
    ULONG Cycles;
    ULONG Seconds = 0;

    Cycles = BenchmarkFullCphTable(Seconds);

    ExitProcess(Cycles);
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
