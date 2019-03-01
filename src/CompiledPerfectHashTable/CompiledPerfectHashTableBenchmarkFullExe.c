
CPH_MAIN()
{
    ULONG Cycles;
    ULONG Seconds = 0;

    Cycles = BENCHMARK_FULL_CPH_ROUTINE(Seconds);

    CPH_EXIT(Cycles);
}

