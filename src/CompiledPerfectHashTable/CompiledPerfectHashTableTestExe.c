extern void ExitProcess(ULONG);

void
__stdcall
mainCRTStartup(
    void
    )
{
    ULONG NumberOfErrors;
    BOOLEAN DebugBreakOnFailure = 0;

    NumberOfErrors = TestCphTable(DebugBreakOnFailure);

    ExitProcess(NumberOfErrors);
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
