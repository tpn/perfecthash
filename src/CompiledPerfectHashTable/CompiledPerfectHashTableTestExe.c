
void
__stdcall
mainCRTStartup(
    void
    )
{
    ULONG NumberOfErrors;
    BOOLEAN DebugBreakOnFailure = 0;

    NumberOfErrors = TEST_CPH_ROUTINE(DebugBreakOnFailure);

    ExitProcess(NumberOfErrors);
}

