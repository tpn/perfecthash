
CPH_MAIN()
{
    ULONG NumberOfErrors;
    BOOLEAN DebugBreakOnFailure = 0;

    NumberOfErrors = TEST_CPH_ROUTINE(DebugBreakOnFailure);

    CPH_EXIT(NumberOfErrors);
}

