//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR PerfectHashBulkCreateExeUsageTextRawCStr[] =
    "Usage: PerfectHashBulkCreate.exe <KeysDirectory> <OutputDirectory> <Algorithm>\n"
    "    <HashFunction> <MaskFunction> <MaximumConcurrency>\n"
    "    [BulkCreateFlags] [KeysLoadFlags] [TableCreateFlags]\n"
    "    [TableCompileFlags] [TableCreateParameters]\n"
    "\n"
    "Bulk Create Flags:\n"
    "\n"
    "    N/A\n"
    "\n"
    "Keys Load Flags:\n"
    "\n"
    "    --TryLargePagesForKeysData\n"
    "\n"
    "        Tries to allocate the keys buffer using large pages.\n"
    "\n"
    "Table Create Flags:\n"
    "\n"
    "    --FirstGraphWins [default]\n"
    "\n"
    "        The first graph that is found is the one used.  This is the default\n"
    "        behavior.\n"
    "\n"
    "    --FindBestGraph\n"
    "\n"
    "        Requires the following two table create parameters to be present:\n"
    "\n"
    "            --BestCoverageNumAttempts=N\n"
    "\n"
    "                Where N is a positive integer, and represents the number of\n"
    "                attempts that will be made at finding a \"best\" graph (based\n"
    "                on the best coverage type requested below) before the create\n"
    "                table routine returns.\n"
    "\n"
    "            --BestCoverageType=<CoverageType>\n"
    "\n"
    "                Indicates the predicate to determine what constitutes the best\n"
    "                graph.\n"
    "\n"
    "                Valid coverage types:\n"
    "\n"
    "                    HighestNumberOfEmptyCacheLines\n"
    "\n"
    "Table Compile Flags:\n"
    "\n"
    "    N/A\n"
    "\n"
    "Table Create Parameters:\n"
    "\n"
    "    --AttemptsBeforeTableResize=N [default = 18]\n"
    "\n"
    "        Specifies the number of attempts at solving the graph that will be made\n"
    "        before a table resize event will occur (assuming that resize events are\n"
    "        permitted, as per the following flag).\n"
    "\n"
    "    --MaxNumberOfTableResizes=N [default = 5]\n"
    "\n"
    "        Maximum number of table resizes that will be permitted before giving up.\n"
    "\n"
    "    See also the --BestCoverage* parameters listed above in the table create\n"
    "    flag's --FindBestGraph section.\n"
    "\n"
;

const STRING PerfectHashBulkCreateExeUsageTextRawCString = {
    sizeof(PerfectHashBulkCreateExeUsageTextRawCStr) - sizeof(CHAR),
    sizeof(PerfectHashBulkCreateExeUsageTextRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&PerfectHashBulkCreateExeUsageTextRawCStr,
};

#ifndef RawCString
#define RawCString (&PerfectHashBulkCreateExeUsageTextRawCString)
#endif
