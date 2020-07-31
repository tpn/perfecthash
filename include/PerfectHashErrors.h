/*++

Copyright (c) 2018-2020 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashErrors.h

Abstract:

    This is the public header file for status codes used by the perfect hash
    library.  It is automatically generated from the messages defined in the
    file src/PerfectHash/PerfectHashErrors.mc by the helper batch script named
    src/PerfectHash/build-message-tables.bat (which must be run whenever the
    .mc file changes).

--*/
//
//  Values are 32 bit values laid out as follows:
//
//   3 3 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1
//   1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0
//  +-+-+-+-+-+---------------------+-------------------------------+
//  |S|R|C|N|r|    Facility         |               Code            |
//  +-+-+-+-+-+---------------------+-------------------------------+
//
//  where
//
//      S - Severity - indicates success/fail
//
//          0 - Success
//          1 - Fail (COERROR)
//
//      R - reserved portion of the facility code, corresponds to NT's
//              second severity bit.
//
//      C - reserved portion of the facility code, corresponds to NT's
//              C field.
//
//      N - reserved portion of the facility code. Used to indicate a
//              mapped NT status value.
//
//      r - reserved portion of the facility code. Reserved for internal
//              use. Used to indicate HRESULT values that are not status
//              values, but are instead message ids for display strings.
//
//      Facility - is the facility code
//
//      Code - is the facility's status code
//
//
// Define the facility codes
//
#define PH_FACILITY_ITF                  0x4


//
// Define the severity codes
//
#define PH_SEVERITY_SUCCESS              0x0
#define PH_SEVERITY_INFORMATIONAL        0x1
#define PH_SEVERITY_WARNING              0x2
#define PH_SEVERITY_FAIL                 0x3


//
// MessageId: PH_S_GRAPH_SOLVED
//
// MessageText:
//
// Graph solved.
//
#define PH_S_GRAPH_SOLVED                ((HRESULT)0x20040001L)

//
// MessageId: PH_S_GRAPH_NOT_SOLVED
//
// MessageText:
//
// Graph not solved.
//
#define PH_S_GRAPH_NOT_SOLVED            ((HRESULT)0x20040002L)

//
// MessageId: PH_S_CONTINUE_GRAPH_SOLVING
//
// MessageText:
//
// Continue graph solving.
//
#define PH_S_CONTINUE_GRAPH_SOLVING      ((HRESULT)0x20040003L)

//
// MessageId: PH_S_STOP_GRAPH_SOLVING
//
// MessageText:
//
// Stop graph solving.
//
#define PH_S_STOP_GRAPH_SOLVING          ((HRESULT)0x20040004L)

//
// MessageId: PH_S_GRAPH_VERIFICATION_SKIPPED
//
// MessageText:
//
// Graph verification skipped.
//
#define PH_S_GRAPH_VERIFICATION_SKIPPED  ((HRESULT)0x20040005L)

//
// MessageId: PH_S_GRAPH_SOLVING_STOPPED
//
// MessageText:
//
// Graph solving has been stopped.
//
#define PH_S_GRAPH_SOLVING_STOPPED       ((HRESULT)0x20040006L)

//
// MessageId: PH_S_TABLE_RESIZE_IMMINENT
//
// MessageText:
//
// Table resize imminent.
//
#define PH_S_TABLE_RESIZE_IMMINENT       ((HRESULT)0x20040007L)

//
// MessageId: PH_S_USE_NEW_GRAPH_FOR_SOLVING
//
// MessageText:
//
// Use new graph for solving.
//
#define PH_S_USE_NEW_GRAPH_FOR_SOLVING   ((HRESULT)0x20040008L)

//
// MessageId: PH_S_NO_KEY_SIZE_EXTRACTED_FROM_FILENAME
//
// MessageText:
//
// No key size extracted from file name.
//
#define PH_S_NO_KEY_SIZE_EXTRACTED_FROM_FILENAME ((HRESULT)0x20040009L)


////////////////////////////////////////////////////////////////////////////////
// PH_SEVERITY_INFORMATIONAL
////////////////////////////////////////////////////////////////////////////////

//
// MessageId: PH_I_CREATE_TABLE_ROUTINE_RECEIVED_SHUTDOWN_EVENT
//
// MessageText:
//
// Create table routine received shutdown event.
//
#define PH_I_CREATE_TABLE_ROUTINE_RECEIVED_SHUTDOWN_EVENT ((HRESULT)0x60040080L)

//
// MessageId: PH_I_CREATE_TABLE_ROUTINE_FAILED_TO_FIND_SOLUTION
//
// MessageText:
//
// Create table routine failed to find perfect hash solution.
//
#define PH_I_CREATE_TABLE_ROUTINE_FAILED_TO_FIND_SOLUTION ((HRESULT)0x60040081L)

//
// MessageId: PH_I_MAXIMUM_NUMBER_OF_TABLE_RESIZE_EVENTS_REACHED
//
// MessageText:
//
// The maximum number of table resize events was reached before a perfect hash table solution could be found.
//
#define PH_I_MAXIMUM_NUMBER_OF_TABLE_RESIZE_EVENTS_REACHED ((HRESULT)0x60040082L)

//
// MessageId: PH_I_REQUESTED_NUMBER_OF_TABLE_ELEMENTS_TOO_LARGE
//
// MessageText:
//
// The requested number of table elements was too large.
//
#define PH_I_REQUESTED_NUMBER_OF_TABLE_ELEMENTS_TOO_LARGE ((HRESULT)0x60040083L)

//
// MessageId: PH_I_FAILED_TO_ALLOCATE_MEMORY_FOR_ALL_GRAPHS
//
// MessageText:
//
// Failed to allocate memory for all graphs.
//
#define PH_I_FAILED_TO_ALLOCATE_MEMORY_FOR_ALL_GRAPHS ((HRESULT)0x60040084L)

//
// MessageId: PH_I_LOW_MEMORY
//
// MessageText:
//
// The system is running low on free memory.
//
#define PH_I_LOW_MEMORY                  ((HRESULT)0x60040085L)

//
// MessageId: PH_I_OUT_OF_MEMORY
//
// MessageText:
//
// The system is out of memory.
//
#define PH_I_OUT_OF_MEMORY               ((HRESULT)0x60040086L)

//
// MessageId: PH_I_TABLE_CREATED_BUT_VALUES_ARRAY_ALLOC_FAILED
//
// MessageText:
//
// The table was created successfully, however, the values array could not be allocated.  The table cannot be used.
//
#define PH_I_TABLE_CREATED_BUT_VALUES_ARRAY_ALLOC_FAILED ((HRESULT)0x60040087L)

//
// MessageId: PH_I_CUDA_OUT_OF_MEMORY
//
// MessageText:
//
// The CUDA device is out of memory.
//
#define PH_I_CUDA_OUT_OF_MEMORY          ((HRESULT)0x60040088L)


////////////////////////////////////////////////////////////////////////////////
// PH_SEVERITY_INFORMATIONAL -- Usage Messages
////////////////////////////////////////////////////////////////////////////////

//
// MessageId: PH_MSG_PERFECT_HASH_ALGO_HASH_MASK_NAMES
//
// MessageText:
//
// Algorithms:
// 
//    ID | Name
//     1   Chm01
// 
// Hash Functions:
// 
//    ID | Name (Number of Seeds)
//     2   Jenkins (2)
//    12   Fnv (2)
//    14   Crc32RotateX (3)
//    15   Crc32RotateXY (3)
//    16   Crc32RotateWXYZ (3)
//    17   RotateMultiplyXorRotate (3)
//    18   ShiftMultiplyXorShift (3)
//    19   ShiftMultiplyXorShift2 (6)
//    20   RotateMultiplyXorRotate2 (6)
//    21   MultiplyRotateR (3)
//    22   MultiplyRotateLR (3)
//    23   MultiplyShiftR (3)
//    24   MultiplyShiftLR (3)
//    25   Multiply (2)
//    26   MultiplyXor (4)
//    27   MultiplyRotateRMultiply (5)
//    28   MultiplyRotateR2 (5)
//    29   MultiplyShiftRMultiply (5)
//    30   MultiplyShiftR2 (5)
//    31   RotateRMultiply (3)
//    32   RotateRMultiplyRotateR (3)
// 
// Mask Functions:
// 
//   ID | Name
//    1   Modulus (does not work!)
//    2   And
// 
//
#define PH_MSG_PERFECT_HASH_ALGO_HASH_MASK_NAMES ((HRESULT)0x60040100L)

//
// MessageId: PH_MSG_PERFECT_HASH_BULK_CREATE_EXE_USAGE
//
// MessageText:
//
// Usage: PerfectHashBulkCreate.exe
//     <KeysDirectory> <OutputDirectory>
//     <Algorithm> <HashFunction> <MaskFunction>
//     <MaximumConcurrency>
//     [BulkCreateFlags] [KeysLoadFlags] [TableCreateFlags]
//     [TableCompileFlags] [TableCreateParameters]
// 
// Bulk Create Flags:
// 
//     --SkipTestAfterCreate
// 
//         Normally, after a table has been successfully created, it is tested.
//         Setting this flag disables this behavior.
// 
//         N.B. This will also disable benchmarking, so no performance information
//              will be present in the .csv output file.
// 
//     --Compile
// 
//         Compiles the table after creation.
// 
//         N.B. msbuild.exe must be on the PATH environment variable for this
//              to work.  (The current error message if this isn't the case is
//              quite cryptic.)
// 
// Keys Load Flags:
// 
//     --TryLargePagesForKeysData
// 
//         Tries to allocate the keys buffer using large pages.
// 
//     --SkipKeysVerification
// 
//         Skips the logic that enumerates all keys after loading and a) ensures
//         they are sorted, and b) constructs a keys bitmap.  If you can be certain
//         the keys are sorted, specifying this flag can provide a speedup when
//         loading large key sets.
// 
//     --DisableImplicitKeyDownsizing
// 
//         When loading keys that are 64-bit (8 bytes), a bitmap is kept that
//         tracks whether or not a given bit was seen across the entire key set.
//         After enumerating the set, the number of zeros in the bitmap are
//         counted; if this number is less than or equal to 32, it means that the
//         entire key set can be compressed into 32-bit values with some parallel
//         bit extraction logic (i.e. _pext_u64()).  As this has beneficial size
//         and performance implications, when detected, the key load operation will
//         implicitly heap-allocate another array and convert all the 64-bit keys
//         into their unique 32-bit equivalent.  Specifying this flag will disable
//         this behavior.
// 
//     --TryInferKeySizeFromKeysFilename
// 
//         The default key size is 32-bit (4 bytes).  When this flag is present,
//         if the keys file name ends with "64.keys" (e.g. "foo64.keys"), the key
//         size will be interpreted as 64-bit (8 bytes).  This flag takes
//         precedence over the table create parameter --KeySizeInBytes.
// 
// Table Create Flags:
// 
//     --Silent
// 
//         Disables console printing of the dots, dashes and other characters used
//         to (crudely) visualize the result of individual table create operations.
// 
//     --NoFileIo
// 
//         Disables writing of all files when a perfect hash solution has been
//         found.  The only time you would use this flag from the console
//         application is to observe the performance of table creation without
//         performing any file I/O operations.
// 
//     --Paranoid
// 
//         Enables redundant checks in the routine that determines whether or not
//         a generated graph is acyclic.
// 
//     --FirstGraphWins | --FindBestGraph
// 
//         --FirstGraphWins [default]
// 
//             This is the default behavior.  When searching for solutions in
//             parallel, the first graph to be found, "wins".  i.e. it's the
//             solution that is subsequently written to disk.
// 
//         --FindBestGraph
// 
//             Requires the following two table create parameters to be present:
// 
//                 --BestCoverageAttempts=N
//                 --BestCoverageType=<CoverageType>
// 
//             The table create routine will then run until it finds the number of
//             best coverage attempts specified.  At that point, the graph that was
//             found to be the "best" based on the coverage type predicate "wins",
//             and is subsequently saved to disk.
// 
//             N.B. This option is significantly more CPU intensive than the
//                  --FirstGraphWins mode, however, it is highly probable that it
//                  will find a graph that is better (based on the predicate) than
//                  when in first graph wins mode.
// 
//     --SkipMemoryCoverageInFirstGraphWinsMode
// 
//         Skips calculating memory coverage information when in "first graph wins"
//         mode.  This will result in the corresponding fields in the .csv output
//         indicating 0.
// 
//     --SkipGraphVerification
// 
//         When present, skips the internal graph verification check that ensures
//         a valid perfect hash solution has been found (i.e. with no collisions
//         across the entire key set).
// 
//     --DisableCsvOutputFile
// 
//         When present, disables writing the .csv output file.  This is required
//         when running multiple instances of the tool against the same output
//         directory in parallel.
// 
//     --OmitCsvRowIfTableCreateFailed
// 
//         When present, omits writing a row in the .csv output file if table
//         creation fails for a given keys file.  Ignored if --DisableCsvOutputFile
//         is speficied.
// 
//     --OmitCsvRowIfTableCreateSucceeded
// 
//         When present, omits writing a row in the .csv output file if table
//         creation succeeded for a given keys file.  Ignored if
//         --DisableCsvOutputFile is specified.
// 
//     --IndexOnly
// 
//         When set, affects the generated C files by defining the C preprocessor
//         macro CPH_INDEX_ONLY, which results in omitting the compiled perfect
//         hash routines that deal with the underlying table values array (i.e.
//         any routine other than Index(); e.g. Insert(), Lookup(), Delete() etc),
//         as well as the array itself.  This results in a size reduction of the
//         final compiled perfect hash binary.  Additionally, only the .dll and
//         BenchmarkIndex projects will be built, as the BenchmarkFull and Test
//         projects require access to a table values array.  This flag is intended
//         to be used if you only need the Index() routine and will be managing the
//         table values array independently.
// 
//     --UseRwsSectionForTableValues
// 
//         When set, tells the linker to use a shared read-write section for the
//         table values array, e.g.: #pragma comment(linker,"/section:.cphval,rws")
//         This will result in the table values array being accessible across
//         multiple processes.  Thus, the array will persist as long as one process
//         maintains an open section (mapping); i.e. keeps the .dll loaded.
// 
//     --UseNonTemporalAvx2Routines
// 
//         When set, uses implementations of RtlCopyPages and RtlFillPages that
//         use non-temporal hints.
// 
//     --ClampNumberOfEdges
// 
//         When present, clamps the number of edges to always be equal to the
//         number of keys, rounded up to a power of two, regardless of the
//         number of table resizes currently in effect.  Normally, when a table
//         is resized, the number of vertices are doubled, and the number of
//         edges are set to the number of vertices shifted right once (divided
//         by two).  When this flag is set, the vertex doubling stays the same,
//         however, the number of edges is always clamped to be equal to the
//         number of keys rounded up to a power of two.  This is a research
//         option used to evaluate the impact of the number of edges on the
//         graph solving probability for a given key set.  Only applies to
//         And masking (i.e. not modulus masking).
// 
//     --UseOriginalSeededHashRoutines
// 
//         When set, uses the original (slower) seeded hash routines (the ones
//         that return an HRESULT return code and write the hash value to an
//         output parameter) -- as opposed to using the newer, faster, "Ex"
//         version of the hash routines.
// 
//         N.B. This flag is incompatible with --HashAllKeysFirst.
// 
//     --HashAllKeysFirst
// 
//         When set, changes the graph solving logic such that vertices (i.e.
//         hash values) are generated for all keys up-front, prior to graph
//         construction.  The hashed keys are stored in a "vertex pair" array.
//         The page table type and page protection applied to this array can be
//         further refined by the following flags.
// 
//         N.B. This flag is incompatible with --UseOriginalSeededHashRoutines.
// 
//     --EnableWriteCombineForVertexPairs
// 
//         When set, allocates the memory for the vertex pairs array with
//         write-combine page protection.
// 
//         N.B. Only applies when --HashAllKeysFirst is set.  Incompatible with
//              --TryLargePagesForVertexPairs.
// 
//     --RemoveWriteCombineAfterSuccessfulHashKeys
// 
//         When set, automatically changes the page protection of the vertex
//         pairs array (after successful hashing of all keys without any vertex
//         collisions) from PAGE_READWRITE|PAGE_WRITECOMBINE to PAGE_READONLY.
// 
//         N.B. Only applies when the flags --EnableWriteCombineForVertexPairs
//              and --HashAllKeysFirst is set.
// 
//     --TryLargePagesForVertexPairs
// 
//         When set, tries to allocate the array for vertex pairs using large
//         pages.
// 
//         N.B. Only applies when HashAllKeysFirst is set.  Incompatible with
//              EnableWriteCombineForVertexPairs.
// 
//     --UsePreviousTableSize
// 
//         When set, uses any previously-recorded table sizes associated with
//         the keys file for the given algorithm, hash function and masking type.
// 
//         N.B. To forcibly delete all previously-recorded table sizes from all
//              keys in a directory, the following PowerShell snippet can be used:
// 
//              PS C:\Temp\keys> Get-Item -Path *.keys -Stream *.TableSize | Remove-Item
// 
//     --IncludeNumberOfTableResizeEventsInOutputPath
// 
//         When set, incorporates the number of table resize events encountered
//         whilst searching for a perfect hash solution into the final output
//         names, e.g.:
// 
//             C:\Temp\output\KernelBase_2485_1_Chm01_Crc32Rotate_And\...
//                                            ^
//                                            Number of resize events.
// 
//     --IncludeNumberOfTableElementsInOutputPath
// 
//         When set, incorporates the number of table elements (i.e. the final
//         table size) into the output path, e.g.:
// 
//             C:\Temp\output\KernelBase_2485_16384_Chm01_Crc32Rotate_And\...
//                                            ^
//                                            Number of table elements.
// 
//         N.B. These two flags can be combined, yielding a path as follows:
// 
//             C:\Temp\output\KernelBase_2485_1_16384_Chm01_Crc32Rotate_And\...
// 
//         N.B. It is important to understand how table resize events impact the
//              behavior of this program if one or both of these flags are present.
//              Using the example above, the initial path that will be used for
//              the solution will be:
// 
//                 C:\Temp\output\KernelBase_2485_0_8192_Chm01_Crc32Rotate_And\...
// 
//              After the maximum number of attempts are reached, a table resize
//              event occurs; the new path component will be:
// 
//                 C:\Temp\output\KernelBase_2485_1_16384_Chm01_Crc32Rotate_And\...
// 
//              However, the actual renaming of the directory is not done until
//              a solution has been found and all files have been written.  If
//              this program is being run repeatedly, then the target directory
//              will already exist.  This complicates things, as, unlike files,
//              we can't just replace an existing directory with a new one.
// 
//              There are two ways this could be handled: a) delete all the
//              existing files under the target path, then delete the directory,
//              then perform the rename, or b) move the target directory somewhere
//              else first, preserving the existing contents, then proceed with
//              the rename.
// 
//              This program takes the latter approach.  The existing directory
//              will be moved to:
// 
//                 C:\Temp\output\old\KernelBase_1_16384_Chm01_Crc32Rotate_And_2018-11-19-011023-512\...
// 
//              The timestamp appended to the directory name is derived from the
//              existing directory's creation time, which should ensure uniqueness.
//              (In the unlikely situation the target directory already exists in
//              the old subdirectory, the whole operation is aborted and the table
//              create routine returns a failure.)
// 
//              The point of mentioning all of this is the following: when one or
//              both of these flags are routinely specified, the number of output
//              files rooted in the output directory's 'old' subdirectory will grow
//              rapidly, consuming a lot of disk space.  Thus, if the old files are
//              not required, it is recommended to regularly delete them manually.
// 
// Table Compile Flags:
// 
//     N/A
// 
// Table Create Parameters:
// 
//     --ValueSizeInBytes=4|8
// 
//         Sets the size, in bytes, of the value element that will be stored in the
//         compiled perfect hash table via Insert().  Defaults to 4 bytes (ULONG).
// 
//     --MainWorkThreadpoolPriority=<High|Normal|Low> [default: Normal]
//     --FileWorkThreadpoolPriority=<High|Normal|Low> [default: Normal]
// 
//         Sets the main work (i.e. the CPU-intensive graph solving) threadpool
//         priority, or the file work threadpool priority, to the given value.
// 
//     --AttemptsBeforeTableResize=N [default = 4,294,967,295 ]
// 
//         Specifies the number of attempts at solving the graph that will be made
//         before a table resize event will occur (assuming that resize events are
//         permitted, as per the following flag).
// 
//     --MaxNumberOfTableResizes=N [default = 5]
// 
//         Maximum number of table resizes that will be permitted before giving up.
// 
//     --InitialNumberOfTableResizes=N [default = 0]
// 
//         Initial number of table resizes to simulate before attempting graph
//         solving.  Each table resize doubles the number of vertices used to
//         solve the graph, which lowers the keys-to-vertices ratio, which will
//         improve graph solving probability.
// 
//         N.B. This parameter is only valid for And masking, not Modulus masking.
// 
//     --BestCoverageAttempts=N
// 
//         Where N is a positive integer, and represents the number of attempts
//         that will be made at finding a "best" graph (based on the best coverage
//         type requested below) before the create table routine returns.
// 
//     --BestCoverageType=<CoverageType>
// 
//         Indicates the predicate to determine what constitutes the best graph.
// 
//         Valid coverage types:
// 
//             HighestNumberOfEmptyPages
//             HighestNumberOfEmptyLargePages
//             HighestNumberOfEmptyCacheLines
//             HighestMaxGraphTraversalDepth
//             HighestTotalGraphTraversals
//             HighestMaxAssignedPerCacheLineCount
//             HighestNumberOfEmptyVertices
//             HighestNumberOfCollisionsDuringAssignment
// 
//             LowestNumberOfEmptyPages
//             LowestNumberOfEmptyLargePages
//             LowestNumberOfEmptyCacheLines
//             LowestMaxGraphTraversalDepth
//             LowestTotalGraphTraversals
//             LowestMaxAssignedPerCacheLineCount
//             LowestNumberOfEmptyVertices
//             LowestNumberOfCollisionsDuringAssignment
// 
//         The following predicates must be used in conjunction with --KeysSubset:
// 
//             HighestMaxAssignedPerCacheLineCountForKeysSubset
//             HighestNumberOfPagesUsedByKeysSubset
//             HighestNumberOfLargePagesUsedByKeysSubset
//             HighestNumberOfCacheLinesUsedByKeysSubset
// 
//             LowestMaxAssignedPerCacheLineCountForKeysSubset
//             LowestNumberOfPagesUsedByKeysSubset
//             LowestNumberOfLargePagesUsedByKeysSubset
//             LowestNumberOfCacheLinesUsedByKeysSubset
// 
//     --Seeds=<n1,...n8>
// 
//         Supplies an optional comma-separated list of up to 8 integers that
//         represent the seed values to use for every graph solving attempt.
// 
//         N.B. This is rarely useful for PerfectHashBulkCreate.exe.  It is more
//              useful for PerfectHashCreate.exe when you want to re-run against
//              a set of known-good seeds.
// 
//     --Seed3Byte1MaskCounts=<n1,...n31>
//     --Seed3Byte2MaskCounts=<n1,...n31>
// 
//         Supplies a comma-separated list of 32 integers that represent weighted
//         counts of seed mask's byte values.  (Experimental.)
// 
//     --SolutionsFoundRatio=<double>
//     --TryUsePredictedAttemptsToLimitMaxConcurrency
// 
//         Supplies a double (64-bit) floating point number indicating the ratio
//         of solutions found (obtained from a prior run).  This is then used to
//         calculate the predicted number of attempts required to solve a given
//         graph; when combined with --TryUsePredictedAttemptsToLimitMaxConcurrency
//         the maximum concurrency used when solving will be the minimum of the
//         predicted attempts and the maximum concurrency indicated on the command
//         line.
// 
//         N.B. These parameters are typically less useful for bulk-create options
//              as each table will have different solving characteristics.
// 
// Console Output Character Legend
// 
//  Char | Meaning
// 
//     .   Table created successfully.
// 
//     +   Table resize event occured.
// 
//     x   Failed to create a table.  The maximum number of attempts at trying to
//         solve the table at a given size was reached, and no more resize attempts
//         were possible (due to the maximum resize limit also being hit).
// 
//     F   Failed to create a table due to a target not being reached by a specific
//         number of attempts or time duration.  Not yet implemented.
// 
//     *   None of the worker threads were able to allocate sufficient memory to
//         attempt solving the graph.
// 
//     !   The system is out of memory.
// 
//     L   The system is running low on memory (a low memory event is triggered
//         at about 90%% RAM usage).  In certain situations we can detect this
//         situation prior to actually running out of memory; in these cases,
//         we abort the current table creation attempt (which will instantly
//         relieve system memory pressure).
// 
//     V   The graph was created successfully, however, we weren't able to allocate
//         enough memory for the table values array in order for the array to be
//         used after creation.  This can be avoided by omitting --TestAfterCreate.
// 
//     T   The requested number of table elements was too large (exceeded 32 bits).
// 
//     S   A shutdown event was received.  This shouldn't be seen unless externally
//         signaling the named shutdown event associated with a context.
// 
//
#define PH_MSG_PERFECT_HASH_BULK_CREATE_EXE_USAGE ((HRESULT)0x60040101L)

//
// MessageId: PH_MSG_PERFECT_HASH_SELF_TEST_EXE_USAGE
//
// MessageText:
//
// Usage: PerfectHashSelfTest.exe
//     <TestDataDirectory> <OutputDirectory>
//     <Algorithm> <HashFunction> <MaskFunction>
//     <MaximumConcurrency>
// 
// N.B. This utility has been deprecated in favor of PerfectHashBulkCreate.exe.
// 
//
#define PH_MSG_PERFECT_HASH_SELF_TEST_EXE_USAGE ((HRESULT)0x60040102L)

//
// MessageId: PH_MSG_PERFECT_HASH_CREATE_EXE_USAGE
//
// MessageText:
//
// Usage: PerfectHashCreate.exe
//     <KeysPath> <OutputDirectory>
//     <Algorithm> <HashFunction> <MaskFunction>
//     <MaximumConcurrency>
//     [KeysLoadFlags] [TableCreateFlags]
//     [TableCompileFlags] [TableCreateParameters]
// 
// Keys Load Flags:
// 
//     --TryLargePagesForKeysData
// 
//         Tries to allocate the keys buffer using large pages.
// 
//     --SkipKeysVerification
// 
//         Skips the logic that enumerates all keys after loading and a) ensures
//         they are sorted, and b) constructs a keys bitmap.  If you can be certain
//         the keys are sorted, specifying this flag can provide a speedup when
//         loading large key sets.
// 
//     --DisableImplicitKeyDownsizing
// 
//         When loading keys that are 64-bit (8 bytes), a bitmap is kept that
//         tracks whether or not a given bit was seen across the entire key set.
//         After enumerating the set, the number of zeros in the bitmap are
//         counted; if this number is less than or equal to 32, it means that the
//         entire key set can be compressed into 32-bit values with some parallel
//         bit extraction logic (i.e. _pext_u64()).  As this has beneficial size
//         and performance implications, when detected, the key load operation will
//         implicitly heap-allocate another array and convert all the 64-bit keys
//         into their unique 32-bit equivalent.  Specifying this flag will disable
//         this behavior.
// 
//     --TryInferKeySizeFromKeysFilename
// 
//         The default key size is 32-bit (4 bytes).  When this flag is present,
//         if the keys file name ends with "64.keys" (e.g. "foo64.keys"), the key
//         size will be interpreted as 64-bit (8 bytes).  This flag takes
//         precedence over the table create parameter --KeySizeInBytes.
// 
// Table Create Flags:
// 
//     --SkipTestAfterCreate
// 
//         Normally, after a table has been successfully created, it is tested.
//         Setting this flag disables this behavior.
// 
//         N.B. This will also disable benchmarking, so no performance information
//              will be present in the .csv output file.
// 
//     --Silent
// 
//         Disables console printing of the dots, dashes and other characters used
//         to (crudely) visualize the result of individual table create operations.
// 
//     --NoFileIo
// 
//         Disables writing of all files when a perfect hash solution has been
//         found.  The only time you would use this flag from the console
//         application is to observe the performance of table creation without
//         performing any file I/O operations.
// 
//     --Paranoid
// 
//         Enables redundant checks in the routine that determines whether or not
//         a generated graph is acyclic.
// 
//     --FirstGraphWins | --FindBestGraph
// 
//         --FirstGraphWins [default]
// 
//             This is the default behavior.  When searching for solutions in
//             parallel, the first graph to be found, "wins".  i.e. it's the
//             solution that is subsequently written to disk.
// 
//         --FindBestGraph
// 
//             Requires the following two table create parameters to be present:
// 
//                 --BestCoverageAttempts=N
//                 --BestCoverageType=<CoverageType>
// 
//             The table create routine will then run until it finds the number of
//             best coverage attempts specified.  At that point, the graph that was
//             found to be the "best" based on the coverage type predicate "wins",
//             and is subsequently saved to disk.
// 
//             N.B. This option is significantly more CPU intensive than the
//                  --FirstGraphWins mode, however, it is highly probable that it
//                  will find a graph that is better (based on the predicate) than
//                  when in first graph wins mode.
// 
//     --SkipMemoryCoverageInFirstGraphWinsMode
// 
//         Skips calculating memory coverage information when in "first graph wins"
//         mode.  This will result in the corresponding fields in the .csv output
//         indicating 0.
// 
//     --SkipGraphVerification
// 
//         When present, skips the internal graph verification check that ensures
//         a valid perfect hash solution has been found (i.e. with no collisions
//         across the entire key set).
// 
//     --DisableCsvOutputFile
// 
//         When present, disables writing the .csv output file.  This is required
//         when running multiple instances of the tool against the same output
//         directory in parallel.
// 
//     --OmitCsvRowIfTableCreateFailed
// 
//         When present, omits writing a row in the .csv output file if table
//         creation fails for a given keys file.  Ignored if --DisableCsvOutputFile
//         is speficied.
// 
//     --OmitCsvRowIfTableCreateSucceeded
// 
//         When present, omits writing a row in the .csv output file if table
//         creation succeeded for a given keys file.  Ignored if
//         --DisableCsvOutputFile is specified.
// 
//     --IndexOnly
// 
//         When set, affects the generated C files by defining the C preprocessor
//         macro CPH_INDEX_ONLY, which results in omitting the compiled perfect
//         hash routines that deal with the underlying table values array (i.e.
//         any routine other than Index(); e.g. Insert(), Lookup(), Delete() etc),
//         as well as the array itself.  This results in a size reduction of the
//         final compiled perfect hash binary.  Additionally, only the .dll and
//         BenchmarkIndex projects will be built, as the BenchmarkFull and Test
//         projects require access to a table values array.  This flag is intended
//         to be used if you only need the Index() routine and will be managing the
//         table values array independently.
// 
//     --UseRwsSectionForTableValues
// 
//         When set, tells the linker to use a shared read-write section for the
//         table values array, e.g.: #pragma comment(linker,"/section:.cphval,rws")
//         This will result in the table values array being accessible across
//         multiple processes.  Thus, the array will persist as long as one process
//         maintains an open section (mapping); i.e. keeps the .dll loaded.
// 
//     --UseNonTemporalAvx2Routines
// 
//         When set, uses implementations of RtlCopyPages and RtlFillPages that
//         use non-temporal hints.
// 
//     --ClampNumberOfEdges
// 
//         When present, clamps the number of edges to always be equal to the
//         number of keys, rounded up to a power of two, regardless of the
//         number of table resizes currently in effect.  Normally, when a table
//         is resized, the number of vertices are doubled, and the number of
//         edges are set to the number of vertices shifted right once (divided
//         by two).  When this flag is set, the vertex doubling stays the same,
//         however, the number of edges is always clamped to be equal to the
//         number of keys rounded up to a power of two.  This is a research
//         option used to evaluate the impact of the number of edges on the
//         graph solving probability for a given key set.  Only applies to
//         And masking (i.e. not modulus masking).
// 
//     --UseOriginalSeededHashRoutines
// 
//         When set, uses the original (slower) seeded hash routines (the ones
//         that return an HRESULT return code and write the hash value to an
//         output parameter) -- as opposed to using the newer, faster, "Ex"
//         version of the hash routines.
// 
//         N.B. This flag is incompatible with --HashAllKeysFirst.
// 
//     --HashAllKeysFirst
// 
//         When set, changes the graph solving logic such that vertices (i.e.
//         hash values) are generated for all keys up-front, prior to graph
//         construction.  The hashed keys are stored in a "vertex pair" array.
//         The page table type and page protection applied to this array can be
//         further refined by the following flags.
// 
//         N.B. This flag is incompatible with --UseOriginalSeededHashRoutines.
// 
//     --EnableWriteCombineForVertexPairs
// 
//         When set, allocates the memory for the vertex pairs array with
//         write-combine page protection.
// 
//         N.B. Only applies when --HashAllKeysFirst is set.  Incompatible with
//              --TryLargePagesForVertexPairs.
// 
//     --RemoveWriteCombineAfterSuccessfulHashKeys
// 
//         When set, automatically changes the page protection of the vertex
//         pairs array (after successful hashing of all keys without any vertex
//         collisions) from PAGE_READWRITE|PAGE_WRITECOMBINE to PAGE_READONLY.
// 
//         N.B. Only applies when the flags --EnableWriteCombineForVertexPairs
//              and --HashAllKeysFirst is set.
// 
//     --TryLargePagesForVertexPairs
// 
//         When set, tries to allocate the array for vertex pairs using large
//         pages.
// 
//         N.B. Only applies when HashAllKeysFirst is set.  Incompatible with
//              EnableWriteCombineForVertexPairs.
// 
//     --UsePreviousTableSize
// 
//         When set, uses any previously-recorded table sizes associated with
//         the keys file for the given algorithm, hash function and masking type.
// 
//         N.B. To forcibly delete all previously-recorded table sizes from all
//              keys in a directory, the following PowerShell snippet can be used:
// 
//              PS C:\Temp\keys> Get-Item -Path *.keys -Stream *.TableSize | Remove-Item
// 
//     --IncludeNumberOfTableResizeEventsInOutputPath
// 
//         When set, incorporates the number of table resize events encountered
//         whilst searching for a perfect hash solution into the final output
//         names, e.g.:
// 
//             C:\Temp\output\KernelBase_2485_1_Chm01_Crc32Rotate_And\...
//                                            ^
//                                            Number of resize events.
// 
//     --IncludeNumberOfTableElementsInOutputPath
// 
//         When set, incorporates the number of table elements (i.e. the final
//         table size) into the output path, e.g.:
// 
//             C:\Temp\output\KernelBase_2485_16384_Chm01_Crc32Rotate_And\...
//                                            ^
//                                            Number of table elements.
// 
//         N.B. These two flags can be combined, yielding a path as follows:
// 
//             C:\Temp\output\KernelBase_2485_1_16384_Chm01_Crc32Rotate_And\...
// 
//         N.B. It is important to understand how table resize events impact the
//              behavior of this program if one or both of these flags are present.
//              Using the example above, the initial path that will be used for
//              the solution will be:
// 
//                 C:\Temp\output\KernelBase_2485_0_8192_Chm01_Crc32Rotate_And\...
// 
//              After the maximum number of attempts are reached, a table resize
//              event occurs; the new path component will be:
// 
//                 C:\Temp\output\KernelBase_2485_1_16384_Chm01_Crc32Rotate_And\...
// 
//              However, the actual renaming of the directory is not done until
//              a solution has been found and all files have been written.  If
//              this program is being run repeatedly, then the target directory
//              will already exist.  This complicates things, as, unlike files,
//              we can't just replace an existing directory with a new one.
// 
//              There are two ways this could be handled: a) delete all the
//              existing files under the target path, then delete the directory,
//              then perform the rename, or b) move the target directory somewhere
//              else first, preserving the existing contents, then proceed with
//              the rename.
// 
//              This program takes the latter approach.  The existing directory
//              will be moved to:
// 
//                 C:\Temp\output\old\KernelBase_1_16384_Chm01_Crc32Rotate_And_2018-11-19-011023-512\...
// 
//              The timestamp appended to the directory name is derived from the
//              existing directory's creation time, which should ensure uniqueness.
//              (In the unlikely situation the target directory already exists in
//              the old subdirectory, the whole operation is aborted and the table
//              create routine returns a failure.)
// 
//              The point of mentioning all of this is the following: when one or
//              both of these flags are routinely specified, the number of output
//              files rooted in the output directory's 'old' subdirectory will grow
//              rapidly, consuming a lot of disk space.  Thus, if the old files are
//              not required, it is recommended to regularly delete them manually.
// 
// Table Compile Flags:
// 
//     N/A
// 
// Table Create Parameters:
// 
//     --ValueSizeInBytes=4|8
// 
//         Sets the size, in bytes, of the value element that will be stored in the
//         compiled perfect hash table via Insert().  Defaults to 4 bytes (ULONG).
// 
//     --MainWorkThreadpoolPriority=<High|Normal|Low> [default: Normal]
//     --FileWorkThreadpoolPriority=<High|Normal|Low> [default: Normal]
// 
//         Sets the main work (i.e. the CPU-intensive graph solving) threadpool
//         priority, or the file work threadpool priority, to the given value.
// 
//     --AttemptsBeforeTableResize=N [default = 4,294,967,295 ]
// 
//         Specifies the number of attempts at solving the graph that will be made
//         before a table resize event will occur (assuming that resize events are
//         permitted, as per the following flag).
// 
//     --MaxNumberOfTableResizes=N [default = 5]
// 
//         Maximum number of table resizes that will be permitted before giving up.
// 
//     --InitialNumberOfTableResizes=N [default = 0]
// 
//         Initial number of table resizes to simulate before attempting graph
//         solving.  Each table resize doubles the number of vertices used to
//         solve the graph, which lowers the keys-to-vertices ratio, which will
//         improve graph solving probability.
// 
//         N.B. This parameter is only valid for And masking, not Modulus masking.
// 
//     --BestCoverageAttempts=N
// 
//         Where N is a positive integer, and represents the number of attempts
//         that will be made at finding a "best" graph (based on the best coverage
//         type requested below) before the create table routine returns.
// 
//     --BestCoverageType=<CoverageType>
// 
//         Indicates the predicate to determine what constitutes the best graph.
// 
//         Valid coverage types:
// 
//             HighestNumberOfEmptyPages
//             HighestNumberOfEmptyLargePages
//             HighestNumberOfEmptyCacheLines
//             HighestMaxGraphTraversalDepth
//             HighestTotalGraphTraversals
//             HighestMaxAssignedPerCacheLineCount
//             HighestNumberOfEmptyVertices
//             HighestNumberOfCollisionsDuringAssignment
// 
//             LowestNumberOfEmptyPages
//             LowestNumberOfEmptyLargePages
//             LowestNumberOfEmptyCacheLines
//             LowestMaxGraphTraversalDepth
//             LowestTotalGraphTraversals
//             LowestMaxAssignedPerCacheLineCount
//             LowestNumberOfEmptyVertices
//             LowestNumberOfCollisionsDuringAssignment
// 
//         The following predicates must be used in conjunction with --KeysSubset:
// 
//             HighestMaxAssignedPerCacheLineCountForKeysSubset
//             HighestNumberOfPagesUsedByKeysSubset
//             HighestNumberOfLargePagesUsedByKeysSubset
//             HighestNumberOfCacheLinesUsedByKeysSubset
// 
//             LowestMaxAssignedPerCacheLineCountForKeysSubset
//             LowestNumberOfPagesUsedByKeysSubset
//             LowestNumberOfLargePagesUsedByKeysSubset
//             LowestNumberOfCacheLinesUsedByKeysSubset
// 
//     --KeysSubset=N,N+1[,N+2,N+3,...] (e.g. --KeysSubset=10,50,123,600,670)
// 
//         Supplies a comma-separated list of keys in ascending key-value order.
//         Must contain two or more elements.
// 
//     --Seeds=<n1,...n8>
// 
//         Supplies an optional comma-separated list of up to 8 integers that
//         represent the seed values to use for every graph solving attempt.
// 
//     --Seed3Byte1MaskCounts=<n1,...n31>
//     --Seed3Byte2MaskCounts=<n1,...n31>
// 
//         Supplies a comma-separated list of 32 integers that represent weighted
//         counts of seed mask's byte values.  (Experimental.)
// 
//     --SolutionsFoundRatio=<double>
//     --TryUsePredictedAttemptsToLimitMaxConcurrency
// 
//         Supplies a double (64-bit) floating point number indicating the ratio
//         of solutions found (obtained from a prior run).  This is then used to
//         calculate the predicted number of attempts required to solve a given
//         graph; when combined with --TryUsePredictedAttemptsToLimitMaxConcurrency
//         the maximum concurrency used when solving will be the minimum of the
//         predicted attempts and the maximum concurrency indicated on the command
//         line.
// 
//
#define PH_MSG_PERFECT_HASH_CREATE_EXE_USAGE ((HRESULT)0x60040103L)


////////////////////////////////////////////////////////////////////////////////
// PH_SEVERITY_FAIL
////////////////////////////////////////////////////////////////////////////////

//
// MessageId: PH_E_CREATE_TABLE_ALREADY_IN_PROGRESS
//
// MessageText:
//
// A table creation operation is in progress for this context.
//
#define PH_E_CREATE_TABLE_ALREADY_IN_PROGRESS ((HRESULT)0xE0040201L)

//
// MessageId: PH_E_TOO_MANY_KEYS
//
// MessageText:
//
// Too many keys.
//
#define PH_E_TOO_MANY_KEYS               ((HRESULT)0xE0040202L)

//
// MessageId: PH_E_INFO_FILE_SMALLER_THAN_HEADER
//
// MessageText:
//
// :Info file is smaller than smallest known table header size.
//
#define PH_E_INFO_FILE_SMALLER_THAN_HEADER ((HRESULT)0xE0040203L)

//
// MessageId: PH_E_INVALID_MAGIC_VALUES
//
// MessageText:
//
// Invalid magic values.
//
#define PH_E_INVALID_MAGIC_VALUES        ((HRESULT)0xE0040204L)

//
// MessageId: PH_E_INVALID_INFO_HEADER_SIZE
//
// MessageText:
//
// Invalid :Info header size.
//
#define PH_E_INVALID_INFO_HEADER_SIZE    ((HRESULT)0xE0040205L)

//
// MessageId: PH_E_NUM_KEYS_MISMATCH_BETWEEN_HEADER_AND_KEYS
//
// MessageText:
//
// The number of keys reported in the keys file does not match the number of keys reported in the header.
//
#define PH_E_NUM_KEYS_MISMATCH_BETWEEN_HEADER_AND_KEYS ((HRESULT)0xE0040206L)

//
// MessageId: PH_E_INVALID_ALGORITHM_ID
//
// MessageText:
//
// Invalid algorithm ID.
//
#define PH_E_INVALID_ALGORITHM_ID        ((HRESULT)0xE0040207L)

//
// MessageId: PH_E_INVALID_HASH_FUNCTION_ID
//
// MessageText:
//
// Invalid hash function ID.
//
#define PH_E_INVALID_HASH_FUNCTION_ID    ((HRESULT)0xE0040208L)

//
// MessageId: PH_E_INVALID_MASK_FUNCTION_ID
//
// MessageText:
//
// Invalid mask function ID.
//
#define PH_E_INVALID_MASK_FUNCTION_ID    ((HRESULT)0xE0040209L)

//
// MessageId: PH_E_HEADER_KEY_SIZE_TOO_LARGE
//
// MessageText:
//
// The key size reported by the header is too large.
//
#define PH_E_HEADER_KEY_SIZE_TOO_LARGE   ((HRESULT)0xE004020AL)

//
// MessageId: PH_E_NUM_KEYS_IS_ZERO
//
// MessageText:
//
// The number of keys is zero.
//
#define PH_E_NUM_KEYS_IS_ZERO            ((HRESULT)0xE004020BL)

//
// MessageId: PH_E_NUM_TABLE_ELEMENTS_IS_ZERO
//
// MessageText:
//
// The number of table elements is zero.
//
#define PH_E_NUM_TABLE_ELEMENTS_IS_ZERO  ((HRESULT)0xE004020CL)

//
// MessageId: PH_E_NUM_KEYS_EXCEEDS_NUM_TABLE_ELEMENTS
//
// MessageText:
//
// The number of keys exceeds the number of table elements.
//
#define PH_E_NUM_KEYS_EXCEEDS_NUM_TABLE_ELEMENTS ((HRESULT)0xE004020DL)

//
// MessageId: PH_E_EXPECTED_EOF_ACTUAL_EOF_MISMATCH
//
// MessageText:
//
// The expected end of file does not match the actual end of file.
//
#define PH_E_EXPECTED_EOF_ACTUAL_EOF_MISMATCH ((HRESULT)0xE004020EL)

//
// MessageId: PH_E_KEYS_FILE_SIZE_NOT_MULTIPLE_OF_KEY_SIZE
//
// MessageText:
//
// The keys file size is not a multiple of the key size.
//
#define PH_E_KEYS_FILE_SIZE_NOT_MULTIPLE_OF_KEY_SIZE ((HRESULT)0xE004020FL)

//
// MessageId: PH_E_KEYS_NUM_SET_BITS_NUM_KEYS_MISMATCH
//
// MessageText:
//
// The number of bits set for the keys bitmap does not match the number of keys.
//
#define PH_E_KEYS_NUM_SET_BITS_NUM_KEYS_MISMATCH ((HRESULT)0xE0040210L)

//
// MessageId: PH_E_DUPLICATE_KEYS_DETECTED
//
// MessageText:
//
// Duplicate keys detected.  Key files must not contain duplicate keys.
//
#define PH_E_DUPLICATE_KEYS_DETECTED     ((HRESULT)0xE0040211L)

//
// Disabled 21st Nov 2018: deprecated in favor of E_OUTOFMEMORY and PH_I_OUT_OF_MEMORY.
// MessageId=0x212
// Severity=Fail
// Facility=ITF
// SymbolicName=PH_E_HEAP_CREATE_FAILED
// Language=English
// A call to HeapCreate() failed.
// .
//
// MessageId: PH_E_RTL_LOAD_SYMBOLS_FROM_MULTIPLE_MODULES_FAILED
//
// MessageText:
//
// A call to RtlLoadSymbolsFromMultipleModules() failed.
//
#define PH_E_RTL_LOAD_SYMBOLS_FROM_MULTIPLE_MODULES_FAILED ((HRESULT)0xE0040213L)

//
// MessageId: PH_E_CONTEXT_SELF_TEST_INVALID_NUM_ARGS
//
// MessageText:
//
// Invalid number of arguments for context self-test.
//
#define PH_E_CONTEXT_SELF_TEST_INVALID_NUM_ARGS ((HRESULT)0xE0040214L)

//
// MessageId: PH_E_INVALID_MAXIMUM_CONCURRENCY
//
// MessageText:
//
// Invalid value for maximum concurrency.
//
#define PH_E_INVALID_MAXIMUM_CONCURRENCY ((HRESULT)0xE0040215L)

//
// MessageId: PH_E_SET_MAXIMUM_CONCURRENCY_FAILED
//
// MessageText:
//
// Setting the maximum concurrency of the context failed.
//
#define PH_E_SET_MAXIMUM_CONCURRENCY_FAILED ((HRESULT)0xE0040216L)

//
// MessageId: PH_E_INITIALIZE_LARGE_PAGES_FAILED
//
// MessageText:
//
// Internal error when attempting to initialize large pages.
//
#define PH_E_INITIALIZE_LARGE_PAGES_FAILED ((HRESULT)0xE0040217L)

//
// MessageId: PH_E_KEYS_NOT_SORTED
//
// MessageText:
//
// The keys file supplied was not sorted.
//
#define PH_E_KEYS_NOT_SORTED             ((HRESULT)0xE0040218L)

//
// MessageId: PH_E_KEYS_NOT_LOADED
//
// MessageText:
//
// A keys file has not been loaded yet.
//
#define PH_E_KEYS_NOT_LOADED             ((HRESULT)0xE0040219L)

//
// MessageId: PH_E_KEYS_LOAD_ALREADY_IN_PROGRESS
//
// MessageText:
//
// A key loading operation is already in progress.
//
#define PH_E_KEYS_LOAD_ALREADY_IN_PROGRESS ((HRESULT)0xE004021AL)

//
// MessageId: PH_E_KEYS_ALREADY_LOADED
//
// MessageText:
//
// A set of keys has already been loaded.
//
#define PH_E_KEYS_ALREADY_LOADED         ((HRESULT)0xE004021BL)

//
// MessageId: PH_E_INVALID_KEYS_LOAD_FLAGS
//
// MessageText:
//
// Invalid key load flags.
//
#define PH_E_INVALID_KEYS_LOAD_FLAGS     ((HRESULT)0xE004021CL)

//
// MessageId: PH_E_INVALID_KEY_SIZE
//
// MessageText:
//
// Invalid key size.
//
#define PH_E_INVALID_KEY_SIZE            ((HRESULT)0xE004021DL)

//
// MessageId: PH_E_TABLE_NOT_LOADED
//
// MessageText:
//
// No table has been loaded yet.
//
#define PH_E_TABLE_NOT_LOADED            ((HRESULT)0xE004021EL)

//
// MessageId: PH_E_TABLE_LOAD_ALREADY_IN_PROGRESS
//
// MessageText:
//
// A table loading operation is already in progress.
//
#define PH_E_TABLE_LOAD_ALREADY_IN_PROGRESS ((HRESULT)0xE004021FL)

//
// Removed 2018-10-01.
//
// MessageId=0x220
// Severity=Fail
// Facility=ITF
// SymbolicName=PH_E_INVALID_CONTEXT_CREATE_TABLE_FLAGS
// Language=English
// Invalid context create table flags.
// .
//
//
// MessageId: PH_E_INVALID_CONTEXT_SELF_TEST_FLAGS
//
// MessageText:
//
// Invalid context self-test flags.
//
#define PH_E_INVALID_CONTEXT_SELF_TEST_FLAGS ((HRESULT)0xE0040221L)

//
// MessageId: PH_E_INVALID_TABLE_LOAD_FLAGS
//
// MessageText:
//
// Invalid table load flags.
//
#define PH_E_INVALID_TABLE_LOAD_FLAGS    ((HRESULT)0xE0040222L)

//
// MessageId: PH_E_TABLE_LOCKED
//
// MessageText:
//
// Table is locked.
//
#define PH_E_TABLE_LOCKED                ((HRESULT)0xE0040223L)

//
// MessageId: PH_E_SYSTEM_CALL_FAILED
//
// MessageText:
//
// System call failed.
//
#define PH_E_SYSTEM_CALL_FAILED          ((HRESULT)0xE0040224L)

//
// MessageId: PH_E_TABLE_ALREADY_CREATED
//
// MessageText:
//
// The table instance has already been created.
//
#define PH_E_TABLE_ALREADY_CREATED       ((HRESULT)0xE0040225L)

//
// MessageId: PH_E_TABLE_ALREADY_LOADED
//
// MessageText:
//
// The table instance has already been loaded.
//
#define PH_E_TABLE_ALREADY_LOADED        ((HRESULT)0xE0040226L)

//
// MessageId: PH_E_INVALID_TABLE_COMPILE_FLAGS
//
// MessageText:
//
// Invalid table compile flags.
//
#define PH_E_INVALID_TABLE_COMPILE_FLAGS ((HRESULT)0xE0040227L)

//
// MessageId: PH_E_TABLE_COMPILATION_NOT_AVAILABLE
//
// MessageText:
//
// Table compilation is not available for the current combination of architecture, algorithm ID, hash function and masking type.
//
#define PH_E_TABLE_COMPILATION_NOT_AVAILABLE ((HRESULT)0xE0040228L)

//
// Disabled 8th Nov 2018: changed to PH_I_MAXIMUM_NUMBER_OF_TABLE_RESIZE_EVENTS_REACHED
// MessageId=0x229
// Severity=Fail
// Facility=ITF
// SymbolicName=PH_E_MAXIMUM_NUMBER_OF_TABLE_RESIZE_EVENTS_REACHED
// Language=English
// The maximum number of table resize events was reached before a perfect hash table solution could be found.
// .
//
// Disabled 8th Nov 2018: changed to PH_I_REQUESTED_NUMBER_OF_TABLE_ELEMENTS_TOO_LARGE
// MessageId=0x22a
// Severity=Fail
// Facility=ITF
// SymbolicName=PH_E_REQUESTED_NUMBER_OF_TABLE_ELEMENTS_TOO_LARGE
// Language=English
// The requested number of table elements was too large.
// .
//
// MessageId: PH_E_ERROR_DURING_PREPARE_TABLE_FILE
//
// MessageText:
//
// Error preparing perfect hash table file.
//
#define PH_E_ERROR_DURING_PREPARE_TABLE_FILE ((HRESULT)0xE004022BL)

//
// MessageId: PH_E_ERROR_DURING_SAVE_TABLE_FILE
//
// MessageText:
//
// Error saving perfect hash table file.
//
#define PH_E_ERROR_DURING_SAVE_TABLE_FILE ((HRESULT)0xE004022CL)

//
// A perfect hash table solution was found, however, it did not
// pass internal validation checks (e.g. collisions were found
// when attempting to independently verify that the perfect hash
// function generated no collisions).
//
//
// MessageId: PH_E_TABLE_VERIFICATION_FAILED
//
// MessageText:
//
// Table verification failed.
//
#define PH_E_TABLE_VERIFICATION_FAILED   ((HRESULT)0xE004022DL)

//
// MessageId: PH_E_TABLE_CROSS_COMPILATION_NOT_AVAILABLE
//
// MessageText:
//
// Table cross-compilation is not available.
//
#define PH_E_TABLE_CROSS_COMPILATION_NOT_AVAILABLE ((HRESULT)0xE004022EL)

//
// MessageId: PH_E_INVALID_CPU_ARCH_ID
//
// MessageText:
//
// The CPU architecture ID was invalid.
//
#define PH_E_INVALID_CPU_ARCH_ID         ((HRESULT)0xE004022FL)

//
// MessageId: PH_E_NOT_IMPLEMENTED
//
// MessageText:
//
// Functionality not yet implemented.
//
#define PH_E_NOT_IMPLEMENTED             ((HRESULT)0xE0040230L)

//
// MessageId: PH_E_WORK_IN_PROGRESS
//
// MessageText:
//
// Work in progress.
//
#define PH_E_WORK_IN_PROGRESS            ((HRESULT)0xE0040231L)

//
// MessageId: PH_E_KEYS_FILE_BASENAME_NOT_VALID_C_IDENTIFIER
//
// MessageText:
//
// Keys file base name is not a valid C identifier.
//
#define PH_E_KEYS_FILE_BASENAME_NOT_VALID_C_IDENTIFIER ((HRESULT)0xE0040232L)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_C_HEADER_FILE
//
// MessageText:
//
// Error preparing C header file.
//
#define PH_E_ERROR_DURING_PREPARE_C_HEADER_FILE ((HRESULT)0xE0040233L)

//
// MessageId: PH_E_ERROR_DURING_SAVE_C_HEADER_FILE
//
// MessageText:
//
// Error saving C header file.
//
#define PH_E_ERROR_DURING_SAVE_C_HEADER_FILE ((HRESULT)0xE0040234L)

//
// MessageId: PH_E_UNREACHABLE_CODE
//
// MessageText:
//
// Unreachable code reached.
//
#define PH_E_UNREACHABLE_CODE            ((HRESULT)0xE0040235L)

//
// MessageId: PH_E_INVARIANT_CHECK_FAILED
//
// MessageText:
//
// Internal invariant check failed.
//
#define PH_E_INVARIANT_CHECK_FAILED      ((HRESULT)0xE0040236L)

//
// MessageId: PH_E_OVERFLOWED_HEADER_FILE_MAPPING_SIZE
//
// MessageText:
//
// The calculated C header file size exceeded 4GB.
//
#define PH_E_OVERFLOWED_HEADER_FILE_MAPPING_SIZE ((HRESULT)0xE0040237L)

//
// MessageId: PH_E_CONTEXT_BASE_OUTPUT_DIRECTORY_NOT_SET
//
// MessageText:
//
// Base output directory has not been set.
//
#define PH_E_CONTEXT_BASE_OUTPUT_DIRECTORY_NOT_SET ((HRESULT)0xE0040238L)

//
// MessageId: PH_E_CONTEXT_LOCKED
//
// MessageText:
//
// The context is locked.
//
#define PH_E_CONTEXT_LOCKED              ((HRESULT)0xE0040239L)

//
// MessageId: PH_E_CONTEXT_RESET_FAILED
//
// MessageText:
//
// Failed to reset context.
//
#define PH_E_CONTEXT_RESET_FAILED        ((HRESULT)0xE004023AL)

//
// MessageId: PH_E_CONTEXT_SET_BASE_OUTPUT_DIRECTORY_FAILED
//
// MessageText:
//
// Failed to set context output directory.
//
#define PH_E_CONTEXT_SET_BASE_OUTPUT_DIRECTORY_FAILED ((HRESULT)0xE004023BL)

//
// MessageId: PH_E_NO_TABLE_CREATED_OR_LOADED
//
// MessageText:
//
// The table has not been created or loaded.
//
#define PH_E_NO_TABLE_CREATED_OR_LOADED  ((HRESULT)0xE004023CL)

//
// MessageId: PH_E_TABLE_PATHS_ALREADY_INITIALIZED
//
// MessageText:
//
// Paths have already been initialized for this table instance.
//
#define PH_E_TABLE_PATHS_ALREADY_INITIALIZED ((HRESULT)0xE004023DL)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_TABLE_INFO_STREAM
//
// MessageText:
//
// Error preparing :Info stream.
//
#define PH_E_ERROR_DURING_PREPARE_TABLE_INFO_STREAM ((HRESULT)0xE004023EL)

//
// MessageId: PH_E_ERROR_DURING_SAVE_TABLE_INFO_STREAM
//
// MessageText:
//
// Error saving :Info stream.
//
#define PH_E_ERROR_DURING_SAVE_TABLE_INFO_STREAM ((HRESULT)0xE004023FL)

//
// MessageId: PH_E_ERROR_DURING_SAVE_C_SOURCE_FILE
//
// MessageText:
//
// Error saving C source file.
//
#define PH_E_ERROR_DURING_SAVE_C_SOURCE_FILE ((HRESULT)0xE0040240L)

//
// MessageId: PH_E_ERROR_DURING_SAVE_C_SOURCE_KEYS_FILE
//
// MessageText:
//
// Error saving C source keys file.
//
#define PH_E_ERROR_DURING_SAVE_C_SOURCE_KEYS_FILE ((HRESULT)0xE0040241L)

//
// MessageId: PH_E_ERROR_DURING_SAVE_C_SOURCE_TABLE_DATA_FILE
//
// MessageText:
//
// Error saving C source table data file.
//
#define PH_E_ERROR_DURING_SAVE_C_SOURCE_TABLE_DATA_FILE ((HRESULT)0xE0040242L)

//
// MessageId: PH_E_FILE_CLOSED
//
// MessageText:
//
// The file has already been closed.
//
#define PH_E_FILE_CLOSED                 ((HRESULT)0xE0040243L)

//
// MessageId: PH_E_FILE_NOT_OPEN
//
// MessageText:
//
// The file has not been opened yet, or has been closed.
//
#define PH_E_FILE_NOT_OPEN               ((HRESULT)0xE0040244L)

//
// MessageId: PH_E_FILE_LOCKED
//
// MessageText:
//
// The file is locked.
//
#define PH_E_FILE_LOCKED                 ((HRESULT)0xE0040245L)

//
// MessageId: PH_E_KEYS_LOCKED
//
// MessageText:
//
// The keys are locked.
//
#define PH_E_KEYS_LOCKED                 ((HRESULT)0xE0040246L)

//
// MessageId: PH_E_MAPPING_SIZE_LESS_THAN_OR_EQUAL_TO_CURRENT_SIZE
//
// MessageText:
//
// Mapping size is less than or equal to current file size.
//
#define PH_E_MAPPING_SIZE_LESS_THAN_OR_EQUAL_TO_CURRENT_SIZE ((HRESULT)0xE0040247L)

//
// MessageId: PH_E_FILE_READONLY
//
// MessageText:
//
// The file is readonly.
//
#define PH_E_FILE_READONLY               ((HRESULT)0xE0040248L)

//
// MessageId: PH_E_FILE_VIEW_CREATED
//
// MessageText:
//
// A file view has already been created.
//
#define PH_E_FILE_VIEW_CREATED           ((HRESULT)0xE0040249L)

//
// MessageId: PH_E_FILE_VIEW_MAPPED
//
// MessageText:
//
// A file view has already been mapped.
//
#define PH_E_FILE_VIEW_MAPPED            ((HRESULT)0xE004024AL)

//
// MessageId: PH_E_FILE_MAPPING_SIZE_IS_ZERO
//
// MessageText:
//
// The mapping size for the file is zero.
//
#define PH_E_FILE_MAPPING_SIZE_IS_ZERO   ((HRESULT)0xE004024BL)

//
// MessageId: PH_E_FILE_MAPPING_SIZE_NOT_SYSTEM_ALIGNED
//
// MessageText:
//
// Mapping size is not aligned to the system allocation granularity.
//
#define PH_E_FILE_MAPPING_SIZE_NOT_SYSTEM_ALIGNED ((HRESULT)0xE004024CL)

//
// MessageId: PH_E_FILE_MAPPING_SIZE_NOT_LARGE_PAGE_ALIGNED
//
// MessageText:
//
// Mapping size is not aligned to the large page granularity.
//
#define PH_E_FILE_MAPPING_SIZE_NOT_LARGE_PAGE_ALIGNED ((HRESULT)0xE004024DL)

//
// MessageId: PH_E_FILE_ALREADY_OPEN
//
// MessageText:
//
// An existing file has already been loaded or created for this file instance.
//
#define PH_E_FILE_ALREADY_OPEN           ((HRESULT)0xE004024EL)

//
// MessageId: PH_E_INVALID_FILE_LOAD_FLAGS
//
// MessageText:
//
// Invalid file load flags.
//
#define PH_E_INVALID_FILE_LOAD_FLAGS     ((HRESULT)0xE004024FL)

//
// MessageId: PH_E_FILE_ALREADY_CLOSED
//
// MessageText:
//
// File already closed.
//
#define PH_E_FILE_ALREADY_CLOSED         ((HRESULT)0xE0040250L)

//
// MessageId: PH_E_FILE_EMPTY
//
// MessageText:
//
// The file is empty.
//
#define PH_E_FILE_EMPTY                  ((HRESULT)0xE0040251L)

//
// MessageId: PH_E_PATH_PARTS_EXTRACTION_FAILED
//
// MessageText:
//
// Failed to extract the path into parts.
//
#define PH_E_PATH_PARTS_EXTRACTION_FAILED ((HRESULT)0xE0040252L)

//
// MessageId: PH_E_PATH_LOCKED
//
// MessageText:
//
// Path is locked.
//
#define PH_E_PATH_LOCKED                 ((HRESULT)0xE0040253L)

//
// MessageId: PH_E_EXISTING_PATH_LOCKED
//
// MessageText:
//
// Existing path parameter is locked.
//
#define PH_E_EXISTING_PATH_LOCKED        ((HRESULT)0xE0040254L)

//
// MessageId: PH_E_PATH_ALREADY_SET
//
// MessageText:
//
// A path has already been set for this instance.
//
#define PH_E_PATH_ALREADY_SET            ((HRESULT)0xE0040255L)

//
// MessageId: PH_E_EXISTING_PATH_NO_PATH_SET
//
// MessageText:
//
// Existing path parameter has not had a path set.
//
#define PH_E_EXISTING_PATH_NO_PATH_SET   ((HRESULT)0xE0040256L)

//
// MessageId: PH_E_NO_PATH_SET
//
// MessageText:
//
// No path set.
//
#define PH_E_NO_PATH_SET                 ((HRESULT)0xE0040257L)

//
// MessageId: PH_E_STRING_BUFFER_OVERFLOW
//
// MessageText:
//
// An internal string buffer has overflowed.
//
#define PH_E_STRING_BUFFER_OVERFLOW      ((HRESULT)0xE0040258L)

//
// MessageId: PH_E_SOURCE_PATH_LOCKED
//
// MessageText:
//
// Source path parameter is locked.
//
#define PH_E_SOURCE_PATH_LOCKED          ((HRESULT)0xE0040259L)

//
// MessageId: PH_E_SOURCE_PATH_NO_PATH_SET
//
// MessageText:
//
// Source path parameter had no path set.
//
#define PH_E_SOURCE_PATH_NO_PATH_SET     ((HRESULT)0xE004025AL)

//
// MessageId: PH_E_INVALID_TABLE
//
// MessageText:
//
// Invalid table.
//
#define PH_E_INVALID_TABLE               ((HRESULT)0xE004025BL)

//
// MessageId: PH_E_INVALID_TABLE_CREATE_FLAGS
//
// MessageText:
//
// Invalid table create flags.
//
#define PH_E_INVALID_TABLE_CREATE_FLAGS  ((HRESULT)0xE004025CL)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_C_SOURCE_FILE
//
// MessageText:
//
// Error preparing C source file.
//
#define PH_E_ERROR_DURING_PREPARE_C_SOURCE_FILE ((HRESULT)0xE004025DL)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_C_SOURCE_KEYS_FILE
//
// MessageText:
//
// Error preparing C source keys file.
//
#define PH_E_ERROR_DURING_PREPARE_C_SOURCE_KEYS_FILE ((HRESULT)0xE004025EL)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_C_SOURCE_TABLE_DATA_FILE
//
// MessageText:
//
// Error preparing C source table data file..
//
#define PH_E_ERROR_DURING_PREPARE_C_SOURCE_TABLE_DATA_FILE ((HRESULT)0xE004025FL)

//
// MessageId: PH_E_FILE_NEVER_OPENED
//
// MessageText:
//
// The file has never been opened.
//
#define PH_E_FILE_NEVER_OPENED           ((HRESULT)0xE0040260L)

//
// MessageId: PH_E_FILE_NO_RENAME_SCHEDULED
//
// MessageText:
//
// The file has not had a rename operation scheduled.
//
#define PH_E_FILE_NO_RENAME_SCHEDULED    ((HRESULT)0xE0040261L)

//
// MessageId: PH_E_FILE_NOT_CLOSED
//
// MessageText:
//
// The file has not yet been closed.
//
#define PH_E_FILE_NOT_CLOSED             ((HRESULT)0xE0040262L)

//
// MessageId: PH_E_INVALID_FILE_WORK_ID
//
// MessageText:
//
// Invalid file work ID.
//
#define PH_E_INVALID_FILE_WORK_ID        ((HRESULT)0xE0040263L)

//
// MessageId: PH_E_INVALID_END_OF_FILE
//
// MessageText:
//
// Invalid end of file (less than or equal to 0).
//
#define PH_E_INVALID_END_OF_FILE         ((HRESULT)0xE0040264L)

//
// MessageId: PH_E_NEW_EOF_LESS_THAN_OR_EQUAL_TO_CURRENT_EOF
//
// MessageText:
//
// New end-of-file is less than or equal to current end-of-file.
//
#define PH_E_NEW_EOF_LESS_THAN_OR_EQUAL_TO_CURRENT_EOF ((HRESULT)0xE0040265L)

//
// MessageId: PH_E_RENAME_PATH_IS_SAME_AS_CURRENT_PATH
//
// MessageText:
//
// The rename path equivalent to the existing path.
//
#define PH_E_RENAME_PATH_IS_SAME_AS_CURRENT_PATH ((HRESULT)0xE0040266L)

//
// MessageId: PH_E_INVALID_FILE_CREATE_FLAGS
//
// MessageText:
//
// Invalid file create flags.
//
#define PH_E_INVALID_FILE_CREATE_FLAGS   ((HRESULT)0xE0040267L)

//
// MessageId: PH_E_INVALID_INTERFACE_ID
//
// MessageText:
//
// Invalid interface ID.
//
#define PH_E_INVALID_INTERFACE_ID        ((HRESULT)0xE0040268L)

//
// MessageId: PH_E_NO_TLS_CONTEXT_SET
//
// MessageText:
//
// PerfectHashTlsEnsureContext() was called but no TLS context was set.
//
#define PH_E_NO_TLS_CONTEXT_SET          ((HRESULT)0xE0040269L)

//
// MessageId: PH_E_NOT_GLOBAL_INTERFACE_ID
//
// MessageText:
//
// The interface ID provided is not a global interface.
//
#define PH_E_NOT_GLOBAL_INTERFACE_ID     ((HRESULT)0xE004026AL)

//
// MessageId: PH_E_INVALID_VALUE_SIZE
//
// MessageText:
//
// Invalid value size.
//
#define PH_E_INVALID_VALUE_SIZE          ((HRESULT)0xE004026BL)

//
// MessageId: PH_E_FILE_ALREADY_BEING_EXTENDED
//
// MessageText:
//
// A file extension operation is already in progress.
//
#define PH_E_FILE_ALREADY_BEING_EXTENDED ((HRESULT)0xE004026CL)

//
// MessageId: PH_E_COLLISIONS_ENCOUNTERED_DURING_GRAPH_VERIFICATION
//
// MessageText:
//
// Collisions encountered during graph verification.
//
#define PH_E_COLLISIONS_ENCOUNTERED_DURING_GRAPH_VERIFICATION ((HRESULT)0xE004026DL)

//
// MessageId: PH_E_NUM_ASSIGNMENTS_NOT_EQUAL_TO_NUM_KEYS_DURING_GRAPH_VERIFICATION
//
// MessageText:
//
// The number of value assignments did not equal the number of keys during graph verification.
//
#define PH_E_NUM_ASSIGNMENTS_NOT_EQUAL_TO_NUM_KEYS_DURING_GRAPH_VERIFICATION ((HRESULT)0xE004026EL)

//
// MessageId: PH_E_INVALID_NUMBER_OF_SEEDS
//
// MessageText:
//
// Invalid number of seeds.
//
#define PH_E_INVALID_NUMBER_OF_SEEDS     ((HRESULT)0xE004026FL)

//
// MessageId: PH_E_CONTEXT_BASE_OUTPUT_DIRECTORY_ALREADY_SET
//
// MessageText:
//
// Base output directory already set.
//
#define PH_E_CONTEXT_BASE_OUTPUT_DIRECTORY_ALREADY_SET ((HRESULT)0xE0040270L)

//
// MessageId: PH_E_DIRECTORY_ALREADY_CLOSED
//
// MessageText:
//
// The directory has already been closed.
//
#define PH_E_DIRECTORY_ALREADY_CLOSED    ((HRESULT)0xE0040271L)

//
// MessageId: PH_E_DIRECTORY_NOT_SET
//
// MessageText:
//
// The directory has not been opened yet, or has been closed.
//
#define PH_E_DIRECTORY_NOT_SET           ((HRESULT)0xE0040272L)

//
// MessageId: PH_E_DIRECTORY_LOCKED
//
// MessageText:
//
// The directory is locked.
//
#define PH_E_DIRECTORY_LOCKED            ((HRESULT)0xE0040273L)

//
// MessageId: PH_E_DIRECTORY_ALREADY_SET
//
// MessageText:
//
// The directory is already set.
//
#define PH_E_DIRECTORY_ALREADY_SET       ((HRESULT)0xE0040274L)

//
// MessageId: PH_E_DIRECTORY_DOES_NOT_EXIST
//
// MessageText:
//
// Directory does not exist.
//
#define PH_E_DIRECTORY_DOES_NOT_EXIST    ((HRESULT)0xE0040275L)

//
// MessageId: PH_E_INVALID_DIRECTORY_OPEN_FLAGS
//
// MessageText:
//
// Invalid directory open flags.
//
#define PH_E_INVALID_DIRECTORY_OPEN_FLAGS ((HRESULT)0xE0040276L)

//
// MessageId: PH_E_INVALID_DIRECTORY_CREATE_FLAGS
//
// MessageText:
//
// Invalid directory create flags.
//
#define PH_E_INVALID_DIRECTORY_CREATE_FLAGS ((HRESULT)0xE0040277L)

//
// MessageId: PH_E_DIRECTORY_NEVER_SET
//
// MessageText:
//
// The directory was never set.
//
#define PH_E_DIRECTORY_NEVER_SET         ((HRESULT)0xE0040278L)

//
// MessageId: PH_E_DIRECTORY_READONLY
//
// MessageText:
//
// The directory is readonly.
//
#define PH_E_DIRECTORY_READONLY          ((HRESULT)0xE0040279L)

//
// MessageId: PH_E_DIRECTORY_NO_RENAME_SCHEDULED
//
// MessageText:
//
// The directory has not had a rename operation scheduled.
//
#define PH_E_DIRECTORY_NO_RENAME_SCHEDULED ((HRESULT)0xE004027AL)

//
// MessageId: PH_E_DIRECTORY_NOT_CLOSED
//
// MessageText:
//
// Directory is not closed.
//
#define PH_E_DIRECTORY_NOT_CLOSED        ((HRESULT)0xE004027BL)

//
// MessageId: PH_E_FILE_ALREADY_ADDED_TO_A_DIRECTORY
//
// MessageText:
//
// The file has already been added to a directory.
//
#define PH_E_FILE_ALREADY_ADDED_TO_A_DIRECTORY ((HRESULT)0xE004027CL)

//
// MessageId: PH_E_FILE_ADDED_TO_DIFFERENT_DIRECTORY
//
// MessageText:
//
// The file was added to a different directory.
//
#define PH_E_FILE_ADDED_TO_DIFFERENT_DIRECTORY ((HRESULT)0xE004027DL)

//
// MessageId: PH_E_DIRECTORY_RENAME_ALREADY_SCHEDULED
//
// MessageText:
//
// Directory rename already scheduled.
//
#define PH_E_DIRECTORY_RENAME_ALREADY_SCHEDULED ((HRESULT)0xE004027EL)

//
// MessageId: PH_E_FILE_NOT_ADDED_TO_DIRECTORY
//
// MessageText:
//
// The file has not been added to a directory.
//
#define PH_E_FILE_NOT_ADDED_TO_DIRECTORY ((HRESULT)0xE004027FL)

//
// MessageId: PH_E_DIRECTORY_CLOSED
//
// MessageText:
//
// Directory is closed.
//
#define PH_E_DIRECTORY_CLOSED            ((HRESULT)0xE0040280L)

//
// MessageId: PH_E_CREATE_RANDOM_OBJECT_NAMES_LENGTH_OF_NAME_TOO_SHORT
//
// MessageText:
//
// LengthOfNameInChars parameter too short.
//
#define PH_E_CREATE_RANDOM_OBJECT_NAMES_LENGTH_OF_NAME_TOO_SHORT ((HRESULT)0xE0040281L)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_VCPROJECT_DLL_FILE
//
// MessageText:
//
// Error preparing Dll.vcxproj file.
//
#define PH_E_ERROR_DURING_PREPARE_VCPROJECT_DLL_FILE ((HRESULT)0xE0040282L)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_C_SOURCE_SUPPORT_FILE
//
// MessageText:
//
// Error preparing C source support file.
//
#define PH_E_ERROR_DURING_PREPARE_C_SOURCE_SUPPORT_FILE ((HRESULT)0xE0040283L)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_C_SOURCE_TEST_FILE
//
// MessageText:
//
// Error preparing C source test file.
//
#define PH_E_ERROR_DURING_PREPARE_C_SOURCE_TEST_FILE ((HRESULT)0xE0040284L)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_C_SOURCE_TEST_EXE_FILE
//
// MessageText:
//
// Error preparing C source test exe file.
//
#define PH_E_ERROR_DURING_PREPARE_C_SOURCE_TEST_EXE_FILE ((HRESULT)0xE0040285L)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_VCPROJECT_TEST_EXE_FILE
//
// MessageText:
//
// Error preparing TestExe.vcxproj file.
//
#define PH_E_ERROR_DURING_PREPARE_VCPROJECT_TEST_EXE_FILE ((HRESULT)0xE0040286L)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_C_SOURCE_BENCHMARK_FULL_FILE
//
// MessageText:
//
// Error preparing C source benchmark full file.
//
#define PH_E_ERROR_DURING_PREPARE_C_SOURCE_BENCHMARK_FULL_FILE ((HRESULT)0xE0040287L)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_C_SOURCE_BENCHMARK_FULL_EXE_FILE
//
// MessageText:
//
// Error preparing C source benchmark full exe file.
//
#define PH_E_ERROR_DURING_PREPARE_C_SOURCE_BENCHMARK_FULL_EXE_FILE ((HRESULT)0xE0040288L)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_VCPROJECT_BENCHMARK_FULL_EXE_FILE
//
// MessageText:
//
// Error preparing BenchmarkFullExe.vcxproj file.
//
#define PH_E_ERROR_DURING_PREPARE_VCPROJECT_BENCHMARK_FULL_EXE_FILE ((HRESULT)0xE0040289L)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_C_SOURCE_BENCHMARK_INDEX_FILE
//
// MessageText:
//
// Error preparing C source benchmark index file.
//
#define PH_E_ERROR_DURING_PREPARE_C_SOURCE_BENCHMARK_INDEX_FILE ((HRESULT)0xE004028AL)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_C_SOURCE_BENCHMARK_INDEX_EXE_FILE
//
// MessageText:
//
// Error preparing C source benchmark index exe file.
//
#define PH_E_ERROR_DURING_PREPARE_C_SOURCE_BENCHMARK_INDEX_EXE_FILE ((HRESULT)0xE004028BL)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_VCPROJECT_BENCHMARK_INDEX_EXE_FILE
//
// MessageText:
//
// Error preparing BenchmarkIndexExe.vcxproj file.
//
#define PH_E_ERROR_DURING_PREPARE_VCPROJECT_BENCHMARK_INDEX_EXE_FILE ((HRESULT)0xE004028CL)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_BUILD_SOLUTION_BATCH_FILE
//
// MessageText:
//
// Error preparing build solution batch file.
//
#define PH_E_ERROR_DURING_PREPARE_BUILD_SOLUTION_BATCH_FILE ((HRESULT)0xE004028DL)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_C_HEADER_COMPILED_PERFECT_HASH_FILE
//
// MessageText:
//
// Error preparing C header CompiledPerfectHash.h file.
//
#define PH_E_ERROR_DURING_PREPARE_C_HEADER_COMPILED_PERFECT_HASH_FILE ((HRESULT)0xE004028EL)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_VCPROPS_COMPILED_PERFECT_HASH_FILE
//
// MessageText:
//
// Error preparing CompiledPerfectHash.props file.
//
#define PH_E_ERROR_DURING_PREPARE_VCPROPS_COMPILED_PERFECT_HASH_FILE ((HRESULT)0xE004028FL)

//
// Spare IDs: 0x290, 0x291.
//
//
// MessageId: PH_E_ERROR_DURING_SAVE_VCPROJECT_DLL_FILE
//
// MessageText:
//
// Error saving Dll.vcxproj file.
//
#define PH_E_ERROR_DURING_SAVE_VCPROJECT_DLL_FILE ((HRESULT)0xE0040292L)

//
// MessageId: PH_E_ERROR_DURING_SAVE_C_SOURCE_SUPPORT_FILE
//
// MessageText:
//
// Error saving C source support file.
//
#define PH_E_ERROR_DURING_SAVE_C_SOURCE_SUPPORT_FILE ((HRESULT)0xE0040293L)

//
// MessageId: PH_E_ERROR_DURING_SAVE_C_SOURCE_TEST_FILE
//
// MessageText:
//
// Error saving C source test file.
//
#define PH_E_ERROR_DURING_SAVE_C_SOURCE_TEST_FILE ((HRESULT)0xE0040294L)

//
// MessageId: PH_E_ERROR_DURING_SAVE_C_SOURCE_TEST_EXE_FILE
//
// MessageText:
//
// Error saving C source test exe file.
//
#define PH_E_ERROR_DURING_SAVE_C_SOURCE_TEST_EXE_FILE ((HRESULT)0xE0040295L)

//
// MessageId: PH_E_ERROR_DURING_SAVE_VCPROJECT_TEST_EXE_FILE
//
// MessageText:
//
// Error saving TestExe.vcxproj file.
//
#define PH_E_ERROR_DURING_SAVE_VCPROJECT_TEST_EXE_FILE ((HRESULT)0xE0040296L)

//
// MessageId: PH_E_ERROR_DURING_SAVE_C_SOURCE_BENCHMARK_FULL_FILE
//
// MessageText:
//
// Error saving C source benchmark full file.
//
#define PH_E_ERROR_DURING_SAVE_C_SOURCE_BENCHMARK_FULL_FILE ((HRESULT)0xE0040297L)

//
// MessageId: PH_E_ERROR_DURING_SAVE_C_SOURCE_BENCHMARK_FULL_EXE_FILE
//
// MessageText:
//
// Error saving C source benchmark full exe file.
//
#define PH_E_ERROR_DURING_SAVE_C_SOURCE_BENCHMARK_FULL_EXE_FILE ((HRESULT)0xE0040298L)

//
// MessageId: PH_E_ERROR_DURING_SAVE_VCPROJECT_BENCHMARK_FULL_EXE_FILE
//
// MessageText:
//
// Error saving BenchmarkFullExe.vcxproj file.
//
#define PH_E_ERROR_DURING_SAVE_VCPROJECT_BENCHMARK_FULL_EXE_FILE ((HRESULT)0xE0040299L)

//
// MessageId: PH_E_ERROR_DURING_SAVE_C_SOURCE_BENCHMARK_INDEX_FILE
//
// MessageText:
//
// Error saving C source benchmark index file.
//
#define PH_E_ERROR_DURING_SAVE_C_SOURCE_BENCHMARK_INDEX_FILE ((HRESULT)0xE004029AL)

//
// MessageId: PH_E_ERROR_DURING_SAVE_C_SOURCE_BENCHMARK_INDEX_EXE_FILE
//
// MessageText:
//
// Error saving C source benchmark index exe file.
//
#define PH_E_ERROR_DURING_SAVE_C_SOURCE_BENCHMARK_INDEX_EXE_FILE ((HRESULT)0xE004029BL)

//
// MessageId: PH_E_ERROR_DURING_SAVE_VCPROJECT_BENCHMARK_INDEX_EXE_FILE
//
// MessageText:
//
// Error saving BenchmarkIndexExe.vcxproj file.
//
#define PH_E_ERROR_DURING_SAVE_VCPROJECT_BENCHMARK_INDEX_EXE_FILE ((HRESULT)0xE004029CL)

//
// MessageId: PH_E_ERROR_DURING_SAVE_BUILD_SOLUTION_BATCH_FILE
//
// MessageText:
//
// Error saving build solution batch file.
//
#define PH_E_ERROR_DURING_SAVE_BUILD_SOLUTION_BATCH_FILE ((HRESULT)0xE004029DL)

//
// MessageId: PH_E_ERROR_DURING_SAVE_C_HEADER_COMPILED_PERFECT_HASH_FILE
//
// MessageText:
//
// Error saving C header CompiledPerfectHash.h file.
//
#define PH_E_ERROR_DURING_SAVE_C_HEADER_COMPILED_PERFECT_HASH_FILE ((HRESULT)0xE004029EL)

//
// MessageId: PH_E_ERROR_DURING_SAVE_VCPROPS_COMPILED_PERFECT_HASH_FILE
//
// MessageText:
//
// Error saving CompiledPerfectHash.props file.
//
#define PH_E_ERROR_DURING_SAVE_VCPROPS_COMPILED_PERFECT_HASH_FILE ((HRESULT)0xE004029FL)

//
// MessageId: PH_E_CONTEXT_MAIN_WORK_LIST_EMPTY
//
// MessageText:
//
// SubmitThreadpoolWork() was called against the main work pool, but no corresponding work item was present on the main work list.
//
#define PH_E_CONTEXT_MAIN_WORK_LIST_EMPTY ((HRESULT)0xE0040300L)

//
// MessageId: PH_E_CONTEXT_FILE_WORK_LIST_EMPTY
//
// MessageText:
//
// SubmitThreadpoolWork() was called against the file work pool, but no corresponding work item was present on the file work list.
//
#define PH_E_CONTEXT_FILE_WORK_LIST_EMPTY ((HRESULT)0xE0040301L)

//
// MessageId: PH_E_GUARDED_LIST_EMPTY
//
// MessageText:
//
// The guarded list is empty.
//
#define PH_E_GUARDED_LIST_EMPTY          ((HRESULT)0xE0040302L)

//
// MessageId: PH_E_INVALID_CHUNK_OP
//
// MessageText:
//
// Invalid chunk op.
//
#define PH_E_INVALID_CHUNK_OP            ((HRESULT)0xE0040303L)

//
// MessageId: PH_E_INVALID_CHUNK_STRING
//
// MessageText:
//
// Invalid chunk string.
//
#define PH_E_INVALID_CHUNK_STRING        ((HRESULT)0xE0040304L)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_C_HEADER_STDAFX_FILE
//
// MessageText:
//
// Error preparing C header stdafx.h file.
//
#define PH_E_ERROR_DURING_PREPARE_C_HEADER_STDAFX_FILE ((HRESULT)0xE0040305L)

//
// MessageId: PH_E_ERROR_DURING_SAVE_C_HEADER_STDAFX_FILE
//
// MessageText:
//
// Error saving C header stdafx.h file.
//
#define PH_E_ERROR_DURING_SAVE_C_HEADER_STDAFX_FILE ((HRESULT)0xE0040306L)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_C_SOURCE_STDAFX_FILE
//
// MessageText:
//
// Error preparing C source stdafx.c file.
//
#define PH_E_ERROR_DURING_PREPARE_C_SOURCE_STDAFX_FILE ((HRESULT)0xE0040307L)

//
// MessageId: PH_E_ERROR_DURING_SAVE_C_SOURCE_STDAFX_FILE
//
// MessageText:
//
// Error saving C source stdafx.c file.
//
#define PH_E_ERROR_DURING_SAVE_C_SOURCE_STDAFX_FILE ((HRESULT)0xE0040308L)

//
// MessageId: PH_E_CONTEXT_FILE_ALREADY_PREPARED
//
// MessageText:
//
// Context file already prepared.
//
#define PH_E_CONTEXT_FILE_ALREADY_PREPARED ((HRESULT)0xE0040309L)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_VSSOLUTION_FILE
//
// MessageText:
//
// Error preparing VS Solution .sln file.
//
#define PH_E_ERROR_DURING_PREPARE_VSSOLUTION_FILE ((HRESULT)0xE004030AL)

//
// MessageId: PH_E_ERROR_DURING_SAVE_VSSOLUTION_FILE
//
// MessageText:
//
// Error saving VS Solution .sln file.
//
#define PH_E_ERROR_DURING_SAVE_VSSOLUTION_FILE ((HRESULT)0xE004030BL)

//
// MessageId: PH_E_INVALID_UUID_STRING
//
// MessageText:
//
// Invalid UUID string.
//
#define PH_E_INVALID_UUID_STRING         ((HRESULT)0xE004030CL)

//
// MessageId: PH_E_NO_INDEX_IMPL_C_STRING_FOUND
//
// MessageText:
//
// No Index() routine raw C string found for the current algorithm, hash function and masking type..
//
#define PH_E_NO_INDEX_IMPL_C_STRING_FOUND ((HRESULT)0xE004030DL)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_C_HEADER_SUPPORT_FILE
//
// MessageText:
//
// Error preparing C header support file.
//
#define PH_E_ERROR_DURING_PREPARE_C_HEADER_SUPPORT_FILE ((HRESULT)0xE004030EL)

//
// MessageId: PH_E_ERROR_DURING_SAVE_C_HEADER_SUPPORT_FILE
//
// MessageText:
//
// Error saving C header support file.
//
#define PH_E_ERROR_DURING_SAVE_C_HEADER_SUPPORT_FILE ((HRESULT)0xE004030FL)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_C_HEADER_COMPILED_PERFECT_HASH_MACRO_GLUE_FILE
//
// MessageText:
//
// Error preparing C header CompiledPerfectHashMacroGlue.h file.
//
#define PH_E_ERROR_DURING_PREPARE_C_HEADER_COMPILED_PERFECT_HASH_MACRO_GLUE_FILE ((HRESULT)0xE0040310L)

//
// MessageId: PH_E_ERROR_DURING_SAVE_C_HEADER_COMPILED_PERFECT_HASH_MACRO_GLUE_FILE
//
// MessageText:
//
// Error saving C header CompiledPerfectHashMacroGlue.h file.
//
#define PH_E_ERROR_DURING_SAVE_C_HEADER_COMPILED_PERFECT_HASH_MACRO_GLUE_FILE ((HRESULT)0xE0040311L)

//
// MessageId: PH_E_TABLE_COMPILATION_FAILED
//
// MessageText:
//
// Table compilation failed.
//
#define PH_E_TABLE_COMPILATION_FAILED    ((HRESULT)0xE0040312L)

//
// MessageId: PH_E_TABLE_NOT_CREATED
//
// MessageText:
//
// Table not created.
//
#define PH_E_TABLE_NOT_CREATED           ((HRESULT)0xE0040313L)

//
// MessageId: PH_E_TOO_MANY_EDGES
//
// MessageText:
//
// Too many edges.
//
#define PH_E_TOO_MANY_EDGES              ((HRESULT)0xE0040314L)

//
// MessageId: PH_E_TOO_MANY_VERTICES
//
// MessageText:
//
// Too many vertices.
//
#define PH_E_TOO_MANY_VERTICES           ((HRESULT)0xE0040315L)

//
// MessageId: PH_E_TOO_MANY_BITS_FOR_BITMAP
//
// MessageText:
//
// Too many bits for bitmap.
//
#define PH_E_TOO_MANY_BITS_FOR_BITMAP    ((HRESULT)0xE0040316L)

//
// MessageId: PH_E_TOO_MANY_TOTAL_EDGES
//
// MessageText:
//
// Too many total edges.
//
#define PH_E_TOO_MANY_TOTAL_EDGES        ((HRESULT)0xE0040317L)

//
// MessageId: PH_E_NUM_VERTICES_LESS_THAN_OR_EQUAL_NUM_EDGES
//
// MessageText:
//
// Number of vertices is less than or equal to the number of edges.
//
#define PH_E_NUM_VERTICES_LESS_THAN_OR_EQUAL_NUM_EDGES ((HRESULT)0xE0040318L)

//
// Disabled 8th Nov 2018: changed to PH_I_CREATE_TABLE_ROUTINE_RECEIVED_SHUTDOWN_EVENT.
// MessageId=0x319
// Severity=Fail
// Facility=ITF
// SymbolicName=PH_E_CREATE_TABLE_ROUTINE_RECEIVED_SHUTDOWN_EVENT
// Language=English
// Create table routine received shutdown event.
// .
//
// MessageId: PH_E_NO_MORE_SEEDS
//
// MessageText:
//
// No more seed data available.
//
#define PH_E_NO_MORE_SEEDS               ((HRESULT)0xE004031AL)

//
// MessageId: PH_E_GRAPH_NO_INFO_SET
//
// MessageText:
//
// No graph information has been set for graph.
//
#define PH_E_GRAPH_NO_INFO_SET           ((HRESULT)0xE004031BL)

//
// Disabled 30th Oct 2018: changed to PH_S_TABLE_RESIZE_IMMINENT.
// MessageId=0x31c
// Severity=Fail
// Facility=ITF
// SymbolicName=PH_E_TABLE_RESIZE_IMMINENT
// Language=English
// Table resize imminent.
// .
//
// Disabled 31st Dec 2018: obsolete after refactoring of table create params.  ;// MessageId=0x31d
// Severity=Fail
// Facility=ITF
// SymbolicName=PH_E_NUM_TABLE_CREATE_PARAMS_IS_ZERO_BUT_PARAMS_POINTER_NOT_NULL
// Language=English
// The number of table create parameters is zero, but table create parameters pointer is not null.
// .
//
// MessageId: PH_E_TABLE_CREATE_PARAMETER_VALIDATION_FAILED
//
// MessageText:
//
// Failed to validate one or more table create parameters.
//
#define PH_E_TABLE_CREATE_PARAMETER_VALIDATION_FAILED ((HRESULT)0xE004031EL)

//
// MessageId: PH_E_INVALID_TABLE_CREATE_PARAMETER_ID
//
// MessageText:
//
// Invalid table create parameter ID.
//
#define PH_E_INVALID_TABLE_CREATE_PARAMETER_ID ((HRESULT)0xE004031FL)

//
// MessageId: PH_E_INVALID_BEST_COVERAGE_TYPE_ID
//
// MessageText:
//
// Invalid best coverage type ID.
//
#define PH_E_INVALID_BEST_COVERAGE_TYPE_ID ((HRESULT)0xE0040320L)

//
// MessageId: PH_E_SPARE_GRAPH
//
// MessageText:
//
// Operation invalid on spare graph.
//
#define PH_E_SPARE_GRAPH                 ((HRESULT)0xE0040321L)

//
// MessageId: PH_E_GRAPH_INFO_ALREADY_LOADED
//
// MessageText:
//
// Graph information already loaded.
//
#define PH_E_GRAPH_INFO_ALREADY_LOADED   ((HRESULT)0xE0040322L)

//
// Disabled 8th Nov 2018: changed to PH_I_CREATE_TABLE_ROUTINE_FAILED_TO_FIND_SOLUTION
// MessageId=0x323
// Severity=Fail
// Facility=ITF
// SymbolicName=PH_E_CREATE_TABLE_ROUTINE_FAILED_TO_FIND_SOLUTION
// Language=English
// Create table routine failed to find perfect hash solution.
// .
//
// MessageId: PH_E_INVALID_TABLE_CREATE_PARAMETERS_FOR_FIND_BEST_GRAPH
//
// MessageText:
//
// Find best graph was requested but one or more mandatory table create parameters were missing or invalid.
//
#define PH_E_INVALID_TABLE_CREATE_PARAMETERS_FOR_FIND_BEST_GRAPH ((HRESULT)0xE0040324L)

//
// MessageId: PH_E_FILE_ALREADY_MAPPED
//
// MessageText:
//
// File is already mapped.
//
#define PH_E_FILE_ALREADY_MAPPED         ((HRESULT)0xE0040325L)

//
// MessageId: PH_E_FILE_ALREADY_UNMAPPED
//
// MessageText:
//
// File is already unmapped.
//
#define PH_E_FILE_ALREADY_UNMAPPED       ((HRESULT)0xE0040326L)

//
// MessageId: PH_E_FILE_NOT_MAPPED
//
// MessageText:
//
// File not mapped.
//
#define PH_E_FILE_NOT_MAPPED             ((HRESULT)0xE0040327L)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_TABLE_FILE
//
// MessageText:
//
// Error closing table file.
//
#define PH_E_ERROR_DURING_CLOSE_TABLE_FILE ((HRESULT)0xE0040328L)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_TABLE_INFO_STREAM
//
// MessageText:
//
// Error closing table info stream.
//
#define PH_E_ERROR_DURING_CLOSE_TABLE_INFO_STREAM ((HRESULT)0xE0040329L)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_C_HEADER_FILE
//
// MessageText:
//
// Error closing C header file.
//
#define PH_E_ERROR_DURING_CLOSE_C_HEADER_FILE ((HRESULT)0xE004032AL)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_C_SOURCE_FILE
//
// MessageText:
//
// Error closing C source file.
//
#define PH_E_ERROR_DURING_CLOSE_C_SOURCE_FILE ((HRESULT)0xE004032BL)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_C_HEADER_STDAFX_FILE
//
// MessageText:
//
// Error closing C header stdafx file.
//
#define PH_E_ERROR_DURING_CLOSE_C_HEADER_STDAFX_FILE ((HRESULT)0xE004032CL)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_C_SOURCE_STDAFX_FILE
//
// MessageText:
//
// Error closing C source stdafx file.
//
#define PH_E_ERROR_DURING_CLOSE_C_SOURCE_STDAFX_FILE ((HRESULT)0xE004032DL)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_C_SOURCE_KEYS_FILE
//
// MessageText:
//
// Error closing C source keys file.
//
#define PH_E_ERROR_DURING_CLOSE_C_SOURCE_KEYS_FILE ((HRESULT)0xE004032EL)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_C_SOURCE_TABLE_DATA_FILE
//
// MessageText:
//
// Error closing C source table data file.
//
#define PH_E_ERROR_DURING_CLOSE_C_SOURCE_TABLE_DATA_FILE ((HRESULT)0xE004032FL)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_C_HEADER_SUPPORT_FILE
//
// MessageText:
//
// Error closing C header support file.
//
#define PH_E_ERROR_DURING_CLOSE_C_HEADER_SUPPORT_FILE ((HRESULT)0xE0040330L)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_C_SOURCE_SUPPORT_FILE
//
// MessageText:
//
// Error closing C source support file.
//
#define PH_E_ERROR_DURING_CLOSE_C_SOURCE_SUPPORT_FILE ((HRESULT)0xE0040331L)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_C_SOURCE_TEST_FILE
//
// MessageText:
//
// Error closing C source test file.
//
#define PH_E_ERROR_DURING_CLOSE_C_SOURCE_TEST_FILE ((HRESULT)0xE0040332L)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_C_SOURCE_TEST_EXE_FILE
//
// MessageText:
//
// Error closing C source test exe file.
//
#define PH_E_ERROR_DURING_CLOSE_C_SOURCE_TEST_EXE_FILE ((HRESULT)0xE0040333L)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_C_SOURCE_BENCHMARK_FULL_FILE
//
// MessageText:
//
// Error closing C source benchmark full file.
//
#define PH_E_ERROR_DURING_CLOSE_C_SOURCE_BENCHMARK_FULL_FILE ((HRESULT)0xE0040334L)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_C_SOURCE_BENCHMARK_FULL_EXE_FILE
//
// MessageText:
//
// Error closing C source benchmark full exe file.
//
#define PH_E_ERROR_DURING_CLOSE_C_SOURCE_BENCHMARK_FULL_EXE_FILE ((HRESULT)0xE0040335L)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_C_SOURCE_BENCHMARK_INDEX_FILE
//
// MessageText:
//
// Error closing C source benchmark index file.
//
#define PH_E_ERROR_DURING_CLOSE_C_SOURCE_BENCHMARK_INDEX_FILE ((HRESULT)0xE0040336L)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_C_SOURCE_BENCHMARK_INDEX_EXE_FILE
//
// MessageText:
//
// Error closing C source benchmark index exe file.
//
#define PH_E_ERROR_DURING_CLOSE_C_SOURCE_BENCHMARK_INDEX_EXE_FILE ((HRESULT)0xE0040337L)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_VCPROJECT_DLL_FILE
//
// MessageText:
//
// Error closing VC project dll file.
//
#define PH_E_ERROR_DURING_CLOSE_VCPROJECT_DLL_FILE ((HRESULT)0xE0040338L)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_VCPROJECT_TEST_EXE_FILE
//
// MessageText:
//
// Error closing VC project test exe file.
//
#define PH_E_ERROR_DURING_CLOSE_VCPROJECT_TEST_EXE_FILE ((HRESULT)0xE0040339L)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_VCPROJECT_BENCHMARK_FULL_EXE_FILE
//
// MessageText:
//
// Error closing VC project benchmark full exe file.
//
#define PH_E_ERROR_DURING_CLOSE_VCPROJECT_BENCHMARK_FULL_EXE_FILE ((HRESULT)0xE004033AL)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_VCPROJECT_BENCHMARK_INDEX_EXE_FILE
//
// MessageText:
//
// Error closing VC project benchmark index exe file.
//
#define PH_E_ERROR_DURING_CLOSE_VCPROJECT_BENCHMARK_INDEX_EXE_FILE ((HRESULT)0xE004033BL)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_VSSOLUTION_FILE
//
// MessageText:
//
// Error closing VS solution file.
//
#define PH_E_ERROR_DURING_CLOSE_VSSOLUTION_FILE ((HRESULT)0xE004033CL)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_C_HEADER_COMPILED_PERFECT_HASH_FILE
//
// MessageText:
//
// Error closing C header compiled perfect hash file.
//
#define PH_E_ERROR_DURING_CLOSE_C_HEADER_COMPILED_PERFECT_HASH_FILE ((HRESULT)0xE004033DL)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_C_HEADER_COMPILED_PERFECT_HASH_MACRO_GLUE_FILE
//
// MessageText:
//
// Error closing C header compiled perfect hash macro glue file.
//
#define PH_E_ERROR_DURING_CLOSE_C_HEADER_COMPILED_PERFECT_HASH_MACRO_GLUE_FILE ((HRESULT)0xE004033EL)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_VCPROPS_COMPILED_PERFECT_HASH_FILE
//
// MessageText:
//
// Error closing vcprops compiled perfect hash file.
//
#define PH_E_ERROR_DURING_CLOSE_VCPROPS_COMPILED_PERFECT_HASH_FILE ((HRESULT)0xE004033FL)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_BUILD_SOLUTION_BATCH_FILE
//
// MessageText:
//
// Error closing build solution batch file.
//
#define PH_E_ERROR_DURING_CLOSE_BUILD_SOLUTION_BATCH_FILE ((HRESULT)0xE0040340L)

//
// MessageId: PH_E_INVALID_CONTEXT_BULK_CREATE_FLAGS
//
// MessageText:
//
// Invalid context bulk create flags.
//
#define PH_E_INVALID_CONTEXT_BULK_CREATE_FLAGS ((HRESULT)0xE0040341L)

//
// MessageId: PH_E_CONTEXT_BULK_CREATE_INVALID_NUM_ARGS
//
// MessageText:
//
// Invalid number of arguments for context bulk create.
//
#define PH_E_CONTEXT_BULK_CREATE_INVALID_NUM_ARGS ((HRESULT)0xE0040342L)

//
// MessageId: PH_E_KEYS_VERIFICATION_SKIPPED
//
// MessageText:
//
// Keys verification skipped.
//
#define PH_E_KEYS_VERIFICATION_SKIPPED   ((HRESULT)0xE0040343L)

//
// MessageId: PH_E_NO_KEYS_FOUND_IN_DIRECTORY
//
// MessageText:
//
// No keys found in directory.
//
#define PH_E_NO_KEYS_FOUND_IN_DIRECTORY  ((HRESULT)0xE0040344L)

//
// MessageId: PH_E_NOT_ALL_BYTES_WRITTEN
//
// MessageText:
//
// Not all bytes written.
//
#define PH_E_NOT_ALL_BYTES_WRITTEN       ((HRESULT)0xE0040345L)

//
// MessageId: PH_E_BULK_CREATE_CSV_HEADER_MISMATCH
//
// MessageText:
//
// Bulk create CSV header mismatch.
//
#define PH_E_BULK_CREATE_CSV_HEADER_MISMATCH ((HRESULT)0xE0040346L)

//
// MessageId: PH_E_INVALID_PATH_CREATE_FLAGS
//
// MessageText:
//
// Invalid path create flags.
//
#define PH_E_INVALID_PATH_CREATE_FLAGS   ((HRESULT)0xE0040347L)

//
// MessageId: PH_E_KEYS_REQUIRED_FOR_TABLE_TEST
//
// MessageText:
//
// Keys required for table test.
//
#define PH_E_KEYS_REQUIRED_FOR_TABLE_TEST ((HRESULT)0xE0040348L)

//
// MessageId: PH_E_CONTEXT_TABLE_CREATE_INVALID_NUM_ARGS
//
// MessageText:
//
// Invalid number of arguments for context table create.
//
#define PH_E_CONTEXT_TABLE_CREATE_INVALID_NUM_ARGS ((HRESULT)0xE0040349L)

//
// MessageId: PH_E_TABLE_CREATE_CSV_HEADER_MISMATCH
//
// MessageText:
//
// Table create CSV header mismatch.
//
#define PH_E_TABLE_CREATE_CSV_HEADER_MISMATCH ((HRESULT)0xE004034AL)

//
// MessageId: PH_E_INVALID_CONTEXT_TABLE_CREATE_FLAGS
//
// MessageText:
//
// Invalid context table create flags.
//
#define PH_E_INVALID_CONTEXT_TABLE_CREATE_FLAGS ((HRESULT)0xE004034BL)

//
// MessageId: PH_E_BEST_COVERAGE_TYPE_REQUIRES_KEYS_SUBSET
//
// MessageText:
//
// Best coverage type requires keys subset, but none was provided.
//
#define PH_E_BEST_COVERAGE_TYPE_REQUIRES_KEYS_SUBSET ((HRESULT)0xE004034CL)

//
// MessageId: PH_E_KEYS_SUBSET_NOT_SORTED
//
// MessageText:
//
// Keys subset not sorted.
//
#define PH_E_KEYS_SUBSET_NOT_SORTED      ((HRESULT)0xE004034DL)

//
// MessageId: PH_E_INVALID_KEYS_SUBSET
//
// MessageText:
//
// Invalid keys subset.
//
#define PH_E_INVALID_KEYS_SUBSET         ((HRESULT)0xE004034EL)

//
// MessageId: PH_E_NOT_SORTED
//
// MessageText:
//
// Not ordered.
//
#define PH_E_NOT_SORTED                  ((HRESULT)0xE004034FL)

//
// MessageId: PH_E_DUPLICATE_DETECTED
//
// MessageText:
//
// Duplicate detected.
//
#define PH_E_DUPLICATE_DETECTED          ((HRESULT)0xE0040350L)

//
// MessageId: PH_E_DUPLICATE_VALUE_DETECTED_IN_KEYS_SUBSET
//
// MessageText:
//
// Duplicate value detected in keys subset.
//
#define PH_E_DUPLICATE_VALUE_DETECTED_IN_KEYS_SUBSET ((HRESULT)0xE0040351L)

//
// MessageId: PH_E_CTRL_C_PRESSED
//
// MessageText:
//
// Ctrl-C pressed.
//
#define PH_E_CTRL_C_PRESSED              ((HRESULT)0xE0040352L)

//
// MessageId: PH_E_INVALID_MAIN_WORK_THREADPOOL_PRIORITY
//
// MessageText:
//
// Invalid main work threadpool priority.
//
#define PH_E_INVALID_MAIN_WORK_THREADPOOL_PRIORITY ((HRESULT)0xE0040353L)

//
// MessageId: PH_E_INVALID_FILE_WORK_THREADPOOL_PRIORITY
//
// MessageText:
//
// Invalid file work threadpool priority.
//
#define PH_E_INVALID_FILE_WORK_THREADPOOL_PRIORITY ((HRESULT)0xE0040354L)

//
// MessageId: PH_E_INVALID_ENUM_ID
//
// MessageText:
//
// Invalid perfect hash enum ID.
//
#define PH_E_INVALID_ENUM_ID             ((HRESULT)0xE0040355L)

//
// MessageId: PH_E_INVALID_ENUM_TYPE_NAME
//
// MessageText:
//
// Invalid perfect hash enum type name.
//
#define PH_E_INVALID_ENUM_TYPE_NAME      ((HRESULT)0xE0040356L)

//
// MessageId: PH_E_INVALID_CPU_ARCH_NAME
//
// MessageText:
//
// Invalid CPU architecture name.
//
#define PH_E_INVALID_CPU_ARCH_NAME       ((HRESULT)0xE0040357L)

//
// MessageId: PH_E_INVALID_INTERFACE_NAME
//
// MessageText:
//
// Invalid interface name.
//
#define PH_E_INVALID_INTERFACE_NAME      ((HRESULT)0xE0040358L)

//
// MessageId: PH_E_INVALID_ALGORITHM_NAME
//
// MessageText:
//
// Invalid algorithm name.
//
#define PH_E_INVALID_ALGORITHM_NAME      ((HRESULT)0xE0040359L)

//
// MessageId: PH_E_INVALID_HASH_FUNCTION_NAME
//
// MessageText:
//
// Invalid hash function name.
//
#define PH_E_INVALID_HASH_FUNCTION_NAME  ((HRESULT)0xE004035AL)

//
// MessageId: PH_E_INVALID_MASK_FUNCTION_NAME
//
// MessageText:
//
// Invalid mask function name.
//
#define PH_E_INVALID_MASK_FUNCTION_NAME  ((HRESULT)0xE004035BL)

//
// MessageId: PH_E_INVALID_BEST_COVERAGE_TYPE_NAME
//
// MessageText:
//
// Invalid best coverage type name.
//
#define PH_E_INVALID_BEST_COVERAGE_TYPE_NAME ((HRESULT)0xE004035CL)

//
// MessageId: PH_E_INVALID_TABLE_CREATE_PARAMETER_NAME
//
// MessageText:
//
// Invalid table create parameter name.
//
#define PH_E_INVALID_TABLE_CREATE_PARAMETER_NAME ((HRESULT)0xE004035DL)

//
// MessageId: PH_E_INVALID_COMMANDLINE_ARG
//
// MessageText:
//
// Invalid command line argument: %1!wZ!
//
#define PH_E_INVALID_COMMANDLINE_ARG     ((HRESULT)0xE004035EL)

//
// MessageId: PH_E_COMMANDLINE_ARG_MISSING_VALUE
//
// MessageText:
//
// Command line argument missing value: %1!wZ!
//
#define PH_E_COMMANDLINE_ARG_MISSING_VALUE ((HRESULT)0xE004035FL)

//
// MessageId: PH_E_INVALID_TABLE_CREATE_PARAMETERS
//
// MessageText:
//
// Invalid table create parameters.
//
#define PH_E_INVALID_TABLE_CREATE_PARAMETERS ((HRESULT)0xE0040360L)

//
// MessageId: PH_E_INVALID_SEEDS
//
// MessageText:
//
// Invalid seeds.
//
#define PH_E_INVALID_SEEDS               ((HRESULT)0xE0040361L)

//
// MessageId: PH_E_INVALID_USER_SEEDS_ELEMENT_SIZE
//
// MessageText:
//
// Invalid user seed element size.
//
#define PH_E_INVALID_USER_SEEDS_ELEMENT_SIZE ((HRESULT)0xE0040362L)

//
// MessageId: PH_E_INVALID_SEED_MASKS_STRUCTURE_SIZE
//
// MessageText:
//
// Invalid seed masks structure size.
//
#define PH_E_INVALID_SEED_MASKS_STRUCTURE_SIZE ((HRESULT)0xE0040363L)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_C_SOURCE_DOWNSIZED_KEYS_FILE
//
// MessageText:
//
// Error preparing C source downsized keys file.
//
#define PH_E_ERROR_DURING_PREPARE_C_SOURCE_DOWNSIZED_KEYS_FILE ((HRESULT)0xE0040364L)

//
// MessageId: PH_E_ERROR_DURING_SAVE_C_SOURCE_DOWNSIZED_KEYS_FILE
//
// MessageText:
//
// Error saving C source downsized keys file.
//
#define PH_E_ERROR_DURING_SAVE_C_SOURCE_DOWNSIZED_KEYS_FILE ((HRESULT)0xE0040365L)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_C_SOURCE_DOWNSIZED_KEYS_FILE
//
// MessageText:
//
// Error closing C source downsized keys file.
//
#define PH_E_ERROR_DURING_CLOSE_C_SOURCE_DOWNSIZED_KEYS_FILE ((HRESULT)0xE0040366L)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_C_HEADER_TYPES_FILE
//
// MessageText:
//
// Error preparing C source table values file.
//
#define PH_E_ERROR_DURING_PREPARE_C_HEADER_TYPES_FILE ((HRESULT)0xE0040367L)

//
// MessageId: PH_E_ERROR_DURING_SAVE_C_HEADER_TYPES_FILE
//
// MessageText:
//
// Error saving C source table values file.
//
#define PH_E_ERROR_DURING_SAVE_C_HEADER_TYPES_FILE ((HRESULT)0xE0040368L)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_C_HEADER_TYPES_FILE
//
// MessageText:
//
// Error closing C source table values file.
//
#define PH_E_ERROR_DURING_CLOSE_C_HEADER_TYPES_FILE ((HRESULT)0xE0040369L)

//
// MessageId: PH_E_INVALID_VALUE_SIZE_IN_BYTES_PARAMETER
//
// MessageText:
//
// Invalid value size in bytes parameter.
//
#define PH_E_INVALID_VALUE_SIZE_IN_BYTES_PARAMETER ((HRESULT)0xE004036AL)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_C_SOURCE_TABLE_VALUES_FILE
//
// MessageText:
//
// Error preparing C source table values file.
//
#define PH_E_ERROR_DURING_PREPARE_C_SOURCE_TABLE_VALUES_FILE ((HRESULT)0xE004036BL)

//
// MessageId: PH_E_ERROR_DURING_SAVE_C_SOURCE_TABLE_VALUES_FILE
//
// MessageText:
//
// Error saving C source table values file.
//
#define PH_E_ERROR_DURING_SAVE_C_SOURCE_TABLE_VALUES_FILE ((HRESULT)0xE004036CL)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_C_SOURCE_TABLE_VALUES_FILE
//
// MessageText:
//
// Error closing C source table values file.
//
#define PH_E_ERROR_DURING_CLOSE_C_SOURCE_TABLE_VALUES_FILE ((HRESULT)0xE004036DL)

//
// MessageId: PH_E_NO_PATH_EXTENSION_PRESENT
//
// MessageText:
//
// No path extension present.
//
#define PH_E_NO_PATH_EXTENSION_PRESENT   ((HRESULT)0xE004036EL)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_MAKEFILE_FILE
//
// MessageText:
//
// Error preparing Makefile file.
//
#define PH_E_ERROR_DURING_PREPARE_MAKEFILE_FILE ((HRESULT)0xE004036FL)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_MAKEFILE_MAIN_MK_FILE
//
// MessageText:
//
// Error preparing Makefile main.mk file.
//
#define PH_E_ERROR_DURING_PREPARE_MAKEFILE_MAIN_MK_FILE ((HRESULT)0xE0040370L)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_MAKEFILE_LIB_MK_FILE
//
// MessageText:
//
// Error preparing Makefile Lib.mk file.
//
#define PH_E_ERROR_DURING_PREPARE_MAKEFILE_LIB_MK_FILE ((HRESULT)0xE0040371L)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_MAKEFILE_SO_MK_FILE
//
// MessageText:
//
// Error preparing Makefile So.mk file.
//
#define PH_E_ERROR_DURING_PREPARE_MAKEFILE_SO_MK_FILE ((HRESULT)0xE0040372L)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_MAKEFILE_TEST_MK_FILE
//
// MessageText:
//
// Error preparing Makefile Test.mk file.
//
#define PH_E_ERROR_DURING_PREPARE_MAKEFILE_TEST_MK_FILE ((HRESULT)0xE0040373L)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_MAKEFILE_BENCHMARK_INDEX_MK_FILE
//
// MessageText:
//
// Error preparing Makefile BenchmarkIndex.mk file.
//
#define PH_E_ERROR_DURING_PREPARE_MAKEFILE_BENCHMARK_INDEX_MK_FILE ((HRESULT)0xE0040374L)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_MAKEFILE_BENCHMARK_FULL_MK_FILE
//
// MessageText:
//
// Error preparing Makefile BenchmarkFull.mk file.
//
#define PH_E_ERROR_DURING_PREPARE_MAKEFILE_BENCHMARK_FULL_MK_FILE ((HRESULT)0xE0040375L)

//
// MessageId: PH_E_ERROR_DURING_PREPARE_C_HEADER_NO_SAL2_FILE
//
// MessageText:
//
// Error preparing C header no_sal2.h file.
//
#define PH_E_ERROR_DURING_PREPARE_C_HEADER_NO_SAL2_FILE ((HRESULT)0xE0040376L)

//
// MessageId: PH_E_ERROR_DURING_SAVE_MAKEFILE_FILE
//
// MessageText:
//
// Error saving Makefile file.
//
#define PH_E_ERROR_DURING_SAVE_MAKEFILE_FILE ((HRESULT)0xE0040377L)

//
// MessageId: PH_E_ERROR_DURING_SAVE_MAKEFILE_MAIN_MK_FILE
//
// MessageText:
//
// Error saving Makefile main.mk file.
//
#define PH_E_ERROR_DURING_SAVE_MAKEFILE_MAIN_MK_FILE ((HRESULT)0xE0040378L)

//
// MessageId: PH_E_ERROR_DURING_SAVE_MAKEFILE_LIB_MK_FILE
//
// MessageText:
//
// Error saving Makefile Lib.mk file.
//
#define PH_E_ERROR_DURING_SAVE_MAKEFILE_LIB_MK_FILE ((HRESULT)0xE0040379L)

//
// MessageId: PH_E_ERROR_DURING_SAVE_MAKEFILE_SO_MK_FILE
//
// MessageText:
//
// Error saving Makefile So.mk file.
//
#define PH_E_ERROR_DURING_SAVE_MAKEFILE_SO_MK_FILE ((HRESULT)0xE004037AL)

//
// MessageId: PH_E_ERROR_DURING_SAVE_MAKEFILE_TEST_MK_FILE
//
// MessageText:
//
// Error saving Makefile Test.mk file.
//
#define PH_E_ERROR_DURING_SAVE_MAKEFILE_TEST_MK_FILE ((HRESULT)0xE004037BL)

//
// MessageId: PH_E_ERROR_DURING_SAVE_MAKEFILE_BENCHMARK_INDEX_MK_FILE
//
// MessageText:
//
// Error saving Makefile BenchmarkIndex.mk file.
//
#define PH_E_ERROR_DURING_SAVE_MAKEFILE_BENCHMARK_INDEX_MK_FILE ((HRESULT)0xE004037CL)

//
// MessageId: PH_E_ERROR_DURING_SAVE_MAKEFILE_BENCHMARK_FULL_MK_FILE
//
// MessageText:
//
// Error saving Makefile BenchmarkFull.mk file.
//
#define PH_E_ERROR_DURING_SAVE_MAKEFILE_BENCHMARK_FULL_MK_FILE ((HRESULT)0xE004037DL)

//
// MessageId: PH_E_ERROR_DURING_SAVE_C_HEADER_NO_SAL2_FILE
//
// MessageText:
//
// Error saving C header no_sal2.h file.
//
#define PH_E_ERROR_DURING_SAVE_C_HEADER_NO_SAL2_FILE ((HRESULT)0xE004037EL)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_MAKEFILE_FILE
//
// MessageText:
//
// Error closing Makefile file.
//
#define PH_E_ERROR_DURING_CLOSE_MAKEFILE_FILE ((HRESULT)0xE004037FL)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_MAKEFILE_MAIN_MK_FILE
//
// MessageText:
//
// Error closing Makefile main.mk file.
//
#define PH_E_ERROR_DURING_CLOSE_MAKEFILE_MAIN_MK_FILE ((HRESULT)0xE0040380L)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_MAKEFILE_LIB_MK_FILE
//
// MessageText:
//
// Error closing Makefile Lib.mk file.
//
#define PH_E_ERROR_DURING_CLOSE_MAKEFILE_LIB_MK_FILE ((HRESULT)0xE0040381L)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_MAKEFILE_SO_MK_FILE
//
// MessageText:
//
// Error closing Makefile So.mk file.
//
#define PH_E_ERROR_DURING_CLOSE_MAKEFILE_SO_MK_FILE ((HRESULT)0xE0040382L)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_MAKEFILE_TEST_MK_FILE
//
// MessageText:
//
// Error closing Makefile Test.mk file.
//
#define PH_E_ERROR_DURING_CLOSE_MAKEFILE_TEST_MK_FILE ((HRESULT)0xE0040383L)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_MAKEFILE_BENCHMARK_INDEX_MK_FILE
//
// MessageText:
//
// Error closing Makefile BenchmarkIndex.mk file.
//
#define PH_E_ERROR_DURING_CLOSE_MAKEFILE_BENCHMARK_INDEX_MK_FILE ((HRESULT)0xE0040384L)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_MAKEFILE_BENCHMARK_FULL_MK_FILE
//
// MessageText:
//
// Error closing Makefile BenchmarkFull.mk file.
//
#define PH_E_ERROR_DURING_CLOSE_MAKEFILE_BENCHMARK_FULL_MK_FILE ((HRESULT)0xE0040385L)

//
// MessageId: PH_E_ERROR_DURING_CLOSE_C_HEADER_NO_SAL2_FILE
//
// MessageText:
//
// Error closing C header no_sal2.h file.
//
#define PH_E_ERROR_DURING_CLOSE_C_HEADER_NO_SAL2_FILE ((HRESULT)0xE0040386L)

//
// MessageId: PH_E_INITIAL_RESIZES_EXCEEDS_MAX_RESIZES
//
// MessageText:
//
// Initial number of table resizes exceeds maximum table resizes limit.
//
#define PH_E_INITIAL_RESIZES_EXCEEDS_MAX_RESIZES ((HRESULT)0xE0040387L)

//
// MessageId: PH_E_INITIAL_RESIZES_NOT_SUPPORTED_FOR_MODULUS_MASKING
//
// MessageText:
//
// Initial number of table resizes not supported for modulus masking.
//
#define PH_E_INITIAL_RESIZES_NOT_SUPPORTED_FOR_MODULUS_MASKING ((HRESULT)0xE0040388L)

//
// MessageId: PH_E_INVALID_NUMBER_OF_SEED_MASK_COUNT_ELEMENTS
//
// MessageText:
//
// Invalid number of seed mask count elements.
//
#define PH_E_INVALID_NUMBER_OF_SEED_MASK_COUNT_ELEMENTS ((HRESULT)0xE0040389L)

//
// MessageId: PH_E_INVALID_NUMBER_OF_SEED_MASK_COUNTS
//
// MessageText:
//
// Invalid number of seed mask counts.
//
#define PH_E_INVALID_NUMBER_OF_SEED_MASK_COUNTS ((HRESULT)0xE004038AL)

//
// MessageId: PH_E_INVALID_SEED3_BYTE1_MASK_COUNTS
//
// MessageText:
//
// Invalid counts for seed 3 byte 1 mask.
//
#define PH_E_INVALID_SEED3_BYTE1_MASK_COUNTS ((HRESULT)0xE004038BL)

//
// MessageId: PH_E_INVALID_SEED3_BYTE2_MASK_COUNTS
//
// MessageText:
//
// Invalid counts for seed 3 byte 2 mask.
//
#define PH_E_INVALID_SEED3_BYTE2_MASK_COUNTS ((HRESULT)0xE004038CL)

//
// MessageId: PH_E_SEED_MASK_COUNT_TOTAL_IS_ZERO
//
// MessageText:
//
// Seed mask count total is zero.
//
#define PH_E_SEED_MASK_COUNT_TOTAL_IS_ZERO ((HRESULT)0xE004038DL)

//
// MessageId: PH_E_SEED_MASK_COUNT_TOTAL_EXCEEDS_MAX_ULONG
//
// MessageText:
//
// Seed mask count total exceeds 32 bits.
//
#define PH_E_SEED_MASK_COUNT_TOTAL_EXCEEDS_MAX_ULONG ((HRESULT)0xE004038EL)

//
// MessageId: PH_E_GRAPH_VERTEX_COLLISION_FAILURE
//
// MessageText:
//
// Graph vertex collision failure.
//
#define PH_E_GRAPH_VERTEX_COLLISION_FAILURE ((HRESULT)0xE004038FL)

//
// MessageId: PH_E_GRAPH_CYCLIC_FAILURE
//
// MessageText:
//
// Cyclic graph failure.
//
#define PH_E_GRAPH_CYCLIC_FAILURE        ((HRESULT)0xE0040390L)

//
// MessageId: PH_E_HASH_ALL_KEYS_FIRST_INCOMPAT_WITH_ORIG_SEEDED_HASH_ROUTINES
//
// MessageText:
//
// --HashAllKeysFirst is incompatible with --UseOriginalSeededHashRoutines.
//
#define PH_E_HASH_ALL_KEYS_FIRST_INCOMPAT_WITH_ORIG_SEEDED_HASH_ROUTINES ((HRESULT)0xE0040391L)

//
// MessageId: PH_E_VERTEX_PAIR_FLAGS_REQUIRE_HASH_ALL_KEYS_FIRST
//
// MessageText:
//
// ---HashAllKeysFirst is required when specifying --TryLargePagesForVertexPairs, --EnableWriteCombineForVertexPairs or --RemoveWriteCombineAfterSuccessfulHashKeys.
//
#define PH_E_VERTEX_PAIR_FLAGS_REQUIRE_HASH_ALL_KEYS_FIRST ((HRESULT)0xE0040392L)

//
// MessageId: PH_E_CANT_WRITE_COMBINE_VERTEX_PAIRS_WHEN_LARGE_PAGES
//
// MessageText:
//
// --EnableWriteCombineForVertexPairs conflicts with --TryLargePagesForVertexPairs (write-combining is not supported for memory backed by large pages).
//
#define PH_E_CANT_WRITE_COMBINE_VERTEX_PAIRS_WHEN_LARGE_PAGES ((HRESULT)0xE0040393L)

//
// MessageId: PH_E_REMOVE_WRITE_COMBINE_REQUIRES_ENABLE_WRITE_COMBINE
//
// MessageText:
//
// --RemoveWriteCombineAfterSuccessfulHashKeys requires --EnableWriteCombineForVertexPairs to be specified.
//
#define PH_E_REMOVE_WRITE_COMBINE_REQUIRES_ENABLE_WRITE_COMBINE ((HRESULT)0xE0040394L)

//
// MessageId: PH_E_NVCUDA_DLL_LOAD_SYMBOLS_FAILED
//
// MessageText:
//
// LoadSymbols failed for nvcuda.dll.
//
#define PH_E_NVCUDA_DLL_LOAD_SYMBOLS_FAILED ((HRESULT)0xE0040395L)

//
// MessageId: PH_E_NVCUDA_DLL_LOAD_LIBRARY_FAILED
//
// MessageText:
//
// LoadLibrary failed for nvcuda.dll.  Make sure it is in your PATH environment variable.
//
#define PH_E_NVCUDA_DLL_LOAD_LIBRARY_FAILED ((HRESULT)0xE0040396L)

//
// MessageId: PH_E_NVCUDA_DLL_LOAD_SYMBOLS_FAILED_TO_LOAD_ALL_SYMBOLS
//
// MessageText:
//
// Failed to load all expected symbols from nvcuda.dll.
//
#define PH_E_NVCUDA_DLL_LOAD_SYMBOLS_FAILED_TO_LOAD_ALL_SYMBOLS ((HRESULT)0xE0040397L)

//
// MessageId: PH_E_CUDA_DRIVER_API_CALL_FAILED
//
// MessageText:
//
// CUDA Driver API call failed.
//
#define PH_E_CUDA_DRIVER_API_CALL_FAILED ((HRESULT)0xE0040398L)

//
// MessageId: PH_E_CU_DEVICE_ORDINALS_NOT_SORTED
//
// MessageText:
//
// CuDeviceOrdinals not sorted.
//
#define PH_E_CU_DEVICE_ORDINALS_NOT_SORTED ((HRESULT)0xE0040399L)

//
// MessageId: PH_E_INVALID_CU_DEVICE_ORDINALS
//
// MessageText:
//
// Invalid CuDeviceOrdinals.
//
#define PH_E_INVALID_CU_DEVICE_ORDINALS  ((HRESULT)0xE004039AL)

//
// MessageId: PH_E_DUPLICATE_VALUE_DETECTED_IN_CU_DEVICE_ORDINALS
//
// MessageText:
//
// Duplicate value detected in CuDeviceOrdinals.
//
#define PH_E_DUPLICATE_VALUE_DETECTED_IN_CU_DEVICE_ORDINALS ((HRESULT)0xE004039BL)

//
// MessageId: PH_E_KEYS_ALREADY_COPIED_TO_A_DIFFERENT_CU_DEVICE
//
// MessageText:
//
// Keys were already copied to a different device.
//
#define PH_E_KEYS_ALREADY_COPIED_TO_A_DIFFERENT_CU_DEVICE ((HRESULT)0xE004039CL)

//
// MessageId: PH_E_FAILED_TO_GENERATE_RANDOM_BYTES
//
// MessageText:
//
// Failed to generate random bytes.
//
#define PH_E_FAILED_TO_GENERATE_RANDOM_BYTES ((HRESULT)0xE004039DL)

//
// MessageId: PH_E_CURAND_DLL_LOAD_SYMBOLS_FAILED
//
// MessageText:
//
// LoadSymbols failed for curand64_NM.dll.
//
#define PH_E_CURAND_DLL_LOAD_SYMBOLS_FAILED ((HRESULT)0xE004039EL)

//
// MessageId: PH_E_CURAND_DLL_LOAD_LIBRARY_FAILED
//
// MessageText:
//
// LoadLibrary failed for curand64_NM.dll.  Make sure it is in your PATH environment variable.
//
#define PH_E_CURAND_DLL_LOAD_LIBRARY_FAILED ((HRESULT)0xE004039FL)

//
// MessageId: PH_E_CURAND_DLL_LOAD_SYMBOLS_FAILED_TO_LOAD_ALL_SYMBOLS
//
// MessageText:
//
// Failed to load all expected symbols from curand64_NM.dll.
//
#define PH_E_CURAND_DLL_LOAD_SYMBOLS_FAILED_TO_LOAD_ALL_SYMBOLS ((HRESULT)0xE00403A0L)

//
// MessageId: PH_E_INVALID_SOLUTIONS_FOUND_RATIO
//
// MessageText:
//
// Invalid SolutionsFoundRatio; must be a double less than 1.0 and greater than 0.0.
//
#define PH_E_INVALID_SOLUTIONS_FOUND_RATIO ((HRESULT)0xE00403A1L)

//
// MessageId: PH_E_CU_CONCURRENCY_EXCEEDS_MAX_CONCURRENCY
//
// MessageText:
//
// CuConcurrency exceeds MaximumConcurrency.
//
#define PH_E_CU_CONCURRENCY_EXCEEDS_MAX_CONCURRENCY ((HRESULT)0xE00403A2L)

//
// MessageId: PH_E_CUDA_OUT_OF_MEMORY
//
// MessageText:
//
// The CUDA device is out of memory.
//
#define PH_E_CUDA_OUT_OF_MEMORY          ((HRESULT)0xE00403A3L)

//
// MessageId: PH_E_INVALID_CU_DEVICES
//
// MessageText:
//
// Invalid --CuDevices.
//
#define PH_E_INVALID_CU_DEVICES          ((HRESULT)0xE00403A4L)

//
// MessageId: PH_E_INVALID_CU_DEVICES_BLOCKS_PER_GRID
//
// MessageText:
//
// Invalid --CuDevicesBlocksPerGrid.
//
#define PH_E_INVALID_CU_DEVICES_BLOCKS_PER_GRID ((HRESULT)0xE00403A5L)

//
// MessageId: PH_E_INVALID_CU_DEVICES_THREADS_PER_BLOCK
//
// MessageText:
//
// Invalid --CuDevicesThreadsPerBlock.
//
#define PH_E_INVALID_CU_DEVICES_THREADS_PER_BLOCK ((HRESULT)0xE00403A6L)

//
// MessageId: PH_E_INVALID_CU_DEVICES_KERNEL_RUNTIME_TARGET_IN_MILLISECONDS
//
// MessageText:
//
// Invalid --CuDevicesKernelRuntimeTargetInMilliseconds.
//
#define PH_E_INVALID_CU_DEVICES_KERNEL_RUNTIME_TARGET_IN_MILLISECONDS ((HRESULT)0xE00403A7L)

//
// MessageId: PH_E_INVALID_CU_CONCURRENCY
//
// MessageText:
//
// --CuConcurrency must be greater than 0 and less than or equal to maximum concurrency.
//
#define PH_E_INVALID_CU_CONCURRENCY      ((HRESULT)0xE00403A8L)

//
// MessageId: PH_E_CU_CONCURRENCY_MANDATORY_FOR_SELECTED_ALGORITHM
//
// MessageText:
//
// --CuConcurrency is mandatory for the selected algorithm.
//
#define PH_E_CU_CONCURRENCY_MANDATORY_FOR_SELECTED_ALGORITHM ((HRESULT)0xE00403A9L)

//
// MessageId: PH_E_CU_BLOCKS_PER_GRID_REQUIRES_CU_DEVICES
//
// MessageText:
//
// --CuDevicesBlocksPerGrid requires --CuDevices.
//
#define PH_E_CU_BLOCKS_PER_GRID_REQUIRES_CU_DEVICES ((HRESULT)0xE00403AAL)

//
// MessageId: PH_E_CU_THREADS_PER_BLOCK_REQUIRES_CU_DEVICES
//
// MessageText:
//
// --CuDevicesThreadsPerBlock requires --CuDevices.
//
#define PH_E_CU_THREADS_PER_BLOCK_REQUIRES_CU_DEVICES ((HRESULT)0xE00403ABL)

//
// MessageId: PH_E_CU_KERNEL_RUNTIME_TARGET_IN_MILLISECONDS_REQUIRES_CU_DEVICES
//
// MessageText:
//
// --CuDevicesKernelRuntimeTargetInMilliseconds requires --CuDevices.
//
#define PH_E_CU_KERNEL_RUNTIME_TARGET_IN_MILLISECONDS_REQUIRES_CU_DEVICES ((HRESULT)0xE00403ACL)

//
// MessageId: PH_E_CU_BLOCKS_PER_GRID_COUNT_MUST_MATCH_CU_DEVICES_COUNT
//
// MessageText:
//
// Number of values supplied to --CuDevicesBlocksPerGrid must match the number of values supplied to --CuDevices.
//
#define PH_E_CU_BLOCKS_PER_GRID_COUNT_MUST_MATCH_CU_DEVICES_COUNT ((HRESULT)0xE00403ADL)

//
// MessageId: PH_E_CU_THREADS_PER_BLOCK_COUNT_MUST_MATCH_CU_DEVICES_COUNT
//
// MessageText:
//
// Number of values supplied to --CuDevicesThreadsPerBlock must match the number of values supplied to --CuDevices.
//
#define PH_E_CU_THREADS_PER_BLOCK_COUNT_MUST_MATCH_CU_DEVICES_COUNT ((HRESULT)0xE00403AEL)

//
// MessageId: PH_E_CU_KERNEL_RUNTIME_TARGET_IN_MILLISECONDS_COUNT_MUST_MATCH_CU_DEVICES_COUNT
//
// MessageText:
//
// Number of values supplied to --CuDevicesKernelRuntimeTargetInMilliseconds must match the number of values supplied to --CuDevices.
//
#define PH_E_CU_KERNEL_RUNTIME_TARGET_IN_MILLISECONDS_COUNT_MUST_MATCH_CU_DEVICES_COUNT ((HRESULT)0xE00403AFL)

//
// MessageId: PH_E_CU_DEVICES_COUNT_MUST_MATCH_CU_CONCONCURRENCY
//
// MessageText:
//
// Number of values supplied to --CuDevices must match the value supplied by --CuConcurrency.
//
#define PH_E_CU_DEVICES_COUNT_MUST_MATCH_CU_CONCONCURRENCY ((HRESULT)0xE00403B0L)

//
// MessageId: PH_E_DUPLICATE_TABLE_CREATE_PARAMETER_DETECTED
//
// MessageText:
//
// Duplicate table create parameter detected.
//
#define PH_E_DUPLICATE_TABLE_CREATE_PARAMETER_DETECTED ((HRESULT)0xE00403B1L)

//
// MessageId: PH_E_INVALID_CU_NUMBER_OF_RANDOM_HOST_SEEDS
//
// MessageText:
//
// Invalid --CuNumberOfRandomHostSeeds.
//
#define PH_E_INVALID_CU_NUMBER_OF_RANDOM_HOST_SEEDS ((HRESULT)0xE00403B2L)

//
// MessageId: PH_E_CU_CUDA_DEV_RUNTIME_LIB_PATH_MANDATORY
//
// MessageText:
//
// --CuCudaDevRuntimeLibPath is mandatory for this algorithm.
//
#define PH_E_CU_CUDA_DEV_RUNTIME_LIB_PATH_MANDATORY ((HRESULT)0xE00403B3L)

