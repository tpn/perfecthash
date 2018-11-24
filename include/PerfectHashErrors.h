/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

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


////////////////////////////////////////////////////////////////////////////////
// PH_SEVERITY_INFORMATIONAL -- Usage Messages
////////////////////////////////////////////////////////////////////////////////

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
//     --TestAfterCreate
// 
//         Tests the perfect hash table after creation.
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
//                 --BestCoverageNumAttempts=N
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
//     --SkipGraphVerification
// 
//         When set, skips the internal graph verification check that ensures a
//         valid perfect hash solution has been found (i.e. with no collisions
//         across the entire key set).
// 
//     --IgnorePreviousTableSize
// 
//         When set, ignores any previously-recorded table sizes associated with
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
//     --AttemptsBeforeTableResize=N [default = 18]
// 
//         Specifies the number of attempts at solving the graph that will be made
//         before a table resize event will occur (assuming that resize events are
//         permitted, as per the following flag).
// 
//     --MaxNumberOfTableResizes=N [default = 5]
// 
//         Maximum number of table resizes that will be permitted before giving up.
// 
//     --BestCoverageNumAttempts=N
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
//             HighestNumberOfEmptyCacheLines
// 
//                 This predicate is based on the notion that a high number of
//                 empty cache lines implies a lower number of cache lines are
//                 required for the table data, which means better clustering of
//                 table data, which could result in fewer cache misses, which
//                 would yield greater performance.
// 
//             HighestNumberOfEmptyPages
//             HighestNumberOfEmptyLargePages
// 
//                 As above, but for pages and large pages, respectively.
// 
// Console Output Character Legend
// 
//  Char | Meaning
// 
//     .   Table created successfully.
// 
//     -   Table resize event occured.
// 
//     x   Failed to create a table.  The maximum number of attempts at trying to
//         solve the table at a given size was reached, and no more resize attempts
//         were possible (due to the maximum resize limit also being hit).
// 
// N.B. The following characters are related to memory issues.
// 
//     *   None of the worker threads were able to allocate sufficient memory to
//         attempt solving the graph.
// 
//     !   The system is out of memory.
// 
//     ^   The system is running low on memory (a low memory event is triggered
//         at about 90% RAM usage).  In certain situations we can detect this
//         situation prior to actually running out of memory; in these cases,
//         we abort the current table creation attempt (which will instantly
//         relieve system memory pressure).
// 
//     %   The graph was created successfully, however, we weren't able to allocate
//         enough memory for the table values array in order for the array to be
//         used after creation.  This can be avoided by omitting --TestAfterCreate.
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
//                 --BestCoverageNumAttempts=N
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
//     --SkipGraphVerification
// 
//         When set, skips the internal graph verification check that ensures a
//         valid perfect hash solution has been found (i.e. with no collisions
//         across the entire key set).
// 
//     --IgnorePreviousTableSize
// 
//         When set, ignores any previously-recorded table sizes associated with
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
//     --AttemptsBeforeTableResize=N [default = 18]
// 
//         Specifies the number of attempts at solving the graph that will be made
//         before a table resize event will occur (assuming that resize events are
//         permitted, as per the following flag).
// 
//     --MaxNumberOfTableResizes=N [default = 5]
// 
//         Maximum number of table resizes that will be permitted before giving up.
// 
//     --BestCoverageNumAttempts=N
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
//             HighestNumberOfEmptyCacheLines
// 
//                 This predicate is based on the notion that a high number of
//                 empty cache lines implies a lower number of cache lines are
//                 required for the table data, which means better clustering of
//                 table data, which could result in fewer cache misses, which
//                 would yield greater performance.
// 
//             HighestNumberOfEmptyPages
//             HighestNumberOfEmptyLargePages
// 
//                 As above, but for pages and large pages, respectively.
// 
//                 N.B. The following predicates must be used in conjunction with
//                      --KeysSubset.
// 
//             LowestNumberOfCacheLinesUsedByKeysSubset
// 
//                 This predicate is used to to search for solutions where the
//                 most frequent keys consume the lowest number of cache lines.
//                 It is useful in scenarios where the frequency of individual
//                 keys being looked up is heavily skewed toward a small subset.
//                 For example, if 90% of the lookups occur for 10% of the keys,
//                 the fewer cache lines occupied by those keys, the better.
// 
//                 This option needs to be combined with the --KeysSubset table
//                 create parameter.
// 
//             LowestNumberOfPagesUsedByKeysSubset
//             LowestNumberOfLargePagesUsedByKeysSubset
// 
//                 As above, but for pages and large pages, respectively.
// 
// 
//     --KeysSubset=N,N+1[,N+2,N+3,...] (e.g. --KeysSubset=10,50,123,600,670)
// 
//         Supplies a comma-separated list of keys in ascending key-value order.
//         Must contain two or more elements.
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
// MessageId: PH_E_ERROR_DURING_PREPARE_TABLE_STATS_TEXT_FILE
//
// MessageText:
//
// Error preparing table stats text file.
//
#define PH_E_ERROR_DURING_PREPARE_TABLE_STATS_TEXT_FILE ((HRESULT)0xE004028DL)

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
// MessageId: PH_E_ERROR_DURING_SAVE_TABLE_STATS_TEXT_FILE
//
// MessageText:
//
// Error saving table stats text file.
//
#define PH_E_ERROR_DURING_SAVE_TABLE_STATS_TEXT_FILE ((HRESULT)0xE004029DL)

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
// MessageId: PH_E_NUM_TABLE_CREATE_PARAMS_IS_ZERO_BUT_PARAMS_POINTER_NOT_NULL
//
// MessageText:
//
// The number of table create parameters is zero, but table create parameters pointer is not null.
//
#define PH_E_NUM_TABLE_CREATE_PARAMS_IS_ZERO_BUT_PARAMS_POINTER_NOT_NULL ((HRESULT)0xE004031DL)

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
// MessageId: PH_E_INVALID_BEST_COVERAGE_TYPE
//
// MessageText:
//
// Invalid best coverage type.
//
#define PH_E_INVALID_BEST_COVERAGE_TYPE  ((HRESULT)0xE0040320L)

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
// MessageId: PH_E_ERROR_DURING_CLOSE_TABLE_STATS_TEXT_FILE
//
// MessageText:
//
// Error closing table stats text file.
//
#define PH_E_ERROR_DURING_CLOSE_TABLE_STATS_TEXT_FILE ((HRESULT)0xE0040340L)

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

