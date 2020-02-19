;/*++
;
;Copyright (c) 2018-2020 Trent Nelson <trent@trent.me>
;
;Module Name:
;
;    PerfectHashErrors.h
;
;Abstract:
;
;    This is the public header file for status codes used by the perfect hash
;    library.  It is automatically generated from the messages defined in the
;    file src/PerfectHash/PerfectHashErrors.mc by the helper batch script named
;    src/PerfectHash/build-message-tables.bat (which must be run whenever the
;    .mc file changes).
;
;--*/

MessageIdTypedef=HRESULT
SeverityNames=(Success=0x0:PH_SEVERITY_SUCCESS
               Informational=0x1:PH_SEVERITY_INFORMATIONAL
               Warning=0x2:PH_SEVERITY_WARNING
               Fail=0x3:PH_SEVERITY_FAIL)
FacilityNames=(ITF=0x4:PH_FACILITY_ITF)
LanguageNames=(English=0x409:English)

MessageId=0x001
Severity=Success
Facility=ITF
SymbolicName=PH_S_GRAPH_SOLVED
Language=English
Graph solved.
.

MessageId=0x002
Severity=Success
Facility=ITF
SymbolicName=PH_S_GRAPH_NOT_SOLVED
Language=English
Graph not solved.
.

MessageId=0x003
Severity=Success
Facility=ITF
SymbolicName=PH_S_CONTINUE_GRAPH_SOLVING
Language=English
Continue graph solving.
.

MessageId=0x004
Severity=Success
Facility=ITF
SymbolicName=PH_S_STOP_GRAPH_SOLVING
Language=English
Stop graph solving.
.

MessageId=0x005
Severity=Success
Facility=ITF
SymbolicName=PH_S_GRAPH_VERIFICATION_SKIPPED
Language=English
Graph verification skipped.
.

MessageId=0x006
Severity=Success
Facility=ITF
SymbolicName=PH_S_GRAPH_SOLVING_STOPPED
Language=English
Graph solving has been stopped.
.

MessageId=0x007
Severity=Success
Facility=ITF
SymbolicName=PH_S_TABLE_RESIZE_IMMINENT
Language=English
Table resize imminent.
.

MessageId=0x008
Severity=Success
Facility=ITF
SymbolicName=PH_S_USE_NEW_GRAPH_FOR_SOLVING
Language=English
Use new graph for solving.
.

MessageId=0x009
Severity=Success
Facility=ITF
SymbolicName=PH_S_NO_KEY_SIZE_EXTRACTED_FROM_FILENAME
Language=English
No key size extracted from file name.
.

;
;////////////////////////////////////////////////////////////////////////////////
;// PH_SEVERITY_INFORMATIONAL
;////////////////////////////////////////////////////////////////////////////////
;

MessageId=0x080
Severity=Informational
Facility=ITF
SymbolicName=PH_I_CREATE_TABLE_ROUTINE_RECEIVED_SHUTDOWN_EVENT
Language=English
Create table routine received shutdown event.
.

MessageId=0x081
Severity=Informational
Facility=ITF
SymbolicName=PH_I_CREATE_TABLE_ROUTINE_FAILED_TO_FIND_SOLUTION
Language=English
Create table routine failed to find perfect hash solution.
.

MessageId=0x082
Severity=Informational
Facility=ITF
SymbolicName=PH_I_MAXIMUM_NUMBER_OF_TABLE_RESIZE_EVENTS_REACHED
Language=English
The maximum number of table resize events was reached before a perfect hash table solution could be found.
.

MessageId=0x083
Severity=Informational
Facility=ITF
SymbolicName=PH_I_REQUESTED_NUMBER_OF_TABLE_ELEMENTS_TOO_LARGE
Language=English
The requested number of table elements was too large.
.

MessageId=0x084
Severity=Informational
Facility=ITF
SymbolicName=PH_I_FAILED_TO_ALLOCATE_MEMORY_FOR_ALL_GRAPHS
Language=English
Failed to allocate memory for all graphs.
.

MessageId=0x085
Severity=Informational
Facility=ITF
SymbolicName=PH_I_LOW_MEMORY
Language=English
The system is running low on free memory.
.

MessageId=0x086
Severity=Informational
Facility=ITF
SymbolicName=PH_I_OUT_OF_MEMORY
Language=English
The system is out of memory.
.

MessageId=0x087
Severity=Informational
Facility=ITF
SymbolicName=PH_I_TABLE_CREATED_BUT_VALUES_ARRAY_ALLOC_FAILED
Language=English
The table was created successfully, however, the values array could not be allocated.  The table cannot be used.
.

;
;////////////////////////////////////////////////////////////////////////////////
;// PH_SEVERITY_INFORMATIONAL -- Usage Messages
;////////////////////////////////////////////////////////////////////////////////
;

MessageId=0x100
Severity=Informational
Facility=ITF
SymbolicName=PH_MSG_PERFECT_HASH_ALGO_HASH_MASK_NAMES
Language=English
Algorithms:

   ID | Name
    1   Chm01

Hash Functions:

   ID | Name (Number of Seeds)
    2   Jenkins (2)
   12   Fnv (2)
   14   Crc32RotateX (3)
   15   Crc32RotateXY (3)
   16   Crc32RotateWXYZ (3)
   17   RotateMultiplyXorRotate (3)
   18   ShiftMultiplyXorShift (3)
   19   ShiftMultiplyXorShift2 (6)
   20   RotateMultiplyXorRotate2 (6)
   21   MultiplyRotateR (3)
   22   MultiplyRotateLR (3)
   23   MultiplyShiftR (3)
   24   MultiplyShiftLR (3)
   25   Multiply (2)
   26   MultiplyXor (4)

Mask Functions:

  ID | Name
   1   Modulus (does not work!)
   2   And

.

MessageId=0x101
Severity=Informational
Facility=ITF
SymbolicName=PH_MSG_PERFECT_HASH_BULK_CREATE_EXE_USAGE
Language=English
Usage: PerfectHashBulkCreate.exe
    <KeysDirectory> <OutputDirectory>
    <Algorithm> <HashFunction> <MaskFunction>
    <MaximumConcurrency>
    [BulkCreateFlags] [KeysLoadFlags] [TableCreateFlags]
    [TableCompileFlags] [TableCreateParameters]

Bulk Create Flags:

    --SkipTestAfterCreate

        Normally, after a table has been successfully created, it is tested.
        Setting this flag disables this behavior.

        N.B. This will also disable benchmarking, so no performance information
             will be present in the .csv output file.

    --Compile

        Compiles the table after creation.

        N.B. msbuild.exe must be on the PATH environment variable for this
             to work.  (The current error message if this isn't the case is
             quite cryptic.)

Keys Load Flags:

    --TryLargePagesForKeysData

        Tries to allocate the keys buffer using large pages.

    --SkipKeysVerification

        Skips the logic that enumerates all keys after loading and a) ensures
        they are sorted, and b) constructs a keys bitmap.  If you can be certain
        the keys are sorted, specifying this flag can provide a speedup when
        loading large key sets.

    --DisableImplicitKeyDownsizing

        When loading keys that are 64-bit (8 bytes), a bitmap is kept that
        tracks whether or not a given bit was seen across the entire key set.
        After enumerating the set, the number of zeros in the bitmap are
        counted; if this number is less than or equal to 32, it means that the
        entire key set can be compressed into 32-bit values with some parallel
        bit extraction logic (i.e. _pext_u64()).  As this has beneficial size
        and performance implications, when detected, the key load operation will
        implicitly heap-allocate another array and convert all the 64-bit keys
        into their unique 32-bit equivalent.  Specifying this flag will disable
        this behavior.

    --TryInferKeySizeFromKeysFilename

        The default key size is 32-bit (4 bytes).  When this flag is present,
        if the keys file name ends with "64.keys" (e.g. "foo64.keys"), the key
        size will be interpreted as 64-bit (8 bytes).  This flag takes
        precedence over the table create parameter --KeySizeInBytes.

Table Create Flags:

    --Silent

        Disables console printing of the dots, dashes and other characters used
        to (crudely) visualize the result of individual table create operations.

    --NoFileIo

        Disables writing of all files when a perfect hash solution has been
        found.  The only time you would use this flag from the console
        application is to observe the performance of table creation without
        performing any file I/O operations.

    --Paranoid

        Enables redundant checks in the routine that determines whether or not
        a generated graph is acyclic.

    --FirstGraphWins | --FindBestGraph

        --FirstGraphWins [default]

            This is the default behavior.  When searching for solutions in
            parallel, the first graph to be found, "wins".  i.e. it's the
            solution that is subsequently written to disk.

        --FindBestGraph

            Requires the following two table create parameters to be present:

                --BestCoverageAttempts=N
                --BestCoverageType=<CoverageType>

            The table create routine will then run until it finds the number of
            best coverage attempts specified.  At that point, the graph that was
            found to be the "best" based on the coverage type predicate "wins",
            and is subsequently saved to disk.

            N.B. This option is significantly more CPU intensive than the
                 --FirstGraphWins mode, however, it is highly probable that it
                 will find a graph that is better (based on the predicate) than
                 when in first graph wins mode.

    --SkipMemoryCoverageInFirstGraphWinsMode

        Skips calculating memory coverage information when in "first graph wins"
        mode.  This will result in the corresponding fields in the .csv output
        indicating 0.

    --SkipGraphVerification

        When present, skips the internal graph verification check that ensures
        a valid perfect hash solution has been found (i.e. with no collisions
        across the entire key set).

    --DisableCsvOutputFile

        When present, disables writing the .csv output file.  This is required
        when running multiple instances of the tool against the same output
        directory in parallel.

    --OmitCsvRowIfTableCreateFailed

        When present, omits writing a row in the .csv output file if table
        creation fails for a given keys file.  Ignored if --DisableCsvOutputFile
        is speficied.

    --OmitCsvRowIfTableCreateSucceeded

        When present, omits writing a row in the .csv output file if table
        creation succeeded for a given keys file.  Ignored if
        --DisableCsvOutputFile is specified.

    --IndexOnly

        When set, affects the generated C files by defining the C preprocessor
        macro CPH_INDEX_ONLY, which results in omitting the compiled perfect
        hash routines that deal with the underlying table values array (i.e.
        any routine other than Index(); e.g. Insert(), Lookup(), Delete() etc),
        as well as the array itself.  This results in a size reduction of the
        final compiled perfect hash binary.  Additionally, only the .dll and
        BenchmarkIndex projects will be built, as the BenchmarkFull and Test
        projects require access to a table values array.  This flag is intended
        to be used if you only need the Index() routine and will be managing the
        table values array independently.

    --UseRwsSectionForTableValues

        When set, tells the linker to use a shared read-write section for the
        table values array, e.g.: #pragma comment(linker,"/section:.cphval,rws")
        This will result in the table values array being accessible across
        multiple processes.  Thus, the array will persist as long as one process
        maintains an open section (mapping); i.e. keeps the .dll loaded.

    --UseNonTemporalAvx2Routines

        When set, uses implementations of RtlCopyPages and RtlFillPages that
        use non-temporal hints.

    --IgnorePreviousTableSize

        When set, ignores any previously-recorded table sizes associated with
        the keys file for the given algorithm, hash function and masking type.

        N.B. To forcibly delete all previously-recorded table sizes from all
             keys in a directory, the following PowerShell snippet can be used:

             PS C:\Temp\keys> Get-Item -Path *.keys -Stream *.TableSize | Remove-Item

    --IncludeNumberOfTableResizeEventsInOutputPath

        When set, incorporates the number of table resize events encountered
        whilst searching for a perfect hash solution into the final output
        names, e.g.:

            C:\Temp\output\KernelBase_2485_1_Chm01_Crc32Rotate_And\...
                                           ^
                                           Number of resize events.

    --IncludeNumberOfTableElementsInOutputPath

        When set, incorporates the number of table elements (i.e. the final
        table size) into the output path, e.g.:

            C:\Temp\output\KernelBase_2485_16384_Chm01_Crc32Rotate_And\...
                                           ^
                                           Number of table elements.

        N.B. These two flags can be combined, yielding a path as follows:

            C:\Temp\output\KernelBase_2485_1_16384_Chm01_Crc32Rotate_And\...

        N.B. It is important to understand how table resize events impact the
             behavior of this program if one or both of these flags are present.
             Using the example above, the initial path that will be used for
             the solution will be:

                C:\Temp\output\KernelBase_2485_0_8192_Chm01_Crc32Rotate_And\...

             After the maximum number of attempts are reached, a table resize
             event occurs; the new path component will be:

                C:\Temp\output\KernelBase_2485_1_16384_Chm01_Crc32Rotate_And\...

             However, the actual renaming of the directory is not done until
             a solution has been found and all files have been written.  If
             this program is being run repeatedly, then the target directory
             will already exist.  This complicates things, as, unlike files,
             we can't just replace an existing directory with a new one.

             There are two ways this could be handled: a) delete all the
             existing files under the target path, then delete the directory,
             then perform the rename, or b) move the target directory somewhere
             else first, preserving the existing contents, then proceed with
             the rename.

             This program takes the latter approach.  The existing directory
             will be moved to:

                C:\Temp\output\old\KernelBase_1_16384_Chm01_Crc32Rotate_And_2018-11-19-011023-512\...

             The timestamp appended to the directory name is derived from the
             existing directory's creation time, which should ensure uniqueness.
             (In the unlikely situation the target directory already exists in
             the old subdirectory, the whole operation is aborted and the table
             create routine returns a failure.)

             The point of mentioning all of this is the following: when one or
             both of these flags are routinely specified, the number of output
             files rooted in the output directory's 'old' subdirectory will grow
             rapidly, consuming a lot of disk space.  Thus, if the old files are
             not required, it is recommended to regularly delete them manually.

Table Compile Flags:

    N/A

Table Create Parameters:

    --ValueSizeInBytes=4|8

        Sets the size, in bytes, of the value element that will be stored in the
        compiled perfect hash table via Insert().  Defaults to 4 bytes (ULONG).

    --MainWorkThreadpoolPriority=<High|Normal|Low> [default: Normal]
    --FileWorkThreadpoolPriority=<High|Normal|Low> [default: Normal]

        Sets the main work (i.e. the CPU-intensive graph solving) threadpool
        priority, or the file work threadpool priority, to the given value.

    --AttemptsBeforeTableResize=N [default = 18]

        Specifies the number of attempts at solving the graph that will be made
        before a table resize event will occur (assuming that resize events are
        permitted, as per the following flag).

    --MaxNumberOfTableResizes=N [default = 5]

        Maximum number of table resizes that will be permitted before giving up.

    --BestCoverageAttempts=N

        Where N is a positive integer, and represents the number of attempts
        that will be made at finding a "best" graph (based on the best coverage
        type requested below) before the create table routine returns.

    --BestCoverageType=<CoverageType>

        Indicates the predicate to determine what constitutes the best graph.

        Valid coverage types:

            HighestNumberOfEmptyPages
            HighestNumberOfEmptyLargePages
            HighestNumberOfEmptyCacheLines
            HighestMaxGraphTraversalDepth
            HighestTotalGraphTraversals
            HighestMaxAssignedPerCacheLineCount

            LowestNumberOfEmptyPages
            LowestNumberOfEmptyLargePages
            LowestNumberOfEmptyCacheLines
            LowestMaxGraphTraversalDepth
            LowestTotalGraphTraversals
            LowestMaxAssignedPerCacheLineCount

        The following predicates must be used in conjunction with --KeysSubset:

            HighestMaxAssignedPerCacheLineCountForKeysSubset
            HighestNumberOfPagesUsedByKeysSubset
            HighestNumberOfLargePagesUsedByKeysSubset
            HighestNumberOfCacheLinesUsedByKeysSubset

            LowestMaxAssignedPerCacheLineCountForKeysSubset
            LowestNumberOfPagesUsedByKeysSubset
            LowestNumberOfLargePagesUsedByKeysSubset
            LowestNumberOfCacheLinesUsedByKeysSubset

Console Output Character Legend

 Char | Meaning

    .   Table created successfully.

    -   Table resize event occured.

    x   Failed to create a table.  The maximum number of attempts at trying to
        solve the table at a given size was reached, and no more resize attempts
        were possible (due to the maximum resize limit also being hit).

    F   Failed to create a table due to a target not being reached by a specific
        number of attempts or time duration.  Not yet implemented.

    *   None of the worker threads were able to allocate sufficient memory to
        attempt solving the graph.

    !   The system is out of memory.

    L   The system is running low on memory (a low memory event is triggered
        at about 90%% RAM usage).  In certain situations we can detect this
        situation prior to actually running out of memory; in these cases,
        we abort the current table creation attempt (which will instantly
        relieve system memory pressure).

    V   The graph was created successfully, however, we weren't able to allocate
        enough memory for the table values array in order for the array to be
        used after creation.  This can be avoided by omitting --TestAfterCreate.

    T   The requested number of table elements was too large (exceeded 32 bits).

    S   A shutdown event was received.  This shouldn't be seen unless externally
        signaling the named shutdown event associated with a context.

.

MessageId=0x102
Severity=Informational
Facility=ITF
SymbolicName=PH_MSG_PERFECT_HASH_SELF_TEST_EXE_USAGE
Language=English
Usage: PerfectHashSelfTest.exe
    <TestDataDirectory> <OutputDirectory>
    <Algorithm> <HashFunction> <MaskFunction>
    <MaximumConcurrency>

N.B. This utility has been deprecated in favor of PerfectHashBulkCreate.exe.

.

MessageId=0x103
Severity=Informational
Facility=ITF
SymbolicName=PH_MSG_PERFECT_HASH_CREATE_EXE_USAGE
Language=English
Usage: PerfectHashCreate.exe
    <KeysPath> <OutputDirectory>
    <Algorithm> <HashFunction> <MaskFunction>
    <MaximumConcurrency>
    [KeysLoadFlags] [TableCreateFlags]
    [TableCompileFlags] [TableCreateParameters]

Keys Load Flags:

    --TryLargePagesForKeysData

        Tries to allocate the keys buffer using large pages.

    --SkipKeysVerification

        Skips the logic that enumerates all keys after loading and a) ensures
        they are sorted, and b) constructs a keys bitmap.  If you can be certain
        the keys are sorted, specifying this flag can provide a speedup when
        loading large key sets.

    --DisableImplicitKeyDownsizing

        When loading keys that are 64-bit (8 bytes), a bitmap is kept that
        tracks whether or not a given bit was seen across the entire key set.
        After enumerating the set, the number of zeros in the bitmap are
        counted; if this number is less than or equal to 32, it means that the
        entire key set can be compressed into 32-bit values with some parallel
        bit extraction logic (i.e. _pext_u64()).  As this has beneficial size
        and performance implications, when detected, the key load operation will
        implicitly heap-allocate another array and convert all the 64-bit keys
        into their unique 32-bit equivalent.  Specifying this flag will disable
        this behavior.

    --TryInferKeySizeFromKeysFilename

        The default key size is 32-bit (4 bytes).  When this flag is present,
        if the keys file name ends with "64.keys" (e.g. "foo64.keys"), the key
        size will be interpreted as 64-bit (8 bytes).  This flag takes
        precedence over the table create parameter --KeySizeInBytes.

Table Create Flags:

    --SkipTestAfterCreate

        Normally, after a table has been successfully created, it is tested.
        Setting this flag disables this behavior.

        N.B. This will also disable benchmarking, so no performance information
             will be present in the .csv output file.

    --Silent

        Disables console printing of the dots, dashes and other characters used
        to (crudely) visualize the result of individual table create operations.

    --NoFileIo

        Disables writing of all files when a perfect hash solution has been
        found.  The only time you would use this flag from the console
        application is to observe the performance of table creation without
        performing any file I/O operations.

    --Paranoid

        Enables redundant checks in the routine that determines whether or not
        a generated graph is acyclic.

    --FirstGraphWins | --FindBestGraph

        --FirstGraphWins [default]

            This is the default behavior.  When searching for solutions in
            parallel, the first graph to be found, "wins".  i.e. it's the
            solution that is subsequently written to disk.

        --FindBestGraph

            Requires the following two table create parameters to be present:

                --BestCoverageAttempts=N
                --BestCoverageType=<CoverageType>

            The table create routine will then run until it finds the number of
            best coverage attempts specified.  At that point, the graph that was
            found to be the "best" based on the coverage type predicate "wins",
            and is subsequently saved to disk.

            N.B. This option is significantly more CPU intensive than the
                 --FirstGraphWins mode, however, it is highly probable that it
                 will find a graph that is better (based on the predicate) than
                 when in first graph wins mode.

    --SkipMemoryCoverageInFirstGraphWinsMode

        Skips calculating memory coverage information when in "first graph wins"
        mode.  This will result in the corresponding fields in the .csv output
        indicating 0.

    --SkipGraphVerification

        When present, skips the internal graph verification check that ensures
        a valid perfect hash solution has been found (i.e. with no collisions
        across the entire key set).

    --DisableCsvOutputFile

        When present, disables writing the .csv output file.  This is required
        when running multiple instances of the tool against the same output
        directory in parallel.

    --OmitCsvRowIfTableCreateFailed

        When present, omits writing a row in the .csv output file if table
        creation fails for a given keys file.  Ignored if --DisableCsvOutputFile
        is speficied.

    --OmitCsvRowIfTableCreateSucceeded

        When present, omits writing a row in the .csv output file if table
        creation succeeded for a given keys file.  Ignored if
        --DisableCsvOutputFile is specified.

    --IndexOnly

        When set, affects the generated C files by defining the C preprocessor
        macro CPH_INDEX_ONLY, which results in omitting the compiled perfect
        hash routines that deal with the underlying table values array (i.e.
        any routine other than Index(); e.g. Insert(), Lookup(), Delete() etc),
        as well as the array itself.  This results in a size reduction of the
        final compiled perfect hash binary.  Additionally, only the .dll and
        BenchmarkIndex projects will be built, as the BenchmarkFull and Test
        projects require access to a table values array.  This flag is intended
        to be used if you only need the Index() routine and will be managing the
        table values array independently.

    --UseRwsSectionForTableValues

        When set, tells the linker to use a shared read-write section for the
        table values array, e.g.: #pragma comment(linker,"/section:.cphval,rws")
        This will result in the table values array being accessible across
        multiple processes.  Thus, the array will persist as long as one process
        maintains an open section (mapping); i.e. keeps the .dll loaded.

    --UseNonTemporalAvx2Routines

        When set, uses implementations of RtlCopyPages and RtlFillPages that
        use non-temporal hints.

    --IgnorePreviousTableSize

        When set, ignores any previously-recorded table sizes associated with
        the keys file for the given algorithm, hash function and masking type.

        N.B. To forcibly delete all previously-recorded table sizes from all
             keys in a directory, the following PowerShell snippet can be used:

             PS C:\Temp\keys> Get-Item -Path *.keys -Stream *.TableSize | Remove-Item

    --IncludeNumberOfTableResizeEventsInOutputPath

        When set, incorporates the number of table resize events encountered
        whilst searching for a perfect hash solution into the final output
        names, e.g.:

            C:\Temp\output\KernelBase_2485_1_Chm01_Crc32Rotate_And\...
                                           ^
                                           Number of resize events.

    --IncludeNumberOfTableElementsInOutputPath

        When set, incorporates the number of table elements (i.e. the final
        table size) into the output path, e.g.:

            C:\Temp\output\KernelBase_2485_16384_Chm01_Crc32Rotate_And\...
                                           ^
                                           Number of table elements.

        N.B. These two flags can be combined, yielding a path as follows:

            C:\Temp\output\KernelBase_2485_1_16384_Chm01_Crc32Rotate_And\...

        N.B. It is important to understand how table resize events impact the
             behavior of this program if one or both of these flags are present.
             Using the example above, the initial path that will be used for
             the solution will be:

                C:\Temp\output\KernelBase_2485_0_8192_Chm01_Crc32Rotate_And\...

             After the maximum number of attempts are reached, a table resize
             event occurs; the new path component will be:

                C:\Temp\output\KernelBase_2485_1_16384_Chm01_Crc32Rotate_And\...

             However, the actual renaming of the directory is not done until
             a solution has been found and all files have been written.  If
             this program is being run repeatedly, then the target directory
             will already exist.  This complicates things, as, unlike files,
             we can't just replace an existing directory with a new one.

             There are two ways this could be handled: a) delete all the
             existing files under the target path, then delete the directory,
             then perform the rename, or b) move the target directory somewhere
             else first, preserving the existing contents, then proceed with
             the rename.

             This program takes the latter approach.  The existing directory
             will be moved to:

                C:\Temp\output\old\KernelBase_1_16384_Chm01_Crc32Rotate_And_2018-11-19-011023-512\...

             The timestamp appended to the directory name is derived from the
             existing directory's creation time, which should ensure uniqueness.
             (In the unlikely situation the target directory already exists in
             the old subdirectory, the whole operation is aborted and the table
             create routine returns a failure.)

             The point of mentioning all of this is the following: when one or
             both of these flags are routinely specified, the number of output
             files rooted in the output directory's 'old' subdirectory will grow
             rapidly, consuming a lot of disk space.  Thus, if the old files are
             not required, it is recommended to regularly delete them manually.

Table Compile Flags:

    N/A

Table Create Parameters:

    --ValueSizeInBytes=4|8

        Sets the size, in bytes, of the value element that will be stored in the
        compiled perfect hash table via Insert().  Defaults to 4 bytes (ULONG).

    --MainWorkThreadpoolPriority=<High|Normal|Low> [default: Normal]
    --FileWorkThreadpoolPriority=<High|Normal|Low> [default: Normal]

        Sets the main work (i.e. the CPU-intensive graph solving) threadpool
        priority, or the file work threadpool priority, to the given value.

    --AttemptsBeforeTableResize=N [default = 18]

        Specifies the number of attempts at solving the graph that will be made
        before a table resize event will occur (assuming that resize events are
        permitted, as per the following flag).

    --MaxNumberOfTableResizes=N [default = 5]

        Maximum number of table resizes that will be permitted before giving up.

    --BestCoverageAttempts=N

        Where N is a positive integer, and represents the number of attempts
        that will be made at finding a "best" graph (based on the best coverage
        type requested below) before the create table routine returns.

    --BestCoverageType=<CoverageType>

        Indicates the predicate to determine what constitutes the best graph.

        Valid coverage types:

            HighestNumberOfEmptyPages
            HighestNumberOfEmptyLargePages
            HighestNumberOfEmptyCacheLines
            HighestMaxGraphTraversalDepth
            HighestTotalGraphTraversals
            HighestMaxAssignedPerCacheLineCount

            LowestNumberOfEmptyPages
            LowestNumberOfEmptyLargePages
            LowestNumberOfEmptyCacheLines
            LowestMaxGraphTraversalDepth
            LowestTotalGraphTraversals
            LowestMaxAssignedPerCacheLineCount

        The following predicates must be used in conjunction with --KeysSubset:

            HighestMaxAssignedPerCacheLineCountForKeysSubset
            HighestNumberOfPagesUsedByKeysSubset
            HighestNumberOfLargePagesUsedByKeysSubset
            HighestNumberOfCacheLinesUsedByKeysSubset

            LowestMaxAssignedPerCacheLineCountForKeysSubset
            LowestNumberOfPagesUsedByKeysSubset
            LowestNumberOfLargePagesUsedByKeysSubset
            LowestNumberOfCacheLinesUsedByKeysSubset

    --KeysSubset=N,N+1[,N+2,N+3,...] (e.g. --KeysSubset=10,50,123,600,670)

        Supplies a comma-separated list of keys in ascending key-value order.
        Must contain two or more elements.

.

;
;////////////////////////////////////////////////////////////////////////////////
;// PH_SEVERITY_FAIL
;////////////////////////////////////////////////////////////////////////////////
;

MessageId=0x201
Severity=Fail
Facility=ITF
SymbolicName=PH_E_CREATE_TABLE_ALREADY_IN_PROGRESS
Language=English
A table creation operation is in progress for this context.
.

MessageId=0x202
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TOO_MANY_KEYS
Language=English
Too many keys.
.

MessageId=0x203
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INFO_FILE_SMALLER_THAN_HEADER
Language=English
:Info file is smaller than smallest known table header size.
.

MessageId=0x204
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_MAGIC_VALUES
Language=English
Invalid magic values.
.

MessageId=0x205
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_INFO_HEADER_SIZE
Language=English
Invalid :Info header size.
.

MessageId=0x206
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NUM_KEYS_MISMATCH_BETWEEN_HEADER_AND_KEYS
Language=English
The number of keys reported in the keys file does not match the number of keys reported in the header.
.

MessageId=0x207
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_ALGORITHM_ID
Language=English
Invalid algorithm ID.
.

MessageId=0x208
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_HASH_FUNCTION_ID
Language=English
Invalid hash function ID.
.

MessageId=0x209
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_MASK_FUNCTION_ID
Language=English
Invalid mask function ID.
.

MessageId=0x20a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_HEADER_KEY_SIZE_TOO_LARGE
Language=English
The key size reported by the header is too large.
.

MessageId=0x20b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NUM_KEYS_IS_ZERO
Language=English
The number of keys is zero.
.

MessageId=0x20c
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NUM_TABLE_ELEMENTS_IS_ZERO
Language=English
The number of table elements is zero.
.

MessageId=0x20d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NUM_KEYS_EXCEEDS_NUM_TABLE_ELEMENTS
Language=English
The number of keys exceeds the number of table elements.
.

MessageId=0x20e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_EXPECTED_EOF_ACTUAL_EOF_MISMATCH
Language=English
The expected end of file does not match the actual end of file.
.

MessageId=0x20f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_KEYS_FILE_SIZE_NOT_MULTIPLE_OF_KEY_SIZE
Language=English
The keys file size is not a multiple of the key size.
.

MessageId=0x210
Severity=Fail
Facility=ITF
SymbolicName=PH_E_KEYS_NUM_SET_BITS_NUM_KEYS_MISMATCH
Language=English
The number of bits set for the keys bitmap does not match the number of keys.
.

MessageId=0x211
Severity=Fail
Facility=ITF
SymbolicName=PH_E_DUPLICATE_KEYS_DETECTED
Language=English
Duplicate keys detected.  Key files must not contain duplicate keys.
.

;//
;// Disabled 21st Nov 2018: deprecated in favor of E_OUTOFMEMORY and PH_I_OUT_OF_MEMORY.
;// MessageId=0x212
;// Severity=Fail
;// Facility=ITF
;// SymbolicName=PH_E_HEAP_CREATE_FAILED
;// Language=English
;// A call to HeapCreate() failed.
;// .

MessageId=0x213
Severity=Fail
Facility=ITF
SymbolicName=PH_E_RTL_LOAD_SYMBOLS_FROM_MULTIPLE_MODULES_FAILED
Language=English
A call to RtlLoadSymbolsFromMultipleModules() failed.
.

MessageId=0x214
Severity=Fail
Facility=ITF
SymbolicName=PH_E_CONTEXT_SELF_TEST_INVALID_NUM_ARGS
Language=English
Invalid number of arguments for context self-test.
.

MessageId=0x215
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_MAXIMUM_CONCURRENCY
Language=English
Invalid value for maximum concurrency.
.

MessageId=0x216
Severity=Fail
Facility=ITF
SymbolicName=PH_E_SET_MAXIMUM_CONCURRENCY_FAILED
Language=English
Setting the maximum concurrency of the context failed.
.

MessageId=0x217
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INITIALIZE_LARGE_PAGES_FAILED
Language=English
Internal error when attempting to initialize large pages.
.

MessageId=0x218
Severity=Fail
Facility=ITF
SymbolicName=PH_E_KEYS_NOT_SORTED
Language=English
The keys file supplied was not sorted.
.

MessageId=0x219
Severity=Fail
Facility=ITF
SymbolicName=PH_E_KEYS_NOT_LOADED
Language=English
A keys file has not been loaded yet.
.

MessageId=0x21a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_KEYS_LOAD_ALREADY_IN_PROGRESS
Language=English
A key loading operation is already in progress.
.

MessageId=0x21b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_KEYS_ALREADY_LOADED
Language=English
A set of keys has already been loaded.
.

MessageId=0x21c
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_KEYS_LOAD_FLAGS
Language=English
Invalid key load flags.
.

MessageId=0x21d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_KEY_SIZE
Language=English
Invalid key size.
.

MessageId=0x21e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TABLE_NOT_LOADED
Language=English
No table has been loaded yet.
.

MessageId=0x21f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TABLE_LOAD_ALREADY_IN_PROGRESS
Language=English
A table loading operation is already in progress.
.

;//
;// Removed 2018-10-01.
;//
;// MessageId=0x220
;// Severity=Fail
;// Facility=ITF
;// SymbolicName=PH_E_INVALID_CONTEXT_CREATE_TABLE_FLAGS
;// Language=English
;// Invalid context create table flags.
;// .
;//

MessageId=0x221
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_CONTEXT_SELF_TEST_FLAGS
Language=English
Invalid context self-test flags.
.

MessageId=0x222
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_TABLE_LOAD_FLAGS
Language=English
Invalid table load flags.
.

MessageId=0x223
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TABLE_LOCKED
Language=English
Table is locked.
.

MessageId=0x224
Severity=Fail
Facility=ITF
SymbolicName=PH_E_SYSTEM_CALL_FAILED
Language=English
System call failed.
.

MessageId=0x225
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TABLE_ALREADY_CREATED
Language=English
The table instance has already been created.
.

MessageId=0x226
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TABLE_ALREADY_LOADED
Language=English
The table instance has already been loaded.
.

MessageId=0x227
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_TABLE_COMPILE_FLAGS
Language=English
Invalid table compile flags.
.

MessageId=0x228
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TABLE_COMPILATION_NOT_AVAILABLE
Language=English
Table compilation is not available for the current combination of architecture, algorithm ID, hash function and masking type.
.

;//
;// Disabled 8th Nov 2018: changed to PH_I_MAXIMUM_NUMBER_OF_TABLE_RESIZE_EVENTS_REACHED
;// MessageId=0x229
;// Severity=Fail
;// Facility=ITF
;// SymbolicName=PH_E_MAXIMUM_NUMBER_OF_TABLE_RESIZE_EVENTS_REACHED
;// Language=English
;// The maximum number of table resize events was reached before a perfect hash table solution could be found.
;// .

;//
;// Disabled 8th Nov 2018: changed to PH_I_REQUESTED_NUMBER_OF_TABLE_ELEMENTS_TOO_LARGE
;// MessageId=0x22a
;// Severity=Fail
;// Facility=ITF
;// SymbolicName=PH_E_REQUESTED_NUMBER_OF_TABLE_ELEMENTS_TOO_LARGE
;// Language=English
;// The requested number of table elements was too large.
;// .

MessageId=0x22b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_TABLE_FILE
Language=English
Error preparing perfect hash table file.
.

MessageId=0x22c
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_TABLE_FILE
Language=English
Error saving perfect hash table file.
.

;//
;// A perfect hash table solution was found, however, it did not
;// pass internal validation checks (e.g. collisions were found
;// when attempting to independently verify that the perfect hash
;// function generated no collisions).
;//

MessageId=0x22d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TABLE_VERIFICATION_FAILED
Language=English
Table verification failed.
.

MessageId=0x22e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TABLE_CROSS_COMPILATION_NOT_AVAILABLE
Language=English
Table cross-compilation is not available.
.

MessageId=0x22f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_CPU_ARCH_ID
Language=English
The CPU architecture ID was invalid.
.

MessageId=0x230
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NOT_IMPLEMENTED
Language=English
Functionality not yet implemented.
.

MessageId=0x231
Severity=Fail
Facility=ITF
SymbolicName=PH_E_WORK_IN_PROGRESS
Language=English
Work in progress.
.

MessageId=0x232
Severity=Fail
Facility=ITF
SymbolicName=PH_E_KEYS_FILE_BASENAME_NOT_VALID_C_IDENTIFIER
Language=English
Keys file base name is not a valid C identifier.
.

MessageId=0x233
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_HEADER_FILE
Language=English
Error preparing C header file.
.

MessageId=0x234
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_HEADER_FILE
Language=English
Error saving C header file.
.

MessageId=0x235
Severity=Fail
Facility=ITF
SymbolicName=PH_E_UNREACHABLE_CODE
Language=English
Unreachable code reached.
.

MessageId=0x236
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVARIANT_CHECK_FAILED
Language=English
Internal invariant check failed.
.

MessageId=0x237
Severity=Fail
Facility=ITF
SymbolicName=PH_E_OVERFLOWED_HEADER_FILE_MAPPING_SIZE
Language=English
The calculated C header file size exceeded 4GB.
.

MessageId=0x238
Severity=Fail
Facility=ITF
SymbolicName=PH_E_CONTEXT_BASE_OUTPUT_DIRECTORY_NOT_SET
Language=English
Base output directory has not been set.
.

MessageId=0x239
Severity=Fail
Facility=ITF
SymbolicName=PH_E_CONTEXT_LOCKED
Language=English
The context is locked.
.

MessageId=0x23a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_CONTEXT_RESET_FAILED
Language=English
Failed to reset context.
.

MessageId=0x23b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_CONTEXT_SET_BASE_OUTPUT_DIRECTORY_FAILED
Language=English
Failed to set context output directory.
.

MessageId=0x23c
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NO_TABLE_CREATED_OR_LOADED
Language=English
The table has not been created or loaded.
.

MessageId=0x23d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TABLE_PATHS_ALREADY_INITIALIZED
Language=English
Paths have already been initialized for this table instance.
.

MessageId=0x23e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_TABLE_INFO_STREAM
Language=English
Error preparing :Info stream.
.

MessageId=0x23f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_TABLE_INFO_STREAM
Language=English
Error saving :Info stream.
.

MessageId=0x240
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_SOURCE_FILE
Language=English
Error saving C source file.
.

MessageId=0x241
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_SOURCE_KEYS_FILE
Language=English
Error saving C source keys file.
.

MessageId=0x242
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_SOURCE_TABLE_DATA_FILE
Language=English
Error saving C source table data file.
.

MessageId=0x243
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_CLOSED
Language=English
The file has already been closed.
.

MessageId=0x244
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_NOT_OPEN
Language=English
The file has not been opened yet, or has been closed.
.

MessageId=0x245
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_LOCKED
Language=English
The file is locked.
.

MessageId=0x246
Severity=Fail
Facility=ITF
SymbolicName=PH_E_KEYS_LOCKED
Language=English
The keys are locked.
.

MessageId=0x247
Severity=Fail
Facility=ITF
SymbolicName=PH_E_MAPPING_SIZE_LESS_THAN_OR_EQUAL_TO_CURRENT_SIZE
Language=English
Mapping size is less than or equal to current file size.
.

MessageId=0x248
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_READONLY
Language=English
The file is readonly.
.

MessageId=0x249
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_VIEW_CREATED
Language=English
A file view has already been created.
.

MessageId=0x24a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_VIEW_MAPPED
Language=English
A file view has already been mapped.
.

MessageId=0x24b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_MAPPING_SIZE_IS_ZERO
Language=English
The mapping size for the file is zero.
.

MessageId=0x24c
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_MAPPING_SIZE_NOT_SYSTEM_ALIGNED
Language=English
Mapping size is not aligned to the system allocation granularity.
.

MessageId=0x24d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_MAPPING_SIZE_NOT_LARGE_PAGE_ALIGNED
Language=English
Mapping size is not aligned to the large page granularity.
.

MessageId=0x24e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_ALREADY_OPEN
Language=English
An existing file has already been loaded or created for this file instance.
.

MessageId=0x24f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_FILE_LOAD_FLAGS
Language=English
Invalid file load flags.
.

MessageId=0x250
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_ALREADY_CLOSED
Language=English
File already closed.
.

MessageId=0x251
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_EMPTY
Language=English
The file is empty.
.

MessageId=0x252
Severity=Fail
Facility=ITF
SymbolicName=PH_E_PATH_PARTS_EXTRACTION_FAILED
Language=English
Failed to extract the path into parts.
.

MessageId=0x253
Severity=Fail
Facility=ITF
SymbolicName=PH_E_PATH_LOCKED
Language=English
Path is locked.
.

MessageId=0x254
Severity=Fail
Facility=ITF
SymbolicName=PH_E_EXISTING_PATH_LOCKED
Language=English
Existing path parameter is locked.
.

MessageId=0x255
Severity=Fail
Facility=ITF
SymbolicName=PH_E_PATH_ALREADY_SET
Language=English
A path has already been set for this instance.
.

MessageId=0x256
Severity=Fail
Facility=ITF
SymbolicName=PH_E_EXISTING_PATH_NO_PATH_SET
Language=English
Existing path parameter has not had a path set.
.

MessageId=0x257
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NO_PATH_SET
Language=English
No path set.
.

MessageId=0x258
Severity=Fail
Facility=ITF
SymbolicName=PH_E_STRING_BUFFER_OVERFLOW
Language=English
An internal string buffer has overflowed.
.

MessageId=0x259
Severity=Fail
Facility=ITF
SymbolicName=PH_E_SOURCE_PATH_LOCKED
Language=English
Source path parameter is locked.
.

MessageId=0x25a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_SOURCE_PATH_NO_PATH_SET
Language=English
Source path parameter had no path set.
.

MessageId=0x25b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_TABLE
Language=English
Invalid table.
.

MessageId=0x25c
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_TABLE_CREATE_FLAGS
Language=English
Invalid table create flags.
.

MessageId=0x25d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_SOURCE_FILE
Language=English
Error preparing C source file.
.

MessageId=0x25e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_SOURCE_KEYS_FILE
Language=English
Error preparing C source keys file.
.

MessageId=0x25f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_SOURCE_TABLE_DATA_FILE
Language=English
Error preparing C source table data file..
.

MessageId=0x260
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_NEVER_OPENED
Language=English
The file has never been opened.
.

MessageId=0x261
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_NO_RENAME_SCHEDULED
Language=English
The file has not had a rename operation scheduled.
.

MessageId=0x262
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_NOT_CLOSED
Language=English
The file has not yet been closed.
.

MessageId=0x263
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_FILE_WORK_ID
Language=English
Invalid file work ID.
.

MessageId=0x264
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_END_OF_FILE
Language=English
Invalid end of file (less than or equal to 0).
.

MessageId=0x265
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NEW_EOF_LESS_THAN_OR_EQUAL_TO_CURRENT_EOF
Language=English
New end-of-file is less than or equal to current end-of-file.
.

MessageId=0x266
Severity=Fail
Facility=ITF
SymbolicName=PH_E_RENAME_PATH_IS_SAME_AS_CURRENT_PATH
Language=English
The rename path equivalent to the existing path.
.

MessageId=0x267
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_FILE_CREATE_FLAGS
Language=English
Invalid file create flags.
.

MessageId=0x268
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_INTERFACE_ID
Language=English
Invalid interface ID.
.

MessageId=0x269
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NO_TLS_CONTEXT_SET
Language=English
PerfectHashTlsEnsureContext() was called but no TLS context was set.
.

MessageId=0x26a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NOT_GLOBAL_INTERFACE_ID
Language=English
The interface ID provided is not a global interface.
.

MessageId=0x26b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_VALUE_SIZE
Language=English
Invalid value size.
.

MessageId=0x26c
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_ALREADY_BEING_EXTENDED
Language=English
A file extension operation is already in progress.
.

MessageId=0x26d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_COLLISIONS_ENCOUNTERED_DURING_GRAPH_VERIFICATION
Language=English
Collisions encountered during graph verification.
.

MessageId=0x26e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NUM_ASSIGNMENTS_NOT_EQUAL_TO_NUM_KEYS_DURING_GRAPH_VERIFICATION
Language=English
The number of value assignments did not equal the number of keys during graph verification.
.

MessageId=0x26f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_NUMBER_OF_SEEDS
Language=English
Invalid number of seeds.
.

MessageId=0x270
Severity=Fail
Facility=ITF
SymbolicName=PH_E_CONTEXT_BASE_OUTPUT_DIRECTORY_ALREADY_SET
Language=English
Base output directory already set.
.

MessageId=0x271
Severity=Fail
Facility=ITF
SymbolicName=PH_E_DIRECTORY_ALREADY_CLOSED
Language=English
The directory has already been closed.
.

MessageId=0x272
Severity=Fail
Facility=ITF
SymbolicName=PH_E_DIRECTORY_NOT_SET
Language=English
The directory has not been opened yet, or has been closed.
.

MessageId=0x273
Severity=Fail
Facility=ITF
SymbolicName=PH_E_DIRECTORY_LOCKED
Language=English
The directory is locked.
.

MessageId=0x274
Severity=Fail
Facility=ITF
SymbolicName=PH_E_DIRECTORY_ALREADY_SET
Language=English
The directory is already set.
.

MessageId=0x275
Severity=Fail
Facility=ITF
SymbolicName=PH_E_DIRECTORY_DOES_NOT_EXIST
Language=English
Directory does not exist.
.

MessageId=0x276
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_DIRECTORY_OPEN_FLAGS
Language=English
Invalid directory open flags.
.

MessageId=0x277
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_DIRECTORY_CREATE_FLAGS
Language=English
Invalid directory create flags.
.

MessageId=0x278
Severity=Fail
Facility=ITF
SymbolicName=PH_E_DIRECTORY_NEVER_SET
Language=English
The directory was never set.
.

MessageId=0x279
Severity=Fail
Facility=ITF
SymbolicName=PH_E_DIRECTORY_READONLY
Language=English
The directory is readonly.
.

MessageId=0x27a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_DIRECTORY_NO_RENAME_SCHEDULED
Language=English
The directory has not had a rename operation scheduled.
.

MessageId=0x27b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_DIRECTORY_NOT_CLOSED
Language=English
Directory is not closed.
.

MessageId=0x27c
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_ALREADY_ADDED_TO_A_DIRECTORY
Language=English
The file has already been added to a directory.
.

MessageId=0x27d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_ADDED_TO_DIFFERENT_DIRECTORY
Language=English
The file was added to a different directory.
.

MessageId=0x27e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_DIRECTORY_RENAME_ALREADY_SCHEDULED
Language=English
Directory rename already scheduled.
.

MessageId=0x27f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_NOT_ADDED_TO_DIRECTORY
Language=English
The file has not been added to a directory.
.

MessageId=0x280
Severity=Fail
Facility=ITF
SymbolicName=PH_E_DIRECTORY_CLOSED
Language=English
Directory is closed.
.

MessageId=0x281
Severity=Fail
Facility=ITF
SymbolicName=PH_E_CREATE_RANDOM_OBJECT_NAMES_LENGTH_OF_NAME_TOO_SHORT
Language=English
LengthOfNameInChars parameter too short.
.

MessageId=0x282
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_VCPROJECT_DLL_FILE
Language=English
Error preparing Dll.vcxproj file.
.

MessageId=0x283
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_SOURCE_SUPPORT_FILE
Language=English
Error preparing C source support file.
.

MessageId=0x284
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_SOURCE_TEST_FILE
Language=English
Error preparing C source test file.
.

MessageId=0x285
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_SOURCE_TEST_EXE_FILE
Language=English
Error preparing C source test exe file.
.

MessageId=0x286
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_VCPROJECT_TEST_EXE_FILE
Language=English
Error preparing TestExe.vcxproj file.
.

MessageId=0x287
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_SOURCE_BENCHMARK_FULL_FILE
Language=English
Error preparing C source benchmark full file.
.

MessageId=0x288
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_SOURCE_BENCHMARK_FULL_EXE_FILE
Language=English
Error preparing C source benchmark full exe file.
.

MessageId=0x289
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_VCPROJECT_BENCHMARK_FULL_EXE_FILE
Language=English
Error preparing BenchmarkFullExe.vcxproj file.
.

MessageId=0x28a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_SOURCE_BENCHMARK_INDEX_FILE
Language=English
Error preparing C source benchmark index file.
.

MessageId=0x28b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_SOURCE_BENCHMARK_INDEX_EXE_FILE
Language=English
Error preparing C source benchmark index exe file.
.

MessageId=0x28c
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_VCPROJECT_BENCHMARK_INDEX_EXE_FILE
Language=English
Error preparing BenchmarkIndexExe.vcxproj file.
.

MessageId=0x28d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_BUILD_SOLUTION_BATCH_FILE
Language=English
Error preparing build solution batch file.
.

MessageId=0x28e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_HEADER_COMPILED_PERFECT_HASH_FILE
Language=English
Error preparing C header CompiledPerfectHash.h file.
.

MessageId=0x28f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_VCPROPS_COMPILED_PERFECT_HASH_FILE
Language=English
Error preparing CompiledPerfectHash.props file.
.

;//
;// Spare IDs: 0x290, 0x291.
;//

MessageId=0x292
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_VCPROJECT_DLL_FILE
Language=English
Error saving Dll.vcxproj file.
.

MessageId=0x293
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_SOURCE_SUPPORT_FILE
Language=English
Error saving C source support file.
.

MessageId=0x294
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_SOURCE_TEST_FILE
Language=English
Error saving C source test file.
.

MessageId=0x295
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_SOURCE_TEST_EXE_FILE
Language=English
Error saving C source test exe file.
.

MessageId=0x296
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_VCPROJECT_TEST_EXE_FILE
Language=English
Error saving TestExe.vcxproj file.
.

MessageId=0x297
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_SOURCE_BENCHMARK_FULL_FILE
Language=English
Error saving C source benchmark full file.
.

MessageId=0x298
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_SOURCE_BENCHMARK_FULL_EXE_FILE
Language=English
Error saving C source benchmark full exe file.
.

MessageId=0x299
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_VCPROJECT_BENCHMARK_FULL_EXE_FILE
Language=English
Error saving BenchmarkFullExe.vcxproj file.
.

MessageId=0x29a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_SOURCE_BENCHMARK_INDEX_FILE
Language=English
Error saving C source benchmark index file.
.

MessageId=0x29b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_SOURCE_BENCHMARK_INDEX_EXE_FILE
Language=English
Error saving C source benchmark index exe file.
.

MessageId=0x29c
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_VCPROJECT_BENCHMARK_INDEX_EXE_FILE
Language=English
Error saving BenchmarkIndexExe.vcxproj file.
.

MessageId=0x29d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_BUILD_SOLUTION_BATCH_FILE
Language=English
Error saving build solution batch file.
.

MessageId=0x29e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_HEADER_COMPILED_PERFECT_HASH_FILE
Language=English
Error saving C header CompiledPerfectHash.h file.
.

MessageId=0x29f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_VCPROPS_COMPILED_PERFECT_HASH_FILE
Language=English
Error saving CompiledPerfectHash.props file.
.

MessageId=0x300
Severity=Fail
Facility=ITF
SymbolicName=PH_E_CONTEXT_MAIN_WORK_LIST_EMPTY
Language=English
SubmitThreadpoolWork() was called against the main work pool, but no corresponding work item was present on the main work list.
.

MessageId=0x301
Severity=Fail
Facility=ITF
SymbolicName=PH_E_CONTEXT_FILE_WORK_LIST_EMPTY
Language=English
SubmitThreadpoolWork() was called against the file work pool, but no corresponding work item was present on the file work list.
.

MessageId=0x302
Severity=Fail
Facility=ITF
SymbolicName=PH_E_GUARDED_LIST_EMPTY
Language=English
The guarded list is empty.
.

MessageId=0x303
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_CHUNK_OP
Language=English
Invalid chunk op.
.

MessageId=0x304
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_CHUNK_STRING
Language=English
Invalid chunk string.
.

MessageId=0x305
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_HEADER_STDAFX_FILE
Language=English
Error preparing C header stdafx.h file.
.

MessageId=0x306
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_HEADER_STDAFX_FILE
Language=English
Error saving C header stdafx.h file.
.

MessageId=0x307
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_SOURCE_STDAFX_FILE
Language=English
Error preparing C source stdafx.c file.
.

MessageId=0x308
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_SOURCE_STDAFX_FILE
Language=English
Error saving C source stdafx.c file.
.

MessageId=0x309
Severity=Fail
Facility=ITF
SymbolicName=PH_E_CONTEXT_FILE_ALREADY_PREPARED
Language=English
Context file already prepared.
.

MessageId=0x30a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_VSSOLUTION_FILE
Language=English
Error preparing VS Solution .sln file.
.

MessageId=0x30b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_VSSOLUTION_FILE
Language=English
Error saving VS Solution .sln file.
.

MessageId=0x30c
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_UUID_STRING
Language=English
Invalid UUID string.
.

MessageId=0x30d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NO_INDEX_IMPL_C_STRING_FOUND
Language=English
No Index() routine raw C string found for the current algorithm, hash function and masking type..
.

MessageId=0x30e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_HEADER_SUPPORT_FILE
Language=English
Error preparing C header support file.
.

MessageId=0x30f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_HEADER_SUPPORT_FILE
Language=English
Error saving C header support file.
.

MessageId=0x310
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_HEADER_COMPILED_PERFECT_HASH_MACRO_GLUE_FILE
Language=English
Error preparing C header CompiledPerfectHashMacroGlue.h file.
.

MessageId=0x311
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_HEADER_COMPILED_PERFECT_HASH_MACRO_GLUE_FILE
Language=English
Error saving C header CompiledPerfectHashMacroGlue.h file.
.

MessageId=0x312
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TABLE_COMPILATION_FAILED
Language=English
Table compilation failed.
.

MessageId=0x313
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TABLE_NOT_CREATED
Language=English
Table not created.
.

MessageId=0x314
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TOO_MANY_EDGES
Language=English
Too many edges.
.

MessageId=0x315
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TOO_MANY_VERTICES
Language=English
Too many vertices.
.

MessageId=0x316
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TOO_MANY_BITS_FOR_BITMAP
Language=English
Too many bits for bitmap.
.

MessageId=0x317
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TOO_MANY_TOTAL_EDGES
Language=English
Too many total edges.
.

MessageId=0x318
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NUM_VERTICES_LESS_THAN_OR_EQUAL_NUM_EDGES
Language=English
Number of vertices is less than or equal to the number of edges.
.

;//
;// Disabled 8th Nov 2018: changed to PH_I_CREATE_TABLE_ROUTINE_RECEIVED_SHUTDOWN_EVENT.
;// MessageId=0x319
;// Severity=Fail
;// Facility=ITF
;// SymbolicName=PH_E_CREATE_TABLE_ROUTINE_RECEIVED_SHUTDOWN_EVENT
;// Language=English
;// Create table routine received shutdown event.
;// .

MessageId=0x31a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NO_MORE_SEEDS
Language=English
No more seed data available.
.

MessageId=0x31b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_GRAPH_NO_INFO_SET
Language=English
No graph information has been set for graph.
.

;//
;// Disabled 30th Oct 2018: changed to PH_S_TABLE_RESIZE_IMMINENT.
;// MessageId=0x31c
;// Severity=Fail
;// Facility=ITF
;// SymbolicName=PH_E_TABLE_RESIZE_IMMINENT
;// Language=English
;// Table resize imminent.
;// .

;//
;// Disabled 31st Dec 2018: obsolete after refactoring of table create params.  ;// MessageId=0x31d
;// Severity=Fail
;// Facility=ITF
;// SymbolicName=PH_E_NUM_TABLE_CREATE_PARAMS_IS_ZERO_BUT_PARAMS_POINTER_NOT_NULL
;// Language=English
;// The number of table create parameters is zero, but table create parameters pointer is not null.
;// .

MessageId=0x31e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TABLE_CREATE_PARAMETER_VALIDATION_FAILED
Language=English
Failed to validate one or more table create parameters.
.

MessageId=0x31f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_TABLE_CREATE_PARAMETER_ID
Language=English
Invalid table create parameter ID.
.

MessageId=0x320
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_BEST_COVERAGE_TYPE_ID
Language=English
Invalid best coverage type ID.
.

MessageId=0x321
Severity=Fail
Facility=ITF
SymbolicName=PH_E_SPARE_GRAPH
Language=English
Operation invalid on spare graph.
.

MessageId=0x322
Severity=Fail
Facility=ITF
SymbolicName=PH_E_GRAPH_INFO_ALREADY_LOADED
Language=English
Graph information already loaded.
.

;//
;// Disabled 8th Nov 2018: changed to PH_I_CREATE_TABLE_ROUTINE_FAILED_TO_FIND_SOLUTION
;// MessageId=0x323
;// Severity=Fail
;// Facility=ITF
;// SymbolicName=PH_E_CREATE_TABLE_ROUTINE_FAILED_TO_FIND_SOLUTION
;// Language=English
;// Create table routine failed to find perfect hash solution.
;// .

MessageId=0x324
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_TABLE_CREATE_PARAMETERS_FOR_FIND_BEST_GRAPH
Language=English
Find best graph was requested but one or more mandatory table create parameters were missing or invalid.
.

MessageId=0x325
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_ALREADY_MAPPED
Language=English
File is already mapped.
.

MessageId=0x326
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_ALREADY_UNMAPPED
Language=English
File is already unmapped.
.

MessageId=0x327
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_NOT_MAPPED
Language=English
File not mapped.
.

MessageId=0x328
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_TABLE_FILE
Language=English
Error closing table file.
.

MessageId=0x329
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_TABLE_INFO_STREAM
Language=English
Error closing table info stream.
.

MessageId=0x32a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_C_HEADER_FILE
Language=English
Error closing C header file.
.

MessageId=0x32b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_C_SOURCE_FILE
Language=English
Error closing C source file.
.

MessageId=0x32c
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_C_HEADER_STDAFX_FILE
Language=English
Error closing C header stdafx file.
.

MessageId=0x32d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_C_SOURCE_STDAFX_FILE
Language=English
Error closing C source stdafx file.
.

MessageId=0x32e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_C_SOURCE_KEYS_FILE
Language=English
Error closing C source keys file.
.

MessageId=0x32f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_C_SOURCE_TABLE_DATA_FILE
Language=English
Error closing C source table data file.
.

MessageId=0x330
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_C_HEADER_SUPPORT_FILE
Language=English
Error closing C header support file.
.

MessageId=0x331
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_C_SOURCE_SUPPORT_FILE
Language=English
Error closing C source support file.
.

MessageId=0x332
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_C_SOURCE_TEST_FILE
Language=English
Error closing C source test file.
.

MessageId=0x333
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_C_SOURCE_TEST_EXE_FILE
Language=English
Error closing C source test exe file.
.

MessageId=0x334
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_C_SOURCE_BENCHMARK_FULL_FILE
Language=English
Error closing C source benchmark full file.
.

MessageId=0x335
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_C_SOURCE_BENCHMARK_FULL_EXE_FILE
Language=English
Error closing C source benchmark full exe file.
.

MessageId=0x336
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_C_SOURCE_BENCHMARK_INDEX_FILE
Language=English
Error closing C source benchmark index file.
.

MessageId=0x337
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_C_SOURCE_BENCHMARK_INDEX_EXE_FILE
Language=English
Error closing C source benchmark index exe file.
.

MessageId=0x338
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_VCPROJECT_DLL_FILE
Language=English
Error closing VC project dll file.
.

MessageId=0x339
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_VCPROJECT_TEST_EXE_FILE
Language=English
Error closing VC project test exe file.
.

MessageId=0x33a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_VCPROJECT_BENCHMARK_FULL_EXE_FILE
Language=English
Error closing VC project benchmark full exe file.
.

MessageId=0x33b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_VCPROJECT_BENCHMARK_INDEX_EXE_FILE
Language=English
Error closing VC project benchmark index exe file.
.

MessageId=0x33c
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_VSSOLUTION_FILE
Language=English
Error closing VS solution file.
.

MessageId=0x33d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_C_HEADER_COMPILED_PERFECT_HASH_FILE
Language=English
Error closing C header compiled perfect hash file.
.

MessageId=0x33e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_C_HEADER_COMPILED_PERFECT_HASH_MACRO_GLUE_FILE
Language=English
Error closing C header compiled perfect hash macro glue file.
.

MessageId=0x33f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_VCPROPS_COMPILED_PERFECT_HASH_FILE
Language=English
Error closing vcprops compiled perfect hash file.
.

MessageId=0x340
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_BUILD_SOLUTION_BATCH_FILE
Language=English
Error closing build solution batch file.
.

MessageId=0x341
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_CONTEXT_BULK_CREATE_FLAGS
Language=English
Invalid context bulk create flags.
.

MessageId=0x342
Severity=Fail
Facility=ITF
SymbolicName=PH_E_CONTEXT_BULK_CREATE_INVALID_NUM_ARGS
Language=English
Invalid number of arguments for context bulk create.
.

MessageId=0x343
Severity=Fail
Facility=ITF
SymbolicName=PH_E_KEYS_VERIFICATION_SKIPPED
Language=English
Keys verification skipped.
.

MessageId=0x344
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NO_KEYS_FOUND_IN_DIRECTORY
Language=English
No keys found in directory.
.

MessageId=0x345
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NOT_ALL_BYTES_WRITTEN
Language=English
Not all bytes written.
.

MessageId=0x346
Severity=Fail
Facility=ITF
SymbolicName=PH_E_BULK_CREATE_CSV_HEADER_MISMATCH
Language=English
Bulk create CSV header mismatch.
.

MessageId=0x347
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_PATH_CREATE_FLAGS
Language=English
Invalid path create flags.
.

MessageId=0x348
Severity=Fail
Facility=ITF
SymbolicName=PH_E_KEYS_REQUIRED_FOR_TABLE_TEST
Language=English
Keys required for table test.
.

MessageId=0x349
Severity=Fail
Facility=ITF
SymbolicName=PH_E_CONTEXT_TABLE_CREATE_INVALID_NUM_ARGS
Language=English
Invalid number of arguments for context table create.
.

MessageId=0x34a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TABLE_CREATE_CSV_HEADER_MISMATCH
Language=English
Table create CSV header mismatch.
.

MessageId=0x34b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_CONTEXT_TABLE_CREATE_FLAGS
Language=English
Invalid context table create flags.
.

MessageId=0x34c
Severity=Fail
Facility=ITF
SymbolicName=PH_E_BEST_COVERAGE_TYPE_REQUIRES_KEYS_SUBSET
Language=English
Best coverage type requires keys subset, but none was provided.
.

MessageId=0x34d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_KEYS_SUBSET_NOT_SORTED
Language=English
Keys subset not sorted.
.

MessageId=0x34e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_KEYS_SUBSET
Language=English
Invalid keys subset.
.

MessageId=0x34f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NOT_SORTED
Language=English
Not ordered.
.

MessageId=0x350
Severity=Fail
Facility=ITF
SymbolicName=PH_E_DUPLICATE_DETECTED
Language=English
Duplicate detected.
.

MessageId=0x351
Severity=Fail
Facility=ITF
SymbolicName=PH_E_DUPLICATE_VALUE_DETECTED_IN_KEYS_SUBSET
Language=English
Duplicate value detected in keys subset.
.

MessageId=0x352
Severity=Fail
Facility=ITF
SymbolicName=PH_E_CTRL_C_PRESSED
Language=English
Ctrl-C pressed.
.

MessageId=0x353
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_MAIN_WORK_THREADPOOL_PRIORITY
Language=English
Invalid main work threadpool priority.
.

MessageId=0x354
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_FILE_WORK_THREADPOOL_PRIORITY
Language=English
Invalid file work threadpool priority.
.

MessageId=0x355
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_ENUM_ID
Language=English
Invalid perfect hash enum ID.
.

MessageId=0x356
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_ENUM_TYPE_NAME
Language=English
Invalid perfect hash enum type name.
.

MessageId=0x357
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_CPU_ARCH_NAME
Language=English
Invalid CPU architecture name.
.

MessageId=0x358
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_INTERFACE_NAME
Language=English
Invalid interface name.
.

MessageId=0x359
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_ALGORITHM_NAME
Language=English
Invalid algorithm name.
.

MessageId=0x35a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_HASH_FUNCTION_NAME
Language=English
Invalid hash function name.
.

MessageId=0x35b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_MASK_FUNCTION_NAME
Language=English
Invalid mask function name.
.

MessageId=0x35c
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_BEST_COVERAGE_TYPE_NAME
Language=English
Invalid best coverage type name.
.

MessageId=0x35d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_TABLE_CREATE_PARAMETER_NAME
Language=English
Invalid table create parameter name.
.

MessageId=0x35e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_COMMANDLINE_ARG
Language=English
Invalid command line argument: %1!wZ!
.

MessageId=0x35f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_COMMANDLINE_ARG_MISSING_VALUE
Language=English
Command line argument missing value: %1!wZ!
.

MessageId=0x360
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_TABLE_CREATE_PARAMETERS
Language=English
Invalid table create parameters.
.

MessageId=0x361
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_SEEDS
Language=English
Invalid seeds.
.

MessageId=0x362
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_USER_SEEDS_ELEMENT_SIZE
Language=English
Invalid user seed element size.
.

MessageId=0x363
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_SEED_MASKS_STRUCTURE_SIZE
Language=English
Invalid seed masks structure size.
.

MessageId=0x364
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_SOURCE_DOWNSIZED_KEYS_FILE
Language=English
Error preparing C source downsized keys file.
.

MessageId=0x365
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_SOURCE_DOWNSIZED_KEYS_FILE
Language=English
Error saving C source downsized keys file.
.

MessageId=0x366
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_C_SOURCE_DOWNSIZED_KEYS_FILE
Language=English
Error closing C source downsized keys file.
.

MessageId=0x367
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_HEADER_TYPES_FILE
Language=English
Error preparing C source table values file.
.

MessageId=0x368
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_HEADER_TYPES_FILE
Language=English
Error saving C source table values file.
.

MessageId=0x369
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_C_HEADER_TYPES_FILE
Language=English
Error closing C source table values file.
.

MessageId=0x36a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_VALUE_SIZE_IN_BYTES_PARAMETER
Language=English
Invalid value size in bytes parameter.
.

MessageId=0x36b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_SOURCE_TABLE_VALUES_FILE
Language=English
Error preparing C source table values file.
.

MessageId=0x36c
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_SOURCE_TABLE_VALUES_FILE
Language=English
Error saving C source table values file.
.

MessageId=0x36d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_C_SOURCE_TABLE_VALUES_FILE
Language=English
Error closing C source table values file.
.

MessageId=0x36e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NO_PATH_EXTENSION_PRESENT
Language=English
No path extension present.
.

MessageId=0x36f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_MAKEFILE_FILE
Language=English
Error preparing Makefile file.
.

MessageId=0x370
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_MAKEFILE_MAIN_MK_FILE
Language=English
Error preparing Makefile main.mk file.
.

MessageId=0x371
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_MAKEFILE_LIB_MK_FILE
Language=English
Error preparing Makefile Lib.mk file.
.

MessageId=0x372
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_MAKEFILE_SO_MK_FILE
Language=English
Error preparing Makefile So.mk file.
.

MessageId=0x373
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_MAKEFILE_TEST_MK_FILE
Language=English
Error preparing Makefile Test.mk file.
.

MessageId=0x374
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_MAKEFILE_BENCHMARK_INDEX_MK_FILE
Language=English
Error preparing Makefile BenchmarkIndex.mk file.
.

MessageId=0x375
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_MAKEFILE_BENCHMARK_FULL_MK_FILE
Language=English
Error preparing Makefile BenchmarkFull.mk file.
.

MessageId=0x376
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_HEADER_NO_SAL2_FILE
Language=English
Error preparing C header no_sal2.h file.
.

MessageId=0x377
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_MAKEFILE_FILE
Language=English
Error saving Makefile file.
.

MessageId=0x378
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_MAKEFILE_MAIN_MK_FILE
Language=English
Error saving Makefile main.mk file.
.

MessageId=0x379
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_MAKEFILE_LIB_MK_FILE
Language=English
Error saving Makefile Lib.mk file.
.

MessageId=0x37a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_MAKEFILE_SO_MK_FILE
Language=English
Error saving Makefile So.mk file.
.

MessageId=0x37b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_MAKEFILE_TEST_MK_FILE
Language=English
Error saving Makefile Test.mk file.
.

MessageId=0x37c
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_MAKEFILE_BENCHMARK_INDEX_MK_FILE
Language=English
Error saving Makefile BenchmarkIndex.mk file.
.

MessageId=0x37d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_MAKEFILE_BENCHMARK_FULL_MK_FILE
Language=English
Error saving Makefile BenchmarkFull.mk file.
.

MessageId=0x37e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_HEADER_NO_SAL2_FILE
Language=English
Error saving C header no_sal2.h file.
.

MessageId=0x37f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_MAKEFILE_FILE
Language=English
Error closing Makefile file.
.

MessageId=0x380
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_MAKEFILE_MAIN_MK_FILE
Language=English
Error closing Makefile main.mk file.
.

MessageId=0x381
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_MAKEFILE_LIB_MK_FILE
Language=English
Error closing Makefile Lib.mk file.
.

MessageId=0x382
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_MAKEFILE_SO_MK_FILE
Language=English
Error closing Makefile So.mk file.
.

MessageId=0x383
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_MAKEFILE_TEST_MK_FILE
Language=English
Error closing Makefile Test.mk file.
.

MessageId=0x384
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_MAKEFILE_BENCHMARK_INDEX_MK_FILE
Language=English
Error closing Makefile BenchmarkIndex.mk file.
.

MessageId=0x385
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_MAKEFILE_BENCHMARK_FULL_MK_FILE
Language=English
Error closing Makefile BenchmarkFull.mk file.
.

MessageId=0x386
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_CLOSE_C_HEADER_NO_SAL2_FILE
Language=English
Error closing C header no_sal2.h file.
.

