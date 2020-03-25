# Perfect Hash

<img src="https://ci.appveyor.com/api/projects/status/github/tpn/perfecthash?svg=true&retina=true" alt="Appveyor Badge">

A library for creating perfect hash tables.

This README is a work-in-progress.

## Usage
### PerfectHashBulkCreate.exe

```
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

    --ClampNumberOfEdges

        When present, clamps the number of edges to always be equal to the
        number of keys, rounded up to a power of two, regardless of the
        number of table resizes currently in effect.  Normally, when a table
        is resized, the number of vertices are doubled, and the number of
        edges are set to the number of vertices shifted right once (divided
        by two).  When this flag is set, the vertex doubling stays the same,
        however, the number of edges is always clamped to be equal to the
        number of keys rounded up to a power of two.  This is a research
        option used to evaluate the impact of the number of edges on the
        graph solving probability for a given key set.  Only applies to
        And masking (i.e. not modulus masking).

    --UsePreviousTableSize

        When set, uses any previously-recorded table sizes associated with
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

    --AttemptsBeforeTableResize=N [default = 4,294,967,295 ]

        Specifies the number of attempts at solving the graph that will be made
        before a table resize event will occur (assuming that resize events are
        permitted, as per the following flag).

    --MaxNumberOfTableResizes=N [default = 5]

        Maximum number of table resizes that will be permitted before giving up.

    --InitialNumberOfTableResizes=N [default = 0]

        Initial number of table resizes to simulate before attempting graph
        solving.  Each table resize doubles the number of vertices used to
        solve the graph, which lowers the keys-to-vertices ratio, which will
        improve graph solving probability.

        N.B. This parameter is only valid for And masking, not Modulus masking.

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

    +   Table resize event occured.

    x   Failed to create a table.  The maximum number of attempts at trying to
        solve the table at a given size was reached, and no more resize attempts
        were possible (due to the maximum resize limit also being hit).

    F   Failed to create a table due to a target not being reached by a specific
        number of attempts or time duration.  Not yet implemented.

    *   None of the worker threads were able to allocate sufficient memory to
        attempt solving the graph.

    !   The system is out of memory.

    L   The system is running low on memory (a low memory event is triggered
        at about 90% RAM usage).  In certain situations we can detect this
        situation prior to actually running out of memory; in these cases,
        we abort the current table creation attempt (which will instantly
        relieve system memory pressure).

    V   The graph was created successfully, however, we weren't able to allocate
        enough memory for the table values array in order for the array to be
        used after creation.  This can be avoided by omitting --TestAfterCreate.

    T   The requested number of table elements was too large (exceeded 32 bits).

    S   A shutdown event was received.  This shouldn't be seen unless externally
        signaling the named shutdown event associated with a context.

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
```

### PerfectHashCreate.exe

Creates a single perfect hash table.  Provides the ability to specify a subset
of keys as part of the "find best graph" predicates.

```
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

    --ClampNumberOfEdges

        When present, clamps the number of edges to always be equal to the
        number of keys, rounded up to a power of two, regardless of the
        number of table resizes currently in effect.  Normally, when a table
        is resized, the number of vertices are doubled, and the number of
        edges are set to the number of vertices shifted right once (divided
        by two).  When this flag is set, the vertex doubling stays the same,
        however, the number of edges is always clamped to be equal to the
        number of keys rounded up to a power of two.  This is a research
        option used to evaluate the impact of the number of edges on the
        graph solving probability for a given key set.  Only applies to
        And masking (i.e. not modulus masking).

    --UsePreviousTableSize

        When set, uses any previously-recorded table sizes associated with
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

    --AttemptsBeforeTableResize=N [default = 4,294,967,295 ]

        Specifies the number of attempts at solving the graph that will be made
        before a table resize event will occur (assuming that resize events are
        permitted, as per the following flag).

    --MaxNumberOfTableResizes=N [default = 5]

        Maximum number of table resizes that will be permitted before giving up.

    --InitialNumberOfTableResizes=N [default = 0]

        Initial number of table resizes to simulate before attempting graph
        solving.  Each table resize doubles the number of vertices used to
        solve the graph, which lowers the keys-to-vertices ratio, which will
        improve graph solving probability.

        N.B. This parameter is only valid for And masking, not Modulus masking.

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

```
