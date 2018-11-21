# Perfect Hash

A library for creating perfect hash tables.

This README is a work-in-progress.

## Usage

```
C:\Users\Trent\Home\src\perfecthash\src>x64\Release\PerfectHashBulkCreate.exe
Invalid number of arguments for context bulk create.
Usage: PerfectHashBulkCreate.exe
    <KeysDirectory> <OutputDirectory>
    <Algorithm> <HashFunction> <MaskFunction>
    <MaximumConcurrency>
    [BulkCreateFlags] [KeysLoadFlags] [TableCreateFlags]
    [TableCompileFlags] [TableCreateParameters]

Bulk Create Flags:

    --TestAfterCreate

        Tests the perfect hash table after creation.

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

Table Create Flags:

    --FirstGraphWins | --FindBestGraph

        --FirstGraphWins [default]

            This is the default behavior.  When searching for solutions in
            parallel, the first graph to be found, "wins".  i.e. it's the
            solution that is subsequently written to disk.

        --FindBestGraph

            Requires the following two table create parameters to be present:

                --BestCoverageNumAttempts=N
                --BestCoverageType=<CoverageType>

            The table create routine will then run until it finds the number of
            best coverage attempts specified.  At that point, the graph that was
            found to be the "best" based on the coverage type predicate "wins",
            and is subsequently saved to disk.

            N.B. This option is significantly more CPU intensive than the
                 --FirstGraphWins mode, however, it is highly probable that it
                 will find a graph that is better (based on the predicate) than
                 when in first graph wins mode.

    --SkipGraphVerification

        When set, skips the internal graph verification check that ensures a
        valid perfect hash solution has been found (i.e. with no collisions
        across the entire key set).

    --IgnoreKeysTableSize

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

    --AttemptsBeforeTableResize=N [default = 18]

        Specifies the number of attempts at solving the graph that will be made
        before a table resize event will occur (assuming that resize events are
        permitted, as per the following flag).

    --MaxNumberOfTableResizes=N [default = 5]

        Maximum number of table resizes that will be permitted before giving up.

    --BestCoverageNumAttempts=N

        Where N is a positive integer, and represents the number of attempts
        that will be made at finding a "best" graph (based on the best coverage
        type requested below) before the create table routine returns.

    --BestCoverageType=<CoverageType>

        Indicates the predicate to determine what constitutes the best graph.

        Valid coverage types:

            HighestNumberOfEmptyCacheLines

                This predicate is based on the notion that a high number of
                empty cache lines implies a lower number of cache lines are
                required for the table data, which means better clustering of
                table data, which could result in fewer cache misses, which
                would yield greater performance.


```
