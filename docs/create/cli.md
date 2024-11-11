<!-- vim: set tw=76 ts=8 sw=4 et ai si: -->

# Create Flags

## SkipTestAfterCreate

Normally, after a table has been successfully created, it is tested. Setting
this flag disables this behavior.

**Note:** This will also disable benchmarking, so no performance information
will be present in the .csv output file.

## Compile

Compiles the table after creation.

**Note:** `msbuild.exe` must be on the `PATH` environment variable for this to
work.

**Note:** Windows only.

# Keys Load Flags

## TryLargePagesForKeysData

Tries to allocate the keys buffer using large pages.  May provide a
performance improvement for large key sets.  Gracefully falls back to
normal allocation if large pages are not available.

**Note:** Requires `SeLockMemoryPrivilege` to be enabled and for the
executable to be run as `Administrator`.  [More information about locking pages in memory](https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-10/security/threat-protection/security-policy-settings/lock-pages-in-memory).

**Note:** Windows only.

## SkipKeysVerification

Skips the logic that enumerates all keys after loading and a) ensures
they are sorted, and b) constructs a keys bitmap.  If you can be certain
the keys are sorted, specifying this flag may provide a small speedup when
loading large key sets.

**Note:** This will disable statistics about the keys (bitmap, longest
continguous bit run, etc.) in the output `.csv` file.

## DisableImplicitKeyDownsizing

When loading keys that are 64-bit (8 bytes), a bitmap is kept that
tracks whether or not a given bit was seen across the entire key set.
After enumerating the set, the number of zeros in the bitmap are
counted; if this number is less than or equal to 32, it means that the
entire key set can be compressed into 32-bit values with some parallel
bit extraction logic (i.e. `_pext_u64()`).  As this has beneficial size
and performance implications, when detected, the key load operation will
implicitly heap-allocate another array and convert all the 64-bit keys
into their unique 32-bit equivalent.  Specifying this flag will disable
this behavior.

## TryInferKeySizeFromKeysFilename

The default key size is 32-bit (4 bytes).  When this flag is present,
if the keys file name ends with `64.keys` (e.g. `foo64.keys`), the key
size will be interpreted as 64-bit (8 bytes).  This flag takes
precedence over the table create parameter `--KeySizeInBytes`.

# Table Create Flags

## Silent

Disables console printing of the dots, dashes and other characters used
to (crudely) visualize the result of individual table create operations,
and disable best graph information from being printed to the console.

**Note:** Implies `--Quiet.`

## Quiet

Disables printing best graph information to the console; only the dots
and dashes etc. will be printed.  This is the default for *Bulk Create*
mode, and is useful when you are running against a directory with a large
number of keys.  It's also handy for visually depicting how fast each table
is being solved in `--FirstGraphWins` mode.

**Note:** Incompatible with `--Silent`.

## NoFileIo

Disables writing of all files when a perfect hash solution has been
found.  The only time you would use this flag from the console
application is to observe the performance of table creation without
performing any file I/O operations.  This is useful for benchmarking
purposes, as writing some 20+ files to disk for each table created adds
a non-neglible amount of time to the overall process.

**Note:** This does not disable writing to the output `.csv` file, which will
still capture all information about the table creation process (including
best graph information, if applicable).  Thus, any solution found when this
flag is present can be re-run without `--NoFileIo` and explicitly providing
the winning table's seeds via the `--Seeds=...` parameter, which will write
all of the participating files to disk such that the table can be compiled
if desired.

## Paranoid

Enables redundant checks in the routine that determines whether or not
a generated graph is acyclic.  This flag was added early in the project's
development before we were confident in the correctness of the algorithm
used to determine if a graph is acyclic.  It is no longer necessary, but
is retained for historical reasons.

## FirstGraphWins

This is the default behavior.  When searching for solutions in parallel, the
first graph to be found, *"wins"*.  i.e. it's the solution that is
subsequently written to disk (assuming `--NoFileIo` is not specified).

**Note:** This is a dummy checkbox that is provided for the sake of
documenting the default graph solving mode.  The CLI does not recognize
the option `--FirstGraphWins`---it is implied by default in the absence of
the `--FindBestGraph` flag.

## FindBestGraph

Attempts to find the *best* graph for a given table based on a given
*coverage type* predicate and target number of best coverage attempts.
Requires the following two table create parameters to be present:
`--BestCoverageAttempts=N` and `--BestCoverageType=<CoverageType>`.

The table create routine will run until it finds the number of best
coverage attempts specified.  At that point, the graph that was found to be
the *"best"* based on the coverage type predicate "wins", and is subsequently
saved to disk.

See also `--TargetNumberOfSolutions` and `--FixedAttempts`.

## SkipMemoryCoverageInFirstGraphWinsMode

Skips calculating memory coverage information when in *first graph wins*
mode.  This will result in the corresponding fields in the `.csv` output
indicating 0.  Calculating memory coverage isn't particularly expensive,
so there's no harm in leaving enabled as default.  Thus, this flag is only
really useful during development or benchmarking.

## SkipGraphVerification

When present, skips the internal graph verification check that ensures
a valid perfect hash solution has been found (i.e. with no collisions
across the entire key set).  Graph verification isn't expensive, so there's
little benefit to disabling it.

## DisableCsvOutputFile

When present, disables writing the .csv output file.  This is required when
running multiple instances of the tool against the same output directory in
parallel.  (Although it's much better practice to run multiple instances in
parallel against different output directories.)

## OmitCsvRowIfTableCreateFailed

When present, omits writing a row in the `.csv` output file if table
creation fails for a given keys file.  Ignored if `--DisableCsvOutputFile`
is speficied.

## OmitCsvRowIfTableCreateSucceeded

When present, omits writing a row in the .csv output file if table
creation succeeded for a given keys file.  Ignored if `--DisableCsvOutputFile`
is specified.

## IndexOnly

When set, affects the generated C files by defining the C preprocessor
macro `CPH_INDEX_ONLY`, which results in omitting the compiled perfect
hash routines that deal with the underlying table values array (i.e.
any routine other than `Index()`; e.g. `Insert()`, `Lookup()`, `Delete()`
etc.), as well as the array itself.  This results in a size reduction of the
final compiled perfect hash binary.

Additionally, only the `.dll` (or `.so`) and `BenchmarkIndex` projects will
be built, as the `BenchmarkFull` and `Test` projects require access to a
table values array.  This flag is intended to be used if you only need the
`Index()` routine and will be managing the table values array independently.

## UseRwsSectionForTableValues

When set, tells the linker to use a shared read-write section for the
table values array, e.g.: `#pragma comment(linker,"/section:.cphval,rws")`.

This will result in the table values array being accessible across multiple
processes.  Thus, the array will persist as long as one process maintains an
open section (mapping); i.e. keeps the `.dll` loaded.

This is enabled by default.

**Note:** Windows only.

## DoNotUseRwsSectionForTableValues

Disables the use of a shared read-write section for the table values array.

**Note:** Windows only.

## UseNonTemporalAvx2Routines

When set, uses implementations of `RtlCopyPages` and `RtlFillPages` that
use non-temporal hints.

**Note:** Windows only.

## ClampNumberOfEdges

When present, clamps the number of edges to always be equal to the number of
keys, rounded up to a power of two, regardless of the number of table
resizes currently in effect.

Normally, when a table is resized, the number of vertices are doubled, and
the number of edges are set to the number of vertices shifted right once
(divided by two).  When this flag is set, the vertex doubling stays the
same, however, the number of edges is always clamped to be equal to the
number of keys rounded up to a power of two.

This is a research option used to evaluate the impact of the number of edges
on the graph solving probability for a given key set.

## UseOriginalSeededHashRoutines

When set, uses the original (slower) seeded hash routines (the ones that
return an `HRESULT` return code and write the hash value to an output
parameter)---as opposed to using the newer, faster, `"Ex"` version of the
hash routines.

**Note:** This flag is incompatible with `--HashAllKeysFirst`.

## HashAllKeysFirst

Hashes all of the keys first and stores the hash values (*vertices*) in a
*vertex pairs* array.  If a vertex collision is detected (when a key hashes
to two identical vertex values), the entire solution is discarded, and a new
solution is attempted using a different set of seeds.

If all keys are hashed without any vertex collisions, the graph solving
logic proceeds to add all the vertices to the graph and then attempts to
determine if the graph is acyclic (with a cyclical graph being the other way
a solution can fail).

This is the implicit default behavior.  The alternative is to hash a key
into two vertices and then immediately add them to the graph.  (This used to
be the default behavior.)

Hashing all keys up front has a few advantages:
- Optimized AVX2 or AVX512 routines can be used to hash all keys in parallel
  for supporting hash functions (e.g. `MultiplyShiftR` and
      `MultiplyShiftRX`).
- Adding two vertices to the graph immediately after hashing a key involves
  writing to memory.  If a vertex collision is encountered toward the tail
  end of the key set, the entire solution is discarded, and thus, all of the
  graph building operations that occurred up to that point were a waste of
  memory bandwidth and CPU cycles.
- It allows consistent benchmarking between CPU and GPU implementations, as
  the experimental GPU implementations implicitly hash all keys first, and
  do not support any other option.

**Note:** The page table type and page protection applied to the *vertex
pairs* array can be further refined by the following flags:
`--EnableWriteCombineForVertexPairs`,
`--RemoveWriteCombineAfterSuccessfulHashKeys`, and
`--TryLargePagesForVertexPairs`.  (The [*Advanced*](#show-advanced) and
[*Uncommon*](#show-uncommon) toggles need to be set to see these flags.)

**Note:** This flag is incompatible with `--UseOriginalSeededHashRoutines`.

**Note:** This flag is required for AVX2 and AVX512 hash routines to be
used if the hash function supports them (`MultiplyShiftR` and
`MultiplyShiftRX`).

## DoNotHashAllKeysFirst

Disables the implicit default `--HashAllKeysFirst` behavior.  This prevents
any of the optimized AVX2 or AVX512 hash routines from being used.  This
option may be more performant if and only if:
- You're using a hash function that doesn't have an AVX2 or AVX512 routine.
- You rarely encounter vertex collisions when hashing keys; you either have
  a high solutions found ratio, or failed solutions are typically due to
  cyclical graphs rather than vertex collisions.

## EnableWriteCombineForVertexPairs

When set, allocates the memory for the *vertex pairs* array with
*write-combine* page protection.  This was added as an experimental option
that in practice has been found to be detrimental to solving performance.

**Note:** Only applies when `--HashAllKeysFirst` is set.  Incompatible with
`--TryLargePagesForVertexPairs`.

**Note:** Windows only.

## RemoveWriteCombineAfterSuccessfulHashKeys

When set, automatically changes the page protection of the *vertex pairs*
array (after successful hashing of all keys without any vertex collisions)
from `PAGE_READWRITE|PAGE_WRITECOMBINE` to `PAGE_READONLY`.

In practice, this has been found to be detrimental to solving performance.

**Note:** Only applies when the flags `--EnableWriteCombineForVertexPairs`
and `--HashAllKeysFirst` is set.

**Note:** Windows only.

## TryLargePagesForVertexPairs

When set, tries to allocate the array for *vertex pairs* using large pages.

**Note:** Only applies when `--HashAllKeysFirst` is set.  Incompatible with
`--EnableWriteCombineForVertexPairs`.

**Note:** Requires `SeLockMemoryPrivilege` to be enabled and for the
executable to be run as `Administrator`.  [More information about locking pages in memory](https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-10/security/threat-protection/security-policy-settings/lock-pages-in-memory).

**Note:** Windows only.

## TryLargePagesForGraphEdgeAndVertexArrays

When set, tries to allocate the edge and vertex arrays used by graphs during
solving using large pages.

**Note:** Requires `SeLockMemoryPrivilege` to be enabled and for the
executable to be run as `Administrator`.  [More information about locking pages in memory](https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-10/security/threat-protection/security-policy-settings/lock-pages-in-memory).

**Note:** Windows only.

## TryLargePagesForGraphTableData

When set, tries to allocate the table data used by graphs during solving
using large pages.

**Note:** Requires `SeLockMemoryPrivilege` to be enabled and for the
executable to be run as `Administrator`.  [More information about locking pages in memory](https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-10/security/threat-protection/security-policy-settings/lock-pages-in-memory).

**Note:** Windows only.

## UsePreviousTableSize

When set, uses any previously-recorded table sizes associated with the keys
file for the given algorithm, hash function and masking type.  On Windows,
this information is stored in an alternate data stream named `:TableSize` on
the `.pht1` output file.

**Note:** To forcibly delete all previously-recorded table sizes from all
keys in a directory, the following PowerShell snippet can be used when on
Windows:

    PS C:\Temp\keys> Get-Item -Path *.keys -Stream *.TableSize | Remove-Item

## IncludeNumberOfTableResizeEventsInOutputPath

When set, incorporates the number of table resize events encountered
whilst searching for a perfect hash solution into the final output
names, e.g.:

    C:\Temp\output\KernelBase_2485_1_Chm01_Crc32Rotate_And\...
                                   ^
                                   Number of resize events.

## IncludeNumberOfTableElementsInOutputPath

When set, incorporates the number of table elements (i.e. the final
table size) into the output path, e.g.:

    C:\Temp\output\KernelBase_2485_16384_Chm01_Crc32Rotate_And\...
                                   ^
                                   Number of table elements.

**Note** This flag can be combined with the one above, yielding a path as
follows:

    C:\Temp\output\KernelBase_2485_1_16384_Chm01_Crc32Rotate_And\...

**Note**: It is important to understand how table resize events impact the
behavior of this program if one or both of these flags are present.  Using
the example above, the initial path that will be used for the solution will
be:

    C:\Temp\output\KernelBase_2485_0_8192_Chm01_Crc32Rotate_And\...

After the maximum number of attempts are reached, a table resize event
occurs; the new path component will be:

    C:\Temp\output\KernelBase_2485_1_16384_Chm01_Crc32Rotate_And\...

However, the actual renaming of the directory is not done until a solution
has been found and all files have been written.  If this program is being
run repeatedly, then the target directory will already exist.  This
complicates things, as, unlike files, we can't just replace an existing
directory with a new one.

There are two ways this could be handled: a) delete all the existing files
under the target path, then delete the directory, then perform the rename,
or b) move the target directory somewhere else first, preserving the
existing contents, then proceed with the rename.

This program takes the latter approach.  The existing directory will be
moved to:

    C:\Temp\output\old\KernelBase_1_16384_Chm01_Crc32Rotate_And_2018-11-19-011023-512\...

The timestamp appended to the directory name is derived from the existing
directory's creation time, which should ensure uniqueness.  (In the unlikely
situation the target directory already exists in the old subdirectory, the
whole operation is aborted and the table create routine returns a failure.)

The point of mentioning all of this is the following: when one or both of
these flags are routinely specified, the number of output files rooted in
the output directory's 'old' subdirectory will grow rapidly, consuming a lot
of disk space.  Thus, if the old files are not required, it is recommended
to regularly delete them manually.

## RngUseRandomStartSeed

Used in conjunction with `--Rng`.  If present, initializes the random
number generator with a random seed (obtained via the operating system).
If not present, the default seed `0x2019090319811025` will be used.

**Note:** If you're benchmarking performance, omit this flag, as starting
from the same default seed is required to get comparable runs.

See also: `--Rng`, `--RngSeed`, `--RngSubsequence`, and `--RngOffset`.

## TryUseAvx2HashFunction

By default, we try and use optimized AVX2 routines for hashing keys, if such
a routine is available for the given hash function.  Only `MultiplyShiftR`
and `MultiplyShiftRX` have AVX2 routines at present.  Will only apply on
CPUs that report support for the AVX2 instruction set.

**Note:** Only applies when `--HashAllKeysFirst` is set.

**Note:** Windows only.

## DoNotTryUseAvx2HashFunction

Disables the implicit default `--TryUseAvx2HashFunction` behavior.  You
would use this flag if you wanted to benchmark the performance of normal C
routines versus AVX2 routines for hashing keys.

**Note:** Windows only.

## TryUseAvx512HashFunction

When set, tries to use optimized AVX512 routines for hashing keys if such
routines are available for the given hash function, and the CPU indicates
support for AVX512.  Only `MultiplyShiftR` and `MultiplyShiftRX` have AVX512
routines at present.  Can be used in conjunction with
`--TryUseAvx2HashFunction`; if the CPU supports AVX512, the AVX512 routines
will be used, otherwise, the AVX2 routines will be used (if available).

**Note:** Only applies when `--HashAllKeysFirst` is set.

**Note:** Windows only.

## TryUseAvx2MemoryCoverageFunction

By default, we try and use optimized AVX2 routines for calculating memory
coverage, provided that the underlying CPU supports the AVX2 instruction
set.

**Note:** Windows only.

## DoNotTryUseAvx2MemoryCoverageFunction

Disables the implicit default `--TryUseAvx2MemoryCoverageFunction` behavior.
You would use this flag if you wanted to benchmark the performance of normal
C routines versus AVX2 routines for calculating memory coverage.

**Note:** Windows only.

## TryUseAvx512MemoryCoverageFunction

When set, tries to use optimized AVX512 routines for calculating memory
coverage, provided that the underlying CPU supports the AVX512 instruction
set.  Can be used in conjunction with `--TryUseAvx2MemoryCoverageFunction`;
if the CPU supports AVX512, the AVX512 routines will be used, otherwise, the
AVX2 routines will be used (if available).

**Note:** Windows only.

## IncludeKeysInCompiledDll

Includes the keys in the compiled `.dll` or `.so` file.  If you want to
benchmark a compiled perfect hash table `Index()` routine against a normal
binary search routine (i.e. `IndexBsearch()`), you'll need to supply this
flag to ensure the keys get built into the binary.

This flag is set by default.

## DoNotIncludeKeysInCompiledDll

This flag disables the implicit default `--IncludeKeysInCompiledDll`
behavior.

## DisableSavingCallbackTableValues

When set, does not attempt to save the runtime table values when running
with a `_penter`-hooked binary.

## DoNotTryUseHash16Impl

By default, if the following conditions exist, the library will
automatically switch to using the `USHORT`, 16-bit implementations of hash
functions and assigned table data seamlessly during graph solving:

- Algorithm is `Chm01`.
- `--GraphImpl` is `3`.
- Number of vertices is <= 65,534 (i.e. `MAX_USHORT-1`).

This provides significant performance improvements, which is why it's
the default.  To disable this behavior, set this flag.  This flag is
intended to be used during debugging and performance comparisons when
benchmarking&mdash;you shouldn't need to use it in normal use.

**Note:** This only affects the solving graph and table instances; the
compiled perfect hash table generated files will still use the appropriate
`USHORT` C-types if applicable (number of vertices less than or equal to
65,534).

## TryUsePredictedAttemptsToLimitMaxConcurrency

When present, the maximum concurrency used when solving will be the minimum
of the predicted attempts and the maximum concurrency indicated on the
command line.

See also: `--SolutionsFoundRatio`.

# Table Create Parameters

## GraphImpl

Selects the backend version of the graph assignment step.
[Version 1](https://github.com/tpn/perfecthash/blob/main/src/PerfectHash/GraphImpl1.c)
matches the original CHM algorithm,
[version 2](https://github.com/tpn/perfecthash/blob/main/src/PerfectHash/GraphImpl2.c)
is faster and was derived from NetBSD's `nbperf` module,
[version 3](https://github.com/tpn/perfecthash/blob/main/src/PerfectHash/GraphImpl3.c)
is even faster and was derived from additional improvements to NetBSD's `nbperf`
module in 2020.

Version 1 recursively walks the graph during the assignment step when it
attempts to delete vertices of degree 1.  This can result in deep call
stacks when many keys have a vertex shared between them.  Version 2 is very
similar to version 1, but has a slightly more efficient assignment step.

Version 3 uses an *XOR* incidence-list trick to avoid the need for recursive
graph traversal during the assignment step.  In almost all scenarios,
version 3 is the fastest solver, and is the default.

It is worth noting that each version will generate different `Assigned[]`
arrays, which means they will exhibit different memory coverage
characteristics, which means that the best graph for a given key set and
coverage type may differ between versions.

Additionally, some best coverage type predicates only work with version 1
and 2:

- `HighestMaxGraphTraversalDepth`
- `LowestMaxGraphTraversalDepth`
- `HighestTotalGraphTraversals`
- `LowestTotalGraphTraversals`
- `HighestNumberOfCollisionsDuringAssignment`
- `LowestNumberOfCollisionsDuringAssignment`

## ValueSizeInBytes

Sets the size, in bytes, of the value element that will be stored in the
compiled perfect hash table via `Insert()`.  Defaults to 4 bytes (`ULONG`).

## MainWorkThreadpoolPriority

Sets the priority of the main work threadpool (i.e. the CPU-intensive graph
solving threadpool) to the given value.  Defaults to `Normal`.  Unlikely to
require customization.

**Note:** Windows only.

## FileWorkThreadpoolPriority

Sets the priority of the file work threadpool (i.e. the threadpool that
handles file I/O operations) to the given value.  Defaults to `Normal`.
Unlikely to require customization.

**Note:** Windows only.

## AttemptsBeforeTableResize

Specifies the number of attempts at solving the graph that will be made
before a table resize event will occur (assuming that resize events are
permitted by the value of `--MaxNumberOfTableResizes`).

Defaults to `MAX_ULONG` (i.e. `4,294,967,295`), which effectively disables
table resizing.

**Note:** Windows only.

## MaxNumberOfTableResizes

Maximum number of table resizes that will be permitted before giving up and
concluding that a solution cannot be found.  Defaults to `5`.

**Note:** Windows only.

## InitialNumberOfTableResizes

Initial number of table resizes to simulate before attempting graph solving.
Each table resize doubles the number of vertices used to solve the graph,
which lowers the *keys-to-vertices* ratio, which will improve graph solving
probability.  Defaults to `0`.

**Note:** Windows only.

## AutoResizeWhenKeysToEdgesRatioExceeds

Supplies a *keys-to-edges* ratio that, if exceeded, results in an auto
resize, i.e. the equivalent of `--InitialNumberOfTableResizes=1`.  Valid
values are above `0.0` and less than `1.0`.  Typical values would be 0.8 to
0.9 depending on the hash function being used.

This will result in much faster solving rates for *nearly-full* key sets
(i.e., when the number of keys is very close to a power of two, e.g.
`HydrogenWorld-31016.keys`).

**Note:** Windows only.

## BestCoverageAttempts

Supplies a positive integer that represents the number of attempts that
will be made at finding a *best* graph (based on the best coverage type
requested) before the create table routine returns.

For example, if this value is set to 5, the solving will stop when the 5th
new best graph is found.  A graph is considered a *"new best"* if its coverage
type predicate is the highest/lowest seen before.

See also: `--BestCoverageType`, and `--MaxNumberOfEqualBestGraphs`.

## BestCoverageType

Indicates the predicate to determine what constitutes the best graph.

**Note:** The terms *best graph* and *best coverage* mean the same thing.
You're either in *first graph wins* mode, or *find best graph* mode.  When
the latter applies, we're looking for the best graph, and that means the one
with the winning coverage predicate.

**Note:** The following predicates only apply to `--GraphImpl=1|2`:

- `HighestMaxGraphTraversalDepth`
- `LowestMaxGraphTraversalDepth`
- `HighestTotalGraphTraversals`
- `LowestTotalGraphTraversals`
- `HighestNumberOfCollisionsDuringAssignment`
- `LowestNumberOfCollisionsDuringAssignment`

## KeysSubset

Supplies a comma-separated list of keys in ascending key-value order.  Must
contain two or more elements.  This parameter is only useful when combined
with one of the following `--BestCoverageType` predicates that are exclusive
to this mode:

- `HighestMaxAssignedPerCacheLineCountForKeysSubset`
- `LowestMaxAssignedPerCacheLineCountForKeysSubset`
- `HighestNumberOfCacheLinesUsedByKeysSubset`
- `LowestNumberOfCacheLinesUsedByKeysSubset`
- `HighestNumberOfLargePagesUsedByKeysSubset`
- `LowestNumberOfLargePagesUsedByKeysSubset`
- `HighestNumberOfPagesUsedByKeysSubset`
- `LowestNumberOfPagesUsedByKeysSubset`

It is intended to cover scenarios where the runtime behavior of a perfect
hash table exhibits highly-skewed cardinality distributions; that is, a
small number of keys are accessed much more frequently than the rest.

If you know apriori which keys are accessed most frequently, you can supply
them as a keys subset here, then select a best coverage type predicate that
will try to find the best solution, for example, that minimizes the number
of cache lines used by the keys subset.  This would mean the most frequently
occurring keys occupy the fewest cache lines, which would result in more
cache lines being available for other data.  In this case, you'd use:

`--KeysSubset=50,70,... --BestCoverageType=LowestNumberOfCacheLinesUsedByKeysSubset`

Alternatively, you may want to maximize the number of cache lines used by
the most frequent keys, perhaps if some sort of interlocked operation is
performed once the key index is obtained.  Having multiple threads attempt
interlocked operations on the same cache line can result in severe performance
degradation.  In this case, you'd use:

`--KeysSubset=50,70,... --BestCoverageType=HighestNumberOfCacheLinesUsedByKeysSubset`

The number of keys in the subset is limited to command line parameter length
restrictions.  There is not currently a way to provide key subsets via a
file.

## MaxNumberOfEqualBestGraphs

Supplies a positive integer that represents the number of times an *equal*
best graph is encountered (based on the best coverage type) before stopping
fur further solving attempts for this graph.

For example, let's say you're using `--BestCoverageType=HighestNumberOfEmptyCacheLines`
and `--BestCoverageAttempts=5`, and that 4th new best graph encountered had a
value of `8` for this coverage type; subsequent graphs that also have a
value of `8` get classed as an *equal* best graph (as we've already found
one with `8`).  If we supply `--MaxNumberOfEqualBestGraphs=10`, then we'll
stop further solving attempts once we see the 10th graph that has `8`
empty cache lines.

This parameter is particularly useful for the *highest* predicates that
aren't restricted by page or cache line quantities, e.g.:

- `HighestMaxGraphTraversalDepth`
- `HighestTotalGraphTraversals`
- `HighestNumberOfCollisionsDuringAssignment`

However, it's still useful for all other predicates as a mechanism for
avoiding never solving a graph (because you never hit the Nth best graph
attempt).

## MinNumberOfKeysForFindBestGraph

Supplies a positive integer that represents the minimum number of keys
that need to be present before `--FindBestGraph` mode is honored.  This
parameter is particularly useful in *Bulk Create* mode when processing many
keys files of different sizes.

Defaults to `512`.  Set to `0` to disable this behavior.

The reasoning behind this parameter is that there's very little benefit in
finding the best graph for a small number of keys, as there's not going to
be enough variation in assigned value cache line occupancy to yield runtime
performance differences.

## BestCoverageTargetValue

Supplies either a floating point number or positive integer that represents the
target value for the best coverage type predicate.  Graph solving will stop
when a solution is found that meets this target value.

The type and value depends on the best coverage type used.  For example,
`HighestRank` and `LowestSlope` use floating point numbers, whereas
`HighestScore` (and almost everything else) uses positive integers.

When provided, graph solving will be stopped if a best graph's coverage
value meets the target value provided by this parameter.  The type of
comparison is derived from the coverage type, e.g., if the following
params are provided:

`--BestCoverageType=HighestRank --BestCoverageTargetValue=0.5`

Then graph solving will stop when a solution is found that has a rank
greater than or equal to `0.5`.  If LowestRank was specified, the reverse
applies: we'd stop solving as soon as we see a solution with a rank
less than or equal to `0.5`.

In *Bulk Create* mode, the most useful predicate is rank, as it is a
normalized score between `[0.0, 1.0)`, and a rank of `0.5` or greater is
usually indicative of a tightly-packed assigned table (which we want).

Other predicates use absolute values, which aren't useful in *Bulk Create*
mode when you have many differing key sizes (e.g. `HighestScore` and
`--BestCoverageTargetValue=200000` does not make sense for *Bulk Create* as
a table has to be a certain size in order to achieve that score).

This parameter can be used in conjunction with other parameters like
`--FixedAttempts=N` or `--TargetNumberOfSolutions=N`.  However, note that
whichever limit is reached first will terminate the solving; i.e. if
you use `--BestCoverageType=HighestRank --BestCoverageTargetValue=0.5`
and `--FixedAttempts=10`, then solving will stop after 10 attempts,
regardless of whether or not the target value is reached.

Also note that this behavior, as with all *find best graph* behavior,
is trumped by the logic that skips finding a best graph if there are
less than the minimum number of keys available (default: `512`).  This
can be altered via `--MinNumberOfKeysForFindBestGraph`.

In general, this parameter is useful for finding a balance between solving
time and solution quality; some key sets may take a lot of attempts to break
a rank of `0.39` to `0.40`, but in general, *most* keys (at least in the
venerable `sys32` set) will eventually yield tables with a `Rank` of `0.5`
or greater within a few seconds to a few minutes.  Anything above 0.5 is
probabilistically harder to achieve, and may take tens of minutes to tens
of hours.  The highest `Rank` observed in practice is `0.503845` on the
`sys32/Hologram-31016.keys` key set.

## FixedAttempts

Supplies a positive integer that represents the number of attempts that
will be made at solving the graph before the create table routine returns
for a given keys file.

This parameter is useful for benchmarking purposes, as it allows you the
most deterministic way to control the number of attempts made at solving
the graph.  When using the default *random number generator* parameters,
independent runs will behave identically if the same number of attempts
are made, which is desirable for benchmarking.

Note that this parameter is ignorant to whether or not a solution is found;
an attempt is counted regardless of the outcome.  Additionally, if a
solution is found before the fixed attempts are exhausted, the routine will
continue to solve the graph until the fixed attempts are exhausted.

## TargetNumberOfSolutions

Supplies a positive integer that represents the number of solutions that
will be attempted to be found before the create table routine returns for
a given keys file.

This is useful when in `--FindBestGraph` mode, as it allows you to target an
arbitrary number of best graphs without having to specify a specific target
value for the best coverage type predicate.

Higher values will result in much longer runtimes.  Too high a value, and
the routine may never return.  `5` is a good starting point for most key
sets; it is not recommended to go above `10` without good reason for a given
coverage type predicate.

## Seeds

Supplies an optional comma-separated list of up to 8 integers that
represent the seed values to use for every graph solving attempt.
Each value may be zero, which tells the algorithm to use a random
seed for this position as per normal.

The logic is also cognizant of the hash function's seed masks, e.g.
`MultiplyShiftR` has a seed mask of `0x1f1f` for seed 3 (which is used to
control the final right shift amount), so, if we use the following:

`--Seeds=0,0,0x1000`

It will use random bytes for the first two seeds.  For the second byte
of the third seed, it'll use `0x10` (as `4096` is `0x1000` in hex), but will
use a random byte for the first byte.  (If we were to use `--Seeds=0,0,16`,
then the first byte will be locked to `0x10` and the second byte will be
random.)

This has proven useful for the hash function `MultiplyShiftR` when using
`--InitialNumberOfTableResizes=1 --Seeds=0,0,0x1010` as it forces all
vertices to be constrained to the first half of the assigned array
(thus negating the overhead of a table resize).  It may be useful in
other contexts, too.

**Note:** Either hex or decimal can be used for the seed values.

## Seed3Byte1MaskCounts

Supplies a comma-separated list of 32 integers that represent weighted
counts of seed 3's first byte mask value.  (Experimental.)

## Seed3Byte2MaskCounts

Supplies a comma-separated list of 32 integers that represent weighted
counts of seed 3's second byte mask value.  (Experimental.)

## SolutionsFoundRatio

Supplies a double (64-bit) floating point number indicating the ratio
of solutions found (obtained from a prior run).  This is then used to
calculate the predicted number of attempts required to solve a given
graph; when combined with `--TryUsePredictedAttemptsToLimitMaxConcurrency`
the maximum concurrency used when solving will be the minimum of the
predicted attempts and the maximum concurrency indicated on the command
line.

See also: `--TryUsePredictedAttemptsToLimitMaxConcurrency`.

## Rng

Supplies the name of a random number generator to use for obtaining the
random bytes needed as part of graph solving.  Valid values:

`Philox43210`

> Uses the Philox 4x32 10-round pseudo-RNG.  This is the default.
> This should be used when benchmarking creation performance, as
> it ensures the random numbers fed to each graph solving attempt
> are identical between runs, resulting in consistent runtimes
> across subsequent runs.  It may result in slower solving times
> versus the System RNG, depending on your key set.

`System`

> Uses the standard operating system facilities for obtaining
> random data.  All other --Rng* parameters are ignored.  This
> should be used when attempting to find legitimate solutions,
> however, due to the inherent randomness, will result in varying
> runtimes across subsequent runs.

## RngSeed

Supplies a 64-bit seed used to initialize the RNG.  Defaults to
`0x2019090319811025`, unless `--RngUseRandomStartSeed` is supplied (in which
case, a random seed will be used, obtained via the operating system).

**Note:** Only applies to `Philox43210` RNG.

## RngSubsequence

Supplies the initial subsequence used by the RNG.  The first graph will
use this sequence, with each additional graph adding 1 to this value for
their subsequence.  This ensures parallel graphs generate different
random numbers (even if the seed is identical) when solving.  (Defaults
to 0.)

**Note:** Only applies to `Philox43210` RNG.

## RngOffset

Supplies the initial offset used by the RNG.  (Defaults to 0.)

**Note:** Only applies to `Philox43210` RNG.

## Remark

Supplies a remark to be associated with the run that will be included
in the .csv output files under the `Remark` column.  An error will
be returned if the provided string contains commas (as this will
break the `.csv` output).

## MaxSolveTimeInSeconds

Supplies the maximum number of seconds to try and solve an individual
graph.  If a solution is not found within this time, the routine will
return with an error.  In *Bulk Create* mode, the program will move onto the
next keys file.

This is useful for ensuring that the routine doesn't get stuck on a single
key set for an extended period of time.

## FunctionHookCallbackDllPath

Supplies a fully-qualified path to a `.dll` file that will be used as the
callback handler for hooked functions.

**Note:** Windows only.

## FunctionHookCallbackFunctionName

Supplies the exported function name to resolve from the callback module
(above) and use as the callback for hooked functions.  The default is
`InterlockedIncrement`.

**Note:** Windows only.

## FunctionHookCallbackIgnoreRip

Supplies a relative RIP to ignore during function callback.  That is,
if a caller matches the supplied relative RIP, the function callback
will not be executed.

**Note:** Windows only.

# Console Output Character Legend

| Char | Meaning |
|------|---------|
| .    | Table created successfully. |
| +    | Table resize event occurred. |
| x    | Failed to create a table. The maximum number of attempts at trying to solve the table at a given size was reached, and no more resize attempts were possible (due to the maximum resize limit also being hit). |
| F    | Failed to create a table due to a target not being reached by a specific number of attempts. |
| *    | None of the worker threads were able to allocate sufficient memory to attempt solving the graph. |
| !    | The system is out of memory. |
| L    | The system is running low on memory (a low memory event is triggered at about 90% RAM usage). In certain situations, we can detect this situation prior to actually running out of memory; in these cases, we abort the current table creation attempt (which will instantly relieve system memory pressure). |
| V    | The graph was created successfully, however, we weren't able to allocate enough memory for the table values array in order for the array to be used after creation. This can be avoided by supplying the command line parameter --SkipTestAfterCreate. |
| T    | The requested number of table elements was too large. |
| S    | A shutdown event was received. This shouldn't be seen unless externally signaling the named shutdown event associated with a context. |
| t    | The solve timeout was reached before a solution was found. |
| ?    | The error code isn't recognized! E-mail trent@trent.me with details. |

