/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTableTest.c

Abstract:

    This module implements the test routine for an individual instance of a
    PERFECT_HASH_TABLE structure.  It is primarily used by the module's
    self-test functionality.

--*/

#include "stdafx.h"

//
// Forward decls.
//

typedef
VOID
(NTAPI PERFECT_HASH_TABLE_BENCHMARK)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ PKEY Key
    );
typedef PERFECT_HASH_TABLE_BENCHMARK *PPERFECT_HASH_TABLE_BENCHMARK;

extern PERFECT_HASH_TABLE_BENCHMARK PerfectHashTableBenchmark;

//
// Disable global optimizations, even in release builds.  Without this, the
// compiler does clever things with regards to optimizing our __debugbreak()
// logic below, such that it's impossible to tell which ASSERT() triggered it.
//

#pragma optimize("", off)

//
// Use a custom ASSERT() macro that optionally issues a __debugbreak() based
// on the caller's DebugBreakOnFailure boolean flag.
//

#undef ASSERT
#define ASSERT(Condition)          \
    if (!(Condition)) {            \
        if (DebugBreakOnFailure) { \
            __debugbreak();        \
        };                         \
        goto Error;                \
    }

PERFECT_HASH_TABLE_TEST PerfectHashTableTest;

_Use_decl_annotations_
HRESULT
PerfectHashTableTest(
    PPERFECT_HASH_TABLE Table,
    PPERFECT_HASH_KEYS Keys,
    BOOLEAN DebugBreakOnFailure
    )
/*++

Routine Description:

    Tests an instance of a PERFECT_HASH_TABLE for correctness.

Arguments:

    Table - Supplies a pointer to an initialized PERFECT_HASH_TABLE structure
        for which the testing will be undertaken.

    Keys - Optionally supplies the keys associatd with the table.  This is
        mandatory if keys were not originally provided when the table was
        loaded.

    DebugBreakOnFailure - Supplies a boolean flag that indicates whether or
        not a __debugbreak() should be issued if a test fails.

Return Value:

    S_OK if all internal tests pass, an appropriate error code if a failure
    is encountered.

--*/
{
    PRTL Rtl;
    ULONG Index;
    ULONG Value;
    ULONG Rotated;
    ULONG Previous;
    ULONG NumberOfKeys;
    ULONG ValueIndex;
    ULONG NumberOfBitsSet;
    KEY Key;
    KEY FirstKey;
    PKEY Source;
    PKEY SourceKeys;
    HRESULT Result = S_OK;
    PALLOCATOR Allocator;
    RTL_BITMAP Indices;
    PLONG BitmapBuffer = NULL;
    ULARGE_INTEGER NumberOfBits;
    ULARGE_INTEGER BitmapBufferSize;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Keys)) {
        Keys = Table->Keys;
    }

    if (!Keys) {
        return PH_E_KEYS_REQUIRED_FOR_TABLE_TEST;
    }

    //
    // Initialize aliases.
    //

    Rtl = Table->Rtl;
    Allocator = Table->Allocator;

    //
    // Sanity check the perfect hash structure size matches what we expect.
    //

    ASSERT(Table->SizeOfStruct == sizeof(*Table));

    //
    // Verify the table has a valid keys structure wired up, and that the number
    // of keys is within MAX_ULONG.
    //

    ASSERT(Keys && Keys->File && Keys->KeyArrayBaseAddress);
    ASSERT(!Keys->NumberOfElements.HighPart);

    NumberOfKeys = Keys->NumberOfElements.LowPart;

    //
    // Calculate the space required for a bitmap buffer that allows us to set
    // a bit for each unique key index.
    //

    NumberOfBits.QuadPart = (ULONGLONG)NumberOfKeys;
    ASSERT(!NumberOfBits.HighPart);

    Indices.SizeOfBitMap = NumberOfBits.LowPart;

    //
    // Align the bitmap up to an 8 byte boundary then divide by 8.
    //

    BitmapBufferSize.QuadPart = (
        ALIGN_UP(NumberOfBits.QuadPart, 8) >> 3ULL
    );

    //
    // Overflow check.
    //

    ASSERT(!BitmapBufferSize.HighPart);

    //
    // Allocate the memory.
    //

    BitmapBuffer = (PLONG)(
        Allocator->Vtbl->Calloc(
            Allocator,
            1,
            BitmapBufferSize.LowPart
        )
    );

    if (!BitmapBuffer) {
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    //
    // N.B. We use PLONG for the BitmapBuffer type instead of PULONG (which is
    //      the type of the RTL_BITMAP Buffer field), because we use the bit
    //      test and set intrinsic below, and that wants PLONG.
    //

    Indices.Buffer = (PULONG)BitmapBuffer;

    //
    // Grab the first key from the underlying keys table such that we've got
    // a valid key that was present in the input set.
    //


    SourceKeys = (PKEY)Keys->KeyArrayBaseAddress;
    FirstKey = Key = *SourceKeys;

    //
    // Rotate the key such that it differs from the original value, but in a
    // way that can be easily reversed.
    //

    Rotated = _rotl(Key, 15);
    ASSERT(Key == _rotr(Rotated, 15));

    //
    // Verify looking up a key that hasn't been inserted returns 0 as the value.
    //

    Result = Table->Vtbl->Lookup(Table, Key, &Value);
    ASSERT(!FAILED(Result));
    ASSERT(Value == 0);

    //
    // Verify insertion.
    //

    Result = Table->Vtbl->Insert(Table, Key, Rotated, &Previous);
    ASSERT(!FAILED(Result));
    ASSERT(Previous == 0);

    //
    // Lookup the inserted key.
    //

    Result = Table->Vtbl->Lookup(Table, Key, &Value);
    ASSERT(!FAILED(Result));
    ASSERT(Value == Rotated);

    //
    // Delete the inserted key.  Returned value should be Rotated.
    //

    Value = 0;
    Result = Table->Vtbl->Delete(Table, Key, &Value);
    ASSERT(!FAILED(Result));
    ASSERT(Value == Rotated);

    //
    // Verify a subsequent lookup returns 0.
    //

    Value = (ULONG)-1;
    Result = Table->Vtbl->Lookup(Table, Key, &Value);
    ASSERT(!FAILED(Result));
    ASSERT(Value == 0);

    //
    // Loop through the entire key set and obtain the values array index for
    // that given key.  Verify that the corresponding bit in our bitmap isn't
    // set -- if it is, it indicates we've been given the same index for two
    // different keys, which is a pretty severe error.
    //

    ValueIndex = 0;

    for (Index = 0, Source = SourceKeys; Index < NumberOfKeys; Index++) {

        Key = *Source++;

        Result = Table->Vtbl->Index(Table, Key, &ValueIndex);
        ASSERT(!FAILED(Result));

        //
        // Verify bit isn't already set and set it.
        //

        ASSERT(!BitTestAndSet(BitmapBuffer, ValueIndex));

    }

    //
    // Count the number of set bits and verify it matches the number of keys.
    //

    NumberOfBitsSet = Rtl->RtlNumberOfSetBits(&Indices);
    ASSERT(NumberOfBitsSet == NumberOfKeys);

    //
    // Loop through the entire key set and insert a rotated version of the key.
    //

    for (Index = 0, Source = SourceKeys; Index < NumberOfKeys; Index++) {

        Key = *Source++;
        Rotated = _rotl(Key, 15);

        Result = Table->Vtbl->Insert(Table, Key, Rotated, &Previous);
        ASSERT(!FAILED(Result));
        ASSERT(Previous == 0);

    }

    //
    // Loop through the entire set again and ensure that lookup returns the
    // rotated version.
    //

    for (Index = 0, Source = SourceKeys; Index < NumberOfKeys; Index++) {

        Key = *Source++;
        Rotated = _rotl(Key, 15);

        Result = Table->Vtbl->Lookup(Table, Key, &Value);
        ASSERT(!FAILED(Result));
        ASSERT(Value == Rotated);

    }

    //
    // Loop through again and delete everything.
    //

    for (Index = 0, Source = SourceKeys; Index < NumberOfKeys; Index++) {

        Key = *Source++;
        Rotated = _rotl(Key, 15);

        Result = Table->Vtbl->Delete(Table, Key, &Previous);
        ASSERT(!FAILED(Result));
        ASSERT(Previous == Rotated);

    }

    //
    // And a final loop through to confirm all lookups now return 0.
    //

    for (Index = 0, Source = SourceKeys; Index < NumberOfKeys; Index++) {

        Key = *Source++;

        Result = Table->Vtbl->Lookup(Table, Key, &Value);
        ASSERT(!FAILED(Result));
        ASSERT(Value == 0);

    }

    //
    // All of the tests completed, so capture some rudimentary benchmarks.
    //

    PerfectHashTableBenchmark(Table, &FirstKey);

    //
    // We're finished!  Indicate success and finish up.
    //

    Result = S_OK;
    goto End;

Error:

    if (Result == S_OK) {
        Result = PH_E_SELF_TEST_OF_HASH_TABLE_FAILED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    if (BitmapBuffer) {
        Allocator->Vtbl->FreePointer(Allocator, &BitmapBuffer);
    }

    return Result;
}

#if 0
#define END_TIMESTAMP END_TIMESTAMP_RDTSC
#define START_TIMESTAMP START_TIMESTAMP_RDTSC
#endif

#if 1
#define END_TIMESTAMP END_TIMESTAMP_RDTSCP
#define START_TIMESTAMP START_TIMESTAMP_RDTSCP
#endif

#define DEFAULT_WARMUPS 10
#define DEFAULT_ATTEMPTS 10
#define DEFAULT_ITERATIONS 100

extern PERFECT_HASH_TABLE_HASH PerfectHashTableHashNull;
extern PERFECT_HASH_TABLE_SEEDED_HASH PerfectHashTableSeededHashNull;

PERFECT_HASH_TABLE_BENCHMARK PerfectHashTableBenchmark;

_Use_decl_annotations_
VOID
PerfectHashTableBenchmark(
    PPERFECT_HASH_TABLE Table,
    PKEY KeyPointer
    )
/*++

Routine Description:

    Benchmarks an instance of a perfect hash table.

Arguments:

    Table - Supplies a pointer to an initialized PERFECT_HASH_TABLE structure
        for which the benchmarking will be undertaken.

    KeyPointer - Supplies a pointer to a valid key to use for routines being
        benchmarked.

Return Value:

    None.

--*/
{
    KEY Key;
    PRTL Rtl;
    ULONG Index;
    ULONG Inner;
    ULONG Outer;
    ULONG Warmups;
    ULONG Attempts;
    ULONG Iterations;
    ULONG NumberOfSeeds;
    PULONG FirstSeed;
    HRESULT Result;
    LARGE_INTEGER Frequency;
    ULARGE_INTEGER Hash;
    PTIMESTAMP SlowIndex;
    PTIMESTAMP SeededHash;
    PTIMESTAMP NullSeededHash;
    PTABLE_INFO_ON_DISK TableInfo;
    PPERFECT_HASH_TABLE_SEEDED_HASH OriginalSeededHashFunc;

    //
    // Initialize aliases and variables.
    //

    Rtl = Table->Rtl;
    Key = *KeyPointer;
    Hash.QuadPart = 0;
    TableInfo = Table->TableInfoOnDisk;
    SlowIndex = &Table->SlowIndexTimestamp;
    SeededHash = &Table->SeededHashTimestamp;
    NullSeededHash = &Table->NullSeededHashTimestamp;

    //
    // If the number of warmups, attempts, or iterations indicate 0, use default
    // values.  We update the corresponding value in the table structure such
    // that the .csv row generation logic picks up whatever value was used.
    //

    Warmups = Table->BenchmarkWarmups;
    if (!Warmups) {
        Warmups = Table->BenchmarkWarmups = DEFAULT_WARMUPS;
    }

    Attempts = Table->BenchmarkAttempts;
    if (!Attempts) {
        Attempts = Table->BenchmarkAttempts = DEFAULT_ATTEMPTS;
    }

    Iterations = Table->BenchmarkIterations;
    if (!Iterations) {
        Iterations = Table->BenchmarkIterations = DEFAULT_ITERATIONS;
    }

    NumberOfSeeds = TableInfo->NumberOfSeeds;
    FirstSeed = &TableInfo->FirstSeed;

    QueryPerformanceFrequency(&Frequency);

    //
    // Perform the seeded hash warmup, then benchmark.
    //

    INIT_TIMESTAMP(SeededHash);

    for (Outer = 0; Outer < Warmups; Outer++) {
        Result = Table->Vtbl->SeededHash(Table,
                                         Key,
                                         NumberOfSeeds,
                                         FirstSeed,
                                         &Hash.QuadPart);
    }

    for (Outer = 0; Outer < Attempts; Outer++) {
        START_TIMESTAMP(SeededHash);
        for (Inner = 0; Inner < Iterations; Inner++) {
            Result = Table->Vtbl->SeededHash(Table,
                                             Key,
                                             NumberOfSeeds,
                                             FirstSeed,
                                             &Hash.QuadPart);
        }
        END_TIMESTAMP(SeededHash);
    }

    //
    // Capture the seeded hash routine and replace it with the null one.
    //

    OriginalSeededHashFunc = Table->Vtbl->SeededHash;
    Table->Vtbl->SeededHash = PerfectHashTableSeededHashNull;

    //
    // Repeat the warmup and benchmark.
    //

    INIT_TIMESTAMP(NullSeededHash);

    for (Outer = 0; Outer < Warmups; Outer++) {
        Result = Table->Vtbl->SeededHash(Table,
                                         Key,
                                         NumberOfSeeds,
                                         FirstSeed,
                                         &Hash.QuadPart);
    }

    for (Outer = 0; Outer < Attempts; Outer++) {
        START_TIMESTAMP(NullSeededHash);
        for (Inner = 0; Inner < Iterations; Inner++) {
            Result = Table->Vtbl->SeededHash(Table,
                                             Key,
                                             NumberOfSeeds,
                                             FirstSeed,
                                             &Hash.QuadPart);
        }
        END_TIMESTAMP(NullSeededHash);
    }

    //
    // Restore the seeded hash vtbl routine.
    //

    Table->Vtbl->SeededHash = OriginalSeededHashFunc;

    //
    // Perform the slow index warmup, then benchmark.
    //

    INIT_TIMESTAMP(SlowIndex);

    for (Outer = 0; Outer < Warmups; Outer++) {
        Result = Table->Vtbl->SlowIndex(Table, Key, &Index);
    }

    for (Outer = 0; Outer < Attempts; Outer++) {
        START_TIMESTAMP(SlowIndex);
        for (Inner = 0; Inner < Iterations; Inner++) {
            Result = Table->Vtbl->SlowIndex(Table, Key, &Index);
        }
        END_TIMESTAMP(SlowIndex);
    }
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
