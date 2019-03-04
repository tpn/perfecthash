/*++

Copyright (c) 2018-2019 Trent Nelson <trent@trent.me>

Module Name:

    CompiledPerfectHash.h

Abstract:

    This is the main public header file for the compiled perfect hash library.
    It defines structures and functions related to loading and using compiled
    perfect hash tables.

--*/

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

//
// Platform-dependent defines.
//

#ifdef _WIN32
#include <sal.h>

//
// The intrinsics headers trigger a lot of warnings when /Wall is on.
//

#pragma warning(push)
#pragma warning(disable: 4255 4514 4668 4820 28251)
#include <intrin.h>
#include <mmintrin.h>
#pragma warning(pop)

#define DEBUGBREAK __debugbreak
#define CPHCALLTYPE __stdcall
#ifndef FORCEINLINE
#define FORCEINLINE __forceinline
#endif
#elif defined(__linux__) || defined(__APPLE__)
#define CPHCALLTYPE
#if defined(__clang__)
#include <x86intrin.h>

//
// Clang doesn't appear to support the rotate intrinsics _rotr and _rotl,
// so, define some static inline versions here.
//

static inline
unsigned int
_rotl(
    unsigned int a,
    unsigned int b
    )
{
    b &= 31;
    return (a << b) | (a >> (32 - b));
}

static inline
unsigned int
_rotr(
    unsigned int a,
    unsigned int b
    )
{
    b &= 31;
    return (a >> b) | (a << (32 - b));
}

#elif defined(__GNUC__)
#include <x86intrin.h>
#else
#error Unrecognized compiler.
#endif
#include <no_sal2.h>
#ifndef FORCEINLINE
#define FORCEINLINE static inline __attribute__((always_inline))
#define DEBUGBREAK __builtin_trap
#endif
#else
#error Unsupported platform.
#endif

#if defined(COMPILED_PERFECT_HASH_DLL_BUILD)
#define CPHAPI __declspec(dllexport)
#elif defined(COMPILED_PERFECT_HASH_EXE_BUILD)
#define CPHAPI __declspec(dllimport)
#else
#define CPHAPI
#endif

#if defined(COMPILED_PERFECT_HASH_DLL_BUILD)
#define CPHAPI __declspec(dllexport)
#elif defined(COMPILED_PERFECT_HASH_EXE_BUILD)
#define CPHAPI __declspec(dllimport)
#else
#define CPHAPI
#endif

#ifdef _M_X64

//
// Define start/end markers for IACA.
//

#define IACA_VC_START() __writegsbyte(111, 111)
#define IACA_VC_END()   __writegsbyte(222, 222)

#endif

//
// Define the main functions exposed by a compiled perfect hash table: index,
// lookup, insert and delete.
//

typedef
CPHAPI
CPHINDEX
(CPHCALLTYPE COMPILED_PERFECT_HASH_TABLE_INDEX)(
    _In_ CPHKEY Key
    );
/*++

Routine Description:

    Looks up given key in a compiled perfect hash table and returns its index.

    N.B. If the given key did not appear in the original set the hash table was
         created from, the behavior of this routine is undefined.  (In practice, the
         key will hash to either an existing key's location or an empty slot,
         so there is potential for returning a non-unique index.)

Arguments:

    Key - Supplies the key to look up.

Return Value:

    The index associated with the given key.

--*/
typedef COMPILED_PERFECT_HASH_TABLE_INDEX *PCOMPILED_PERFECT_HASH_TABLE_INDEX;

#ifndef CPH_INDEX_ONLY

typedef
CPHAPI
CPHVALUE
(CPHCALLTYPE COMPILED_PERFECT_HASH_TABLE_LOOKUP)(
    _In_ CPHKEY Key
    );
/*++

Routine Description:

    Looks up given key in a compiled perfect hash table and returns the value
    present.  If no insertion has taken place for this key, this routine
    guarantees to return 0 as the value.

    N.B. If the given key did not appear in the original set the hash table was
         created from, the behavior of this routine is undefined.  (In practice, the
         value returned will be the value for some other key in the table that
         hashes to the same location -- or potentially an empty slot in the
         table.)

Arguments:

    Key - Supplies the key to look up.

Return Value:

    The value at the given location.

--*/
typedef COMPILED_PERFECT_HASH_TABLE_LOOKUP *PCOMPILED_PERFECT_HASH_TABLE_LOOKUP;


typedef
CPHAPI
CPHVALUE
(CPHCALLTYPE COMPILED_PERFECT_HASH_TABLE_INSERT)(
    _In_ CPHKEY Key,
    _In_ CPHVALUE Value
    );
/*++

Routine Description:

    Inserts value at key into a compiled hash table, and returns the previously
    set value (which will be 0 if no prior insert occurred).

    N.B. If the given key did not appear in the original set the hash table was
         created from, the behavior of this routine is undefined.  (In practice, the
         key will hash to either an existing key's location or an empty slot, so
         there is potential to corrupt the table in the sense that previously
         inserted values will be trampled over.)

Arguments:

    Key - Supplies the key for which the value will be inserted.

    Value - Supplies the value to insert.

Return Value:

    Previous value at the relevant table location prior to this insertion.

--*/
typedef COMPILED_PERFECT_HASH_TABLE_INSERT *PCOMPILED_PERFECT_HASH_TABLE_INSERT;


typedef
CPHAPI
CPHVALUE
(CPHCALLTYPE COMPILED_PERFECT_HASH_TABLE_DELETE)(
    _In_ CPHKEY Key
    );
/*++

Routine Description:

    Deletes a key from a perfect hash table, optionally returning the value
    prior to deletion back to the caller.  Deletion simply clears the value
    associated with the key, and thus, is a simple O(1) operation.  Deleting
    a key that has not yet been inserted has no effect other than potentially
    returning 0 as the previous value.  That is, a caller can safely issue
    deletes of keys regardless of whether or not said keys were inserted first.

    N.B. If the given key did not appear in the original set the hash table
         was created from, the behavior of this routine is undefined.  (In
         practice, the key will hash to either an existing key's location or
         an empty slot, so there is potential to corrupt the table in the
         sense that a previously inserted value for an unrelated, valid key
         will be cleared.)

Arguments:

    Key - Supplies the key to delete.

Return Value:

    Previous value at the given key's location prior to deletion.  If no prior
    insertion, the previous value is guaranteed to be 0.

--*/
typedef COMPILED_PERFECT_HASH_TABLE_DELETE *PCOMPILED_PERFECT_HASH_TABLE_DELETE;

//
// Typedefs of methods for testing and benchmarking.
//

typedef
_Success_(return == 0)
ULONG
(CPHCALLTYPE TEST_COMPILED_PERFECT_HASH_TABLE)(
    _In_opt_ BOOLEAN DebugBreakOnFailure
    );
/*++

Routine Description:

    Tests a compiled perfect hash table for correctness.

Arguments:

    DebugBreakOnFailure - Supplies a boolean flag that indicates whether or
        not a __debugbreak() should be issued as soon as a test fails.

Return Value:

    The number of failed tests.  If 0, all tests passed.

--*/
typedef TEST_COMPILED_PERFECT_HASH_TABLE *PTEST_COMPILED_PERFECT_HASH_TABLE;

typedef
ULONG
(CPHCALLTYPE BENCHMARK_FULL_COMPILED_PERFECT_HASH_TABLE)(
    _In_ ULONG Seconds
    );
/*++

Routine Description:

    Benchmarks the time it takes to insert all keys into a table, lookup the
    inserted value, then delete all keys.

Arguments:

    Seconds - TBD.

Return Value:

    Number of cycles.

--*/
typedef BENCHMARK_FULL_COMPILED_PERFECT_HASH_TABLE
      *PBENCHMARK_FULL_COMPILED_PERFECT_HASH_TABLE;

#endif // #ifndef CPH_INDEX_ONLY

typedef
ULONG
(CPHCALLTYPE BENCHMARK_INDEX_COMPILED_PERFECT_HASH_TABLE)(
    _In_ ULONG Seconds
    );
/*++

Routine Description:

    Benchmarks the time it takes to perform just the Index() operation.

Arguments:

    Seconds - TBD.

Return Value:

    Number of cycles.

--*/
typedef BENCHMARK_INDEX_COMPILED_PERFECT_HASH_TABLE
      *PBENCHMARK_INDEX_COMPILED_PERFECT_HASH_TABLE;

#ifdef __cplusplus
} // extern "C"
#endif

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
