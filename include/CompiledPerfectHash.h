/*++

Copyright (c) 2018-2024 Trent Nelson <trent@trent.me>

Module Name:

    CompiledPerfectHash.h

Abstract:

    This is the main public header file for the compiled perfect hash library.
    It defines structures and functions related to loading and using compiled
    perfect hash tables.

--*/

#ifndef _COMPILED_PERFECT_HASH_H_
#define _COMPILED_PERFECT_HASH_H_


#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

//
// Static inline C versions of rotate and bit extraction.
//

//
// Equivalent to _rotl intrinsic on x64.
//

static inline
uint32_t
RotateLeft32_C(
    uint32_t a,
    uint32_t b
    )
{
    b &= 31;
    return (a << b) | (a >> (32 - b));
}

//
// Equivalent to _rotr intrinsic on x64.
//

static inline
uint32_t
RotateRight32_C(
    uint32_t a,
    uint32_t b
    )
{
    b &= 31;
    return (a >> b) | (a << (32 - b));
}

//
// Equivalent to _rotl64 intrinsic on x64.
//

static inline
uint64_t
RotateLeft64_C(
    uint64_t a,
    uint64_t b
    )
{
    b &= 63;
    return (a << b) | (a >> (64 - b));
}

//
// Equivalent to _rotr64 intrinsic on x64.
//

static inline
uint64_t
RotateRight64_C(
    uint64_t a,
    uint64_t b
    )
{
    b &= 63;
    return (a >> b) | (a << (64 - b));
}

//
// Equivalent to _pext_u64 intrinsic on x64.
//

static inline
uint64_t
ExtractBits64_C(
    uint64_t value,
    uint64_t mask
    )
{
    uint64_t result = 0;
    uint64_t bit_position = 0;

    while (mask != 0) {
        if (mask & 1) {
            result |= (value & 1) << bit_position;
            bit_position++;
        }

        mask >>= 1;
        value >>= 1;
    }

    return result;
}

//
// Software CRC32C (Castagnoli) fallback.
//

static inline
uint32_t
Crc32cU8_C(
    uint32_t crc,
    uint8_t value
    )
{
    uint32_t i;
    crc ^= value;
    for (i = 0; i < 8; i++) {
        uint32_t mask = (uint32_t)-(int)(crc & 1);
        crc = (crc >> 1) ^ (0x82F63B78u & mask);
    }
    return crc;
}

static inline
uint32_t
Crc32u32_C(
    uint32_t crc,
    uint32_t value
    )
{
    crc = Crc32cU8_C(crc, (uint8_t)value);
    crc = Crc32cU8_C(crc, (uint8_t)(value >> 8));
    crc = Crc32cU8_C(crc, (uint8_t)(value >> 16));
    crc = Crc32cU8_C(crc, (uint8_t)(value >> 24));
    return crc;
}

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
#if defined(__clang__) || defined(__GNUC__)
#   if defined(__aarch64__) || defined(__arm64__)

#define RotateRight32 RotateRight32_C
#define RotateLeft32  RotateLeft32_C
#define RotateRight64 RotateRight64_C
#define RotateLeft64  RotateLeft64_C
#define ExtractBits64 ExtractBits64_C
#define Crc32u32      Crc32u32_C

#   else // !arm64
#       include <x86intrin.h>

#define RotateRight32 _rotr
#define RotateLeft32  _rotl
#define RotateRight64 _rotr64
#define RotateLeft64  _rotl64
#define ExtractBits64 _pext_u64
#define Crc32u32      _mm_crc32_u32

#   endif // arm64
#else
#error Unrecognized compiler.
#endif

#ifndef PH_UNITY
#include <no_sal2.h>
#endif
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

#define RotateRight32 _rotr
#define RotateLeft32  _rotl
#define RotateRight64 _rotr64
#define RotateLeft64  _rotl64
#define ExtractBits64 _pext_u64
#define Crc32u32      _mm_crc32_u32

//
// Define start/end markers for IACA.
//

#define IACA_VC_START() __writegsbyte(111, 111)
#define IACA_VC_END()   __writegsbyte(222, 222)

#else
#define IACA_VC_START()
#define IACA_VC_END()
#endif

#ifdef __APPLE__
#ifdef __arm64__
#include <time.h>
#define __rdtsc() clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
#endif
#endif

#ifdef __linux__
#if defined(__aarch64__) || defined(__arm64__)
#include <time.h>
static inline uint64_t
__rdtsc(void)
{
    struct timespec ts;
#if defined(CLOCK_MONOTONIC_RAW)
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return ((uint64_t)ts.tv_sec * 1000000000ull) + (uint64_t)ts.tv_nsec;
}
#endif
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

    N.B. If the given key did not appear in the original set from which the
         hash table was created, the behavior of this routine is undefined.
         (In practice, the key will hash to either an existing key's location
         or an empty slot, so there is potential for returning a non-unique
         index.)

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

    N.B. If the given key did not appear in the original set from which the
         hash table was created, the behavior of this routine is undefined.
         (In practice, the value returned will be the value for some other
         key in the table that hashes to the same location, or, potentially,
         an empty slot in the table.)

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

    N.B. If the given key did not appear in the original set from which the
         hash table was created, the behavior of this routine is undefined.
         (In practice, the key will hash to either an existing key's location
         or an empty slot, so there is potential to corrupt the table in the
         sense that previously inserted values will be trampled over.)

Arguments:

    Key - Supplies the key for which the value will be inserted.

    Value - Supplies the value to insert.

Return Value:

    Previous value at the relevant table location prior to this insertion.

--*/
typedef COMPILED_PERFECT_HASH_TABLE_INSERT
      *PCOMPILED_PERFECT_HASH_TABLE_INSERT;


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

    N.B. If the given key did not appear in the original set from which the
         hash table was created, the behavior of this routine is undefined.
         (In practice, the key will hash to either an existing key's location
         or an empty slot, so there is potential to corrupt the table in the
         sense that a previously inserted value for an unrelated, valid key
         will be cleared.)

Arguments:

    Key - Supplies the key to delete.

Return Value:

    Previous value at the given key's location prior to deletion.  If no prior
    insertion, the previous value is guaranteed to be 0.

--*/
typedef COMPILED_PERFECT_HASH_TABLE_DELETE *PCOMPILED_PERFECT_HASH_TABLE_DELETE;

typedef
CPHAPI
CPHVALUE
(CPHCALLTYPE COMPILED_PERFECT_HASH_TABLE_INTERLOCKED_INCREMENT)(
    _In_ CPHKEY Key
    );
/*++

Routine Description:

    Increments the value associated with a key.

Arguments:

    Key - Supplies the key to increment.

Return Value:

    Previous value.

--*/
typedef COMPILED_PERFECT_HASH_TABLE_INTERLOCKED_INCREMENT
      *PCOMPILED_PERFECT_HASH_TABLE_INTERLOCKED_INCREMENT;

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

#endif // _COMPILED_PERFECT_HASH_H_

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
