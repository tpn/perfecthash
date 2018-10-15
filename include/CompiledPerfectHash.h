/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

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

#include <sal.h>

#ifndef COMPILED_PERFECT_HASH_DLL_BUILD
#define CPHAPI __declspec(dllimport)
#else
#define CPHAPI __declspec(dllexport)
#endif

//
// Define start/end markers for IACA.
//

#define IACA_VC_START() __writegsbyte(111, 111)
#define IACA_VC_END()   __writegsbyte(222, 222)

//
// Define basic NT types and macros used by this header file.
//

#define CPHCALLTYPE __stdcall
#define FORCEINLINE __forceinline

typedef char BOOLEAN;
typedef unsigned char BYTE;
typedef BYTE *PBYTE;
typedef long LONG;
typedef long long LONGLONG;
typedef unsigned long ULONG;
typedef unsigned long *PULONG;
typedef unsigned long long ULONGLONG;
typedef void *PVOID;

//
// Disable the anonymous union/struct warning.
//

#pragma warning(push)
#pragma warning(disable: 4201 4094)

typedef union _LARGE_INTEGER {
    struct {
        ULONG LowPart;
        LONG HighPart;
    };
    LONGLONG QuadPart;
} LARGE_INTEGER;
typedef LARGE_INTEGER *PLARGE_INTEGER;

typedef union _ULARGE_INTEGER {
    struct {
        ULONG LowPart;
        ULONG HighPart;
    };
    ULONGLONG QuadPart;
} ULARGE_INTEGER;
typedef ULARGE_INTEGER *PULARGE_INTEGER;

//
// Define the main functions exposed by a compiled perfect hash table: index,
// lookup, insert and delete.
//

typedef
CPHAPI
ULONG
(CPHCALLTYPE COMPILED_PERFECT_HASH_TABLE_INDEX)(
    _In_ ULONG Key
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


typedef
CPHAPI
ULONG
(CPHCALLTYPE COMPILED_PERFECT_HASH_TABLE_LOOKUP)(
    _In_ ULONG Key
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
ULONG
(CPHCALLTYPE COMPILED_PERFECT_HASH_TABLE_INSERT)(
    _In_ ULONG Key,
    _In_ ULONG Value
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
ULONG
(CPHCALLTYPE COMPILED_PERFECT_HASH_TABLE_DELETE)(
    _In_ ULONG Key
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
typedef COMPILED_PERFECT_HASH_TABLE_DELETE
      *PCOMPILED_PERFECT_HASH_TABLE_DELETE;

//
// Define the vtbl structure encapsulating the compiled perfect hash table's
// function pointers.
//

typedef struct _COMPILED_PERFECT_HASH_TABLE_VTBL {
    PCOMPILED_PERFECT_HASH_TABLE_INDEX  Index;
    PCOMPILED_PERFECT_HASH_TABLE_LOOKUP Lookup;
    PCOMPILED_PERFECT_HASH_TABLE_INSERT Insert;
    PCOMPILED_PERFECT_HASH_TABLE_DELETE Delete;
} COMPILED_PERFECT_HASH_TABLE_VTBL;
typedef COMPILED_PERFECT_HASH_TABLE_VTBL *PCOMPILED_PERFECT_HASH_TABLE_VTBL;

typedef struct _Struct_size_bytes_(SizeOfStruct) _COMPILED_PERFECT_HASH_TABLE {

    COMPILED_PERFECT_HASH_TABLE_VTBL Vtbl;

    //
    // Size of the structure, in bytes.
    //

    _Field_range_(==, sizeof(struct _COMPILED_PERFECT_HASH_TABLE))
        ULONG SizeOfStruct;

    //
    // Size of an individual key element, in bytes.
    //

    ULONG KeySizeInBytes;

    //
    // Algorithm that was used.
    //

    ULONG AlgorithmId;

    //
    // Hash function that was used.
    //

    ULONG HashFunctionId;

    //
    // Masking type.
    //

    ULONG MaskFunctionId;

    //
    // Padding.
    //

    ULONG Padding;

    //
    // Number of keys in the input set.  This is used to size an appropriate
    // array for storing values.
    //

    ULARGE_INTEGER NumberOfKeys;

    //
    // Final number of elements in the underlying table.  This will vary
    // depending on how the graph was created.  If modulus masking is in use,
    // this will reflect the number of keys (unless a custom table size was
    // requested during creation).  Otherwise, this will be the number of keys
    // rounded up to the next power of 2.  (That is, take the number of keys,
    // round up to a power of 2, then round that up to the next power of 2.)
    //

    ULARGE_INTEGER NumberOfTableElements;

    //
    // Hash and index sizes and masks.
    //

    ULONG HashSize;
    ULONG IndexSize;

    ULONG HashMask;
    ULONG IndexMask;

    //
    // Base addresses of the table data and values array.
    //

    union {
        PVOID DataBaseAddress;
        PULONG Data;
    };

    union {
        PVOID ValuesBaseAddress;
        PULONG Values;
    };

    //
    // Seed data.
    //

    ULONG NumberOfSeeds;

    union {
        ULONG Seed1;
        ULONG FirstSeed;
    };

    ULONG Seed2;
    ULONG Seed3;

    union {
        ULONG Seed4;
        ULONG LastSeed;
    };

    ULONG Padding2;

} COMPILED_PERFECT_HASH_TABLE;
typedef COMPILED_PERFECT_HASH_TABLE *PCOMPILED_PERFECT_HASH_TABLE;

typedef
_Success_(return != 0)
BOOLEAN
(CPHCALLTYPE GET_COMPILED_PERFECT_HASH_TABLE)(
    _In_ _Field_range_(==, sizeof(Table)) ULONG SizeOfTable,
    _Out_writes_bytes_(SizeOfTable) PCOMPILED_PERFECT_HASH_TABLE Table
    );
/*++

Routine Description:

    Obtains a compiled perfect hash table structure for a given module.

Arguments:

    SizeOfTable - Supplies the size, in bytes, of the Table structure.

    Table - Supplies a pointer to a COMPILED_PERFECT_HASH_TABLE structure for
        which the table instance will be copied.

Return Value:

    TRUE on success, FALSE on failure.  The only possible way for this routine
    to fail is if SizeOfTable isn't correct.

--*/
typedef GET_COMPILED_PERFECT_HASH_TABLE *PGET_COMPILED_PERFECT_HASH_TABLE;

//
// Define helper macro for defining functions.
//

#define DEFINE_COMPILED_PERFECT_HASH_ROUTINES(TableName) \
CPHAPI COMPILED_PERFECT_HASH_TABLE_INDEX                 \
    CompiledPerfectHash_##TableName##_Index;             \
                                                         \
CPHAPI COMPILED_PERFECT_HASH_TABLE_LOOKUP                \
    CompiledPerfectHash_##TableName##_Lookup;            \
                                                         \
CPHAPI COMPILED_PERFECT_HASH_TABLE_INSERT                \
    CompiledPerfectHash_##TableName##_Insert;            \
                                                         \
CPHAPI COMPILED_PERFECT_HASH_TABLE_DELETE                \
    CompiledPerfectHash_##TableName##_Delete

//
// Define a helper macro for implementing the Lookup, Insert and Delete routines
// once an Index routine has been implemented.
//

#define DECLARE_COMPILED_PERFECT_HASH_ROUTINES(TableName) \
CPHAPI COMPILED_PERFECT_HASH_TABLE_LOOKUP                 \
    CompiledPerfectHash_##TableName##_Lookup;             \
                                                          \
_Use_decl_annotations_                                    \
ULONG                                                     \
CompiledPerfectHash_##TableName##_Lookup(                 \
    ULONG Key                                             \
    )                                                     \
{                                                         \
    ULONG Index;                                          \
                                                          \
    Index = CompiledPerfectHash_##TableName##_Index(Key); \
    return TableName##_TableValues[Index];                \
}                                                         \
                                                          \
CPHAPI COMPILED_PERFECT_HASH_TABLE_INSERT                 \
    CompiledPerfectHash_##TableName##_Insert;             \
                                                          \
_Use_decl_annotations_                                    \
ULONG                                                     \
CompiledPerfectHash_##TableName##_Insert(                 \
    ULONG Key,                                            \
    ULONG Value                                           \
    )                                                     \
{                                                         \
    ULONG Index;                                          \
    ULONG Previous;                                       \
                                                          \
    Index = CompiledPerfectHash_##TableName##_Index(Key); \
    Previous = TableName##_TableValues[Index];            \
    TableName##_TableValues[Index] = Value;               \
    return Previous;                                      \
}                                                         \
                                                          \
CPHAPI COMPILED_PERFECT_HASH_TABLE_DELETE                 \
    CompiledPerfectHash_##TableName##_Delete;             \
                                                          \
_Use_decl_annotations_                                    \
ULONG                                                     \
CompiledPerfectHash_##TableName##_Delete(                 \
    ULONG Key                                             \
    )                                                     \
{                                                         \
    ULONG Index;                                          \
    ULONG Previous;                                       \
                                                          \
    Index = CompiledPerfectHash_##TableName##_Index(Key); \
    Previous = TableName##_TableValues[Index];            \
    TableName##_TableValues[Index] = 0;                   \
    return Previous;                                      \
}

//
// Inline versions of above.
//

#define DECLARE_COMPILED_PERFECT_HASH_ROUTINES_INLINE(TableName) \
FORCEINLINE                                                      \
ULONG                                                            \
CompiledPerfectHash_##TableName##_LookupInline(                  \
    _In_ ULONG Key                                               \
    )                                                            \
{                                                                \
    ULONG Index;                                                 \
                                                                 \
    Index = CompiledPerfectHash_##TableName##_IndexInline(Key);  \
    return TableName##_TableValues[Index];                       \
}                                                                \
                                                                 \
FORCEINLINE                                                      \
ULONG                                                            \
CompiledPerfectHash_##TableName##_InsertInline(                  \
    ULONG Key,                                                   \
    ULONG Value                                                  \
    )                                                            \
{                                                                \
    ULONG Index;                                                 \
    ULONG Previous;                                              \
                                                                 \
    Index = CompiledPerfectHash_##TableName##_IndexInline(Key);  \
    Previous = TableName##_TableValues[Index];                   \
    TableName##_TableValues[Index] = Value;                      \
    return Previous;                                             \
}                                                                \
                                                                 \
FORCEINLINE                                                      \
ULONG                                                            \
CompiledPerfectHash_##TableName##_DeleteInline(                  \
    ULONG Key                                                    \
    )                                                            \
{                                                                \
    ULONG Index;                                                 \
    ULONG Previous;                                              \
                                                                 \
    Index = CompiledPerfectHash_##TableName##_IndexInline(Key);  \
    Previous = TableName##_TableValues[Index];                   \
    TableName##_TableValues[Index] = 0;                          \
    return Previous;                                             \
}

//
// Helper macros for declaring C hash functions.
//

//
// CRC32Rotate
//

#define DECLARE_CHM01_CRC32ROTATE_AND_INDEX_ROUTINE(    \
    TableName, Seed1, Seed2, HashMask, IndexMask        \
    )                                                   \
CPHAPI                                                  \
ULONG                                                   \
CompiledPerfectHash_##TableName##_Index(                \
    ULONG Key                                           \
    )                                                   \
{                                                       \
    ULONG Index;                                        \
    ULONG Vertex1;                                      \
    ULONG Vertex2;                                      \
    ULONG MaskedLow;                                    \
    ULONG MaskedHigh;                                   \
    ULONGLONG Combined;                                 \
                                                        \
    Vertex1 = _mm_crc32_u32(Seed1, Key);                \
    Vertex2 = _mm_crc32_u32(Seed2, _rotl(Key, 15));     \
                                                        \
    MaskedLow = Vertex1 & HashMask;                     \
    MaskedHigh = Vertex2 & HashMask;                    \
                                                        \
    Vertex1 = TableName##_TableData[MaskedLow];         \
    Vertex2 = TableName##_TableData[MaskedHigh];        \
                                                        \
    Combined = (ULONGLONG)Vertex1 + (ULONGLONG)Vertex2; \
                                                        \
    Index = Combined & IndexMask;                       \
                                                        \
    return Index;                                       \
}

#define DECLARE_CHM01_CRC32ROTATE_AND_INDEX_ROUTINE_INLINE( \
    TableName, Seed1, Seed2, HashMask, IndexMask            \
    )                                                       \
FORCEINLINE                                                 \
ULONG                                                       \
CompiledPerfectHash_##TableName##_IndexInline(              \
    ULONG Key                                               \
    )                                                       \
{                                                           \
    ULONG Index;                                            \
    ULONG Vertex1;                                          \
    ULONG Vertex2;                                          \
    ULONG MaskedLow;                                        \
    ULONG MaskedHigh;                                       \
    ULONGLONG Combined;                                     \
                                                            \
    Vertex1 = _mm_crc32_u32(Seed1, Key);                    \
    Vertex2 = _mm_crc32_u32(Seed2, _rotl(Key, 15));         \
                                                            \
    MaskedLow = Vertex1 & HashMask;                         \
    MaskedHigh = Vertex2 & HashMask;                        \
                                                            \
    Vertex1 = TableName##_TableData[MaskedLow];             \
    Vertex2 = TableName##_TableData[MaskedHigh];            \
                                                            \
    Combined = (ULONGLONG)Vertex1 + (ULONGLONG)Vertex2;     \
                                                            \
    Index = Combined & IndexMask;                           \
                                                            \
    return Index;                                           \
}

//
// Jenkins
//

#define DECLARE_CHM01_JENKINS_AND_INDEX_ROUTINE( \
    TableName, Seed1, Seed2, HashMask, IndexMask \
    )                                            \
CPHAPI                                           \
ULONG                                            \
CompiledPerfectHash_##TableName##_Index(         \
    ULONG Key                                    \
    )                                            \
{                                                \
    ULONG A;                                     \
    ULONG B;                                     \
    ULONG C;                                     \
    ULONG D;                                     \
    ULONG E;                                     \
    ULONG F;                                     \
    PBYTE Byte;                                  \
    ULONG Index;                                 \
    ULONG Vertex1;                               \
    ULONG Vertex2;                               \
    ULONG MaskedLow;                             \
    ULONG MaskedHigh;                            \
    ULONGLONG Combined;                          \
                                                 \
    Byte = (PBYTE)&Key;                          \
                                                 \
    A = B = 0x9e3779b9;                          \
    C = Seed1;                                   \
                                                 \
    A += (((ULONG)Byte[3]) << 24);               \
    A += (((ULONG)Byte[2]) << 16);               \
    A += (((ULONG)Byte[1]) <<  8);               \
    A += ((ULONG)Byte[0]);                       \
                                                 \
    A -= B; A -= C; A ^= (C >> 13);              \
    B -= C; B -= A; B ^= (A <<  8);              \
    C -= A; C -= B; C ^= (B >> 13);              \
    A -= B; A -= C; A ^= (C >> 12);              \
    B -= C; B -= A; B ^= (A << 16);              \
    C -= A; C -= B; C ^= (B >>  5);              \
    A -= B; A -= C; A ^= (C >>  3);              \
    B -= C; B -= A; B ^= (A << 10);              \
    C -= A; C -= B; C ^= (B >> 15);              \
                                                 \
    Vertex1 = C;                                 \
                                                 \
    D = E = 0x9e3779b9;                          \
    F = Seed2;                                   \
                                                 \
    D += (((ULONG)Byte[3]) << 24);               \
    D += (((ULONG)Byte[2]) << 16);               \
    D += (((ULONG)Byte[1]) <<  8);               \
    D += ((ULONG)Byte[0]);                       \
                                                 \
    D -= E; D -= F; D ^= (F >> 13);              \
    E -= F; E -= D; E ^= (D <<  8);              \
    F -= D; F -= E; F ^= (E >> 13);              \
    D -= E; D -= F; D ^= (F >> 12);              \
    E -= F; E -= D; E ^= (D << 16);              \
    F -= D; F -= E; F ^= (E >>  5);              \
    D -= E; D -= F; D ^= (F >>  3);              \
    E -= F; E -= D; E ^= (D << 10);              \
    F -= D; F -= E; F ^= (E >> 15);              \
                                                 \
    Vertex2 = F;                                 \
                                                 \
    MaskedLow = Vertex1 & HashMask;              \
    MaskedHigh = Vertex2 & HashMask;             \
                                                 \
    Vertex1 = TableName##_TableData[MaskedLow];  \
    Vertex2 = TableName##_TableData[MaskedHigh]; \
                                                 \
    Combined = (                                 \
        (ULONGLONG)Vertex1 +                     \
        (ULONGLONG)Vertex2                       \
    );                                           \
                                                 \
    Index = Combined & IndexMask;                \
                                                 \
    return Index;                                \
}

#define DECLARE_CHM01_JENKINS_AND_INDEX_ROUTINE_INLINE( \
    TableName, Seed1, Seed2, HashMask, IndexMask        \
    )                                                   \
FORCEINLINE                                             \
ULONG                                                   \
CompiledPerfectHash_##TableName##_IndexInline(          \
    ULONG Key                                           \
    )                                                   \
{                                                       \
    ULONG A;                                            \
    ULONG B;                                            \
    ULONG C;                                            \
    ULONG D;                                            \
    ULONG E;                                            \
    ULONG F;                                            \
    PBYTE Byte;                                         \
    ULONG Index;                                        \
    ULONG Vertex1;                                      \
    ULONG Vertex2;                                      \
    ULONG MaskedLow;                                    \
    ULONG MaskedHigh;                                   \
    ULONGLONG Combined;                                 \
                                                        \
    Byte = (PBYTE)&Key;                                 \
                                                        \
    A = B = 0x9e3779b9;                                 \
    C = Seed1;                                          \
                                                        \
    A += (((ULONG)Byte[3]) << 24);                      \
    A += (((ULONG)Byte[2]) << 16);                      \
    A += (((ULONG)Byte[1]) <<  8);                      \
    A += ((ULONG)Byte[0]);                              \
                                                        \
    A -= B; A -= C; A ^= (C >> 13);                     \
    B -= C; B -= A; B ^= (A <<  8);                     \
    C -= A; C -= B; C ^= (B >> 13);                     \
    A -= B; A -= C; A ^= (C >> 12);                     \
    B -= C; B -= A; B ^= (A << 16);                     \
    C -= A; C -= B; C ^= (B >>  5);                     \
    A -= B; A -= C; A ^= (C >>  3);                     \
    B -= C; B -= A; B ^= (A << 10);                     \
    C -= A; C -= B; C ^= (B >> 15);                     \
                                                        \
    Vertex1 = C;                                        \
                                                        \
    D = E = 0x9e3779b9;                                 \
    F = Seed2;                                          \
                                                        \
    D += (((ULONG)Byte[3]) << 24);                      \
    D += (((ULONG)Byte[2]) << 16);                      \
    D += (((ULONG)Byte[1]) <<  8);                      \
    D += ((ULONG)Byte[0]);                              \
                                                        \
    D -= E; D -= F; D ^= (F >> 13);                     \
    E -= F; E -= D; E ^= (D <<  8);                     \
    F -= D; F -= E; F ^= (E >> 13);                     \
    D -= E; D -= F; D ^= (F >> 12);                     \
    E -= F; E -= D; E ^= (D << 16);                     \
    F -= D; F -= E; F ^= (E >>  5);                     \
    D -= E; D -= F; D ^= (F >>  3);                     \
    E -= F; E -= D; E ^= (D << 10);                     \
    F -= D; F -= E; F ^= (E >> 15);                     \
                                                        \
    Vertex2 = F;                                        \
                                                        \
    MaskedLow = Vertex1 & HashMask;                     \
    MaskedHigh = Vertex2 & HashMask;                    \
                                                        \
    Vertex1 = TableName##_TableData[MaskedLow];         \
    Vertex2 = TableName##_TableData[MaskedHigh];        \
                                                        \
    Combined = (                                        \
        (ULONGLONG)Vertex1 +                            \
        (ULONGLONG)Vertex2                              \
    );                                                  \
                                                        \
    Index = Combined & IndexMask;                       \
                                                        \
    return Index;                                       \
}

//
// Typedefs of methods for testing and benchmarking.
//

typedef
_Success_(return == 0)
ULONG
(CPHCALLTYPE TEST_COMPILED_PERFECT_HASH_TABLE)(
    _In_opt_ BOOLEAN DebugBreakOnFailure
    );
typedef TEST_COMPILED_PERFECT_HASH_TABLE
      *PTEST_COMPILED_PERFECT_HASH_TABLE;

typedef
ULONG
(CPHCALLTYPE BENCHMARK_INDEX_COMPILED_PERFECT_HASH_TABLE)(
    _In_ ULONG Seconds
    );
typedef BENCHMARK_INDEX_COMPILED_PERFECT_HASH_TABLE
      *PBENCHMARK_INDEX_COMPILED_PERFECT_HASH_TABLE;

typedef
ULONG
(CPHCALLTYPE BENCHMARK_FULL_COMPILED_PERFECT_HASH_TABLE)(
    _In_ ULONG Seconds
    );
typedef BENCHMARK_FULL_COMPILED_PERFECT_HASH_TABLE
      *PBENCHMARK_FULL_COMPILED_PERFECT_HASH_TABLE;

#define DEFINE_TEST_AND_BENCHMARK_COMPILED_PERFECT_HASH_TABLE_ROUTINES(Tbl) \
extern TEST_COMPILED_PERFECT_HASH_TABLE                                     \
    TestCompiledPerfectHashTable_##Tbl;                                     \
                                                                            \
extern BENCHMARK_INDEX_COMPILED_PERFECT_HASH_TABLE                          \
    BenchmarkIndexCompiledPerfectHashTable_##Tbl;                           \
                                                                            \
extern BENCHMARK_FULL_COMPILED_PERFECT_HASH_TABLE                           \
    BenchmarkFullCompiledPerfectHashTable_##Tbl

//
// The following macros are intended to be used by the Test.c, BenchmarkFull.c
// and BenchmarkIndex.c files such that they can generate the proper function
// header (with a bit of macro glue) without having to have the table name
// hardcoded.
//

#define DECLARE_TEST_COMPILED_PERFECT_HASH_TABLE_ROUTINE_HEADER(T) \
TEST_COMPILED_PERFECT_HASH_TABLE                                   \
    TestCompiledPerfectHashTable_##T;                              \
                                                                   \
_Use_decl_annotations_                                             \
ULONG                                                              \
TestCompiledPerfectHashTable_##T##(                                \
    BOOLEAN DebugBreakOnFailure                                    \
    )

#define DECLARE_BENCHMARK_FULL_COMPILED_PERFECT_HASH_TABLE_ROUTINE_HEADER(T) \
BENCHMARK_FULL_COMPILED_PERFECT_HASH_TABLE                                   \
    BenchmarkFullCompiledPerfectHashTable_##T;                               \
                                                                             \
_Use_decl_annotations_                                                       \
ULONG                                                                        \
BenchmarkFullCompiledPerfectHashTable_##T##(                                 \
    ULONG Seconds                                                            \
    )

#define DECLARE_BENCHMARK_INDEX_COMPILED_PERFECT_HASH_TABLE_ROUTINE_HEADER(T) \
BENCHMARK_INDEX_COMPILED_PERFECT_HASH_TABLE                                   \
    BenchmarkIndexCompiledPerfectHashTable_##T;                               \
                                                                              \
_Use_decl_annotations_                                                        \
ULONG                                                                         \
BenchmarkIndexCompiledPerfectHashTable_##T##(                                 \
    ULONG Seconds                                                             \
    )

#pragma warning(pop)

#ifdef __cplusplus
} // extern "C"
#endif

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
