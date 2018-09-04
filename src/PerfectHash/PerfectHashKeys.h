/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashKeys.h

Abstract:

    This is the private header file for the PERFECT_HASH_KEYS
    component of the perfect hash table library.  It defines the structure,
    and function pointer typedefs for the initialize and rundown functions.

--*/

#pragma once

#include "stdafx.h"

//
// Define the PERFECT_HASH_KEYS_FLAGS structure.
//

typedef union _PERFECT_HASH_KEYS_STATE {
    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, indicates a keys file has been loaded successfully.
        //

        ULONG Loaded:1;

        //
        // Unused bits.
        //

        ULONG Unused:31;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_KEYS_STATE;
C_ASSERT(sizeof(PERFECT_HASH_KEYS_STATE) == sizeof(ULONG));
typedef PERFECT_HASH_KEYS_STATE *PPERFECT_HASH_KEYS_STATE;

typedef union _PERFECT_HASH_KEYS_FLAGS {
    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, indicates the keys were mapped using large pages.
        //

        ULONG MappedWithLargePages:1;

        //
        // When set, indicates the keys are a sequential linear array of
        // values.
        //

        ULONG Linear:1;

        //
        // Unused bits.
        //

        ULONG Unused:30;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_KEYS_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_KEYS_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_KEYS_FLAGS *PPERFECT_HASH_KEYS_FLAGS;

//
// Define the PERFECT_HASH_KEYS_STATS structure.
//

typedef struct _PERFECT_HASH_KEYS_STATS {

    PERFECT_HASH_KEYS_BITMAP KeysBitmap;

    ULONG MinValue;
    ULONG MaxValue;

    ULONG BitCount[32];
    ULONG PopCount[32];

} PERFECT_HASH_KEYS_STATS;
typedef PERFECT_HASH_KEYS_STATS *PPERFECT_HASH_KEYS_STATS;

//
// Define the PERFECT_HASH_KEYS structure.
//

typedef struct _Struct_size_bytes_(SizeOfStruct) _PERFECT_HASH_KEYS {

    COMMON_COMPONENT_HEADER(PERFECT_HASH_KEYS);

    //
    // Slim read/write lock guarding the structure.
    //

    SRWLOCK Lock;

    //
    // Pointer to an initialized RTL structure.
    //

    PRTL Rtl;

    //
    // Pointer to an initialized ALLOCATOR structure.
    //

    PALLOCATOR Allocator;

    //
    // Number of keys in the mapping.
    //

    ULARGE_INTEGER NumberOfElements;

    //
    // Handle to the underlying keys file.
    //

    HANDLE FileHandle;

    //
    // Handle to the memory mapping for the keys file.
    //

    HANDLE MappingHandle;

    //
    // Base address of the memory map.
    //

    union {
        PVOID BaseAddress;
        PULONG Keys;
    };

    //
    // If we were able to allocate a large page buffer of sufficient size,
    // BaseAddress above will point to it, and the following variable will
    // capture the original mapped address.
    //

    PVOID MappedAddress;

    //
    // Fully-qualified, NULL-terminated path of the source keys file.
    //

    UNICODE_STRING Path;

    //
    // Capture simple statistics about the keys that were loaded.
    //

    PERFECT_HASH_KEYS_STATS Stats;

    //
    // Backing interface.
    //

    PERFECT_HASH_KEYS_VTBL Interface;

} PERFECT_HASH_KEYS;
typedef PERFECT_HASH_KEYS *PPERFECT_HASH_KEYS;

#define TryAcquirePerfectHashKeysLockExclusive(Keys) \
    TryAcquireSRWLockExclusive(&Keys->Lock)

#define ReleasePerfectHashKeysLockExclusive(Keys) \
    ReleaseSRWLockExclusive(&Keys->Lock)

typedef
HRESULT
(NTAPI PERFECT_HASH_KEYS_INITIALIZE)(
    _In_ PPERFECT_HASH_KEYS Keys
    );
typedef PERFECT_HASH_KEYS_INITIALIZE
      *PPERFECT_HASH_KEYS_INITIALIZE;

typedef
VOID
(NTAPI PERFECT_HASH_KEYS_RUNDOWN)(
    _In_ _Post_ptr_invalid_ PPERFECT_HASH_KEYS Keys
    );
typedef PERFECT_HASH_KEYS_RUNDOWN
      *PPERFECT_HASH_KEYS_RUNDOWN;

typedef
HRESULT
(NTAPI PERFECT_HASH_KEYS_LOAD_STATS)(
    _In_ PPERFECT_HASH_KEYS Keys
    );
typedef PERFECT_HASH_KEYS_LOAD_STATS
      *PPERFECT_HASH_KEYS_LOAD_STATS;


extern PERFECT_HASH_KEYS_INITIALIZE PerfectHashKeysInitialize;
extern PERFECT_HASH_KEYS_RUNDOWN PerfectHashKeysRundown;
extern PERFECT_HASH_KEYS_LOAD_STATS PerfectHashKeysLoadStats;
extern PERFECT_HASH_KEYS_LOAD PerfectHashKeysLoad;
extern PERFECT_HASH_KEYS_GET_BITMAP PerfectHashKeysGetBitmap;

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
