/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashPath.h

Abstract:

    This is the private header file for the PERFECT_HASH_PATH component of the
    perfect hash table library.  It defines the structure, and function pointer
    typedefs for the initialize and rundown functions.

    This component encapsulates generic path functionality as required by other
    components of the library.

--*/

#pragma once

#include "stdafx.h"

//
// Define private vtbl method(s).
//

typedef
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Path->Lock)
HRESULT
(NTAPI PERFECT_HASH_PATH_EXTRACT_PARTS)(
    _In_ PPERFECT_HASH_PATH Path
    );
typedef PERFECT_HASH_PATH_EXTRACT_PARTS *PPERFECT_HASH_PATH_EXTRACT_PARTS;

//
// Define the private vtbl.
//
typedef struct _PERFECT_HASH_PATH_VTBL {
    DECLARE_COMPONENT_VTBL_HEADER(PERFECT_HASH_PATH);
    PPERFECT_HASH_PATH_COPY Copy;
    PPERFECT_HASH_PATH_CREATE Create;
    PPERFECT_HASH_PATH_RESET Reset;
    PPERFECT_HASH_PATH_GET_PARTS GetParts;

    //
    // Begin private methods.
    //

    PPERFECT_HASH_PATH_EXTRACT_PARTS ExtractParts;

} PERFECT_HASH_PATH_VTBL;
typedef PERFECT_HASH_PATH_VTBL *PPERFECT_HASH_PATH_VTBL;

//
// Define the PERFECT_HASH_PATH_STATE structure.
//

typedef union _PERFECT_HASH_PATH_STATE {
    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, a path has been loaded via Copy() or Create().  The
        // instance must be reset via Reset() before a subsequent Copy()
        // or Create() can be dispatched.
        //

        ULONG PathSet:1;

        //
        // Unused bits.
        //

        ULONG Unused:31;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_PATH_STATE;
C_ASSERT(sizeof(PERFECT_HASH_PATH_STATE) == sizeof(ULONG));
typedef PERFECT_HASH_PATH_STATE *PPERFECT_HASH_PATH_STATE;

//
// Helper macros for discerning file state.
//

#define IsPathSet(Path) (Path->State.PathSet)

//
// Define the PERFECT_HASH_PATH_FLAGS structure.
//

typedef union _PERFECT_HASH_PATH_FLAGS {
    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, indicates a path's base name is a valid C identifer.
        //

        ULONG BaseNameIsValidCIdentifier:1;

        //
        // Unused bits.
        //

        ULONG Unused:31;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_PATH_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_PATH_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_PATH_FLAGS *PPERFECT_HASH_PATH_FLAGS;

#define IsBaseNameValidCIdentifier(Path) \
    (Path->Flags.IsBaseNameValidCIdentifier)

#define ClearPathState(Path) (Path->State.AsULong = 0)
#define ClearPathFlags(Path) (Path->Flags.AsULong = 0)

//
// Define the PERFECT_HASH_PATH structure.
//

typedef struct _Struct_size_bytes_(SizeOfStruct) _PERFECT_HASH_PATH {

    COMMON_COMPONENT_HEADER(PERFECT_HASH_PATH);

    //
    // We inline the parts structure for convenience.
    //

    _Guarded_by_(Path->Lock)
    union {

        PERFECT_HASH_PATH_PARTS Parts;

        struct {
            UNICODE_STRING FullPath;
            UNICODE_STRING Drive;
            UNICODE_STRING Directory;
            UNICODE_STRING BaseName;
            STRING BaseNameA;
            UNICODE_STRING FileName;
            UNICODE_STRING Extension;
            UNICODE_STRING StreamName;
        };
    };


    //
    // Backing interface.
    //

    PERFECT_HASH_PATH_VTBL Interface;

} PERFECT_HASH_PATH;
typedef PERFECT_HASH_PATH *PPERFECT_HASH_PATH;

//
// Private non-vtbl methods.
//

typedef
HRESULT
(NTAPI PERFECT_HASH_PATH_INITIALIZE)(
    _In_ PPERFECT_HASH_PATH Path
    );
typedef PERFECT_HASH_PATH_INITIALIZE *PPERFECT_HASH_PATH_INITIALIZE;

typedef
VOID
(NTAPI PERFECT_HASH_PATH_RUNDOWN)(
    _In_ _Post_ptr_invalid_ PPERFECT_HASH_PATH Path
    );
typedef PERFECT_HASH_PATH_RUNDOWN *PPERFECT_HASH_PATH_RUNDOWN;

extern PERFECT_HASH_PATH_INITIALIZE PerfectHashPathInitialize;
extern PERFECT_HASH_PATH_RUNDOWN PerfectHashPathRundown;
extern PERFECT_HASH_PATH_COPY PerfectHashPathCopy;
extern PERFECT_HASH_PATH_CREATE PerfectHashPathCreate;
extern PERFECT_HASH_PATH_GET_PARTS PerfectHashPathGetParts;
extern PERFECT_HASH_PATH_RESET PerfectHashPathReset;
extern PERFECT_HASH_PATH_EXTRACT_PARTS PerfectHashPathExtractParts;

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
