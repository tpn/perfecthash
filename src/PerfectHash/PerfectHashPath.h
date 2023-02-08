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

        ULONG IsSet:1;

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
// Helper macros for discerning path state.
//

#define IsPathSet(Path) (Path->State.IsSet)

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
        // When set, disables any char replacement logic.
        //

        ULONG DisableCharReplacement:1;

        //
        // Unused bits.
        //

        ULONG Unused:30;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_PATH_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_PATH_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_PATH_FLAGS *PPERFECT_HASH_PATH_FLAGS;

#define IsBaseNameValidCIdentifier(Path) \
    ((Path)->Flags.BaseNameIsValidCIdentifier)

#define IsCharReplacementDisabled(Path) ((Path)->Flags.DisableCharReplacement)

#define ClearPathState(Path) ((Path)->State.AsULong = 0)
#define ClearPathFlags(Path) ((Path)->Flags.AsULong = 0)

FORCEINLINE
BOOLEAN
IsReplaceableBaseNameChar(
    WCHAR Wide
    )
{
    BOOLEAN Replace;

    Replace = (
        Wide == L'-' ||
        Wide == L' ' ||
        Wide == L'.' ||
        Wide == L',' ||
        Wide == L'#' ||
        Wide == L'^' ||
        Wide == L'!' ||
        Wide == L'~' ||
        Wide == L'%' ||
        Wide == L'@' ||
        Wide == L'*'
    );

    return Replace;
}

//
// Helper routine for determining if a wide string starts with \\?\.
//

FORCEINLINE
BOOLEAN
IsDevicePathInDrivePathFormat(
    _In_ PCWSZ Start,
    _In_ PCWSZ End
    )
{
#ifdef PH_WINDOWS
    PCWSZ Char;
    BOOLEAN IsDrivePath;

    //
    // Check for device path format where the drive is explicit,
    // e.g. \\?\C:\Temp.
    //

    Char = Start;

    if ((ULONG_PTR)(Char+4) > (ULONG_PTR)End) {
        return FALSE;
    }

    IsDrivePath = (
        *Char++ == L'\\' &&
        *Char++ == L'\\' &&
        *Char++ == L'?'  &&
        *Char++ == L'\\'
    );

    return IsDrivePath;
#endif
    return FALSE;
}

//
// Define the PERFECT_HASH_PATH structure.
//

typedef struct _Struct_size_bytes_(SizeOfStruct) _PERFECT_HASH_PATH {

    COMMON_COMPONENT_HEADER(PERFECT_HASH_PATH);

    //
    // Inline the parts structure for convenience.
    //

    _Guarded_by_(Path->Lock)
    union {

        PERFECT_HASH_PATH_PARTS Parts;

        struct {
            union {
                UNICODE_STRING FullPath;
                UNICODE_STRING FirstPart;
            };
            UNICODE_STRING Drive;
            UNICODE_STRING Directory;
            UNICODE_STRING BaseName;
            UNICODE_STRING FileName;
            UNICODE_STRING Extension;
            UNICODE_STRING StreamName;
            STRING BaseNameA;
            STRING TableNameA;
            STRING BaseNameUpperA;
            union {
                STRING TableNameUpperA;
                STRING LastPart;
            };
        };
    };

    //
    // The offset of the additional suffix added to the keys file name to
    // create the table name, relative to the BaseNameA and BaseNameUpperA.
    // This is used by the PERFECT_HASH_TABLE component to capture where
    // the algorithm name (that has been automatically appended) starts
    // within a base name/table name.
    //
    // N.B. This field is not used by the path component; downstream
    //      components are free to use it for whatever they want.
    //

    USHORT AdditionalSuffixAOffset;

    USHORT Padding[3];

    //
    // Backing interface.
    //

    PERFECT_HASH_PATH_VTBL Interface;

} PERFECT_HASH_PATH;
typedef PERFECT_HASH_PATH *PPERFECT_HASH_PATH;

//
// Verify we've manually inlined the structure correctly.  If compilation fails
// on this next line, verify the inline representation matches the structure
// definition of PERFECT_HASH_PATH_PARTS.
//

C_ASSERT(
    RTL_FIELD_SIZE(PERFECT_HASH_PATH, Parts) == ((
        FIELD_OFFSET(PERFECT_HASH_PATH, LastPart) -
        FIELD_OFFSET(PERFECT_HASH_PATH, FirstPart)
    ) + RTL_FIELD_SIZE(PERFECT_HASH_PATH, LastPart))
);

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
