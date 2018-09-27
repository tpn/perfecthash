/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashFile.h

Abstract:

    This is the private header file for the PERFECT_HASH_FILE component of the
    perfect hash table library.  It defines the structure, and function pointer
    typedefs for the initialize and rundown functions.

    This component encapsulates generic file functionality as required by other
    components of the library (keys, tables, etc).

--*/

#pragma once

#include "stdafx.h"

//
// Private vtbl methods.
//

typedef
_Check_return_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(File->Lock)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_FILE_CLOSE)(
    _In_ PPERFECT_HASH_FILE File,
    _In_opt_ PLARGE_INTEGER NewEndOfFile
    );
typedef PERFECT_HASH_FILE_CLOSE *PPERFECT_HASH_FILE_CLOSE;

typedef
_Check_return_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(File->Lock)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_FILE_EXTEND)(
    _In_ PPERFECT_HASH_FILE File,
    _In_ ULARGE_INTEGER NewMappingSize
    );
typedef PERFECT_HASH_FILE_EXTEND *PPERFECT_HASH_FILE_EXTEND;

typedef
_Check_return_
_Success_(return >= 0)
HRESULT
_Requires_exclusive_lock_held_(File->Lock)
(STDAPICALLTYPE PERFECT_HASH_FILE_TRUNCATE)(
    _In_ PPERFECT_HASH_FILE File,
    _In_ LARGE_INTEGER NewEndOfFile
    );
typedef PERFECT_HASH_FILE_TRUNCATE *PPERFECT_HASH_FILE_TRUNCATE;

typedef
_Check_return_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(File->Lock)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_FILE_MAP)(
    _In_ PPERFECT_HASH_FILE File
    );
typedef PERFECT_HASH_FILE_MAP *PPERFECT_HASH_FILE_MAP;

typedef
_Check_return_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(File->Lock)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_FILE_UNMAP)(
    _In_ PPERFECT_HASH_FILE File
    );
typedef PERFECT_HASH_FILE_UNMAP *PPERFECT_HASH_FILE_UNMAP;

typedef
_Check_return_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(File->Lock)
_Requires_lock_not_held_(NewPath->Lock)
_Acquires_exclusive_lock_(NewPath->Lock)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_FILE_SCHEDULE_RENAME)(
    _In_ PPERFECT_HASH_FILE File,
    _In_ PPERFECT_HASH_PATH NewPath
    );
typedef PERFECT_HASH_FILE_SCHEDULE_RENAME *PPERFECT_HASH_FILE_SCHEDULE_RENAME;

typedef
_Check_return_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(File->Lock)
_Pre_satisfies_(
    File->RenamePath != NULL &&
    File->State.IsOpen == FALSE &&
    File->State.IsClosed == TRUE
)
_Post_satisfies_(File->RenamePath == NULL)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_FILE_DO_RENAME)(
    _In_ PPERFECT_HASH_FILE File
    );
typedef PERFECT_HASH_FILE_DO_RENAME *PPERFECT_HASH_FILE_DO_RENAME;

//
// Define our private PERFECT_HASH_FILE_VTBL structure.
//

typedef struct _PERFECT_HASH_FILE_VTBL {
    DECLARE_COMPONENT_VTBL_HEADER(PERFECT_HASH_FILE);
    PPERFECT_HASH_FILE_LOAD Load;
    PPERFECT_HASH_FILE_CREATE Create;
    PPERFECT_HASH_FILE_GET_FLAGS GetFlags;
    PPERFECT_HASH_FILE_GET_PATH GetPath;
    PPERFECT_HASH_FILE_GET_RESOURCES GetResources;

    //
    // Begin private methods.
    //

    PPERFECT_HASH_FILE_CLOSE Close;
    PPERFECT_HASH_FILE_EXTEND Extend;
    PPERFECT_HASH_FILE_TRUNCATE Truncate;
    PPERFECT_HASH_FILE_MAP Map;
    PPERFECT_HASH_FILE_UNMAP Unmap;
    PPERFECT_HASH_FILE_SCHEDULE_RENAME ScheduleRename;
    PPERFECT_HASH_FILE_DO_RENAME DoRename;
} PERFECT_HASH_FILE_VTBL;
typedef PERFECT_HASH_FILE_VTBL *PPERFECT_HASH_FILE_VTBL;

//
// Define the PERFECT_HASH_FILE_STATE structure.
//

typedef union _PERFECT_HASH_FILE_STATE {
    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // File is open.
        //

        ULONG IsOpen:1;

        //
        // File was open, but has been subsequently closed.
        //

        ULONG IsClosed:1;

        //
        // File is read-only.  This will only ever be set if the file is open.
        // (That is, it is cleared when the file is closed, regardless of
        // whether or not the file *was* read-only when open.)
        //

        ULONG IsReadOnly:1;

        //
        // Unused bits.
        //

        ULONG Unused:29;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_FILE_STATE;
C_ASSERT(sizeof(PERFECT_HASH_FILE_STATE) == sizeof(ULONG));
typedef PERFECT_HASH_FILE_STATE *PPERFECT_HASH_FILE_STATE;

//
// Helper macros for discerning file state.
//

#define IsFileOpen(File) (File->State.IsOpen)
#define IsFileClosed(File) (File->State.IsClosed)
#define FileNeverOpened(File) (!IsFileOpen(File) && !IsFileClosed(File))
#define IsFileReadOnly(File) (File->State.IsReadOnly)
#define IsViewMapped(File) (File->MappedAddress != NULL)
#define IsViewCreated(File) (File->MappingHandle != NULL)
#define WantsLargePages(File) (!File->Flags.DoesNotWantLargePages)
#define IsRenameScheduled(File) (File->RenamePath)

#define SetFileOpened(File)      \
    File->State.IsOpen = TRUE;   \
    File->State.IsClosed = FALSE

#define SetFileClosed(File)     \
    File->State.IsOpen = FALSE; \
    File->State.IsClosed = TRUE

#define SetFileLoaded(File)     \
    File->Flags.Loaded = TRUE;  \
    File->Flags.Created = FALSE

#define SetFileCreated(File)    \
    File->Flags.Loaded = FALSE; \
    File->Flags.Created = TRUE

//
// Define the PERFECT_HASH_FILE structure.
//

typedef struct _Struct_size_bytes_(SizeOfStruct) _PERFECT_HASH_FILE {

    COMMON_COMPONENT_HEADER(PERFECT_HASH_FILE);

    //
    // Flags specified to load or create methods.
    //

    PERFECT_HASH_FILE_LOAD_FLAGS FileLoadFlags;
    PERFECT_HASH_FILE_CREATE_FLAGS FileCreateFlags;

    //
    // System allocation granularity.
    //

    ULONG AllocationGranularity;

    //
    // Align up to an 8-byte boundary.
    //

    ULONG Padding;

    //
    // Pointer to the path instance from which we were loaded or created.
    //

    volatile PPERFECT_HASH_PATH Path;

    //
    // Pointer to the new path name requested by a call to ScheduleRename().
    //

    volatile PPERFECT_HASH_PATH RenamePath;

    //
    // Handle to the underlying file.
    //

    HANDLE FileHandle;

    //
    // Handle to the memory mapping for the file.
    //

    HANDLE MappingHandle;

    //
    // Base address of the memory map.
    //

    PVOID BaseAddress;

    //
    // If we were able to allocate a large page buffer of sufficient size,
    // BaseAddress above will point to it, and the following variable will
    // capture the original mapped address.
    //

    PVOID MappedAddress;

    //
    // Size of the memory map.
    //

    ULARGE_INTEGER MappingSize;

    //
    // File info for the current file.  Updated whenever any routine mutates
    // end of file or allocation size.
    //

    FILE_STANDARD_INFO FileInfo;

    //
    // Number of bytes written to BaseAddress if the file is writable.  This
    // is used to adjust the file's end-of-file during rundown.
    //

    LARGE_INTEGER NumberOfBytesWritten;

    //
    // Backing interface.
    //

    PERFECT_HASH_FILE_VTBL Interface;

} PERFECT_HASH_FILE;
typedef PERFECT_HASH_FILE *PPERFECT_HASH_FILE;

#define TryAcquirePerfectHashFileLockExclusive(File) \
    TryAcquireSRWLockExclusive(&File->Lock)

#define ReleasePerfectHashFileLockExclusive(File) \
    ReleaseSRWLockExclusive(&File->Lock)

#define TryAcquirePerfectHashFileLockShared(File) \
    TryAcquireSRWLockShared(&File->Lock)

#define ReleasePerfectHashFileLockShared(File) \
    ReleaseSRWLockShared(&File->Lock)

//
// Private methods.
//

typedef
HRESULT
(NTAPI PERFECT_HASH_FILE_INITIALIZE)(
    _In_ PPERFECT_HASH_FILE File
    );
typedef PERFECT_HASH_FILE_INITIALIZE
      *PPERFECT_HASH_FILE_INITIALIZE;

typedef
VOID
(NTAPI PERFECT_HASH_FILE_RUNDOWN)(
    _In_ _Post_ptr_invalid_ PPERFECT_HASH_FILE File
    );
typedef PERFECT_HASH_FILE_RUNDOWN
      *PPERFECT_HASH_FILE_RUNDOWN;

extern PERFECT_HASH_FILE_INITIALIZE PerfectHashFileInitialize;
extern PERFECT_HASH_FILE_RUNDOWN PerfectHashFileRundown;
extern PERFECT_HASH_FILE_LOAD PerfectHashFileLoad;
extern PERFECT_HASH_FILE_CREATE PerfectHashFileCreate;
extern PERFECT_HASH_FILE_GET_FLAGS PerfectHashFileGetFlags;
extern PERFECT_HASH_FILE_GET_PATH PerfectHashFileGetPath;
extern PERFECT_HASH_FILE_GET_RESOURCES PerfectHashFileGetResources;
extern PERFECT_HASH_FILE_CLOSE PerfectHashFileClose;
extern PERFECT_HASH_FILE_EXTEND PerfectHashFileExtend;
extern PERFECT_HASH_FILE_TRUNCATE PerfectHashFileTruncate;
extern PERFECT_HASH_FILE_MAP PerfectHashFileMap;
extern PERFECT_HASH_FILE_UNMAP PerfectHashFileUnmap;
extern PERFECT_HASH_FILE_SCHEDULE_RENAME PerfectHashFileScheduleRename;
extern PERFECT_HASH_FILE_DO_RENAME PerfectHashFileDoRename;

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
