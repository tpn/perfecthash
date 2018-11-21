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
_Must_inspect_result_
_Success_(return >= 0)
_Requires_lock_not_held_(File->Lock)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_FILE_CLOSE)(
    _In_ PPERFECT_HASH_FILE File,
    _In_opt_ PLARGE_INTEGER NewEndOfFile
    );
typedef PERFECT_HASH_FILE_CLOSE *PPERFECT_HASH_FILE_CLOSE;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(File->Lock)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_FILE_EXTEND)(
    _In_ PPERFECT_HASH_FILE File,
    _In_ PLARGE_INTEGER NewEndOfFile
    );
typedef PERFECT_HASH_FILE_EXTEND *PPERFECT_HASH_FILE_EXTEND;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(File->Lock)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_FILE_TRUNCATE)(
    _In_ PPERFECT_HASH_FILE File,
    _In_opt_ PLARGE_INTEGER NewEndOfFile
    );
typedef PERFECT_HASH_FILE_TRUNCATE *PPERFECT_HASH_FILE_TRUNCATE;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(File->Lock)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_FILE_MAP)(
    _In_ PPERFECT_HASH_FILE File
    );
typedef PERFECT_HASH_FILE_MAP *PPERFECT_HASH_FILE_MAP;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(File->Lock)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_FILE_UNMAP)(
    _In_ PPERFECT_HASH_FILE File
    );
typedef PERFECT_HASH_FILE_UNMAP *PPERFECT_HASH_FILE_UNMAP;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_lock_not_held_(File->Lock)
_Requires_lock_not_held_(NewPath->Lock)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_FILE_SCHEDULE_RENAME)(
    _In_ PPERFECT_HASH_FILE File,
    _In_ PPERFECT_HASH_PATH NewPath
    );
typedef PERFECT_HASH_FILE_SCHEDULE_RENAME *PPERFECT_HASH_FILE_SCHEDULE_RENAME;

typedef
_Must_inspect_result_
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
        // The file is in the process of being extended.
        //

        ULONG IsBeingExtended:1;

        //
        // The file is mapped (Map() was successful and there have been no
        // subsequent Unmap() calls.)
        //

        ULONG IsMapped:1;

        //
        // The file is unmapped (Unmap() was successfully called against a file
        // that was previously successfully mapped via Map().)
        //

        ULONG IsUnmapped:1;

        //
        // Unused bits.
        //

        ULONG Unused:26;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_FILE_STATE;
C_ASSERT(sizeof(PERFECT_HASH_FILE_STATE) == sizeof(ULONG));
typedef PERFECT_HASH_FILE_STATE *PPERFECT_HASH_FILE_STATE;

//
// Helper macros.
//

#define IsFileOpen(File) (File->State.IsOpen)
#define IsFileClosed(File) (File->State.IsClosed)
#define IsFileMapped(File) (File->State.IsMapped)
#define IsFileUnmapped(File) (File->State.IsUnmapped)
#define FileNeverOpened(File) (!IsFileOpen(File) && !IsFileClosed(File))
#define IsFileReadOnly(File) (File->State.IsReadOnly)
#define IsFileBeingExtended(File) (File->State.IsBeingExtended)
#define IsViewMapped(File) (File->MappedAddress != NULL)
#define IsViewCreated(File) (File->MappingHandle != NULL)
#define WantsLargePages(File) (!File->Flags.DoesNotWantLargePages)
#define IsFileRenameScheduled(File) (File->RenamePath)
#define IsFileStream(File) (File->Path->StreamName.Buffer != NULL)
#define WasFileLoaded(File) (File->Flags.Loaded == TRUE)
#define WasFileCreated(File) (File->Flags.Created == TRUE)

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

#define SetFileMapped(File)        \
    File->State.IsMapped = TRUE;   \
    File->State.IsUnmapped = FALSE

#define SetFileUnmapped(File)     \
    File->State.IsMapped = FALSE; \
    File->State.IsUnmapped = TRUE

#define NumberOfPagesForFile(File) \
    (ULONG)BYTES_TO_PAGES(File->FileInfo.AllocationSize.QuadPart)

#define NumberOfPagesForPendingEndOfFile(File) \
    (ULONG)BYTES_TO_PAGES(File->PendingEndOfFile.QuadPart)

#define GetActivePath(File) (                              \
    (File)->RenamePath ? (File)->RenamePath : (File)->Path \
)

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
    // Optionally capture the file ID associated with this file.
    //

    FILE_ID FileId;

    //
    // Pointer to the path instance from which we were loaded or created.
    //

    volatile PPERFECT_HASH_PATH Path;

    //
    // Pointer to the new path name requested by a call to ScheduleRename().
    //

    volatile PPERFECT_HASH_PATH RenamePath;

    //
    // Pointer to our parent directory, if applicable.
    //

    PPERFECT_HASH_DIRECTORY ParentDirectory;

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
    // File info for the current file.  Updated whenever any routine mutates
    // end of file or allocation size.
    //

    FILE_STANDARD_INFO FileInfo;

    //
    // Number of bytes written to BaseAddress if the file is writable.  This
    // is used to adjust the file's end-of-file during Close() if no explicit
    // end-of-file has been provided.
    //

    LARGE_INTEGER NumberOfBytesWritten;

    //
    // If the file was opened via Create(), and the NoTruncate flag was
    // specified, this field represents the initial file size of the file
    // when first opened.  This takes precedence during Close() if a 0 value
    // is passed in (indicating an error).
    //

    LARGE_INTEGER InitialEndOfFile;

    //
    // If there's a pending end-of-file change that needs to be made, that is
    // captured here.  This is set by Close(), and subsequently read by Unmap(),
    // which uses it to determine how many pages to copy back from the large
    // page buffer if one was allocated and the file is not read-only.
    //

    LARGE_INTEGER PendingEndOfFile;

    //
    // Certain files, such as VC Project (.vcxproj) files, have UUIDs associated
    // with them, which will be captured in the following field.
    //

    STRING Uuid;

    //
    // Backing interface.
    //

    PERFECT_HASH_FILE_VTBL Interface;

} PERFECT_HASH_FILE;
typedef PERFECT_HASH_FILE *PPERFECT_HASH_FILE;

#define TryAcquirePerfectHashFileLockExclusive(File) \
    TryAcquireSRWLockExclusive(&File->Lock)

#define AcquirePerfectHashFileLockExclusive(File) \
    AcquireSRWLockExclusive(&File->Lock)

#define ReleasePerfectHashFileLockExclusive(File) \
    ReleaseSRWLockExclusive(&File->Lock)

#define TryAcquirePerfectHashFileLockShared(File) \
    TryAcquireSRWLockShared(&File->Lock)

#define AcquirePerfectHashFileLockShared(File) \
    AcquireSRWLockShared(&File->Lock)

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
