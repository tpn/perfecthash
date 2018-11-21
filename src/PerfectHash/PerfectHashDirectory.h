/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashDirectory.h

Abstract:

    This is the private header file for the PERFECT_HASH_DIRECTORY component of the
    perfect hash table library.  It defines the structure, private vtbl and
    function pointer typedefs for the initialize and rundown functions.

    This component encapsulates generic direcotry functionality as required by
    other components of the library (keys, tables, etc).

--*/

#pragma once

#include "stdafx.h"

//
// Private vtbl methods.
//

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_lock_not_held_(Directory->Lock)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_DIRECTORY_CLOSE)(
    _In_ PPERFECT_HASH_DIRECTORY Directory
    );
typedef PERFECT_HASH_DIRECTORY_CLOSE *PPERFECT_HASH_DIRECTORY_CLOSE;

typedef
_Check_return_
_Success_(return >= 0)
_Requires_lock_not_held_(Directory->Lock)
_Requires_lock_not_held_(NewPath->Lock)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_DIRECTORY_SCHEDULE_RENAME)(
    _In_ PPERFECT_HASH_DIRECTORY Directory,
    _In_ PPERFECT_HASH_PATH NewPath
    );
typedef PERFECT_HASH_DIRECTORY_SCHEDULE_RENAME *PPERFECT_HASH_DIRECTORY_SCHEDULE_RENAME;

typedef
_Check_return_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Directory->Lock)
_Pre_satisfies_(
    Directory->RenamePath != NULL &&
    Directory->State.IsSet == FALSE &&
    Directory->State.IsClosed == TRUE
)
_Post_satisfies_(Directory->RenamePath == NULL)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_DIRECTORY_DO_RENAME)(
    _In_ PPERFECT_HASH_DIRECTORY Directory
    );
typedef PERFECT_HASH_DIRECTORY_DO_RENAME *PPERFECT_HASH_DIRECTORY_DO_RENAME;

typedef
_Check_return_
_Success_(return >= 0)
_Requires_lock_not_held_(Directory->Lock)
_Requires_exclusive_lock_held_(File->Lock)
_Pre_satisfies_(File->ParentDirectory == NULL)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_DIRECTORY_ADD_FILE)(
    _In_ PPERFECT_HASH_DIRECTORY Directory,
    _In_ PPERFECT_HASH_FILE File
    );
typedef PERFECT_HASH_DIRECTORY_ADD_FILE *PPERFECT_HASH_DIRECTORY_ADD_FILE;

typedef
_Check_return_
_Success_(return >= 0)
_Requires_lock_not_held_(Directory->Lock)
_Requires_exclusive_lock_held_(File->Lock)
_Pre_satisfies_(File->ParentDirectory != NULL)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_DIRECTORY_REMOVE_FILE)(
    _In_ PPERFECT_HASH_DIRECTORY Directory,
    _In_ PPERFECT_HASH_FILE File
    );
typedef PERFECT_HASH_DIRECTORY_REMOVE_FILE
      *PPERFECT_HASH_DIRECTORY_REMOVE_FILE;

//
// Define our private PERFECT_HASH_DIRECTORY_VTBL structure.
//

typedef struct _PERFECT_HASH_DIRECTORY_VTBL {
    DECLARE_COMPONENT_VTBL_HEADER(PERFECT_HASH_DIRECTORY);
    PPERFECT_HASH_DIRECTORY_OPEN Open;
    PPERFECT_HASH_DIRECTORY_CREATE Create;
    PPERFECT_HASH_DIRECTORY_GET_FLAGS GetFlags;
    PPERFECT_HASH_DIRECTORY_GET_PATH GetPath;

    //
    // Begin private methods.
    //

    PPERFECT_HASH_DIRECTORY_CLOSE Close;
    PPERFECT_HASH_DIRECTORY_SCHEDULE_RENAME ScheduleRename;
    PPERFECT_HASH_DIRECTORY_DO_RENAME DoRename;
    PPERFECT_HASH_DIRECTORY_ADD_FILE AddFile;
    PPERFECT_HASH_DIRECTORY_REMOVE_FILE RemoveFile;
} PERFECT_HASH_DIRECTORY_VTBL;
typedef PERFECT_HASH_DIRECTORY_VTBL *PPERFECT_HASH_DIRECTORY_VTBL;

//
// Define the PERFECT_HASH_DIRECTORY_STATE structure.
//

typedef union _PERFECT_HASH_DIRECTORY_STATE {
    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // The directory has been "set" by way of an Open() or Create() call.
        //

        ULONG IsSet:1;

        //
        // A Close() call has been issued on the directory.
        //

        ULONG IsClosed:1;

        //
        // Directory is read-only.  This will only ever be set if the directory
        // is set.  (That is, it is cleared when the directory is closed,
        // regardless of whether or not the directory *was* read-only when set.)
        //
        // When set, indicates the directory has been set via Open() instead of
        // Create().
        //

        ULONG IsReadOnly:1;

        //
        // When set, indicates the underlying filesystem directory was created
        // by us.  When clear, indicates the directory already existed.
        //

        ULONG WasCreatedByUs:1;

        //
        // Unused bits.
        //

        ULONG Unused:28;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_DIRECTORY_STATE;
C_ASSERT(sizeof(PERFECT_HASH_DIRECTORY_STATE) == sizeof(ULONG));
typedef PERFECT_HASH_DIRECTORY_STATE *PPERFECT_HASH_DIRECTORY_STATE;

//
// Helper macros.
//

#define IsDirectorySet(Directory) (Directory->State.IsSet)
#define IsDirectoryClosed(Directory) (Directory->State.IsClosed)
#define IsDirectoryReadOnly(Directory) (Directory->State.IsReadOnly)
#define IsDirectoryRenameScheduled(Directory) (Directory->RenamePath)
#define DirectoryWasCreatedByUs(Directory) (Directory->State.WasCreatedByUs)

#define DirectoryNeverSet(File) (                         \
    !Directory->Flags.Opened && !Directory->Flags.Created \
)

#define SetDirectoryOpened(Directory)  \
    Directory->State.IsSet = TRUE;     \
    Directory->State.IsClosed = FALSE; \
    Directory->Flags.Opened = TRUE;    \
    Directory->Flags.Created = FALSE

#define SetDirectoryCreated(Directory) \
    Directory->State.IsSet = TRUE;     \
    Directory->State.IsClosed = FALSE; \
    Directory->Flags.Opened = FALSE;   \
    Directory->Flags.Created = TRUE

#define SetDirectoryClosed(Directory) \
    Directory->State.IsClosed = TRUE

#define GetActiveDirPath(Dir) (Dir->RenamePath ? Dir->RenamePath : Dir->Path)

//
// Define the PERFECT_HASH_DIRECTORY structure.
//

typedef struct _Struct_size_bytes_(SizeOfStruct) _PERFECT_HASH_DIRECTORY {

    COMMON_COMPONENT_HEADER(PERFECT_HASH_DIRECTORY);

    //
    // Flags specified to load or create methods.
    //

    PERFECT_HASH_DIRECTORY_OPEN_FLAGS DirectoryOpenFlags;
    PERFECT_HASH_DIRECTORY_CREATE_FLAGS DirectoryCreateFlags;

    //
    // Handle to the underlying directory.
    //

    HANDLE DirectoryHandle;

    //
    // Pointer to the path instance from which we were opened or created.
    //

    volatile PPERFECT_HASH_PATH Path;

    //
    // Pointer to the new path name requested by a call to ScheduleRename().
    //

    volatile PPERFECT_HASH_PATH RenamePath;

    //
    // Pointer to the guarded list used to track files that need renaming.
    //

    PGUARDED_LIST FilesList;

    //
    // Backing interface.
    //

    PERFECT_HASH_DIRECTORY_VTBL Interface;

} PERFECT_HASH_DIRECTORY;
typedef PERFECT_HASH_DIRECTORY *PPERFECT_HASH_DIRECTORY;

#define TryAcquirePerfectHashDirectoryLockExclusive(Directory) \
    TryAcquireSRWLockExclusive(&Directory->Lock)

#define AcquirePerfectHashDirectoryLockExclusive(Directory) \
    AcquireSRWLockExclusive(&Directory->Lock)

#define ReleasePerfectHashDirectoryLockExclusive(Directory) \
    ReleaseSRWLockExclusive(&Directory->Lock)

#define TryAcquirePerfectHashDirectoryLockShared(Directory) \
    TryAcquireSRWLockShared(&Directory->Lock)

#define AcquirePerfectHashDirectoryLockShared(Directory) \
    AcquireSRWLockShared(&Directory->Lock)

#define ReleasePerfectHashDirectoryLockShared(Directory) \
    ReleaseSRWLockShared(&Directory->Lock)

//
// Private non-vtbl methods.
//

typedef
HRESULT
(NTAPI PERFECT_HASH_DIRECTORY_INITIALIZE)(
    _In_ PPERFECT_HASH_DIRECTORY Directory
    );
typedef PERFECT_HASH_DIRECTORY_INITIALIZE *PPERFECT_HASH_DIRECTORY_INITIALIZE;

typedef
VOID
(NTAPI PERFECT_HASH_DIRECTORY_RUNDOWN)(
    _In_ _Post_ptr_invalid_ PPERFECT_HASH_DIRECTORY Directory
    );
typedef PERFECT_HASH_DIRECTORY_RUNDOWN *PPERFECT_HASH_DIRECTORY_RUNDOWN;

extern PERFECT_HASH_DIRECTORY_INITIALIZE PerfectHashDirectoryInitialize;
extern PERFECT_HASH_DIRECTORY_RUNDOWN PerfectHashDirectoryRundown;
extern PERFECT_HASH_DIRECTORY_OPEN PerfectHashDirectoryOpen;
extern PERFECT_HASH_DIRECTORY_CREATE PerfectHashDirectoryCreate;
extern PERFECT_HASH_DIRECTORY_GET_FLAGS PerfectHashDirectoryGetFlags;
extern PERFECT_HASH_DIRECTORY_GET_PATH PerfectHashDirectoryGetPath;
extern PERFECT_HASH_DIRECTORY_CLOSE PerfectHashDirectoryClose;
extern PERFECT_HASH_DIRECTORY_SCHEDULE_RENAME
    PerfectHashDirectoryScheduleRename;
extern PERFECT_HASH_DIRECTORY_DO_RENAME PerfectHashDirectoryDoRename;
extern PERFECT_HASH_DIRECTORY_ADD_FILE PerfectHashDirectoryAddFile;
extern PERFECT_HASH_DIRECTORY_REMOVE_FILE PerfectHashDirectoryRemoveFile;

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
