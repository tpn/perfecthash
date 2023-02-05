/*++

Copyright (c) 2023 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashCompat.c

Abstract:

    Implementations of "compat" (i.e. non-Windows) routines.

--*/

#include "stdafx.h"
#include "PerfectHashEventsPrivate.h"

#ifdef PH_WINDOWS
#error This file is not for Windows.
#endif

#ifndef PH_COMPAT
#error PH_COMPAT should be defined for this file.
#endif

//
// Includes.
//

#include <stdio.h>
#include <unistd.h>
#include <sys/sysinfo.h>

DWORD LastError;

DWORD
GetLastError(
    VOID
    )
{
    return LastError;
}

VOID
SetLastError(
    DWORD Error
    )
{
    LastError = Error;
}

DWORD
GetCurrentThreadId(
    VOID
    )
{
    return (DWORD)pthread_self();
    //return syscall(__NR_gettid);
}

HANDLE
GetCurrentProcess(
    VOID
    )
{
    return NULL;
}


DWORD
GetMaximumProcessorCount(
    _In_ WORD GroupNumber
    )
{
    return get_nprocs();
}

_Success_(return != 0)
BOOL
WINAPI
GetComputerNameA (
    _Out_writes_to_opt_(*nSize, *nSize + 1) LPSTR Buffer,
    _Inout_ LPDWORD Size
    )
{
    LastError = gethostname(Buffer, *Size);
    Buffer[MAX_COMPUTERNAME_LENGTH-1] = '\0';
    *Size = strlen(Buffer);
    return (LastError == 0);
}

WINBASEAPI
HANDLE
WINAPI
GetStdHandle(
    _In_ DWORD StdHandle
    )
{
    switch (StdHandle) {
        case STD_INPUT_HANDLE:
            return (HANDLE)stdin;

        case STD_OUTPUT_HANDLE:
            return (HANDLE)stdout;

        case STD_ERROR_HANDLE:
            return (HANDLE)stderr;

        default:
            return INVALID_HANDLE_VALUE;
    }
}

WINBASEAPI
VOID
WINAPI
OutputDebugStringA(
    _In_opt_ LPCSTR lpOutputString
    )
{
    fprintf(stderr, "%s\n", lpOutputString);
}


//
// File times.
//


VOID
GetSystemTime(
    _Out_ LPSYSTEMTIME lpSystemTime
    )
{
    return;
}

VOID
GetSystemTimeAsFileTime(
    _Out_ LPFILETIME lpSystemTimeAsFileTime
    )
{
    return;
}

VOID
GetLocalTime(
    _Out_ LPSYSTEMTIME lpSystemTime
    )
{
    return;
}

_Success_(return != FALSE)
BOOL
FileTimeToSystemTime(
    _In_ CONST FILETIME* lpFileTime,
    _Out_ LPSYSTEMTIME lpSystemTime
    )
{
    return FALSE;
}

_Success_(return != FALSE)
BOOL
SystemTimeToFileTime(
    _In_ CONST SYSTEMTIME* lpSystemTime,
    _Out_ LPFILETIME lpFileTime
    )
{
    return FALSE;
}

#include <time.h>

WINBASEAPI
ULONGLONG
WINAPI
GetTickCount64(
    VOID
    )
{
    ULONGLONG t = 0;
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    t  = ts.tv_nsec / 1000000;
    t += ts.tv_sec * 1000;
    return t;
}


WINBASEAPI
BOOL
WINAPI
QueryPerformanceCounter(
    _Out_ LARGE_INTEGER* lpPerformanceCount
    )
{
    ULONGLONG t = 0;
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);

    t  = ts.tv_nsec;
    t += ts.tv_sec * 1e9;

    lpPerformanceCount->QuadPart = t;

    return TRUE;
}

WINBASEAPI
BOOL
WINAPI
QueryPerformanceFrequency(
    _Out_ LARGE_INTEGER* lpFrequency
    )
{
    lpFrequency->QuadPart = 1000;
}

//
// Files and directories.
//

WINBASEAPI
BOOL
WINAPI
GetFileTime(
    _In_ HANDLE hFile,
    _Out_opt_ LPFILETIME lpCreationTime,
    _Out_opt_ LPFILETIME lpLastAccessTime,
    _Out_opt_ LPFILETIME lpLastWriteTime
    )
{
    return FALSE;
}


WINBASEAPI
HANDLE
WINAPI
CreateFileW(
    _In_ LPCWSTR lpFileName,
    _In_ DWORD dwDesiredAccess,
    _In_ DWORD dwShareMode,
    _In_opt_ LPSECURITY_ATTRIBUTES lpSecurityAttributes,
    _In_ DWORD dwCreationDisposition,
    _In_ DWORD dwFlagsAndAttributes,
    _In_opt_ HANDLE hTemplateFile
    )
{
    return INVALID_HANDLE_VALUE;
}

WINBASEAPI
BOOL
WINAPI
DeleteFileW(
    _In_ LPCWSTR lpFileName
    )
{
    return FALSE;
}

WINBASEAPI
BOOL
WINAPI
CreateDirectoryW(
    _In_ LPCWSTR lpPathName,
    _In_opt_ LPSECURITY_ATTRIBUTES lpSecurityAttributes
    )
{
    return FALSE;
}

WINBASEAPI
BOOL
WINAPI
SetEndOfFile(
    _In_ HANDLE hFile
    )
{
    return FALSE;
}

WINBASEAPI
_Ret_maybenull_
LPVOID
WINAPI
MapViewOfFile(
    _In_ HANDLE hFileMappingObject,
    _In_ DWORD dwDesiredAccess,
    _In_ DWORD dwFileOffsetHigh,
    _In_ DWORD dwFileOffsetLow,
    _In_ SIZE_T dwNumberOfBytesToMap
    )
{
    return NULL;
}

WINBASEAPI
_Ret_maybenull_
HANDLE
WINAPI
CreateFileMappingW(
    _In_ HANDLE hFile,
    _In_opt_ LPSECURITY_ATTRIBUTES lpFileMappingAttributes,
    _In_ DWORD flProtect,
    _In_ DWORD dwMaximumSizeHigh,
    _In_ DWORD dwMaximumSizeLow,
    _In_opt_ LPCWSTR lpName
    )
{
    return INVALID_HANDLE_VALUE;
}


WINBASEAPI
BOOL
WINAPI
UnmapViewOfFile(
    _In_ LPCVOID lpBaseAddress
    )
{
    return FALSE;
}

WINBASEAPI
_Ret_maybenull_
LPVOID
WINAPI
MapViewOfFileEx(
    _In_ HANDLE hFileMappingObject,
    _In_ DWORD dwDesiredAccess,
    _In_ DWORD dwFileOffsetHigh,
    _In_ DWORD dwFileOffsetLow,
    _In_ SIZE_T dwNumberOfBytesToMap,
    _In_opt_ LPVOID lpBaseAddress
    )
{
    return NULL;
}

WINBASEAPI
DWORD
WINAPI
SetFilePointer(
    _In_ HANDLE hFile,
    _In_ LONG lDistanceToMove,
    _Inout_opt_ PLONG lpDistanceToMoveHigh,
    _In_ DWORD dwMoveMethod
    )
{
    return 0;
}

WINBASEAPI
BOOL
WINAPI
SetFilePointerEx(
    _In_ HANDLE hFile,
    _In_ LARGE_INTEGER liDistanceToMove,
    _Out_opt_ PLARGE_INTEGER lpNewFilePointer,
    _In_ DWORD dwMoveMethod
    )
{
    return FALSE;
}

WINBASEAPI
BOOL
WINAPI
MoveFileExW(
    _In_     LPCWSTR lpExistingFileName,
    _In_opt_ LPCWSTR lpNewFileName,
    _In_     DWORD    dwFlags
    )
{
    return FALSE;
}

WINBASEAPI
BOOL
WINAPI
GetFileInformationByHandle(
    _In_ HANDLE hFile,
    _Out_ LPBY_HANDLE_FILE_INFORMATION lpFileInformation
    )
{
    return FALSE;
}

WINBASEAPI
BOOL
WINAPI
GetFileInformationByHandleEx(
    _In_  HANDLE hFile,
    _In_  FILE_INFO_BY_HANDLE_CLASS FileInformationClass,
    _Out_writes_bytes_(dwBufferSize) LPVOID lpFileInformation,
    _In_  DWORD dwBufferSize
    )
{
    return FALSE;
}

WINBASEAPI
BOOL
WINAPI
RemoveDirectoryW(
    _In_ LPCWSTR lpPathName
    )
{
    return FALSE;
}

WINBASEAPI
BOOL
WINAPI
CloseHandle(
    _In_ _Post_ptr_invalid_ HANDLE hObject
    )
{
    return FALSE;
}

WINBASEAPI
BOOL
WINAPI
WriteFile(
    _In_ HANDLE hFile,
    _In_reads_bytes_opt_(nNumberOfBytesToWrite) LPCVOID lpBuffer,
    _In_ DWORD nNumberOfBytesToWrite,
    _Out_opt_ LPDWORD lpNumberOfBytesWritten,
    _Inout_opt_ LPOVERLAPPED lpOverlapped
    )
{
    LARGE_INTEGER fd;
    SIZE_T BytesWritten;

    fd.QuadPart = (LONGLONG)hFile;

    BytesWritten = write(fd.LowPart, lpBuffer, nNumberOfBytesToWrite);

    if (ARGUMENT_PRESENT(lpNumberOfBytesWritten)) {
        *lpNumberOfBytesWritten = BytesWritten;
    }

    if (BytesWritten == -1) {
        SetLastError(errno);
        SYS_ERROR(write);
        return FALSE;
    }

    return TRUE;
}

//
// SRW locks.
//
// TODO: change to phtread_rwlock_*.
//

WINBASEAPI
VOID
WINAPI
InitializeSRWLock(
    _Out_ PSRWLOCK SRWLock
    )
{
    LastError = pthread_mutex_init(SRWLock, NULL);
    if (LastError != 0) {
        PH_RAISE(PH_E_SYSTEM_CALL_FAILED);
    }
}

WINBASEAPI
_Releases_exclusive_lock_(*SRWLock)
VOID
WINAPI
ReleaseSRWLockExclusive(
    _Inout_ PSRWLOCK SRWLock
    )
{
    LastError = pthread_mutex_unlock(SRWLock);
    if (LastError != 0) {
        PH_RAISE(PH_E_SYSTEM_CALL_FAILED);
    }
}

WINBASEAPI
_Releases_shared_lock_(*SRWLock)
VOID
WINAPI
ReleaseSRWLockShared(
    _Inout_ PSRWLOCK SRWLock
    )
{
    LastError = pthread_mutex_unlock(SRWLock);
    if (LastError != 0) {
        PH_RAISE(PH_E_SYSTEM_CALL_FAILED);
    }
}

WINBASEAPI
_Acquires_exclusive_lock_(*SRWLock)
VOID
WINAPI
AcquireSRWLockExclusive(
    _Inout_ PSRWLOCK SRWLock
    )
{
    LastError = pthread_mutex_lock(SRWLock);
    if (LastError != 0) {
        PH_RAISE(PH_E_SYSTEM_CALL_FAILED);
    }
}

WINBASEAPI
_Acquires_shared_lock_(*SRWLock)
VOID
WINAPI
AcquireSRWLockShared(
    _Inout_ PSRWLOCK SRWLock
    )
{
    LastError = pthread_mutex_lock(SRWLock);
    if (LastError != 0) {
        PH_RAISE(PH_E_SYSTEM_CALL_FAILED);
    }
}

WINBASEAPI
_When_(return!=0, _Acquires_exclusive_lock_(*SRWLock))
BOOLEAN
WINAPI
TryAcquireSRWLockExclusive(
    _Inout_ PSRWLOCK SRWLock
    )
{
    LastError = pthread_mutex_trylock(SRWLock);
    if (LastError != 0 && LastError != EBUSY) {
        PH_RAISE(PH_E_SYSTEM_CALL_FAILED);
    }
    return (LastError == 0);
}

WINBASEAPI
_When_(return!=0, _Acquires_shared_lock_(*SRWLock))
BOOLEAN
WINAPI
TryAcquireSRWLockShared(
    _Inout_ PSRWLOCK SRWLock
    )
{
    LastError = pthread_mutex_trylock(SRWLock);
    if (LastError != 0 && LastError != EBUSY) {
        PH_RAISE(PH_E_SYSTEM_CALL_FAILED);
    }
    return (LastError == 0);
}

//
// Hacky init once with global vars.
//

PINIT_ONCE InitOnceInitOnce;
PVOID InitOnceParameter;
PVOID InitOnceContext;
PINIT_ONCE_FN InitOnceFunction;

C_ASSERT(sizeof(INIT_ONCE) >= sizeof(pthread_once));

typedef VOID (INIT_CALLBACK)(VOID);
typedef INIT_CALLBACK *PINIT_CALLBACK;

VOID
InitOnceWrapper(
    VOID
    )
{
    InitOnceFunction(InitOnceInitOnce, InitOnceParameter, InitOnceContext);
}

WINBASEAPI
BOOL
WINAPI
InitOnceExecuteOnce(
    _Inout_ PINIT_ONCE InitOnce,
    _In_ __callback PINIT_ONCE_FN InitFn,
    _Inout_opt_ PVOID Parameter,
    _Outptr_opt_result_maybenull_ LPVOID* Context
    )
{
    InitOnceContext = Context;
    InitOnceFunction = InitFn;
    InitOnceParameter = Parameter;
    pthread_once((pthread_once_t *)InitOnce, InitOnceWrapper);
    return TRUE;
}

//
// SYSTEM_INFO
//

WINBASEAPI
VOID
WINAPI
GetSystemInfo(
    _Out_ LPSYSTEM_INFO lpSystemInfo
    )
{
    ZeroStructPointerInline(lpSystemInfo);
    lpSystemInfo->dwPageSize = 4096;
    lpSystemInfo->dwNumberOfProcessors = get_nprocs();
}

//
// Critical sections.
//
// TODO: change to pthread_spin_*.
//

C_ASSERT(sizeof(CRITICAL_SECTION) >= sizeof(pthread_mutex_t));

WINBASEAPI
_Must_inspect_result_
BOOL
WINAPI
InitializeCriticalSectionAndSpinCount(
    _Out_ LPCRITICAL_SECTION lpCriticalSection,
    _In_ DWORD dwSpinCount
    )
{
    LastError = pthread_mutex_init((pthread_mutex_t*)lpCriticalSection, NULL);
    if (LastError != 0) {
        PH_RAISE(PH_E_SYSTEM_CALL_FAILED);
    }
    return TRUE;
}

WINBASEAPI
VOID
WINAPI
EnterCriticalSection(
    _Inout_ LPCRITICAL_SECTION lpCriticalSection
    )
{
    LastError = pthread_mutex_lock((pthread_mutex_t*)lpCriticalSection);
    if (LastError != 0) {
        PH_RAISE(PH_E_SYSTEM_CALL_FAILED);
    }
}

WINBASEAPI
VOID
WINAPI
LeaveCriticalSection(
    _Inout_ LPCRITICAL_SECTION lpCriticalSection
    )
{
    LastError = pthread_mutex_unlock((pthread_mutex_t*)lpCriticalSection);
    if (LastError != 0) {
        PH_RAISE(PH_E_SYSTEM_CALL_FAILED);
    }
}

WINBASEAPI
VOID
WINAPI
DeleteCriticalSection(
    _Inout_ LPCRITICAL_SECTION lpCriticalSection
    )
{
    return;
}

//
// Events
//

typedef union _Struct_size_bytes_(sizeof(ULONG)) _PH_EVENT_STATE {
    struct {

        //
        // When set, indicates the event is signaled.
        //

        ULONG Signaled:1;

        //
        // When set, indicates the event is to be manually reset.
        //

        ULONG ManualReset:1;

        //
        // When set, indicates the mutex has been initialized.
        //

        ULONG MutexInitialized:1;

        //
        // When set, indicates the condition has been initialized.
        //

        ULONG ConditionInitialized:1;

        //
        // When set, indicates the name has been allocated.
        //

        ULONG NameAllocated:1;

        //
        // Unused bits.
        //

        ULONG Unused:27;

    };

    LONG AsLong;
    ULONG AsULong;
} PH_EVENT_STATE, *PPH_EVENT_STATE;

typedef struct _PH_EVENT {
    PH_EVENT_STATE State;
    ULONG Padding1;
    PSTR Name;
    pthread_mutex_t Mutex;
    pthread_cond_t Condition;
} PH_EVENT, *PPH_EVENT;

WINBASEAPI
_Ret_maybenull_
HANDLE
WINAPI
CreateEventW(
    _In_opt_ LPSECURITY_ATTRIBUTES EventAttributes,
    _In_ BOOL ManualReset,
    _In_ BOOL InitialState,
    _In_opt_ LPCWSTR Name
    )
{
    LONG Error;
    BOOL Success;
    PWSTR LocalName;
    PPH_EVENT Event;
    SIZE_T NameLengthInChars;

    UNREFERENCED_PARAMETER(EventAttributes);

    Success = FALSE;

    Event = (PPH_EVENT)calloc(1, sizeof(*Event));
    if (Event == NULL) {
        SetLastError(ENOMEM);
        return INVALID_HANDLE_VALUE;
    }

    if (Name != NULL) {
        NameLengthInChars = wcslen(Name);
        LocalName = Event->Name = (PSTR)calloc(1, NameLengthInChars+1);
        AppendWStrToCharBufferFast(&LocalName, (PWSTR)Name);
        Event->State.NameAllocated = TRUE;
    }

    //
    // Initialize the mutex.
    //

    Error = pthread_mutex_init(&Event->Mutex, NULL);
    if (Error != 0) {
        SetLastError(Error);
        goto End;
    }
    Event->State.MutexInitialized = TRUE;

    //
    // Initialize the condition variable.
    //

    Error = pthread_cond_init(&Event->Condition, NULL);
    if (Error != 0) {
        SetLastError(Error);
        goto End;
    }
    Event->State.ConditionInitialized = TRUE;

    //
    // Initialize the remaining state.
    //

    Event->State.Signaled = InitialState;
    Event->State.ManualReset = ManualReset;

    //
    // We're done, finish up.
    //

    Success = TRUE;

End:

    if (!Success) {
        CloseEvent(Event);
    }

    return (HANDLE)Event;
}

WINBASEAPI
BOOL
WINAPI
CloseEvent(
    _In_ _Post_ptr_invalid_ HANDLE Object
    )
{
    BOOL Success;
    LONG Error;
    PPH_EVENT Event;

    Success = FALSE;
    Event = (PPH_EVENT)Object;

    if (Event == NULL || Event == INVALID_HANDLE_VALUE) {
        goto End;
    }

    if (Event->State.MutexInitialized) {
        Error = pthread_mutex_destroy(&Event->Mutex);
        if (Error != 0) {
            SetLastError(Error);
            SYS_ERROR(pthread_mutex_destroy);
        }
    }

    if (Event->State.ConditionInitialized) {
        Error = pthread_cond_destroy(&Event->Condition);
        if (Error != 0) {
            SetLastError(Error);
            SYS_ERROR(pthread_cond_destroy);
        }
    }

    if (Event->State.NameAllocated) {
        ASSERT(Event->Name != NULL);
        free(Event->Name);
        Event->Name = NULL;
    } else {
        ASSERT(Event->Name == NULL);
    }

    free(Event);

    Success = TRUE;

End:

    return Success;
}

WINBASEAPI
BOOL
WINAPI
SetEvent(
    _In_ HANDLE Handle
    )
{
    LONG Error;
    BOOL Success;
    PPH_EVENT Event;

    Event = (PPH_EVENT)Handle;

    Success = FALSE;

    Error = pthread_mutex_lock(&Event->Mutex);
    if (Error != 0) {
        SetLastError(Error);
        SYS_ERROR(pthread_mutex_lock);
        goto End;
    }

    Event->State.Signaled = TRUE;

    Error = pthread_mutex_unlock(&Event->Mutex);
    if (Error != 0) {
        SetLastError(Error);
        SYS_ERROR(pthread_mutex_unlock);
        goto End;
    }

    Error = pthread_cond_broadcast(&Event->Condition);
    if (Error != 0) {
        SetLastError(Error);
        SYS_ERROR(pthread_cond_broadcast);
        goto End;
    }

    Success = TRUE;

End:

    return Success;
}

WINBASEAPI
BOOL
WINAPI
ResetEvent(
    _In_ HANDLE Handle
    )
{
    LONG Error;
    BOOL Success;
    PPH_EVENT Event;

    Event = (PPH_EVENT)Handle;

    Success = FALSE;

    Error = pthread_mutex_lock(&Event->Mutex);
    if (Error != 0) {
        SetLastError(Error);
        SYS_ERROR(pthread_mutex_lock);
        goto End;
    }

    Event->State.Signaled = FALSE;

    Error = pthread_mutex_unlock(&Event->Mutex);
    if (Error != 0) {
        SetLastError(Error);
        SYS_ERROR(pthread_mutex_unlock);
        goto End;
    }

    Success = TRUE;

End:

    return Success;
}

WINBASEAPI
DWORD
WINAPI
WaitForSingleObject(
    _In_ HANDLE hHandle,
    _In_ DWORD dwMilliseconds
    )
{
    LONG Error;
    PPH_EVENT Event;
    DWORD WaitResult;

    ASSERT(dwMilliseconds == INFINITE);

    Event = (PPH_EVENT)hHandle;
    WaitResult = WAIT_FAILED;

    Error = pthread_mutex_lock(&Event->Mutex);
    if (Error != 0) {
        SetLastError(Error);
        SYS_ERROR(pthread_mutex_lock);
        goto End;
    }

    while (Event->State.Signaled == FALSE) {

        Error = pthread_cond_wait(&Event->Condition, &Event->Mutex);
        if (Error != 0) {
            SetLastError(Error);
            SYS_ERROR(pthread_cond_wait);
            goto End;
        }

    }

    if (Event->State.ManualReset == FALSE) {
        Event->State.Signaled = FALSE;
    }

    Error = pthread_mutex_unlock(&Event->Mutex);
    if (Error != 0) {
        SetLastError(Error);
        SYS_ERROR(pthread_mutex_unlock);
        goto End;
    }

    WaitResult = WAIT_OBJECT_0;

End:

    return WaitResult;
}

WINBASEAPI
DWORD
WINAPI
WaitForMultipleObjects(
    _In_ DWORD nCount,
    _In_reads_(nCount) CONST HANDLE* lpHandles,
    _In_ BOOL bWaitAll,
    _In_ DWORD dwMilliseconds
    )
{
    return WAIT_FAILED;
}


//
// Memory.
//



#include <linux/mman.h>
#include <sys/mman.h>

WINBASEAPI
SIZE_T
WINAPI
GetLargePageMinimum(
    VOID
    )
{
    //
    // 2MB
    //

    return 2 * 1024 * 1024;
}

WINBASEAPI
HANDLE
WINAPI
GetProcessHeap(
    VOID
    )
{
    return NULL;
}

WINBASEAPI
_Ret_maybenull_
_Post_writable_byte_size_(dwBytes)
LPVOID
WINAPI
HeapAlloc(
    _In_ HANDLE Heap,
    _In_ DWORD Flags,
    _In_ SIZE_T SizeInBytes
    )
{
    UNREFERENCED_PARAMETER(Heap);

    if ((Flags & HEAP_ZERO_MEMORY) != 0) {
        return calloc(1, SizeInBytes);
    } else {
        return malloc(SizeInBytes);
    }
}

WINBASEAPI
_Success_(return != 0)
_Ret_maybenull_
_Post_writable_byte_size_(dwBytes)
LPVOID
WINAPI
HeapReAlloc(
    _Inout_ HANDLE Heap,
    _In_ DWORD Flags,
    _Frees_ptr_opt_ LPVOID Mem,
    _In_ SIZE_T SizeInBytes
    )
{
    UNREFERENCED_PARAMETER(Heap);

    if ((Flags & HEAP_ZERO_MEMORY) != 0) {

        //
        // Erm, can't do anything here as we don't know the original size for
        // a memcpy, e.g.:
        //
        //  New = calloc(1, SizeInBytes);
        //  memcpy(New, Mem, OriginalSizeInBytes);
        //  free(Mem);
        //  return New;
        //
        // The only code that uses Allocator->Vtbl->ReCalloc() is ExtractArg.c
        // for resizing the table create parameters.
        //

        NOTHING;
    }

    return realloc(Mem, SizeInBytes);
}
WINBASEAPI
_Success_(return != FALSE)
BOOL
WINAPI
HeapFree(
    _Inout_ HANDLE Heap,
    _In_ DWORD Flags,
    __drv_freesMem(Mem) _Frees_ptr_opt_ LPVOID Mem
    )
{
    UNREFERENCED_PARAMETER(Heap);
    UNREFERENCED_PARAMETER(Flags);

    free(Mem);

    return TRUE;
}

//
// Virtual alloc and friends.
//

WINBASEAPI
_Ret_maybenull_
_Post_writable_byte_size_(dwSize)
LPVOID
WINAPI
VirtualAlloc(
    _In_opt_ LPVOID Address,
    _In_ SIZE_T Size,
    _In_ DWORD AllocationType,
    _In_ DWORD Protect
    )
{
    int prot;
    int flags;
    void *addr;

    ASSERT(BooleanFlagOn(AllocationType, MEM_COMMIT));

    //
    // Convert flags.
    //

    flags = MAP_ANONYMOUS | MAP_PRIVATE;
    if (BooleanFlagOn(AllocationType, MEM_LARGE_PAGES)) {
        flags |= (MAP_HUGETLB | MAP_HUGE_2MB);
    }

    //
    // Convert protection.
    //

    prot = 0;
    if (BooleanFlagOn(Protect, PAGE_READWRITE)) {
        prot = PROT_READ | PROT_WRITE;
    } else if (BooleanFlagOn(Protect, PAGE_NOACCESS)) {
        prot = PROT_NONE;
    } else if (BooleanFlagOn(Protect, PAGE_READONLY)) {
        prot = PROT_READ;
    }

    addr = mmap(Address, Size, prot, flags, 0, 0);
    if (addr == NULL || addr == (void *)-1) {
        SetLastError(errno);
        SYS_ERROR(mmap);
        addr = NULL;
    }

    return addr;
}

WINBASEAPI
_Ret_maybenull_
_Post_writable_byte_size_(dwSize)
LPVOID
WINAPI
VirtualAllocEx(
    _In_ HANDLE Process,
    _In_opt_ LPVOID Address,
    _In_ SIZE_T Size,
    _In_ DWORD AllocationType,
    _In_ DWORD Protect
    )
{
    return VirtualAlloc(Address, Size, AllocationType, Protect);
}

WINBASEAPI
_Success_(return != FALSE)
BOOL
WINAPI
VirtualProtect(
    _In_ LPVOID Address,
    _In_ SIZE_T Size,
    _In_ DWORD NewProtect,
    _Out_ PDWORD OldProtect
    )
{
    LONG Error;
    int prot;

    if (BooleanFlagOn(NewProtect, PAGE_NOACCESS)) {
        prot = PROT_NONE;
    } else if (BooleanFlagOn(NewProtect, PAGE_READONLY)) {
        prot = PROT_READ;
    }

    Error = mprotect(Address, Size, prot);
    if (Error != 0) {
        SetLastError(errno);
        SYS_ERROR(mprotect);
        return FALSE;
    }

    return TRUE;
}

WINBASEAPI
_Success_(return != FALSE)
BOOL
WINAPI
VirtualProtectEx(
    _In_ HANDLE Process,
    _In_ LPVOID Address,
    _In_ SIZE_T Size,
    _In_ DWORD NewProtect,
    _Out_ PDWORD OldProtect
    )
{
    UNREFERENCED_PARAMETER(Process);
    UNREFERENCED_PARAMETER(OldProtect);

    return VirtualProtect(Address, Size, NewProtect, OldProtect);
}


WINBASEAPI
BOOL
WINAPI
VirtualFree(
    _In_ LPVOID Address,
    _In_ SIZE_T Size,
    _In_ DWORD FreeType
    )
{
    LONG Error;

    ASSERT(FreeType == MEM_RELEASE);

    Error = munmap(Address, Size);
    if (Error != 0) {
        SetLastError(errno);
        SYS_ERROR(munmap);
        return FALSE;
    }

    return TRUE;
}

WINBASEAPI
BOOL
WINAPI
VirtualFreeEx(
    _In_ HANDLE Process,
    _In_ LPVOID Address,
    _In_ SIZE_T Size,
    _In_ DWORD FreeType
    )
{
    return VirtualFree(Address, Size, FreeType);
}


HLOCAL
WINAPI
LocalFree(
    _Frees_ptr_opt_ HLOCAL hMem
    )
{
    free(hMem);
    return 0;
}

//
// TLS.
//

TLS_KEY_TYPE PerfectHashTlsIndex;

pthread_once_t PerfectHashTlsInitOnce = PTHREAD_ONCE_INIT;
int PerfectHashTlsCreateError = 0;

VOID
PerfectHashTlsAlloc(
    VOID
    )
{
    PerfectHashTlsCreateError = pthread_key_create(&PerfectHashTlsIndex, NULL);
    if (PerfectHashTlsCreateError != 0) {
        PH_ERROR(phtread_key_create, PerfectHashTlsCreateError);
        exit(PerfectHashTlsCreateError);
    }
}

__attribute__((constructor))
VOID
PerfectHashTlsProcessAttach(
    VOID
    )
{
    pthread_once(&PerfectHashTlsInitOnce, PerfectHashTlsAlloc);
}

WINBASEAPI
LPVOID
WINAPI
TlsGetValue(
    _In_ DWORD TlsIndex
    )
{
    TLS_KEY_TYPE Key;

    Key = (TLS_KEY_TYPE)TlsIndex;

    return pthread_getspecific(Key);
}

WINBASEAPI
BOOL
WINAPI
TlsSetValue(
    _In_ DWORD TlsIndex,
    _In_opt_ LPVOID TlsValue
    )
{
    TLS_KEY_TYPE Key;

    Key = (TLS_KEY_TYPE)TlsIndex;

    return (pthread_setspecific(Key, TlsValue) == 0);
}

WINBASEAPI
BOOL
WINAPI
TlsFree(
    _In_ DWORD TlsIndex
    )
{
    UNREFERENCED_PARAMETER(TlsIndex);
    return TRUE;
}

//
// Misc Rtl.
//

HRESULT
EnableLockMemoryPrivilege(
    PRTL Rtl
    )
{
    return S_OK;
}

_Use_decl_annotations_
HRESULT
RtlGenerateRandomBytes(
    PRTL Rtl,
    ULONG SizeOfBufferInBytes,
    PBYTE Buffer
    )
/*++

Routine Description:

    This routine writes random bytes into a given buffer using the system's
    random data generation facilities.

Arguments:

    Rtl - Supplies a pointer to an RTL instance.

    SizeOfBufferInBytes - Supplies the size of the Buffer parameter, in bytes.

    Buffer - Supplies the address for which random bytes will be written.

Return Value:

    S_OK - Success.

    E_POINTER - Rtl or Buffer is NULL.

    E_INVALIDARG - SizeOfBufferInBytes is 0.

    PH_E_FAILED_TO_GENERATE_RANDOM_BYTES - System routine failed to generate
        random bytes.

--*/
{
    ULONG Seed;
    ULONG Index;
    ULONG Count;
    PULONG Dest;
    ULONG Random;
    HRESULT Result;
    PBYTE DestBytes;
    PBYTE SourceBytes;
    ULONG NumberOfLongs;
    ULONG TrailingBytes;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Rtl)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Buffer)) {
        return E_POINTER;
    }

    if (SizeOfBufferInBytes == 0) {
        return E_INVALIDARG;
    }

    //
    // Argument validation complete.  Continue.
    //

    EVENT_WRITE_RTL_RANDOM_BYTES_START(SizeOfBufferInBytes);

    NumberOfLongs = SizeOfBufferInBytes / sizeof(ULONG);
    TrailingBytes = SizeOfBufferInBytes % sizeof(ULONG);

    //
    // Seed rand().
    //

    srand(time(NULL));

    //
    // Get the number of 32-bit random values we need.
    //

    Dest = (PULONG)Buffer;
    for (Index = 0; Index < NumberOfLongs; Index++) {
        *Dest++ = rand_r(&Seed);
    }

    if (TrailingBytes) {

        DestBytes = (PBYTE)Dest;
        Random = rand_r(&Seed);
        SourceBytes = (PBYTE)&Random;

        for (Index = 0; Index < TrailingBytes; Index++) {
            *DestBytes++ = *SourceBytes++;
        }
    }

    EVENT_WRITE_RTL_RANDOM_BYTES_STOP(SizeOfBufferInBytes, Result);

    return S_OK;
}

_Use_decl_annotations_
HRESULT
RtlCreateUuidString(
    PRTL Rtl,
    PSTRING String
    )
{
    USHORT Index;
    USHORT Count;
    PCHAR Buffer;
    CHAR Char;
    CHAR Upper;
    GUID Guid;
    HRESULT Result;
    PSTR GuidCStr = NULL;

    UNREFERENCED_PARAMETER(Rtl);

    Result = RtlGenerateRandomBytes(Rtl, sizeof(Guid), (PBYTE)&Guid);
    if (FAILED(Result)) {
        SYS_ERROR(UuidCreate);
        goto End;
    }

    GuidCStr = (PSTR)calloc(1, UUID_STRING_LENGTH+1);
    if (!GuidCStr) {
        return E_OUTOFMEMORY;
    }

    sprintf(GuidCStr,
            "%08x-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x",
            Guid.Data1, Guid.Data2, Guid.Data3,
            Guid.Data4[0], Guid.Data4[1], Guid.Data4[2],
            Guid.Data4[3], Guid.Data4[4], Guid.Data4[5],
            Guid.Data4[6], Guid.Data4[7]);

    String->Buffer = (PCHAR)GuidCStr;
    String->Length = (USHORT)strlen(String->Buffer);
    String->MaximumLength = String->Length + 1;
    ASSERT(String->Length == UUID_STRING_LENGTH);
    ASSERT(String->Buffer[String->Length] == '\0');

    //
    // Convert the UUID into uppercase.
    //

    Buffer = (PCHAR)GuidCStr;
    Count = UUID_STRING_LENGTH;

    for (Index = 0; Index < Count; Index++, Buffer++) {
        Upper = Char = *Buffer;

        if (Char >= 'a' && Char <= 'f') {
            Upper -= 0x20;
            *Buffer = Upper;
        }

    }

End:

    return Result;
}

_Use_decl_annotations_
HRESULT
RtlFreeUuidString(
    PRTL Rtl,
    PSTRING String
    )
{
    HRESULT Result = S_OK;

    if (!IsValidUuidString(String)) {
        Result = PH_E_INVALID_UUID_STRING;
        goto End;
    }

    free(&String->Buffer);
    ZeroStructPointer(String);

End:

    return Result;
}


//
// Threadpools.
//

WINBASEAPI
_Must_inspect_result_
PTP_POOL
WINAPI
CreateThreadpool(
    _Reserved_ PVOID reserved
    )
{
    return NULL;
}

WINBASEAPI
VOID
WINAPI
SetThreadpoolThreadMaximum(
    _Inout_ PTP_POOL ptpp,
    _In_ DWORD cthrdMost
    )
{
    return;
}

WINBASEAPI
BOOL
WINAPI
SetThreadpoolThreadMinimum(
    _Inout_ PTP_POOL ptpp,
    _In_ DWORD cthrdMic
    )
{
    return FALSE;
}

WINBASEAPI
VOID
WINAPI
CloseThreadpool(
    _Inout_ PTP_POOL ptpp
    )
{
    return;
}

WINBASEAPI
_Must_inspect_result_
PTP_CLEANUP_GROUP
WINAPI
CreateThreadpoolCleanupGroup(
    VOID
    )
{
    return NULL;
}

WINBASEAPI
VOID
WINAPI
CloseThreadpoolCleanupGroupMembers(
    _Inout_ PTP_CLEANUP_GROUP ptpcg,
    _In_ BOOL fCancelPendingCallbacks,
    _Inout_opt_ PVOID pvCleanupContext
    )
{
    return;
}

WINBASEAPI
VOID
WINAPI
CloseThreadpoolCleanupGroup(
    _Inout_ PTP_CLEANUP_GROUP ptpcg
    )
{
    return;
}

WINBASEAPI
VOID
WINAPI
SetEventWhenCallbackReturns(
    _Inout_ PTP_CALLBACK_INSTANCE pci,
    _In_ HANDLE evt
    )
{
    return;
}

WINBASEAPI
_Must_inspect_result_
PTP_WORK
WINAPI
CreateThreadpoolWork(
    _In_ PTP_WORK_CALLBACK pfnwk,
    _Inout_opt_ PVOID pv,
    _In_opt_ PTP_CALLBACK_ENVIRON pcbe
    )
{
    return NULL;
}

WINBASEAPI
VOID
WINAPI
SubmitThreadpoolWork(
    _Inout_ PTP_WORK pwk
    )
{
    return;
}

WINBASEAPI
VOID
WINAPI
WaitForThreadpoolWorkCallbacks(
    _Inout_ PTP_WORK pwk,
    _In_ BOOL fCancelPendingCallbacks
    )
{
    return;
}

WINBASEAPI
VOID
WINAPI
CloseThreadpoolWork(
    _Inout_ PTP_WORK pwk
    )
{
    return;
}

WINBASEAPI
_Must_inspect_result_
PTP_TIMER
WINAPI
CreateThreadpoolTimer(
    _In_ PTP_TIMER_CALLBACK pfnti,
    _Inout_opt_ PVOID pv,
    _In_opt_ PTP_CALLBACK_ENVIRON pcbe
    )
{
    return NULL;
}

WINBASEAPI
VOID
WINAPI
SetThreadpoolTimer(
    _Inout_ PTP_TIMER pti,
    _In_opt_ PFILETIME pftDueTime,
    _In_ DWORD msPeriod,
    _In_opt_ DWORD msWindowLength
    )
{
    return;
}

WINBASEAPI
VOID
WINAPI
WaitForThreadpoolTimerCallbacks(
    _Inout_ PTP_TIMER pti,
    _In_ BOOL fCancelPendingCallbacks
    )
{
    return;
}

WINBASEAPI
VOID
WINAPI
CloseThreadpoolTimer(
    _Inout_ PTP_TIMER pti
    )
{
    return;
}

WINBASEAPI
_Must_inspect_result_
PTP_WAIT
WINAPI
CreateThreadpoolWait(
    _In_ PTP_WAIT_CALLBACK pfnwa,
    _Inout_opt_ PVOID pv,
    _In_opt_ PTP_CALLBACK_ENVIRON pcbe
    )
{
    return NULL;
}

WINBASEAPI
VOID
WINAPI
SetThreadpoolWait(
    _Inout_ PTP_WAIT pwa,
    _In_opt_ HANDLE h,
    _In_opt_ PFILETIME pftTimeout
    )
{
    return;
}

WINBASEAPI
VOID
WINAPI
WaitForThreadpoolWaitCallbacks(
    _Inout_ PTP_WAIT pwa,
    _In_ BOOL fCancelPendingCallbacks
    )
{
    return;
}

WINBASEAPI
VOID
WINAPI
CloseThreadpoolWait(
    _Inout_ PTP_WAIT pwa
    )
{
    return;
}

//
// Errors.
//

WINBASEAPI
_Success_(return != 0)
DWORD
WINAPI
FormatMessageA(
    _In_     DWORD dwFlags,
    _In_opt_ LPCVOID lpSource,
    _In_     DWORD dwMessageId,
    _In_     DWORD dwLanguageId,
    _When_((dwFlags & FORMAT_MESSAGE_ALLOCATE_BUFFER) != 0, _At_((LPSTR*)lpBuffer, _Outptr_result_z_))
    _When_((dwFlags & FORMAT_MESSAGE_ALLOCATE_BUFFER) == 0, _Out_writes_z_(nSize))
             LPSTR lpBuffer,
    _In_     DWORD nSize,
    _In_opt_ va_list *Arguments
    )
{
    return 0;
}

//
// Chm01 stubs.
//

_Must_inspect_result_
_Success_(return >= 0)
HRESULT
PrintCurrentContextStatsChm01(
    _In_ PPERFECT_HASH_CONTEXT Context
    )
{
    return S_OK;
}

VOID
ProcessConsoleCallbackChm01(
    PPERFECT_HASH_CONTEXT Context
    )
{
    return;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :