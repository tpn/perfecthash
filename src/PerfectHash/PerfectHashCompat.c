/*++

Copyright (c) 2023-2026 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashCompat.c

Abstract:

    Implementations of "compat" (i.e. non-Windows) routines.

--*/

#include "stdafx.h"
#include "PerfectHashEventsPrivate.h"
#include <sys/time.h>

#if defined(PH_LINUX) || defined(PH_MAC)
#include "PerfectHashErrors_EnglishBin.h"
#endif

#define GetSystemAllocationGranularity() (max(getpagesize(), 65536))

PSTR
CreateStringFromWide(
    _In_ PCWSTR WideString
    )
{
    INT Index;
    PCHAR Char;
    PSTR String;
    PWCHAR Wide;
    SIZE_T Count;

    Count = wcslen(WideString) + 1;
    String = (PSTR)calloc(1, Count);
    if (!String) {
        SetLastError(ENOMEM);
        return NULL;
    }

    Char = String;
    Wide = (PWCHAR)WideString;
    for (Index = 0; Index < Count; Index++) {
        *Char++ = *Wide++;
    }

    return String;
}

//
// Library.
//

#include <dlfcn.h>

_Ret_maybenull_
HMODULE
LoadLibraryA(
    _In_ LPCSTR LibFileName
    )
{
    PSTR Error;
    PVOID Handle;

    Handle = dlopen(LibFileName, RTLD_LAZY);
    if (Handle == NULL) {
        Error = dlerror();
        if (Error != NULL) {
            fprintf(stderr, "dlopen failed: %s\n", Error);
        }
        SetLastError(PH_E_SYSTEM_CALL_FAILED);
    }

    return (HMODULE)Handle;
}

BOOL
FreeLibrary (
    _In_ HMODULE Module
    )
{
    (VOID)dlclose((PVOID)Module);
    return TRUE;
}

FARPROC
GetProcAddress(
    _In_ HMODULE Module,
    _In_ LPCSTR ProcName
    )
{
    PSTR Error;
    PVOID Proc;

    Proc = dlsym((PVOID)Module, ProcName);
    if (Proc == NULL) {
        Error = dlerror();
        if (Error != NULL) {
            fprintf(stderr,
                    "dlsym(%p, '%s') failed: %s\n",
                    (PVOID)Module,
                    ProcName,
                    Error);
        }
        SetLastError(PH_E_SYSTEM_CALL_FAILED);
    }

    return Proc;
}


//
// Misc.
//

DWORD LastError;

static DWORD
PhGetNumberOfProcessors(
    VOID
    )
{
#ifdef PH_LINUX
    long Count = get_nprocs();
#elif defined(PH_MAC)
    int Count = 0;
    size_t Size = sizeof(Count);
    if (sysctlbyname("hw.logicalcpu", &Count, &Size, NULL, 0) != 0 ||
        Count <= 0) {
        long SysconfCount = sysconf(_SC_NPROCESSORS_ONLN);
        if (SysconfCount > 0) {
            Count = (int)SysconfCount;
        }
    }
#else
    long Count = sysconf(_SC_NPROCESSORS_ONLN);
#endif
    if (Count < 1) {
        Count = 1;
    }
    return (DWORD)Count;
}

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
#ifdef PH_MAC
    uint64_t ThreadId = 0;
    if (pthread_threadid_np(NULL, &ThreadId) == 0) {
        return (DWORD)ThreadId;
    }
#endif
    return (DWORD)(uintptr_t)pthread_self();
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
    UNREFERENCED_PARAMETER(GroupNumber);
    return PhGetNumberOfProcessors();
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
// Time.
//

//
// Win32 Epoch (Jan 1, 1601) to Unix Epoch (Jan 1, 1970).  365 days in a year,
// 369 years between 1601 and 1970, 89 leap days between 1601 and 1970.
//

#define DAYS_BETWEEN_EPOCHS ((ULONGLONG)(((369 * 365) + 89)))

//
// 86400 seconds in a day.
//

#define SECONDS_IN_DAYS_BETWEEN_EPOCHS (DAYS_BETWEEN_EPOCHS * 86400LL)

C_ASSERT(SECONDS_IN_DAYS_BETWEEN_EPOCHS == 11644473600LL);

//
// Convert a struct timeval to a FILETIME.
//

VOID
ConvertStructTimevalToFileTime(
    _In_ struct timeval *Timeval,
    _Out_ LPFILETIME FileTime
    )
{
    ULONGLONG Time;
    ULARGE_INTEGER Quad;

    Time = (Timeval->tv_sec + SECONDS_IN_DAYS_BETWEEN_EPOCHS) * 1e9;

    //
    // Instead of doing the following, just multiply by 10:
    //
    //  Time += ((Timeval->tv_usec * 1000) / 100);
    //

    Time += Timeval->tv_usec * 10;

    Quad.QuadPart = Time;
    FileTime->dwLowDateTime = Quad.LowPart;
    FileTime->dwHighDateTime = Quad.HighPart;
}


VOID
ConvertUnixTimeToSystemTime(
    _In_ struct tm *Time,
    _In_ struct timeval *Timeval,
    _Out_ LPSYSTEMTIME SystemTime
    )
{
    SystemTime->wYear = (WORD)(Time->tm_year + 1900);
    SystemTime->wMonth = (WORD)(Time->tm_mon + 1);
    SystemTime->wDayOfWeek = (WORD)Time->tm_wday;
    SystemTime->wDay = (WORD)Time->tm_mday;
    SystemTime->wHour = (WORD)Time->tm_hour;
    SystemTime->wMinute = (WORD)Time->tm_min;
    SystemTime->wSecond = (WORD)Time->tm_sec;
    SystemTime->wMilliseconds = (WORD)(Timeval->tv_usec / 1000);
}

VOID
GetSystemTime(
    _Out_ LPSYSTEMTIME SystemTime
    )
{
    time_t t;
    struct tm *tmp;
    struct timeval tv;
    struct tm tm_utc = { 0, };

    if (gettimeofday(&tv, NULL) == -1) {
        PH_ERROR(gettimeofday, PH_E_SYSTEM_CALL_FAILED);
        PH_RAISE(PH_E_SYSTEM_CALL_FAILED);
    }

    t = tv.tv_sec;

    tmp = gmtime_r(&t, &tm_utc);
    if (tmp == NULL) {
        PH_ERROR(gmtime, PH_E_SYSTEM_CALL_FAILED);
        PH_RAISE(PH_E_SYSTEM_CALL_FAILED);
    }

    ConvertUnixTimeToSystemTime(tmp, &tv, SystemTime);

    return;
}

VOID
GetSystemTimeAsFileTime(
    _Out_ LPFILETIME FileTime
    )
{
    struct timeval tv;

    if (gettimeofday(&tv, NULL) == -1) {
        PH_ERROR(gettimeofday, PH_E_SYSTEM_CALL_FAILED);
        PH_RAISE(PH_E_SYSTEM_CALL_FAILED);
    }

    ConvertStructTimevalToFileTime(&tv, FileTime);
    return;
}

VOID
GetLocalTime(
    _Out_ LPSYSTEMTIME SystemTime
    )
{
    time_t t;
    struct tm *tmp;
    struct timeval tv;
    struct tm tm_local = { 0, };

    if (gettimeofday(&tv, NULL) == -1) {
        PH_ERROR(gettimeofday, PH_E_SYSTEM_CALL_FAILED);
        PH_RAISE(PH_E_SYSTEM_CALL_FAILED);
    }

    t = tv.tv_sec;

    tmp = localtime_r(&t, &tm_local);
    if (tmp == NULL) {
        PH_ERROR(gmtime, PH_E_SYSTEM_CALL_FAILED);
        PH_RAISE(PH_E_SYSTEM_CALL_FAILED);
    }

    ConvertUnixTimeToSystemTime(tmp, &tv, SystemTime);

    return;
}

//
// Convert a FILETIME to SYSTEMTIME.  Thanks ChatGPT!
//

_Success_(return != FALSE)
BOOL
FileTimeToSystemTime(
    _In_ CONST FILETIME* FileTime,
    _Out_ LPSYSTEMTIME SystemTime
    )
{
    ULONGLONG Days;
    ULONGLONG Seconds;
    ULONGLONG Interval;
    ULARGE_INTEGER Quad;
    ULONGLONG SecondsIntoDay;
    ULONGLONG DaysSinceLeapYear;
    DWORD Year;
    DWORD Month;
    DWORD YearDay;
    DWORD DayOfWeek;
    DWORD LeapYears;
    DWORD DayOfMonth;

    if (FileTime == NULL || SystemTime == NULL) {
        SetLastError(E_INVALIDARG);
        return FALSE;
    }

    //
    // Convert the FILETIME value to the number of 100-nanosecond intervals
    // since January 1, 1601.
    //

    Quad.LowPart = FileTime->dwLowDateTime;
    Quad.HighPart = FileTime->dwHighDateTime;
    Interval = Quad.QuadPart;

    //
    // Calculate the number of seconds since January 1, 1970 (Unix epoch).
    //

    Seconds = Interval / 10000000ULL - SECONDS_IN_DAYS_BETWEEN_EPOCHS;

    //
    // Calculate the number of seconds into the current day.
    //

    SecondsIntoDay = Seconds % 86400;

    //
    // Calculate the number of days since January 1, 1970.
    //

    Days = Seconds / 86400;

    //
    // Calculate the day of the week (0=Sunday, 1=Monday, etc.).
    //

    DayOfWeek = (DWORD)((Days + 1) % 7);

    //
    // Calculate the year, month, and day of the month.
    //

    LeapYears = (DWORD)((Days - 1) / 1461);
    Year = (DWORD)(1970 + 4 * LeapYears);
    DaysSinceLeapYear = Days - LeapYears * 1461;
    if (DaysSinceLeapYear >= 366) {
        LeapYears++;
        Year++;
        DaysSinceLeapYear -= 366;
    }
    YearDay = (DWORD)DaysSinceLeapYear;
    for (Month = 1; Month <= 12; Month++) {
        DWORD DaysInMonth = 31;
        if (Month == 4 || Month == 6 || Month == 9 || Month == 11) {
            DaysInMonth = 30;
        } else if (Month == 2) {
            DaysInMonth = Year % 4 == 0 && (
                Year % 100 != 0 || Year % 400 == 0
            ) ? 29 : 28;
        }
        if (YearDay < DaysInMonth) {
            DayOfMonth = YearDay + 1;
            break;
        }
        YearDay -= DaysInMonth;
    }

    //
    // Set the fields in the SYSTEMTIME structure.
    //

    SystemTime->wYear = (WORD)Year;
    SystemTime->wMonth = (WORD)Month;
    SystemTime->wDayOfWeek = (WORD)DayOfWeek;
    SystemTime->wDay = (WORD)DayOfMonth;
    SystemTime->wHour = (WORD)(SecondsIntoDay / 3600);
    SystemTime->wMinute = (WORD)((SecondsIntoDay % 3600) / 60);
    SystemTime->wSecond = (WORD)(SecondsIntoDay % 60);
    SystemTime->wMilliseconds = (WORD)(Interval % 10000);

    return TRUE;
}

//
// Convert SYSTEMTIME to FILETIME.  Thanks ChatGPT!
//

_Success_(return != FALSE)
BOOL
SystemTimeToFileTime(
    _In_ CONST SYSTEMTIME* SystemTime,
    _Out_ LPFILETIME FileTime
    )
{
    ULONGLONG Seconds;
    ULONGLONG Interval;
    DWORD Year;
    DWORD Month;
    DWORD DayOfWeek;
    DWORD LeapYears;
    DWORD DayOfMonth;
    DWORD DaysInMonth;
    DWORD DaysSince1970;
    DWORD DaysSinceLeapYear;

    if (SystemTime == NULL || FileTime == NULL)
    {
        PH_RAISE(E_INVALIDARG);
        return FALSE;
    }

    //
    // Calculate the number of seconds since January 1, 1970 (Unix epoch).
    //

    Year = SystemTime->wYear;
    Month = SystemTime->wMonth;
    DayOfMonth = SystemTime->wDay;
    LeapYears = (Year - 1969) / 4 - (Year - 1901) / 100 + (Year - 1601) / 400;
    DaysSince1970 = (Year - 1970) * 365 + LeapYears;
    switch (Month)
    {
        case 12:
            DaysSince1970 += 30;
        case 11:
            DaysSince1970 += 31;
        case 10:
            DaysSince1970 += 30;
        case 9:
            DaysSince1970 += 31;
        case 8:
            DaysSince1970 += 31;
        case 7:
            DaysSince1970 += 30;
        case 6:
            DaysSince1970 += 31;
        case 5:
            DaysSince1970 += 30;
        case 4:
            DaysSince1970 += 31;
        case 3:
            DaysSince1970 += Year % 4 == 0 && (
                Year % 100 != 0 || Year % 400 == 0
            ) ? 29 : 28;
        case 2:
            DaysSince1970 += 31;
        case 1:
        default:
            break;
    }

    DaysSince1970 += DayOfMonth - 1;
    DayOfWeek = (DWORD)((DaysSince1970 + 4) % 7);
    Seconds = DaysSince1970 * 86400ULL +
        SystemTime->wHour * 3600ULL +
        SystemTime->wMinute * 60ULL +
        SystemTime->wSecond;

    //
    // Convert the number of seconds to a FILETIME value.
    //

    Interval = Seconds + SECONDS_IN_DAYS_BETWEEN_EPOCHS;
    Interval *= 10000000ULL;
    FileTime->dwLowDateTime = (DWORD)Interval;
    FileTime->dwHighDateTime = (DWORD)(Interval >> 32);

    return TRUE;
}


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
    return TRUE;
}

//
// Files and directories.
//

WINBASEAPI
BOOL
WINAPI
GetFileTime(
    _In_ HANDLE File,
    _Out_opt_ LPFILETIME CreationTime,
    _Out_opt_ LPFILETIME LastAccessTime,
    _Out_opt_ LPFILETIME LastWriteTime
    )
{
    INT Result;
    PH_HANDLE Fd = { 0 };
    struct stat Stat;

    Fd.AsHandle = File;
    Result = fstat(Fd.AsFileDescriptor, &Stat);
    if (Result != 0) {
        SetLastError(errno);
        return FALSE;
    }

    CreationTime->dwLowDateTime = (DWORD)Stat.st_ctime;
    CreationTime->dwHighDateTime = (DWORD)(Stat.st_ctime >> 32);

    LastAccessTime->dwLowDateTime = (DWORD)Stat.st_atime;
    LastAccessTime->dwHighDateTime = (DWORD)(Stat.st_atime >> 32);

    LastWriteTime->dwLowDateTime = (DWORD)Stat.st_mtime;
    LastWriteTime->dwHighDateTime = (DWORD)(Stat.st_mtime >> 32);

    return TRUE;
}


WINBASEAPI
HANDLE
WINAPI
CreateFileW(
    _In_ LPCWSTR FileName,
    _In_ DWORD DesiredAccess,
    _In_ DWORD ShareMode,
    _In_opt_ LPSECURITY_ATTRIBUTES SecurityAttributes,
    _In_ DWORD CreationDisposition,
    _In_ DWORD FlagsAndAttributes,
    _In_opt_ HANDLE TemplateFile
    )
{
    PSTR Path = NULL;
    DWORD Access;
    HANDLE Handle;
    BOOL IsRead;
    BOOL IsWrite;
    BOOL IsReadWrite;
    BOOL HasRetried = FALSE;
    int Flags;
    mode_t Mode;
    PH_HANDLE Fd = { 0 };

    UNREFERENCED_PARAMETER(SecurityAttributes);
    UNREFERENCED_PARAMETER(TemplateFile);

    Flags = 0;
    Handle = NULL;
    Access = DesiredAccess;

    IsRead = BooleanFlagOn(Access, GENERIC_READ);
    IsWrite = BooleanFlagOn(Access, GENERIC_WRITE);
    IsReadWrite = (IsRead && IsWrite);

    if (IsReadWrite) {
        Flags = O_RDWR;
    } else if (IsRead) {
        Flags = O_RDONLY;
    } else if (IsWrite) {
        Flags = O_WRONLY;
    } else {
        PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
    }

    if (CreationDisposition == OPEN_ALWAYS) {
        Flags |= (O_CREAT | O_EXCL);
    } else if (CreationDisposition == OPEN_EXISTING) {
        NOTHING;
    } else {
        PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
    }

    Path = CreateStringFromWide(FileName);
    if (!Path) {
        goto End;
    }

Retry:
    Mode = 0664;
    if ((Fd.AsFileDescriptor = open(Path, Flags, Mode)) == -1) {
        if (errno != EEXIST) {
            SetLastError(errno);
            goto End;
        } else if (!HasRetried && CreationDisposition == OPEN_ALWAYS) {
            SetLastError(ERROR_ALREADY_EXISTS);
            HasRetried = TRUE;

            //
            // Remove the O_CREAT and O_EXCL flags and try again.
            //

            Flags &= ~(O_CREAT | O_EXCL);
            goto Retry;
        } else {
            SetLastError(errno);
        }
    } else {
        Handle = Fd.AsHandle;
    }

End:

    FREE_PTR(&Path);

    return Handle;
}

WINBASEAPI
BOOL
WINAPI
DeleteFileW(
    _In_ LPCWSTR FileName
    )
{
    BOOL Success;
    PSTR Path;

    Success = FALSE;
    Path = CreateStringFromWide(FileName);
    if (!Path) {
        goto End;
    }

    if (unlink(Path) == -1) {
        SetLastError(errno);
        goto End;
    }

    Success = TRUE;

End:

    FREE_PTR(&Path);

    return Success;
}

BOOL
CloseFile(
    _In_ _Post_ptr_invalid_ HANDLE Object
    )
{
    PH_HANDLE Fd = { 0 };

    Fd.AsHandle = Object;
    if (close(Fd.AsFileDescriptor) == -1) {
        SetLastError(errno);
        return FALSE;
    }

    return TRUE;
}

WINBASEAPI
BOOL
WINAPI
CreateDirectoryW(
    _In_ LPCWSTR PathName,
    _In_opt_ LPSECURITY_ATTRIBUTES SecurityAttributes
    )
{
    PSTR Path;
    INT Result;

    Path = CreateStringFromWide(PathName);
    if (!Path) {
        return FALSE;
    }

    Result = mkdir(Path, 0755);

    if (Result != 0) {
        if (errno == EEXIST) {
            SetLastError(ERROR_ALREADY_EXISTS);
        } else {
            SetLastError(errno);
        }
    }

    FREE_PTR(&Path);

    return (Result == 0);
}

BOOL
CloseDirectory(
    _In_ HANDLE DirectoryHandle
    )
{
    DIR* Directory = (DIR*)DirectoryHandle;
    INT Result;

    if (Directory) {
        Result = closedir(Directory);
        if (Result != 0) {
            SetLastError(errno);
            return FALSE;
        }
    }

    return TRUE;
}

WINBASEAPI
BOOL
WINAPI
RemoveDirectoryW(
    _In_ LPCWSTR PathName
    )
{
    PSTR Path;
    INT Result;

    Path = CreateStringFromWide(PathName);
    if (!Path) {
        return FALSE;
    }

    Result = rmdir(Path);

    if (Result != 0) {
        if (errno == ENOTEMPTY) {
            SetLastError(ERROR_DIR_NOT_EMPTY);
        } else {
            SetLastError(errno);
        }
    }

    FREE_PTR(&Path);

    return (Result == 0);
}

typedef struct _PH_FILE_MAPPING {
    ULONG Signature;
    int FileDescriptor;
} PH_FILE_MAPPING, *PPH_FILE_MAPPING;

#define PH_FILE_MAPPING_SIGNATURE 0x4D504846u // 'FHP M'

typedef struct _PH_MAPPED_VIEW {
    void *Address;
    size_t Size;
    struct _PH_MAPPED_VIEW *Next;
} PH_MAPPED_VIEW, *PPH_MAPPED_VIEW;

static pthread_mutex_t PhMappedViewLock = PTHREAD_MUTEX_INITIALIZER;
static PPH_MAPPED_VIEW PhMappedViewList;

typedef struct _PH_VIRTUAL_ALLOC {
    void *Address;
    size_t Size;
    struct _PH_VIRTUAL_ALLOC *Next;
} PH_VIRTUAL_ALLOC, *PPH_VIRTUAL_ALLOC;

static pthread_mutex_t PhVirtualAllocLock = PTHREAD_MUTEX_INITIALIZER;
static PPH_VIRTUAL_ALLOC PhVirtualAllocList;

static VOID
PhTrackMappedView(
    _In_ void *Address,
    _In_ size_t Size
    )
{
    PPH_MAPPED_VIEW View;

    View = (PPH_MAPPED_VIEW)calloc(1, sizeof(*View));
    if (!View) {
        return;
    }

    View->Address = Address;
    View->Size = Size;

    pthread_mutex_lock(&PhMappedViewLock);
    View->Next = PhMappedViewList;
    PhMappedViewList = View;
    pthread_mutex_unlock(&PhMappedViewLock);
}

static BOOL
PhUntrackMappedView(
    _In_ void *Address,
    _Out_ size_t *Size
    )
{
    BOOL Found;
    PPH_MAPPED_VIEW Prev;
    PPH_MAPPED_VIEW View;

    Found = FALSE;
    Prev = NULL;

    pthread_mutex_lock(&PhMappedViewLock);
    View = PhMappedViewList;
    while (View) {
        if (View->Address == Address) {
            if (Prev) {
                Prev->Next = View->Next;
            } else {
                PhMappedViewList = View->Next;
            }
            *Size = View->Size;
            free(View);
            Found = TRUE;
            break;
        }
        Prev = View;
        View = View->Next;
    }
    pthread_mutex_unlock(&PhMappedViewLock);

    return Found;
}

static VOID
PhTrackVirtualAlloc(
    _In_ void *Address,
    _In_ size_t Size
    )
{
    PPH_VIRTUAL_ALLOC Entry;

    Entry = (PPH_VIRTUAL_ALLOC)calloc(1, sizeof(*Entry));
    if (!Entry) {
        return;
    }

    Entry->Address = Address;
    Entry->Size = Size;

    pthread_mutex_lock(&PhVirtualAllocLock);
    Entry->Next = PhVirtualAllocList;
    PhVirtualAllocList = Entry;
    pthread_mutex_unlock(&PhVirtualAllocLock);
}

static BOOL
PhUntrackVirtualAlloc(
    _In_ void *Address,
    _Out_ size_t *Size
    )
{
    BOOL Found;
    PPH_VIRTUAL_ALLOC Prev;
    PPH_VIRTUAL_ALLOC Entry;

    Found = FALSE;
    Prev = NULL;

    pthread_mutex_lock(&PhVirtualAllocLock);
    Entry = PhVirtualAllocList;
    while (Entry) {
        if (Entry->Address == Address) {
            if (Prev) {
                Prev->Next = Entry->Next;
            } else {
                PhVirtualAllocList = Entry->Next;
            }
            *Size = Entry->Size;
            free(Entry);
            Found = TRUE;
            break;
        }
        Prev = Entry;
        Entry = Entry->Next;
    }
    pthread_mutex_unlock(&PhVirtualAllocLock);

    return Found;
}

static BOOL
PhTryGetFileMappingFd(
    _In_ HANDLE MappingHandle,
    _Out_ int *FileDescriptor
    )
{
    PPH_FILE_MAPPING Mapping;

    Mapping = (PPH_FILE_MAPPING)MappingHandle;
    if (!Mapping || Mapping == INVALID_HANDLE_VALUE) {
        return FALSE;
    }

    if (Mapping->Signature == PH_FILE_MAPPING_SIGNATURE) {
        *FileDescriptor = Mapping->FileDescriptor;
        return TRUE;
    }

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
    int fd;
    int Prot;
    int Flags;
    off_t Offset;
    size_t Size;
    void *Address;
    struct stat Stat;

    if (!PhTryGetFileMappingFd(hFileMappingObject, &fd)) {
        fd = (int)(intptr_t)hFileMappingObject;
    }

    Offset = ((off_t)dwFileOffsetHigh << 32) | dwFileOffsetLow;
    Size = (size_t)dwNumberOfBytesToMap;

    if (Size == 0) {
        if (fstat(fd, &Stat) != 0) {
            SetLastError(errno);
            return NULL;
        }
        if ((off_t)Stat.st_size <= Offset) {
            SetLastError(EINVAL);
            return NULL;
        }
        Size = (size_t)(Stat.st_size - Offset);
    }

    Prot = 0;
    if (dwDesiredAccess & FILE_MAP_READ) {
        Prot |= PROT_READ;
    }
    if (dwDesiredAccess & FILE_MAP_WRITE) {
        Prot |= PROT_WRITE;
    }
    if (dwDesiredAccess & FILE_MAP_EXECUTE) {
        Prot |= PROT_EXEC;
    }
    if (Prot == 0) {
        Prot = PROT_READ;
    }

    Flags = MAP_SHARED;

    Address = mmap(NULL, Size, Prot, Flags, fd, Offset);
    if (Address == MAP_FAILED) {
        SetLastError(errno);
        return NULL;
    }

    PhTrackMappedView(Address, Size);
    return Address;
}

WINBASEAPI
_Ret_maybenull_
HANDLE
WINAPI
CreateFileMappingW(
    _In_ HANDLE File,
    _In_opt_ LPSECURITY_ATTRIBUTES FileMappingAttributes,
    _In_ DWORD Protect,
    _In_ DWORD MaximumSizeHigh,
    _In_ DWORD MaximumSizeLow,
    _In_opt_ LPCWSTR Name
    )
{
    PH_HANDLE Fd = { 0 };
    PPH_FILE_MAPPING Mapping;
    int DupFd;

    ASSERT(FileMappingAttributes == NULL);
    ASSERT(Name == NULL);
    ASSERT(MaximumSizeHigh == 0);
    ASSERT(MaximumSizeLow == 0);

    if (File == INVALID_HANDLE_VALUE || File == NULL) {
        SetLastError(EBADF);
        return NULL;
    }

    Fd.AsHandle = File;
    DupFd = dup(Fd.AsFileDescriptor);
    if (DupFd == -1) {
        SetLastError(errno);
        return NULL;
    }

    Mapping = (PPH_FILE_MAPPING)calloc(1, sizeof(*Mapping));
    if (!Mapping) {
        close(DupFd);
        SetLastError(ENOMEM);
        return NULL;
    }

    Mapping->Signature = PH_FILE_MAPPING_SIGNATURE;
    Mapping->FileDescriptor = DupFd;

    UNREFERENCED_PARAMETER(Protect);

    return (HANDLE)Mapping;
}


WINBASEAPI
BOOL
WINAPI
UnmapViewOfFile(
    _In_ LPCVOID lpBaseAddress
    )
{
    size_t Size;

    if (!PhUntrackMappedView((void *)lpBaseAddress, &Size)) {
        SetLastError(EINVAL);
        return FALSE;
    }

    if (munmap((void *)lpBaseAddress, Size) != 0) {
        SetLastError(errno);
        SYS_ERROR(munmap);
        return FALSE;
    }

    return TRUE;
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
    int fd;
    int Prot;
    int Flags;
    off_t Offset;
    size_t Size;
    void *Address;
    struct stat Stat;

    if (!PhTryGetFileMappingFd(hFileMappingObject, &fd)) {
        fd = (int)(intptr_t)hFileMappingObject;
    }

    Offset = ((off_t)dwFileOffsetHigh << 32) | dwFileOffsetLow;
    Size = (size_t)dwNumberOfBytesToMap;

    if (Size == 0) {
        if (fstat(fd, &Stat) != 0) {
            SetLastError(errno);
            return NULL;
        }
        if ((off_t)Stat.st_size <= Offset) {
            SetLastError(EINVAL);
            return NULL;
        }
        Size = (size_t)(Stat.st_size - Offset);
    }

    Prot = 0;
    if (dwDesiredAccess & FILE_MAP_READ) {
        Prot |= PROT_READ;
    }
    if (dwDesiredAccess & FILE_MAP_WRITE) {
        Prot |= PROT_WRITE;
    }
    if (dwDesiredAccess & FILE_MAP_EXECUTE) {
        Prot |= PROT_EXEC;
    }
    if (Prot == 0) {
        Prot = PROT_READ;
    }

    Flags = MAP_SHARED;

    Address = mmap(lpBaseAddress, Size, Prot, Flags, fd, Offset);
    if (Address == MAP_FAILED) {
        SetLastError(errno);
        return NULL;
    }

    PhTrackMappedView(Address, Size);
    return Address;
}

WINBASEAPI
BOOL
WINAPI
SetFilePointerEx(
    _In_ HANDLE File,
    _In_ LARGE_INTEGER DistanceToMove,
    _Out_opt_ PLARGE_INTEGER NewFilePointer,
    _In_ DWORD MoveMethod
    )
{
    INT Error;
    PH_HANDLE Fd = { 0 };
    off_t Result;
    struct stat Stat;

    ASSERT(NewFilePointer == NULL);
    ASSERT(MoveMethod == FILE_BEGIN);

    Fd.AsHandle = File;

    Error = fstat(Fd.AsFileDescriptor, &Stat);
    if (Error != 0) {
        SetLastError(errno);
        return FALSE;
    }

    if (Stat.st_size > DistanceToMove.QuadPart) {
        Error = ftruncate(Fd.AsFileDescriptor, DistanceToMove.QuadPart);
        if (Error != 0) {
            SetLastError(errno);
            return FALSE;
        }
    } else if (Stat.st_size < DistanceToMove.QuadPart) {

#ifdef PH_MAC
        Error = ftruncate(Fd.AsFileDescriptor, DistanceToMove.QuadPart);
        if (Error != 0) {
            SetLastError(errno);
            return FALSE;
        }
#else
        Error = posix_fallocate(Fd.AsFileDescriptor,
                                0,
                                DistanceToMove.QuadPart);
        if (Error != 0) {
            if (Error == EINVAL && DistanceToMove.QuadPart == 0) {
                NOTHING;
            } else {
                SetLastError(Error);
                return FALSE;
            }
        }
#endif

        Result = lseek(Fd.AsFileDescriptor, DistanceToMove.QuadPart, SEEK_SET);
        if (Result == -1) {
            SetLastError(errno);
            return FALSE;
        }
    }

    return TRUE;
}

WINBASEAPI
BOOL
WINAPI
SetEndOfFile(
    _In_ HANDLE hFile
    )
{
    //
    // This is a no-op on POSIX as we've already set the end of file in
    // SetFilePointerEx.
    //

    return TRUE;
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
    int Result;
    BOOL ReplaceExisting;
    PSTR ExistingPath;
    PSTR NewPath;
    struct stat Stat;

    ReplaceExisting = BooleanFlagOn(dwFlags, MOVEFILE_REPLACE_EXISTING);

    if (!lpExistingFileName || !lpNewFileName) {
        SetLastError(EINVAL);
        return FALSE;
    }

    ExistingPath = CreateStringFromWide(lpExistingFileName);
    if (!ExistingPath) {
        return FALSE;
    }

    NewPath = CreateStringFromWide(lpNewFileName);
    if (!NewPath) {
        FREE_PTR(&ExistingPath);
        return FALSE;
    }

    if (!ReplaceExisting && stat(NewPath, &Stat) == 0) {
        FREE_PTR(&ExistingPath);
        FREE_PTR(&NewPath);
        SetLastError(EEXIST);
        return FALSE;
    }

    Result = rename(ExistingPath, NewPath);
    if (Result != 0) {
        SetLastError(errno);
    }

    FREE_PTR(&ExistingPath);
    FREE_PTR(&NewPath);

    return (Result == 0);
}

WINBASEAPI
BOOL
WINAPI
GetFileInformationByHandle(
    _In_ HANDLE hFile,
    _Out_ LPBY_HANDLE_FILE_INFORMATION lpFileInformation
    )
{
    INT Result;
    PH_HANDLE Fd = { 0 };
    struct stat Stat;

    if (!lpFileInformation) {
        SetLastError(EINVAL);
        return FALSE;
    }

    Fd.AsHandle = hFile;
    Result = fstat(Fd.AsFileDescriptor, &Stat);
    if (Result != 0) {
        SetLastError(errno);
        return FALSE;
    }

    ZeroStructPointerInline(lpFileInformation);

    if (S_ISDIR(Stat.st_mode)) {
        lpFileInformation->dwFileAttributes = FILE_ATTRIBUTE_DIRECTORY;
    } else {
        lpFileInformation->dwFileAttributes = FILE_ATTRIBUTE_NORMAL;
    }

    lpFileInformation->ftCreationTime.dwLowDateTime = (DWORD)Stat.st_ctime;
    lpFileInformation->ftCreationTime.dwHighDateTime = (DWORD)(Stat.st_ctime >> 32);
    lpFileInformation->ftLastAccessTime.dwLowDateTime = (DWORD)Stat.st_atime;
    lpFileInformation->ftLastAccessTime.dwHighDateTime = (DWORD)(Stat.st_atime >> 32);
    lpFileInformation->ftLastWriteTime.dwLowDateTime = (DWORD)Stat.st_mtime;
    lpFileInformation->ftLastWriteTime.dwHighDateTime = (DWORD)(Stat.st_mtime >> 32);
    lpFileInformation->nFileSizeHigh = (DWORD)((uint64_t)Stat.st_size >> 32);
    lpFileInformation->nFileSizeLow = (DWORD)((uint64_t)Stat.st_size & 0xffffffffu);
    lpFileInformation->nNumberOfLinks = (DWORD)Stat.st_nlink;
    lpFileInformation->nFileIndexHigh = (DWORD)((uint64_t)Stat.st_ino >> 32);
    lpFileInformation->nFileIndexLow = (DWORD)((uint64_t)Stat.st_ino & 0xffffffffu);

    return TRUE;
}

WINBASEAPI
BOOL
WINAPI
GetFileInformationByHandleEx(
    _In_  HANDLE File,
    _In_  FILE_INFO_BY_HANDLE_CLASS FileInformationClass,
    _Out_writes_bytes_(BufferSize) LPVOID FileInformation,
    _In_  DWORD BufferSize
    )
{
    PH_HANDLE Fd = { 0 };
    if (FileInformationClass == FileStandardInfo) {
        INT Result;
        struct stat Stat;
        PFILE_STANDARD_INFO StandardInfo = (PFILE_STANDARD_INFO)FileInformation;

        Fd.AsHandle = File;
        Result = fstat(Fd.AsFileDescriptor, &Stat);
        if (Result != 0) {
            SetLastError(errno);
            return FALSE;
        }

        StandardInfo->AllocationSize.QuadPart = Stat.st_size;
        StandardInfo->EndOfFile.QuadPart = Stat.st_size;

        return TRUE;
    } else {
        PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
        return FALSE;
    }
}


WINBASEAPI
BOOL
WINAPI
CloseHandle(
    _In_ _Post_ptr_invalid_ HANDLE hObject
    )
{
    int Result;
    int FileDescriptor;
    PPH_FILE_MAPPING Mapping;

    if (hObject == NULL || hObject == INVALID_HANDLE_VALUE) {
        SetLastError(EBADF);
        return FALSE;
    }

    if (hObject == (HANDLE)stdin ||
        hObject == (HANDLE)stdout ||
        hObject == (HANDLE)stderr) {
        return TRUE;
    }

    if (PhTryGetFileMappingFd(hObject, &FileDescriptor)) {
        Mapping = (PPH_FILE_MAPPING)hObject;
        Result = close(FileDescriptor);
        free(Mapping);
        return (Result == 0);
    }

    if ((uintptr_t)hObject <= INT_MAX) {
        Result = close((int)(intptr_t)hObject);
        if (Result != 0) {
            SetLastError(errno);
            return FALSE;
        }
        return TRUE;
    }

    if (((PPH_EVENT)hObject)->Signature == PH_EVENT_SIGNATURE) {
        return CloseEvent(hObject);
    }

    Result = close((int)(intptr_t)hObject);
    if (Result != 0) {
        SetLastError(errno);
        return FALSE;
    }

    return TRUE;
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
    int fd;
    ssize_t BytesWritten;

    UNREFERENCED_PARAMETER(lpOverlapped);

    if (hFile == (HANDLE)stdout) {
        fd = fileno(stdout);
    } else if (hFile == (HANDLE)stderr) {
        fd = fileno(stderr);
    } else if (hFile == (HANDLE)stdin) {
        fd = fileno(stdin);
    } else {
        fd = (int)(intptr_t)hFile;
    }

    BytesWritten = write(fd, lpBuffer, nNumberOfBytesToWrite);

    if (ARGUMENT_PRESENT(lpNumberOfBytesWritten)) {
        if (BytesWritten < 0) {
            *lpNumberOfBytesWritten = 0;
        } else {
            *lpNumberOfBytesWritten = (DWORD)BytesWritten;
        }
    }

    if (BytesWritten < 0) {
        __debugbreak();
        SetLastError(errno);
        SYS_ERROR(write);
        return FALSE;
    }

    return TRUE;
}

typedef struct _PH_FIND_HANDLE {
    PCSTR Extension;
    PCSTR DirectoryName;
    DIR *Directory;
    struct dirent Entry;
    struct dirent *Result;
} PH_FIND_HANDLE, *PPH_FIND_HANDLE;

FORCEINLINE
BOOLEAN
StringEndsWith(PCSTR String, PCSTR Suffix)
{
    BOOLEAN EndsWith;
    SIZE_T StringLength;
    SIZE_T SuffixLength;

    StringLength = strlen(String);
    SuffixLength = strlen(Suffix);

    if (StringLength < SuffixLength) {
        return FALSE;
    }

    EndsWith = (
        strncmp(String + StringLength - SuffixLength,
                Suffix,
                SuffixLength) == 0
    );

    return EndsWith;
}

WINBASEAPI
BOOL
WINAPI
FindNextFileW(
    _In_ HANDLE FindHandle,
    _Out_ LPWIN32_FIND_DATAW FindFileData
    )
{
    LONG Index;
    LONG Result;
    LONG Length;
    PWSTR Dest;
    PCSTR Source;
    PCSTR FileName;
    PPH_FIND_HANDLE Handle;

    Handle = (PPH_FIND_HANDLE)FindHandle;

    while (TRUE) {

        Result = readdir_r(Handle->Directory,
                           &Handle->Entry,
                           &Handle->Result);

        if (Result != 0) {
            SetLastError(Result);
            return FALSE;
        } else if (!Handle->Result) {
            return FALSE;
        }

        FileName = Handle->Entry.d_name;
        if (!StringEndsWith(FileName, Handle->Extension)) {
            continue;
        }

        Length = strlen(FileName);
        if (Length >= MAX_PATH) {
            fprintf(stderr, "Path too long: %s\n", FileName);
            continue;
        }

        Dest = &FindFileData->cFileName[0];
        Source = FileName;
        for (Index = 0; Index < Length; Index++) {
            *Dest++ = (WCHAR)*Source++;
        }
        *Dest = '\0';

        break;
    }

    return TRUE;
}

WINBASEAPI
HANDLE
WINAPI
FindFirstFileW(
    _In_ LPCWSTR WildcardPath,
    _Out_ LPWIN32_FIND_DATAW FindFileData
    )
{
    PSTR Path;
    LONG Index;
    LONG Length;
    LONG Result;
    BOOL FoundDot;
    BOOL FoundSep;
    PSTR FileName;
    PPH_FIND_HANDLE Handle;

    Handle = (PPH_FIND_HANDLE)calloc(1, sizeof(*Handle));
    if (!Handle) {
        SetLastError(ENOMEM);
        FREE_PTR(&Path);
        goto End;
    }

    Path = CreateStringFromWide(WildcardPath);
    if (!Path) {
        goto End;
    }

    //
    // Reverse through the path until we find the last dot, then continue
    // and look for the last path separator.
    //

    FoundDot = FALSE;
    FoundSep = FALSE;
    Length = strlen(Path);

    for (Index = Length - 1; Index >= 0; Index--) {
        if (!FoundDot) {
            if (Path[Index] == '.') {
                ASSERT(Index != Length - 1);
                ASSERT(Handle->Extension == NULL);
                FoundDot = TRUE;
                Handle->Extension = &Path[Index];
            }
        } else if (Path[Index] == PATHSEP_A) {
            ASSERT(Index != Length - 1);
            ASSERT(Handle->DirectoryName == NULL);
            FoundSep = TRUE;
            Handle->DirectoryName = &Path[0];
            Path[Index] = '\0';
            break;
        }
    }

    if (!FoundDot || !FoundSep) {
        PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
    }

    Handle->Directory = opendir(Handle->DirectoryName);
    if (!Handle->Directory) {
        if (errno == ENOENT) {
            SetLastError(ERROR_FILE_NOT_FOUND);
        } else {
            SetLastError(errno);
        }
        goto Error;
    }

    SetLastError(ERROR_SUCCESS);
    if (!FindNextFileW((HANDLE)Handle, FindFileData)) {
        if (GetLastError() == ERROR_SUCCESS) {
            SetLastError(ERROR_FILE_NOT_FOUND);
        }
        goto Error;
    }

    goto End;

Error:

    if (Handle != NULL) {
        FindClose((HANDLE)Handle);
        Handle = NULL;
    }

End:

    return (HANDLE)Handle;
}


WINBASEAPI
BOOL
WINAPI
FindClose(
    _Inout_ HANDLE FindFile
    )
{
    PPH_FIND_HANDLE Handle;

    Handle = (PPH_FIND_HANDLE)FindFile;

    if (!Handle) {
        SetLastError(E_INVALIDARG);
        return FALSE;
    }

    if (Handle->Directory) {
        closedir(Handle->Directory);
    }

    free(Handle);
    Handle = NULL;
    return TRUE;
}

//
// SRW locks.
//

WINBASEAPI
VOID
WINAPI
InitializeSRWLock(
    _Out_ PSRWLOCK SRWLock
    )
{
    LastError = pthread_rwlock_init(SRWLock, NULL);
    if (LastError != 0) {
        PH_RAISE(PH_E_SYSTEM_CALL_FAILED);
    }
}

static int
PhEnsureSrwLockInitialized(
    _Inout_ PSRWLOCK SRWLock
    )
{
    int Result;

    Result = pthread_rwlock_init(SRWLock, NULL);
    if (Result == 0) {
        return 0;
    }

    if (Result == EBUSY) {
        return 0;
    }

    return Result;
}

WINBASEAPI
_Releases_exclusive_lock_(*SRWLock)
VOID
WINAPI
ReleaseSRWLockExclusive(
    _Inout_ PSRWLOCK SRWLock
    )
{
    LastError = pthread_rwlock_unlock(SRWLock);
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
    LastError = pthread_rwlock_unlock(SRWLock);
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
    LastError = pthread_rwlock_wrlock(SRWLock);
    if (LastError == EINVAL) {
        int InitError = PhEnsureSrwLockInitialized(SRWLock);
        if (InitError == 0) {
            LastError = pthread_rwlock_wrlock(SRWLock);
        } else {
            LastError = InitError;
        }
    }
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
    LastError = pthread_rwlock_rdlock(SRWLock);
    if (LastError == EINVAL) {
        int InitError = PhEnsureSrwLockInitialized(SRWLock);
        if (InitError == 0) {
            LastError = pthread_rwlock_rdlock(SRWLock);
        } else {
            LastError = InitError;
        }
    }
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
    LastError = pthread_rwlock_trywrlock(SRWLock);
    if (LastError == EINVAL) {
        int InitError = PhEnsureSrwLockInitialized(SRWLock);
        if (InitError == 0) {
            LastError = pthread_rwlock_trywrlock(SRWLock);
        } else {
            LastError = InitError;
        }
    }
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
    LastError = pthread_rwlock_tryrdlock(SRWLock);
    if (LastError == EINVAL) {
        int InitError = PhEnsureSrwLockInitialized(SRWLock);
        if (InitError == 0) {
            LastError = pthread_rwlock_tryrdlock(SRWLock);
        } else {
            LastError = InitError;
        }
    }
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
VOID
WINAPI
InitOnceInitialize(
    _Out_ PINIT_ONCE InitOnce
    )
{
    static const pthread_once_t PhOnceInit = PTHREAD_ONCE_INIT;
    InitOnce->Once = PhOnceInit;
    InitOnce->Context = NULL;
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
    InitOnceInitOnce = InitOnce;
    InitOnceContext = Context;
    InitOnceFunction = InitFn;
    InitOnceParameter = Parameter;
    pthread_once((pthread_once_t *)InitOnce, InitOnceWrapper);
    if (ARGUMENT_PRESENT(Context)) {
        *Context = InitOnce->Context;
    }
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
    lpSystemInfo->dwPageSize = (DWORD)sysconf(_SC_PAGESIZE);
    lpSystemInfo->dwNumberOfProcessors = PhGetNumberOfProcessors();
    lpSystemInfo->dwAllocationGranularity = GetSystemAllocationGranularity();
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

    Event->Signature = PH_EVENT_SIGNATURE;

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

BOOL
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
        SetLastError(EBADF);
        goto End;
    }

    if (Event->Signature != PH_EVENT_SIGNATURE) {
        SetLastError(EBADF);
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

    if (Event->Signature != PH_EVENT_SIGNATURE) {
        SetLastError(EBADF);
        goto End;
    }

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

    if (Event->Signature != PH_EVENT_SIGNATURE) {
        SetLastError(EBADF);
        goto End;
    }

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

    ASSERT((dwMilliseconds == INFINITE) ||
           (dwMilliseconds == 0));

    Event = (PPH_EVENT)hHandle;
    WaitResult = WAIT_FAILED;

    if (Event->Signature != PH_EVENT_SIGNATURE) {
        SetLastError(EBADF);
        goto End;
    }

    Error = pthread_mutex_lock(&Event->Mutex);
    if (Error != 0) {
        SetLastError(Error);
        SYS_ERROR(pthread_mutex_lock);
        goto End;
    }

    if (dwMilliseconds == 0) {
        if (Event->State.Signaled == FALSE) {
            WaitResult = WAIT_TIMEOUT;
        } else {
            WaitResult = WAIT_OBJECT_0;
        }
    } else {

        while (Event->State.Signaled == FALSE) {

            Error = pthread_cond_wait(&Event->Condition, &Event->Mutex);
            if (Error != 0) {
                SetLastError(Error);
                SYS_ERROR(pthread_cond_wait);
                goto Unlock;
            }
        }

        WaitResult = WAIT_OBJECT_0;
    }

    if (Event->State.ManualReset == FALSE) {
        Event->State.Signaled = FALSE;
    }

Unlock:
    Error = pthread_mutex_unlock(&Event->Mutex);
    if (Error != 0) {
        SetLastError(Error);
        SYS_ERROR(pthread_mutex_unlock);
        goto End;
    }

End:

    return WaitResult;
}

WINBASEAPI
DWORD
WINAPI
WaitForMultipleObjects(
    _In_ DWORD Count,
    _In_reads_(Count) CONST HANDLE* Handles,
    _In_ BOOL WaitAll,
    _In_ DWORD Milliseconds
    )
{
    DWORD Index;
    DWORD WaitResult;

    if (WaitAll == FALSE) {
        return WAIT_FAILED;
    }

    for (Index = 0; Index < Count; Index++) {
        WaitResult = WaitForSingleObject(Handles[Index], Milliseconds);
        if (WaitResult != WAIT_OBJECT_0) {
            return WaitResult;
        }
    }

    return WAIT_OBJECT_0;
}


//
// Memory.
//


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

#ifndef PH_WINDOWS

typedef struct _PH_HEAP_ALLOCATION {
    struct _PH_HEAP_ALLOCATION *Next;
    struct _PH_HEAP_ALLOCATION *Prev;
    struct _PH_HEAP *Heap;
    SIZE_T Size;
} PH_HEAP_ALLOCATION;
typedef PH_HEAP_ALLOCATION *PPH_HEAP_ALLOCATION;

typedef struct _PH_HEAP {
    ULONG Magic;
    SRWLOCK Lock;
    PPH_HEAP_ALLOCATION Head;
} PH_HEAP;
typedef PH_HEAP *PPH_HEAP;

#define PH_HEAP_MAGIC 0x50484850u

static
PPH_HEAP
GetPerfectHashHeap(
    _In_opt_ HANDLE Heap
    )
{
    PPH_HEAP MaybeHeap = (PPH_HEAP)Heap;

    if (!MaybeHeap || MaybeHeap->Magic != PH_HEAP_MAGIC) {
        return NULL;
    }

    return MaybeHeap;
}

WINBASEAPI
HANDLE
WINAPI
HeapCreate(
    _In_ DWORD flOptions,
    _In_ SIZE_T dwInitialSize,
    _In_ SIZE_T dwMaximumSize
    )
{
    PPH_HEAP Heap;

    UNREFERENCED_PARAMETER(flOptions);
    UNREFERENCED_PARAMETER(dwInitialSize);
    UNREFERENCED_PARAMETER(dwMaximumSize);

    Heap = (PPH_HEAP)calloc(1, sizeof(*Heap));
    if (!Heap) {
        SetLastError(ENOMEM);
        return NULL;
    }

    Heap->Magic = PH_HEAP_MAGIC;
    InitializeSRWLock(&Heap->Lock);

    return (HANDLE)Heap;
}

WINBASEAPI
BOOL
WINAPI
HeapDestroy(
    _In_ HANDLE hHeap
    )
{
    PPH_HEAP Heap;
    PPH_HEAP_ALLOCATION Allocation;
    PPH_HEAP_ALLOCATION Next;

    Heap = GetPerfectHashHeap(hHeap);
    if (!Heap) {
        return FALSE;
    }

    AcquireSRWLockExclusive(&Heap->Lock);

    Allocation = Heap->Head;
    while (Allocation) {
        Next = Allocation->Next;
        free(Allocation);
        Allocation = Next;
    }

    Heap->Head = NULL;

    ReleaseSRWLockExclusive(&Heap->Lock);

    free(Heap);

    return TRUE;
}

#endif // PH_WINDOWS

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
    PPH_HEAP PerfectHeap;
    PPH_HEAP_ALLOCATION Allocation;
    SIZE_T TotalSize;

    PerfectHeap = GetPerfectHashHeap(Heap);
    if (PerfectHeap) {
        TotalSize = sizeof(*Allocation) + SizeInBytes;
        if ((Flags & HEAP_ZERO_MEMORY) != 0) {
            Allocation = (PPH_HEAP_ALLOCATION)calloc(1, TotalSize);
        } else {
            Allocation = (PPH_HEAP_ALLOCATION)malloc(TotalSize);
        }

        if (!Allocation) {
            SetLastError(ENOMEM);
            return NULL;
        }

        Allocation->Heap = PerfectHeap;
        Allocation->Size = SizeInBytes;

        AcquireSRWLockExclusive(&PerfectHeap->Lock);

        Allocation->Next = PerfectHeap->Head;
        Allocation->Prev = NULL;
        if (PerfectHeap->Head) {
            PerfectHeap->Head->Prev = Allocation;
        }
        PerfectHeap->Head = Allocation;

        ReleaseSRWLockExclusive(&PerfectHeap->Lock);

        return (LPVOID)(Allocation + 1);
    }

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
    PPH_HEAP PerfectHeap;
    PPH_HEAP_ALLOCATION Allocation;
    PPH_HEAP_ALLOCATION NewAllocation;
    SIZE_T TotalSize;
    SIZE_T OldSize;

    PerfectHeap = GetPerfectHashHeap(Heap);
    if (PerfectHeap) {
        if (!Mem) {
            return HeapAlloc(Heap, Flags, SizeInBytes);
        }

        if (SizeInBytes == 0) {
            HeapFree(Heap, 0, Mem);
            return NULL;
        }

        Allocation = ((PPH_HEAP_ALLOCATION)Mem) - 1;
        OldSize = Allocation->Size;
        TotalSize = sizeof(*Allocation) + SizeInBytes;

        AcquireSRWLockExclusive(&PerfectHeap->Lock);

        if (Allocation->Prev) {
            Allocation->Prev->Next = Allocation->Next;
        } else {
            PerfectHeap->Head = Allocation->Next;
        }
        if (Allocation->Next) {
            Allocation->Next->Prev = Allocation->Prev;
        }

        ReleaseSRWLockExclusive(&PerfectHeap->Lock);

        NewAllocation = (PPH_HEAP_ALLOCATION)realloc(Allocation, TotalSize);
        if (!NewAllocation) {
            AcquireSRWLockExclusive(&PerfectHeap->Lock);

            Allocation->Next = PerfectHeap->Head;
            Allocation->Prev = NULL;
            if (PerfectHeap->Head) {
                PerfectHeap->Head->Prev = Allocation;
            }
            PerfectHeap->Head = Allocation;

            ReleaseSRWLockExclusive(&PerfectHeap->Lock);

            SetLastError(ENOMEM);
            return NULL;
        }

        NewAllocation->Heap = PerfectHeap;
        NewAllocation->Size = SizeInBytes;

        if ((Flags & HEAP_ZERO_MEMORY) != 0 && SizeInBytes > OldSize) {
            memset((PBYTE)(NewAllocation + 1) + OldSize,
                   0,
                   SizeInBytes - OldSize);
        }

        AcquireSRWLockExclusive(&PerfectHeap->Lock);

        NewAllocation->Next = PerfectHeap->Head;
        NewAllocation->Prev = NULL;
        if (PerfectHeap->Head) {
            PerfectHeap->Head->Prev = NewAllocation;
        }
        PerfectHeap->Head = NewAllocation;

        ReleaseSRWLockExclusive(&PerfectHeap->Lock);

        return (LPVOID)(NewAllocation + 1);
    }

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
    UNREFERENCED_PARAMETER(Flags);

    PPH_HEAP PerfectHeap;
    PPH_HEAP_ALLOCATION Allocation;

    if (!Mem) {
        return TRUE;
    }

    PerfectHeap = GetPerfectHashHeap(Heap);
    if (PerfectHeap) {
        Allocation = ((PPH_HEAP_ALLOCATION)Mem) - 1;

        AcquireSRWLockExclusive(&PerfectHeap->Lock);

        if (Allocation->Prev) {
            Allocation->Prev->Next = Allocation->Next;
        } else {
            PerfectHeap->Head = Allocation->Next;
        }
        if (Allocation->Next) {
            Allocation->Next->Prev = Allocation->Prev;
        }

        ReleaseSRWLockExclusive(&PerfectHeap->Lock);

        free(Allocation);
        return TRUE;
    }

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
    } else {
        PhTrackVirtualAlloc(addr, Size);
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
    size_t PageSize;
    size_t ProtectSize;
    uintptr_t Addr;
    uintptr_t Aligned;

    prot = 0;
    if (BooleanFlagOn(NewProtect, PAGE_NOACCESS)) {
        prot = PROT_NONE;
    } else {
        if (BooleanFlagOn(NewProtect, PAGE_READONLY) ||
            BooleanFlagOn(NewProtect, PAGE_READWRITE) ||
            BooleanFlagOn(NewProtect, PAGE_EXECUTE_READ) ||
            BooleanFlagOn(NewProtect, PAGE_EXECUTE_READWRITE) ||
            BooleanFlagOn(NewProtect, PAGE_EXECUTE_WRITECOPY)) {
            prot |= PROT_READ;
        }
        if (BooleanFlagOn(NewProtect, PAGE_READWRITE) ||
            BooleanFlagOn(NewProtect, PAGE_EXECUTE_READWRITE) ||
            BooleanFlagOn(NewProtect, PAGE_EXECUTE_WRITECOPY)) {
            prot |= PROT_WRITE;
        }
        if (BooleanFlagOn(NewProtect, PAGE_EXECUTE) ||
            BooleanFlagOn(NewProtect, PAGE_EXECUTE_READ) ||
            BooleanFlagOn(NewProtect, PAGE_EXECUTE_READWRITE) ||
            BooleanFlagOn(NewProtect, PAGE_EXECUTE_WRITECOPY)) {
            prot |= PROT_EXEC;
        }
        if (prot == 0) {
            prot = PROT_READ | PROT_WRITE;
        }
    }

    UNREFERENCED_PARAMETER(OldProtect);

    PageSize = (size_t)sysconf(_SC_PAGESIZE);
    Addr = (uintptr_t)Address;
    Aligned = Addr & ~(uintptr_t)(PageSize - 1);
    ProtectSize = Size + (Addr - Aligned);
    ProtectSize = (ProtectSize + PageSize - 1) & ~(PageSize - 1);

    Error = mprotect((void *)Aligned, ProtectSize, prot);
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
    size_t ActualSize;

    ASSERT(FreeType == MEM_RELEASE);

    ActualSize = Size;
    if (ActualSize == 0) {
        if (!PhUntrackVirtualAlloc(Address, &ActualSize)) {
            SetLastError(EINVAL);
            return FALSE;
        }
    } else {
        size_t TrackedSize;
        PhUntrackVirtualAlloc(Address, &TrackedSize);
    }

    if (ActualSize == 0) {
        SetLastError(EINVAL);
        return FALSE;
    }

    Error = munmap(Address, ActualSize);
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

    FREE_PTR(&String->Buffer);
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
    BYTE Index;
    BYTE Count;
    PTP_POOL Pool;
    PTPP_QUEUE Queue;

    Pool = (PTP_POOL)calloc(1, sizeof(*Pool));
    if (!Pool) {
        SetLastError(ENOMEM);
        goto Error;
    }

    Pool->WorkerWaitEvent = CreateEventW(NULL, FALSE, FALSE, NULL);

    Count = (BYTE)ARRAYSIZE(Pool->TaskQueue);
    for (Index = 0; Index < Count; Index++) {
        Queue = (PTPP_QUEUE)calloc(1, sizeof(TPP_QUEUE));
        if (!Queue) {
            SetLastError(ENOMEM);
            goto Error;
        }
        InitializeSRWLock(&Queue->Lock);
        InitializeListHead(&Queue->Queue);
        Pool->TaskQueue[Index] = Queue;
    }

    Pool->Refcount = 1;
    Pool->MaximumThreads = 500;
    Pool->MinimumThreads = 0;

    InitializeSRWLock(&Pool->Lock);
    InitializeSRWLock(&Pool->ShutdownLock);

    InitializeListHead(&Pool->PoolObjectList);
    InitializeListHead(&Pool->WorkerList);
    InitializeListHead(&Pool->PoolLinks);

    goto End;

Error:

    if (Pool) {
        CloseThreadpool(Pool);
        Pool = NULL;
    }

End:

    return Pool;
}

WINBASEAPI
VOID
WINAPI
SetThreadpoolThreadMaximum(
    _Inout_ PTP_POOL Pool,
    _In_ DWORD MaxThreads
    )
{
    AcquireSRWLockExclusive(&Pool->Lock);
    Pool->MaximumThreads = MaxThreads;
    ReleaseSRWLockExclusive(&Pool->Lock);
    return;
}

VOID
ThreadpoolWorkerThreadEntry2 (
    _In_ PVOID Context
    )
{
    ULONG WaitResult;
    PTP_POOL Pool;
    HANDLE Event;
    PTPP_WORKER Worker;
    PTP_CALLBACK_INSTANCE Instance;

    Worker = (PTPP_WORKER)Context;
    Instance = &Worker->CallbackInstance;

    Pool = Instance->Pool;
    Event = (HANDLE)Pool->WorkerWaitEvent;

    InterlockedIncrement(&Pool->ActiveWorkerCount);

    while (TRUE) {
        WaitResult = WaitForSingleObject(Event, INFINITE);
        if (WaitResult != WAIT_OBJECT_0) {
            SYS_ERROR(WaitForSingleObject);
            break;
        }

        if (Pool->PendingWorkCount == 0) {
            continue;
        }

        break;
    }

    InterlockedDecrement(&Pool->ActiveWorkerCount);
}

BOOL
CreateThreadpoolWorker(
    _Inout_ PTP_POOL Pool
    )
{
    INT Result;
    PTPP_WORKER Worker;
    PTP_CALLBACK_INSTANCE Instance;

    Worker = (PTPP_WORKER)calloc(1, sizeof(TPP_WORKER));
    if (!Worker) {
        return FALSE;
    }

    Instance = &Worker->CallbackInstance;
    Instance->Pool = Pool;

    Result = pthread_create(&Worker->ThreadId,
                            NULL,
                            ThreadpoolWorkerThreadEntry2,
                            Worker);

    if (Result != 0) {
        SetLastError(errno);
        SYS_ERROR(pthread_create);
        FREE_PTR(&Worker);
        return FALSE;
    }

    InsertTailList(&Pool->WorkerList, &Worker->ListEntry);

    Pool->NumberOfWorkers++;

    return TRUE;
}

BOOL
DestroyThreadpoolWorker(
    _Inout_ PTP_POOL Pool
    )
{
    return TRUE;
}

WINBASEAPI
BOOL
WINAPI
SetThreadpoolThreadMinimum(
    _Inout_ PTP_POOL Pool,
    _In_ DWORD MinThreads
    )
{
    DWORD Index;
    DWORD Count;
    AcquireSRWLockExclusive(&Pool->Lock);

    if (MinThreads > Pool->MaximumThreads) {
        ReleaseSRWLockExclusive(&Pool->Lock);
        return FALSE;
    }

    if (MinThreads > Pool->MinimumThreads) {
        Count = MinThreads - Pool->MinimumThreads;
        for (Index = 0; Index < Count; Index++) {
            if (!CreateThreadpoolWorker(Pool)) {
                ReleaseSRWLockExclusive(&Pool->Lock);
                return FALSE;
            }
        }
    } else if (MinThreads < Pool->MinimumThreads) {
        Count = Pool->MinimumThreads - MinThreads;
        for (Index = 0; Index < Count; Index++) {
            DestroyThreadpoolWorker(Pool);
        }
    }

    Pool->MinimumThreads = MinThreads;
    ReleaseSRWLockExclusive(&Pool->Lock);
    return TRUE;
}

WINBASEAPI
VOID
WINAPI
CloseThreadpool(
    _Inout_ PTP_POOL Pool
    )
{
    BYTE Index;
    BYTE Count;
    PTPP_QUEUE Queue;

    if (Pool->WorkerWaitEvent) {
        SetEvent(Pool->WorkerWaitEvent);
        CloseEvent(Pool->WorkerWaitEvent);
    }

    Count = (BYTE)ARRAYSIZE(Pool->TaskQueue);
    for (Index = 0; Index < Count; Index++) {
        FREE_PTR(&Pool->TaskQueue[Index]);
    }

    free(Pool);
}

WINBASEAPI
_Must_inspect_result_
PTP_CLEANUP_GROUP
WINAPI
CreateThreadpoolCleanupGroup(
    VOID
    )
{
    PTP_CLEANUP_GROUP Cleanup;

    Cleanup = (PTP_CLEANUP_GROUP)calloc(1, sizeof(*Cleanup));
    if (!Cleanup) {
        SetLastError(ENOMEM);
        goto Error;
    }

    Cleanup->Refcount = 1;
    InitializeSRWLock(&Cleanup->MemberLock);
    InitializeListHead(&Cleanup->MemberList);

    //
    // Init barrier here?
    //

    InitializeSRWLock(&Cleanup->CleanupLock);
    InitializeListHead(&Cleanup->CleanupList);

    goto End;

Error:

    if (Cleanup) {
        FREE_PTR(&Cleanup);
    }

End:

    return Cleanup;
}

WINBASEAPI
VOID
WINAPI
CloseThreadpoolCleanupGroupMembers(
    _Inout_ PTP_CLEANUP_GROUP CleanupGroup,
    _In_ BOOL CancelPendingCallbacks,
    _Inout_opt_ PVOID CleanupContext
    )
{
    //
    // for members in group:
    //      cleanup member
    //

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
    _In_ PTP_WORK_CALLBACK Callback,
    _Inout_opt_ PVOID Context,
    _In_opt_ PTP_CALLBACK_ENVIRON CallbackEnv
    )
{
    PTP_POOL Pool;
    PTP_WORK Work;
    PTP_CLEANUP_GROUP Group;
    PTPP_CLEANUP_GROUP_MEMBER Member;

    ASSERT(CallbackEnv != NULL);
    if (CallbackEnv == NULL) {
        SetLastError(EINVAL);
        return NULL;
    }
    ASSERT(CallbackEnv->Pool != NULL);
    if (CallbackEnv->Pool == NULL) {
        SetLastError(EINVAL);
        return NULL;
    }

    Work = (PTP_WORK)calloc(1, sizeof(*Work));
    if (!Work) {
        SetLastError(ENOMEM);
        goto Error;
    }

    Member = &Work->CleanupGroupMember;
    Group = CallbackEnv->CleanupGroup;

    if (Group == NULL) {
        Group = CreateThreadpoolCleanupGroup();
        if (Group == NULL) {
            SetLastError(ENOMEM);
            goto Error;
        }
        CallbackEnv->CleanupGroup = Group;
    }

    //
    // Add member to group list.
    //

    AcquireSRWLockExclusive(&Group->MemberLock);
    InsertTailList(&Group->MemberList, &Member->CleanupGroupMemberLinks);
    ReleaseSRWLockExclusive(&Group->MemberLock);

    Member->Context = Context;
    Member->WorkCallback = Callback;
    Pool = Member->Pool = CallbackEnv->Pool;

    Work->Callbacks.ExecuteCallback = (PVOID)Callback;

    AcquireSRWLockExclusive(&Pool->Lock);
    InsertTailList(&Pool->WorkerList, &Work->ListEntry);
    ReleaseSRWLockExclusive(&Pool->Lock);

    goto End;

Error:

    if (Work) {
        FREE_PTR(&Work);
    }

End:

    return Work;
}

VOID
ThreadpoolWorkerThreadEntry (
    _In_ PVOID Context
    )
{
    ULONG WaitResult;
    PTP_POOL Pool;
    HANDLE Event;
    PTPP_WORKER Worker;
    PTP_CALLBACK_INSTANCE Instance;

    Worker = (PTPP_WORKER)Context;
    Instance = &Worker->CallbackInstance;

    Pool = Instance->Pool;

    InterlockedIncrement(&Pool->ActiveWorkerCount);

    while (TRUE) {
        WaitResult = WaitForSingleObject(Event, INFINITE);
        if (WaitResult != WAIT_OBJECT_0) {
            SYS_ERROR(WaitForSingleObject);
            break;
        }

        if (Pool->PendingWorkCount == 0) {
            continue;
        }

        break;
    }

    InterlockedDecrement(&Pool->ActiveWorkerCount);
}


WINBASEAPI
VOID
WINAPI
SubmitThreadpoolWork(
    _Inout_ PTP_WORK Work
    )
{
    return;
}

#if 0
WINBASEAPI
VOID
WINAPI
SubmitThreadpoolWork(
    _Inout_ PTP_WORK Work
    )
{
    INT Result;
    PTP_POOL Pool;
    PTPP_WORKER Worker;
    PTP_CALLBACK_INSTANCE Instance;

    Worker = (PTPP_WORKER)calloc(1, sizeof(TPP_WORKER));
    if (!Worker) {
        return;
    }

    Instance = &Worker->CallbackInstance;
    Instance->CleanupGroupMember = &Work->CleanupGroupMember;
    Pool = Instance->Pool = Work->CleanupGroupMember.Pool;

    Result = pthread_create(&Worker->ThreadId,
                            NULL,
                            ThreadpoolWorkerThreadEntry,
                            Worker);

    if (Result != 0) {
        SetLastError(errno);
        SYS_ERROR(pthread_create);
        FREE_PTR(&Worker);
        return;
    }

    AcquireSRWLockExclusive(&Pool->Lock);
    InsertTailList(&Pool->WorkerList, &Worker->ListEntry);
    Pool->NumberOfWorkers++;
    ReleaseSRWLockExclusive(&Pool->Lock);

    return;
}

WINBASEAPI
VOID
WINAPI
SubmitThreadpoolWork2(
    _Inout_ PTP_WORK Work
    )
{
    PTP_POOL Pool;

    Pool = Work->CleanupGroupMember.Pool;

    InterlockedIncrement(&Pool->PendingWorkCount);
    SetEvent(Pool->WorkerWaitEvent);

    return;
}
#endif

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

//
// Error handling.
//

PERFECT_HASH_PRINT_ERROR PerfectHashPrintError;

_Use_decl_annotations_
HRESULT
PerfectHashPrintError(
    PCSZ FunctionName,
    PCSZ FileName,
    ULONG LineNumber,
    ULONG Error
    )
{
    PCSZ CodeString;
    HRESULT Result = S_OK;
    const STRING Prefix1 = RTL_CONSTANT_STRING(
        "%s: %u: %s failed with error: 0x%x\n"
    );
    const STRING Prefix2 = RTL_CONSTANT_STRING(
        "%s: %u: %s failed with error: 0x%x: %s\n"
    );

    if (Error == S_OK) {
        fprintf(stderr,
                Prefix1.Buffer,
                FileName,
                LineNumber,
                FunctionName,
                Error);
    } else {
        Result = PerfectHashGetErrorCodeString(NULL, Error, &CodeString);
        if (FAILED(Result)) {
            fprintf(stderr,
                    "PhtPrintError: PerfectHashGetErrorCodeString() "
                    "failed for error 0x%x", Error);
        } else {
            fprintf(stderr,
                    Prefix2.Buffer,
                    FileName,
                    LineNumber,
                    FunctionName,
                    Error,
                    CodeString);
        }
    }

    return Result;
}

#if defined(PH_LINUX) || defined(PH_MAC)

#define PH_MESSAGE_RESOURCE_UNICODE 0x0001

static
USHORT
ReadU16LE(
    _In_reads_(2) const UCHAR *Buffer
    )
{
    return (USHORT)(
        ((USHORT)Buffer[0]) |
        ((USHORT)Buffer[1] << 8)
    );
}

static
ULONG
ReadU32LE(
    _In_reads_(4) const UCHAR *Buffer
    )
{
    return (ULONG)(
        ((ULONG)Buffer[0]) |
        ((ULONG)Buffer[1] << 8) |
        ((ULONG)Buffer[2] << 16) |
        ((ULONG)Buffer[3] << 24)
    );
}

static
HRESULT
PerfectHashLookupMessageEntry(
    _In_ ULONG Code,
    _Outptr_result_maybenull_ const UCHAR **Text,
    _Out_ PUSHORT TextLength,
    _Out_ PUSHORT Flags
    )
{
    ULONG Index;
    ULONG LowId;
    ULONG HighId;
    ULONG Offset;
    ULONG NumberOfBlocks;
    ULONG Id;
    USHORT EntryLength;
    USHORT EntryFlags;
    size_t BaseOffset;
    size_t BlocksSize;
    size_t EntryOffset;
    size_t DataSize;
    const UCHAR *Data;

    if (!Text || !TextLength || !Flags) {
        return E_POINTER;
    }

    *Text = NULL;
    *TextLength = 0;
    *Flags = 0;

    Data = PerfectHashErrorsEnglishBin;
    DataSize = PerfectHashErrorsEnglishBinSize;
    if (DataSize < sizeof(ULONG)) {
        return E_FAIL;
    }

    NumberOfBlocks = ReadU32LE(Data);
    BlocksSize = (size_t)NumberOfBlocks * 12;
    if (DataSize < (sizeof(ULONG) + BlocksSize)) {
        return E_FAIL;
    }

    for (Index = 0; Index < NumberOfBlocks; Index++) {
        BaseOffset = sizeof(ULONG) + ((size_t)Index * 12);
        LowId = ReadU32LE(Data + BaseOffset);
        HighId = ReadU32LE(Data + BaseOffset + 4);
        Offset = ReadU32LE(Data + BaseOffset + 8);

        if (Code < LowId || Code > HighId) {
            continue;
        }

        if (Offset >= DataSize) {
            return E_FAIL;
        }

        EntryOffset = Offset;
        for (Id = LowId; ; Id++) {
            if ((EntryOffset + 4) > DataSize) {
                return E_FAIL;
            }

            EntryLength = ReadU16LE(Data + EntryOffset);
            EntryFlags = ReadU16LE(Data + EntryOffset + 2);

            if (EntryLength < 4) {
                return E_FAIL;
            }

            if ((EntryOffset + EntryLength) > DataSize) {
                return E_FAIL;
            }

            if (Id == Code) {
                *Text = Data + EntryOffset + 4;
                *TextLength = (USHORT)(EntryLength - 4);
                *Flags = EntryFlags;
                return S_OK;
            }

            EntryOffset += EntryLength;

            if (Id == HighId) {
                break;
            }
        }
    }

    return S_FALSE;
}

static
SIZE_T
PerfectHashCollapsePercentEscapes(
    _Inout_updates_bytes_(Length) PSTR Buffer,
    _In_ SIZE_T Length
    );

static
HRESULT
PerfectHashConvertMessageEntryToAnsi(
    _In_reads_(TextLength) const UCHAR *Text,
    _In_ USHORT TextLength,
    _In_ USHORT Flags,
    _Outptr_result_maybenull_ PSTR *MessageBuffer,
    _Out_ PSIZE_T MessageLength
    )
{
    SIZE_T Index;
    SIZE_T Count;
    SIZE_T Written;
    USHORT Wide;
    PSTR Buffer;

    if (!MessageBuffer || !MessageLength) {
        return E_POINTER;
    }

    *MessageBuffer = NULL;
    *MessageLength = 0;

    if (Flags & PH_MESSAGE_RESOURCE_UNICODE) {
        if ((TextLength % sizeof(USHORT)) != 0) {
            return E_FAIL;
        }

        Count = TextLength / sizeof(USHORT);
        Buffer = (PSTR)calloc(Count + 1, sizeof(CHAR));
        if (!Buffer) {
            return E_OUTOFMEMORY;
        }

        Written = 0;
        for (Index = 0; Index < Count; Index++) {
            Wide = ReadU16LE(Text + (Index * sizeof(USHORT)));
            if (Wide == 0) {
                break;
            }
            Buffer[Written++] = (Wide <= 0x7f) ? (CHAR)Wide : '?';
        }

        Written = PerfectHashCollapsePercentEscapes(Buffer, Written);
        *MessageBuffer = Buffer;
        *MessageLength = Written;
        return S_OK;
    }

    Buffer = (PSTR)calloc(TextLength + 1, sizeof(CHAR));
    if (!Buffer) {
        return E_OUTOFMEMORY;
    }

    Written = 0;
    for (Index = 0; Index < TextLength; Index++) {
        if (Text[Index] == '\0') {
            break;
        }
        Buffer[Written++] = (CHAR)Text[Index];
    }

    Written = PerfectHashCollapsePercentEscapes(Buffer, Written);
    *MessageBuffer = Buffer;
    *MessageLength = Written;
    return S_OK;
}

static
SIZE_T
PerfectHashCollapsePercentEscapes(
    _Inout_updates_bytes_(Length) PSTR Buffer,
    _In_ SIZE_T Length
    )
{
    SIZE_T ReadIndex;
    SIZE_T WriteIndex;

    ReadIndex = 0;
    WriteIndex = 0;

    while (ReadIndex < Length) {
        if (Buffer[ReadIndex] == '%' &&
            (ReadIndex + 1) < Length &&
            Buffer[ReadIndex + 1] == '%') {
            Buffer[WriteIndex++] = '%';
            ReadIndex += 2;
            continue;
        }

        Buffer[WriteIndex++] = Buffer[ReadIndex++];
    }

    Buffer[WriteIndex] = '\0';
    return WriteIndex;
}

static
HRESULT
PerfectHashGetMessageStringFromBin(
    _In_ ULONG Code,
    _Outptr_result_maybenull_ PSTR *MessageBuffer,
    _Out_ PSIZE_T MessageLength
    )
{
    USHORT Flags;
    USHORT TextLength;
    const UCHAR *Text;
    HRESULT Result;

    if (!MessageBuffer || !MessageLength) {
        return E_POINTER;
    }

    Result = PerfectHashLookupMessageEntry(Code, &Text, &TextLength, &Flags);
    if (Result != S_OK) {
        return Result;
    }

    return PerfectHashConvertMessageEntryToAnsi(Text,
                                                TextLength,
                                                Flags,
                                                MessageBuffer,
                                                MessageLength);
}

static
HRESULT
PerfectHashCreateStringFromUnicodeString(
    _In_opt_ PCUNICODE_STRING UnicodeString,
    _Outptr_result_maybenull_ PSTR *StringPointer,
    _Out_ PSIZE_T StringLength
    )
{
    SIZE_T Count;
    PSTR Buffer;
    PCHAR Dest;

    if (!StringPointer || !StringLength) {
        return E_POINTER;
    }

    *StringPointer = NULL;
    *StringLength = 0;

    if (!UnicodeString || !UnicodeString->Buffer || UnicodeString->Length == 0) {
        Buffer = (PSTR)calloc(1, sizeof(CHAR));
        if (!Buffer) {
            return E_OUTOFMEMORY;
        }
        *StringPointer = Buffer;
        *StringLength = 0;
        return S_OK;
    }

    Count = (SIZE_T)(UnicodeString->Length / sizeof(WCHAR));
    Buffer = (PSTR)calloc(Count + 1, sizeof(CHAR));
    if (!Buffer) {
        return E_OUTOFMEMORY;
    }

    Dest = Buffer;
    AppendUnicodeStringToCharBufferFast(&Dest, UnicodeString);

    *StringPointer = Buffer;
    *StringLength = (SIZE_T)(Dest - Buffer);
    return S_OK;
}

static
HRESULT
PerfectHashFormatMessageArgs(
    _In_ PCSZ Message,
    _In_ SIZE_T MessageLength,
    _Inout_ va_list *Args,
    _Outptr_result_maybenull_ PSTR *FormattedMessage,
    _Out_ PSIZE_T FormattedLength
    )
{
    PCSZ Cursor;
    PCSZ Match;
    PSTR ArgString;
    SIZE_T ArgLength;
    SIZE_T OutputLength;
    SIZE_T PrefixLength;
    SIZE_T SuffixLength;
    SIZE_T TotalLength;
    PSTR Output;
    const SIZE_T MarkerLength = sizeof("%1!wZ!") - 1;
    const PCSZ Marker = "%1!wZ!";
    HRESULT Result;

    if (!FormattedMessage || !FormattedLength) {
        return E_POINTER;
    }

    *FormattedMessage = NULL;
    *FormattedLength = 0;

    Cursor = Message;
    Match = strstr(Cursor, Marker);
    if (!Match) {
        return S_FALSE;
    }

    Result = PerfectHashCreateStringFromUnicodeString(
        va_arg(*Args, PCUNICODE_STRING),
        &ArgString,
        &ArgLength
    );
    if (FAILED(Result)) {
        return Result;
    }

    OutputLength = 0;
    Cursor = Message;
    while ((Match = strstr(Cursor, Marker)) != NULL) {
        OutputLength += (SIZE_T)(Match - Cursor);
        OutputLength += ArgLength;
        Cursor = Match + MarkerLength;
    }
    OutputLength += strlen(Cursor);

    Output = (PSTR)calloc(OutputLength + 1, sizeof(CHAR));
    if (!Output) {
        free(ArgString);
        return E_OUTOFMEMORY;
    }

    Cursor = Message;
    TotalLength = 0;
    while ((Match = strstr(Cursor, Marker)) != NULL) {
        PrefixLength = (SIZE_T)(Match - Cursor);
        CopyMemoryInline(Output + TotalLength, Cursor, PrefixLength);
        TotalLength += PrefixLength;

        CopyMemoryInline(Output + TotalLength, ArgString, ArgLength);
        TotalLength += ArgLength;

        Cursor = Match + MarkerLength;
    }

    SuffixLength = strlen(Cursor);
    CopyMemoryInline(Output + TotalLength, Cursor, SuffixLength);
    TotalLength += SuffixLength;

    Output[TotalLength] = '\0';
    *FormattedMessage = Output;
    *FormattedLength = TotalLength;

    free(ArgString);

    UNREFERENCED_PARAMETER(MessageLength);
    return S_OK;
}

#endif // PH_LINUX || PH_MAC

PERFECT_HASH_PRINT_MESSAGE PerfectHashPrintMessage;

_Use_decl_annotations_
HRESULT
PerfectHashPrintMessage(
    ULONG Code,
    ...
    )
{
    PCSZ CodeString;
    HRESULT Result = S_OK;
#if defined(PH_LINUX) || defined(PH_MAC)
    PSTR Message;
    PSTR FormattedMessage;
    SIZE_T MessageLength;
    SIZE_T FormattedLength;
    va_list Args;
#endif

#if defined(PH_LINUX) || defined(PH_MAC)
    Message = NULL;
    FormattedMessage = NULL;
    MessageLength = 0;
    FormattedLength = 0;

    Result = PerfectHashGetMessageStringFromBin(Code,
                                                &Message,
                                                &MessageLength);
    if (Result == S_OK) {
        va_start(Args, Code);
        Result = PerfectHashFormatMessageArgs(Message,
                                              MessageLength,
                                              &Args,
                                              &FormattedMessage,
                                              &FormattedLength);
        va_end(Args);

        if (Result == S_OK) {
            fwrite(FormattedMessage, 1, FormattedLength, stderr);
            free(FormattedMessage);
        } else {
            fwrite(Message, 1, MessageLength, stderr);
        }

        free(Message);
        Result = S_OK;

        if (DoesErrorCodeWantAlgoHashMaskTableAppended(Code)) {
            HRESULT NewResult;

            NewResult = PerfectHashPrintMessage(
                PH_MSG_PERFECT_HASH_ALGO_HASH_MASK_NAMES
            );
            if (FAILED(NewResult)) {
                Result = NewResult;
            }
        }

        return Result;
    }
#endif

    Result = PerfectHashGetErrorCodeString(NULL, Code, &CodeString);
    if (FAILED(Result)) {
        fprintf(stderr,
                "PhtPrintError: PerfectHashGetErrorCodeString() "
                "failed for error 0x%x", Code);
    } else {
        fprintf(stderr, "%s\n", CodeString);
    }

    return Result;
}

RTL_PRINT_SYS_ERROR RtlPrintSysError;

_Use_decl_annotations_
HRESULT
RtlPrintSysError(
    PRTL Rtl,
    PCSZ FunctionName,
    PCSZ FileName,
    ULONG LineNumber
    )
{
    return E_FAIL;
}

PCHAR
CommandLineArgvAToString(
    _In_ INT NumberOfArguments,
    _In_reads_(NumberOfArguments) PSTR *ArgvA
    )
{
    INT Index;
    PCHAR String;
    SIZE_T TotalSizeInBytes;

    TotalSizeInBytes = 0;

    for (Index = 0; Index < NumberOfArguments; Index++) {
        TotalSizeInBytes += strlen(ArgvA[Index]);
    }

    //
    // Account for space and trailing \0.
    //

    TotalSizeInBytes += NumberOfArguments + 1;

    String = (PCHAR)calloc(1, TotalSizeInBytes);
    if (!String) {
        return NULL;
    }

    for (Index = 0; Index < NumberOfArguments; Index++) {
        strcat(String, ArgvA[Index]);
        if (Index < (NumberOfArguments - 1)) {
            strcat(String, " ");
        }
    }

    return String;
}

PWSTR
CommandLineArgvAToStringW(
    _In_ INT NumberOfArguments,
    _In_reads_(NumberOfArguments) PSTR *ArgvA
    )
{
    INT Index;
    INT Inner;
    CHAR Char;
    PWCHAR Wide;
    PSTR Source;
    SIZE_T Length;
    PWCHAR String;
    SIZE_T TotalSizeInBytes;

    TotalSizeInBytes = 0;

    for (Index = 0; Index < NumberOfArguments; Index++) {
        TotalSizeInBytes += strlen(ArgvA[Index]) + 1;
    }

    //
    // Account for space and trailing \0.
    //

    TotalSizeInBytes += NumberOfArguments + 1;

    //
    // Account for char -> wchar_t.
    //

    TotalSizeInBytes *= sizeof(WCHAR);

    String = (PWCHAR)calloc(1, TotalSizeInBytes);
    if (!String) {
        return NULL;
    }

    Wide = (PWCHAR)String;

    for (Index = 0; Index < NumberOfArguments; Index++) {
        Source = ArgvA[Index];
        Length = strlen(Source);

        for (Inner = 0; Inner < Length; Inner++) {
            Char = Source[Inner];
            *Wide++ = (WCHAR)Char;
        }
    }

    return String;
}

PWSTR *
CommandLineArgvAToArgvW(
    _In_ INT NumberOfArguments,
    _In_reads_(NumberOfArguments) PSTR *ArgvA
    )
{
    INT Index;
    INT Inner;
    CHAR Char;
    PWSTR Wide;
    PSTR Source;
    PWSTR *ArgvW;
    SIZE_T Length;
    SIZE_T TotalSizeInBytes;
    SIZE_T ArraySizeInBytes;

    TotalSizeInBytes = 0;

    for (Index = 0; Index < NumberOfArguments; Index++) {
        TotalSizeInBytes += strlen(ArgvA[Index]);
    }

    //
    // Account for space and trailing \0.
    //

    TotalSizeInBytes += NumberOfArguments + 1;

    //
    // Account for char -> wchar_t.
    //

    TotalSizeInBytes *= sizeof(WCHAR);

    //
    // Account for the array of pointers, plus a trailing NULL pointer.
    //

    ArraySizeInBytes = (sizeof(Wide) * (NumberOfArguments + 1));
    TotalSizeInBytes += ArraySizeInBytes;

    ArgvW = (PWSTR *)calloc(1, TotalSizeInBytes);
    if (!ArgvW) {
        return NULL;
    }

    //
    // Wire up Wide to point to after the array.
    //

    Wide = (PWSTR)RtlOffsetToPointer(ArgvW, ArraySizeInBytes);

    for (Index = 0; Index < NumberOfArguments; Index++) {
        Source = ArgvA[Index];
        Length = strlen(Source);
        ArgvW[Index] = Wide;

        for (Inner = 0; Inner < Length; Inner++) {
            Char = Source[Inner];
            *Wide++ = (WCHAR)Char;
        }
        *Wide++ = L'\0';
    }

    return ArgvW;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
