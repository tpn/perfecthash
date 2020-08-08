/*++

Copyright (c) 2020 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashCuda.h

Abstract:

    nvcc-friendly include for the perfect hash library.

--*/

#pragma once

#ifndef __CUDA_ARCH__
#error PerfectHashCuda.h included by non-nvcc compiler.
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <no_sal2.h>
#define _Ret_reallocated_bytes_(Address, Size)
#define _Frees_ptr_opt_

//
// Define NT-style typedefs.
//

typedef char CHAR;
typedef short SHORT;
typedef long LONG;
typedef wchar_t WCHAR;    // wc,   16-bit UNICODE character

typedef WCHAR *PWCHAR, *LPWCH, *PWCH, *PWSTR, *LPWSTR;

typedef CHAR *PCHAR, *LPCH, *PCH, *PSTR;

typedef float FLOAT;
typedef double DOUBLE;
typedef FLOAT *PFLOAT;
typedef DOUBLE *PDOUBLE;

typedef unsigned char BYTE;
typedef unsigned char UCHAR;
typedef unsigned short USHORT;
typedef unsigned short WORD;
typedef unsigned long ULONG;
typedef unsigned long DWORD;

typedef int BOOL;
typedef BYTE BOOLEAN;
typedef BOOLEAN *PBOOLEAN;

typedef UCHAR *PUCHAR;
typedef USHORT *PUSHORT;
typedef ULONG *PULONG;

typedef BYTE *PBYTE;
typedef CHAR *PCHAR;
typedef SHORT *PSHORT;
typedef LONG *PLONG;

typedef long long LONGLONG;
typedef long long LONG64;
typedef long long LONG_PTR;
typedef unsigned long long ULONGLONG;
typedef unsigned long long ULONG64;
typedef unsigned long long ULONG_PTR;

typedef LONG64 *PLONG64;
typedef ULONG64 *PULONG64;

typedef long long *PLONGLONG;
typedef unsigned long long *PULONGLONG;

typedef __int64 LONG64, *PLONG64;
typedef unsigned __int64 ULONG64, *PULONG64;
typedef unsigned __int64 DWORD64, *PDWORD64;

//
// Vector types.
//

typedef int1 INT;
typedef uint1 UINT;
typedef int2 INT2;
typedef int4 INT4;
typedef uint2 UINT2;
typedef uint4 UINT4;
typedef int2 LONG2;
typedef int4 LONG4;
typedef uint2 ULONG2;
typedef uint4 ULONG4;
typedef longlong2 LONGLONG2;
typedef longlong4 LONGLONG4;
typedef ulonglong2 ULONGLONG2;
typedef ulonglong4 ULONGLONG4;

typedef INT2 *PINT2;
typedef INT4 *PINT4;
typedef UINT2 *PUINT2;
typedef UINT4 *PUINT4;
typedef ULONG2 *PULONG2;
typedef ULONG4 *PULONG4;
typedef LONGLONG2 *PLONGLONG2;
typedef LONGLONG4 *PLONGLONG4;
typedef ULONGLONG2 *PULONGLONG2;
typedef ULONGLONG4 *PULONGLONG4;

#define CONST const
#define VOID void
typedef void *PVOID;
typedef void *LPVOID;

typedef size_t SIZE_T;

typedef PVOID HANDLE;
typedef HANDLE *PHANDLE;
typedef HANDLE HMODULE;
typedef HANDLE HINSTANCE;

typedef struct _GUID {
    unsigned long  Data1;
    unsigned short Data2;
    unsigned short Data3;
    unsigned char  Data4[8];
} GUID;

typedef GUID *LPGUID;
typedef const GUID *LPCGUID;
typedef GUID IID;

#define REFGUID const GUID *
#define REFIID const IID *
#define REFCLSID const IID *

#define RTL_NUMBER_OF_V1(A) (sizeof(A)/sizeof((A)[0]))
#define ARRAYSIZE(A) RTL_NUMBER_OF_V1(A)

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

typedef struct _RTL_BITMAP {

    //
    // Number of bits in the bitmap.
    //

    ULONG SizeOfBitMap;

    //
    // Pad out to an 8-byte boundary.
    //

    ULONG Padding;

    //
    // Pointer to bitmap buffer.
    //

    PULONG Buffer;

} RTL_BITMAP;
typedef RTL_BITMAP *PRTL_BITMAP;

struct _LIST_ENTRY {
   struct _LIST_ENTRY *Flink;
   struct _LIST_ENTRY *Blink;
};
typedef struct _LIST_ENTRY LIST_ENTRY;
typedef LIST_ENTRY *PLIST_ENTRY;

typedef enum _TP_CALLBACK_PRIORITY {
    TP_CALLBACK_PRIORITY_HIGH,
    TP_CALLBACK_PRIORITY_NORMAL,
    TP_CALLBACK_PRIORITY_LOW,
    TP_CALLBACK_PRIORITY_INVALID,
    TP_CALLBACK_PRIORITY_COUNT = TP_CALLBACK_PRIORITY_INVALID
} TP_CALLBACK_PRIORITY;

typedef struct _RTL_CRITICAL_SECTION_DEBUG {
    WORD   Type;
    WORD   CreatorBackTraceIndex;
    struct _RTL_CRITICAL_SECTION *CriticalSection;
    LIST_ENTRY ProcessLocksList;
    DWORD EntryCount;
    DWORD ContentionCount;
    DWORD Flags;
    WORD   CreatorBackTraceIndexHigh;
    WORD   SpareWORD  ;
} RTL_CRITICAL_SECTION_DEBUG, *PRTL_CRITICAL_SECTION_DEBUG, RTL_RESOURCE_DEBUG, *PRTL_RESOURCE_DEBUG;

//
// These flags define the upper byte of the critical section SpinCount field
//
#define RTL_CRITICAL_SECTION_FLAG_NO_DEBUG_INFO         0x01000000
#define RTL_CRITICAL_SECTION_FLAG_DYNAMIC_SPIN          0x02000000
#define RTL_CRITICAL_SECTION_FLAG_STATIC_INIT           0x04000000
#define RTL_CRITICAL_SECTION_FLAG_RESOURCE_TYPE         0x08000000
#define RTL_CRITICAL_SECTION_FLAG_FORCE_DEBUG_INFO      0x10000000
#define RTL_CRITICAL_SECTION_ALL_FLAG_BITS              0xFF000000
#define RTL_CRITICAL_SECTION_FLAG_RESERVED              (RTL_CRITICAL_SECTION_ALL_FLAG_BITS & (~(RTL_CRITICAL_SECTION_FLAG_NO_DEBUG_INFO | RTL_CRITICAL_SECTION_FLAG_DYNAMIC_SPIN | RTL_CRITICAL_SECTION_FLAG_STATIC_INIT | RTL_CRITICAL_SECTION_FLAG_RESOURCE_TYPE | RTL_CRITICAL_SECTION_FLAG_FORCE_DEBUG_INFO)))

//
// These flags define possible values stored in the Flags field of a critsec debuginfo.
//
#define RTL_CRITICAL_SECTION_DEBUG_FLAG_STATIC_INIT     0x00000001


#pragma pack(push, 8)

typedef struct _RTL_CRITICAL_SECTION {
    PRTL_CRITICAL_SECTION_DEBUG DebugInfo;

    //
    //  The following three fields control entering and exiting the critical
    //  section for the resource
    //

    LONG LockCount;
    LONG RecursionCount;
    HANDLE OwningThread;        // from the thread's ClientId->UniqueThread
    HANDLE LockSemaphore;
    ULONG_PTR SpinCount;        // force size on 64-bit systems when packed
} RTL_CRITICAL_SECTION, *PRTL_CRITICAL_SECTION;

#pragma pack(pop)

typedef struct _RTL_SRWLOCK {
        PVOID Ptr;
} RTL_SRWLOCK, *PRTL_SRWLOCK;
#define RTL_SRWLOCK_INIT {0}
typedef struct _RTL_CONDITION_VARIABLE {
        PVOID Ptr;
} RTL_CONDITION_VARIABLE, *PRTL_CONDITION_VARIABLE;

typedef RTL_CRITICAL_SECTION CRITICAL_SECTION;
typedef RTL_SRWLOCK SRWLOCK;

#define STDAPICALLTYPE
#define NTAPI
#define FORCEINLINE static inline
#define C_ASSERT(e) typedef char __C_ASSERT__[(e)?1:-1]

typedef _Return_type_success_(return >= 0) long HRESULT;
typedef HRESULT *PHRESULT;
#define _HRESULT_TYPEDEF_(_sc) ((HRESULT)_sc)

#define S_OK            ((HRESULT)0L)
#define S_FALSE         ((HRESULT)1L)
#define E_POINTER       _HRESULT_TYPEDEF_(0x80004003L)
#define E_FAIL          _HRESULT_TYPEDEF_(0x80004005L)
#define E_UNEXPECTED    _HRESULT_TYPEDEF_(0x8000FFFFL)
#define E_OUTOFMEMORY   _HRESULT_TYPEDEF_(0x8007000EL)
#define E_INVALIDARG    _HRESULT_TYPEDEF_(0x80070057L)
#define E_NOTIMPL       _HRESULT_TYPEDEF_(0x80000001L)

#ifndef RtlOffsetToPointer
#define RtlOffsetToPointer(B,O)    ((PCHAR)(((PCHAR)(B)) + ((ULONG_PTR)(O))))
#endif

#ifndef RtlOffsetFromPointer
#define RtlOffsetFromPointer(B,O)  ((PCHAR)(((PCHAR)(B)) - ((ULONG_PTR)(O))))
#endif

#ifndef RtlPointerToOffset
#define RtlPointerToOffset(B,P)    ((ULONG_PTR)(((PCHAR)(P)) - ((PCHAR)(B))))
#endif

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#define SUCCEEDED(hr) (((HRESULT)(hr)) >= 0)
#define FAILED(hr) (((HRESULT)(hr)) < 0)

FORCEINLINE
VOID
CopyMemoryInline(
    _Out_writes_bytes_all_(SizeInBytes) PVOID Dst,
    _In_ const VOID *Src,
    _In_ SIZE_T SizeInBytes
    )
{
    PDWORD64 Dest = (PDWORD64)Dst;
    PDWORD64 Source = (PDWORD64)Src;
    PCHAR TrailingDest;
    PCHAR TrailingSource;
    SIZE_T TrailingBytes;
    SIZE_T NumberOfQuadwords;

    NumberOfQuadwords = SizeInBytes >> 3;
    TrailingBytes = SizeInBytes - (NumberOfQuadwords << 3);

    while (NumberOfQuadwords) {

        //
        // N.B. If you hit an exception on this next line, and the call stack
        //      contains PrepareBulkCreateCsvFile(), you probably need to adjust
        //      the number of pages used for the temporary row buffer in either
        //      the BulkCreateBestCsv.h or BulkCreateCsv.h header (e.g. bump
        //      BULK_CREATE_BEST_CSV_ROW_BUFFER_NUMBER_OF_PAGES by one).
        //

        *Dest++ = *Source++;
        NumberOfQuadwords--;
    }

    TrailingDest = (PCHAR)Dest;
    TrailingSource = (PCHAR)Source;

    while (TrailingBytes) {
        *TrailingDest++ = *TrailingSource++;
        TrailingBytes--;
    }
}

#define CopyInline CopyMemoryInline

//
// Ditto for ZeroMemory.
//

FORCEINLINE
VOID
ZeroMemoryInline(
    _Out_writes_bytes_all_(SizeInBytes) PVOID Dst,
    _In_ SIZE_T SizeInBytes,
    _In_ BOOLEAN AllOnes
    )
{
    PDWORD64 Dest = (PDWORD64)Dst;
    DWORD64 FillQuad;
    BYTE Fill;
    PCHAR TrailingDest;
    SIZE_T TrailingBytes;
    SIZE_T NumberOfQuadwords;

    NumberOfQuadwords = SizeInBytes >> 3;
    TrailingBytes = SizeInBytes - (NumberOfQuadwords << 3);

    if (AllOnes) {
        FillQuad = ~0ULL;
        Fill = (BYTE)~0;
    } else {
        FillQuad = 0;
        Fill = 0;
    }

    while (NumberOfQuadwords) {
        *Dest++ = (DWORD64)FillQuad;
        NumberOfQuadwords--;
    }

    TrailingDest = (PCHAR)Dest;

    while (TrailingBytes) {
        *TrailingDest++ = Fill;
        TrailingBytes--;
    }
}

#define ZeroInline(Dest, Size) ZeroMemoryInline(Dest, Size, FALSE)
#define ZeroArrayInline(Name) ZeroInline(Name, sizeof(Name))

#define AllOnesInline(Dest, Size) ZeroMemoryInline(Dest, Size, TRUE)

#include <PerfectHash.h>

//
// Define helper macros for component definition.
//

#define COMMON_COMPONENT_HEADER(Name) \
    P##Name##_VTBL Vtbl;              \
    SRWLOCK Lock;                     \
    LIST_ENTRY ListEntry;             \
    struct _RTL *Rtl;                 \
    struct _ALLOCATOR *Allocator;     \
    PIUNKNOWN OuterUnknown;           \
    volatile LONG ReferenceCount;     \
    PERFECT_HASH_INTERFACE_ID Id;     \
    ULONG SizeOfStruct;               \
    Name##_STATE State;               \
    Name##_FLAGS Flags;               \
    ULONG Reserved

#define DEFINE_UNUSED_STATE(Name)                  \
typedef union _##Name##_STATE {                    \
    struct {                                       \
        ULONG Unused:32;                           \
    };                                             \
    LONG AsLong;                                   \
    ULONG AsULong;                                 \
} Name##_STATE;                                    \
C_ASSERT(sizeof(Name##_STATE) == sizeof(ULONG));   \
typedef Name##_STATE *P##Name##_STATE

#define DEFINE_UNUSED_FLAGS(Name)                  \
typedef union _##Name##_FLAGS {                    \
    struct {                                       \
        ULONG Unused:32;                           \
    };                                             \
    LONG AsLong;                                   \
    ULONG AsULong;                                 \
} Name##_FLAGS;                                    \
C_ASSERT(sizeof(Name##_FLAGS) == sizeof(ULONG));   \
typedef Name##_FLAGS *P##Name##_FLAGS

typedef enum _TYPE {
    ByteType = 0,
    ShortType = 1,
    LongType = 2,
    LongLongType = 3,
    XmmType = 4,
    YmmType = 5,
    ZmmType = 6,
} TYPE;
typedef TYPE *PTYPE;

#ifdef __cplusplus
#ifndef EXTERN_C
#define EXTERN_C extern "C"
#endif
#ifndef EXTERN_C_BEGIN
#define EXTERN_C_BEGIN EXTERN_C {
#endif
#ifndef EXTERN_C_END
#define EXTERN_C_END }
#endif
#else
#ifndef EXTERN_C
#define EXTERN_C
#endif
#ifndef EXTERN_C_BEGIN
#define EXTERN_C_BEGIN
#endif
#ifndef EXTERN_C_END
#define EXTERN_C_END
#endif
#endif

//
// Define CUDA macros and typedefs in NT style.
//

#define HOST __host__
#define GLOBAL __global__
#define DEVICE __device__
#define SHARED __shared__
#define CONSTANT __constant__
#define GridDim gridDim
#define BlockDim blockDim
#define BlockIndex blockIdx
#define ThreadIndex threadIdx

#define KERNEL EXTERN_C GLOBAL

#define FOR_EACH_1D(Index, Total)                           \
    for (Index = BlockIndex.x * BlockDim.x + ThreadIndex.x; \
         Index < Total;                                     \
         Index += BlockDim.x * GridDim.x)

DEVICE
static inline
void
ClockBlock(
    _In_ LONGLONG ClockCount
    )
{
    LONGLONG Start = clock64();
    LONGLONG Offset = 0;
    while (Offset < ClockCount) {
        Offset = clock64() - Start;
    }
}

//
// Define CUDA Device API Typedefs.
//

typedef ULONG_PTR CU_DEVICE_POINTER;

#if 0
typedef LONG CU_DEVICE;
typedef CU_DEVICE *PCU_DEVICE;
typedef CU_DEVICE **PPCU_DEVICE;
typedef CU_DEVICE_POINTER *PCU_DEVICE_POINTER;
typedef CU_DEVICE_POINTER **PPCU_DEVICE_POINTER;

struct CU_CONTEXT;
typedef struct CU_CONTEXT *PCU_CONTEXT;
typedef struct CU_CONTEXT **PPCU_CONTEXT;

struct CU_MODULE;
typedef struct CU_MODULE *PCU_MODULE;
typedef struct CU_MODULE **PPCU_MODULE;

struct CU_EVENT;
typedef struct CU_EVENT *PCU_EVENT;
typedef struct CU_EVENT **PPCU_EVENT;

struct CU_STREAM;
typedef struct CU_STREAM *PCU_STREAM;
typedef struct CU_STREAM **PPCU_STREAM;

struct CU_FUNCTION;
typedef struct CU_FUNCTION *PCU_FUNCTION;
typedef struct CU_FUNCTION **PPCU_FUNCTION;
#endif

#define CU_SUCCEEDED(Result) (Result == 0)
#define CU_FAILED(Result) (Result != 0)


#ifdef __cplusplus
} // extern "C"
#endif

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab syntax=cuda                         :
