/*++

Copyright (c) 2020 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashCuda.h

Abstract:

    nvcc-friendly include for the perfect hash library.

--*/

#pragma once

#ifndef PH_CUDA
#error PerfectHashCuda.h included by non-CUDA component.
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if 0
#include <no_sal2.h>
#define _Ret_reallocated_bytes_(Address, Size)
#define _Frees_ptr_opt_

//
// Define NT-style typedefs.
//

typedef char CHAR;
typedef short SHORT;
typedef int LONG;
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

#endif

//
// Vector types.
//

typedef int2 INT2;
typedef int4 INT4;
typedef int2 LONG2;
typedef int4 LONG4;
typedef uint2 UINT2;
typedef uint4 UINT4;
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


#if 0
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
#endif

#define AllOnesInline(Dest, Size) ZeroMemoryInline(Dest, Size, TRUE)

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

#define DEFINE_UNUSED_STATE(Name)                \
typedef union _##Name##_STATE {                  \
    struct {                                     \
        ULONG Unused:32;                         \
    };                                           \
    LONG AsLong;                                 \
    ULONG AsULong;                               \
} Name##_STATE;                                  \
C_ASSERT(sizeof(Name##_STATE) == sizeof(ULONG)); \
typedef Name##_STATE *P##Name##_STATE

#define DEFINE_UNUSED_FLAGS(Name)                \
typedef union _##Name##_FLAGS {                  \
    struct {                                     \
        ULONG Unused:32;                         \
    };                                           \
    LONG AsLong;                                 \
    ULONG AsULong;                               \
} Name##_FLAGS;                                  \
C_ASSERT(sizeof(Name##_FLAGS) == sizeof(ULONG)); \
typedef Name##_FLAGS *P##Name##_FLAGS

#if 0
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
#endif


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

#define GlobalThreadIndex() (BlockIndex.x * BlockDim.x + ThreadIndex.x)

#if 0
DEVICE
static inline
void
ClockBlock(
    int64_t ClockCount
    )
{
    int64_t Start = clock64();
    int64_t Offset = 0;
    while (Offset < ClockCount) {
        Offset = clock64() - Start;
    }
}
#endif

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

#define CU_SUCCEEDED(Result) (Result == 0)
#define CU_FAILED(Result) (Result != 0)
#endif

#ifdef __cplusplus
} // extern "C"
#endif

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab syntax=cuda                         :
