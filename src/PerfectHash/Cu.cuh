/*++

Copyright (c) 2017-2020 Trent Nelson <trent@trent.me>

Module Name:

    Cu.cuh

Abstract:

    This module is the main header file for the Cu component.

--*/

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <no_sal2.h>

//
// Define NT-style typedefs.
//

typedef char CHAR;
typedef short SHORT;
typedef long LONG;
typedef wchar_t WCHAR;    // wc,   16-bit UNICODE character

typedef WCHAR *PWCHAR, *LPWCH, *PWCH;

typedef CHAR *PCHAR, *LPCH, *PCH;

typedef float FLOAT;
typedef double DOUBLE;
typedef FLOAT *PFLOAT;
typedef DOUBLE *PDOUBLE;

typedef unsigned char BYTE;
typedef unsigned char UCHAR;
typedef unsigned short USHORT;
typedef unsigned long ULONG;

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

typedef int2 INT2;
typedef int4 INT4;

typedef INT2 *PINT2;
typedef INT4 *PINT4;

#define VOID void
typedef void *PVOID;

union _ULONG_BYTES {
    struct _Struct_size_bytes_(sizeof(ULONG)) {
        BYTE Byte1;
        BYTE Byte2;
        BYTE Byte3;
        BYTE Byte4;
    };

    struct _Struct_size_bytes_(sizeof(ULONG)) {
        CHAR Char1;
        CHAR Char2;
        CHAR Char3;
        CHAR Char4;
    };

    struct _Struct_size_bytes_(sizeof(ULONG)) {
        SHORT Word1;
        SHORT Word2;
    };

    struct _Struct_size_bytes_(sizeof(ULONG)) {
        USHORT UWord1;
        USHORT UWord2;
    };

    LONG AsLong;
    ULONG AsULong;
};
typedef union _ULONG_BYTES ULONG_BYTES;
typedef ULONG_BYTES *PULONG_BYTES;

union _LARGE_INTEGER {
    struct {
        ULONG LowPart;
        LONG HighPart;
    };
    struct {
        ULONG LowPart;
        LONG HighPart;
    };
    LONGLONG QuadPart;
};
typedef union _LARGE_INTEGER LARGE_INTEGER;

union _ULARGE_INTEGER {
    struct {
        ULONG LowPart;
        ULONG HighPart;
    };
    struct {
        ULONG LowPart;
        ULONG HighPart;
    };
    ULONGLONG QuadPart;
};
typedef union _ULARGE_INTEGER ULARGE_INTEGER;

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

typedef LONG CU_DEVICE;
typedef ULONG_PTR CU_DEVICE_POINTER;
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


#ifdef __cplusplus
}
#endif

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab syntax=cuda                         :
