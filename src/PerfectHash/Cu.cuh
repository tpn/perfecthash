/*++

Copyright (c) 2017 Trent Nelson <trent@trent.me>

Module Name:

    Cu.cuh

Abstract:

    This module is the main header file for the Cu component.

--*/

#ifdef __cplusplus
extern "C" {
#endif

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

typedef unsigned char UCHAR;
typedef unsigned short USHORT;
typedef unsigned long ULONG;

typedef UCHAR *PUCHAR;
typedef USHORT *PUSHORT;
typedef ULONG *PULONG;

typedef CHAR *PCHAR;
typedef SHORT *PSHORT;
typedef LONG *PLONG;

typedef long long LONGLONG;
typedef long long LONG64;
typedef unsigned long long ULONGLONG;
typedef unsigned long long ULONG64;

typedef LONG64 *PLONG64;
typedef ULONG64 *PULONG64;

#define VOID    void

//
// Define CUDA macros and typedefs in NT style.
//

#define HOST __host__
#define GLOBAL __global__
#define DEVICE __device__
#define GridDim gridDim
#define BlockDim blockDim
#define BlockIndex blockIdx
#define ThreadIndex threadIdx


#define FOR_EACH_1D(Index, Total)                           \
    for (Index = BlockIndex.x * BlockDim.x + ThreadIndex.x; \
         Index < Total;                                     \
         Index += BlockDim.x * GridDim.x)

#ifdef __cplusplus
}
#endif

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab syntax=cuda                         :
