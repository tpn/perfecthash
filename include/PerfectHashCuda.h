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


//
// Define CUDA Device API Typedefs.
//

typedef ULONG_PTR CU_DEVICE_POINTER;

#ifdef __cplusplus
} // extern "C"
#endif

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab syntax=cuda                         :
