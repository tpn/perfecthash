/*++

Copyright (c) 2018-2023 Trent Nelson <trent@trent.me>

Module Name:

    Rtl.h

Abstract:

    This is the main header file for the Rtl (Run-time Library) component.
    It is named after the NT kernel's Rtl component.  It has two main roles:
    provide an interface to useful NT kernel primitives (such as bitmaps,
    prefix trees, hash tables, splay trees, AVL trees, etc) without the need
    to include ntddk.h or link to ntdll.lib ntoskrnl.lib.  (Ntddk.h can't be
    included if Windows.h is also included, and ntdll.lib and ntoskrnl.lib
    require the DDK to be installed.)

    Type definitions for the data structures (e.g. RTL_BITMAP) are mirrored,
    and function type definitions and pointer types are provided for the NT
    functions we use.  They are made accessible as function pointers through a
    structure named RTL.

    In addition to NT functionality, this module also defines structures and
    functions for additional functionality we implement, such as convenience
    functions for string and path handling, memory management, etc.  In most
    cases, our helper routines are exposed through the Rtl->Vtbl interface and
    thus conform to COM semantics (i.e. they return a HRESULT and take PRTL
    as their first parameter).

--*/

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#if PH_WINDOWS
#include <minwindef.h>
#endif

#ifdef PH_WINDOWS
#define PATHSEP L'\\'
#else
#define PATHSEP   L'/'
#define PATHSEP_W L'/'
#define PATHSEP_A  '/'
#endif

//
// NT typedefs.
//

#define RTL_CONSTANT_STRING(s) { \
    sizeof(s) - sizeof((s)[0]),  \
    sizeof(s),                   \
    0,                           \
    s                            \
}

#define RTL_LAST_CHAR(s) (                              \
    (s)->Buffer[(s)->Length / (sizeof((s)->Buffer[0]))] \
)

#define RTL_SECOND_LAST_CHAR(s) (                         \
    (s)->Buffer[(s)->Length / (sizeof((s)->Buffer[0]))-1] \
)

typedef union _WIDE_CHARACTER {
    struct {
        CHAR  LowPart;
        CHAR  HighPart;
    };
    WCHAR WidePart;
} WIDE_CHARACTER, *PWIDE_CHARACTER;

typedef _Null_terminated_ const CHAR *PCSZ;
typedef const CHAR *PCCHAR;

typedef _Null_terminated_ const WCHAR *PCWSZ;
typedef const WCHAR *PCWCHAR;

//
// Calls to VirtualFree() need to have their size parameter wrapped by
// VFS() to ensure it works on Windows and non-Windows platforms.  (Windows
// enforces size == 0 when MEM_RELEASE, but we need the size on *nix when we
// unmap the backing address.)
//

#ifdef PH_WINDOWS
#define VFS(Size) 0
#else
#define VFS(Size) Size
#endif


#ifndef PAGE_SHIFT
#define PAGE_SHIFT 12
#endif

#ifndef PAGE_SIZE
#define PAGE_SIZE (1 << PAGE_SHIFT) // 4096
#endif

#ifndef LARGE_PAGE_SHIFT
#define LARGE_PAGE_SHIFT 21
#endif

#ifndef LARGE_PAGE_SIZE
#define LARGE_PAGE_SIZE (1 << LARGE_PAGE_SHIFT) // 2097152, or 2MB.
#endif

#ifndef CACHE_LINE_SHIFT
#define CACHE_LINE_SHIFT 6
#endif

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE (1 << CACHE_LINE_SHIFT) // 64
#endif

#ifndef PAGE_ALIGN
#define PAGE_ALIGN(A) ((PVOID)((ULONG_PTR)(A) & ~(PAGE_SIZE - 1)))
#endif

#ifndef LARGE_PAGE_ALIGN
#define LARGE_PAGE_ALIGN(A) ((PVOID)((ULONG_PTR)(A) & ~(LARGE_PAGE_SIZE - 1)))
#endif

#ifndef CACHE_LINE_ALIGN
#define CACHE_LINE_ALIGN(A) ((PVOID)((ULONG_PTR)(A) & ~(CACHE_LINE_SIZE - 1)))
#endif

#ifndef ROUND_TO_PAGES
#define ROUND_TO_PAGES(Size) (                             \
    ((ULONG_PTR)(Size) + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1) \
)
#endif

#ifndef ROUND_TO_LARGE_PAGES
#define ROUND_TO_LARGE_PAGES(Size) (                                   \
    ((ULONG_PTR)(Size) + LARGE_PAGE_SIZE - 1) & ~(LARGE_PAGE_SIZE - 1) \
)
#endif

#ifndef ROUND_TO_CACHE_LINES
#define ROUND_TO_CACHE_LINES(Size) (                                   \
    ((ULONG_PTR)(Size) + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1) \
)
#endif

#ifndef BYTES_TO_PAGES
#define BYTES_TO_PAGES(Size) (                                   \
    (((Size) >> PAGE_SHIFT) + (((Size) & (PAGE_SIZE - 1)) != 0)) \
)
#endif

#ifndef BYTES_TO_LARGE_PAGES
#define BYTES_TO_LARGE_PAGES(Size) (                                         \
    (((Size) >> LARGE_PAGE_SHIFT) + (((Size) & (LARGE_PAGE_SIZE - 1)) != 0)) \
)
#endif

#ifndef BYTES_TO_CACHE_LINES
#define BYTES_TO_CACHE_LINES(Size) (                                         \
    (((Size) >> CACHE_LINE_SHIFT) + (((Size) & (CACHE_LINE_SIZE - 1)) != 0)) \
)
#endif

#define BYTES_TO_QUADWORDS(Bytes) ((Bytes) >> 3)

#define QUADWORD_SIZEOF(Type) (BYTES_TO_QUADWORDS(sizeof(Type)))

#ifndef ADDRESS_AND_SIZE_TO_SPAN_PAGES
#define ADDRESS_AND_SIZE_TO_SPAN_PAGES(Va,Size)                             \
    ((BYTE_OFFSET (Va) + ((SIZE_T) (Size)) + (PAGE_SIZE - 1)) >> PAGE_SHIFT
#endif

#ifndef ALIGN_DOWN
#define ALIGN_DOWN(Address, Alignment)                     \
    ((ULONG_PTR)(Address) & (~((ULONG_PTR)(Alignment)-1)))
#endif

#ifndef ALIGN_UP
#define ALIGN_UP(Address, Alignment) (                        \
    (((ULONG_PTR)(Address)) + (((ULONG_PTR)(Alignment))-1)) & \
    ~(((ULONG_PTR)(Alignment))-1)                             \
)
#endif

#ifndef ALIGN_UP_POINTER
#define ALIGN_UP_POINTER(Address) (ALIGN_UP(Address, sizeof(ULONG_PTR)))
#endif

#ifndef ALIGN_UP_SYSTEM
#define ALIGN_UP_SYSTEM(Address) (                 \
    ALIGN_UP(Address, MEMORY_ALLOCATION_ALIGNMENT) \
)
#endif

#ifndef ALIGN_UP_LARGE_PAGE
#define ALIGN_UP_LARGE_PAGE(Address) ( \
    ALIGN_UP(Address, LARGE_PAGE_SIZE) \
)
#endif

#ifndef ALIGN_DOWN_POINTER
#define ALIGN_DOWN_POINTER(Address) (ALIGN_DOWN(Address, sizeof(ULONG_PTR)))
#endif

#ifndef ALIGN_DOWN_PAGE
#define ALIGN_DOWN_PAGE(Address) (ALIGN_DOWN(Address, PAGE_SIZE))
#endif

#ifndef ALIGN_DOWN_USHORT_TO_POINTER_SIZE
#define ALIGN_DOWN_USHORT_TO_POINTER_SIZE(Value)                   \
    (USHORT)(ALIGN_DOWN((USHORT)Value, (USHORT)sizeof(ULONG_PTR)))
#endif

#ifndef ALIGN_UP_USHORT_TO_POINTER_SIZE
#define ALIGN_UP_USHORT_TO_POINTER_SIZE(Value)                   \
    (USHORT)(ALIGN_UP((USHORT)Value, (USHORT)sizeof(ULONG_PTR)))
#endif

#define XMMWORD_ALIGNMENT 16
#define YMMWORD_ALIGNMENT 32
#define ZMMWORD_ALIGNMENT 64

#ifdef _M_X64

//
// XMM, YMM, and ZMM registers.
//

#ifdef PH_WINDOWS
typedef __m128i DECLSPEC_ALIGN(XMMWORD_ALIGNMENT) XMMWORD, *PXMMWORD;
typedef __m256i DECLSPEC_ALIGN(YMMWORD_ALIGNMENT) YMMWORD, *PYMMWORD;
typedef __m512i DECLSPEC_ALIGN(ZMMWORD_ALIGNMENT) ZMMWORD, *PZMMWORD;
C_ASSERT(sizeof(XMMWORD) == XMMWORD_ALIGNMENT);
C_ASSERT(sizeof(YMMWORD) == YMMWORD_ALIGNMENT);
C_ASSERT(sizeof(ZMMWORD) == ZMMWORD_ALIGNMENT);
#else
typedef __m128i XMMWORD, *PXMMWORD;
typedef __m256i YMMWORD, *PYMMWORD;
typedef __m512i ZMMWORD, *PZMMWORD;
#endif

//
// AVX-512 masks.
//

typedef __mmask16 ZMASK8, *PZMASK8;
typedef __mmask16 ZMASK16, *PZMASK16;
typedef __mmask32 ZMASK32, *PZMASK32;
typedef __mmask64 ZMASK64, *PZMASK64;

//
// Helper structures for index tables fed to AVX512 permute routines such as
// _mm512_permutex2var_epi32().
//

typedef union _ZMM_PERMUTEXVAR_INDEX16 {
    struct _Struct_size_bytes_(sizeof(ULONG)) {
        ULONG Index:5;
        ULONG Unused:27;
    };

    ULONG AsULong;
} ZMM_PERMUTEXVAR_INDEX16;
C_ASSERT(sizeof(ZMM_PERMUTEXVAR_INDEX16) == sizeof(ULONG));

typedef union _ZMM_PERMUTEX2VAR_INDEX_BYTE {
    struct _Struct_size_bytes_(sizeof(BYTE)) {
        BYTE Index:3;
        BYTE Selector:1;
        BYTE Unused:4;
    };

    BYTE AsByte;
} ZMM_PERMUTEX2VAR_INDEX_BYTE;
C_ASSERT(sizeof(ZMM_PERMUTEX2VAR_INDEX_BYTE) == sizeof(BYTE));

typedef union _ZMM_PERMUTEX2VAR_INDEX16 {
    struct _Struct_size_bytes_(sizeof(ULONG)) {
        ULONG Index:5;
        ULONG Selector:1;
        ULONG Unused:26;
    };

    LONG AsLong;
    ULONG AsULong;
} ZMM_PERMUTEX2VAR_INDEX16;
C_ASSERT(sizeof(ZMM_PERMUTEX2VAR_INDEX16) == sizeof(ULONG));

typedef union _ZMM_PERMUTEX2VAR_INDEX32 {
    struct _Struct_size_bytes_(sizeof(ULONG)) {
        ULONG Index:4;
        ULONG Selector:1;
        ULONG Unused:27;
    };

    LONG AsLong;
    ULONG AsULong;
} ZMM_PERMUTEX2VAR_INDEX32;
C_ASSERT(sizeof(ZMM_PERMUTEX2VAR_INDEX32) == sizeof(ULONG));
#endif

#ifndef ALIGN_UP_XMMWORD
#define ALIGN_UP_XMMWORD(Address) (ALIGN_UP(Address, XMMWORD_ALIGNMENT))
#endif

#ifndef ALIGN_UP_YMMWORD
#define ALIGN_UP_YMMWORD(Address) (ALIGN_UP(Address, YMMWORD_ALIGNMENT))
#endif

#ifndef ALIGN_UP_ZMMWORD
#define ALIGN_UP_ZMMWORD(Address) (ALIGN_UP(Address, ZMMWORD_ALIGNMENT))
#endif

//
// The performance impact of keeping ASSERT() in non-debug builds is nil,
// so keep it.
//

#if 1
#define ASSERT(Condition) \
    if (!(Condition)) {   \
        __debugbreak();   \
    }
#else
#ifdef ASSERT
#undef ASSERT
#endif

#ifdef _DEBUG
#define ASSERT(Condition) \
    if (!(Condition)) {   \
        __debugbreak();   \
    }
#else
#define ASSERT(Condition)
#endif
#endif

#ifndef ARGUMENT_PRESENT
#define ARGUMENT_PRESENT(ArgumentPointer) (                  \
    (CHAR *)((ULONG_PTR)(ArgumentPointer)) != (CHAR *)(NULL) \
)
#endif

#ifndef RTL_FIELD_SIZE
#define RTL_FIELD_SIZE(Type, Field) (sizeof(((Type *)0)->Field))
#endif

//
// Similar to RTL_FIELD_SIZE, but captures the size of the pointed-to
// Field in type Type.
//

#define RTL_ELEMENT_SIZE(Type, Field) (sizeof(*((Type *)0)->Field))

#ifndef LARGE_INTEGER_TO_SIZE_T
#ifdef _WIN32
#define LARGE_INTEGER_TO_SIZE_T(Name) ((SIZE_T)Name.LowPart)
#else
#define LARGE_INTEGER_TO_SIZE_T(Name) ((SIZE_T)Name.QuadPart)
#endif
#endif

#ifndef PLARGE_INTEGER_TO_SIZE_T
#ifdef _WIN32
#define PLARGE_INTEGER_TO_SIZE_T(Name) ((SIZE_T)Name->LowPart)
#else
#define PLARGE_INTEGER_TO_SIZE_T(Name) ((SIZE_T)Name->QuadPart)
#endif
#endif

#ifndef NOTHING
#define NOTHING
#endif

#if PH_WINDOWS

#ifndef DECLSPEC_NOINLINE
#define DECLSPEC_NOINLINE __declspec(noinline)
#endif

#ifndef NOINLINE
#define NOINLINE __declspec(noinline)
#endif

#ifndef INLINE
#define INLINE __inline
#endif

#ifndef FORCEINLINE
#define FORCEINLINE __forceinline
#endif

#else // PH_WINDOWS

#define FORCEINLINE static inline __attribute__((always_inline))
#define DECLSPEC_NOINLINE __attribute__((noinline))
#define NOINLINE __attribute__((noinline))

#endif

#ifdef PH_WINDOWS
#ifdef RtlCopyMemory
#undef RtlCopyMemory
#endif

#ifdef RtlZeroMemory
#undef RtlZeroMemory
#endif

#ifdef RtlFillMemory
#undef RtlFillMemory
#endif

#ifdef RtlMoveMemory
#undef RtlMoveMemory
#endif

#ifdef CopyMemory
#undef CopyMemory
#endif
#define CopyMemory Rtl->RtlCopyMemory

#ifdef MoveMemory
#undef MoveMemory
#endif
#define MoveMemory Rtl->RtlMoveMemory

#ifdef FillMemory
#undef FillMemory
#endif
#define FillMemory(Dest, Length, Fill) Rtl->RtlFillMemory(Dest, Length, Fill)

#ifdef ZeroMemory
#undef ZeroMemory
#endif
#define ZeroMemory(Dest, Length) FillMemory(Dest, Length, 0)
#else // PH_WINDOWS

#ifdef CopyMemory
#undef CopyMemory
#endif
#define CopyMemory RtlCopyMemory

#ifdef MoveMemory
#undef MoveMemory
#endif
#define MoveMemory RtlMoveMemory

#ifdef FillMemory
#undef FillMemory
#endif
#define FillMemory RtlFillMemory

#ifdef ZeroMemory
#undef ZeroMemory
#endif
#define ZeroMemory RtlZeroMemory

#endif // PH_WINDOWS

#ifndef ZeroStruct
#define ZeroStruct(Name) ZeroMemory(&Name, sizeof(Name))
#endif

#ifndef ZeroArray
#define ZeroArray(Name) ZeroMemory(Name, sizeof(Name))
#endif

#ifndef ZeroStructPointer
#define ZeroStructPointer(Name) ZeroMemory(Name, sizeof(*Name))
#endif

#ifndef RtlOffsetToPointer
#define RtlOffsetToPointer(B,O)    ((PCHAR)(((PCHAR)(B)) + ((ULONG_PTR)(O))))
#endif

#ifndef RtlOffsetFromPointer
#define RtlOffsetFromPointer(B,O)  ((PCHAR)(((PCHAR)(B)) - ((ULONG_PTR)(O))))
#endif

#ifndef RtlPointerToOffset
#define RtlPointerToOffset(B,P)    ((ULONG_PTR)(((PCHAR)(P)) - ((PCHAR)(B))))
#endif

#ifdef PH_WINDOWS
#define RtlInitOnceToPointer(A) (                                           \
    ((ULONG_PTR)(A) & ~(ULONG_PTR)((1 << INIT_ONCE_CTX_RESERVED_BITS) - 1)) \
)
#else
#define RtlInitOnceToPointer(A) (((PINIT_ONCE)(A))->Context)
#endif

#ifndef FlagOn
#define FlagOn(_F,_SF)        ((_F) & (_SF))
#endif

#ifndef BooleanFlagOn
#define BooleanFlagOn(F,SF)   ((BOOLEAN)(((F) & (SF)) != 0))
#endif

#ifndef SetFlag
#define SetFlag(_F,_SF)       ((_F) |= (_SF))
#endif

#ifndef ClearFlag
#define ClearFlag(_F,_SF)     ((_F) &= ~(_SF))
#endif

#if defined(_M_AMD64) || defined(_M_X64) || defined(_M_IX86)
////////////////////////////////////////////////////////////////////////////////
// Intel/AMD x86/x64 CPU Features
////////////////////////////////////////////////////////////////////////////////

typedef union _CPU_INFO {
    struct {
        LONG Eax;
        LONG Ebx;
        LONG Ecx;
        LONG Edx;
    };
    struct {
        ULONG_BYTES EaxBytes;
        ULONG_BYTES EbxBytes;
        ULONG_BYTES EcxBytes;
        ULONG_BYTES EdxBytes;
    };
    INT AsIntArray[4];
    LONG AsLongArray[4];
    ULONG AsULongArray[4];
    CHAR AsCharArray[16];
} CPU_INFO;
typedef CPU_INFO *PCPU_INFO;

#define EXPAND_AS_CPU_FEATURE_BYTE(Name, Offset, Bits) \
    BYTE Name[Bits];

#define EXPAND_AS_CPU_FEATURE_BITFLAG(Name, Offset, Bits) \
    ULONG Name:Bits;

#define EXPAND_AS_CPU_FEATURE_C_ASSERT_OFFSET(Name, Offset, Bits) \
    C_ASSERT(FIELD_OFFSET(Type, Name) == Offset);

//
// F1_ECX
//

#define RTL_CPU_FEATURES_F1_ECX_TABLE(FIRST_ENTRY, \
                                      ENTRY,       \
                                      LAST_ENTRY)  \
    FIRST_ENTRY(SSE3, 0, 1)                        \
    ENTRY(PCLMULQDQ, 1, 1)                         \
    ENTRY(DTES64, 2, 1)                            \
    ENTRY(MONITOR, 3, 1)                           \
    ENTRY(DSCPL, 4, 1)                             \
    ENTRY(VMX, 5, 1)                               \
    ENTRY(SMX, 6, 1)                               \
    ENTRY(EIST, 7, 1)                              \
    ENTRY(TM2, 8, 1)                               \
    ENTRY(SSSE3, 9, 1)                             \
    ENTRY(CNXTID, 10, 1)                           \
    ENTRY(SDBG, 11, 1)                             \
    ENTRY(FMA, 12, 1)                              \
    ENTRY(CMPXCHG16B, 13, 1)                       \
    ENTRY(XTPRUC, 14, 1)                           \
    ENTRY(PDCM, 15, 1)                             \
    ENTRY(_Reserved_16_F1_ECX, 16, 1)              \
    ENTRY(PCID, 17, 1)                             \
    ENTRY(DCA, 18, 1)                              \
    ENTRY(SSE41, 19, 1)                            \
    ENTRY(SSE42, 20, 1)                            \
    ENTRY(X2APIC, 21, 1)                           \
    ENTRY(MOVBE, 22, 1)                            \
    ENTRY(POPCNT, 23, 1)                           \
    ENTRY(TSCD, 24, 1)                             \
    ENTRY(AESNI, 25, 1)                            \
    ENTRY(XSAVE, 26, 1)                            \
    ENTRY(OSXSAVE, 27, 1)                          \
    ENTRY(AVX, 28, 1)                              \
    ENTRY(F16C, 29, 1)                             \
    ENTRY(RDRAND, 30, 1)                           \
    LAST_ENTRY(_NotUsed_31_F1_ECX, 31, 1)

#define RTL_CPU_FEATURES_F1_ECX_TABLE_ENTRY(ENTRY) \
    RTL_CPU_FEATURES_F1_ECX_TABLE(ENTRY,           \
                                  ENTRY,           \
                                  ENTRY)

#pragma pack(push, 1)
typedef struct _RTL_CPU_FEATURES_F1_ECX_AS_BYTES {
    RTL_CPU_FEATURES_F1_ECX_TABLE_ENTRY(EXPAND_AS_CPU_FEATURE_BYTE)
} RTL_CPU_FEATURES_F1_ECX_AS_BYTES;
#pragma pack(pop)
C_ASSERT(sizeof(RTL_CPU_FEATURES_F1_ECX_AS_BYTES) ==
         sizeof(ULONG) << 3);

#define Type RTL_CPU_FEATURES_F1_ECX_AS_BYTES
RTL_CPU_FEATURES_F1_ECX_TABLE_ENTRY(
    EXPAND_AS_CPU_FEATURE_C_ASSERT_OFFSET
);
#undef Type

typedef union _RTL_CPU_FEATURES_F1_ECX {
    struct _Struct_size_bytes_(sizeof(ULONG)) {
        RTL_CPU_FEATURES_F1_ECX_TABLE_ENTRY(
            EXPAND_AS_CPU_FEATURE_BITFLAG
        )
    };
    LONG AsLong;
    ULONG AsULong;
} RTL_CPU_FEATURES_F1_ECX;
C_ASSERT(sizeof(RTL_CPU_FEATURES_F1_ECX) == sizeof(ULONG));
typedef RTL_CPU_FEATURES_F1_ECX *PRTL_CPU_FEATURES_F1_ECX;

//
// F1_EDX
//

#define RTL_CPU_FEATURES_F1_EDX_TABLE(FIRST_ENTRY, \
                                      ENTRY,       \
                                      LAST_ENTRY)  \
    FIRST_ENTRY(FPU, 0, 1)                         \
    ENTRY(VME, 1, 1)                               \
    ENTRY(DE, 2, 1)                                \
    ENTRY(PSE, 3, 1)                               \
    ENTRY(TSC, 4, 1)                               \
    ENTRY(MSR, 5, 1)                               \
    ENTRY(PAE, 6, 1)                               \
    ENTRY(MCE, 7, 1)                               \
    ENTRY(CX8, 8, 1)                               \
    ENTRY(APIC, 9, 1)                              \
    ENTRY(_Reserved_10_F1_EDX, 10, 1)              \
    ENTRY(SEP, 11, 1)                              \
    ENTRY(MTRR, 12, 1)                             \
    ENTRY(PGE, 13, 1)                              \
    ENTRY(MCA, 14, 1)                              \
    ENTRY(CMOV, 15, 1)                             \
    ENTRY(PAT, 16, 1)                              \
    ENTRY(PSE36, 17, 1)                            \
    ENTRY(PSN, 18, 1)                              \
    ENTRY(CLFSH, 19, 1)                            \
    ENTRY(_Reserved_20_F1_EDX, 20, 1)              \
    ENTRY(DS, 21, 1)                               \
    ENTRY(ACPI, 22, 1)                             \
    ENTRY(MMX, 23, 1)                              \
    ENTRY(FXSR, 24, 1)                             \
    ENTRY(SSE, 25, 1)                              \
    ENTRY(SSE2, 26, 1)                             \
    ENTRY(SS, 27, 1)                               \
    ENTRY(HTT, 28, 1)                              \
    ENTRY(TM, 29, 1)                               \
    ENTRY(_Reserved_30_F1_EDX, 30, 1)              \
    LAST_ENTRY(PBE, 31, 1)

#define RTL_CPU_FEATURES_F1_EDX_TABLE_ENTRY(ENTRY) \
    RTL_CPU_FEATURES_F1_EDX_TABLE(ENTRY,           \
                                  ENTRY,           \
                                  ENTRY)

#pragma pack(push, 1)
typedef struct _RTL_CPU_FEATURES_F1_EDX_AS_BYTES {
    RTL_CPU_FEATURES_F1_EDX_TABLE_ENTRY(EXPAND_AS_CPU_FEATURE_BYTE)
} RTL_CPU_FEATURES_F1_EDX_AS_BYTES;
#pragma pack(pop)
C_ASSERT(sizeof(RTL_CPU_FEATURES_F1_EDX_AS_BYTES) ==
         sizeof(ULONG) << 3);

#define Type RTL_CPU_FEATURES_F1_EDX_AS_BYTES
RTL_CPU_FEATURES_F1_EDX_TABLE_ENTRY(
    EXPAND_AS_CPU_FEATURE_C_ASSERT_OFFSET
);
#undef Type

typedef union _RTL_CPU_FEATURES_F1_EDX {
    struct _Struct_size_bytes_(sizeof(ULONG)) {
        RTL_CPU_FEATURES_F1_EDX_TABLE_ENTRY(
            EXPAND_AS_CPU_FEATURE_BITFLAG
        )
    };
    LONG AsLong;
    ULONG AsULong;
} RTL_CPU_FEATURES_F1_EDX;
C_ASSERT(sizeof(RTL_CPU_FEATURES_F1_EDX) == sizeof(ULONG));
typedef RTL_CPU_FEATURES_F1_EDX *PRTL_CPU_FEATURES_F1_EDX;

//
// F7_EBX
//

#define RTL_CPU_FEATURES_F7_EBX_TABLE(FIRST_ENTRY, \
                                      ENTRY,       \
                                      LAST_ENTRY)  \
    FIRST_ENTRY(FSGSBASE, 0, 1)                    \
    ENTRY(IA32_TSC_ADJUST_MSR, 1, 1)               \
    ENTRY(SGX, 2, 1)                               \
    ENTRY(BMI1, 3, 1)                              \
    ENTRY(HLE, 4, 1)                               \
    ENTRY(AVX2, 5, 1)                              \
    ENTRY(FDP_EXCPTN_ONLY, 6, 1)                   \
    ENTRY(SMEP, 7, 1)                              \
    ENTRY(BMI2, 8, 1)                              \
    ENTRY(ERMS, 9, 1)                              \
    ENTRY(INVPCID, 10, 1)                          \
    ENTRY(RTM, 11, 1)                              \
    ENTRY(RDTM, 12, 1)                             \
    ENTRY(NOFPUCSDS, 13, 1)                        \
    ENTRY(MPX, 14, 1)                              \
    ENTRY(RDTA, 15, 1)                             \
    ENTRY(AVX512F, 16, 1)                          \
    ENTRY(AVX512DQ, 17, 1)                         \
    ENTRY(RDSEED, 18, 1)                           \
    ENTRY(ADX, 19, 1)                              \
    ENTRY(SMAP, 20, 1)                             \
    ENTRY(AVX512_IFMA, 21, 1)                      \
    ENTRY(_Reserved_22_F7_EBX, 22, 1)              \
    ENTRY(CLFLUSHOPT, 23, 1)                       \
    ENTRY(CLWB, 24, 1)                             \
    ENTRY(IPT, 25, 1)                              \
    ENTRY(AVX512PF, 26, 1)                         \
    ENTRY(AVX512ER, 27, 1)                         \
    ENTRY(AVX512CD, 28, 1)                         \
    ENTRY(SHA, 29, 1)                              \
    ENTRY(AVX512BW, 30, 1)                         \
    LAST_ENTRY(AVX512VL, 31, 1)

#define RTL_CPU_FEATURES_F7_EBX_TABLE_ENTRY(ENTRY) \
    RTL_CPU_FEATURES_F7_EBX_TABLE(ENTRY,           \
                                  ENTRY,           \
                                  ENTRY)

#pragma pack(push, 1)
typedef struct _RTL_CPU_FEATURES_F7_EBX_AS_BYTES {
    RTL_CPU_FEATURES_F7_EBX_TABLE_ENTRY(EXPAND_AS_CPU_FEATURE_BYTE)
} RTL_CPU_FEATURES_F7_EBX_AS_BYTES;
#pragma pack(pop)
C_ASSERT(sizeof(RTL_CPU_FEATURES_F7_EBX_AS_BYTES) ==
         sizeof(ULONG) << 3);

#define Type RTL_CPU_FEATURES_F7_EBX_AS_BYTES
RTL_CPU_FEATURES_F7_EBX_TABLE_ENTRY(
    EXPAND_AS_CPU_FEATURE_C_ASSERT_OFFSET
);
#undef Type

typedef union _RTL_CPU_FEATURES_F7_EBX {
    struct _Struct_size_bytes_(sizeof(ULONG)) {
        RTL_CPU_FEATURES_F7_EBX_TABLE_ENTRY(
            EXPAND_AS_CPU_FEATURE_BITFLAG
        )
    };
    LONG AsLong;
    ULONG AsULong;
} RTL_CPU_FEATURES_F7_EBX;
C_ASSERT(sizeof(RTL_CPU_FEATURES_F7_EBX) == sizeof(ULONG));
typedef RTL_CPU_FEATURES_F7_EBX *PRTL_CPU_FEATURES_F7_EBX;

//
// F7_ECX
//

#define RTL_CPU_FEATURES_F7_ECX_TABLE(FIRST_ENTRY, \
                                      ENTRY,       \
                                      LAST_ENTRY)  \
    FIRST_ENTRY(PREFETCHWT1, 0, 1)                 \
    ENTRY(AVX512_VBMI, 1, 1)                       \
    ENTRY(UMIP, 2, 1)                              \
    ENTRY(PKU, 3, 1)                               \
    ENTRY(OSPKE, 4, 1)                             \
    ENTRY(WAITPKG, 5, 1)                           \
    ENTRY(_Reserved_6_7_F7_ECX, 6, 2)              \
    ENTRY(GFNI, 8, 1)                              \
    ENTRY(_Reserved_9_13_F7_ECX, 9, 5)             \
    ENTRY(AVX512_VPOPCNTDQ, 14, 1)                 \
    ENTRY(_Reserved_15_16_F7_ECX, 15, 2)           \
    ENTRY(MAWAU, 17, 5)                            \
    ENTRY(RDPID_IA32_TSC_AUX, 22, 1)               \
    ENTRY(_Reserved_23_24_F7_ECX, 23, 2)           \
    ENTRY(CLDEMOTE, 25, 1)                         \
    ENTRY(_Reserved_26_F7_ECX, 26, 1)              \
    ENTRY(MOVDIRI, 27, 1)                          \
    ENTRY(MOVDIR64B, 28, 1)                        \
    ENTRY(_Reserved_29_F7_ECX, 29, 1)              \
    ENTRY(SGX_LC, 30, 1)                           \
    LAST_ENTRY(_Reserved_30_F7_ECX, 31, 1)

#define RTL_CPU_FEATURES_F7_ECX_TABLE_ENTRY(ENTRY) \
    RTL_CPU_FEATURES_F7_ECX_TABLE(ENTRY,           \
                                  ENTRY,           \
                                  ENTRY)

#pragma pack(push, 1)
typedef struct _RTL_CPU_FEATURES_F7_ECX_AS_BYTES {
    RTL_CPU_FEATURES_F7_ECX_TABLE_ENTRY(EXPAND_AS_CPU_FEATURE_BYTE)
} RTL_CPU_FEATURES_F7_ECX_AS_BYTES;
#pragma pack(pop)
C_ASSERT(sizeof(RTL_CPU_FEATURES_F7_ECX_AS_BYTES) ==
         sizeof(ULONG) << 3);

#define Type RTL_CPU_FEATURES_F7_ECX_AS_BYTES
RTL_CPU_FEATURES_F7_ECX_TABLE_ENTRY(
    EXPAND_AS_CPU_FEATURE_C_ASSERT_OFFSET
);
#undef Type

typedef union _RTL_CPU_FEATURES_F7_ECX {
    struct _Struct_size_bytes_(sizeof(ULONG)) {
        RTL_CPU_FEATURES_F7_ECX_TABLE_ENTRY(
            EXPAND_AS_CPU_FEATURE_BITFLAG
        )
    };
    LONG AsLong;
    ULONG AsULong;
} RTL_CPU_FEATURES_F7_ECX;
C_ASSERT(sizeof(RTL_CPU_FEATURES_F7_ECX) == sizeof(ULONG));
typedef RTL_CPU_FEATURES_F7_ECX *PRTL_CPU_FEATURES_F7_ECX;

//
// F7_EDX
//

#define RTL_CPU_FEATURES_F7_EDX_TABLE(FIRST_ENTRY, \
                                      ENTRY,       \
                                      LAST_ENTRY)  \
    FIRST_ENTRY(_Reserved_0_1_F7_EDX, 0, 2)        \
    ENTRY(AVX512_4VNNIW, 2, 1)                     \
    ENTRY(AVX512_4FMAPS, 3, 1)                     \
    ENTRY(_Reserved_4_25_F7_EDX, 4, 22)            \
    ENTRY(IBRS, 26, 1)                             \
    ENTRY(STIBP, 27, 1)                            \
    ENTRY(L1D_FLUSH, 28, 1)                        \
    ENTRY(IA32_ARCH_CAPABILITIES, 29, 1)           \
    ENTRY(IA32_CORE_CAPABILITIES, 30, 1)           \
    LAST_ENTRY(SSBD, 31, 1)

#define RTL_CPU_FEATURES_F7_EDX_TABLE_ENTRY(ENTRY) \
    RTL_CPU_FEATURES_F7_EDX_TABLE(ENTRY,           \
                                  ENTRY,           \
                                  ENTRY)

#pragma pack(push, 1)
typedef struct _RTL_CPU_FEATURES_F7_EDX_AS_BYTES {
    RTL_CPU_FEATURES_F7_EDX_TABLE_ENTRY(EXPAND_AS_CPU_FEATURE_BYTE)
} RTL_CPU_FEATURES_F7_EDX_AS_BYTES;
#pragma pack(pop)
C_ASSERT(sizeof(RTL_CPU_FEATURES_F7_EDX_AS_BYTES) ==
         sizeof(ULONG) << 3);

#define Type RTL_CPU_FEATURES_F7_EDX_AS_BYTES
RTL_CPU_FEATURES_F7_EDX_TABLE_ENTRY(
    EXPAND_AS_CPU_FEATURE_C_ASSERT_OFFSET
);
#undef Type

typedef union _RTL_CPU_FEATURES_F7_EDX {
    struct _Struct_size_bytes_(sizeof(ULONG)) {
        RTL_CPU_FEATURES_F7_EDX_TABLE_ENTRY(
            EXPAND_AS_CPU_FEATURE_BITFLAG
        )
    };
    LONG AsLong;
    ULONG AsULong;
} RTL_CPU_FEATURES_F7_EDX;
C_ASSERT(sizeof(RTL_CPU_FEATURES_F7_EDX) == sizeof(ULONG));
typedef RTL_CPU_FEATURES_F7_EDX *PRTL_CPU_FEATURES_F7_EDX;

//
// F81_ECX
//

#define RTL_CPU_FEATURES_F81_ECX_TABLE(FIRST_ENTRY, \
                                       ENTRY,       \
                                       LAST_ENTRY)  \
    FIRST_ENTRY(LAHFSAHF, 0, 1)                     \
    ENTRY(_Reserved_1_4_F81_ECX, 1, 4)              \
    ENTRY(LZCNT, 5, 1)                              \
    ENTRY(_Reserved_6_7_F81_ECX, 6, 2)              \
    ENTRY(PREFETCHW, 8, 1)                          \
    LAST_ENTRY(_Reserved_9_31_F81_ECX, 9, 23)

#define RTL_CPU_FEATURES_F81_ECX_TABLE_ENTRY(ENTRY) \
    RTL_CPU_FEATURES_F81_ECX_TABLE(ENTRY,           \
                                   ENTRY,           \
                                   ENTRY)

#pragma pack(push, 1)
typedef struct _RTL_CPU_FEATURES_F81_ECX_AS_BYTES {
    RTL_CPU_FEATURES_F81_ECX_TABLE_ENTRY(EXPAND_AS_CPU_FEATURE_BYTE)
} RTL_CPU_FEATURES_F81_ECX_AS_BYTES;
#pragma pack(pop)
C_ASSERT(sizeof(RTL_CPU_FEATURES_F81_ECX_AS_BYTES) ==
         sizeof(ULONG) << 3);

#define Type RTL_CPU_FEATURES_F81_ECX_AS_BYTES
RTL_CPU_FEATURES_F81_ECX_TABLE_ENTRY(
    EXPAND_AS_CPU_FEATURE_C_ASSERT_OFFSET
);
#undef Type

typedef union _RTL_CPU_FEATURES_F81_ECX {
    struct _Struct_size_bytes_(sizeof(ULONG)) {
        RTL_CPU_FEATURES_F81_ECX_TABLE_ENTRY(
            EXPAND_AS_CPU_FEATURE_BITFLAG
        )
    };
    LONG AsLong;
    ULONG AsULong;
} RTL_CPU_FEATURES_F81_ECX;
C_ASSERT(sizeof(RTL_CPU_FEATURES_F81_ECX) == sizeof(ULONG));
typedef RTL_CPU_FEATURES_F81_ECX *PRTL_CPU_FEATURES_F81_ECX;

//
// F81_EDX
//

#define RTL_CPU_FEATURES_F81_EDX_TABLE(FIRST_ENTRY, \
                                       ENTRY,       \
                                       LAST_ENTRY)  \
    FIRST_ENTRY(_Reserved_0_10_F81_EDX, 0, 11)      \
    ENTRY(SYSCALLSYSRET, 11, 1)                     \
    ENTRY(_Reserved_12_19_F81_EDX, 12, 8)           \
    ENTRY(EDB, 20, 1)                               \
    ENTRY(_Reserved_21_25_F81_EDX, 21, 5)           \
    ENTRY(ONEGBPAGES, 26, 1)                        \
    ENTRY(RDTSCP_IA32_TSC_AUX, 27, 1)               \
    ENTRY(_Reserved_28_F81_EDX, 28, 1)              \
    ENTRY(IA64, 29, 1)                              \
    LAST_ENTRY(_Reserved_30_31_F81_EDX, 30, 2)

#define RTL_CPU_FEATURES_F81_EDX_TABLE_ENTRY(ENTRY) \
    RTL_CPU_FEATURES_F81_EDX_TABLE(ENTRY,           \
                                   ENTRY,           \
                                   ENTRY)

#pragma pack(push, 1)
typedef struct _RTL_CPU_FEATURES_F81_EDX_AS_BYTES {
    RTL_CPU_FEATURES_F81_EDX_TABLE_ENTRY(EXPAND_AS_CPU_FEATURE_BYTE)
} RTL_CPU_FEATURES_F81_EDX_AS_BYTES;
#pragma pack(pop)
C_ASSERT(sizeof(RTL_CPU_FEATURES_F81_EDX_AS_BYTES) ==
         sizeof(ULONG) << 3);

#define Type RTL_CPU_FEATURES_F81_EDX_AS_BYTES
RTL_CPU_FEATURES_F81_EDX_TABLE_ENTRY(
    EXPAND_AS_CPU_FEATURE_C_ASSERT_OFFSET
);
#undef Type

typedef union _RTL_CPU_FEATURES_F81_EDX {
    struct _Struct_size_bytes_(sizeof(ULONG)) {
        RTL_CPU_FEATURES_F81_EDX_TABLE_ENTRY(
            EXPAND_AS_CPU_FEATURE_BITFLAG
        )
    };
    LONG AsLong;
    ULONG AsULong;
} RTL_CPU_FEATURES_F81_EDX;
C_ASSERT(sizeof(RTL_CPU_FEATURES_F81_EDX) == sizeof(ULONG));
typedef RTL_CPU_FEATURES_F81_EDX *PRTL_CPU_FEATURES_F81_EDX;

//
// Define the union of all CPU feature flags we support for x86/x64.
//

typedef union _CPU_VENDOR {
    struct {
        ULONG Unknown:1;
        ULONG IsIntel:1;
        ULONG IsAMD:1;
        ULONG Unused:29;
    };
    LONG AsLong;
    ULONG AsULong;
} CPU_VENDOR;
typedef CPU_VENDOR *PCPU_VENDOR;
C_ASSERT(sizeof(CPU_VENDOR) == sizeof(ULONG));

//
// Internal CPU feature flags (i.e. not related to the actual CPU, but to the
// structure itself, such as availability of certain data).
//

typedef union _RTL_CPU_FEATURES_FLAGS {
    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, indicates the ProcessorInfo structure has been successfully
        // initialized and can be used.
        //

        ULONG HasProcessorInformation:1;

        //
        // Unused bits.
        //

        ULONG Unused:31;
    };

    LONG AsLong;
    ULONG AsULong;
} RTL_CPU_FEATURES_FLAGS, *PRTL_CPU_FEATURES_FLAGS;
C_ASSERT(sizeof(RTL_CPU_FEATURES_FLAGS) == sizeof(ULONG));

typedef struct _SYSTEM_LOGICAL_PROCESSOR_INFO_ARRAY {
    SIZE_T Count;
    SIZE_T SizeInBytes;

    _Readable_elements_(Count)
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ProcInfo;
} SYSTEM_LOGICAL_PROCESSOR_INFO_ARRAY, *PSYSTEM_LOGICAL_PROCESSOR_INFO_ARRAY;

typedef union _CPU_CACHE_LEVEL {
    struct {
        CACHE_DESCRIPTOR Unified;
        CACHE_DESCRIPTOR Instruction;
        CACHE_DESCRIPTOR Data;
        CACHE_DESCRIPTOR Trace;
    };

    CACHE_DESCRIPTOR AsArray[4];
} CPU_CACHE_LEVEL, *PCPU_CACHE_LEVEL;

typedef struct _CPU_CACHES {
    BYTE NumberOfLevels;
    BYTE Padding[3];

    union {
        struct {
            CPU_CACHE_LEVEL L1;
            CPU_CACHE_LEVEL L2;
            CPU_CACHE_LEVEL L3;
            CPU_CACHE_LEVEL L4;
        };

        CPU_CACHE_LEVEL Level[4];
    };
} CPU_CACHES, *PCPU_CACHES;

typedef struct _RTL_CPU_FEATURES {

    CPU_VENDOR Vendor;
    LONG HighestFeatureId;
    LONG HighestExtendedFeatureId;

    //
    // AMD-specific features that we set manually.
    //

    struct _Struct_size_bytes_(sizeof(ULONG)) {
        ULONG ABM:1;
        ULONG SSE4A:1;
        ULONG XOP:1;
        ULONG TBM:1;
        ULONG SVM:1;
        ULONG IBS:1;
        ULONG LWP:1;
        ULONG MMXEXT:1;
        ULONG THREEDNOW:1;
        ULONG THREEDNOWEXT:1;
    } AMD;

    //
    // Intel-specific features that we set manually.
    //

    struct _Struct_size_bytes_(sizeof(ULONG)) {
        ULONG HLE:1;
        ULONG RTM:1;
        ULONG BMI1:1;
        ULONG BMI2:1;
        ULONG LZCNT:1;
        ULONG POPCNT:1;
        ULONG SYSCALL:1;
        ULONG RDTSCP:1;
    } Intel;

    RTL_CPU_FEATURES_FLAGS Flags;

    union {

        //
        // F1_ECX
        // F1_EDX
        // F7_EBX
        // F7_ECX
        // F81_ECX
        // F81_EDX
        //

        struct {
            union {
                struct {
                    RTL_CPU_FEATURES_F1_ECX_TABLE_ENTRY(
                        EXPAND_AS_CPU_FEATURE_BITFLAG
                    )
                };
                RTL_CPU_FEATURES_F1_ECX F1Ecx;
            };
            union {
                struct {
                    RTL_CPU_FEATURES_F1_EDX_TABLE_ENTRY(
                        EXPAND_AS_CPU_FEATURE_BITFLAG
                    )
                };
                RTL_CPU_FEATURES_F1_EDX F1Edx;
            };
            union {
                struct {
                    RTL_CPU_FEATURES_F7_EBX_TABLE_ENTRY(
                        EXPAND_AS_CPU_FEATURE_BITFLAG
                    )
                };
                RTL_CPU_FEATURES_F7_EBX F7Ebx;
            };
            union {
                struct {
                    RTL_CPU_FEATURES_F7_ECX_TABLE_ENTRY(
                        EXPAND_AS_CPU_FEATURE_BITFLAG
                    )
                };
                RTL_CPU_FEATURES_F7_ECX F7Ecx;
            };
            union {
                struct {
                    RTL_CPU_FEATURES_F81_ECX_TABLE_ENTRY(
                        EXPAND_AS_CPU_FEATURE_BITFLAG
                    )
                };
                RTL_CPU_FEATURES_F81_ECX F81Ecx;
            };
            union {
                struct {
                    RTL_CPU_FEATURES_F81_EDX_TABLE_ENTRY(
                        EXPAND_AS_CPU_FEATURE_BITFLAG
                    )
                };
                RTL_CPU_FEATURES_F81_EDX F81Edx;
            };
        };

        INT AsIntArray[6];
        LONG AsLongArray[6];
        ULONG AsULongArray[6];
    };

    //
    // CPU Brand String.
    //

    STRING Brand;
    CHAR BrandBuffer[48];

    //
    // CPU Information.
    //

    ULONG LogicalProcessorCount;
    ULONG NumaNodeCount;
    ULONG ProcessorCoreCount;
    ULONG ProcessorL1CacheCount;
    ULONG ProcessorL2CacheCount;
    ULONG ProcessorL3CacheCount;
    ULONG ProcessorPackageCount;
    ULONG Padding2;

    CPU_CACHES Caches;

    ULONG Padding3;
    SYSTEM_LOGICAL_PROCESSOR_INFO_ARRAY ProcInfoArray;

} RTL_CPU_FEATURES;
typedef RTL_CPU_FEATURES *PRTL_CPU_FEATURES;

#endif // defined(_M_AMD64) || defined(_M_X64) || defined(_M_IX86)

////////////////////////////////////////////////////////////////////////////////
// Memory/String
////////////////////////////////////////////////////////////////////////////////

typedef
VOID
(NTAPI RTL_ZERO_MEMORY)(
    _Out_writes_bytes_all_(Length) PVOID Destination,
    _In_ ULONG_PTR Length
    );
typedef RTL_ZERO_MEMORY *PRTL_ZERO_MEMORY;

typedef
VOID
(NTAPI RTL_FILL_MEMORY)(
    _Out_writes_bytes_all_(Length) PVOID Destination,
    _In_ ULONG_PTR Length,
    _In_ BYTE Fill
    );
typedef RTL_FILL_MEMORY *PRTL_FILL_MEMORY;

typedef
VOID
(NTAPI RTL_MOVE_MEMORY)(
    _Out_writes_bytes_all_(Length) PVOID Destination,
    _In_ const PVOID Source,
    _In_ ULONG_PTR Length
    );
typedef RTL_MOVE_MEMORY *PRTL_MOVE_MEMORY;

typedef
SIZE_T
(NTAPI RTL_COMPARE_MEMORY)(
    _In_ const VOID *Source1,
    _In_ const VOID *Source2,
    _In_ SIZE_T Length
    );
typedef RTL_COMPARE_MEMORY *PRTL_COMPARE_MEMORY;

typedef
NTSTATUS
(NTAPI RTL_CHAR_TO_INTEGER)(
    _In_ PCSZ String,
    _In_ ULONG Base,
    _Out_ PULONG Value
    );
typedef RTL_CHAR_TO_INTEGER *PRTL_CHAR_TO_INTEGER;

typedef
NTSTATUS
(NTAPI RTL_INTEGER_TO_CHAR)(
    _In_ ULONG Value,
    _In_ ULONG Base,
    _In_ ULONG SizeOfBuffer,
    _Out_writes_bytes_(SizeOfBuffer) PCHAR Buffer
    );
typedef RTL_INTEGER_TO_CHAR *PRTL_INTEGER_TO_CHAR;

typedef
NTSTATUS
(NTAPI RTL_UNICODE_STRING_TO_INTEGER)(
    _In_ PCUNICODE_STRING String,
    _In_ ULONG Base,
    _Out_ PULONG Value
    );
typedef RTL_UNICODE_STRING_TO_INTEGER *PRTL_UNICODE_STRING_TO_INTEGER;

typedef
NTSTATUS
(RTL_UNICODE_STRING_TO_INT64)(
    _In_ PCUNICODE_STRING String,
    _In_ ULONG Base,
    _Out_ PLONG64 Number,
    _Out_opt_ PWSTR *EndPointer
    );
typedef RTL_UNICODE_STRING_TO_INT64 *PRTL_UNICODE_STRING_TO_INT64;

typedef
BOOLEAN
(NTAPI RTL_EQUAL_UNICODE_STRING)(
    _In_ PCUNICODE_STRING String1,
    _In_ PCUNICODE_STRING String2,
    _In_ BOOLEAN CaseInSensitive
    );
typedef RTL_EQUAL_UNICODE_STRING *PRTL_EQUAL_UNICODE_STRING;

typedef
BOOLEAN
(NTAPI RTL_PREFIX_UNICODE_STRING)(
    _In_ PCUNICODE_STRING String1,
    _In_ PCUNICODE_STRING String2,
    _In_ BOOLEAN CaseInSensitive
    );
typedef RTL_PREFIX_UNICODE_STRING *PRTL_PREFIX_UNICODE_STRING;

typedef
NTSTATUS
(NTAPI RTL_APPEND_UNICODE_STRING_TO_STRING)(
    _Inout_ PUNICODE_STRING  Destination,
    _In_    PCUNICODE_STRING Source
    );
typedef RTL_APPEND_UNICODE_STRING_TO_STRING
      *PRTL_APPEND_UNICODE_STRING_TO_STRING;

typedef
VOID
(NTAPI RTL_COPY_MEMORY)(
    _Out_writes_bytes_all_(Length) PVOID Destination,
    _In_ const PVOID Source,
    _In_ ULONG_PTR Length
    );
typedef RTL_COPY_MEMORY *PRTL_COPY_MEMORY;

//
// In some situations, an Rtl pointer may not be available, which means the
// CopyMemory() macro can't be used, as it expands to Rtl->RtlCopyMemory().
// CopyMemoryInline() routine can be used instead.
//

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

#if defined(_M_X64) && defined(PH_WINDOWS)
    __stosq(Dest, FillQuad, NumberOfQuadwords);
#else
    while (NumberOfQuadwords) {
        *Dest++ = (DWORD64)FillQuad;
        NumberOfQuadwords--;
    }
#endif

    TrailingDest = (PCHAR)Dest;

    while (TrailingBytes) {
        *TrailingDest++ = Fill;
        TrailingBytes--;
    }
}

#define ZeroInline(Dest, Size) ZeroMemoryInline(Dest, Size, FALSE)
#define ZeroArrayInline(Name) ZeroInline(Name, sizeof(Name))

#define AllOnesInline(Dest, Size) ZeroMemoryInline(Dest, Size, TRUE)

//
// Structures are guaranteed to be aligned to an 8 byte boundary on x64, so
// just use __stosq().
//

#if defined(_M_X64) && defined(PH_WINDOWS)

#define ZeroStructInline(Name) \
    __stosq((PDWORD64)&Name, 0, (sizeof(Name) >> 3))

#define ZeroStructPointerInline(Name) \
    __stosq((PDWORD64)Name, 0, (sizeof(*Name) >> 3))

#else
#define ZeroStructInline(Name) \
    memset((PDWORD64)&Name, 0, sizeof(Name))

#define ZeroStructPointerInline(Name) \
    memset((PDWORD64)Name, 0, (sizeof(*Name)))
#endif

#ifndef PH_WINDOWS
#define SecureZeroMemory ZeroInline
#endif

////////////////////////////////////////////////////////////////////////////////
// Crypto
////////////////////////////////////////////////////////////////////////////////

typedef
_Success_(return != 0)
BOOL
(WINAPI CRYPT_BINARY_TO_STRING_A)(
    _In_reads_bytes_(cbBinary) CONST BYTE *pbBinary,
    _In_ DWORD cbBinary,
    _In_ DWORD dwFlags,
    _Out_writes_to_opt_(*pcchString, *pcchString) LPSTR pszString,
    _Inout_ DWORD *pcchString
    );
typedef CRYPT_BINARY_TO_STRING_A *PCRYPT_BINARY_TO_STRING_A;

typedef
_Success_(return != 0)
BOOL
(WINAPI CRYPT_BINARY_TO_STRING_W)(
    _In_reads_bytes_(cbBinary) CONST BYTE *pbBinary,
    _In_ DWORD cbBinary,
    _In_ DWORD dwFlags,
    _Out_writes_to_opt_(*pcchString, *pcchString) LPWSTR pszString,
    _Inout_ DWORD *pcchString
    );
typedef CRYPT_BINARY_TO_STRING_W *PCRYPT_BINARY_TO_STRING_W;

////////////////////////////////////////////////////////////////////////////////
// Searching
////////////////////////////////////////////////////////////////////////////////

typedef
INT
(__cdecl CRTCOMPARE)(
    _In_ const void *Key,
    _In_ const void *Datum
    );
typedef CRTCOMPARE *PCRTCOMPARE;

typedef
INT
(__cdecl CRTCOMPARE_S)(
    _In_ PVOID Context,
    _In_ const void *Key,
    _In_ const void *Datum
    );
typedef CRTCOMPARE_S *PCRTCOMPARE_S;

typedef
PVOID
(__cdecl BSEARCH)(
    _In_ const void *Key,
    _In_reads_bytes_(NumberOfElements * WidthOfElement) const void *Base,
    _In_ SIZE_T NumberOfElements,
    _In_ SIZE_T WidthOfElement,
    _In_ PCRTCOMPARE Compare
    );
typedef BSEARCH *PBSEARCH;

typedef
PVOID
(__cdecl BSEARCH_S)(
    _In_ const void *Key,
    _In_reads_bytes_(NumberOfElements * WidthOfElement) const void *Base,
    _In_ SIZE_T NumberOfElements,
    _In_ SIZE_T WidthOfElement,
    _In_ PCRTCOMPARE_S Compare,
    _In_opt_ PVOID Context
    );
typedef BSEARCH_S *PBSEARCH_S;

typedef
VOID
(__cdecl QSORT)(
    _Inout_updates_bytes_(NumberOfElements * SizeOfElements) PVOID Base,
    _In_ SIZE_T NumberOfElements,
    _In_ SIZE_T SizeOfElements,
    _In_ PCRTCOMPARE Compare
    );
typedef QSORT *PQSORT;

typedef
VOID
(__cdecl QSORT_S)(
    _Inout_updates_bytes_(NumberOfElements * SizeOfElements) PVOID Base,
    _In_ SIZE_T NumberOfElements,
    _In_ SIZE_T SizeOfElements,
    _In_ PCRTCOMPARE_S Compare,
    _In_opt_ PVOID Context
    );
typedef QSORT_S *PQSORT_S;

//
// N.B. ntdll.dll doesn't export _lfind_s, which is why there's no secure
//      LFIND_S equivalent here.
//

typedef
PVOID
(__cdecl LFIND)(
    _In_ const void *Key,
    _In_reads_bytes_((*NumberOfElements) * WidthOfElement) const void *Base,
    _Inout_ LONG *NumberOfElements,
    _Inout_ ULONG WidthOfElement,
    _In_ PCRTCOMPARE Compare
    );
typedef LFIND *PLFIND;

////////////////////////////////////////////////////////////////////////////////
// Stdio
////////////////////////////////////////////////////////////////////////////////

typedef
_Success_(return >= 0)
_Check_return_opt_
LONG
(__cdecl SPRINTF_S)(
    _Out_writes_(_BufferCount) _Always_(_Post_z_) char* const _Buffer,
    _In_ size_t const _BufferCount,
    _In_z_ _Printf_format_string_ char const* const _Format,
    ...);
typedef SPRINTF_S *PSPRINTF_S;

typedef
_Success_(return >= 0)
_Check_return_opt_
LONG
(__cdecl SWPRINTF_S)(
    _Out_writes_(_BufferCount) _Always_(_Post_z_) wchar_t* const _Buffer,
    _In_ size_t const _BufferCount,
    _In_z_ _Printf_format_string_ char const* const _Format,
    ...);
typedef SWPRINTF_S *PSWPRINTF_S;

typedef
_Success_(return >= 0)
LONG
(__cdecl VSPRINTF_S)(
    _Out_writes_(cchDest) _Always_(_Post_z_) char* const pszDest,
    _In_ size_t cchDest,
    _In_ _Printf_format_string_ char* const pszFormat,
    _In_ va_list argList
    );
typedef VSPRINTF_S *PVSPRINTF_S;

typedef
_Success_(return >= 0)
LONG
(__cdecl VSWPRINTF_S)(
    _Out_writes_(_BufferCount) _Always_(_Post_z_) wchar_t* const _Buffer,
    _In_ size_t const _BufferCount,
    _In_z_ _Printf_format_string_ wchar_t const* const _Format,
    va_list _ArgList
    );
typedef VSWPRINTF_S *PVSWPRINTF_S;

typedef
LONG
(__cdecl _WSPLITPATH_S)(
   const _In_ WCHAR *Path,
   _Out_writes_bytes_(SizeOfDrive) WCHAR *Drive,
   _In_ SIZE_T SizeOfDrive,
   _Out_writes_bytes_(SizeOfDir) WCHAR *Dir,
   _In_ SIZE_T SizeOfDir,
   _Out_writes_bytes_(SizeOfFileName) WCHAR *FileName,
   _In_ SIZE_T SizeOfFileName,
   _Out_writes_bytes_(SizeOfExt) WCHAR *Ext,
   _In_ SIZE_T SizeOfExt
   );
typedef _WSPLITPATH_S *P_WSPLITPATH_S;

////////////////////////////////////////////////////////////////////////////////
// Bitmaps
////////////////////////////////////////////////////////////////////////////////

#ifdef PH_WINDOWS
typedef struct _RTL_BITMAP {

    //
    // Number of bits in the bitmap.
    //

    ULONG SizeOfBitMap;

#ifdef _WIN64
    union {
        ULONG Hash;
        ULONG Padding;
    };
#endif

    //
    // Pointer to bitmap buffer.
    //

    PULONG Buffer;

} RTL_BITMAP;
typedef RTL_BITMAP *PRTL_BITMAP;
#endif

typedef struct _RTL_BITMAP_RUN {
    ULONG StartingIndex;
    ULONG NumberOfBits;
} RTL_BITMAP_RUN;
typedef RTL_BITMAP_RUN *PRTL_BITMAP_RUN;

//
// The various bitmap find functions return 0xFFFFFFFF if they couldn't find the
// requested bit pattern.
//

#define BITS_NOT_FOUND 0xFFFFFFFF

//
// Function pointer typedefs.
//

typedef
VOID
(NTAPI *PRTL_INITIALIZE_BITMAP)(
    _Out_ PRTL_BITMAP BitMapHeader,
    _In_opt_ PULONG BitMapBuffer,
    _In_ ULONG SizeOfBitMap
    );

typedef
VOID
(NTAPI *PRTL_CLEAR_BIT)(
    _In_ PRTL_BITMAP BitMapHeader,
    _In_range_(<, BitMapHeader->SizeOfBitMap) ULONG BitNumber
    );

typedef
VOID
(NTAPI *PRTL_SET_BIT)(
    _In_ PRTL_BITMAP BitMapHeader,
    _In_range_(<, BitMapHeader->SizeOfBitMap) ULONG BitNumber
    );

typedef
BOOLEAN
(NTAPI *PRTL_TEST_BIT)(
    _In_ PRTL_BITMAP BitMapHeader,
    _In_range_(<, BitMapHeader->SizeOfBitMap) ULONG BitNumber
    );

typedef
VOID
(NTAPI *PRTL_CLEAR_ALL_BITS)(
    _In_ PRTL_BITMAP BitMapHeader
    );

typedef
VOID
(NTAPI *PRTL_SET_ALL_BITS)(
    _In_ PRTL_BITMAP BitMapHeader
    );

typedef
ULONG
(NTAPI *PRTL_FIND_CLEAR_BITS)(
    _In_ PRTL_BITMAP BitMapHeader,
    _In_ ULONG NumberToFind,
    _In_ ULONG HintIndex
    );

typedef
ULONG
(NTAPI *PRTL_FIND_SET_BITS)(
    _In_ PRTL_BITMAP BitMapHeader,
    _In_ ULONG NumberToFind,
    _In_ ULONG HintIndex
    );

typedef
ULONG
(NTAPI *PRTL_FIND_CLEAR_BITS_AND_SET)(
    _In_ PRTL_BITMAP BitMapHeader,
    _In_ ULONG NumberToFind,
    _In_ ULONG HintIndex
    );

typedef
ULONG
(NTAPI *PRTL_FIND_SET_BITS_AND_CLEAR)(
    _In_ PRTL_BITMAP BitMapHeader,
    _In_ ULONG NumberToFind,
    _In_ ULONG HintIndex
    );

typedef
VOID
(NTAPI *PRTL_CLEAR_BITS)(
    _In_ PRTL_BITMAP BitMapHeader,
    _In_ ULONG StartingIndex,
    _In_ ULONG NumberToClear
    );

typedef
VOID
(NTAPI *PRTL_SET_BITS)(
    _In_ PRTL_BITMAP BitMapHeader,
    _In_ ULONG StartingIndex,
    _In_ ULONG NumberToSet
    );

typedef
ULONG
(NTAPI *PRTL_FIND_CLEAR_BITS)(
    _In_ PRTL_BITMAP BitMapHeader,
    _In_ ULONG NumberToFind,
    _In_ ULONG HintIndex
    );

typedef
ULONG
(NTAPI *PRTL_FIND_SET_BITS)(
    _In_ PRTL_BITMAP BitMapHeader,
    _In_ ULONG NumberToFind,
    _In_ ULONG HintIndex
    );

typedef
ULONG
(NTAPI *PRTL_FIND_CLEAR_BITS_AND_SET)(
    _In_ PRTL_BITMAP BitMapHeader,
    _In_ ULONG NumberToFind,
    _In_ ULONG HintIndex
    );

typedef
ULONG
(NTAPI *PRTL_FIND_SET_BITS_AND_CLEAR)(
    _In_ PRTL_BITMAP BitMapHeader,
    _In_ ULONG NumberToFind,
    _In_ ULONG HintIndex
    );

typedef
VOID
(NTAPI *PRTL_CLEAR_BITS)(
    _In_ PRTL_BITMAP BitMapHeader,
    _In_ ULONG StartingIndex,
    _In_ ULONG NumberToClear
    );

typedef
VOID
(NTAPI *PRTL_SET_BITS)(
    _In_ PRTL_BITMAP BitMapHeader,
    _In_ ULONG StartingIndex,
    _In_ ULONG NumberToSet
    );

typedef
ULONG
(NTAPI *PRTL_FIND_CLEAR_RUNS)(
    _In_ PRTL_BITMAP BitMapHeader,
    _Out_ PRTL_BITMAP_RUN RunArray,
    _In_ ULONG SizeOfRunArray,
    _In_ BOOLEAN LocateLongestRuns
    );

typedef
ULONG
(NTAPI RTL_FIND_LONGEST_RUN_CLEAR)(
    _In_ PRTL_BITMAP BitMapHeader,
    _Out_ PULONG StartingIndex
    );
typedef RTL_FIND_LONGEST_RUN_CLEAR *PRTL_FIND_LONGEST_RUN_CLEAR;

typedef
ULONG
(NTAPI *PRTL_FIND_FIRST_RUN_CLEAR)(
    _In_ PRTL_BITMAP BitMapHeader,
    _Out_ PULONG StartingIndex
    );

typedef
ULONG
(NTAPI *PRTL_NUMBER_OF_CLEAR_BITS)(
    _In_ PRTL_BITMAP BitMapHeader
    );

typedef
ULONG
(NTAPI RTL_NUMBER_OF_SET_BITS)(
    _In_ PRTL_BITMAP BitMapHeader
    );
typedef RTL_NUMBER_OF_SET_BITS *PRTL_NUMBER_OF_SET_BITS;

typedef
BOOLEAN
(NTAPI *PRTL_ARE_BITS_CLEAR)(
    _In_ PRTL_BITMAP BitMapHeader,
    _In_ ULONG StartingIndex,
    _In_ ULONG Length
    );

typedef
BOOLEAN
(NTAPI *PRTL_ARE_BITS_SET)(
    _In_ PRTL_BITMAP BitMapHeader,
    _In_ ULONG StartingIndex,
    _In_ ULONG Length
    );

typedef
ULONG
(NTAPI *PRTL_FIND_NEXT_FORWARD_RUN_CLEAR)(
    _In_ PRTL_BITMAP BitMapHeader,
    _In_ ULONG FromIndex,
    _Out_ PULONG StartingRunIndex
    );

typedef
ULONG
(NTAPI *PRTL_FIND_LAST_BACKWARD_RUN_CLEAR)(
    _In_ PRTL_BITMAP BitMapHeader,
    _In_ ULONG FromIndex,
    _Out_ PULONG StartingRunIndex
    );

////////////////////////////////////////////////////////////////////////////////
// Singly-linked Lists
////////////////////////////////////////////////////////////////////////////////

typedef
PSLIST_ENTRY
(NTAPI RTL_FIRST_ENTRY_SLIST)(
    _In_ const SLIST_HEADER *ListHead
    );
typedef RTL_FIRST_ENTRY_SLIST *PRTL_FIRST_ENTRY_SLIST;

////////////////////////////////////////////////////////////////////////////////
// Doubly-linked Lists
////////////////////////////////////////////////////////////////////////////////

//
// N.B. This is literally a verbatim copy-and-paste from ntddk.h.
//

#define RTL_STATIC_LIST_HEAD(x) LIST_ENTRY x = { &x, &x }

FORCEINLINE
VOID
InitializeListHead(
    _In_ PLIST_ENTRY ListHead
    )

{

    ListHead->Flink = ListHead->Blink = ListHead;
    return;
}

FORCEINLINE
BOOLEAN
IsListEmpty(
    _In_ const LIST_ENTRY * ListHead
    )

{
    return (BOOLEAN)(ListHead->Flink == ListHead);
}

FORCEINLINE
BOOLEAN
RemoveEntryList(
    _In_ PLIST_ENTRY Entry
    )
{

    PLIST_ENTRY Blink;
    PLIST_ENTRY Flink;

    Flink = Entry->Flink;
    Blink = Entry->Blink;
    Blink->Flink = Flink;
    Flink->Blink = Blink;
    return (BOOLEAN)(Flink == Blink);
}

FORCEINLINE
PLIST_ENTRY
RemoveHeadList(
    _Inout_ PLIST_ENTRY ListHead
    )
{

    PLIST_ENTRY Flink;
    PLIST_ENTRY Entry;

    Entry = ListHead->Flink;
    Flink = Entry->Flink;
    ListHead->Flink = Flink;
    Flink->Blink = ListHead;
    return Entry;
}



FORCEINLINE
PLIST_ENTRY
RemoveTailList(
    _Inout_ PLIST_ENTRY ListHead
    )

{

    PLIST_ENTRY Blink;
    PLIST_ENTRY Entry;

    Entry = ListHead->Blink;
    Blink = Entry->Blink;
    ListHead->Blink = Blink;
    Blink->Flink = ListHead;
    return Entry;
}


FORCEINLINE
VOID
InsertTailList(
    _Inout_ PLIST_ENTRY ListHead,
    _Inout_ __drv_aliasesMem PLIST_ENTRY Entry
    )
{

    PLIST_ENTRY Blink;

    Blink = ListHead->Blink;
    Entry->Flink = ListHead;
    Entry->Blink = Blink;
    Blink->Flink = Entry;
    ListHead->Blink = Entry;
    return;
}


FORCEINLINE
VOID
InsertHeadList(
    _Inout_ PLIST_ENTRY ListHead,
    _Inout_ __drv_aliasesMem PLIST_ENTRY Entry
    )
{

    PLIST_ENTRY Flink;

    Flink = ListHead->Flink;
    Entry->Flink = Flink;
    Entry->Blink = ListHead;
    Flink->Blink = Entry;
    ListHead->Flink = Entry;
    return;
}

FORCEINLINE
VOID
AppendTailList(
    _Inout_ PLIST_ENTRY ListHead,
    _Inout_ PLIST_ENTRY ListToAppend
    )
{
    PLIST_ENTRY ListEnd = ListHead->Blink;

    ListHead->Blink->Flink = ListToAppend;
    ListHead->Blink = ListToAppend->Blink;
    ListToAppend->Blink->Flink = ListHead;
    ListToAppend->Blink = ListEnd;
    return;
}

#define FOR_EACH_LIST_ENTRY(Head, Entry) \
    for (Entry = Head->Flink;            \
         Entry != Head;                  \
         Entry = Entry->Flink)

#define FOR_EACH_LIST_ENTRY_REVERSE(Head, Entry) \
    for (Entry = Head->Blink;                    \
         Entry != Head;                          \
         Entry = Entry->Blink)

////////////////////////////////////////////////////////////////////////////////
// End of NT/Windows decls.
////////////////////////////////////////////////////////////////////////////////

//
// The following routines and typedefs are not part of NT/Windows, but make
// the most sense to colocate with the Rtl component (from a "this is a general
// runtime function" perspective).
//

typedef struct _RTL RTL;
typedef RTL *PRTL;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE RTL_GENERATE_RANDOM_BYTES)(
    _In_ PRTL Rtl,
    _In_ ULONG SizeOfBufferInBytes,
    _Inout_updates_bytes_(SizeOfBufferInBytes) PBYTE Buffer
    );
typedef RTL_GENERATE_RANDOM_BYTES *PRTL_GENERATE_RANDOM_BYTES;

typedef
_Success_(return >= 0)
_Check_return_opt_
HRESULT
(STDAPICALLTYPE RTL_PRINT_SYS_ERROR)(
    _In_ PRTL Rtl,
    _In_ PCSZ FunctionName,
    _In_ PCSZ FileName,
    _In_ ULONG LineNumber
    );
typedef RTL_PRINT_SYS_ERROR *PRTL_PRINT_SYS_ERROR;

#define RTL_SYS_ERROR(Name) \
    Rtl->Vtbl->PrintSysError(Rtl, #Name, __FILE__, __LINE__)

//
// SEH macro glue.
//

//
// N.B. The TRY_* variants facilitate quick source code grepping for certain
//      SEH operations.
//

#define TRY __try
#define TRY_TSX __try
#define TRY_AVX __try
#define TRY_AVX2 __try
#define TRY_AVX512 __try
#define TRY_AVX_ALIGNED __try
#define TRY_AVX_UNALIGNED __try

#define TRY_SSE42 __try
#define TRY_SSE42_ALIGNED __try
#define TRY_SSE42_UNALIGNED __try

#define TRY_PROBE_MEMORY __try
#define TRY_MAPPED_MEMORY_OP __try

#define EXCEPT __except
#define EXCEPT_FILTER(Name) __except( \
    Name(GetExceptionCode(),          \
         GetExceptionInformation())   \
    )

#define CATCH_EXCEPTION_ILLEGAL_INSTRUCTION __except(     \
    GetExceptionCode() == EXCEPTION_ILLEGAL_INSTRUCTION ? \
        EXCEPTION_EXECUTE_HANDLER :                       \
        EXCEPTION_CONTINUE_SEARCH                         \
    )

#define CATCH_EXCEPTION_ACCESS_VIOLATION __except(     \
    GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ? \
        EXCEPTION_EXECUTE_HANDLER :                    \
        EXCEPTION_CONTINUE_SEARCH                      \
    )

#define CATCH_STATUS_IN_PAGE_ERROR __except(     \
    GetExceptionCode() == STATUS_IN_PAGE_ERROR ? \
        EXCEPTION_EXECUTE_HANDLER :              \
        EXCEPTION_CONTINUE_SEARCH                \
    )

#define CATCH_STATUS_IN_PAGE_ERROR_OR_ACCESS_VIOLATION __except( \
    GetExceptionCode() == STATUS_IN_PAGE_ERROR ||                \
    GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ?           \
        EXCEPTION_EXECUTE_HANDLER :                              \
        EXCEPTION_CONTINUE_SEARCH                                \
    )

#if 0
FORCEINLINE
ULONG_PTR
GetAddressAlignment(_In_ ULONG_PTR Address)
{
    ULONG_PTR One = 1;
    ULONG_PTR Integer = Address;
    ULONG_PTR NumTrailingZeros = TrailingZerosPointer(Integer);
    return (One << NumTrailingZeros);
}

FORCEINLINE
BOOLEAN
PointerToOffsetCrossesPageBoundary(
    _In_ ULONG_PTR Pointer,
    _In_ LONG_PTR Offset
    )
{
    LONG_PTR ThisPage;
    LONG_PTR NextPage;

    ThisPage = ALIGN_DOWN(Pointer,        PAGE_SIZE);
    NextPage = ALIGN_DOWN(Pointer+Offset, PAGE_SIZE);

    return (ThisPage != NextPage);
}

FORCEINLINE
BOOLEAN
IsSamePage(
    _In_ PVOID Left,
    _In_ PVOID Right
    )
{
    LONG_PTR LeftPage;
    LONG_PTR RightPage;

    LeftPage = ALIGN_DOWN(Left, PAGE_SIZE);
    RightPage = ALIGN_DOWN(Right, PAGE_SIZE);
    return (LeftPage == RightPage);
}

FORCEINLINE
VOID
AssertAligned(
    _In_ ULONG_PTR Address,
    _In_ USHORT Alignment
    )
{
    ULONG_PTR CurrentAlignment;
    ULONG_PTR ExpectedAlignment;

    CurrentAlignment = GetAddressAlignment(Address);
    ExpectedAlignment = ALIGN_UP(CurrentAlignment, Alignment);

    ASSERT(CurrentAlignment == ExpectedAlignment);
}

#define AssertAligned8(Address)      AssertAligned((ULONG_PTR)Address, 8)
#define AssertAligned16(Address)     AssertAligned((ULONG_PTR)Address, 16)
#define AssertAligned32(Address)     AssertAligned((ULONG_PTR)Address, 32)
#define AssertAligned64(Address)     AssertAligned((ULONG_PTR)Address, 64)
#define AssertAligned512(Address)    AssertAligned((ULONG_PTR)Address, 512)
#define AssertAligned1024(Address)   AssertAligned((ULONG_PTR)Address, 1024)
#define AssertAligned2048(Address)   AssertAligned((ULONG_PTR)Address, 2048)
#define AssertAligned4096(Address)   AssertAligned((ULONG_PTR)Address, 4096)
#define AssertAligned8192(Address)   AssertAligned((ULONG_PTR)Address, 8192)

#define AssertPageAligned(Address)   AssertAligned4096(Address)

#define AssertSystemAligned(Address) ( \
    AssertAligned(                     \
        (ULONG_PTR)Address,            \
        MEMORY_ALLOCATION_ALIGNMENT    \
    )                                  \
)

#define AssertPointerAligned(Address) ( \
    AssertAligned(                      \
        (ULONG_PTR)Address,             \
        sizeof(ULONG_PTR)               \
    )                                   \
)

FORCEINLINE
_Success_(return != 0)
BOOLEAN
IsAligned(
    _In_ ULONG_PTR Address,
    _In_ USHORT Alignment
    )
{
    ULONG_PTR CurrentAlignment = GetAddressAlignment(Address);
    ULONG_PTR ExpectedAlignment = ALIGN_UP(CurrentAlignment, Alignment);

    return (CurrentAlignment == ExpectedAlignment);
}

#define IsAligned8(Address)      IsAligned((ULONG_PTR)Address, 8)
#define IsAligned16(Address)     IsAligned((ULONG_PTR)Address, 16)
#define IsAligned32(Address)     IsAligned((ULONG_PTR)Address, 32)
#define IsAligned64(Address)     IsAligned((ULONG_PTR)Address, 64)
#define IsAligned512(Address)    IsAligned((ULONG_PTR)Address, 512)
#define IsAligned1024(Address)   IsAligned((ULONG_PTR)Address, 1024)
#define IsAligned2048(Address)   IsAligned((ULONG_PTR)Address, 2048)
#define IsAligned4096(Address)   IsAligned((ULONG_PTR)Address, 4096)
#define IsAligned8192(Address)   IsAligned((ULONG_PTR)Address, 8192)

#define IsPageAligned(Address)   IsAligned4096(Address)
#define IsSystemAligned(Address) IsAligned((ULONG_PTR)Address,          \
                                           MEMORY_ALLOCATION_ALIGNMENT)

#define PrefaultPage(Address) (*((volatile char *)(PCHAR)(Address)))

#define PrefaultNextPage(Address)                              \
    (*(volatile char *)(PCHAR)((ULONG_PTR)Address + PAGE_SIZE))

#endif

//
// Helper enum and function pointer for determining an appropriate C type for
// a given power-of-2 value.
//

typedef enum _TYPE {        // Number of bytes
    ByteType = 0,           //  1
    ShortType = 1,          //  2
    LongType = 2,           //  4
    LongLongType = 3,       //  8
    XmmType = 4,            //  16
    YmmType = 5,            //  32
    ZmmType = 6,            //  64
} TYPE;
typedef TYPE *PTYPE;

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI GET_CONTAINING_TYPE)(
    _In_ PRTL Rtl,
    _In_ ULONG_PTR Value,
    _Out_ PTYPE Type
    );
typedef GET_CONTAINING_TYPE *PGET_CONTAINING_TYPE;

#ifndef __INTELLISENSE__
extern GET_CONTAINING_TYPE GetContainingType;
#endif

//
// String validation helpers.
//

FORCEINLINE
BOOLEAN
IsValidNullTerminatedUnicodeStringWithMinimumLengthInChars(
    _In_ PCUNICODE_STRING String,
    _In_ USHORT MinimumLengthInChars
    )
{
    //
    // Add 1 to account for the NULL.
    //

    USHORT Length = (MinimumLengthInChars + 1) * sizeof(WCHAR);
    USHORT MaximumLength = Length + sizeof(WCHAR);

    return (
        String != NULL &&
        String->Buffer != NULL &&
        String->Length >= Length &&
        String->MaximumLength >= MaximumLength &&
        sizeof(WCHAR) == (String->MaximumLength - String->Length) &&
        String->Buffer[String->Length / sizeof(WCHAR)] == L'\0'
    );
}

FORCEINLINE
BOOLEAN
IsValidNullTerminatedStringWithMinimumLengthInChars(
    _In_ PCSTRING String,
    _In_ USHORT MinimumLengthInChars
    )
{
    //
    // Add 1 to account for the NULL.
    //

    USHORT Length = (MinimumLengthInChars + 1) * sizeof(CHAR);
    USHORT MaximumLength = Length + sizeof(CHAR);

    return (
        String != NULL &&
        String->Buffer != NULL &&
        String->Length >= Length &&
        String->MaximumLength >= MaximumLength &&
        sizeof(CHAR) == (String->MaximumLength - String->Length) &&
        RTL_LAST_CHAR(String) == '\0'
    );
}

FORCEINLINE
BOOLEAN
IsValidUnicodeStringWithMinimumLengthInChars(
    _In_ PCUNICODE_STRING String,
    _In_ USHORT MinimumLengthInChars
    )
{
    USHORT Length = MinimumLengthInChars * sizeof(WCHAR);

    return (
        String != NULL &&
        String->Buffer != NULL &&
        String->Length >= Length &&
        String->MaximumLength >= Length &&
        String->MaximumLength >= String->Length
    );
}

FORCEINLINE
BOOLEAN
IsValidStringWithMinimumLengthInChars(
    _In_ PCSTRING String,
    _In_ USHORT MinimumLengthInChars
    )
{
    USHORT Length = MinimumLengthInChars * sizeof(CHAR);

    return (
        String != NULL &&
        String->Buffer != NULL &&
        String->Length >= Length &&
        String->MaximumLength >= Length &&
        String->MaximumLength >= String->Length
    );
}

FORCEINLINE
BOOLEAN
IsValidMinimumDirectoryUnicodeString(
    _In_ PCUNICODE_STRING String
    )
{
    return IsValidUnicodeStringWithMinimumLengthInChars(
        String,
        4
    );
}

FORCEINLINE
BOOLEAN
IsValidNullTerminatedUnicodeString(
    _In_ PCUNICODE_STRING String
    )
{
    return IsValidNullTerminatedUnicodeStringWithMinimumLengthInChars(
        String,
        1
    );
}

FORCEINLINE
BOOLEAN
IsValidNullTerminatedString(
    _In_ PCSTRING String
    )
{
    return IsValidNullTerminatedStringWithMinimumLengthInChars(String, 1);
}

FORCEINLINE
BOOLEAN
IsValidMinimumDirectoryNullTerminatedUnicodeString(
    _In_ PCUNICODE_STRING String
    )
{
    //
    // Minimum length: "C:\a" -> 4.
    //

    return IsValidNullTerminatedUnicodeStringWithMinimumLengthInChars(
        String,
        4
    );
}

FORCEINLINE
BOOLEAN
IsValidUnicodeString(
    _In_ PCUNICODE_STRING String
    )
{
    return (
        String != NULL &&
        String->Buffer != NULL &&
        String->Length >= 1 &&
        String->MaximumLength >= String->Length
    );
}

FORCEINLINE
BOOLEAN
IsEmptyUnicodeString(
    _In_ PCUNICODE_STRING String
    )
{
    return (
        String != NULL &&
        String->Length == 0
    );
}

FORCEINLINE
BOOLEAN
IsValidOrEmptyUnicodeString(
    _In_ PCUNICODE_STRING String
    )
{
    return IsEmptyUnicodeString(String) || (
        String != NULL &&
        String->Buffer != NULL &&
        String->Length >= sizeof(*String->Buffer) &&
        String->MaximumLength >= String->Length
    );
}

FORCEINLINE
BOOLEAN
IsValidString(
    _In_ PCSTRING String
    )
{
    return (
        String != NULL &&
        String->Buffer != NULL &&
        String->Length >= 1 &&
        String->MaximumLength >= String->Length
    );
}

FORCEINLINE
BOOLEAN
IsEmptyString(
    _In_ PCSTRING String
    )
{
    return (
        String != NULL &&
        String->Buffer == NULL &&
        String->Length == 0 &&
        String->MaximumLength == 0
    );
}

FORCEINLINE
BOOLEAN
IsValidOrEmptyString(
    _In_ PCSTRING String
    )
{
    return IsEmptyString(String) || (
        String != NULL &&
        String->Buffer != NULL &&
        String->Length >= 1 &&
        String->MaximumLength >= String->Length
    );
}

FORCEINLINE
BOOLEAN
FindCharInUnicodeString(
    _In_ PCUNICODE_STRING String,
    _In_ WCHAR Char,
    _In_ _Field_range_(<=, String->Length / sizeof(WCHAR))
        USHORT StartAtCharOffset,
    _Out_opt_ PUSHORT FoundAtCharOffset
    )
{
    USHORT Index;
    USHORT Count;
    PWSTR Wide;
    BOOLEAN Found = FALSE;
    USHORT Start = StartAtCharOffset;

    Count = String->Length / sizeof(WCHAR);

    if (!StartAtCharOffset || StartAtCharOffset > Count) {
        Start = 0;
    }

    Wide = String->Buffer;
    for (Index = 0; Index < Count; Index++, Wide++) {
        if (*Wide == Char) {
            Found = TRUE;
            break;
        }
    }

    if (!Found) {
        Index = 0;
    }

    if (ARGUMENT_PRESENT(FoundAtCharOffset)) {
        *FoundAtCharOffset = Index;
    }

    return Found;
}

FORCEINLINE
BOOLEAN
VerifyNoSlashInUnicodeString(
    _In_ PCUNICODE_STRING String
    )
{
    return !FindCharInUnicodeString(String, PATHSEP, 0, NULL);
}

//
// Buffer-related functions.
//

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE RTL_CREATE_BUFFER)(
    _In_ PRTL Rtl,
    _Inout_opt_ PHANDLE TargetProcessHandle,
    _In_ ULONG NumberOfPages,
    _In_opt_ PULONG AdditionalProtectionFlags,
    _Out_ PULONGLONG UsableBufferSizeInBytes,
    _Out_ PVOID *BufferAddress
    );
typedef RTL_CREATE_BUFFER *PRTL_CREATE_BUFFER;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE RTL_CREATE_MULTIPLE_BUFFERS)(
    _In_ PRTL Rtl,
    _Inout_opt_ PHANDLE TargetProcessHandle,
    _In_ ULONG PageSize,
    _In_ ULONG NumberOfBuffers,
    _In_ ULONG NumberOfPagesPerBuffer,
    _In_opt_ PULONG AdditionalProtectionFlags,
    _In_opt_ PULONG AdditionalAllocationTypeFlags,
    _Out_ PULONGLONG UsableBufferSizeInBytesPerBuffer,
    _Out_ PULONGLONG TotalBufferSizeInBytes,
    _Out_ PVOID *BufferAddress
    );
typedef RTL_CREATE_MULTIPLE_BUFFERS *PRTL_CREATE_MULTIPLE_BUFFERS;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE RTL_DESTROY_BUFFER)(
    _In_ PRTL Rtl,
    _In_ HANDLE ProcessHandle,
    _Inout_ PVOID *BufferAddress,
    _In_ ULONGLONG Size
    );
typedef RTL_DESTROY_BUFFER *PRTL_DESTROY_BUFFER;

//
// Page filling and copying functions.
//

typedef
HRESULT
(RTL_COPY_PAGES)(
    _In_ PRTL Rtl,
    _Out_writes_bytes_all_(NumberOfPages * 4096) PCHAR Dest,
    _In_reads_bytes_(NumberOfPages * 4096) const PCHAR Source,
    _In_ ULONG NumberOfPages
    );
typedef RTL_COPY_PAGES *PRTL_COPY_PAGES;

typedef
HRESULT
(RTL_FILL_PAGES)(
    _In_ PRTL Rtl,
    _Out_writes_bytes_all_(NumberOfPages * 4096) PCHAR Dest,
    _In_ BYTE Byte,
    _In_ ULONG NumberOfPages
    );
typedef RTL_FILL_PAGES *PRTL_FILL_PAGES;

//
// Privilege-related functions.
//

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE SET_PRIVILEGE)(
    _In_ PRTL Rtl,
    _In_ PWSTR PrivilegeName,
    _In_ BOOLEAN Enable
    );
typedef SET_PRIVILEGE *PSET_PRIVILEGE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE ENABLE_PRIVILEGE)(
    _In_ PRTL Rtl,
    _In_ PWSTR PrivilegeName
    );
typedef ENABLE_PRIVILEGE *PENABLE_PRIVILEGE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE DISABLE_PRIVILEGE)(
    _In_ PRTL Rtl,
    _In_ PWSTR PrivilegeName
    );
typedef DISABLE_PRIVILEGE *PDISABLE_PRIVILEGE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE ENABLE_MANAGE_VOLUME_PRIVILEGE)(
    _In_ PRTL Rtl
    );
typedef ENABLE_MANAGE_VOLUME_PRIVILEGE *PENABLE_MANAGE_VOLUME_PRIVILEGE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE DISABLE_MANAGE_VOLUME_PRIVILEGE)(
    _In_ PRTL Rtl
    );
typedef DISABLE_MANAGE_VOLUME_PRIVILEGE *PDISABLE_MANAGE_VOLUME_PRIVILEGE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE ENABLE_LOCK_MEMORY_PRIVILEGE)(
    _In_ PRTL Rtl
    );
typedef ENABLE_LOCK_MEMORY_PRIVILEGE *PENABLE_LOCK_MEMORY_PRIVILEGE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE DISABLE_LOCK_MEMORY_PRIVILEGE)(
    _In_ PRTL Rtl
    );
typedef DISABLE_LOCK_MEMORY_PRIVILEGE *PDISABLE_LOCK_MEMORY_PRIVILEGE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE ENABLE_DEBUG_PRIVILEGE)(
    _In_ PRTL Rtl
    );
typedef ENABLE_DEBUG_PRIVILEGE *PENABLE_DEBUG_PRIVILEGE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE DISABLE_DEBUG_PRIVILEGE)(
    _In_ PRTL Rtl
    );
typedef DISABLE_DEBUG_PRIVILEGE *PDISABLE_DEBUG_PRIVILEGE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE ENABLE_SYSTEM_PROFILE_PRIVILEGE)(
    _In_ PRTL Rtl
    );
typedef ENABLE_SYSTEM_PROFILE_PRIVILEGE *PENABLE_SYSTEM_PROFILE_PRIVILEGE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE DISABLE_SYSTEM_PROFILE_PRIVILEGE)(
    _In_ PRTL Rtl
    );
typedef DISABLE_SYSTEM_PROFILE_PRIVILEGE *PDISABLE_SYSTEM_PROFILE_PRIVILEGE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE ENABLE_PROFILE_SINGLE_PROCESS_PRIVILEGE)(
    _In_ PRTL Rtl
    );
typedef ENABLE_PROFILE_SINGLE_PROCESS_PRIVILEGE
      *PENABLE_PROFILE_SINGLE_PROCESS_PRIVILEGE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE DISABLE_PROFILE_SINGLE_PROCESS_PRIVILEGE)(
    _In_ PRTL Rtl
    );
typedef DISABLE_PROFILE_SINGLE_PROCESS_PRIVILEGE
      *PDISABLE_PROFILE_SINGLE_PROCESS_PRIVILEGE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE ENABLE_INCREASE_WORKING_SET_PRIVILEGE)(
    _In_ PRTL Rtl
    );
typedef ENABLE_INCREASE_WORKING_SET_PRIVILEGE
      *PENABLE_INCREASE_WORKING_SET_PRIVILEGE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE DISABLE_INCREASE_WORKING_SET_PRIVILEGE)(
    _In_ PRTL Rtl
    );
typedef DISABLE_INCREASE_WORKING_SET_PRIVILEGE
      *PDISABLE_INCREASE_WORKING_SET_PRIVILEGE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE ENABLE_CREATE_SYMBOLIC_LINK_PRIVILEGE)(
    _In_ PRTL Rtl
    );
typedef ENABLE_CREATE_SYMBOLIC_LINK_PRIVILEGE
      *PENABLE_CREATE_SYMBOLIC_LINK_PRIVILEGE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE DISABLE_CREATE_SYMBOLIC_LINK_PRIVILEGE)(
    _In_ PRTL Rtl
    );
typedef DISABLE_CREATE_SYMBOLIC_LINK_PRIVILEGE
      *PDISABLE_CREATE_SYMBOLIC_LINK_PRIVILEGE;

//
// Large page-specific routines.
//

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE RTL_INITIALIZE_LARGE_PAGES)(
    _In_ PRTL Rtl
    );
typedef RTL_INITIALIZE_LARGE_PAGES *PRTL_INITIALIZE_LARGE_PAGES;

typedef
_Must_inspect_result_
_Ret_maybenull_
_Post_writable_byte_size_(dwSize)
LPVOID
(STDAPICALLTYPE RTL_TRY_LARGE_PAGE_VIRTUAL_ALLOC)(
    _In_      PRTL   Rtl,
    _In_opt_  LPVOID lpAddress,
    _In_      SIZE_T dwSize,
    _In_      DWORD  flAllocationType,
    _In_      DWORD  flProtect,
    _Inout_   PBOOLEAN LargePages
    );
typedef RTL_TRY_LARGE_PAGE_VIRTUAL_ALLOC
      *PRTL_TRY_LARGE_PAGE_VIRTUAL_ALLOC;

typedef
_Must_inspect_result_
_Ret_maybenull_
_Post_writable_byte_size_(dwSize)
LPVOID
(STDAPICALLTYPE RTL_TRY_LARGE_PAGE_VIRTUAL_ALLOC_EX)(
    _In_      PRTL   Rtl,
    _In_      HANDLE hProcess,
    _In_opt_  LPVOID lpAddress,
    _In_      SIZE_T dwSize,
    _In_      DWORD  flAllocationType,
    _In_      DWORD  flProtect,
    _Inout_   PBOOLEAN LargePages
    );
typedef RTL_TRY_LARGE_PAGE_VIRTUAL_ALLOC_EX
      *PRTL_TRY_LARGE_PAGE_VIRTUAL_ALLOC_EX;

typedef
_Ret_maybenull_
HANDLE
(STDAPICALLTYPE RTL_TRY_LARGE_PAGE_CREATE_FILE_MAPPING_W)(
    _In_ PRTL Rtl,
    _In_ HANDLE hFile,
    _In_opt_ LPSECURITY_ATTRIBUTES lpFileMappingAttributes,
    _In_ DWORD flProtect,
    _In_ DWORD dwMaximumSizeHigh,
    _In_ DWORD dwMaximumSizeLow,
    _In_opt_ LPCWSTR lpName,
    _Inout_ PBOOLEAN LargePages
    );
typedef RTL_TRY_LARGE_PAGE_CREATE_FILE_MAPPING_W
      *PRTL_TRY_LARGE_PAGE_CREATE_FILE_MAPPING_W;

//
// Random object names.
//

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE RTL_CREATE_RANDOM_OBJECT_NAMES)(
    _In_ PRTL Rtl,
    _In_ PALLOCATOR TemporaryAllocator,
    _In_ PALLOCATOR WideBufferAllocator,
    _In_ USHORT NumberOfNames,
    _In_ USHORT LengthOfNameInChars,
    _In_opt_ PUNICODE_STRING NamespacePrefix,
    _Inout_ PPUNICODE_STRING NamesArrayPointer,
    _In_opt_ PPUNICODE_STRING PrefixArrayPointer,
    _Inout_ PULONG SizeOfWideBufferInBytes,
    _Out_writes_bytes_all_(*SizeOfWideBufferInBytes) PWSTR *WideBufferPointer
    );
typedef RTL_CREATE_RANDOM_OBJECT_NAMES *PRTL_CREATE_RANDOM_OBJECT_NAMES;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE RTL_CREATE_SINGLE_RANDOM_OBJECT_NAME)(
    _In_ PRTL Rtl,
    _In_ PALLOCATOR TemporaryAllocator,
    _In_ PALLOCATOR WideBufferAllocator,
    _In_opt_ PCUNICODE_STRING Prefix,
    _Inout_ PUNICODE_STRING Name
    );
typedef RTL_CREATE_SINGLE_RANDOM_OBJECT_NAME
      *PRTL_CREATE_SINGLE_RANDOM_OBJECT_NAME;

//
// UUID String Helpers.
//

#define UUID_STRING_LENGTH 36

FORCEINLINE
BOOLEAN
IsValidUuidString(
    _In_ PCSTRING String
    )
{
    return (
        String &&
        String->Buffer != NULL &&
        String->Length == UUID_STRING_LENGTH &&
        String->MaximumLength >= UUID_STRING_LENGTH+1
    );
}

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI RTL_CREATE_UUID_STRING)(
    _In_ PRTL Rtl,
    _In_ PSTRING String
    );
typedef RTL_CREATE_UUID_STRING *PRTL_CREATE_UUID_STRING;
extern RTL_CREATE_UUID_STRING RtlCreateUuidString;

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI RTL_FREE_UUID_STRING)(
    _In_ PRTL Rtl,
    _In_ PSTRING String
    );
typedef RTL_FREE_UUID_STRING *PRTL_FREE_UUID_STRING;
extern RTL_FREE_UUID_STRING RtlFreeUuidString;

//
// COM-specific glue for the Rtl component.
//

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE RTL_QUERY_INTERFACE)(
    _In_ PRTL Rtl,
    _In_ REFIID InterfaceId,
    _COM_Outptr_ PVOID *Interface
    );
typedef RTL_QUERY_INTERFACE *PRTL_QUERY_INTERFACE;

typedef
ULONG
(STDAPICALLTYPE RTL_ADD_REF)(
    _In_ PRTL Rtl
    );
typedef RTL_ADD_REF *PRTL_ADD_REF;

typedef
ULONG
(STDAPICALLTYPE RTL_RELEASE)(
    _In_ PRTL Rtl
    );
typedef RTL_RELEASE *PRTL_RELEASE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE RTL_CREATE_INSTANCE)(
    _In_ PRTL Rtl,
    _In_opt_ PIUNKNOWN UnknownOuter,
    _In_ REFIID InterfaceId,
    _COM_Outptr_ PVOID *Instance
    );
typedef RTL_CREATE_INSTANCE *PRTL_CREATE_INSTANCE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE RTL_LOCK_SERVER)(
    _In_ PRTL Rtl,
    _In_ BOOL Lock
    );
typedef RTL_LOCK_SERVER *PRTL_LOCK_SERVER;

typedef struct _RTL_VTBL {
    PRTL_QUERY_INTERFACE QueryInterface;
    PRTL_ADD_REF AddRef;
    PRTL_RELEASE Release;
    PRTL_CREATE_INSTANCE CreateInstance;
    PRTL_LOCK_SERVER LockServer;
    PRTL_GENERATE_RANDOM_BYTES GenerateRandomBytes;
    PRTL_PRINT_SYS_ERROR PrintSysError;
    PRTL_CREATE_BUFFER CreateBuffer;
    PRTL_CREATE_MULTIPLE_BUFFERS CreateMultipleBuffers;
    PRTL_DESTROY_BUFFER DestroyBuffer;
    PRTL_COPY_PAGES CopyPages;
    PRTL_FILL_PAGES FillPages;
    PRTL_CREATE_RANDOM_OBJECT_NAMES CreateRandomObjectNames;
    PRTL_CREATE_SINGLE_RANDOM_OBJECT_NAME CreateSingleRandomObjectName;
    PRTL_TRY_LARGE_PAGE_VIRTUAL_ALLOC TryLargePageVirtualAlloc;
    PRTL_TRY_LARGE_PAGE_VIRTUAL_ALLOC_EX TryLargePageVirtualAllocEx;
    PRTL_TRY_LARGE_PAGE_CREATE_FILE_MAPPING_W TryLargePageCreateFileMappingW;
} RTL_VTBL;
typedef RTL_VTBL *PRTL_VTBL;

//
// Define the RTL_FUNCTION_TABLE X-macro.  Each macro receives (Upper, Name)
// as its parameters, where Upper represents the pointer type name (excluding
// the leading 'P'), and name is the capitalized name of the function, e.g.:
//
//      (RTL_INITIALIZE_BITMAP, RtlInitializeBitmap)
//

#define RTL_FUNCTION_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
                                                           \
    FIRST_ENTRY(                                           \
        RTL_INITIALIZE_BITMAP,                             \
        RtlInitializeBitMap                                \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_CLEAR_BIT,                                     \
        RtlClearBit                                        \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_SET_BIT,                                       \
        RtlSetBit                                          \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_TEST_BIT,                                      \
        RtlTestBit                                         \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_CLEAR_ALL_BITS,                                \
        RtlClearAllBits                                    \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_SET_ALL_BITS,                                  \
        RtlSetAllBits                                      \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_FIND_CLEAR_BITS,                               \
        RtlFindClearBits                                   \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_FIND_SET_BITS,                                 \
        RtlFindSetBits                                     \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_FIND_CLEAR_BITS_AND_SET,                       \
        RtlFindClearBitsAndSet                             \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_FIND_SET_BITS_AND_CLEAR,                       \
        RtlFindSetBitsAndClear                             \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_CLEAR_BITS,                                    \
        RtlClearBits                                       \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_SET_BITS,                                      \
        RtlSetBits                                         \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_FIND_CLEAR_RUNS,                               \
        RtlFindClearRuns                                   \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_FIND_LONGEST_RUN_CLEAR,                        \
        RtlFindLongestRunClear                             \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_NUMBER_OF_CLEAR_BITS,                          \
        RtlNumberOfClearBits                               \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_NUMBER_OF_SET_BITS,                            \
        RtlNumberOfSetBits                                 \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_ARE_BITS_CLEAR,                                \
        RtlAreBitsClear                                    \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_ARE_BITS_SET,                                  \
        RtlAreBitsSet                                      \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_FIND_NEXT_FORWARD_RUN_CLEAR,                   \
        RtlFindNextForwardRunClear                         \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_FIND_LAST_BACKWARD_RUN_CLEAR,                  \
        RtlFindLastBackwardRunClear                        \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_COPY_MEMORY,                                   \
        RtlCopyMemory                                      \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_MOVE_MEMORY,                                   \
        RtlMoveMemory                                      \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_ZERO_MEMORY,                                   \
        RtlZeroMemory                                      \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_FILL_MEMORY,                                   \
        RtlFillMemory                                      \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_COMPARE_MEMORY,                                \
        RtlCompareMemory                                   \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_FIRST_ENTRY_SLIST,                             \
        RtlFirstEntrySList                                 \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_CHAR_TO_INTEGER,                               \
        RtlCharToInteger                                   \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_INTEGER_TO_CHAR,                               \
        RtlIntegerToChar                                   \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_UNICODE_STRING_TO_INTEGER,                     \
        RtlUnicodeStringToInteger                          \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_UNICODE_STRING_TO_INT64,                       \
        RtlUnicodeStringToInt64                            \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_EQUAL_UNICODE_STRING,                          \
        RtlEqualUnicodeString                              \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_PREFIX_UNICODE_STRING,                         \
        RtlPrefixUnicodeString                             \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        RTL_APPEND_UNICODE_STRING_TO_STRING,               \
        RtlAppendUnicodeStringToString                     \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        CRYPT_BINARY_TO_STRING_A,                          \
        CryptBinaryToStringA                               \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        CRYPT_BINARY_TO_STRING_W,                          \
        CryptBinaryToStringW                               \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        BSEARCH,                                           \
        bsearch                                            \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        BSEARCH_S,                                         \
        bsearch_s                                          \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        QSORT,                                             \
        qsort                                              \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        QSORT_S,                                           \
        qsort_s                                            \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        LFIND,                                             \
        _lfind                                             \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        __C_SPECIFIC_HANDLER,                              \
        __C_specific_handler                               \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        _WSPLITPATH_S,                                     \
        _wsplitpath_s                                      \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        SPRINTF_S,                                         \
        sprintf_s                                          \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        SWPRINTF_S,                                        \
        swprintf_s                                         \
    )                                                      \
                                                           \
    ENTRY(                                                 \
        VSPRINTF_S,                                        \
        vsprintf_s                                         \
    )                                                      \
                                                           \
    LAST_ENTRY(                                            \
        VSWPRINTF_S,                                       \
        vswprintf_s                                        \
    )

#define RTL_FUNCTION_TABLE_ENTRY(ENTRY) \
    RTL_FUNCTION_TABLE(ENTRY, ENTRY, ENTRY)

#define EXPAND_AS_RTL_FUNCTION_STRUCT(Upper, Name) \
    P##Upper Name;

typedef struct _RTL_FUNCTIONS {
    RTL_FUNCTION_TABLE_ENTRY(EXPAND_AS_RTL_FUNCTION_STRUCT)
} RTL_FUNCTIONS;

DEFINE_UNUSED_STATE(RTL);

typedef union _RTL_FLAGS {
    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, indicates large pages are available.
        //

        ULONG IsLargePageEnabled:1;

        //
        // Unused bits.
        //

        ULONG Unused:31;

    };
    LONG AsLong;
    ULONG AsULong;
} RTL_FLAGS;
C_ASSERT(sizeof(RTL_FLAGS) == sizeof(ULONG));
typedef RTL_FLAGS *PRTL_FLAGS;

typedef struct _RTL {
    COMMON_COMPONENT_HEADER(RTL);

    HANDLE SysErrorOutputHandle;
    SRWLOCK SysErrorMessageBufferLock;

    _Guarded_by_(SysErrorMessageBufferLock)
    struct {
        PCHAR SysErrorMessageBuffer;
        SIZE_T SizeOfSysErrorMessageBufferInBytes;
    };

    DEBUGGER_CONTEXT DebuggerContext;

    ULONG Padding1;

    HCRYPTPROV CryptProv;

    SIZE_T LargePageMinimum;

    union {
        HMODULE Modules[4];
        struct {
            union {
                PVOID FirstModule;
                HMODULE NtdllModule;
            };

            HMODULE NtoskrnlModule;
            HMODULE Advapi32Module;

            union {
                PVOID LastModule;
                HMODULE Crypt32Module;
            };
        };
    };

    ULONG NumberOfModules;

    ULONG ProcInfoBufferSizeInBytes;

    //
    // Inline the Rtl functions for convenience.
    //

    union {

        struct {
            RTL_FUNCTION_TABLE_ENTRY(EXPAND_AS_RTL_FUNCTION_STRUCT)
        };

        RTL_FUNCTIONS RtlFunctions;
    };

    //
    // Inline bit manipulation functions for convenience.
    //

    union {

#define EXPAND_AS_FUNCTION_POINTER(Upper, Name, Unused3, Unused4) \
    P##Upper Name;
        struct {
            RTL_BIT_MANIPULATION_FUNCTION_TABLE_ENTRY(
                EXPAND_AS_FUNCTION_POINTER
            )
        };
#undef EXPAND_AS_FUNCTION_POINTER

        RTL_BIT_MANIPULATION_FUNCTIONS RtlBitManipulationFunctions;

    };

    //
    // CPU Features.
    //

    RTL_CPU_FEATURES CpuFeatures;

    RTL_VTBL Interface;
} RTL;
typedef RTL *PRTL;

#define AcquireRtlSysErrorMessageBufferLock(Rtl) \
    AcquireSRWLockExclusive(&Rtl->SysErrorMessageBufferLock)

#define ReleaseRtlSysErrorMessageBufferLock(Rtl) \
    ReleaseSRWLockExclusive(&Rtl->SysErrorMessageBufferLock)

#define _RTL_MODULE_NAMES_HEAD  \
    (PCZPCWSTR)L"ntdll.dll",    \
    (PCZPCWSTR)L"ntoskrnl.exe", \
    (PCZPCWSTR)L"advapi32.dll", \
    (PCZPCWSTR)L"crypt32.dll"

FORCEINLINE
BYTE
GetNumberOfRtlModules(
    _In_ PRTL Rtl
    )
{
    BYTE NumberOfModules;

    //
    // Calculate the number of module handles based on the first and last
    // module fields of the RTL structure.  The additional sizeof(HMODULE)
    // accounts for the fact that we're going from 0-based address offsets
    // to 1-based counts.
    //

    NumberOfModules = (BYTE)(

        (ULONG_PTR)(

            sizeof(HMODULE) +

            RtlOffsetFromPointer(
                &Rtl->LastModule,
                &Rtl->FirstModule
            )

        ) / (ULONG_PTR)sizeof(HMODULE)
    );

    ASSERT(NumberOfModules == ARRAYSIZE(Rtl->Modules));

    return NumberOfModules;
}

typedef
HRESULT
(NTAPI RTL_INITIALIZE)(
    _In_ PRTL Rtl
    );
typedef RTL_INITIALIZE *PRTL_INITIALIZE;

typedef
VOID
(NTAPI RTL_RUNDOWN)(
    _In_ _Post_ptr_invalid_ PRTL Rtl
    );
typedef RTL_RUNDOWN *PRTL_RUNDOWN;

typedef
HRESULT
(NTAPI RTL_INITIALIZE_CPU_FEATURES)(
    _In_ PRTL Rtl
    );
typedef RTL_INITIALIZE_CPU_FEATURES *PRTL_INITIALIZE_CPU_FEATURES;

typedef
HRESULT
(NTAPI RTL_INITIALIZE_BIT_MANIPULATION_FUNCTION_POINTERS)(
    _In_ PRTL Rtl
    );
typedef RTL_INITIALIZE_BIT_MANIPULATION_FUNCTION_POINTERS
      *PRTL_INITIALIZE_BIT_MANIPULATION_FUNCTION_POINTERS;

extern RTL_INITIALIZE RtlInitialize;
extern RTL_INITIALIZE_CPU_FEATURES RtlInitializeCpuFeatures;
extern RTL_INITIALIZE_BIT_MANIPULATION_FUNCTION_POINTERS
    RtlInitializeBitManipulationFunctionPointers;
extern RTL_RUNDOWN RtlRundown;
extern RTL_GENERATE_RANDOM_BYTES RtlGenerateRandomBytes;
extern RTL_PRINT_SYS_ERROR RtlPrintSysError;
extern RTL_CREATE_BUFFER RtlCreateBuffer;
extern RTL_CREATE_MULTIPLE_BUFFERS RtlCreateMultipleBuffers;
extern RTL_DESTROY_BUFFER RtlDestroyBuffer;
extern RTL_CREATE_RANDOM_OBJECT_NAMES RtlCreateRandomObjectNames;
extern RTL_CREATE_SINGLE_RANDOM_OBJECT_NAME RtlCreateSingleRandomObjectName;
extern RTL_INITIALIZE_LARGE_PAGES RtlInitializeLargePages;
extern RTL_TRY_LARGE_PAGE_VIRTUAL_ALLOC RtlTryLargePageVirtualAlloc;
extern RTL_TRY_LARGE_PAGE_VIRTUAL_ALLOC_EX RtlTryLargePageVirtualAllocEx;
extern RTL_TRY_LARGE_PAGE_CREATE_FILE_MAPPING_W
    RtlTryLargePageCreateFileMappingW;
extern RTL_COPY_PAGES RtlCopyPages;
extern RTL_FILL_PAGES RtlFillPages;

#if defined(_M_AMD64) || defined(_M_X64)
extern RTL_COPY_PAGES RtlCopyPagesNonTemporal_AVX2;
extern RTL_FILL_PAGES RtlFillPagesNonTemporal_AVX2;
extern RTL_COPY_PAGES RtlCopyPages_AVX2;
extern RTL_FILL_PAGES RtlFillPages_AVX2;
#endif

#ifdef PH_COMPAT
extern RTL_COPY_MEMORY RtlCopyMemory;
extern RTL_MOVE_MEMORY RtlMoveMemory;
extern RTL_COMPARE_MEMORY RtlCompareMemory;
extern RTL_NUMBER_OF_SET_BITS RtlNumberOfSetBits;
extern RTL_EQUAL_UNICODE_STRING RtlEqualUnicodeString;
extern RTL_FIND_LONGEST_RUN_CLEAR RtlFindLongestRunClear;
extern RTL_UNICODE_STRING_TO_INT64 RtlUnicodeStringToInt64;
extern RTL_UNICODE_STRING_TO_INTEGER RtlUnicodeStringToInteger;
extern RTL_APPEND_UNICODE_STRING_TO_STRING RtlAppendUnicodeStringToString;
#endif


//
// Compat glue.
//

#ifdef PH_WINDOWS
BOOL
CloseEvent(
    _In_ _Post_ptr_invalid_ HANDLE Object
    );

BOOL
CloseDirectory(
    _In_ _Post_ptr_invalid_ HANDLE Object
    );

BOOL
CloseFile(
    _In_ _Post_ptr_invalid_ HANDLE Object
    );
#endif

FORCEINLINE
VOID
ResetSRWLock(PSRWLOCK Lock)
{
#ifdef PH_WINDOWS
    Lock->Ptr = NULL;
#else
    ZeroStructPointerInline(Lock);
#endif
}

#ifdef PH_WINDOWS
#define TLS_KEY_TYPE ULONG
#else
#define TLS_KEY_TYPE pthread_key_t
#endif

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
