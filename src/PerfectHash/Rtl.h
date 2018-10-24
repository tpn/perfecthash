/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

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

#include <minwindef.h>

//
// NT typedefs.
//

#ifdef _WIN64
#define RTL_CONSTANT_STRING(s) { \
    sizeof(s) - sizeof((s)[0]),  \
    sizeof( s ),                 \
    0,                           \
    s                            \
}
#else
#define RTL_CONSTANT_STRING(s) { \
    sizeof(s) - sizeof((s)[0]),  \
    sizeof( s ),                 \
    s                            \
}
#endif

typedef union _ULONG_INTEGER {
    struct {
        USHORT  LowPart;
        USHORT  HighPart;
    };
    ULONG   LongPart;

} ULONG_INTEGER, *PULONG_INTEGER;

typedef union _LONG_INTEGER {
    struct {
        USHORT  LowPart;
        SHORT   HighPart;
    };
    LONG   LongPart;
} LONG_INTEGER, *PLONG_INTEGER;

typedef union _USHORT_INTEGER {
    struct {
        BYTE  LowPart;
        BYTE  HighPart;
    };
    USHORT   ShortPart;
} USHORT_INTEGER, *PUSHORT_INTEGER;

typedef union _SHORT_INTEGER {
    struct {
        CHAR  LowPart;
        CHAR  HighPart;
    };
    SHORT   ShortPart;
} SHORT_INTEGER, *PSHORT_INTEGER;

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

#ifdef _M_X64
typedef __m128i DECLSPEC_ALIGN(16) XMMWORD, *PXMMWORD, **PPXMMWORD;
typedef __m256i DECLSPEC_ALIGN(32) YMMWORD, *PYMMWORD, **PPYMMWORD;
typedef __m512i DECLSPEC_ALIGN(64) ZMMWORD, *PZMMWORD, **PPZMMWORD;
#endif

#ifndef PAGE_SIZE
#define PAGE_SIZE 4096
#endif

#ifndef PAGE_SHIFT
#define PAGE_SHIFT 12L
#endif

#ifndef PAGE_ALIGN
#define PAGE_ALIGN(Va) ((PVOID)((ULONG_PTR)(Va) & ~(PAGE_SIZE - 1)))
#endif

#ifndef ROUND_TO_PAGES
#define ROUND_TO_PAGES(Size) (                             \
    ((ULONG_PTR)(Size) + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1) \
)
#endif

#ifndef BYTES_TO_PAGES
#define BYTES_TO_PAGES(Size) (                                   \
    (((Size) >> PAGE_SHIFT) + (((Size) & (PAGE_SIZE - 1)) != 0)) \
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
#define ALIGN_UP_LARGE_PAGE(Address) (       \
    ALIGN_UP(Address, GetLargePageMinimum()) \
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

#ifdef _M_X64
#ifndef ALIGN_UP_XMMWORD
#define ALIGN_UP_XMMWORD(Address) (ALIGN_UP(Address, sizeof(XMMWORD)))
#endif

#ifndef ALIGN_UP_YMMWORD
#define ALIGN_UP_YMMWORD(Address) (ALIGN_UP(Address, sizeof(YMMWORD)))
#endif

#ifndef ALIGN_UP_ZMMWORD
#define ALIGN_UP_ZMMWORD(Address) (ALIGN_UP(Address, sizeof(ZMMWORD)))
#endif
#else _M_X64
#ifndef ALIGN_UP_XMMWORD
#define ALIGN_UP_XMMWORD(Address) (ALIGN_UP(Address, 16))
#endif

#ifndef ALIGN_UP_YMMWORD
#define ALIGN_UP_YMMWORD(Address) (ALIGN_UP(Address, 32))
#endif

#ifndef ALIGN_UP_ZMMWORD
#define ALIGN_UP_ZMMWORD(Address) (ALIGN_UP(Address, 64))
#endif
#endif

#ifndef ASSERT
#define ASSERT(Condition) \
    if (!(Condition)) {   \
        __debugbreak();   \
    }
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


#ifdef RtlCopyMemory
#undef RtlCopyMemory
#endif

#ifdef RtlZeroMemory
#undef RtlZeroMemory
#endif

#ifdef RtlFillMemory
#undef RtlFillMemory
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

#ifndef BitTestAndSet
#define BitTestAndSet _bittestandset
#endif

#define FastSetBit(Bitmap, BitNumber) (             \
    BitTestAndSet((PLONG)Bitmap->Buffer, BitNumber) \
)

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

////////////////////////////////////////////////////////////////////////////////
// Memory/String
////////////////////////////////////////////////////////////////////////////////

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
    _In_opt_ ULONG Base,
    _Out_ PULONG Value
    );
typedef RTL_CHAR_TO_INTEGER *PRTL_CHAR_TO_INTEGER;

typedef
NTSTATUS
(NTAPI RTL_INTEGER_TO_CHAR)(
    _In_ ULONG Value,
    _In_opt_ ULONG Base,
    _In_ ULONG SizeOfBuffer,
    _Out_writes_bytes_(SizeOfBuffer) PCHAR Buffer
    );
typedef RTL_INTEGER_TO_CHAR *PRTL_INTEGER_TO_CHAR;

typedef
NTSTATUS
(NTAPI RTL_UNICODE_STRING_TO_INTEGER)(
    _In_ PCUNICODE_STRING String,
    _In_opt_ ULONG Base,
    _Out_ PULONG Value
    );
typedef RTL_UNICODE_STRING_TO_INTEGER *PRTL_UNICODE_STRING_TO_INTEGER;

typedef
BOOLEAN
(NTAPI RTL_EQUAL_UNICODE_STRING)(
    _In_ PCUNICODE_STRING String1,
    _In_ PCUNICODE_STRING String2,
    _In_ BOOLEAN CaseInSensitive
    );
typedef RTL_EQUAL_UNICODE_STRING *PRTL_EQUAL_UNICODE_STRING;

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

#ifdef _M_X64
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

#ifdef _M_X64

#define ZeroStructInline(Name) \
    __stosq((PDWORD64)&Name, 0, (sizeof(Name) >> 3))

#define ZeroStructPointerInline(Name) \
    __stosq((PDWORD64)Name, 0, (sizeof(*Name) >> 3))

#else
#define ZeroStructInline(Name) ZeroInline(&Name, sizeof(Name))
#define ZeroStructPointer(Name) ZeroInline(Name, sizeof(*Name))
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
    _In_opt_ ULONG SizeOfBitMap
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
(NTAPI *PRTL_FIND_LONGEST_RUN_CLEAR)(
    _In_ PRTL_BITMAP BitMapHeader,
    _Out_ PULONG StartingIndex
    );

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
(NTAPI *PRTL_NUMBER_OF_SET_BITS)(
    _In_ PRTL_BITMAP BitMapHeader
    );

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
    _In_opt_ ULONG LineNumber
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

//
// Alignment and power-of-2 helpers.
//

FORCEINLINE
ULONG
TrailingZeros(
    _In_ ULONG Integer
    )
{
    return _tzcnt_u32(Integer);
}

FORCEINLINE
ULONG
LeadingZeros(
    _In_ ULONG Integer
    )
{
    return _lzcnt_u32(Integer);
}

#ifdef _WIN64
FORCEINLINE
ULONGLONG
TrailingZeros64(
    _In_ ULONGLONG Integer
    )
{
    return _tzcnt_u64(Integer);
}

FORCEINLINE
ULONGLONG
LeadingZeros64(
    _In_ ULONGLONG Integer
    )
{
    return _lzcnt_u64(Integer);
}
#endif

FORCEINLINE
ULONG_PTR
TrailingZerosPointer(
    _In_ ULONG_PTR Integer
    )
{
#ifdef _WIN64
    return _tzcnt_u64((ULONGLONG)Integer);
#else
    return _tzcnt_u32((ULONG)Integer);
#endif
}

FORCEINLINE
ULONGLONG
LeadingZerosPointer(
    _In_ ULONG_PTR Integer
    )
{
#ifdef _WIN64
    return _lzcnt_u64((ULONGLONG)Integer);
#else
    return _lzcnt_u32((ULONG)Integer);
#endif
}

FORCEINLINE
ULONG
PopulationCount32(
    _In_ ULONG Integer
    )
{
    return __popcnt(Integer);
}

#if 0
FORCEINLINE
ULONGLONG
PopulationCount64(
    _In_ ULONGLONG Integer
    )
{
    return __popcnt64(Integer);
}

FORCEINLINE
ULONG
PopulationCountPointer(
    _In_ ULONG_PTR Integer
    )
{
#ifdef _WIN64
    return PopulationCount64(Integer);
#else
    return PopulationCount(Integer);
#endif
}
#endif

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
    ULONG_PTR CurrentAlignment = GetAddressAlignment(Address);
    ULONG_PTR ExpectedAlignment = ALIGN_UP(CurrentAlignment, Alignment);

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

FORCEINLINE
BOOLEAN
IsPowerOf2(
    _In_ ULONGLONG Value
    )
{
    if (Value <= 1) {
        return FALSE;
    }

    return ((Value & (Value - 1)) == 0);
}

FORCEINLINE
ULONGLONG
RoundUpPowerOf2(
    _In_ ULONG Input
    )
{
    if (Input <= 1) {
        return 2;
    }

    if (IsPowerOf2(Input)) {
        return Input;
    }

    return 1ULL << (32 - LeadingZeros(Input - 1));
}

FORCEINLINE
ULONGLONG
RoundUpNextPowerOf2(
    _In_ ULONG Input
    )
{
    if (Input <= 1) {
        return 2;
    }

    if (IsPowerOf2(Input)) {
        Input += 1;
    }

    return 1ULL << (32 - LeadingZeros(Input - 1));
}

#define PrefaultPage(Address) (*((volatile char *)(PCHAR)(Address)))

#define PrefaultNextPage(Address)                              \
    (*(volatile char *)(PCHAR)((ULONG_PTR)Address + PAGE_SIZE))


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
        String->Buffer[String->Length >> 1] == L'\0'
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
        String->Buffer[String->Length >> 1] == '\0'
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
        String->Buffer == NULL &&
        String->Length == 0 &&
        String->MaximumLength == 0
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
        String->Length >= 1 &&
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
    _In_opt_ _Field_range_(<=, String->Length >> 1) USHORT StartAtCharOffset,
    _Out_opt_ PUSHORT FoundAtCharOffset
    )
{
    USHORT Index;
    USHORT Count;
    PWSTR Wide;
    BOOLEAN Found = FALSE;
    USHORT Start = StartAtCharOffset;

    Count = String->Length >> 1;

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
    return !FindCharInUnicodeString(String, L'\\', 0, NULL);
}

//
// Buffer-related functions.
//

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE RTL_CREATE_BUFFER)(
    _In_ PRTL Rtl,
    _In_opt_ PHANDLE TargetProcessHandle,
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
    _In_opt_ PHANDLE TargetProcessHandle,
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
    _Inout_ PVOID *BufferAddress
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
    _In_opt_ BYTE Byte,
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
_Check_return_
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
_Check_return_
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
_Check_return_
_Success_(return >= 0)
HRESULT
(NTAPI RTL_CREATE_UUID_STRING)(
    _In_ PRTL Rtl,
    _In_ PSTRING String
    );
typedef RTL_CREATE_UUID_STRING *PRTL_CREATE_UUID_STRING;
extern RTL_CREATE_UUID_STRING RtlCreateUuidString;

typedef
_Check_return_
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
    _In_opt_ BOOL Lock
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
// N.B. Keep the function names and function pointers in sync; they need to
//      be identical in order for LoadSymbols() to work.
//

#define _RTL_FUNCTION_NAMES_HEAD      \
    "RtlInitializeBitMap",            \
    "RtlClearBit",                    \
    "RtlSetBit",                      \
    "RtlTestBit",                     \
    "RtlClearAllBits",                \
    "RtlSetAllBits",                  \
    "RtlFindClearBits",               \
    "RtlFindSetBits",                 \
    "RtlFindClearBitsAndSet",         \
    "RtlFindSetBitsAndClear",         \
    "RtlClearBits",                   \
    "RtlSetBits",                     \
    "RtlFindClearRuns",               \
    "RtlFindLongestRunClear",         \
    "RtlNumberOfClearBits",           \
    "RtlNumberOfSetBits",             \
    "RtlAreBitsClear",                \
    "RtlAreBitsSet",                  \
    "RtlFindNextForwardRunClear",     \
    "RtlFindLastBackwardRunClear",    \
    "RtlCopyMemory",                  \
    "RtlMoveMemory",                  \
    "RtlFillMemory",                  \
    "RtlCompareMemory",               \
    "RtlFirstEntrySList",             \
    "RtlCharToInteger",               \
    "RtlIntegerToChar",               \
    "RtlUnicodeStringToInteger",      \
    "RtlEqualUnicodeString",          \
    "RtlAppendUnicodeStringToString", \
    "CryptBinaryToStringA",           \
    "CryptBinaryToStringW",           \
    "__C_specific_handler",           \
    "_wsplitpath_s",                  \
    "sprintf_s",                      \
    "swprintf_s",                     \
    "vsprintf_s",                     \
    "vswprintf_s"

#define _RTL_FUNCTIONS_HEAD                                              \
    PRTL_INITIALIZE_BITMAP RtlInitializeBitMap;                          \
    PRTL_CLEAR_BIT RtlClearBit;                                          \
    PRTL_SET_BIT RtlSetBit;                                              \
    PRTL_TEST_BIT RtlTestBit;                                            \
    PRTL_CLEAR_ALL_BITS RtlClearAllBits;                                 \
    PRTL_SET_ALL_BITS RtlSetAllBits;                                     \
    PRTL_FIND_CLEAR_BITS RtlFindClearBits;                               \
    PRTL_FIND_SET_BITS RtlFindSetBits;                                   \
    PRTL_FIND_CLEAR_BITS_AND_SET RtlFindClearBitsAndSet;                 \
    PRTL_FIND_SET_BITS_AND_CLEAR RtlFindSetBitsAndClear;                 \
    PRTL_CLEAR_BITS RtlClearBits;                                        \
    PRTL_SET_BITS RtlSetBits;                                            \
    PRTL_FIND_CLEAR_RUNS RtlFindClearRuns;                               \
    PRTL_FIND_LONGEST_RUN_CLEAR RtlFindLongestRunClear;                  \
    PRTL_NUMBER_OF_CLEAR_BITS RtlNumberOfClearBits;                      \
    PRTL_NUMBER_OF_SET_BITS RtlNumberOfSetBits;                          \
    PRTL_ARE_BITS_CLEAR RtlAreBitsClear;                                 \
    PRTL_ARE_BITS_SET RtlAreBitsSet;                                     \
    PRTL_FIND_NEXT_FORWARD_RUN_CLEAR RtlFindNextForwardRunClear;         \
    PRTL_FIND_LAST_BACKWARD_RUN_CLEAR RtlFindLastBackwardRunClear;       \
    PRTL_COPY_MEMORY RtlCopyMemory;                                      \
    PRTL_MOVE_MEMORY RtlMoveMemory;                                      \
    PRTL_FILL_MEMORY RtlFillMemory;                                      \
    PRTL_COMPARE_MEMORY RtlCompareMemory;                                \
    PRTL_FIRST_ENTRY_SLIST RtlFirstEntrySList;                           \
    PRTL_CHAR_TO_INTEGER RtlCharToInteger;                               \
    PRTL_INTEGER_TO_CHAR RtlIntegerToChar;                               \
    PRTL_UNICODE_STRING_TO_INTEGER RtlUnicodeStringToInteger;            \
    PRTL_EQUAL_UNICODE_STRING RtlEqualUnicodeString;                     \
    PRTL_APPEND_UNICODE_STRING_TO_STRING RtlAppendUnicodeStringToString; \
    PCRYPT_BINARY_TO_STRING_A CryptBinaryToStringA;                      \
    PCRYPT_BINARY_TO_STRING_W CryptBinaryToStringW;                      \
    P__C_SPECIFIC_HANDLER __C_specific_handler;                          \
    P_WSPLITPATH_S _wsplitpath_s;                                        \
    PSPRINTF_S sprintf_s;                                                \
    PSWPRINTF_S swprintf_s;                                              \
    PVSPRINTF_S vsprintf_s;                                              \
    PVSWPRINTF_S vswprintf_s

typedef struct _RTL_FUNCTIONS {
    _RTL_FUNCTIONS_HEAD;
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
    ULONG Padding2;

    union {
        struct {
            _RTL_FUNCTIONS_HEAD;
        };
        RTL_FUNCTIONS RtlFunctions;
    };

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

extern RTL_INITIALIZE RtlInitialize;
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
extern RTL_COPY_PAGES RtlCopyPagesNonTemporalAvx2_v1;
extern RTL_FILL_PAGES RtlFillPagesNonTemporalAvx2_v1;
#endif

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
