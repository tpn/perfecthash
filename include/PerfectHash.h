/*++

Copyright (c) 2018-2023 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHash.h

Abstract:

    This is the main public header file for the PerfectHash component.
    It defines structures and functions related to creating perfect hash
    tables, contexts, loading keys, testing and benchmarking.

--*/

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <PerfectHashErrors.h>

#ifdef PH_WINDOWS

//
// N.B. The warning disable glue is necessary to get the system headers to
//      include with all errors enabled (/Wall).
//

//
// Disable the "function selected for inlining" and "function not inlined"
// warnings.
//

#pragma warning(disable: 4710 4711)

//
// 4255:
//      winuser.h(6502): warning C4255: 'EnableMouseInPointerForThread':
//          no function prototype given: converting '()' to '(void)'
//
// 4668:
//      winioctl.h(8910): warning C4668: '_WIN32_WINNT_WIN10_TH2'
//          is not defined as a preprocessor macro, replacing with
//          '0' for '#if/#elif'
//
//

#ifndef __CUDA_ARCH__
#pragma warning(push)
#pragma warning(disable: 4255)
#pragma warning(disable: 4668)
#include <Windows.h>
#pragma warning(pop)
#endif

#ifndef __CUDA_ARCH__
#include <sal.h>
#include <specstrings.h>
#else
#include <PerfectHashCuda.h>
#endif

//
// Disable the anonymous union/struct warning.
//

#pragma warning(disable: 4201)

//
// Disable "bit field types other than int" warning.
//

#pragma warning(disable: 4214)

//
// Disable the (plethora!) of spectre warnings that get reported when compiling
// with Visual Studio 2019 (v142) toolset.
//

#pragma warning(disable: 5045)

//
// Disable (temporarily, hopefully) concurrency warnings being flagged after
// upgrading to Visual Studio 2019 (v142) toolset.
//
//      warning C26110: Caller failing to hold lock 'Graph->Lock' before
//                      calling function 'ReleaseSRWLockExclusive'.
//
//      warning C26167: Possibly releasing unheld lock 'Graph->Lock' in
//                      function 'CreatePerfectHashTableImplChm01'.
//
//      warning C26165: Possibly failing to release lock 'Graph->Lock' in
//                      function 'CreatePerfectHashTableImplChm01'.
//

#pragma warning(disable: 26110 26165 26167)

#else // PH_WINDOWS
#include <PerfectHashCompat.h>
#endif

//
// Helper macro for casting to void **.
//

#define PPV(P) ((void **)(P))

//
// NT DDK types.
//

typedef struct _STRING {
    USHORT Length;
    USHORT MaximumLength;
#ifndef __CUDA_ARCH__
#ifdef _WIN64
    union {
        LONG Hash;
        LONG Padding;
    };
#endif
#else
    ULONG Padding;
#endif
    PCHAR Buffer;
} STRING, ANSI_STRING, *PSTRING, *PANSI_STRING, **PPSTRING, **PPANSI_STRING;
typedef const STRING *PCSTRING;

typedef struct _UNICODE_STRING {
    USHORT Length;
    USHORT MaximumLength;
#ifndef __CUDA_ARCH__
#ifdef _WIN64
    union {
        LONG Hash;
        LONG Padding;
    };
#endif
#else
    ULONG Padding;
#endif
    PWSTR Buffer;
} UNICODE_STRING, *PUNICODE_STRING, **PPUNICODE_STRING, ***PPPUNICODE_STRING;
typedef const UNICODE_STRING *PCUNICODE_STRING;
#define UNICODE_NULL ((WCHAR)0)

typedef _Null_terminated_ CONST CHAR *PCSZ;

typedef DOUBLE *PDOUBLE;

//
// Define a helper union that allows easy access to the bytes and shorts
// making up a ULONG.  This is predominantly used by the hash routines that
// have randomized shift/rotate instructions.
//

typedef union _ULONG_BYTES {
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
} ULONG_BYTES;
C_ASSERT(sizeof(ULONG_BYTES) == sizeof(ULONG));
typedef ULONG_BYTES *PULONG_BYTES;

#ifndef ARGUMENT_PRESENT
#define ARGUMENT_PRESENT(ArgumentPointer) (                  \
    (CHAR *)((ULONG_PTR)(ArgumentPointer)) != (CHAR *)(NULL) \
)
#endif

typedef HRESULT *PHRESULT;

#define IsValidHandle(Handle) (Handle != NULL && Handle != INVALID_HANDLE_VALUE)

#ifdef PH_WINDOWS
#define InterlockedIncrementULongPtr(Ptr) InterlockedIncrement64((PLONG64)Ptr)
#define InterlockedDecrementULongPtr(Ptr) InterlockedDecrement64((PLONG64)Ptr)
#define InterlockedAddULongPtr(Ptr, Val) \
    InterlockedAdd64((PLONG64)Ptr, (LONG64)Val)
#else
#endif

//
// Bitmap macro helpers.  (Thanks ChatGPT!)
//

#define TestBit32(Address, Bit) (                          \
    ((((PLONG32)(Address))[(Bit >> 5)] >> (Bit & 31)) & 1) \
)

#define TestBit64(Address, Bit) (                          \
    ((((PLONG64)(Address))[(Bit >> 6)] >> (Bit & 63)) & 1) \
)

#define SetBit32(Address, Bit) (                               \
    ((((PLONG32)(Address))[(Bit) >> 5] |= (1L << (Bit & 31)))) \
)

#define SetBit64(Address, Bit) (                                \
    ((((PLONG64)(Address))[(Bit) >> 6] |= (1LL << (Bit & 63)))) \
)

//
// Define start/end markers for IACA.
//

#define IACA_VC_START() __writegsbyte(111, 111)
#define IACA_VC_END()   __writegsbyte(222, 222)

//
// Define an enumeration for identifying CPU architectures.
//

#define PERFECT_HASH_CPU_ARCH_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    FIRST_ENTRY(x86, X86)                                           \
    ENTRY(x64, X64)                                                 \
    ENTRY(Arm, ARM)                                                 \
    ENTRY(Arm64, ARM64)                                             \
    LAST_ENTRY(Cuda, CUDA)

#define PERFECT_HASH_CPU_ARCH_TABLE_ENTRY(ENTRY) \
    PERFECT_HASH_CPU_ARCH_TABLE(ENTRY, ENTRY, ENTRY)

#define EXPAND_AS_CPU_ARCH_ID_ENUM(Name, Upper) \
    PerfectHash##Name##CpuArchId,

typedef enum PERFECT_HASH_CPU_ARCH_ID {

    //
    // Explicitly define a null ID to take the 0-index slot.  This makes enum
    // validation easier.
    //

    PerfectHashNullCpuArchId = 0,

    //
    // Begin valid CPU architectures.
    //

    PERFECT_HASH_CPU_ARCH_TABLE_ENTRY(EXPAND_AS_CPU_ARCH_ID_ENUM)

    //
    // End valid CPU architectures.
    //

    //
    // N.B. Keep the next value last.
    //

    PerfectHashInvalidCpuArchId

} PERFECT_HASH_CPU_ARCH_ID;
typedef PERFECT_HASH_CPU_ARCH_ID *PPERFECT_HASH_CPU_ARCH_ID;

//
// Provide a simple inline CPU architecture enum validation routine.
//

FORCEINLINE
BOOLEAN
IsValidPerfectHashCpuArchId(
    _In_ PERFECT_HASH_CPU_ARCH_ID CpuArchId
    )
{
    return (
        CpuArchId > PerfectHashNullCpuArchId &&
        CpuArchId < PerfectHashInvalidCpuArchId
    );
}

//
// Provide a simple inline routine for obtaining the current CPU architecture.
//

FORCEINLINE
PERFECT_HASH_CPU_ARCH_ID
PerfectHashGetCurrentCpuArch(
    VOID
    )
{
#ifdef _M_AMD64
    return PerfectHashx64CpuArchId;
#elif defined(_M_IX86)
    return PerfectHashx86CpuArchId;
#elif defined(_M_ARM64)
    return PerfectHashArm64CpuArchId;
#elif defined(_M_ARM)
    return PerfectHashArmCpuArchId;
#elif defined(__CUDA_ARCH__)
    return PerfectHashCudaArchId;
#else
#error Unknown CPU architecture.
#endif
}

//
// Define an X-macro for COM interfaces.  The ENTRY macros receive the following
// parameters: (Name, Upper, Guid).
//

#define GUID_EX(l, w1, w2, b1, b2, b3, b4, b5, b6, b7, b8) \
    { l, w1, w2, { b1, b2,  b3,  b4,  b5,  b6,  b7,  b8 } }

#define PERFECT_HASH_INTERFACE_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
                                                                     \
    FIRST_ENTRY(                                                     \
        IUnknown,                                                    \
        IUNKNOWN,                                                    \
        GUID_EX(                                                     \
            0x00000000, 0x0000, 0x0000,                              \
            0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46           \
        )                                                            \
    )                                                                \
                                                                     \
    ENTRY(                                                           \
        IClassFactory,                                               \
        ICLASSFACTORY,                                               \
        GUID_EX(                                                     \
            0x00000001, 0x0000, 0x0000,                              \
            0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46           \
        )                                                            \
    )                                                                \
                                                                     \
    ENTRY(                                                           \
        Keys,                                                        \
        KEYS,                                                        \
        GUID_EX(                                                     \
            0x7e43ebea, 0x8671, 0x47ba,                              \
            0xb8, 0x44, 0x76, 0xb, 0x7a, 0x9e, 0xa9, 0x21            \
        )                                                            \
    )                                                                \
                                                                     \
    ENTRY(                                                           \
        Context,                                                     \
        CONTEXT,                                                     \
        GUID_EX(                                                     \
            0xd4b24571, 0x99d7, 0x44ba,                              \
            0x8a, 0x27, 0x63, 0xd8, 0x73, 0x9f, 0x9b, 0x81           \
        )                                                            \
    )                                                                \
                                                                     \
    ENTRY(                                                           \
        Table,                                                       \
        TABLE,                                                       \
        GUID_EX(                                                     \
            0xc265816f, 0xc6a9, 0x4b44,                              \
            0xbc, 0xee, 0xec, 0x5a, 0x12, 0xab, 0xe1, 0xef           \
        )                                                            \
    )                                                                \
                                                                     \
    ENTRY(                                                           \
        Rtl,                                                         \
        RTL,                                                         \
        GUID_EX(                                                     \
            0x9c05a3d6, 0xbc30, 0x45e6,                              \
            0xbe, 0xa6, 0x50, 0x4f, 0xcc, 0x92, 0x43, 0xa8           \
        )                                                            \
    )                                                                \
                                                                     \
    ENTRY(                                                           \
        Allocator,                                                   \
        ALLOCATOR,                                                   \
        GUID_EX(                                                     \
            0xf87564d2, 0xb3c7, 0x4cca,                              \
            0x90, 0x13, 0xeb, 0x59, 0xc1, 0xe2, 0x53, 0xb7           \
        )                                                            \
    )                                                                \
                                                                     \
    ENTRY(                                                           \
        File,                                                        \
        FILE,                                                        \
        GUID_EX(                                                     \
            0x27549274, 0x968a, 0x499a,                              \
            0x83, 0x49, 0x31, 0x33, 0xe3, 0xd5, 0xe6, 0x49           \
        )                                                            \
    )                                                                \
                                                                     \
    ENTRY(                                                           \
        Path,                                                        \
        PATH,                                                        \
        GUID_EX(                                                     \
            0x267623b1, 0xc5d, 0x47b1,                               \
            0xa2, 0x97, 0xdf, 0xe, 0x54, 0x67, 0xaf, 0xd1            \
        )                                                            \
    )                                                                \
                                                                     \
    ENTRY(                                                           \
        Directory,                                                   \
        DIRECTORY,                                                   \
        GUID_EX(                                                     \
            0x5d673839, 0x1686, 0x411e,                              \
            0x99, 0x2, 0x46, 0xc6, 0xe9, 0x7c, 0xa5, 0x67            \
        )                                                            \
    )                                                                \
                                                                     \
    ENTRY(                                                           \
        GuardedList,                                                 \
        GUARDED_LIST,                                                \
        GUID_EX(                                                     \
            0x14a25ba2, 0x3c18, 0x413f,                              \
            0x8c, 0x76, 0xa7, 0xa9, 0x1e, 0xc8, 0x8c, 0x2a           \
        )                                                            \
    )                                                                \
                                                                     \
    ENTRY(                                                           \
        Graph,                                                       \
        GRAPH,                                                       \
        GUID_EX(                                                     \
            0xb906f824, 0xcb59, 0x4696,                              \
            0x84, 0x77, 0x44, 0xd4, 0xba, 0x9, 0xda, 0x94            \
        )                                                            \
    )                                                                \
                                                                     \
    ENTRY(                                                           \
        GraphCu,                                                     \
        GRAPH_CU,                                                    \
        GUID_EX(                                                     \
            0x5067a808, 0xd72b, 0x4f7e,                              \
            0xb3, 0xd0, 0xa3, 0xe8, 0xcf, 0x6f, 0x23, 0xc7           \
                                                                     \
        )                                                            \
    )                                                                \
                                                                     \
    ENTRY(                                                           \
        Cu,                                                          \
        CU,                                                          \
        GUID_EX(                                                     \
            0x8e124a55, 0xf609, 0x45be,                              \
            0xa7, 0x2e, 0x96, 0x74, 0x2d, 0xbb, 0x1, 0xf5            \
        )                                                            \
    )                                                                \
                                                                     \
    LAST_ENTRY(                                                      \
        Rng,                                                         \
        RNG,                                                         \
        GUID_EX(                                                     \
            0xfd84eebe, 0x2571, 0x4517,                              \
            0xa5, 0x14, 0x7a, 0xe4, 0x50, 0x32, 0x7d, 0x48           \
        )                                                            \
    )

#define PERFECT_HASH_INTERFACE_TABLE_ENTRY(ENTRY) \
    PERFECT_HASH_INTERFACE_TABLE(ENTRY, ENTRY, ENTRY)

#define EXPAND_AS_INTERFACE_ENUM_FIRST(Name, Upper, Guid)         \
    PerfectHash##Name##InterfaceId,                               \
    PerfectHashFirstInterfaceId = PerfectHash##Name##InterfaceId,

#define EXPAND_AS_INTERFACE_ENUM(Name, Upper, Guid) \
    PerfectHash##Name##InterfaceId,

#define EXPAND_AS_INTERFACE_ENUM_LAST(Name, Upper, Guid)         \
    PerfectHash##Name##InterfaceId,                              \
    PerfectHashLastInterfaceId = PerfectHash##Name##InterfaceId,

typedef enum _PERFECT_HASH_INTERFACE_ID {

    //
    // Explicitly define a null ID to take the 0-index slot.  This makes enum
    // validation easier.
    //

    PerfectHashNullInterfaceId = 0,

    //
    // Begin valid interfaces.
    //

    PERFECT_HASH_INTERFACE_TABLE(EXPAND_AS_INTERFACE_ENUM_FIRST,
                                 EXPAND_AS_INTERFACE_ENUM,
                                 EXPAND_AS_INTERFACE_ENUM_LAST)

    //
    // End valid interfaces.
    //

    //
    // N.B. Keep the next value last.
    //

    PerfectHashInvalidInterfaceId,

} PERFECT_HASH_INTERFACE_ID;

//
// Provide a simple inline interface enum validation routine.
//

FORCEINLINE
BOOLEAN
IsValidPerfectHashInterfaceId(
    _In_ PERFECT_HASH_INTERFACE_ID InterfaceId
    )
{
    return (
        InterfaceId > PerfectHashNullInterfaceId &&
        InterfaceId < PerfectHashInvalidInterfaceId
    );
}

#define DEFINE_GUID_EX(Name, l, w1, w2, b1, b2, b3, b4, b5, b6, b7, b8) \
    static const GUID Name                                              \
        = { l, w1, w2, { b1, b2,  b3,  b4,  b5,  b6,  b7,  b8 } }

typedef GUID *PGUID;
typedef const GUID CGUID;
typedef GUID const *PCGUID;

//
// Define static const GUIDs for the perfect hash library's class ID, then all
// interface IDs.
//

//
// CLSID_PERFECT_HASH: 402045FD-72F4-4A05-902E-D22B7C1877B4
//

DEFINE_GUID_EX(CLSID_PERFECT_HASH, 0x402045fd, 0x72f4, 0x4a05,
               0x90, 0x2e, 0xd2, 0x2b, 0x7c, 0x18, 0x77, 0xb4);

#define EXPAND_AS_DEFINE_GUID_EX(Name, Upper, Guid) \
    static const GUID IID_PERFECT_HASH_##Upper = Guid;

PERFECT_HASH_INTERFACE_TABLE_ENTRY(EXPAND_AS_DEFINE_GUID_EX);

//
// GUID array.
//

#define EXPAND_AS_IID_ADDRESS(Name, Upper, Guid) &IID_PERFECT_HASH_##Upper,

static const PCGUID PerfectHashInterfaceGuids[] = {
    NULL,
    PERFECT_HASH_INTERFACE_TABLE_ENTRY(EXPAND_AS_IID_ADDRESS)
    NULL
};

#ifndef __CUDA_ARCH__
static const BYTE NumberOfPerfectHashInterfaceGuids =
    ARRAYSIZE(PerfectHashInterfaceGuids);

//
// Convert a GUID to an interface ID.
//

FORCEINLINE
PERFECT_HASH_INTERFACE_ID
PerfectHashInterfaceGuidToId(
    _In_ REFIID Guid
    )
{
    BYTE Index;
    BYTE Count;
    PERFECT_HASH_INTERFACE_ID Id = PerfectHashNullInterfaceId;

    if (!Guid) {
        return PerfectHashInvalidInterfaceId;
    }

    //
    // We start the index at 1 in order to skip the first NULL entry.
    //

    Count = NumberOfPerfectHashInterfaceGuids;

    for (Index = 1; Index < Count; Index++) {
        if (InlineIsEqualGUID(Guid, PerfectHashInterfaceGuids[Index])) {
            Id = (PERFECT_HASH_INTERFACE_ID)Index;
            break;
        }
    }

    return Id;
}

#endif // __CUDA_ARCH__

//
// COM-related funtion pointer typedefs.
//

typedef
HRESULT
(CO_INITIALIZE_EX)(
    _In_opt_ LPVOID Reserved,
    _In_ DWORD CoInit
    );
typedef CO_INITIALIZE_EX *PCO_INITIALIZE_EX;

typedef
_Must_inspect_result_
HRESULT
(STDAPICALLTYPE DLL_GET_CLASS_OBJECT)(
    _In_ REFCLSID ClassId,
    _In_ REFIID InterfaceId,
    _COM_Outptr_ LPVOID *Interface
    );
typedef DLL_GET_CLASS_OBJECT *PDLL_GET_CLASS_OBJECT;

typedef
HRESULT
(STDAPICALLTYPE DLL_CAN_UNLOAD_NOW)(
    VOID
    );
typedef DLL_CAN_UNLOAD_NOW *PDLL_CAN_UNLOAD_NOW;


//
//
// Define helper macros to handle COM interface glue.
//
// N.B. We abuse the COM spec a bit here in that all of our components,
//      including IUnknown, actually implement IClassFactory.
//


#define DECLARE_COMPONENT(Name, Upper)                           \
    typedef struct _##Upper Upper;                               \
    typedef Upper *P##Upper;                                     \
    typedef const Upper *PC##Upper;                              \
                                                                 \
    typedef                                                      \
    _Must_inspect_result_                                        \
    _Success_(return >= 0)                                       \
    HRESULT                                                      \
    (STDAPICALLTYPE Upper##_QUERY_INTERFACE)(                    \
        _In_ P##Upper Name,                                      \
        _In_ REFIID InterfaceId,                                 \
        _COM_Outptr_ PVOID *Interface                            \
        );                                                       \
    typedef Upper##_QUERY_INTERFACE *P##Upper##_QUERY_INTERFACE; \
                                                                 \
    typedef                                                      \
    ULONG                                                        \
    (STDAPICALLTYPE Upper##_ADD_REF)(                            \
        _In_ P##Upper Name                                       \
        );                                                       \
    typedef Upper##_ADD_REF *P##Upper##_ADD_REF;                 \
                                                                 \
    typedef                                                      \
    ULONG                                                        \
    (STDAPICALLTYPE Upper##_RELEASE)(                            \
        _In_ P##Upper Name                                       \
        );                                                       \
    typedef Upper##_RELEASE *P##Upper##_RELEASE;                 \
                                                                 \
    typedef                                                      \
    _Must_inspect_result_                                        \
    _Success_(return >= 0)                                       \
    HRESULT                                                      \
    (STDAPICALLTYPE Upper##_CREATE_INSTANCE)(                    \
        _In_ P##Upper Name,                                      \
        _In_opt_ PIUNKNOWN UnknownOuter,                         \
        _In_ REFIID InterfaceId,                                 \
        _COM_Outptr_ PVOID *Interface                            \
        );                                                       \
    typedef Upper##_CREATE_INSTANCE *P##Upper##_CREATE_INSTANCE; \
                                                                 \
    typedef                                                      \
    _Must_inspect_result_                                        \
    _Success_(return >= 0)                                       \
    HRESULT                                                      \
    (STDAPICALLTYPE Upper##_LOCK_SERVER)(                        \
        _In_ P##Upper Name,                                      \
        _In_ BOOL Lock                                           \
        );                                                       \
    typedef Upper##_LOCK_SERVER *P##Upper##_LOCK_SERVER


#define DECLARE_COMPONENT_VTBL_HEADER(Upper)   \
    P##Upper##_QUERY_INTERFACE QueryInterface; \
    P##Upper##_ADD_REF AddRef;                 \
    P##Upper##_RELEASE Release;                \
    P##Upper##_CREATE_INSTANCE CreateInstance; \
    P##Upper##_LOCK_SERVER LockServer

//
// Define our COM interfaces.  We include IUnknown and IClassFactory just so
// we're consistent with our Cutler-Normal-Form naming scheme.
//

//
// IUnknown
//

DECLARE_COMPONENT(Unknown, IUNKNOWN);

typedef struct _IUNKNOWN_VTBL {
    DECLARE_COMPONENT_VTBL_HEADER(IUNKNOWN);
} IUNKNOWN_VTBL;
typedef IUNKNOWN_VTBL *PIUNKNOWN_VTBL;

#ifndef _PERFECT_HASH_INTERNAL_BUILD
typedef struct _IUNKNOWN {
    PIUNKNOWN_VTBL Vtbl;
} IUNKNOWN;
typedef IUNKNOWN *PIUNKNOWN;
#endif

//
// IClassFactory
//

DECLARE_COMPONENT(ClassFactory, ICLASSFACTORY);

typedef struct _ICLASSFACTORY_VTBL {
    DECLARE_COMPONENT_VTBL_HEADER(ICLASSFACTORY);
} ICLASSFACTORY_VTBL;
typedef ICLASSFACTORY_VTBL *PICLASSFACTORY_VTBL;

#ifndef _PERFECT_HASH_INTERNAL_BUILD
typedef struct _ICLASSFACTORY {
    PICLASSFACTORY_VTBL Vtbl;
} ICLASSFACTORY;
typedef ICLASSFACTORY *PICLASSFACTORY;
#endif

//
// Define the ALLOCATOR interface.
//

DECLARE_COMPONENT(Allocator, ALLOCATOR);

typedef
_Must_inspect_result_
_Ret_maybenull_
_Success_(return != 0)
_Post_writable_byte_size_(Size)
PVOID
(STDAPICALLTYPE ALLOCATOR_MALLOC)(
    _In_ PALLOCATOR Allocator,
    _In_ SIZE_T Size
    );
typedef ALLOCATOR_MALLOC *PALLOCATOR_MALLOC;

typedef
_Must_inspect_result_
_Ret_maybenull_
_Success_(return != 0)
_Post_writable_byte_size_(NumberOfElements * ElementSize)
PVOID
(STDAPICALLTYPE ALLOCATOR_CALLOC)(
    _In_ PALLOCATOR Allocator,
    _In_ SIZE_T NumberOfElements,
    _In_ SIZE_T ElementSize
    );
typedef ALLOCATOR_CALLOC *PALLOCATOR_CALLOC;

typedef
_Must_inspect_result_
_Ret_reallocated_bytes_(Address, Size)
PVOID
(STDAPICALLTYPE ALLOCATOR_REALLOC)(
    _In_ PALLOCATOR Allocator,
    _Frees_ptr_opt_ PVOID Address,
    _In_ SIZE_T Size
    );
typedef ALLOCATOR_REALLOC *PALLOCATOR_REALLOC;

typedef
_Must_inspect_result_
_Ret_reallocated_bytes_(Address, NumberOfElements * ElementSize)
PVOID
(STDAPICALLTYPE ALLOCATOR_RECALLOC)(
    _In_ PALLOCATOR Allocator,
    _Frees_ptr_opt_ PVOID Address,
    _In_ SIZE_T NumberOfElements,
    _In_ SIZE_T ElementSize
    );
typedef ALLOCATOR_RECALLOC *PALLOCATOR_RECALLOC;

typedef
VOID
(STDAPICALLTYPE ALLOCATOR_FREE)(
    _In_ PALLOCATOR Allocator,
    _Frees_ptr_opt_ PVOID Address
    );
typedef ALLOCATOR_FREE *PALLOCATOR_FREE;

typedef
VOID
(STDAPICALLTYPE ALLOCATOR_FREE_POINTER)(
    _In_ PALLOCATOR Allocator,
    _Inout_ PVOID *AddressPointer
    );
typedef ALLOCATOR_FREE_POINTER *PALLOCATOR_FREE_POINTER;

typedef
VOID
(STDAPICALLTYPE ALLOCATOR_FREE_STRING_BUFFER)(
    _In_ PALLOCATOR Allocator,
    _In_ PSTRING String
    );
typedef ALLOCATOR_FREE_STRING_BUFFER
      *PALLOCATOR_FREE_STRING_BUFFER;

typedef
VOID
(STDAPICALLTYPE ALLOCATOR_FREE_UNICODE_STRING_BUFFER)(
    _In_ PALLOCATOR Allocator,
    _In_ PUNICODE_STRING String
    );
typedef ALLOCATOR_FREE_UNICODE_STRING_BUFFER
      *PALLOCATOR_FREE_UNICODE_STRING_BUFFER;

typedef
_Must_inspect_result_
_Success_(return != 0)
_Ret_maybenull_
_Post_writable_byte_size_(Size)
PVOID
(STDAPICALLTYPE ALLOCATOR_ALIGNED_MALLOC)(
    _In_ PALLOCATOR Allocator,
    _In_ SIZE_T Size,
    _In_ SIZE_T Alignment
    );
typedef ALLOCATOR_ALIGNED_MALLOC *PALLOCATOR_ALIGNED_MALLOC;

typedef
_Must_inspect_result_
_Success_(return != 0)
_Ret_maybenull_
_Post_writable_byte_size_(NumberOfElements * ElementSize)
PVOID
(STDAPICALLTYPE ALLOCATOR_ALIGNED_CALLOC)(
    _In_ PALLOCATOR Allocator,
    _In_ SIZE_T NumberOfElements,
    _In_ SIZE_T ElementSize,
    _In_ SIZE_T Alignment
    );
typedef ALLOCATOR_ALIGNED_CALLOC *PALLOCATOR_ALIGNED_CALLOC;

typedef
_Must_inspect_result_
_Ret_maybenull_
_Ret_reallocated_bytes_(Address, Size)
PVOID
(STDAPICALLTYPE ALLOCATOR_ALIGNED_REALLOC)(
    _In_ PALLOCATOR Allocator,
    _Frees_ptr_opt_ PVOID Address,
    _In_ SIZE_T Size,
    _In_ SIZE_T Alignment
    );
typedef ALLOCATOR_ALIGNED_REALLOC *PALLOCATOR_ALIGNED_REALLOC;

typedef
_Must_inspect_result_
_Ret_maybenull_
_Ret_reallocated_bytes_(Address, NumberOfElements * ElementSize)
PVOID
(STDAPICALLTYPE ALLOCATOR_ALIGNED_RECALLOC)(
    _In_ PALLOCATOR Allocator,
    _Frees_ptr_opt_ PVOID Address,
    _In_ SIZE_T NumberOfElements,
    _In_ SIZE_T ElementSize,
    _In_ SIZE_T Alignment
    );
typedef ALLOCATOR_ALIGNED_RECALLOC *PALLOCATOR_ALIGNED_RECALLOC;

typedef
VOID
(STDAPICALLTYPE ALLOCATOR_ALIGNED_FREE)(
    _In_ PALLOCATOR Allocator,
    _Frees_ptr_opt_ PVOID Address
    );
typedef ALLOCATOR_ALIGNED_FREE *PALLOCATOR_ALIGNED_FREE;

typedef
VOID
(STDAPICALLTYPE ALLOCATOR_ALIGNED_FREE_POINTER)(
    _In_ PALLOCATOR Allocator,
    _Inout_ PVOID *AddressPointer
    );
typedef ALLOCATOR_ALIGNED_FREE_POINTER *PALLOCATOR_ALIGNED_FREE_POINTER;

typedef
_Must_inspect_result_
_Success_(return != 0)
_Ret_maybenull_
_Post_writable_byte_size_(Size)
PVOID
(STDAPICALLTYPE ALLOCATOR_ALIGNED_OFFSET_MALLOC)(
    _In_ PALLOCATOR Allocator,
    _In_ SIZE_T Size,
    _In_ SIZE_T Alignment,
    _In_ SIZE_T Offset
    );
typedef ALLOCATOR_ALIGNED_OFFSET_MALLOC *PALLOCATOR_ALIGNED_OFFSET_MALLOC;

typedef
_Must_inspect_result_
_Success_(return != 0)
_Ret_maybenull_
_Post_writable_byte_size_(NumberOfElements * ElementSize)
PVOID
(STDAPICALLTYPE ALLOCATOR_ALIGNED_OFFSET_CALLOC)(
    _In_ PALLOCATOR Allocator,
    _In_ SIZE_T NumberOfElements,
    _In_ SIZE_T ElementSize,
    _In_ SIZE_T Alignment,
    _In_ SIZE_T Offset
    );
typedef ALLOCATOR_ALIGNED_OFFSET_CALLOC *PALLOCATOR_ALIGNED_OFFSET_CALLOC;

typedef
_Must_inspect_result_
_Ret_maybenull_
_Ret_reallocated_bytes_(Address, Size)
PVOID
(STDAPICALLTYPE ALLOCATOR_ALIGNED_OFFSET_REALLOC)(
    _In_ PALLOCATOR Allocator,
    _Frees_ptr_opt_ PVOID Address,
    _In_ SIZE_T Size,
    _In_ SIZE_T Alignment,
    _In_ SIZE_T Offset
    );
typedef ALLOCATOR_ALIGNED_OFFSET_REALLOC *PALLOCATOR_ALIGNED_OFFSET_REALLOC;

typedef
_Must_inspect_result_
_Ret_maybenull_
_Ret_reallocated_bytes_(Address, NumberOfElements * ElementSize)
PVOID
(STDAPICALLTYPE ALLOCATOR_ALIGNED_OFFSET_RECALLOC)(
    _In_ PALLOCATOR Allocator,
    _Frees_ptr_opt_ PVOID Address,
    _In_ SIZE_T NumberOfElements,
    _In_ SIZE_T ElementSize,
    _In_ SIZE_T Alignment,
    _In_ SIZE_T Offset
    );
typedef ALLOCATOR_ALIGNED_OFFSET_RECALLOC *PALLOCATOR_ALIGNED_OFFSET_RECALLOC;

typedef struct _ALLOCATOR_VTBL {
    DECLARE_COMPONENT_VTBL_HEADER(ALLOCATOR);
    PALLOCATOR_MALLOC Malloc;
    PALLOCATOR_CALLOC Calloc;
    PALLOCATOR_REALLOC ReAlloc;
    PALLOCATOR_RECALLOC ReCalloc;
    PALLOCATOR_FREE Free;
    PALLOCATOR_FREE_POINTER FreePointer;
    PALLOCATOR_FREE_STRING_BUFFER FreeStringBuffer;
    PALLOCATOR_FREE_UNICODE_STRING_BUFFER FreeUnicodeStringBuffer;
    PALLOCATOR_ALIGNED_MALLOC AlignedMalloc;
    PALLOCATOR_ALIGNED_CALLOC AlignedCalloc;
    PALLOCATOR_ALIGNED_REALLOC AlignedReAlloc;
    PALLOCATOR_ALIGNED_RECALLOC AlignedReCalloc;
    PALLOCATOR_ALIGNED_FREE AlignedFree;
    PALLOCATOR_ALIGNED_FREE_POINTER AlignedFreePointer;
    PALLOCATOR_ALIGNED_OFFSET_MALLOC AlignedOffsetMalloc;
    PALLOCATOR_ALIGNED_OFFSET_CALLOC AlignedOffsetCalloc;
    PALLOCATOR_ALIGNED_OFFSET_REALLOC AlignedOffsetReAlloc;
    PALLOCATOR_ALIGNED_OFFSET_RECALLOC AlignedOffsetReCalloc;
} ALLOCATOR_VTBL;
typedef ALLOCATOR_VTBL *PALLOCATOR_VTBL;

#ifndef _PERFECT_HASH_INTERNAL_BUILD
typedef struct _ALLOCATOR {
    PALLOCATOR_VTBL Vtbl;
} ALLOCATOR;
typedef ALLOCATOR *PALLOCATOR;
#endif

//
// Define the PERFECT_HASH_PATH interface.
//

DECLARE_COMPONENT(Path, PERFECT_HASH_PATH);

typedef struct _PERFECT_HASH_PATH_PARTS {

    //
    // Fully-qualified, NULL-terminated path of the file.  Path.Buffer is
    // owned by this component, and allocated via the Allocator.
    //

    UNICODE_STRING FullPath;

    //
    // N.B. The following path component fields all point within Path.Buffer.
    //

    //
    // Drive (e.g. "C" or "\\??\\...".
    //

    UNICODE_STRING Drive;

    //
    // Fully-qualified directory, including drive.
    //

    UNICODE_STRING Directory;

    //
    // Base name of the file (i.e. file name excluding extension).
    //

    UNICODE_STRING BaseName;

    //
    // File name (includes extension).
    //

    UNICODE_STRING FileName;

    //
    // File extension (e.g. ".keys").
    //

    UNICODE_STRING Extension;

    //
    // Stream name if applicable (e.g. ":Info").
    //

    UNICODE_STRING StreamName;

    //
    // If the base name (above) is a valid C identifier, an ASCII-encoded
    // version of the string will be available in the following field.  This
    // is guaranteed to only use ASCII (7-bit) characters and will be a valid
    // C identifier (only starts with _, a-z or A-Z, and only contains _, 0-9,
    // a-z and A-Z characters).
    //

    STRING BaseNameA;

    //
    // As above, but captures just the "table name" part of the file.  This
    // will exclude any additional suffix string data appended to the file name
    // during creation.  This buffer is identical to BaseNameA.Buffer, however,
    // the lengths differ.
    //

    STRING TableNameA;

    //
    // As above, but an uppercase version of the base name and table name if
    // applicable.
    //

    STRING BaseNameUpperA;
    STRING TableNameUpperA;

} PERFECT_HASH_PATH_PARTS;
typedef PERFECT_HASH_PATH_PARTS *PPERFECT_HASH_PATH_PARTS;
typedef const PERFECT_HASH_PATH_PARTS *PCPERFECT_HASH_PATH_PARTS;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_lock_not_held_(Path->Lock)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_PATH_COPY)(
    _In_ PPERFECT_HASH_PATH Path,
    _In_ PCUNICODE_STRING Source,
    _Out_opt_ PCPERFECT_HASH_PATH_PARTS *Parts,
    _Reserved_ PVOID Reserved
    );
typedef PERFECT_HASH_PATH_COPY *PPERFECT_HASH_PATH_COPY;

//
// Define path creation method and supporting flags.
//

typedef union _PERFECT_HASH_PATH_CREATE_FLAGS {

    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, disables the logic that automatically replaces characters
        // like whitespace, comma, hyphen etc with underscores.
        //

        ULONG DisableCharReplacement:1;

        //
        // Unused bits.
        //

        ULONG Unused:31;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_PATH_CREATE_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_PATH_CREATE_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_PATH_CREATE_FLAGS *PPERFECT_HASH_PATH_CREATE_FLAGS;

FORCEINLINE
HRESULT
IsValidPathCreateFlags(
    _In_ PPERFECT_HASH_PATH_CREATE_FLAGS PathCreateFlags
    )
{

    if (!ARGUMENT_PRESENT(PathCreateFlags)) {
        return E_POINTER;
    }

    if (PathCreateFlags->Unused != 0) {
        return E_FAIL;
    }

    return S_OK;
}

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_lock_not_held_(Path->Lock)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_PATH_CREATE)(
    _In_ PPERFECT_HASH_PATH Path,
    _In_ PPERFECT_HASH_PATH ExistingPath,
    _In_opt_ PCUNICODE_STRING NewDirectory,
    _In_opt_ PCUNICODE_STRING DirectorySuffix,
    _In_opt_ PCUNICODE_STRING NewBaseName,
    _In_opt_ PCUNICODE_STRING BaseNameSuffix,
    _In_opt_ PCUNICODE_STRING NewExtension,
    _In_opt_ PCUNICODE_STRING NewStreamName,
    _Out_opt_ PCPERFECT_HASH_PATH_PARTS *Parts,
    _In_opt_ PPERFECT_HASH_PATH_CREATE_FLAGS PathCreateFlags
    );
typedef PERFECT_HASH_PATH_CREATE *PPERFECT_HASH_PATH_CREATE;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Path->Lock)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_PATH_RESET)(
    _In_ PPERFECT_HASH_PATH Path
    );
typedef PERFECT_HASH_PATH_RESET *PPERFECT_HASH_PATH_RESET;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_lock_not_held_(Path->Lock)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_PATH_GET_PARTS)(
    _In_ PPERFECT_HASH_PATH Path,
    _Out_ PCPERFECT_HASH_PATH_PARTS *Parts
    );
typedef PERFECT_HASH_PATH_GET_PARTS *PPERFECT_HASH_PATH_GET_PARTS;

#ifndef _PERFECT_HASH_INTERNAL_BUILD
typedef struct _PERFECT_HASH_PATH_VTBL {
    DECLARE_COMPONENT_VTBL_HEADER(PERFECT_HASH_PATH);
    PPERFECT_HASH_PATH_COPY Copy;
    PPERFECT_HASH_PATH_CREATE Create;
    PPERFECT_HASH_PATH_RESET Reset;
    PPERFECT_HASH_PATH_GET_PARTS GetParts;
} PERFECT_HASH_PATH_VTBL;
typedef PERFECT_HASH_PATH_VTBL *PPERFECT_HASH_PATH_VTBL;

typedef struct _PERFECT_HASH_PATH {
    PPERFECT_HASH_PATH_VTBL Vtbl;
    SRWLOCK Lock;
} PERFECT_HASH_PATH;
typedef PERFECT_HASH_PATH *PPERFECT_HASH_PATH;
#endif

//
// Define helper macros for acquiring and releasing the perfect hash path lock
// in shared and exclusive mode.
//

#define TryAcquirePerfectHashPathLockExclusive(Path) \
    TryAcquireSRWLockExclusive(&Path->Lock)

#define AcquirePerfectHashPathLockExclusive(Path) \
    AcquireSRWLockExclusive(&Path->Lock)

#define ReleasePerfectHashPathLockExclusive(Path) \
    ReleaseSRWLockExclusive(&Path->Lock)

#define TryAcquirePerfectHashPathLockShared(Path) \
    TryAcquireSRWLockShared(&Path->Lock)

#define AcquirePerfectHashPathLockShared(Path) \
    AcquireSRWLockShared(&Path->Lock)

#define ReleasePerfectHashPathLockShared(Path) \
    ReleaseSRWLockShared(&Path->Lock)

//
// Define the PERFECT_HASH_DIRECTORY interface.
//

DECLARE_COMPONENT(Directory, PERFECT_HASH_DIRECTORY);

typedef union _PERFECT_HASH_DIRECTORY_OPEN_FLAGS {

    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // Unused bits.
        //

        ULONG Unused:32;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_DIRECTORY_OPEN_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_DIRECTORY_OPEN_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_DIRECTORY_OPEN_FLAGS *PPERFECT_HASH_DIRECTORY_OPEN_FLAGS;

FORCEINLINE
HRESULT
IsValidDirectoryOpenFlags(
    _In_ PPERFECT_HASH_DIRECTORY_OPEN_FLAGS DirectoryOpenFlags
    )
{

    if (!ARGUMENT_PRESENT(DirectoryOpenFlags)) {
        return E_POINTER;
    }

    if (DirectoryOpenFlags->Unused != 0) {
        return E_FAIL;
    }

    return S_OK;
}

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_DIRECTORY_OPEN)(
    _In_ PPERFECT_HASH_DIRECTORY Directory,
    _In_ PPERFECT_HASH_PATH SourcePath,
    _In_opt_ PPERFECT_HASH_DIRECTORY_OPEN_FLAGS DirectoryOpenFlags
    );
typedef PERFECT_HASH_DIRECTORY_OPEN *PPERFECT_HASH_DIRECTORY_OPEN;

//
// Define Create() method and supporting flags.
//

typedef union _PERFECT_HASH_DIRECTORY_CREATE_FLAGS {

    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // Unused bits.
        //

        ULONG Unused:32;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_DIRECTORY_CREATE_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_DIRECTORY_CREATE_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_DIRECTORY_CREATE_FLAGS
      *PPERFECT_HASH_DIRECTORY_CREATE_FLAGS;

FORCEINLINE
HRESULT
IsValidDirectoryCreateFlags(
    _In_ PPERFECT_HASH_DIRECTORY_CREATE_FLAGS DirectoryCreateFlags
    )
{

    if (!ARGUMENT_PRESENT(DirectoryCreateFlags)) {
        return E_POINTER;
    }

    if (DirectoryCreateFlags->Unused != 0) {
        return E_FAIL;
    }

    return S_OK;
}

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_DIRECTORY_CREATE)(
    _In_ PPERFECT_HASH_DIRECTORY Directory,
    _In_ PPERFECT_HASH_PATH SourcePath,
    _In_opt_ PPERFECT_HASH_DIRECTORY_CREATE_FLAGS DirectoryCreateFlags
    );
typedef PERFECT_HASH_DIRECTORY_CREATE *PPERFECT_HASH_DIRECTORY_CREATE;

//
// Define the PERFECT_HASH_DIRECTORY_FLAGS structure.
//

typedef union _PERFECT_HASH_DIRECTORY_FLAGS {
    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, indicates the directory was initialized via Open().
        //
        // Invariant:
        //
        //      If Opened == TRUE:
        //          Assert Created == FALSE
        //

        ULONG Opened:1;

        //
        // When set, indicates the directory was initialized via Create().
        //
        // Invariant:
        //
        //      If Created == TRUE:
        //          Assert Loaded == FALSE
        //

        ULONG Created:1;

        //
        // Unused bits.
        //

        ULONG Unused:30;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_DIRECTORY_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_DIRECTORY_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_DIRECTORY_FLAGS *PPERFECT_HASH_DIRECTORY_FLAGS;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_DIRECTORY_GET_FLAGS)(
    _In_ PPERFECT_HASH_DIRECTORY Directory,
    _In_ ULONG SizeOfFlags,
    _Out_writes_bytes_(SizeOfFlags) PPERFECT_HASH_DIRECTORY_FLAGS Flags
    );
typedef PERFECT_HASH_DIRECTORY_GET_FLAGS *PPERFECT_HASH_DIRECTORY_GET_FLAGS;

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_DIRECTORY_GET_PATH)(
    _In_ PPERFECT_HASH_DIRECTORY Directory,
    _Inout_opt_ PPERFECT_HASH_PATH *Path
    );
typedef PERFECT_HASH_DIRECTORY_GET_PATH *PPERFECT_HASH_DIRECTORY_GET_PATH;

#ifndef _PERFECT_HASH_INTERNAL_BUILD
typedef struct _PERFECT_HASH_DIRECTORY_VTBL {
    DECLARE_COMPONENT_VTBL_HEADER(PERFECT_HASH_DIRECTORY);
    PPERFECT_HASH_DIRECTORY_OPEN Open;
    PPERFECT_HASH_DIRECTORY_CREATE Create;
    PPERFECT_HASH_DIRECTORY_GET_FLAGS GetFlags;
    PPERFECT_HASH_DIRECTORY_GET_PATH GetPath;
} PERFECT_HASH_DIRECTORY_VTBL;
typedef PERFECT_HASH_DIRECTORY_VTBL *PPERFECT_HASH_DIRECTORY_VTBL;

typedef struct _PERFECT_HASH_DIRECTORY {
    PPERFECT_HASH_DIRECTORY_VTBL Vtbl;
} PERFECT_HASH_DIRECTORY;
typedef PERFECT_HASH_DIRECTORY *PPERFECT_HASH_DIRECTORY;
#endif

//
// Define the PERFECT_HASH_FILE interface.
//

DECLARE_COMPONENT(File, PERFECT_HASH_FILE);

typedef union _PERFECT_HASH_FILE_LOAD_FLAGS {

    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, tries to allocate the file buffer using large pages.  The
        // caller is responsible for ensuring the process can create large pages
        // first by enabling the lock memory privilege.  If large pages can't be
        // allocated (because the lock memory privilege hasn't been enabled, or
        // there are insufficient large pages available to the system), the file
        // will be accessed via the normal memory-mapped address of the
        // underlying file.
        //
        // To determine whether or not the large page allocation was successful,
        // check the UsesLargePages bit of the PERFECT_HASH_FILE_FLAGS enum
        // enum (the flags can be obtained via the GetFlags() vtbl function).
        //

        ULONG TryLargePagesForFileData:1;

        //
        // Unused bits.
        //

        ULONG Unused:31;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_FILE_LOAD_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_FILE_LOAD_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_FILE_LOAD_FLAGS *PPERFECT_HASH_FILE_LOAD_FLAGS;

FORCEINLINE
HRESULT
IsValidFileLoadFlags(
    _In_ PPERFECT_HASH_FILE_LOAD_FLAGS FileLoadFlags
    )
{

    if (!ARGUMENT_PRESENT(FileLoadFlags)) {
        return E_POINTER;
    }

    if (FileLoadFlags->Unused != 0) {
        return E_FAIL;
    }

    return S_OK;
}

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_FILE_LOAD)(
    _In_ PPERFECT_HASH_FILE File,
    _In_ PPERFECT_HASH_PATH SourcePath,
    _Out_opt_ PLARGE_INTEGER EndOfFile,
    _In_opt_ PPERFECT_HASH_FILE_LOAD_FLAGS FileLoadFlags
    );
typedef PERFECT_HASH_FILE_LOAD *PPERFECT_HASH_FILE_LOAD;

//
// Define Create() method and supporting flags.
//

typedef union _PERFECT_HASH_FILE_CREATE_FLAGS {

    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, tries to allocate the file buffer using large pages.  The
        // caller is responsible for ensuring the process can create large pages
        // first by enabling the lock memory privilege.  If large pages can't be
        // allocated (because the lock memory privilege hasn't been enabled, or
        // there are insufficient large pages available to the system), the file
        // will be accessed via the normal memory-mapped address of the
        // underlying file.
        //
        // To determine whether or not the large page allocation was successful,
        // check the UsesLargePages bit of the PERFECT_HASH_FILE_FLAGS enum
        // enum (the flags can be obtained via the GetFlags() vtbl function).
        //

        ULONG TryLargePagesForFileData:1;

        //
        // Normally, when files are opened via Create(), the file will be
        // truncated if it already exists.  When this bit is set, no truncation
        // will be performed.  Unless EndOfFileIsExtensionSizeIfFileExists bit
        // is set, the EndOfFile parameter passed to Create() will be ignored
        // on input (it will still receive the final mapping size used).
        //

        ULONG NoTruncate:1;

        //
        // Used in conjunction with NoTruncate, when set, indicates that the
        // EndOfFile parameter should be treated as the number of bytes to
        // extend the file if it already exists (aligned up to allocation
        // granularity boundaries as necessary).  Ignored if NoTruncate is
        // not also set.
        //

        ULONG EndOfFileIsExtensionSizeIfFileExists:1;

        //
        // Unused bits.
        //

        ULONG Unused:29;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_FILE_CREATE_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_FILE_CREATE_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_FILE_CREATE_FLAGS *PPERFECT_HASH_FILE_CREATE_FLAGS;

FORCEINLINE
HRESULT
IsValidFileCreateFlags(
    _In_ PPERFECT_HASH_FILE_CREATE_FLAGS FileCreateFlags
    )
{

    if (!ARGUMENT_PRESENT(FileCreateFlags)) {
        return E_POINTER;
    }

    if (FileCreateFlags->Unused != 0) {
        return E_FAIL;
    }

    return S_OK;
}

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_FILE_CREATE)(
    _In_ PPERFECT_HASH_FILE File,
    _In_ PPERFECT_HASH_PATH SourcePath,
    _Inout_ PLARGE_INTEGER EndOfFile,
    _In_opt_ PPERFECT_HASH_DIRECTORY ParentDirectory,
    _In_opt_ PPERFECT_HASH_FILE_CREATE_FLAGS FileCreateFlags
    );
typedef PERFECT_HASH_FILE_CREATE *PPERFECT_HASH_FILE_CREATE;

//
// Define the PERFECT_HASH_FILE_FLAGS structure.
//

typedef union _PERFECT_HASH_FILE_FLAGS {
    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, indicates the file was initialized via Load().
        //
        // Invariant:
        //
        //      If Loaded == TRUE:
        //          Assert Created == FALSE
        //

        ULONG Loaded:1;

        //
        // When set, indicates the file was initialized via Create().
        //
        // Invariant:
        //
        //      If Created == TRUE:
        //          Assert Loaded == FALSE
        //

        ULONG Created:1;

        //
        // When set, indicates the caller did not set the relevant bit for
        // trying large pages in the create or load flags.
        //

        ULONG DoesNotWantLargePages:1;

        //
        // When set, indicates the file data resides in a memory allocation
        // backed by large pages.  In this case, BaseAddress represents the
        // large page address, and MappedAddress represents the original
        // address the file was mapped at.
        //

        ULONG UsesLargePages:1;

        //
        // Unused bits.
        //

        ULONG Unused:28;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_FILE_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_FILE_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_FILE_FLAGS *PPERFECT_HASH_FILE_FLAGS;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_FILE_GET_FLAGS)(
    _In_ PPERFECT_HASH_FILE File,
    _In_ ULONG SizeOfFlags,
    _Out_writes_bytes_(SizeOfFlags) PPERFECT_HASH_FILE_FLAGS Flags
    );
typedef PERFECT_HASH_FILE_GET_FLAGS *PPERFECT_HASH_FILE_GET_FLAGS;

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_FILE_GET_PATH)(
    _In_ PPERFECT_HASH_FILE File,
    _Inout_opt_ PPERFECT_HASH_PATH *Path
    );
typedef PERFECT_HASH_FILE_GET_PATH *PPERFECT_HASH_FILE_GET_PATH;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_FILE_GET_RESOURCES)(
    _In_ PPERFECT_HASH_FILE File,
    _Out_opt_ PHANDLE FileHandle,
    _Out_opt_ PHANDLE MappingHandle,
    _Out_opt_ PVOID *BaseAddress,
    _Out_opt_ PVOID *MappedAddress,
    _Out_opt_ PLARGE_INTEGER EndOfFile
    );
typedef PERFECT_HASH_FILE_GET_RESOURCES
      *PPERFECT_HASH_FILE_GET_RESOURCES;

#ifndef _PERFECT_HASH_INTERNAL_BUILD
typedef struct _PERFECT_HASH_FILE_VTBL {
    DECLARE_COMPONENT_VTBL_HEADER(PERFECT_HASH_FILE);
    PPERFECT_HASH_FILE_LOAD Load;
    PPERFECT_HASH_FILE_CREATE Create;
    PPERFECT_HASH_FILE_GET_FLAGS GetFlags;
    PPERFECT_HASH_FILE_GET_PATH GetPath;
    PPERFECT_HASH_FILE_GET_RESOURCES GetResources;
} PERFECT_HASH_FILE_VTBL;
typedef PERFECT_HASH_FILE_VTBL *PPERFECT_HASH_FILE_VTBL;

typedef struct _PERFECT_HASH_FILE {
    PPERFECT_HASH_FILE_VTBL Vtbl;
} PERFECT_HASH_FILE;
typedef PERFECT_HASH_FILE *PPERFECT_HASH_FILE;
#endif


//
// Define the PERFECT_HASH_KEYS interface and supporting structures.
//

typedef struct _VALUE_ARRAY {
    _Writable_elements_(NumberOfValues)
    PULONG Values;
    ULONG NumberOfValues;
    ULONG ValueSizeInBytes;
} VALUE_ARRAY;
typedef VALUE_ARRAY *PVALUE_ARRAY;
typedef const VALUE_ARRAY *PCVALUE_ARRAY;

typedef VALUE_ARRAY KEYS_SUBSET;
typedef KEYS_SUBSET *PKEYS_SUBSET;
typedef const KEYS_SUBSET *PCKEYS_SUBSET;

typedef union _PERFECT_HASH_KEYS_BITMAP_FLAGS {

    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, indicates the bitmap for the keys contains a single
        // contiguous run of set bits.  That is, between the highest set
        // bit and lowest set bit, all intermediate bits are set.  E.g.:
        //
        //  Contiguous:     00000000000111111111111111110000
        //  Not Contiguous: 00110000000111111111111111110000
        //

        ULONG Contiguous:1;

        //
        // When set, indicates that the key set contains a zero.
        //

        ULONG HasZero:1;

        //
        // Unused bits.
        //

        ULONG Unused:29;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_KEYS_BITMAP_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_KEYS_BITMAP_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_KEYS_BITMAP_FLAGS *PPERFECT_HASH_KEYS_BITMAP_FLAGS;

typedef struct _PERFECT_HASH_KEYS_BITMAP {

    PERFECT_HASH_KEYS_BITMAP_FLAGS Flags;

    BYTE LongestRunLength;
    BYTE LongestRunStart;
    BYTE TrailingZeros;
    BYTE LeadingZeros;

    ULONGLONG Bitmap;
    ULONGLONG ShiftedMask;

    CHAR String[64];

} PERFECT_HASH_KEYS_BITMAP;
typedef PERFECT_HASH_KEYS_BITMAP *PPERFECT_HASH_KEYS_BITMAP;

DECLARE_COMPONENT(Keys, PERFECT_HASH_KEYS);

typedef union _PERFECT_HASH_KEYS_LOAD_FLAGS {

    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, tries to allocate the keys buffer using large pages.  The
        // caller is responsible for ensuring the process can create large pages
        // first by enabling the lock memory privilege.  If large pages can't be
        // allocated (because the lock memory privilege hasn't been enabled, or
        // there are insufficient large pages available to the system), the keys
        // will be accessed via the normal memory-mapped address of the
        // underlying file.
        //
        // To determine whether or not the large page allocation was successful,
        // check the KeysDataUsesLargePages bit of the PERFECT_HASH_KEYS_FLAGS
        // enum (the flags can be obtained after loading via the GetFlags() vtbl
        // function).
        //

        ULONG TryLargePagesForKeysData:1;

        //
        // When set, skips the verification of keys during loading.
        // Specifically, skips enumerating all keys and verifying that the
        // keys are sorted, as well as constructing the keys bitmap.
        //

        ULONG SkipKeysVerification:1;

        //
        // When loading keys that are 64-bit (8 bytes), a bitmap is kept that
        // tracks whether or not a given bit was seen across the entire key set.
        // After enumerating the set, the number of zeros (bit not set) in the
        // bitmap are counted; if this number is less than or equal to 32, it
        // means that the entire key set can be compressed into 32-bit values
        // with some parallel bit extraction logic (i.e. _pext_u64()).  As this
        // has beneficial size and performance implications, when detected, the
        // key load operation will implicitly heap-allocate another array and
        // convert all the 64-bit keys into their unique 32-bit equivalent.
        // When set, this flag disables that behavior.
        //

        ULONG DisableImplicitKeyDownsizing:1;

        //
        // When set, attempts to infer the key size, in bits, from the last
        // digits in the .keys file name.  By default, the default key size is
        // assumed to be 32-bit (4 bytes; ULONG); when this flag is present, if
        // 64 appears prior to the final period preceding the file name (e.g.
        // "foo64.keys"), it has the same effect as specifying the table create
        // parameter --KeySizeInBytes=8.
        //
        // This flag is most useful when using the bulk-create command against
        // a directory of key files that have differing key sizes.
        //

        ULONG TryInferKeySizeFromKeysFilename:1;

        //
        // Unused bits.
        //

        ULONG Unused:28;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_KEYS_LOAD_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_KEYS_LOAD_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_KEYS_LOAD_FLAGS *PPERFECT_HASH_KEYS_LOAD_FLAGS;

FORCEINLINE
HRESULT
IsValidKeysLoadFlags(
    _In_ PPERFECT_HASH_KEYS_LOAD_FLAGS LoadFlags
    )
{
    if (!ARGUMENT_PRESENT(LoadFlags)) {
        return E_POINTER;
    }

    if (LoadFlags->Unused != 0) {
        return E_FAIL;
    }

    return S_OK;
}

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_KEYS_LOAD)(
    _In_ PPERFECT_HASH_KEYS Keys,
    _In_opt_ PPERFECT_HASH_KEYS_LOAD_FLAGS LoadFlags,
    _In_ PCUNICODE_STRING Path,
    _In_ ULONG KeySizeInBytes
    );
typedef PERFECT_HASH_KEYS_LOAD *PPERFECT_HASH_KEYS_LOAD;

//
// Define the keys flags union and get flags function.
//

typedef union _PERFECT_HASH_KEYS_FLAGS {
    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, indicates the keys were mapped using large pages.
        //

        ULONG KeysDataUsesLargePages:1;

        //
        // When set, indicates the keys are a sequential linear array of
        // values.
        //

        ULONG Linear:1;

        //
        // When set, indicates that key downsizing has occurred.  See the
        // DisableImplicitKeyDownsizing comment in the key load flags for more
        // information.
        //

        ULONG DownsizingOccurred:1;

        //
        // Unused bits.
        //

        ULONG Unused:29;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_KEYS_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_KEYS_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_KEYS_FLAGS *PPERFECT_HASH_KEYS_FLAGS;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_KEYS_GET_FLAGS)(
    _In_ PPERFECT_HASH_KEYS Keys,
    _In_ ULONG SizeOfFlags,
    _Out_writes_bytes_(SizeOfFlags) PPERFECT_HASH_KEYS_FLAGS Flags
    );
typedef PERFECT_HASH_KEYS_GET_FLAGS *PPERFECT_HASH_KEYS_GET_FLAGS;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_KEYS_GET_ADDRESS)(
    _In_ PPERFECT_HASH_KEYS Keys,
    _Out_ PVOID *BaseAddress,
    _Out_ PULARGE_INTEGER NumberOfElements
    );
typedef PERFECT_HASH_KEYS_GET_ADDRESS *PPERFECT_HASH_KEYS_GET_ADDRESS;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_KEYS_GET_BITMAP)(
    _In_ PPERFECT_HASH_KEYS Keys,
    _In_ ULONG SizeOfBitmap,
    _Out_writes_bytes_(SizeOfBitmap) PPERFECT_HASH_KEYS_BITMAP Bitmap
    );
typedef PERFECT_HASH_KEYS_GET_BITMAP *PPERFECT_HASH_KEYS_GET_BITMAP;

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_KEYS_GET_FILE)(
    _In_ PPERFECT_HASH_KEYS Keys,
    _Inout_ PPERFECT_HASH_FILE *File
    );
typedef PERFECT_HASH_KEYS_GET_FILE *PPERFECT_HASH_KEYS_GET_FILE;

typedef struct _PERFECT_HASH_KEYS_VTBL {
    DECLARE_COMPONENT_VTBL_HEADER(PERFECT_HASH_KEYS);
    PPERFECT_HASH_KEYS_LOAD Load;
    PPERFECT_HASH_KEYS_GET_FLAGS GetFlags;
    PPERFECT_HASH_KEYS_GET_ADDRESS GetAddress;
    PPERFECT_HASH_KEYS_GET_BITMAP GetBitmap;
    PPERFECT_HASH_KEYS_GET_FILE GetFile;
} PERFECT_HASH_KEYS_VTBL;
typedef PERFECT_HASH_KEYS_VTBL *PPERFECT_HASH_KEYS_VTBL;

#ifndef _PERFECT_HASH_INTERNAL_BUILD
typedef struct _PERFECT_HASH_KEYS {
    PPERFECT_HASH_KEYS_VTBL Vtbl;
} PERFECT_HASH_KEYS;
typedef PERFECT_HASH_KEYS *PPERFECT_HASH_KEYS;
#endif

//
// Define an X-macro for algorithm IDs.  The ENTRY macro receives the following
// parameters: (Name, Upper).
//

#define PERFECT_HASH_ALGORITHM_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    FIRST_ENTRY(Chm01, CHM01)                                        \
    LAST_ENTRY(Chm02, CHM02)

#define PERFECT_HASH_ALGORITHM_TABLE_ENTRY(ENTRY) \
    PERFECT_HASH_ALGORITHM_TABLE(ENTRY, ENTRY, ENTRY)

#define EXPAND_AS_ALGORITHM_ENUM(Name, Upper) \
    PerfectHash##Name##AlgorithmId,

typedef enum _PERFECT_HASH_ALGORITHM_ID {
    PerfectHashNullAlgorithmId = 0,
    PERFECT_HASH_ALGORITHM_TABLE_ENTRY(EXPAND_AS_ALGORITHM_ENUM)
    PerfectHashInvalidAlgorithmId,
} PERFECT_HASH_ALGORITHM_ID;
typedef PERFECT_HASH_ALGORITHM_ID *PPERFECT_HASH_ALGORITHM_ID;

//
// Provide a simple inline algorithm validation routine.
//

FORCEINLINE
BOOLEAN
IsValidPerfectHashAlgorithmId(
    _In_ PERFECT_HASH_ALGORITHM_ID AlgorithmId
    )
{
    return (
        AlgorithmId > PerfectHashNullAlgorithmId &&
        AlgorithmId < PerfectHashInvalidAlgorithmId
    );
}

//
// Define a seed masks structure.  The number of elements must match the maximum
// number of seeds used by all hash functions (currently 8).
//

#define MAX_NUMBER_OF_SEEDS 8

typedef struct _SEED_MASKS {
    LONG Mask1;
    LONG Mask2;
    LONG Mask3;
    LONG Mask4;
    LONG Mask5;
    LONG Mask6;
    LONG Mask7;
    LONG Mask8;
} SEED_MASKS;
typedef SEED_MASKS *PSEED_MASKS;
typedef const SEED_MASKS *PCSEED_MASKS;

FORCEINLINE
BOOLEAN
IsValidSeedMasks(
    _In_ PCSEED_MASKS Masks
    )
{
    return (Masks->Mask1 != -1);
}

//
// Define an X-macro for hash functions.  The ENTRY macros receive the following
// parameters: (Name, NumberOfSeeds, SeedMasks).
//

#define NO_SEED_MASKS { -1, }

#define DECL_SEED_MASKS(m1, m2, m3, m4, m5, m6, m7, m8) \
    { m1, m2, m3, m4, m5, m6, m7, m8 }

#define PERFECT_HASH_HASH_FUNCTION_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    FIRST_ENTRY(Crc32Rotate15, 2, NO_SEED_MASKS)                         \
    ENTRY(Jenkins, 2, NO_SEED_MASKS)                                     \
    ENTRY(JenkinsMod, 2, NO_SEED_MASKS)                                  \
    ENTRY(RotateXor, 4, NO_SEED_MASKS)                                   \
    ENTRY(AddSubXor, 4, NO_SEED_MASKS)                                   \
    ENTRY(Xor, 2, NO_SEED_MASKS)                                         \
    ENTRY(Dummy, 3, NO_SEED_MASKS)                                       \
    ENTRY(Crc32RotateXor, 3, NO_SEED_MASKS)                              \
    ENTRY(Crc32, 2, NO_SEED_MASKS)                                       \
    ENTRY(Djb, 2, NO_SEED_MASKS)                                         \
    ENTRY(DjbXor, 2, NO_SEED_MASKS)                                      \
    ENTRY(Fnv, 2, NO_SEED_MASKS)                                         \
    ENTRY(Crc32Not, 2, NO_SEED_MASKS)                                    \
    ENTRY(                                                               \
        Crc32RotateX,                                                    \
        3,                                                               \
        DECL_SEED_MASKS(0, 0, 0x1f, 0, 0, 0, 0, 0)                       \
    )                                                                    \
    ENTRY(                                                               \
        Crc32RotateXY,                                                   \
        3,                                                               \
        DECL_SEED_MASKS(0, 0, 0x1f1f, 0, 0, 0, 0, 0)                     \
    )                                                                    \
    ENTRY(                                                               \
        Crc32RotateWXYZ,                                                 \
        3,                                                               \
        DECL_SEED_MASKS(0, 0, 0x1f1f1f1f, 0, 0, 0, 0, 0)                 \
    )                                                                    \
    ENTRY(                                                               \
        RotateMultiplyXorRotate,                                         \
        3,                                                               \
        DECL_SEED_MASKS(0, 0, 0x1f1f1f1f, 0, 0, 0, 0, 0)                 \
    )                                                                    \
    ENTRY(                                                               \
        ShiftMultiplyXorShift,                                           \
        3,                                                               \
        DECL_SEED_MASKS(0, 0, 0x1f1f1f1f, 0, 0, 0, 0, 0)                 \
    )                                                                    \
    ENTRY(                                                               \
        ShiftMultiplyXorShift2,                                          \
        6,                                                               \
        DECL_SEED_MASKS(0, 0, 0x1f1f1f, 0, 0, 0x1f1f1f, 0, 0)            \
    )                                                                    \
    ENTRY(                                                               \
        RotateMultiplyXorRotate2,                                        \
        6,                                                               \
        DECL_SEED_MASKS(0, 0, 0x1f1f1f, 0, 0, 0x1f1f1f, 0, 0)            \
    )                                                                    \
    ENTRY(                                                               \
        MultiplyRotateR,                                                 \
        3,                                                               \
        DECL_SEED_MASKS(0, 0, 0x1f1f, 0, 0, 0, 0, 0)                     \
    )                                                                    \
    ENTRY(                                                               \
        MultiplyRotateLR,                                                \
        3,                                                               \
        DECL_SEED_MASKS(0, 0, 0x1f1f, 0, 0, 0, 0, 0)                     \
    )                                                                    \
    ENTRY(                                                               \
        MultiplyShiftR,                                                  \
        3,                                                               \
        DECL_SEED_MASKS(0, 0, 0x1f1f, 0, 0, 0, 0, 0)                     \
    )                                                                    \
    ENTRY(                                                               \
        MultiplyShiftLR,                                                 \
        3,                                                               \
        DECL_SEED_MASKS(0, 0, 0x1f1f, 0, 0, 0, 0, 0)                     \
    )                                                                    \
    ENTRY(Multiply, 2, NO_SEED_MASKS)                                    \
    ENTRY(MultiplyXor, 4, NO_SEED_MASKS)                                 \
    ENTRY(                                                               \
        MultiplyRotateRMultiply,                                         \
        5,                                                               \
        DECL_SEED_MASKS(0, 0, 0x1f1f, 0, 0, 0, 0, 0)                     \
    )                                                                    \
    ENTRY(                                                               \
        MultiplyRotateR2,                                                \
        5,                                                               \
        DECL_SEED_MASKS(0, 0, 0x1f1f1f1f, 0, 0, 0, 0, 0)                 \
    )                                                                    \
    ENTRY(                                                               \
        MultiplyShiftRMultiply,                                          \
        5,                                                               \
        DECL_SEED_MASKS(0, 0, 0x1f1f, 0, 0, 0, 0, 0)                     \
    )                                                                    \
    ENTRY(                                                               \
        MultiplyShiftR2,                                                 \
        5,                                                               \
        DECL_SEED_MASKS(0, 0, 0x1f1f1f1f, 0, 0, 0, 0, 0)                 \
    )                                                                    \
    ENTRY(                                                               \
        RotateRMultiply,                                                 \
        3,                                                               \
        DECL_SEED_MASKS(0, 0, 0x1f1f, 0, 0, 0, 0, 0)                     \
    )                                                                    \
    ENTRY(                                                               \
        RotateRMultiplyRotateR,                                          \
        3,                                                               \
        DECL_SEED_MASKS(0, 0, 0x1f1f1f1f, 0, 0, 0, 0, 0)                 \
    )                                                                    \
    ENTRY(                                                               \
        Multiply643ShiftR,                                               \
        5,                                                               \
        DECL_SEED_MASKS(0, 0, 0x1f1f, 0, 0, 0, 0, 0)                     \
    )                                                                    \
    ENTRY(                                                               \
        Multiply644ShiftR,                                               \
        5,                                                               \
        DECL_SEED_MASKS(0, 0, 0x3f3f, 0, 0, 0, 0, 0)                     \
    )                                                                    \
    ENTRY(                                                               \
        MultiplyShiftRX,                                                 \
        3,                                                               \
        DECL_SEED_MASKS(0, 0, 0x1f1f, 0, 0, 0, 0, 0)                     \
    )                                                                    \
    LAST_ENTRY(Scratch, 8, NO_SEED_MASKS)

#define PERFECT_HASH_HASH_FUNCTION_TABLE_ENTRY(ENTRY) \
    PERFECT_HASH_HASH_FUNCTION_TABLE(ENTRY, ENTRY, ENTRY)

#define EXPAND_AS_HASH_FUNCTION_ENUM(Name, NumberOfSeeds, SeedMasks) \
    PerfectHashHash##Name##FunctionId,

//
// Define an enumeration for identifying which hash function variant to use.
//

typedef enum _PERFECT_HASH_HASH_FUNCTION_ID {

    //
    // Explicitly define a null algorithm to take the 0-index slot.
    // This makes enum validation easier.
    //

    PerfectHashNullHashFunctionId               = 0,

    //
    // Begin valid hash functions.
    //

    PERFECT_HASH_HASH_FUNCTION_TABLE_ENTRY(EXPAND_AS_HASH_FUNCTION_ENUM)

    //
    // End valid hash functions.
    //

    //
    // N.B. Keep the next value last.
    //

    PerfectHashInvalidHashFunctionId,

} PERFECT_HASH_HASH_FUNCTION_ID;
typedef PERFECT_HASH_HASH_FUNCTION_ID *PPERFECT_HASH_HASH_FUNCTION_ID;

//
// Provide a simple inline hash function validation routine.
//

FORCEINLINE
BOOLEAN
IsValidPerfectHashHashFunctionId(
    _In_ PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId
    )
{
    return (
        HashFunctionId > PerfectHashNullHashFunctionId &&
        HashFunctionId < PerfectHashInvalidHashFunctionId
    );
}

//
// The final 'AND' of a vertex value by a hash mask can be avoided for hash
// functions that right-shift their final value by "32 minus the number
// of trailing zeros of the number of vertices".  This yields slightly faster
// final compiled routines, as it avoids the need to do additional AND mask
// operations.
//
// E.g. HologramWorld-31016.keys has 32768 edges and 65536 vertices.  65536
//      has 16 trailing zeros, thus, if the final operation of a hash function
//      is the right-shift the final vertex value by 16 (32 - 16 = 16), there
//      is no need to then also do a logical AND 0xffff operation.  512 keys
//      would result in a final right-shift of 23 (1 << 9 = 512; 32 - 9 = 23).
//

FORCEINLINE
BOOLEAN
IsAndHashMaskRequired(
    _In_ PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId
    )
{
    return !(
        HashFunctionId == PerfectHashHashMultiplyShiftRXFunctionId
    );
}

//
// Define an enum for capturing the subset of hash functions that are unusable.
// If you want to write your own hash function, feel free to repurpose one of
// these instead of adding a new one.
//
// N.B. "Unusable" is any hash function that either a) can't find a solution
//      for any keys file in the perfecthash-keys/sys32 directory without
//      resorting to a table resize, or b) takes orders of magnitude longer
//      than MultiplyShiftR (the slowest-solving algorithm) when processing
//
// N.B. This enum isn't used anywhere in the C code per se.  It is included
//      in this header in order to generate a symbol that can be identified
//      by the perfecthash.sourcefile.PerfectHashPdbexHeaderFile() class
//      which uses it to skip these hash functions in the hash_functions()
//      routine.
//

typedef enum _PERFECT_HASH_DISABLED_HASH_FUNCTION_ID {
    //
    // Keep this first.
    //

    PerfectHashDisabledNullFunctionId,

    //
    // Poor hash functions that can't solve sys32 keys such as
    // HologramWorld-31016.keys.
    //

    PerfectHashDisabledHashJenkinsModFunctionId,
    PerfectHashDisabledHashDummyFunctionId,
    PerfectHashDisabledHashScratchFunctionId,
    PerfectHashDisabledHashXorFunctionId,
    PerfectHashDisabledHashAddSubXorFunctionId,
    PerfectHashDisabledHashCrc32RotateXYFunctionId,
    PerfectHashDisabledHashCrc32RotateWXYZFunctionId,
    PerfectHashDisabledHashRotateXorFunctionId,
    PerfectHashDisabledHashCrc32FunctionId,
    PerfectHashDisabledHashDjbFunctionId,
    PerfectHashDisabledHashDjbXorFunctionId,
    PerfectHashDisabledHashCrc32Rotate15FunctionId,
    PerfectHashDisabledHashCrc32RotateXorFunctionId,
    PerfectHashDisabledHashCrc32NotFunctionId,
    PerfectHashDisabledHashCrc32RotateXFunctionId,
    PerfectHashDisabledHashMultiplyFunctionId,
    PerfectHashDisabledHashMultiplyXorFunctionId,
    PerfectHashDisabledHashMultiplyShiftLRFunctionId,
    PerfectHashDisabledHashRotateRMultiplyFunctionId,
    PerfectHashDisabledHashFnvFunctionId,

    //
    // Hash functions that can solve the sys32-2k+ keys, but don't do so within
    // acceptable time limits (i.e. orders of magnitude slower than our slowest
    // working function MultiplyShiftR).
    //

    PerfectHashDisabledHashMultiplyRotateRMultiplyFunctionId,
    PerfectHashDisabledHashMultiplyShiftRMultiplyFunctionId,

    //
    // Keep this last.
    //

    PerfectHashDisabledInvalidFunctionId,
} PERFECT_HASH_DISABLED_HASH_FUNCTION_ID;


//
// Define the seed mask counts structure, which inherits from VALUE_ARRAY.
//

typedef struct _SEED_MASK_COUNTS {

    //
    // Begin VALUE_ARRAY.
    //

    _Writable_elements_(NumberOfValues)
    PULONG Values;
    ULONG NumberOfValues;
    ULONG ValueSizeInBytes;

    //
    // End VALUE_ARRAY.  SEED_MASK_COUNTS-specific fields start here.
    //

    _Field_range_(1, 8) BYTE SeedNumber;
    _Field_range_(1, 4) BYTE ByteNumber;

    USHORT Padding1;

    //
    // Total sum of all counts in the values array.
    //

    ULONG Total;

    //
    // Cumulative sum of counts in the values array.
    //

    ULONG Cumulative[32];

    //
    // String representation of the counts with spaces instead of commas.  The
    // struct owns the underlying Buffer pointer and must free it during param
    // deallocation.
    //

    UNICODE_STRING CountsString;

} SEED_MASK_COUNTS;
typedef SEED_MASK_COUNTS *PSEED_MASK_COUNTS;
typedef const SEED_MASK_COUNTS *PCSEED_MASK_COUNTS;

//
// Define an enumeration for identifying the type of table masking used by the
// underlying perfect hash table.  This has performance and size implications.
// Modulus masking typically results in smaller tables at the expenses of slower
// modulus-based hash functions.  Non-modulus masking requires power-of-2 sized
// tables, which will be larger, but the resulting mask function can be done
// by logical AND instructions, which are fast.
//
// N.B. Modulus masking does not work.
//

#define PERFECT_HASH_MASK_FUNCTION_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    FIRST_ENTRY(Modulus, MODULUS)                                        \
    LAST_ENTRY(And, AND)

#define PERFECT_HASH_MASK_FUNCTION_TABLE_ENTRY(ENTRY) \
    PERFECT_HASH_MASK_FUNCTION_TABLE(ENTRY, ENTRY, ENTRY)

#define EXPAND_AS_MASK_FUNCTION_ENUM(Name, NumberOfSeeds) \
    PerfectHash##Name##MaskFunctionId,

typedef enum _PERFECT_HASH_MASK_FUNCTION_ID {

    //
    // Null masking type.
    //

    PerfectHashNullMaskFunctionId          = 0,

    PERFECT_HASH_MASK_FUNCTION_TABLE_ENTRY(EXPAND_AS_MASK_FUNCTION_ENUM)

    //
    // N.B. Keep the next value last.
    //

    PerfectHashInvalidMaskFunctionId,


} PERFECT_HASH_MASK_FUNCTION_ID;
typedef PERFECT_HASH_MASK_FUNCTION_ID
      *PPERFECT_HASH_MASK_FUNCTION_ID;

//
// Provide a simple inline masking type validation routine.
//

FORCEINLINE
BOOLEAN
IsValidPerfectHashMaskFunctionId(
    _In_ PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId
    )
{
    return (
        MaskFunctionId > PerfectHashNullMaskFunctionId &&
        MaskFunctionId < PerfectHashInvalidMaskFunctionId
    );
}

//
// Masking tends to fall into one of two buckets: modulus and not-modulus.
// Provide an inline routine that guarantees to match all current and future
// modulus masking function IDs.
//

FORCEINLINE
BOOLEAN
IsModulusMasking(
    _In_ PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId
    )
{
    return MaskFunctionId == PerfectHashModulusMaskFunctionId;
}

//
// Define the PERFECT_HASH_CONTEXT interface.  This interface is responsible for
// encapsulating threadpool resources and allows perfect hash table solutions to
// be found in parallel.  An instance of this interface must be provided to the
// PERFECT_HASH_TABLE interface's creation routine.
//

DECLARE_COMPONENT(Context, PERFECT_HASH_CONTEXT);

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_CONTEXT_SET_MAXIMUM_CONCURRENCY)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ ULONG MaximumConcurrency
    );
typedef PERFECT_HASH_CONTEXT_SET_MAXIMUM_CONCURRENCY
      *PPERFECT_HASH_CONTEXT_SET_MAXIMUM_CONCURRENCY;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_CONTEXT_GET_MAXIMUM_CONCURRENCY)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _Out_ PULONG MaximumConcurrency
    );
typedef PERFECT_HASH_CONTEXT_GET_MAXIMUM_CONCURRENCY
      *PPERFECT_HASH_CONTEXT_GET_MAXIMUM_CONCURRENCY;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_CONTEXT_SET_BASE_OUTPUT_DIRECTORY)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PCUNICODE_STRING BaseOutputDirectory
    );
typedef PERFECT_HASH_CONTEXT_SET_BASE_OUTPUT_DIRECTORY
      *PPERFECT_HASH_CONTEXT_SET_BASE_OUTPUT_DIRECTORY;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_CONTEXT_GET_BASE_OUTPUT_DIRECTORY)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _Inout_ PPERFECT_HASH_DIRECTORY *BaseOutputDirectory
    );
typedef PERFECT_HASH_CONTEXT_GET_BASE_OUTPUT_DIRECTORY
      *PPERFECT_HASH_CONTEXT_GET_BASE_OUTPUT_DIRECTORY;

//
// Define the context bulk-create flags.
//

typedef union _PERFECT_HASH_CONTEXT_BULK_CREATE_FLAGS {

    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // Normally, after a table has been successfully created, it is tested
        // (i.e. the table's Test() method is invoked).  When this bit is set,
        // testing is not performed.
        //

        ULONG SkipTestAfterCreate:1;

        //
        // When set, compiles each successfully-created table as part of the
        // bulk create operation (i.e. invokes the table's Compile() method).
        //
        // N.B. Requires msbuild.exe on the PATH; currently generates a cryptic
        //      error when this is not the case.
        //

        ULONG Compile:1;

        //
        // Unused bits.
        //

        ULONG Unused:30;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_CONTEXT_BULK_CREATE_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_CONTEXT_BULK_CREATE_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_CONTEXT_BULK_CREATE_FLAGS
      *PPERFECT_HASH_CONTEXT_BULK_CREATE_FLAGS;

FORCEINLINE
HRESULT
IsValidContextBulkCreateFlags(
    _In_ PPERFECT_HASH_CONTEXT_BULK_CREATE_FLAGS BulkCreateFlags
    )
{

    if (!ARGUMENT_PRESENT(BulkCreateFlags)) {
        return E_POINTER;
    }

    if (BulkCreateFlags->Unused != 0) {
        return E_FAIL;
    }

    return S_OK;
}

//
// Define the context table-create flags.
//

typedef union _PERFECT_HASH_CONTEXT_TABLE_CREATE_FLAGS {

    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // Normally, after a table has been successfully created, it is tested
        // (i.e. the table's Test() method is invoked).  When this bit is set,
        // testing is not performed.
        //

        ULONG SkipTestAfterCreate:1;

        //
        // When set, compiles the table after successful creation (i.e. invokes
        // the table's Compile() method).
        //
        // N.B. Requires msbuild.exe on the PATH; currently generates a cryptic
        //      error when this is not the case.
        //
        // N.B. Compilation can take anywhere from 5-15 seconds per table.
        //

        ULONG Compile:1;

        //
        // Unused bits.
        //

        ULONG Unused:30;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_CONTEXT_TABLE_CREATE_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_CONTEXT_TABLE_CREATE_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_CONTEXT_TABLE_CREATE_FLAGS
      *PPERFECT_HASH_CONTEXT_TABLE_CREATE_FLAGS;

FORCEINLINE
HRESULT
IsValidContextTableCreateFlags(
    _In_ PPERFECT_HASH_CONTEXT_TABLE_CREATE_FLAGS ContextTableCreateFlags
    )
{

    if (!ARGUMENT_PRESENT(ContextTableCreateFlags)) {
        return E_POINTER;
    }

    if (ContextTableCreateFlags->Unused != 0) {
        return E_FAIL;
    }

    return S_OK;
}

//
// Define the table create flags.
//

typedef union _PERFECT_HASH_TABLE_CREATE_FLAGS {

    struct _Struct_size_bytes_(sizeof(ULONGLONG)) {

        //
        // When set, disables the default "first solved graph wins" behavior
        // and enables the "find best memory coverage" behavior.  This must
        // be used in conjunction with create table parameters that specify
        // the number of attempts at finding a best solution, as well as a
        // predicate for evaluating what constitutes "best" (e.g. highest
        // number of empty cache lines in the final assigned array).  This
        // option is considerably more CPU intensive than the "first graph
        // wins" behavior, as the create table routine will not return until
        // it has attempted the requested number of solutions.
        //
        // N.B. "Best" is being used in the relative sense here, i.e. there
        //      are no guarantees that the resulting table *is* actually the
        //      best (best what?  best performing?); only that it was the
        //      table with the highest/lowest value for a given predicate.
        //
        // N.B. See ../src/PerfectHash/Graph.h for some more information about
        //      the best memory coverage behavior.
        //

        ULONGLONG FindBestGraph:1;

        //
        // When set, skips the internal graph verfication check that ensures a
        // valid perfect hash solution has been found (i.e. with no collisions
        // across the entire key set).  This means that perfect hash tables
        // could be created that subsequently fail their self-test via the
        // Table->Vtbl->Test() routine, or their compiled perfect hash table
        // test .exe file fails.
        //
        // N.B. Once a given algorithm, hash function, and masking type have
        //      been observed to generate valid solutions, it is safe to assume
        //      all future solutions found will be valid, assuming no changes to
        //      the parameters or underlying implementations.
        //
        // N.B. When modulus masking is being used, the graph verification logic
        //      correctly detects that invalid solutions are being generated.
        //

        ULONGLONG SkipGraphVerification:1;

        //
        // When set, indicates that the resulting table will not be used after
        // creation.  That is, none of the table's hash methods (e.g. Index(),
        // Lookup(), Insert() etc) will be used after the table's Create()
        // method returns.
        //
        // Setting this flag allows the table creation routine to omit various
        // buffer allocations and memory copies in order to get the table into
        // a state where it can be used as if Load() were called on it.
        //
        // If this flag is set, any attempts at calling the hash vtbl methods
        // like Index() etc will result in an access violation.
        //

        ULONGLONG CreateOnly:1;

        //
        // When set, tries to allocate the table data using large pages.  This
        // only applies to the post-create instance of the table, assuming the
        // CreateOnly flag is not set.
        //
        // Analogous to TryLargePagesForTableData flag in table load flags.
        //

        ULONGLONG TryLargePagesForTableData:1;

        //
        // When set, tries to allocate the values array using large pages.
        // This only applies to the post-create instance of the table, assuming
        // the CreateOnly flag is not set.
        //
        // Analogous to TryLargePagesForValuesArray flag in table load flags.
        //

        ULONGLONG TryLargePagesForValuesArray:1;

        //
        // When set, uses any previous table size information associated with
        // the keys file for the given combination of algorithm, hash function
        // and masking type.
        //

        ULONGLONG UsePreviousTableSize:1;

        //
        // When set, incorporates the number of table resize events encountered
        // whilst searching for a perfect hash solution into the final output
        // name.
        //

        ULONGLONG IncludeNumberOfTableResizeEventsInOutputPath:1;

        //
        // When set, incorporates the number of table elements (i.e. the size)
        // of the winning perfect hash solution into the final output name.
        //

        ULONGLONG IncludeNumberOfTableElementsInOutputPath:1;

        //
        // When set, disables all file work (I/O).  This will prevent generation
        // of the .pht1 file that can later be Load()'d, as well as all the
        // supporting source files for the compiled perfect hash table.
        //

        ULONGLONG NoFileIo:1;

        //
        // When set, does not print any console output related to table creation
        // (i.e. the normal dots, dashes etc., or any best graph output).
        //
        // N.B. Incompatible with flag Quiet.
        //

        ULONGLONG Silent:1;

        //
        // Enables redundant checks in the routine that determines whether or
        // not a generated graph is acyclic.  This shouldn't be necessary in
        // normal operation.  It may help identify bugs during development of
        // the graph solving logic, though.
        //

        ULONGLONG Paranoid:1;

        //
        // Skips calculating assigned memory coverage when in "first graph wins"
        // mode.
        //

        ULONGLONG SkipMemoryCoverageInFirstGraphWinsMode:1;

        //
        // When set, tries to allocate the edge and vertex arrays used by graphs
        // during solving using large pages.
        //

        ULONGLONG TryLargePagesForGraphEdgeAndVertexArrays:1;

        //
        // When set, tries to allocate the table data used by graphs during
        // solving using large pages.
        //

        ULONGLONG TryLargePagesForGraphTableData:1;

        //
        // When set, omits writing a row in the applicable .csv file if table
        // creation failed.
        //

        ULONGLONG OmitCsvRowIfTableCreateFailed:1;

        //
        // When set, omits writing a row in the applicable .csv file if table
        // creation succeeded.
        //

        ULONGLONG OmitCsvRowIfTableCreateSucceeded:1;

        //
        // When set, causes the C preprocessor macro CPH_INDEX_ONLY to be
        // defined, which has the effect of omitting the compiled perfect hash
        // routines that deal with the underlying table values array (i.e. any
        // routine other than Index(); Insert(), Lookup(), Delete(), etc), as
        // well as the array itself.  This will result in a size reduction of
        // the final compiled perfect hash binary.  It is intended to be used
        // if you only need the Index() routine and will be managing your own
        // table values independently.
        //

        ULONGLONG IndexOnly:1;

        //
        // When set, uses a shared read-write section for the table values
        // array in the compiled perfect hash table C files.  This will result
        // in the values array being visible across multiple processes.
        //
        // N.B. Has no effect if --IndexOnly is also specified.
        //

        ULONGLONG UseRwsSectionForTableValues:1;

        //
        // When set, uses implementations of RtlCopyPages and RtlFillPages that
        // use non-temporal hints.  Only applies when running on AMD64.  See
        // ../src/PerfectHash/RtlCopyPages_x64.asm and RtlFillPages_x64.asm for
        // more info.
        //

        ULONGLONG UseNonTemporalAvx2Routines:1;

        //
        // When set, disables writing the output .csv file.
        //

        ULONGLONG DisableCsvOutputFile:1;

        //
        // When set, clamps the number of edges to always be equal to the
        // number of keys, rounded up to a power of two, regardless of the
        // number of table resizes currently in effect.  Normally, when a table
        // is resized, the number of vertices are doubled, and the number of
        // edges are set to the number of vertices shifted right once (divided
        // by two).  When this flag is set, the vertex doubling stays the same,
        // however, the number of edges is always clamped to be equal to the
        // number of keys rounded up to a power of two.  This is a research
        // option used to evaluate the impact of the number of edges on the
        // graph solving probability for a given key set.  Only applies to
        // And masking (i.e. not modulus masking).
        //

        ULONGLONG ClampNumberOfEdges:1;

        //
        // When set, uses the original (slower) seeded hash routines (the ones
        // that return an HRESULT return code and write the hash value to an
        // output parameter) -- as opposed to using the newer, faster, "Ex"
        // version of the hash routines.
        //
        // N.B. This flag is incompatible with HashAllKeysFirst.
        //

        ULONGLONG UseOriginalSeededHashRoutines:1;

        //
        // When set, changes the graph solving logic such that vertices (i.e.
        // hash values) are generated for all keys up-front, prior to graph
        // construction.  (Experimental.)
        //
        // N.B. This flag is incompatible with UseOriginalSeededHashRoutines.
        //

        ULONGLONG HashAllKeysFirst:1;

        //
        // When set, allocates the memory for the vertex pairs array with
        // write-combine page protection.
        //
        // N.B. Only applies when HashAllKeysFirst is set.  Incompatible with
        //      TryLargePagesForVertexPairs.
        //

        ULONGLONG EnableWriteCombineForVertexPairs:1;

        //
        // When set, automatically changes the page protection of the vertex
        // pairs array (after successful hashing of all keys without any vertex
        // collisions) from PAGE_READWRITE|PAGE_WRITECOMBINE to PAGE_READONLY.
        //
        // N.B. Only applies when the flags EnableWriteCombineForVertexPairs
        //      and HashAllKeysFirst is set.
        //

        ULONGLONG RemoveWriteCombineAfterSuccessfulHashKeys:1;

        //
        // When set, tries to allocate the array for vertex pairs using large
        // pages.
        //
        // N.B. Only applies when HashAllKeysFirst is set.  Incompatible with
        //      EnableWriteCombineForVertexPairs.
        //

        ULONGLONG TryLargePagesForVertexPairs:1;

        //
        // When set, if a non-zero value is available in the table's predicted
        // attempts field (which will be the case if a solutions-found ratio has
        // been supplied), use it to limit the maximum concurrency used when
        // dispatching parallel graph solving attempts.
        //

        ULONGLONG TryUsePredictedAttemptsToLimitMaxConcurrency:1;

        //
        // When set, if uses a random seed obtained from the operating system to
        // initialize the selected RNG.  Requires --Rng.
        //

        ULONGLONG RngUseRandomStartSeed:1;

        //
        // When set, tries to use optimized AVX2 routines for hashing keys, if
        // applicable.
        //
        // N.B. Only applies when HashAllKeysFirst is set.
        //
        // N.B. Currently only implemented for the MultiplyShiftR hash function.
        //

        ULONGLONG TryUseAvx2HashFunction:1;

        //
        // When set, tries to use optimized AVX512 routines for hashing keys, if
        // applicable.
        //
        // N.B. Only applies when HashAllKeysFirst is set.
        //
        // N.B. Currently only implemented for the MultiplyShiftR hash function.
        //

        ULONGLONG TryUseAvx512HashFunction:1;

        //
        // When set, disables automatically using the AVX2 version of the
        // calculate memory coverage routine.
        //

        ULONGLONG DoNotTryUseAvx2MemoryCoverageFunction:1;

        //
        // When set, disables the best graph console output and only prints the
        // normal dots and dashes etc.
        //
        // N.B. Incompatible with flag Silent.
        //

        ULONGLONG Quiet:1;

        //
        // When set, includes the table keys in the generated compiled perfect
        // hash table DLL.
        //

        ULONGLONG IncludeKeysInCompiledDll:1;

        //
        // When set, disables saving the table values if function hooking is
        // active and the callback DLL has a TableValues export (which it will
        // if it's a perfect hash compiled DLL).
        //

        ULONGLONG DisableSavingCallbackTableValues:1;

        //
        // When set, disables any attempt at using the 16-bit hashing mechanisms
        // if the right conditions exist.
        //

        ULONGLONG DoNotTryUseHash16Impl:1;

        //
        // When set, always try to respect the kernel runtime limit (supplied
        // via --CuDevicesKernelRuntimeTargetInMilliseconds), even if the
        // device indicates it has no kernel runtime limit (i.e. is in TCC
        // mode).  Only applies to GPU solver graphs.
        //

        ULONGLONG AlwaysRespectCuKernelRuntimeLimit:1;

        //
        // Unused bits.
        //

        ULONGLONG Unused:28;
    };

    LONGLONG AsLongLong;
    ULONGLONG AsULongLong;
} PERFECT_HASH_TABLE_CREATE_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_TABLE_CREATE_FLAGS) == sizeof(ULONGLONG));
typedef PERFECT_HASH_TABLE_CREATE_FLAGS
      *PPERFECT_HASH_TABLE_CREATE_FLAGS;

FORCEINLINE
HRESULT
LoadDefaultTableCreateFlags(
    _In_ PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags
    )
{

    if (!ARGUMENT_PRESENT(TableCreateFlags)) {
        return E_POINTER;
    }

    TableCreateFlags->HashAllKeysFirst = TRUE;
    TableCreateFlags->TryUseAvx2HashFunction = TRUE;
    TableCreateFlags->IncludeKeysInCompiledDll = TRUE;
    TableCreateFlags->UseRwsSectionForTableValues = TRUE;

    return S_OK;
}

FORCEINLINE
HRESULT
IsValidTableCreateFlags(
    _In_ PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags
    )
{

    if (!ARGUMENT_PRESENT(TableCreateFlags)) {
        return PH_E_INVALID_TABLE_CREATE_FLAGS;
    }

    if (TableCreateFlags->Silent && TableCreateFlags->Quiet) {
        return PH_E_SILENT_INCOMPATIBLE_WITH_QUIET;
    }

    if (TableCreateFlags->UseOriginalSeededHashRoutines &&
        TableCreateFlags->HashAllKeysFirst) {
        return PH_E_HASH_ALL_KEYS_FIRST_INCOMPAT_WITH_ORIG_SEEDED_HASH_ROUTINES;
    }

    if (!TableCreateFlags->HashAllKeysFirst) {

        //
        // The flags related to vertex pairs are not applicable when we're not
        // hashing all keys up-front (as that's the only time when we use a
        // vertex pair array).
        //

        if (TableCreateFlags->TryLargePagesForVertexPairs ||
            TableCreateFlags->EnableWriteCombineForVertexPairs ||
            TableCreateFlags->RemoveWriteCombineAfterSuccessfulHashKeys) {
            return PH_E_VERTEX_PAIR_FLAGS_REQUIRE_HASH_ALL_KEYS_FIRST;
        }

        //
        // Likewise for the flag related to AVX hash function routines.
        //

        if (TableCreateFlags->TryUseAvx2HashFunction) {
            return PH_E_TRY_USE_AVX2_HASH_FUNC_FLAG_REQUIRE_HASH_ALL_KEYS_FIRST;
        }

        if (TableCreateFlags->TryUseAvx2HashFunction) {
            return PH_E_TRY_USE_AVX512_HASH_FUNC_FLAG_REQUIRE_HASH_ALL_KEYS_FIRST;
        }

    } else if (TableCreateFlags->TryLargePagesForVertexPairs) {

        //
        // We can't use write-combine page protection on vertex pairs if they're
        // backed by large pages.
        //

        if (TableCreateFlags->EnableWriteCombineForVertexPairs) {
            return PH_E_CANT_WRITE_COMBINE_VERTEX_PAIRS_WHEN_LARGE_PAGES;
        }

    } else if (TableCreateFlags->RemoveWriteCombineAfterSuccessfulHashKeys) {

        //
        // If write-combining hasn't been requested for vertex pairs, it makes
        // no sense to request automatic write-combine removal.
        //

        if (!TableCreateFlags->EnableWriteCombineForVertexPairs) {
            return PH_E_REMOVE_WRITE_COMBINE_REQUIRES_ENABLE_WRITE_COMBINE;
        }

    }

    return S_OK;
}

//
// Define the table load flags.
//

typedef union _PERFECT_HASH_TABLE_LOAD_FLAGS {

    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, tries to allocate the table data using large pages.  The
        // caller is responsible for ensuring the process can create large pages
        // first by enabling the lock memory privilege.  If large pages can't be
        // allocated (because the lock memory privilege hasn't been enabled, or
        // there are insufficient large pages available to the system), the
        // table data will be accessed via the normal memory-mapped address of
        // the underlying file.
        //
        // To determine whether or not the large page allocation was successful,
        // check the TableDataUsesLargePages bit of PERFECT_HASH_TABLE_FLAGS
        // enum (the flags can be obtained after loading via the GetFlags() vtbl
        // function).
        //

        ULONG TryLargePagesForTableData:1;

        //
        // As above, but for the values array (i.e. the memory used to save the
        // values inserted into the table via Insert()).  Corresponds to the
        // ValuesArrayUsesLargePages bit of PERFECT_HASH_TABLE_FLAGS.
        //

        ULONG TryLargePagesForValuesArray:1;

        //
        // Unused bits.
        //

        ULONG Unused:30;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_TABLE_LOAD_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_TABLE_LOAD_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_TABLE_LOAD_FLAGS *PPERFECT_HASH_TABLE_LOAD_FLAGS;

FORCEINLINE
HRESULT
IsValidTableLoadFlags(
    _In_ PPERFECT_HASH_TABLE_LOAD_FLAGS LoadFlags
    )
{
    if (!ARGUMENT_PRESENT(LoadFlags)) {
        return E_POINTER;
    }

    if (LoadFlags->Unused != 0) {
        return E_FAIL;
    }

    return S_OK;
}

//
// Define table compilation flags.
//

typedef union _PERFECT_HASH_TABLE_COMPILE_FLAGS {

    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, ignores any optimized assembly routines available for
        // compilation and uses the standard C routines.  If no assembly
        // routine has been written for the given table configuration (i.e.
        // algorithm, hash function and masking type), setting this bit has no
        // effect.
        //

        ULONG IgnoreAssembly:1;

        //
        // Unused bits.
        //

        ULONG Unused:31;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_TABLE_COMPILE_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_TABLE_COMPILE_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_TABLE_COMPILE_FLAGS *PPERFECT_HASH_TABLE_COMPILE_FLAGS;

FORCEINLINE
HRESULT
IsValidTableCompileFlags(
    _In_ PPERFECT_HASH_TABLE_COMPILE_FLAGS CompileFlags
    )
{
    if (!ARGUMENT_PRESENT(CompileFlags)) {
        return E_POINTER;
    }

    if (CompileFlags->Unused != 0) {
        return E_FAIL;
    }

    return S_OK;
}

//
// Define an X-macro for random number generators.  The ENTRY macros receive
// the following parameters: (Name, Upper).
//

#define PERFECT_HASH_RNG_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    FIRST_ENTRY(System, SYSTEM)                                \
    LAST_ENTRY(Philox43210, PHILOX43210)

#define PERFECT_HASH_RNG_TABLE_ENTRY(ENTRY) \
    PERFECT_HASH_RNG_TABLE(ENTRY, ENTRY, ENTRY)

#define EXPAND_AS_RNG_ENUM(Name, Upper) \
    PerfectHashRng##Name##Id,

//
// Define an enumeration for identifying which random number generator to use.
//

typedef enum _PERFECT_HASH_RNG_ID {

    //
    // Explicitly define a null ID to take the 0-index slot.
    //

    PerfectHashNullRngId = 0,

    //
    // Begin valid RNGs.
    //

    PERFECT_HASH_RNG_TABLE_ENTRY(EXPAND_AS_RNG_ENUM)

    //
    // End valid RNGs.
    //

    //
    // N.B.  Keep the next value last.
    //

    PerfectHashInvalidRngId,

} PERFECT_HASH_RNG_ID;
typedef PERFECT_HASH_RNG_ID *PPERFECT_HASH_RNG_ID;

//
// Provide a simple inline RNG validation routine.
//

FORCEINLINE
BOOLEAN
IsValidPerfectHashRngId(
    _In_ PERFECT_HASH_RNG_ID RngId
    )
{
    return (
        RngId > PerfectHashNullRngId &&
        RngId < PerfectHashInvalidRngId
    );
}

//
// Declare the RNG component.
//

DECLARE_COMPONENT(Rng, RNG);

//
// Define RNG function typedefs.
//

typedef union _RNG_FLAGS {
    struct {

        //
        // When set, indicates new seeds should be obtained from the system
        // random number generator during the RNG initialization routines (e.g.
        // InitializePseudo()).  This is set automatically when the command line
        // parameter --RngUseRandomStartSeed is supplied.
        //
        // Invariant: When set, UseDefaultStartSeed must not be set.
        //

        ULONG UseRandomStartSeed:1;

        //
        // When set, indicates the default random seed should be used.  This is
        // set automatically if both --RngSeed and --RngUseRandomStartSeed are
        // not present on the command line.
        //
        // Invariant: When set, UseRandomStartSeed must not be set.
        //

        ULONG UseDefaultStartSeed:1;

        //
        // Unused bits.
        //

        ULONG Unused:30;

    };
    LONG AsLong;
    ULONG AsULong;
} RNG_FLAGS;
C_ASSERT(sizeof(RNG_FLAGS) == sizeof(ULONG));
typedef RNG_FLAGS *PRNG_FLAGS;

FORCEINLINE
HRESULT
IsValidRngFlags(
    _In_ PRNG_FLAGS Flags
    )
{
    if (!ARGUMENT_PRESENT(Flags)) {
        return E_POINTER;
    }

    if (Flags->Unused != 0) {
        return E_FAIL;
    }

    if (Flags->UseRandomStartSeed != FALSE) {
        if (Flags->UseDefaultStartSeed != FALSE) {
            return E_INVALIDARG;
        }
    }

    return S_OK;
}

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE RNG_INITIALIZE_PSEUDO)(
    _In_ PRNG Rng,
    _In_ PERFECT_HASH_RNG_ID RngId,
    _In_ PRNG_FLAGS Flags,
    _In_ ULONGLONG Seed,
    _In_ ULONGLONG Subsequence,
    _In_ ULONGLONG Offset
    );
typedef RNG_INITIALIZE_PSEUDO *PRNG_INITIALIZE_PSEUDO;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_At_(SizeOfBufferInBytes, _Pre_satisfies_(SizeOfBufferInBytes % 4 == 0))
HRESULT
(STDAPICALLTYPE RNG_GENERATE_RANDOM_BYTES)(
    _In_ PRNG Rng,
    _In_ SIZE_T SizeOfBufferInBytes,
    _Out_writes_(SizeOfBufferInBytes) PBYTE Buffer
    );
typedef RNG_GENERATE_RANDOM_BYTES *PRNG_GENERATE_RANDOM_BYTES;

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE RNG_GET_CURRENT_OFFSET)(
    _In_ PRNG Rng,
    _Out_ PULONGLONG Offset
    );
typedef RNG_GET_CURRENT_OFFSET *PRNG_GET_CURRENT_OFFSET;

//
// Define the RNG vtable.
//

typedef struct _RNG_VTBL {
    DECLARE_COMPONENT_VTBL_HEADER(RNG);
    PRNG_INITIALIZE_PSEUDO InitializePseudo;
    PRNG_GENERATE_RANDOM_BYTES GenerateRandomBytes;
    PRNG_GET_CURRENT_OFFSET GetCurrentOffset;
} RNG_VTBL;
typedef RNG_VTBL *PRNG_VTBL;

//
// Define an X-macro for CURAND random number generators.  The ENTRY macros
// receive the following parameters: (Name, Upper, IsImplemented).
//

#define PERFECT_HASH_CU_RNG_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    FIRST_ENTRY(Philox43210, PHILOX43210, TRUE)                   \
    ENTRY(XorWow, XORWOW, FALSE)                                  \
    ENTRY(MRG32k3a, MRG32K3A, FALSE)                              \
    ENTRY(MTGP32, MTGP32, FALSE)                                  \
    ENTRY(Sobol32, SOBOL32, FALSE)                                \
    ENTRY(Sobol64, SOBOL64, FALSE)                                \
    LAST_ENTRY(Scratch, SCRATCH, FALSE)

#define PERFECT_HASH_CU_RNG_TABLE_ENTRY(ENTRY) \
    PERFECT_HASH_CU_RNG_TABLE(ENTRY, ENTRY, ENTRY)

#define EXPAND_AS_CU_RNG_ENUM(Name, Upper, IsImplemented) \
    PerfectHashCuRng##Name##Id,

//
// Define an enumeration for identifying which CUDA random number generator
// to use.
//

typedef enum _PERFECT_HASH_CU_RNG_ID {

    //
    // Explicitly define a null ID to take the 0-index slot.
    //

    PerfectHashNullCuRngId = 0,

    //
    // Begin valid RNGs.
    //

    PERFECT_HASH_CU_RNG_TABLE_ENTRY(EXPAND_AS_CU_RNG_ENUM)

    //
    // End valid RNGs.
    //

    //
    // N.B.  Keep the next value last.
    //

    PerfectHashInvalidCuRngId,

} PERFECT_HASH_CU_RNG_ID;
typedef PERFECT_HASH_CU_RNG_ID *PPERFECT_HASH_CU_RNG_ID;

//
// Provide a simple inline RNG validation routine.
//

FORCEINLINE
BOOLEAN
IsValidPerfectHashCuRngId(
    _In_ PERFECT_HASH_CU_RNG_ID RngId
    )
{
    return (
        RngId > PerfectHashNullCuRngId &&
        RngId < PerfectHashInvalidCuRngId
    );
}

//
// Define the X-macro for table create parameters.
//

#define TABLE_CREATE_PARAMETER_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    FIRST_ENTRY(AttemptsBeforeTableResize)                           \
    ENTRY(MaxNumberOfTableResizes)                                   \
    ENTRY(MaxNumberOfEqualBestGraphs)                                \
    ENTRY(InitialNumberOfTableResizes)                               \
    ENTRY(MinAttempts)                                               \
    ENTRY(MaxAttempts)                                               \
    ENTRY(FixedAttempts)                                             \
    ENTRY(TargetNumberOfSolutions)                                   \
    ENTRY(BestCoverageAttempts)                                      \
    ENTRY(BestCoverageType)                                          \
    ENTRY(MaxNumberOfEqualBestGraphs)                                \
    ENTRY(MinNumberOfKeysForFindBestGraph)                           \
    ENTRY(KeysSubset)                                                \
    ENTRY(MainWorkThreadpoolPriority)                                \
    ENTRY(FileWorkThreadpoolPriority)                                \
    ENTRY(Seeds)                                                     \
    ENTRY(ValueSizeInBytes)                                          \
    ENTRY(KeySizeInBytes)                                            \
    ENTRY(SolutionsFoundRatio)                                       \
    ENTRY(GraphImpl)                                                 \
    ENTRY(Rng)                                                       \
    ENTRY(RngSeed)                                                   \
    ENTRY(RngSubsequence)                                            \
    ENTRY(RngOffset)                                                 \
    ENTRY(Seed3Byte1MaskCounts)                                      \
    ENTRY(Seed3Byte2MaskCounts)                                      \
    ENTRY(MaxSolveTimeInSeconds)                                     \
    ENTRY(AutoResizeWhenKeysToEdgesRatioExceeds)                     \
    ENTRY(FunctionHookCallbackDllPath)                               \
    ENTRY(FunctionHookCallbackFunctionName)                          \
    ENTRY(FunctionHookCallbackIgnoreRip)                             \
    ENTRY(BestCoverageTargetValue)                                   \
    ENTRY(CuConcurrency)                                             \
    ENTRY(CuPtxPath)                                                 \
    ENTRY(CuDevices)                                                 \
    ENTRY(CuDevicesBlocksPerGrid)                                    \
    ENTRY(CuDevicesThreadsPerBlock)                                  \
    ENTRY(CuDevicesKernelRuntimeTargetInMilliseconds)                \
    ENTRY(CuCudaDevRuntimeLibPath)                                   \
    ENTRY(CuNumberOfRandomHostSeeds)                                 \
    LAST_ENTRY(Remark)

#define TABLE_CREATE_PARAMETER_TABLE_ENTRY(ENTRY) \
    TABLE_CREATE_PARAMETER_TABLE(ENTRY, ENTRY, ENTRY)

#define EXPAND_AS_TABLE_CREATE_PARAMETER_ENUM(Name) \
    TableCreateParameter##Name##Id,                 \
    PerfectHashTableCreateParameter##Name##Id =     \
        TableCreateParameter##Name##Id,

//
// N.B. We need the PerfectHashTable prefix in order for certain X-macro
//      consumers to work, however, the final name is inconveniently long,
//      so we also export identical enum names with the prefix removed.
//

typedef enum _PERFECT_HASH_TABLE_CREATE_PARAMETER_ID {
    TableCreateParameterNullId = 0,
    PerfectHashNullTableCreateParameterId = TableCreateParameterNullId,

    TABLE_CREATE_PARAMETER_TABLE_ENTRY(EXPAND_AS_TABLE_CREATE_PARAMETER_ENUM)

    TableCreateParameterInvalidId,
    PerfectHashInvalidTableCreateParameterId = TableCreateParameterInvalidId,
} PERFECT_HASH_TABLE_CREATE_PARAMETER_ID;

FORCEINLINE
BOOLEAN
IsValidPerfectHashTableCreateParameterId(
    _In_ PERFECT_HASH_TABLE_CREATE_PARAMETER_ID TableCreateParameter
    )
{
    return (
        TableCreateParameter > TableCreateParameterNullId &&
        TableCreateParameter < TableCreateParameterInvalidId
    );
}

//
// Define an X-macro for the best coverage types.  The ENTRY macros receive
// (Name, Comparison, Comparator) as their arguments, e.g.:
//
//      (NumberOfEmptyPages,            Highest, >)
//      (NumberOfPagesUsedByKeysSubset, Lowest,  <)
//
// N.B. See GraphRegisterSolved() for how these predicates are used in
//      registering the best graph, which should give more clarity to the
//      role of the X-macro.
//

#define BEST_COVERAGE_TYPE_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    FIRST_ENTRY(NumberOfEmptyPages, Highest, >)                  \
    ENTRY(NumberOfEmptyLargePages, Highest, >)                   \
    ENTRY(NumberOfEmptyCacheLines, Highest, >)                   \
    ENTRY(NumberOfEmptyPages, Lowest, <)                         \
    ENTRY(NumberOfEmptyLargePages, Lowest, <)                    \
    ENTRY(NumberOfEmptyCacheLines, Lowest, <)                    \
    ENTRY(NumberOfUsedPages, Highest, >)                         \
    ENTRY(NumberOfUsedLargePages, Highest, >)                    \
    ENTRY(NumberOfUsedCacheLines, Highest, >)                    \
    ENTRY(NumberOfUsedPages, Lowest, <)                          \
    ENTRY(NumberOfUsedLargePages, Lowest, <)                     \
    ENTRY(NumberOfUsedCacheLines, Lowest, <)                     \
    ENTRY(MaxGraphTraversalDepth, Highest, >)                    \
    ENTRY(MaxGraphTraversalDepth, Lowest, <)                     \
    ENTRY(TotalGraphTraversals, Highest, >)                      \
    ENTRY(TotalGraphTraversals, Lowest, <)                       \
    ENTRY(NumberOfEmptyVertices, Highest, >)                     \
    ENTRY(NumberOfEmptyVertices, Lowest, <)                      \
    ENTRY(NumberOfCollisionsDuringAssignment, Highest, >)        \
    ENTRY(NumberOfCollisionsDuringAssignment, Lowest, <)         \
    ENTRY(MaxAssignedPerCacheLineCount, Highest, >)              \
    ENTRY(MaxAssignedPerCacheLineCountForKeysSubset, Highest, >) \
    ENTRY(MaxAssignedPerCacheLineCount, Lowest, <)               \
    ENTRY(MaxAssignedPerCacheLineCountForKeysSubset, Lowest, <)  \
    ENTRY(NumberOfPagesUsedByKeysSubset, Lowest, <)              \
    ENTRY(NumberOfLargePagesUsedByKeysSubset, Lowest, <)         \
    ENTRY(NumberOfCacheLinesUsedByKeysSubset, Lowest, <)         \
    ENTRY(NumberOfPagesUsedByKeysSubset, Highest, >)             \
    ENTRY(NumberOfLargePagesUsedByKeysSubset, Highest, >)        \
    ENTRY(NumberOfCacheLinesUsedByKeysSubset, Highest, >)        \
    ENTRY(PredictedNumberOfFilledCacheLines, Lowest, <)          \
    ENTRY(PredictedNumberOfFilledCacheLines, Highest, >)         \
    ENTRY(Slope, Lowest, <)                                      \
    ENTRY(Slope, Highest, >)                                     \
    ENTRY(Score, Lowest , <)                                     \
    ENTRY(Score, Highest , >)                                    \
    ENTRY(Rank, Lowest , <)                                      \
    LAST_ENTRY(Rank, Highest , >)

#define BEST_COVERAGE_TYPE_TABLE_ENTRY(ENTRY) \
    BEST_COVERAGE_TYPE_TABLE(ENTRY, ENTRY, ENTRY)

#define EXPAND_AS_BEST_COVERAGE_TYPE_ENUM(Name, Comparison, Comparator) \
    BestCoverageType##Comparison##Name##Id,                             \
    PerfectHashTableBestCoverageType##Comparison##Name##Id =            \
        BestCoverageType##Comparison##Name##Id,

//
// N.B. We need the PerfectHashTable prefix in order for certain X-macro
//      consumers to work, however, the final name is inconveniently long,
//      so we also export identical enum names with the prefix removed.
//

typedef enum _PERFECT_HASH_TABLE_BEST_COVERAGE_TYPE_ID {
    BestCoverageTypeNullId = 0,
    PerfectHashNullBestCoverageTypeId = BestCoverageTypeNullId,

    BEST_COVERAGE_TYPE_TABLE_ENTRY(EXPAND_AS_BEST_COVERAGE_TYPE_ENUM)

    BestCoverageTypeInvalidId,
    PerfectHashInvalidBestCoverageTypeId = BestCoverageTypeInvalidId,
} PERFECT_HASH_TABLE_BEST_COVERAGE_TYPE_ID;

FORCEINLINE
BOOLEAN
IsValidPerfectHashBestCoverageTypeId(
    _In_ PERFECT_HASH_TABLE_BEST_COVERAGE_TYPE_ID CoverageType
    )
{
    return (
        CoverageType > BestCoverageTypeNullId &&
        CoverageType < BestCoverageTypeInvalidId
    );
}

FORCEINLINE
BOOLEAN
DoesBestCoverageTypeRequireKeysSubset(
    _In_ PERFECT_HASH_TABLE_BEST_COVERAGE_TYPE_ID Type
    )
{
    return (
        Type ==
          BestCoverageTypeHighestMaxAssignedPerCacheLineCountForKeysSubsetId ||
        Type ==
          BestCoverageTypeLowestMaxAssignedPerCacheLineCountForKeysSubsetId  ||
        Type == BestCoverageTypeLowestNumberOfPagesUsedByKeysSubsetId        ||
        Type == BestCoverageTypeLowestNumberOfLargePagesUsedByKeysSubsetId   ||
        Type == BestCoverageTypeLowestNumberOfCacheLinesUsedByKeysSubsetId   ||
        Type == BestCoverageTypeHighestNumberOfPagesUsedByKeysSubsetId       ||
        Type == BestCoverageTypeHighestNumberOfLargePagesUsedByKeysSubsetId  ||
        Type == BestCoverageTypeHighestNumberOfCacheLinesUsedByKeysSubsetId
    );
}

FORCEINLINE
BOOLEAN
IsSeedMaskCountParameter(
    _In_ PERFECT_HASH_TABLE_CREATE_PARAMETER_ID Id
    )
{
    return (
        Id == TableCreateParameterSeed3Byte1MaskCountsId ||
        Id == TableCreateParameterSeed3Byte2MaskCountsId
    );
}

FORCEINLINE
BOOLEAN
DoesTableCreateParameterRequireDeallocation(
    _In_ PERFECT_HASH_TABLE_CREATE_PARAMETER_ID Id
    )
{

    //
    // All parameters that accept comma-separated lists require deallocation.
    //

    return (
        Id == TableCreateParameterKeysSubsetId ||
        Id == TableCreateParameterSeedsId ||
        Id == TableCreateParameterCuDevicesId ||
        Id == TableCreateParameterCuDevicesBlocksPerGridId ||
        Id == TableCreateParameterCuDevicesThreadsPerBlockId ||
        Id == TableCreateParameterCuDevicesKernelRuntimeTargetInMillisecondsId ||
        IsSeedMaskCountParameter(Id)
    );
}

FORCEINLINE
BOOLEAN
DoesBestCoverageTypeUseValueArray(
    _In_ PERFECT_HASH_TABLE_BEST_COVERAGE_TYPE_ID CoverageType
    )
{
    return (
        DoesBestCoverageTypeRequireKeysSubset(CoverageType)
    );
}

FORCEINLINE
BOOLEAN
DoesBestCoverageTypeUseDouble(
    _In_ PERFECT_HASH_TABLE_BEST_COVERAGE_TYPE_ID Type
    )
{
    return (
        Type == BestCoverageTypeLowestSlopeId                              ||
        Type == BestCoverageTypeHighestSlopeId                             ||
        Type == BestCoverageTypeLowestPredictedNumberOfFilledCacheLinesId  ||
        Type == BestCoverageTypeHighestPredictedNumberOfFilledCacheLinesId ||
        Type == BestCoverageTypeLowestRankId                               ||
        Type == BestCoverageTypeHighestRankId
    );
}

typedef struct _PERFECT_HASH_TABLE_CREATE_PARAMETER {
    PERFECT_HASH_TABLE_CREATE_PARAMETER_ID Id;
    ULONG Padding;
    union {
        PVOID AsVoidPointer;
        LONG AsLong;
        ULONG AsULong;
        DOUBLE AsDouble;
        STRING AsString;
        LONGLONG AsLongLong;
        ULONGLONG AsULongLong;
        LARGE_INTEGER AsLargeInteger;
        ULARGE_INTEGER AsULargeInteger;
        STRING AsString;
        UNICODE_STRING AsUnicodeString;
        TP_CALLBACK_PRIORITY AsTpCallbackPriority;
        PERFECT_HASH_RNG_ID AsRngId;
        PERFECT_HASH_CU_RNG_ID AsCuRngId;
        PERFECT_HASH_TABLE_BEST_COVERAGE_TYPE_ID AsBestCoverageType;
        VALUE_ARRAY AsValueArray;
        KEYS_SUBSET AsKeysSubset;
        SEED_MASK_COUNTS AsSeedMaskCounts;
    };
} PERFECT_HASH_TABLE_CREATE_PARAMETER;
typedef PERFECT_HASH_TABLE_CREATE_PARAMETER
      *PPERFECT_HASH_TABLE_CREATE_PARAMETER;

//
// Define table create parameter flags.
//

typedef union _PERFECT_HASH_TABLE_CREATE_PARAMETERS_FLAGS {

    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, indicates at least one seed mask count parameter is
        // present.
        //

        ULONG HasSeedMaskCounts:1;

        //
        // When set, indicates a best coverage target value has been provided.
        //

        ULONG HasBestCoverageTargetValue:1;

        //
        // Unused bits.
        //

        ULONG Unused:30;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_TABLE_CREATE_PARAMETERS_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_TABLE_CREATE_PARAMETERS_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_TABLE_CREATE_PARAMETERS_FLAGS
      *PPERFECT_HASH_TABLE_CREATE_PARAMETERS_FLAGS;

FORCEINLINE
HRESULT
IsValidTableCreateParametersFlags(
    _In_ PPERFECT_HASH_TABLE_CREATE_PARAMETERS_FLAGS Flags
    )
{

    if (!ARGUMENT_PRESENT(Flags)) {
        return E_POINTER;
    }

    if (Flags->Unused != 0) {
        return E_FAIL;
    }

    return S_OK;
}

typedef
struct _Struct_size_bytes_(SizeOfStruct) _PERFECT_HASH_TABLE_CREATE_PARAMETERS {
    ULONG SizeOfStruct;
    ULONG NumberOfElements;
    PALLOCATOR Allocator;
    PPERFECT_HASH_TABLE_CREATE_PARAMETER Params;
    PERFECT_HASH_TABLE_CREATE_PARAMETERS_FLAGS Flags;
    ULONG Padding1;
} PERFECT_HASH_TABLE_CREATE_PARAMETERS;
typedef PERFECT_HASH_TABLE_CREATE_PARAMETERS
      *PPERFECT_HASH_TABLE_CREATE_PARAMETERS;

//
// Bulk Create
//

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_CONTEXT_BULK_CREATE)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PCUNICODE_STRING KeysDirectory,
    _In_ PCUNICODE_STRING BaseOutputDirectory,
    _In_ PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    _In_ PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    _In_ PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    _In_opt_ PPERFECT_HASH_CONTEXT_BULK_CREATE_FLAGS ContextBulkCreateFlags,
    _In_opt_ PPERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags,
    _In_opt_ PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags,
    _In_opt_ PPERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags,
    _In_opt_ PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    );
typedef PERFECT_HASH_CONTEXT_BULK_CREATE *PPERFECT_HASH_CONTEXT_BULK_CREATE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_CONTEXT_BULK_CREATE_ARGVW)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ ULONG NumberOfArguments,
    _In_ LPWSTR *ArgvW,
    _In_ LPWSTR CommandLineW
    );
typedef PERFECT_HASH_CONTEXT_BULK_CREATE_ARGVW
      *PPERFECT_HASH_CONTEXT_BULK_CREATE_ARGVW;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_CONTEXT_EXTRACT_BULK_CREATE_ARGS_FROM_ARGVW)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ ULONG NumberOfArguments,
    _In_ LPWSTR *ArgvW,
    _In_ LPWSTR CommandLineW,
    _In_ PUNICODE_STRING KeysDirectory,
    _In_ PUNICODE_STRING BaseOutputDirectory,
    _Inout_ PPERFECT_HASH_ALGORITHM_ID AlgorithmId,
    _Inout_ PPERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    _Inout_ PPERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    _Inout_ PULONG MaximumConcurrency,
    _Inout_ PPERFECT_HASH_CONTEXT_BULK_CREATE_FLAGS ContextBulkCreateFlags,
    _Inout_ PPERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags,
    _Inout_ PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags,
    _Inout_ PPERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags,
    _In_ PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    );
typedef PERFECT_HASH_CONTEXT_EXTRACT_BULK_CREATE_ARGS_FROM_ARGVW
      *PPERFECT_HASH_CONTEXT_EXTRACT_BULK_CREATE_ARGS_FROM_ARGVW;

//
// Table Create
//

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_CONTEXT_TABLE_CREATE)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PCUNICODE_STRING KeysPath,
    _In_ PCUNICODE_STRING BaseOutputDirectory,
    _In_ PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    _In_ PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    _In_ PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    _In_opt_ PPERFECT_HASH_CONTEXT_TABLE_CREATE_FLAGS ContextTableCreateFlags,
    _In_opt_ PPERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags,
    _In_opt_ PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags,
    _In_opt_ PPERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags,
    _In_ PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    );
typedef PERFECT_HASH_CONTEXT_TABLE_CREATE *PPERFECT_HASH_CONTEXT_TABLE_CREATE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_CONTEXT_TABLE_CREATE_ARGVW)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ ULONG NumberOfArguments,
    _In_ LPWSTR *ArgvW,
    _In_ LPWSTR CommandLineW
    );
typedef PERFECT_HASH_CONTEXT_TABLE_CREATE_ARGVW
      *PPERFECT_HASH_CONTEXT_TABLE_CREATE_ARGVW;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_CONTEXT_EXTRACT_TABLE_CREATE_ARGS_FROM_ARGVW)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ ULONG NumberOfArguments,
    _In_ LPWSTR *ArgvW,
    _In_ LPWSTR CommandLineW,
    _In_ PUNICODE_STRING KeysPath,
    _In_ PUNICODE_STRING BaseOutputDirectory,
    _Inout_ PPERFECT_HASH_ALGORITHM_ID AlgorithmId,
    _Inout_ PPERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    _Inout_ PPERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    _Inout_ PULONG MaximumConcurrency,
    _Inout_ PPERFECT_HASH_CONTEXT_TABLE_CREATE_FLAGS ContextTableCreateFlags,
    _Inout_ PPERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags,
    _Inout_ PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags,
    _Inout_ PPERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags,
    _In_ PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    );
typedef PERFECT_HASH_CONTEXT_EXTRACT_TABLE_CREATE_ARGS_FROM_ARGVW
      *PPERFECT_HASH_CONTEXT_EXTRACT_TABLE_CREATE_ARGS_FROM_ARGVW;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_CONTEXT_TABLE_CREATE_ARGVA)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ ULONG NumberOfArguments,
    _In_ LPSTR *ArgvA
    );
typedef PERFECT_HASH_CONTEXT_TABLE_CREATE_ARGVA
      *PPERFECT_HASH_CONTEXT_TABLE_CREATE_ARGVA;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_CONTEXT_BULK_CREATE_ARGVA)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ ULONG NumberOfArguments,
    _In_ LPSTR *ArgvA
    );
typedef PERFECT_HASH_CONTEXT_BULK_CREATE_ARGVA
      *PPERFECT_HASH_CONTEXT_BULK_CREATE_ARGVA;

typedef struct _PERFECT_HASH_CONTEXT_VTBL {
    DECLARE_COMPONENT_VTBL_HEADER(PERFECT_HASH_CONTEXT);

    PPERFECT_HASH_CONTEXT_SET_MAXIMUM_CONCURRENCY SetMaximumConcurrency;
    PPERFECT_HASH_CONTEXT_GET_MAXIMUM_CONCURRENCY GetMaximumConcurrency;
    PPERFECT_HASH_CONTEXT_SET_BASE_OUTPUT_DIRECTORY SetBaseOutputDirectory;
    PPERFECT_HASH_CONTEXT_GET_BASE_OUTPUT_DIRECTORY GetBaseOutputDirectory;

    PPERFECT_HASH_CONTEXT_BULK_CREATE BulkCreate;
    PPERFECT_HASH_CONTEXT_BULK_CREATE_ARGVW BulkCreateArgvW;
    PPERFECT_HASH_CONTEXT_EXTRACT_BULK_CREATE_ARGS_FROM_ARGVW
        ExtractBulkCreateArgsFromArgvW;

    PPERFECT_HASH_CONTEXT_TABLE_CREATE TableCreate;

    PPERFECT_HASH_CONTEXT_TABLE_CREATE_ARGVW TableCreateArgvW;
    PPERFECT_HASH_CONTEXT_EXTRACT_TABLE_CREATE_ARGS_FROM_ARGVW
        ExtractTableCreateArgsFromArgvW;

    //
    // N.B. These two routines will only be present on non-Windows platforms.
    //

    PPERFECT_HASH_CONTEXT_TABLE_CREATE_ARGVA TableCreateArgvA;
    PPERFECT_HASH_CONTEXT_BULK_CREATE_ARGVA BulkCreateArgvA;

} PERFECT_HASH_CONTEXT_VTBL;
typedef PERFECT_HASH_CONTEXT_VTBL *PPERFECT_HASH_CONTEXT_VTBL;

#ifndef _PERFECT_HASH_INTERNAL_BUILD
typedef struct _PERFECT_HASH_CONTEXT {
    PPERFECT_HASH_CONTEXT_VTBL Vtbl;
} PERFECT_HASH_CONTEXT;
typedef PERFECT_HASH_CONTEXT *PPERFECT_HASH_CONTEXT;
#endif

//
// Define the PERFECT_HASH_TABLE interface.
//

DECLARE_COMPONENT(Table, PERFECT_HASH_TABLE);

typedef struct _PERFECT_HASH_TABLE PERFECT_HASH_TABLE;
typedef PERFECT_HASH_TABLE *PPERFECT_HASH_TABLE;

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_CREATE)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    _In_ PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    _In_ PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    _In_ PPERFECT_HASH_KEYS Keys,
    _In_opt_ PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags,
    _In_opt_ PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    );
typedef PERFECT_HASH_TABLE_CREATE *PPERFECT_HASH_TABLE_CREATE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_LOAD)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_opt_ PPERFECT_HASH_TABLE_LOAD_FLAGS LoadFlags,
    _In_ PCUNICODE_STRING Path,
    _In_opt_ PPERFECT_HASH_KEYS Keys
    );
typedef PERFECT_HASH_TABLE_LOAD *PPERFECT_HASH_TABLE_LOAD;

//
// Define the table flags enum and associated function pointer to obtain the
// flags from a loaded table.
//

typedef union _PERFECT_HASH_TABLE_FLAGS {
    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, indicates the table was created via a context.
        //
        // Invariant:
        //
        //      If Created == TRUE:
        //          Assert Loaded == FALSE
        //

        ULONG Created:1;

        //
        // When set, indicates the table was loaded from disk via a previously
        // created instance.
        //
        // Invariant:
        //
        //      If Loaded == TRUE:
        //          Assert Created == FALSE
        //

        ULONG Loaded:1;

        //
        // When set, indicates large pages are in use for the memory backing
        // the table data (e.g. the vertices array).  This will only ever be
        // set for loaded tables, not created ones.
        //

        ULONG TableDataUsesLargePages:1;

        //
        // When set, indicates the values array was allocated with large pages.
        // This will only ever be set for loaded tables, not created ones.
        //

        ULONG ValuesArrayUsesLargePages:1;

        //
        // When set, indicates the vertex pairs array was allocated with large
        // pages (at the individual graph level).  Only applies when the table
        // create flags --HashAllKeysFirst and --TryLargePagesForVertexPairs
        // are present.
        //

        ULONG VertexPairsArrayUsesLargePages:1;

        //
        // When set, indicates that the graph used an optimized AVX2 version
        // of the hash function during graph solving.
        //

        ULONG UsedAvx2HashFunction:1;

        //
        // When set, indicates that the graph used an optimized AVX512 version
        // of the hash function during graph solving.
        //

        ULONG UsedAvx512HashFunction:1;

        //
        // When set, indicates the AVX2 memory coverage function was used.
        //

        ULONG UsedAvx2MemoryCoverageFunction:1;

        //
        // Unused bits.
        //

        ULONG Unused:24;
    };

    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_TABLE_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_TABLE_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_TABLE_FLAGS *PPERFECT_HASH_TABLE_FLAGS;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_GET_FLAGS)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ ULONG SizeOfFlags,
    _Out_writes_bytes_(SizeOfFlags) PPERFECT_HASH_TABLE_FLAGS Flags
    );
typedef PERFECT_HASH_TABLE_GET_FLAGS *PPERFECT_HASH_TABLE_GET_FLAGS;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_COMPILE)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_opt_ PPERFECT_HASH_TABLE_COMPILE_FLAGS CompileFlags,
    _In_ PERFECT_HASH_CPU_ARCH_ID CpuArchId
    );
typedef PERFECT_HASH_TABLE_COMPILE *PPERFECT_HASH_TABLE_COMPILE;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_TEST)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_opt_ PPERFECT_HASH_KEYS Keys,
    _In_ BOOLEAN DebugBreakOnFailure
    );
typedef PERFECT_HASH_TABLE_TEST *PPERFECT_HASH_TABLE_TEST;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_INSERT)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ ULONG Key,
    _In_ ULONG Value,
    _Out_opt_ PULONG PreviousValue
    );
typedef PERFECT_HASH_TABLE_INSERT *PPERFECT_HASH_TABLE_INSERT;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_LOOKUP)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ ULONG Key,
    _Out_ PULONG Value
    );
typedef PERFECT_HASH_TABLE_LOOKUP *PPERFECT_HASH_TABLE_LOOKUP;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_DELETE)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ ULONG Key,
    _Out_opt_ PULONG PreviousValue
    );
typedef PERFECT_HASH_TABLE_DELETE *PPERFECT_HASH_TABLE_DELETE;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_INDEX)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ ULONG Key,
    _Out_ PULONG Index
    );
typedef PERFECT_HASH_TABLE_INDEX *PPERFECT_HASH_TABLE_INDEX;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_HASH)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ ULONG Key,
    _Out_ PULONGLONG Hash
    );
typedef PERFECT_HASH_TABLE_HASH *PPERFECT_HASH_TABLE_HASH;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_SEEDED_HASH)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ ULONG Key,
    _In_ ULONG NumberOfSeeds,
    _In_reads_(NumberOfSeeds) PULONG Seeds,
    _Out_ PULONGLONG Hash
    );
typedef PERFECT_HASH_TABLE_SEEDED_HASH *PPERFECT_HASH_TABLE_SEEDED_HASH;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_MASK_HASH)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ ULONG Key,
    _Out_ PULONG Masked
    );
typedef PERFECT_HASH_TABLE_MASK_HASH *PPERFECT_HASH_TABLE_MASK_HASH;

typedef
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_MASK_INDEX)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ ULONGLONG Key,
    _Out_ PULONG Masked
    );
typedef PERFECT_HASH_TABLE_MASK_INDEX *PPERFECT_HASH_TABLE_MASK_INDEX;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_GET_ALGORITHM_NAME)(
    _In_ PPERFECT_HASH_TABLE Table,
    _Out_ PCUNICODE_STRING *Name
    );
typedef PERFECT_HASH_TABLE_GET_ALGORITHM_NAME
      *PPERFECT_HASH_TABLE_GET_ALGORITHM_NAME;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_GET_HASH_FUNCTION_NAME)(
    _In_ PPERFECT_HASH_TABLE Table,
    _Out_ PCUNICODE_STRING *Name
    );
typedef PERFECT_HASH_TABLE_GET_HASH_FUNCTION_NAME
      *PPERFECT_HASH_TABLE_GET_HASH_FUNCTION_NAME;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_GET_MASK_FUNCTION_NAME)(
    _In_ PPERFECT_HASH_TABLE Table,
    _Out_ PCUNICODE_STRING *Name
    );
typedef PERFECT_HASH_TABLE_GET_MASK_FUNCTION_NAME
      *PPERFECT_HASH_TABLE_GET_MASK_FUNCTION_NAME;

typedef
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_GET_FILE)(
    _In_ PPERFECT_HASH_TABLE Table,
    _Inout_opt_ PPERFECT_HASH_FILE *File
    );
typedef PERFECT_HASH_TABLE_GET_FILE *PPERFECT_HASH_TABLE_GET_FILE;

typedef
ULONGLONG
(STDAPICALLTYPE PERFECT_HASH_TABLE_HASH_EX)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ ULONG Key
    );
typedef PERFECT_HASH_TABLE_HASH_EX *PPERFECT_HASH_TABLE_HASH_EX;

typedef
ULONGLONG
(STDAPICALLTYPE PERFECT_HASH_TABLE_SEEDED_HASH_EX)(
    _In_ ULONG Key,
    _In_ PULONG Seeds,
    _In_ ULONG Mask
    );
typedef PERFECT_HASH_TABLE_SEEDED_HASH_EX *PPERFECT_HASH_TABLE_SEEDED_HASH_EX;

typedef
ULONG
(STDAPICALLTYPE PERFECT_HASH_TABLE_SEEDED_HASH16_EX)(
    _In_ ULONG Key,
    _In_ PULONG Seeds,
    _In_ USHORT Mask
    );
typedef PERFECT_HASH_TABLE_SEEDED_HASH16_EX *PPERFECT_HASH_TABLE_SEEDED_HASH16_EX;

typedef struct _PERFECT_HASH_TABLE_VTBL {
    DECLARE_COMPONENT_VTBL_HEADER(PERFECT_HASH_TABLE);
    PPERFECT_HASH_TABLE_CREATE Create;
    PPERFECT_HASH_TABLE_LOAD Load;
    PPERFECT_HASH_TABLE_GET_FLAGS GetFlags;
    PPERFECT_HASH_TABLE_COMPILE Compile;
    PPERFECT_HASH_TABLE_TEST Test;
    PPERFECT_HASH_TABLE_INSERT Insert;
    PPERFECT_HASH_TABLE_LOOKUP Lookup;
    PPERFECT_HASH_TABLE_DELETE Delete;
    PPERFECT_HASH_TABLE_INDEX Index;
    PPERFECT_HASH_TABLE_HASH Hash;
    PPERFECT_HASH_TABLE_MASK_HASH MaskHash;
    PPERFECT_HASH_TABLE_MASK_INDEX MaskIndex;
    PPERFECT_HASH_TABLE_SEEDED_HASH SeededHash;
    PPERFECT_HASH_TABLE_INDEX FastIndex;
    PPERFECT_HASH_TABLE_INDEX SlowIndex;
    PPERFECT_HASH_TABLE_GET_ALGORITHM_NAME GetAlgorithmName;
    PPERFECT_HASH_TABLE_GET_HASH_FUNCTION_NAME GetHashFunctionName;
    PPERFECT_HASH_TABLE_GET_MASK_FUNCTION_NAME GetMaskFunctionName;
    PPERFECT_HASH_TABLE_GET_FILE GetFile;
    PPERFECT_HASH_TABLE_HASH_EX HashEx;
    PPERFECT_HASH_TABLE_SEEDED_HASH_EX SeededHashEx;
    PPERFECT_HASH_TABLE_SEEDED_HASH16_EX SeededHash16Ex;
} PERFECT_HASH_TABLE_VTBL;
typedef PERFECT_HASH_TABLE_VTBL *PPERFECT_HASH_TABLE_VTBL;

//
// Helper functions for obtaining the string representation of enumeration IDs.
//

typedef
_Success_(return != 0)
BOOLEAN
(NTAPI GET_PERFECT_HASH_TABLE_ALGORITHM_NAME)(
    _In_ PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    _Out_ PCUNICODE_STRING *Name
    );
typedef GET_PERFECT_HASH_TABLE_ALGORITHM_NAME
      *PGET_PERFECT_HASH_TABLE_ALGORITHM_NAME;

typedef
_Success_(return != 0)
BOOLEAN
(NTAPI GET_PERFECT_HASH_TABLE_HASH_FUNCTION_NAME)(
    _In_ PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    _Out_ PCUNICODE_STRING *Name
    );
typedef GET_PERFECT_HASH_TABLE_HASH_FUNCTION_NAME
      *PGET_PERFECT_HASH_TABLE_HASH_FUNCTION_NAME;

typedef
_Success_(return != 0)
BOOLEAN
(NTAPI GET_PERFECT_HASH_TABLE_MASK_FUNCTION_NAME)(
    _In_ PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    _Out_ PCUNICODE_STRING *Name
    );
typedef GET_PERFECT_HASH_TABLE_MASK_FUNCTION_NAME
      *PGET_PERFECT_HASH_TABLE_MASK_FUNCTION_NAME;

#ifdef PH_WINDOWS

#ifndef __CUDA_ARCH__

//
// Scaffolding required to support structured exception handling via __try
// blocks without having to link to the C runtime library.
//

typedef
EXCEPTION_DISPOSITION
(__cdecl __C_SPECIFIC_HANDLER)(
    PEXCEPTION_RECORD ExceptionRecord,
    ULONG_PTR Frame,
    PCONTEXT Context,
    struct _DISPATCHER_CONTEXT *Dispatch
    );
typedef __C_SPECIFIC_HANDLER *P__C_SPECIFIC_HANDLER;

typedef
EXCEPTION_DISPOSITION
(__cdecl RTL_EXCEPTION_HANDLER)(
    PEXCEPTION_RECORD ExceptionRecord,
    ULONG_PTR Frame,
    PCONTEXT Context,
    struct _DISPATCHER_CONTEXT *Dispatch
    );
typedef RTL_EXCEPTION_HANDLER *PRTL_EXCEPTION_HANDLER;

typedef RTL_EXCEPTION_HANDLER __C_SPECIFIC_HANDLER;
typedef __C_SPECIFIC_HANDLER *P__C_SPECIFIC_HANDLER;

typedef
VOID
(SET_C_SPECIFIC_HANDLER)(
    _In_ P__C_SPECIFIC_HANDLER Handler
    );
typedef SET_C_SPECIFIC_HANDLER *PSET_C_SPECIFIC_HANDLER;

typedef
VOID
(__cdecl __SECURITY_INIT_COOKIE)(
    VOID
    );
typedef __SECURITY_INIT_COOKIE *P__SECURITY_INIT_COOKIE;

extern __SECURITY_INIT_COOKIE __security_init_cookie;

//
// _penter function hooking scaffolding.
//

typedef
VOID
(NTAPI FUNCTION_ENTRY_CALLBACK)(
    _In_ PVOID ReturnRip,
    _In_ PVOID Context
    );
typedef FUNCTION_ENTRY_CALLBACK *PFUNCTION_ENTRY_CALLBACK;

typedef
VOID
(NTAPI SET_FUNCTION_ENTRY_CALLBACK)(
    _In_ PFUNCTION_ENTRY_CALLBACK Callback,
    _In_ PVOID Context,
    _In_ PVOID ModuleBaseAddress,
    _In_ ULONG ModuleSizeInBytes,
    _In_ ULONG IgnoreRip
    );
typedef SET_FUNCTION_ENTRY_CALLBACK *PSET_FUNCTION_ENTRY_CALLBACK;

typedef
VOID
(NTAPI GET_FUNCTION_ENTRY_CALLBACK)(
    _Out_ PFUNCTION_ENTRY_CALLBACK *Callback,
    _Out_ PVOID *Context,
    _Out_ PVOID *ModuleBaseAddress,
    _Out_ ULONG *ModuleSizeInBytes,
    _Out_ ULONG *IgnoreRip
    );
typedef GET_FUNCTION_ENTRY_CALLBACK *PGET_FUNCTION_ENTRY_CALLBACK;

typedef
VOID
(NTAPI CLEAR_FUNCTION_ENTRY_CALLBACK)(
    _Out_opt_ PFUNCTION_ENTRY_CALLBACK *Callback,
    _Out_opt_ PVOID *Context,
    _Out_opt_ PVOID *ModuleBaseAddress,
    _Out_opt_ ULONG *ModuleSizeInBytes,
    _Out_opt_ ULONG *IgnoreRip
    );
typedef CLEAR_FUNCTION_ENTRY_CALLBACK *PCLEAR_FUNCTION_ENTRY_CALLBACK;

typedef
BOOLEAN
(NTAPI IS_FUNCTION_ENTRY_CALLBACK_ENABLED)(
    VOID
    );
typedef IS_FUNCTION_ENTRY_CALLBACK_ENABLED
      *PIS_FUNCTION_ENTRY_CALLBACK_ENABLED;

typedef
VOID
(_PENTER)(
    VOID
    );
typedef _PENTER *P_PENTER;

#else // PH_WINDOWS

typedef
ULONG
(__cdecl __C_SPECIFIC_HANDLER)(
    PVOID ExceptionRecord,
    ULONG_PTR Frame,
    PVOID Context,
    PVOID Dispatch
    );
typedef __C_SPECIFIC_HANDLER *P__C_SPECIFIC_HANDLER;

#endif

//
// Define bootstrap helpers.
//

typedef
_Success_(return >= 0)
HRESULT
(NTAPI PERFECT_HASH_PRINT_ERROR)(
    _In_ PCSZ FunctionName,
    _In_ PCSZ FileName,
    _In_ ULONG LineNumber,
    _In_ ULONG Error
    );
typedef PERFECT_HASH_PRINT_ERROR *PPERFECT_HASH_PRINT_ERROR;

typedef
_Success_(return >= 0)
HRESULT
(NTAPI PERFECT_HASH_PRINT_MESSAGE)(
    _In_ ULONG Code,
    ...
    );
typedef PERFECT_HASH_PRINT_MESSAGE *PPERFECT_HASH_PRINT_MESSAGE;

#endif // ifndef __CUDA_ARCH__

//
// Define an X-macro for the enum types used by the library.  The ENTRY macros
// receive (Name, Upper) as their arguments, e.g.: (CpuArch, CPU_ARCH)
//

#define PERFECT_HASH_ENUM_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    FIRST_ENTRY(CpuArch, CPU_ARCH)                              \
    ENTRY(Interface, INTERFACE)                                 \
    ENTRY(Algorithm, ALGORITHM)                                 \
    ENTRY(HashFunction, HASH_FUNCTION)                          \
    ENTRY(MaskFunction, MASK_FUNCTION)                          \
    ENTRY(BestCoverageType, BEST_COVERAGE_TYPE)                 \
    ENTRY(TableCreateParameter, TABLE_CREATE_PARAMETER)         \
    ENTRY(CuRng, CU_RNG)                                        \
    LAST_ENTRY(Rng, RNG)

#define PERFECT_HASH_ENUM_TABLE_ENTRY(ENTRY) \
    PERFECT_HASH_ENUM_TABLE(ENTRY, ENTRY, ENTRY)

#define EXPAND_AS_ENUM_ID_ENUM(Name, Upper) \
    PerfectHash##Name##EnumId,

//
// Define an enum of all enum types used by the library.
//

typedef enum _PERFECT_HASH_ENUM_ID {
    PerfectHashNullEnumId = 0,
    PERFECT_HASH_ENUM_TABLE_ENTRY(EXPAND_AS_ENUM_ID_ENUM)
    PerfectHashInvalidEnumId
} PERFECT_HASH_ENUM_ID;
typedef PERFECT_HASH_ENUM_ID *PPERFECT_HASH_ENUM_ID;

//
// Provide a simple inline CPU architecture enum validation routine.
//

FORCEINLINE
BOOLEAN
IsValidPerfectHashEnumId(
    _In_ PERFECT_HASH_ENUM_ID EnumId
    )
{
    return (
        EnumId > PerfectHashNullEnumId &&
        EnumId < PerfectHashInvalidEnumId
    );
}

//
// Define helper macros for printing errors to stdout.  Requires the symbol
// PerfectHashPrintError to be in scope.
//

#ifndef __CUDA_ARCH__
#define SYS_ERROR(Name) \
    PerfectHashPrintError(#Name, __FILE__, __LINE__, GetLastError())

#define PH_ERROR(Name, Result)           \
    PerfectHashPrintError(#Name,         \
                          __FILE__,      \
                          __LINE__,      \
                          (ULONG)Result)

#define PH_MESSAGE(Result) \
    PerfectHashPrintMessage((ULONG)Result)

#define PH_MESSAGE_ARGS(Result, ...) \
    PerfectHashPrintMessage((ULONG)Result, __VA_ARGS__)

//
// Our usage text has exceeded the maximum message limit of 64k, so we have
// to use multiple messages.
//

#define PH_USAGE()                                                         \
    PerfectHashPrintMessage((ULONG)PH_MSG_PERFECT_HASH_USAGE);             \
    PerfectHashPrintMessage((ULONG)PH_MSG_PERFECT_HASH_USAGE_CONTINUED_1);

#define PH_BREAK() __debugbreak()
#endif // ifndef __CUDA_ARCH__

//
// Helper inline that decorates RaiseException() with _Analysis_noreturn_,
// which we need for SAL to grok our PH_RAISE() calls correctly.
//

_Analysis_noreturn_
FORCEINLINE
VOID
PhRaiseException(
    _In_ DWORD dwExceptionCode,
    _In_ DWORD dwExceptionFlags,
    _In_ DWORD nNumberOfArguments,
    _In_reads_opt_(nNumberOfArguments) CONST ULONG_PTR* lpArguments
    )
{
#if PH_WINDOWS
    RaiseException(dwExceptionCode,
                   dwExceptionFlags,
                   nNumberOfArguments,
                   lpArguments);
#else
    __debugbreak();
    exit(dwExceptionCode);
#endif
}

//
// Helper macro for raising non-continuable exceptions.
//

#ifdef _DEBUG
#define PH_RAISE(Result)                                               \
    __debugbreak();                                                    \
    PhRaiseException((DWORD)Result, EXCEPTION_NONCONTINUABLE, 0, NULL)
#else
#define PH_RAISE(Result)                                               \
    __debugbreak();                                                    \
    PhRaiseException((DWORD)Result, EXCEPTION_NONCONTINUABLE, 0, NULL)
#endif

//
// Build type static strings.
//

#ifdef PERFECT_HASH_CMAKE

#ifndef STRINGAFY
#define STRINGAFY(x) #x
#endif

static const char PerfectHashBuildConfigString[] =
    STRINGAFY(PERFECT_HASH_BUILD_CONFIG);

#else

#ifdef PERFECT_HASH_BUILD_CONFIG_PGI
static const char PerfectHashBuildConfigString[] = "PGInstrument";
#elif defined(PERFECT_HASH_BUILD_CONFIG_PGU)
static const char PerfectHashBuildConfigString[] = "PGUpdate";
#elif defined(PERFECT_HASH_BUILD_CONFIG_PGO)
static const char PerfectHashBuildConfigString[] = "PGOptimize";
#elif defined(PERFECT_HASH_BUILD_CONFIG_RELEASE)
static const char PerfectHashBuildConfigString[] = "Release";
#elif defined(PERFECT_HASH_BUILD_CONFIG_DEBUG)
static const char PerfectHashBuildConfigString[] = "Debug";
#elif defined(__CUDA_ARCH__)
static const char PerfectHashBuildConfigString[] = "CUDA";
#else
#error Unknown build config type.
#endif

#ifndef __CUDA_ARCH__
#ifndef _PERFECT_HASH_INTERNAL_BUILD
#ifdef PH_WINDOWS
FORCEINLINE
_Success_(return >= 0)
HRESULT
PerfectHashBootstrap(
    _Out_ PICLASSFACTORY *ClassFactoryPointer,
    _Out_ PPERFECT_HASH_PRINT_ERROR *PrintErrorPointer,
    _Out_ PPERFECT_HASH_PRINT_MESSAGE *PrintMessagePointer,
    _Out_ HMODULE *ModulePointer
    )
/*++

Routine Description:

    This is a simple helper routine that loads the PerfectHash.dll library,
    obtains an IClassFactory interface, and an error handling function pointer.
    It is useful for using the library without needing any underlying system
    COM support (e.g. CoCreateInstance).

Arguments:

    ClassFactoryPointer - Supplies the address of a variable that will receive
        an instance of an ICLASSFACTORY interface if the routine is successful.
        Caller is responsible for calling ClassFactory->Vtbl->Release() when
        finished with the instance.

    PrintErrorPointer - Supplies the address of a variable that will receive the
        function pointer to the PerfectHashPrintError error handling routine.

    PrintMessagePointer - Supplies the address of a variable that will receive
        the function pointer to the PerfectHashPrintMessage routine.

    ModulePointer - Supplies the address of a variable that will receive the
        handle of the loaded module.  Callers can call FreeLibrary() against
        this handle when finished with the library.

Return Value:

    S_OK - Routine completed successfully.

    The following error codes may also be returned.  This is not an exhaustive
    list.

    E_FAIL - Failed to load PerfectHash.dll.

    E_UNEXPECTED - Failed to resolve symbol from loaded PerfectHash.dll.

--*/
{
    PVOID Proc;
    HRESULT Result;
    HMODULE Module;
    PPERFECT_HASH_PRINT_ERROR PerfectHashPrintError;
    PPERFECT_HASH_PRINT_MESSAGE PerfectHashPrintMessage;
    PDLL_GET_CLASS_OBJECT PhDllGetClassObject;
    PICLASSFACTORY ClassFactory;

    *ClassFactoryPointer = NULL;
    *PrintErrorPointer = NULL;
    *ModulePointer = NULL;

    Module = LoadLibraryA("PerfectHash.dll");
    if (!Module) {
        return E_FAIL;
    }

    Proc = (PVOID)GetProcAddress(Module, "PerfectHashPrintError");
    if (!Proc) {
        FreeLibrary(Module);
        return E_UNEXPECTED;
    }
    PerfectHashPrintError = (PPERFECT_HASH_PRINT_ERROR)Proc;

    Proc = (PVOID)GetProcAddress(Module, "PerfectHashPrintMessage");
    if (!Proc) {
        FreeLibrary(Module);
        return E_UNEXPECTED;
    }
    PerfectHashPrintMessage = (PPERFECT_HASH_PRINT_MESSAGE)Proc;

    Proc = (PVOID)GetProcAddress(Module, "PerfectHashDllGetClassObject");
    if (!Proc) {
        SYS_ERROR(GetProcAddress);
        FreeLibrary(Module);
        return E_UNEXPECTED;
    }
    PhDllGetClassObject = (PDLL_GET_CLASS_OBJECT)Proc;

    Result = PhDllGetClassObject(&CLSID_PERFECT_HASH,
                                 &IID_PERFECT_HASH_ICLASSFACTORY,
                                 &ClassFactory);

    if (FAILED(Result)) {
        FreeLibrary(Module);
        return Result;
    }

    *ClassFactoryPointer = ClassFactory;
    *PrintErrorPointer = PerfectHashPrintError;
    *PrintMessagePointer = PerfectHashPrintMessage;
    *ModulePointer = Module;

    return S_OK;
}
#else // ifdef PH_WINDOWS

extern PERFECT_HASH_PRINT_ERROR PerfectHashPrintError;
extern PERFECT_HASH_PRINT_MESSAGE PerfectHashPrintMessage;
extern DLL_GET_CLASS_OBJECT PerfectHashDllGetClassObject;

FORCEINLINE
_Success_(return >= 0)
HRESULT
PerfectHashBootstrap(
    _Out_ PICLASSFACTORY *ClassFactoryPointer,
    _Out_ PPERFECT_HASH_PRINT_ERROR *PrintErrorPointer,
    _Out_ PPERFECT_HASH_PRINT_MESSAGE *PrintMessagePointer,
    _Out_ HMODULE *ModulePointer
    )
/*++

Routine Description:

    This is a simple helper routine that loads the PerfectHash.dll library,
    obtains an IClassFactory interface, and an error handling function pointer.
    It is useful for using the library without needing any underlying system
    COM support (e.g. CoCreateInstance).

Arguments:

    ClassFactoryPointer - Supplies the address of a variable that will receive
        an instance of an ICLASSFACTORY interface if the routine is successful.
        Caller is responsible for calling ClassFactory->Vtbl->Release() when
        finished with the instance.

    PrintErrorPointer - Supplies the address of a variable that will receive the
        function pointer to the PerfectHashPrintError error handling routine.

    PrintMessagePointer - Supplies the address of a variable that will receive
        the function pointer to the PerfectHashPrintMessage routine.

    ModulePointer - Supplies the address of a variable that will receive the
        handle of the loaded module.  Callers can call FreeLibrary() against
        this handle when finished with the library.

Return Value:

    S_OK - Routine completed successfully.

    The following error codes may also be returned.  This is not an exhaustive
    list.

    E_FAIL - Failed to load PerfectHash.dll.

    E_UNEXPECTED - Failed to resolve symbol from loaded PerfectHash.dll.

--*/
{
    HRESULT Result;
    PICLASSFACTORY ClassFactory;

    Result = PerfectHashDllGetClassObject(&CLSID_PERFECT_HASH,
                                          &IID_PERFECT_HASH_ICLASSFACTORY,
                                          PPV(&ClassFactory));

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashDllGetClassObject, Result);
        return Result;
    }

    *ClassFactoryPointer = ClassFactory;
    *PrintErrorPointer = PerfectHashPrintError;
    *PrintMessagePointer = PerfectHashPrintMessage;
    *ModulePointer = NULL;

    return S_OK;
}
#endif // PH_WINDOWS
#endif // _PERFECT_HASH_INTERNAL_BUILD
#endif // __CUDA_ARCH__

#ifdef __cplusplus
} // extern "C"
#endif

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
