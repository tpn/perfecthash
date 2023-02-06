/*++

Copyright (c) 2018-2023 Trent Nelson <trent@trent.me>

Module Name:

    Rtl.c

Abstract:

    This module implements functionality related to the Rtl component of
    the perfect hash table library.

--*/

#include "stdafx.h"


#ifdef PH_WINDOWS
BOOL
CloseEvent(
    _In_ _Post_ptr_invalid_ HANDLE Object
    )
{
    return CloseHandle(Object);
}

BOOL
CloseDirectory(
    _In_ _Post_ptr_invalid_ HANDLE Object
    )
{
    return CloseHandle(Object);
}
#endif

//
// Forward definitions.
//

#ifdef PH_WINDOWS
LOAD_SYMBOLS_FROM_MULTIPLE_MODULES LoadSymbolsFromMultipleModules;
SET_C_SPECIFIC_HANDLER SetCSpecificHandler;
#endif

#define EXPAND_AS_RTL_FUNCTION_NAME(Upper, Name) \
    #Name,

const PCSTR RtlFunctionNames[] = {
    RTL_FUNCTION_TABLE_ENTRY(EXPAND_AS_RTL_FUNCTION_NAME)
};
#define Names RtlFunctionNames

const PCZPCWSTR RtlModuleNames[] = {
    _RTL_MODULE_NAMES_HEAD
};
#define ModuleNames RtlModuleNames


#ifdef PH_WINDOWS

//
// As we don't link to the CRT, we don't get a __C_specific_handler entry,
// which the linker will complain about as soon as we use __try/__except.
// What we do is define a __C_specific_handler_impl pointer to the original
// function (that lives in ntoskrnl), then implement our own function by the
// same name that calls the underlying impl pointer.  In order to do this
// we have to disable some compiler/linker warnings regarding mismatched
// stuff.
//

P__C_SPECIFIC_HANDLER __C_specific_handler_impl = NULL;

#pragma warning(push)
#pragma warning(disable: 4028 4273 28251)

EXCEPTION_DISPOSITION
__cdecl
__C_specific_handler(
    PEXCEPTION_RECORD ExceptionRecord,
    ULONG_PTR Frame,
    PCONTEXT Context,
    struct _DISPATCHER_CONTEXT *Dispatch
    )
{
    return __C_specific_handler_impl(ExceptionRecord,
                                     Frame,
                                     Context,
                                     Dispatch);
}

#pragma warning(pop)

INIT_ONCE InitOnceCSpecificHandler = INIT_ONCE_STATIC_INIT;

BOOL
CALLBACK
SetCSpecificHandlerCallback(
    PINIT_ONCE InitOnce,
    PVOID Parameter,
    PVOID *Context
    )
{
    UNREFERENCED_PARAMETER(InitOnce);
    UNREFERENCED_PARAMETER(Context);

    __C_specific_handler_impl = (P__C_SPECIFIC_HANDLER)Parameter;

    return TRUE;
}

VOID
SetCSpecificHandler(
    _In_ P__C_SPECIFIC_HANDLER Handler
    )
{
    BOOL Status;

    Status = InitOnceExecuteOnce(&InitOnceCSpecificHandler,
                                 SetCSpecificHandlerCallback,
                                 (PVOID)Handler,
                                 NULL);

    //
    // This should never return FALSE.
    //

    ASSERT(Status);
}

#endif // PH_WINDOWS


//
// Initialize and rundown functions.
//

RTL_INITIALIZE RtlInitialize;

_Use_decl_annotations_
HRESULT
RtlInitialize(
    PRTL Rtl
    )
{
    BOOL Success;
    ULONG Index;
    HRESULT Result = S_OK;
    HMODULE *Module;
    PWSTR *Name;
    ULONG NumberOfModules;
    ULONG NumberOfSymbols;
    ULONG NumberOfResolvedSymbols;

#ifdef PH_WINDOWS

    //
    // Define an appropriately sized bitmap we can passed to LoadSymbols().
    //

    ULONG BitmapBuffer[(ALIGN_UP(ARRAYSIZE(Names),
                        sizeof(ULONG) << 3) >> 5)+1] = { 0 };
    RTL_BITMAP FailedBitmap;

    FailedBitmap.SizeOfBitMap = ARRAYSIZE(Names) + 1;
    FailedBitmap.Buffer = (PULONG)&BitmapBuffer;

    if (!ARGUMENT_PRESENT(Rtl)) {
        return E_POINTER;
    }

    //
    // Attempt to initialize the error handling scaffolding such that SYS_ERROR
    // is available as early as possible.
    //

    Rtl->SysErrorOutputHandle = GetStdHandle(STD_ERROR_HANDLE);

    //
    // Create an error message buffer.
    //

    InitializeSRWLock(&Rtl->SysErrorMessageBufferLock);
    AcquireRtlSysErrorMessageBufferLock(Rtl);

    Rtl->SizeOfSysErrorMessageBufferInBytes = PAGE_SIZE;

    Rtl->SysErrorMessageBuffer = (PCHAR)(
        HeapAlloc(GetProcessHeap(),
                  HEAP_ZERO_MEMORY,
                  Rtl->SizeOfSysErrorMessageBufferInBytes)
    );

    if (!Rtl->SysErrorMessageBuffer) {
        ReleaseRtlSysErrorMessageBufferLock(Rtl);
        return E_OUTOFMEMORY;
    }

    ReleaseRtlSysErrorMessageBufferLock(Rtl);

    //
    // Load the modules.
    //

    Name = (PWSTR *)ModuleNames;
    Module = Rtl->Modules;
    NumberOfModules = GetNumberOfRtlModules(Rtl);

    for (Index = 0; Index < NumberOfModules; Index++, Module++, Name++) {
        *Module = LoadLibraryW(*Name);
        if (!*Module) {
            SYS_ERROR(LoadLibraryW);
            goto Error;
        }
        Rtl->NumberOfModules++;
    }
    ASSERT(Rtl->NumberOfModules == NumberOfModules);

    //
    // Calculate the number of symbols in the function pointer array.
    //

    NumberOfSymbols = sizeof(RTL_FUNCTIONS) / sizeof(ULONG_PTR);
    ASSERT(NumberOfSymbols == ARRAYSIZE(Names));

    //
    // Load the symbols.
    //

    Success = LoadSymbolsFromMultipleModules(Names,
                                             NumberOfSymbols,
                                             (PULONG_PTR)&Rtl->RtlFunctions,
                                             NumberOfSymbols,
                                             Rtl->Modules,
                                             (USHORT)Rtl->NumberOfModules,
                                             &FailedBitmap,
                                             &NumberOfResolvedSymbols);

    if (!Success) {
        Result = PH_E_RTL_LOAD_SYMBOLS_FROM_MULTIPLE_MODULES_FAILED;
        goto Error;
    }

    ASSERT(NumberOfSymbols == NumberOfResolvedSymbols);

    SetCSpecificHandler(Rtl->__C_specific_handler);

    Success = CryptAcquireContextW(&Rtl->CryptProv,
                                   NULL,
                                   NULL,
                                   PROV_RSA_FULL,
                                   CRYPT_VERIFYCONTEXT);

    if (!Success) {
        SYS_ERROR(CryptAcquireContextW);
        goto Error;
    }

    Result = RtlInitializeCpuFeatures(Rtl);
    if (FAILED(Result)) {
        PH_ERROR(RtlInitializeCpuFeatures, Result);
        goto Error;
    }

    Result = RtlInitializeLargePages(Rtl);
    if (FAILED(Result)) {
        PH_ERROR(RtlInitializeLargePages, Result);
        goto Error;
    }

#if defined(_M_AMD64) || defined(_M_X64)
    if (Rtl->CpuFeatures.AVX2 != FALSE) {
        Rtl->Vtbl->CopyPages = RtlCopyPages_AVX2;
        Rtl->Vtbl->FillPages = RtlFillPages_AVX2;
    }
#endif
#else // PH_WINDOWS

    //
    // Compat initialization.
    //

    Rtl->RtlNumberOfSetBits = RtlNumberOfSetBits;
    Rtl->RtlEqualUnicodeString = RtlEqualUnicodeString;
    Rtl->RtlFindLongestRunClear = RtlFindLongestRunClear;
    Rtl->RtlUnicodeStringToInt64 = RtlUnicodeStringToInt64;
    Rtl->RtlUnicodeStringToInteger = RtlUnicodeStringToInteger;
    Rtl->RtlAppendUnicodeStringToString = RtlAppendUnicodeStringToString;

#endif

    //
    // We're done, indicate success and finish up.
    //

    Result = S_OK;
    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}

RTL_RUNDOWN RtlRundown;

_Use_decl_annotations_
VOID
RtlRundown(
    PRTL Rtl
    )
{
    ULONG Index;
    HMODULE *Module;
    PVOID Buffer;

    if (!ARGUMENT_PRESENT(Rtl)) {
        return;
    }

#ifdef PH_WINDOWS
    if (Rtl->CryptProv) {

        if (!CryptReleaseContext(Rtl->CryptProv, 0)) {
            SYS_ERROR(CryptReleaseContext);
        }
        Rtl->CryptProv = 0;
    }

    Buffer = Rtl->CpuFeatures.ProcInfoArray.ProcInfo;
    if (Buffer != NULL) {
        if (!VirtualFree(Buffer, 0, MEM_RELEASE)) {
            SYS_ERROR(VirtualFree);
        }
        Rtl->CpuFeatures.ProcInfoArray.ProcInfo = NULL;
    }

    //
    // Free any loaded modules.
    //

    Module = Rtl->Modules;

    for (Index = 0; Index < Rtl->NumberOfModules; Index++) {
        if (!FreeLibrary(*Module)) {
            SYS_ERROR(FreeLibrary);
        }
        *Module++ = 0;
    }

    //
    // Free the sys error buffer.
    //

    if (Rtl->SysErrorMessageBuffer) {
        if (!HeapFree(GetProcessHeap(), 0, Rtl->SysErrorMessageBuffer)) {

            //
            // We can't reliably report this with SYS_ERROR() as the buffer is
            // in an unknown state.
            //

            NOTHING;
        }

        Rtl->SysErrorMessageBuffer = NULL;
    }
#endif

}

_Use_decl_annotations_
HRESULT
GetContainingType(
    PRTL Rtl,
    ULONG_PTR Value,
    PTYPE TypePointer
    )
/*++

Routine Description:

    Obtain the appropriate type for a given power-of-2 based value.

Arguments:

    Rtl - Supplies a pointer to an RTL instance.

    Value - Supplies a power-of-2 value for which the type is to be obtained.

    TypePointer - Receives the corresponding type on success.

Return Value:

    S_OK - Type was obtained successfully.

    E_INVALIDARG - Value was not a power-of-2 (more than 1 bit was set).

    E_UNEXPECTED - Internal error.

--*/
{
    TYPE Type;
    ULONG_PTR Bits;
    ULONG_PTR Trailing;

    //
    // If no bits are set in the incoming value, default to ByteType.
    //

    if (Value == 0) {
        *TypePointer = ByteType;
        return S_OK;
    }

    //
    // Count the number of bits in the value.  If more than one bit is present,
    // error out; the value isn't a power-of-2.
    //

    Bits = Rtl->PopulationCountPointer(Value);
    if (Bits > 1) {
        return E_INVALIDARG;
    }

    //
    // Count the number of trailing zeros.
    //

    Trailing = Rtl->TrailingZerosPointer(Value);

    switch (Trailing) {
        case 0:
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
            Type = ByteType;
            break;

        case 8:
        case 9:
        case 10:
        case 11:
        case 12:
        case 13:
        case 14:
        case 15:
        case 16:
            Type = ShortType;
            break;

        case 17:
        case 18:
        case 19:
        case 20:
        case 21:
        case 22:
        case 23:
        case 24:
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
        case 30:
        case 31:
            Type = LongType;
            break;

        default:
            Type = LongLongType;
            break;

    }

    //
    // Update the caller's pointer and return success.
    //

    *TypePointer = Type;
    return S_OK;
}


#ifdef PH_WINDOWS
#if defined(_M_AMD64) || defined(_M_X64) || defined(_M_IX86)
HRESULT
RtlInitializeCpuFeaturesLogicalProcessors(
    _In_ PRTL Rtl
    )
/*++

Routine Description:

    This routine initializes the logical processor information of the CPU
    features structure in the provided Rtl instance.

Arguments:

    Rtl - Supplies a pointer to an RTL instance.

Return Value:

    S_OK - Success.

    PH_E_SYSTEM_CALL_FAILED - System call failed.

--*/
{
    SIZE_T Index;
    BOOL Success;
    HRESULT Result;
    DWORD LastError;
    PVOID ProcInfoBuffer;
    DWORD ProcInfoLength;
    ULONG ProcessorMaskBits;
    PCPU_CACHES Caches;
    PCPU_CACHE_LEVEL CacheLevel;
    PCACHE_DESCRIPTOR Cache;
    PCACHE_DESCRIPTOR CacheDesc;
    PRTL_CPU_FEATURES Features;
    PSYSTEM_LOGICAL_PROCESSOR_INFO_ARRAY ProcInfoArray;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ProcInfo;

    Features = &Rtl->CpuFeatures;

    //
    // Obtain the processor info.
    //

    ProcInfoLength = 0;
    Success = GetLogicalProcessorInformation(NULL, &ProcInfoLength);
    if (Success) {
        PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
    } else {
        LastError = GetLastError();
        if (LastError != ERROR_INSUFFICIENT_BUFFER) {
            SYS_ERROR(GetLogicalProcessorInformation);
            Result = PH_E_SYSTEM_CALL_FAILED;
            goto End;
        }
    }

    ProcInfoBuffer = VirtualAlloc(NULL,
                                  ProcInfoLength,
                                  MEM_COMMIT | MEM_RESERVE,
                                  PAGE_READWRITE);

    if (!ProcInfoBuffer) {
        Result = E_OUTOFMEMORY;
        goto End;
    }

    Success = GetLogicalProcessorInformation(ProcInfoBuffer, &ProcInfoLength);
    if (!Success) {
        SYS_ERROR(GetLogicalProcessorInformation);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto End;
    }

    ProcInfoArray = &Features->ProcInfoArray;
    ProcInfoArray->Count = ProcInfoLength / sizeof(*ProcInfo);
    ProcInfoArray->SizeInBytes = ProcInfoLength;

    ProcInfoArray->ProcInfo = (
        (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)ProcInfoBuffer
    );

    Caches = &Features->Caches;

    for (Index = 0; Index < ProcInfoArray->Count; Index++) {

        ProcInfo = &ProcInfoArray->ProcInfo[Index];

        switch (ProcInfo->Relationship) {
            case RelationNumaNode:
                Features->NumaNodeCount++;
                break;

            case RelationProcessorCore:
                Features->ProcessorCoreCount++;

                ProcessorMaskBits = (ULONG)(
                    Rtl->PopulationCountPointer(ProcInfo->ProcessorMask)
                );
                Features->LogicalProcessorCount += ProcessorMaskBits;
                break;

            case RelationProcessorPackage:
                Features->ProcessorPackageCount++;
                break;

            case RelationCache:
                CacheDesc = &ProcInfo->Cache;

                if (CacheDesc->Level > 4) {
                    break;
                }

                Caches->NumberOfLevels = max(Caches->NumberOfLevels,
                                             CacheDesc->Level);

                CacheLevel = &Caches->Level[CacheDesc->Level-1];
                Cache = &CacheLevel->AsArray[CacheDesc->Type];
                if (Cache->Level == 0) {
                    CopyMemoryInline(Cache, CacheDesc, sizeof(*Cache));
                } else {
                    Cache->Size += CacheDesc->Size;
                }
                break;

            case RelationAll:
                break;

            case RelationGroup:
                break;

            case RelationNumaNodeEx:
                break;

            case RelationProcessorDie:
                break;

            case RelationProcessorModule:
                break;

            default:
                break;
        }

    }

    Features->Flags.HasProcessorInformation = TRUE;
    Result = S_OK;

    //
    // Intentional follow-on.
    //

End:

    return Result;
}


_Use_decl_annotations_
HRESULT
RtlInitializeCpuFeatures(
    PRTL Rtl
    )
/*++

Routine Description:

    This routine calls the x86/x64 CPUID function and initializes the CPU
    features structure in the provided Rtl instance.

Arguments:

    Rtl - Supplies a pointer to an RTL instance.

Return Value:

    S_OK - Success.

    PH_E_SYSTEM_CALL_FAILED - System call failed.

--*/
{
    HRESULT Result;
    CPU_INFO CpuInfo;
    PRTL_CPU_FEATURES Features;
    const LONG BaseExtendedId = 0x80000000;
    LONG ExtendedId;
    LONG HighestId;
    LONG HighestExtendedId;
    PSTRING Brand;

    Features = &Rtl->CpuFeatures;
    ZeroStruct(CpuInfo);

    __cpuid((PINT)&CpuInfo.AsIntArray, 0);

    HighestId = CpuInfo.Eax;

    Features->Vendor.IsIntel = (
        CpuInfo.Ebx == (LONG)'uneG' &&
        CpuInfo.Ecx == (LONG)'letn' &&
        CpuInfo.Edx == (LONG)'Ieni'
    );

    if (!Features->Vendor.IsIntel) {
        Features->Vendor.IsAMD = (
            CpuInfo.Ebx == (LONG)'htuA' &&
            CpuInfo.Ecx == (LONG)'DMAc' &&
            CpuInfo.Edx == (LONG)'itne'
        );

        if (!Features->Vendor.IsAMD) {
            Features->Vendor.Unknown = TRUE;
        }
    }

    if (HighestId >= 1) {
        ZeroStruct(CpuInfo);
        __cpuidex((PINT)&CpuInfo.AsIntArray, 1, 0);

        Features->F1Ecx.AsLong = CpuInfo.Ecx;
        Features->F1Edx.AsLong = CpuInfo.Edx;
    }

    if (HighestId >= 7) {
        ZeroStruct(CpuInfo);
        __cpuidex((PINT)&CpuInfo.AsIntArray, 7, 0);

        Features->F7Ebx.AsLong = CpuInfo.Ebx;
        Features->F7Ecx.AsLong = CpuInfo.Ecx;
    }

    ZeroStruct(CpuInfo);
    __cpuid((PINT)&CpuInfo.AsIntArray, BaseExtendedId);
    HighestExtendedId = CpuInfo.Eax;

    ExtendedId = BaseExtendedId + 1;
    if (HighestExtendedId >= ExtendedId) {
        ZeroStruct(CpuInfo);
        __cpuidex((PINT)&CpuInfo.AsIntArray, ExtendedId, 0);

        Features->F81Ecx.AsLong = CpuInfo.Ecx;
        Features->F81Edx.AsLong = CpuInfo.Edx;
    }

    Features->HighestFeatureId = HighestId;
    Features->HighestExtendedFeatureId = HighestExtendedId;

    if (Features->Vendor.IsIntel) {
        Features->Intel.HLE = Features->HLE;
        Features->Intel.RTM = Features->RTM;
        Features->Intel.BMI1 = Features->BMI1;
        Features->Intel.BMI2 = Features->BMI2;
        Features->Intel.LZCNT = Features->LZCNT;
        Features->Intel.POPCNT = Features->POPCNT;
        Features->Intel.SYSCALL = Features->SYSCALLSYSRET;
        Features->Intel.RDTSCP = Features->RDTSCP_IA32_TSC_AUX;
    } else if (Features->Vendor.IsAMD) {
        LONG F81Ecx;
        LONG F81Edx;

        F81Ecx = Features->F81Ecx.AsLong;
        F81Edx = Features->F81Edx.AsLong;

        Features->AMD.ABM = BooleanFlagOn(F81Ecx, 1 << 5);
        Features->AMD.SSE4A = BooleanFlagOn(F81Ecx, 1 << 6);
        Features->AMD.XOP = BooleanFlagOn(F81Ecx, 1 << 11);
        Features->AMD.TBM = BooleanFlagOn(F81Ecx, 1 << 21);
        Features->AMD.SVM = BooleanFlagOn(F81Ecx, 1 << 2);
        Features->AMD.IBS = BooleanFlagOn(F81Ecx, 1 << 10);
        Features->AMD.LWP = BooleanFlagOn(F81Ecx, 1 << 15);
        Features->AMD.MMXEXT = BooleanFlagOn(F81Edx, 1 << 22);
        Features->AMD.THREEDNOW = BooleanFlagOn(F81Edx, 1 << 31);
        Features->AMD.THREEDNOWEXT = BooleanFlagOn(F81Edx, 1 << 30);
    }

    //
    // Capture the CPU brand string if available.
    //

    Brand = &Features->Brand;
    Brand->Length = 0;
    Brand->MaximumLength = sizeof(Features->BrandBuffer);
    Brand->Buffer = (PCHAR)&Features->BrandBuffer;

    if (HighestExtendedId >= (BaseExtendedId + 4)) {

        __cpuid((PINT)&CpuInfo.AsIntArray, BaseExtendedId + 2);
        CopyMemory(Brand->Buffer, CpuInfo.AsCharArray, 16);

        __cpuid((PINT)&CpuInfo.AsIntArray, BaseExtendedId + 3);
        CopyMemory(Brand->Buffer + 16, CpuInfo.AsCharArray, 16);

        __cpuid((PINT)&CpuInfo.AsIntArray, BaseExtendedId + 4);
        CopyMemory(Brand->Buffer + 32, CpuInfo.AsCharArray, 16);

        Brand->Length = (USHORT)strlen(Brand->Buffer);
        _Analysis_assume_(Brand->Length <= 48);
        ASSERT(Brand->Length < Brand->MaximumLength);

        //
        // Disable the following warning:
        //
        //  warning C6385: Reading invalid data from 'Brand->Buffer':
        //      the readable size is '_Old_13`16' bytes, but
        //      'Brand->Length' bytes may be read.
        //

#pragma warning(push)
#pragma warning(disable: 6385)
        ASSERT(Brand->Buffer[Brand->Length] == '\0');
#pragma warning(pop)

    }

    //
    // Initialize the bit manipulation features next, then the processor info
    // (which needs PopulationCount).
    //

    Result = RtlInitializeBitManipulationFunctionPointers(Rtl);
    if (FAILED(Result)) {
        PH_ERROR(RtlInitializeBitManipulationFunctionPointers, Result);
        goto End;
    }

    Result = RtlInitializeCpuFeaturesLogicalProcessors(Rtl);
    if (FAILED(Result)) {
        PH_ERROR(RtlInitializeCpuFeaturesLogicalProcessors, Result);
        goto End;
    }

    //
    // Intentional follow-on.
    //

End:

    return Result;
}
#endif


_Use_decl_annotations_
HRESULT
RtlInitializeBitManipulationFunctionPointers(
    PRTL Rtl
    )
/*++

Routine Description:

    This routine initializes the bit mainpulation function pointers in the
    provided Rtl instance based on the CPU features.

Arguments:

    Rtl - Supplies a pointer to an RTL instance.

Return Value:

    S_OK - Success.

--*/
{
    PRTL_CPU_FEATURES Features;
    PRTL_BIT_MANIPULATION_FUNCTIONS Functions;

    Features = &Rtl->CpuFeatures;
    Functions = &Rtl->RtlBitManipulationFunctions;

    if (Features->Vendor.IsIntel != FALSE) {

#define EXPAND_AS_INTEL_FUNCTION_ASSIGNMENT(Upper,        \
                                            Name,         \
                                            IntelFeature, \
                                            Unused4)      \
    if (Features->Intel.IntelFeature != FALSE) {          \
        Functions->Name = Name##_##IntelFeature;          \
    } else {                                              \
        Functions->Name = Name##_C;                       \
    }

    RTL_BIT_MANIPULATION_FUNCTION_TABLE_ENTRY(
        EXPAND_AS_INTEL_FUNCTION_ASSIGNMENT
    )

#undef EXPAND_AS_INTEL_FUNCTION_ASSIGNMENT

    } else if (Features->Vendor.IsAMD != FALSE) {

#define EXPAND_AS_AMD_FUNCTION_ASSIGNMENT(Upper,        \
                                          Name,         \
                                          IntelFeature, \
                                          AmdFeature)   \
    if (Features->AMD.AmdFeature != FALSE) {            \
        Functions->Name = Name##_##IntelFeature;        \
    } else {                                            \
        Functions->Name = Name##_C;                     \
    }

    RTL_BIT_MANIPULATION_FUNCTION_TABLE_ENTRY(EXPAND_AS_AMD_FUNCTION_ASSIGNMENT)

#undef EXPAND_AS_AMD_FUNCTION_ASSIGNMENT

    } else {

#define EXPAND_AS_C_FUNCTION_ASSIGNMENT(Upper, Name, Unused3, Unused4) \
    Functions->Name = Name##_C;

    RTL_BIT_MANIPULATION_FUNCTION_TABLE_ENTRY(EXPAND_AS_C_FUNCTION_ASSIGNMENT);

#undef EXPAND_AS_C_FUNCTION_ASSIGNMENT

    }

    return S_OK;

}
#endif // defined(_M_AMD64) || defined(_M_X64) || defined(_M_IX86)

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
