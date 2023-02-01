/*++

Copyright (c) 2018-2023. Trent Nelson <trent@trent.me>

Module Name:

    Component.c

Abstract:

    This module implements COM-related routines for the perfect hash library.

--*/

#include "stdafx.h"

//
// Forward decls.
//

_Must_inspect_result_
_Success_(return != 0)
PCOMPONENT
CreateComponent(
    _In_ PERFECT_HASH_INTERFACE_ID Id,
    _Reserved_ PIUNKNOWN OuterUnknown
    );

//
// Globals for capturing component counts and server locks.
//

volatile LONG ComponentCount = 0;
volatile LONG ServerLockCount = 0;

//
// The "global" global components structure.
//

GLOBAL_COMPONENTS GlobalComponents = { 0 };

_Requires_exclusive_lock_held_(GlobalComponents.Lock)
FORCEINLINE
PINIT_ONCE
GetGlobalComponentInitOnce(
    _In_ PERFECT_HASH_INTERFACE_ID Id
    )
{
    SHORT Offset;
    PINIT_ONCE InitOnce;

    if (!IsValidPerfectHashInterfaceId(Id)) {
        PH_RAISE(PH_E_INVALID_INTERFACE_ID);
    }

    Offset = GlobalComponentsInterfaceOffsets[Id];

    if (Offset < 0) {
        PH_RAISE(PH_E_NOT_GLOBAL_INTERFACE_ID);
    }

    InitOnce = (PINIT_ONCE)RtlOffsetToPointer(&GlobalComponents, Offset);
    return InitOnce;
}


_Success_(return != 0)
_Requires_exclusive_lock_held_(GlobalComponents.Lock)
BOOL
CALLBACK
CreateGlobalComponentCallback(
    _Inout_ PINIT_ONCE InitOnce,
    _Inout_ PVOID Parameter,
    _Inout_opt_ PVOID *Context
    )
{
    SHORT Offset;
    PCOMPONENT Component;
    PINIT_ONCE ExpectedInitOnce;
    PCREATE_COMPONENT_PARAMS Params;

    Params = (PCREATE_COMPONENT_PARAMS)Parameter;

    if (!IsGlobalComponentInterfaceId(Params->Id)) {
        PH_RAISE(PH_E_NOT_GLOBAL_INTERFACE_ID);
    }

    Offset = GlobalComponentsInterfaceOffsets[Params->Id];
    ExpectedInitOnce = (PINIT_ONCE)(
        RtlOffsetToPointer(
            &GlobalComponents,
            Offset
        )
    );

    if (InitOnce != ExpectedInitOnce) {
        PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
    }

    Component = CreateComponent(Params->Id, NULL);

    if (!Component) {
        return FALSE;
    }

    if (ARGUMENT_PRESENT(Context)) {
        *Context = Component;
    }

    return TRUE;
}

PCOMPONENT
CreateGlobalComponent(
    PERFECT_HASH_INTERFACE_ID Id,
    PIUNKNOWN OuterUnknown
    )
{
    BOOL Success;
    PCOMPONENT Component;
    PINIT_ONCE InitOnce = NULL;
    CREATE_COMPONENT_PARAMS Params = { 0 };
    PPERFECT_HASH_TLS_CONTEXT TlsContext;

    TlsContext = PerfectHashTlsEnsureContext();

    AcquireGlobalComponentsLockExclusive();

    InitOnce = GetGlobalComponentInitOnce(Id);

    Params.Id = Id;
    Params.OuterUnknown = OuterUnknown;

    Success = InitOnceExecuteOnce(InitOnce,
                                  CreateGlobalComponentCallback,
                                  &Params,
                                  PPV(&Component));

    ReleaseGlobalComponentsLockExclusive();

    if (!Success) {
        SYS_ERROR(InitOnceExecuteOnce);
        TlsContext->LastError = GetLastError();
        if (TlsContext->LastResult == S_OK) {
            TlsContext->LastResult = PH_E_SYSTEM_CALL_FAILED;
        }
        return NULL;
    }

    return Component;
}

//
// Component functions.
//

_Use_decl_annotations_
PCOMPONENT
CreateComponent(
    PERFECT_HASH_INTERFACE_ID Id,
    PIUNKNOWN OuterUnknown
    )
{
    PVOID Interface;
    SIZE_T AllocSize;
    HRESULT Result;
    HANDLE HeapHandle;
    PIUNKNOWN Unknown;
    USHORT InterfaceOffset;
    SIZE_T Index;
    SIZE_T NumberOfFunctions;
    SIZE_T InterfaceSizeInBytes;
    PULONG_PTR DestFunction;
    PULONG_PTR SourceFunction;
    PCOMPONENT Component;
    PCOMPONENT_INITIALIZE Initialize;
    PPERFECT_HASH_TLS_CONTEXT TlsContext;

    if (!IsValidPerfectHashInterfaceId(Id)) {
        PH_ERROR(PerfectHashCreateComponent, PH_E_INVALID_INTERFACE_ID);
        return NULL;
    }

    TlsContext = PerfectHashTlsEnsureContext();

    HeapHandle = GetProcessHeap();

    AllocSize = ComponentSizes[Id];

    Component = (PCOMPONENT)HeapAlloc(HeapHandle, HEAP_ZERO_MEMORY, AllocSize);
    if (!Component) {
        SYS_ERROR(HeapAlloc);
        TlsContext->LastError = GetLastError();
        TlsContext->LastResult = E_OUTOFMEMORY;
        return NULL;
    }

    Component->Id = Id;
    Component->SizeOfStruct = (ULONG)AllocSize;
    Component->OuterUnknown = OuterUnknown;

    InitializeSRWLock(&Component->Lock);
    InitializeListHead(&Component->ListEntry);

    //
    // Wire up the vtable.
    //

    InterfaceOffset = ComponentInterfaceOffsets[Id];
    Component->Vtbl = (PCOMPONENT_VTBL)(
        RtlOffsetToPointer(
            Component,
            InterfaceOffset
        )
    );

    //
    // Copy the interface.
    //
    // N.B. We can't use CopyMemory() here (which expands to Rtl->RtlCopyMemory)
    //      as Rtl isn't available in this context.
    //

    Interface = (PVOID)ComponentInterfaces[Id];
    InterfaceSizeInBytes = ComponentInterfaceSizes[Id];
    NumberOfFunctions = InterfaceSizeInBytes / sizeof(ULONG_PTR);
    DestFunction = (PULONG_PTR)Component->Vtbl;
    SourceFunction = (PULONG_PTR)Interface;

    for (Index = 0; Index < NumberOfFunctions; Index++) {
        *DestFunction++ = *SourceFunction++;
    }

    //
    // AddRef() before we call the custom initializer, if applicable, such that
    // we can invoke the appropriate rundown functionality via Release() if
    // the initializer fails.  Ditto for incrementing the component count.
    //

    Unknown = &Component->Unknown;
    ASSERT(Unknown->ReferenceCount == 0);
    Unknown->Vtbl->AddRef(Unknown);
    ASSERT(Unknown->ReferenceCount == 1);

    if (InterlockedIncrement(&ComponentCount) == 1) {

        AcquireGlobalComponentsLockExclusive();

        if (GlobalComponents.FirstComponent != NULL) {
            PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
        }

        GlobalComponents.FirstComponent = Component;

        ReleaseGlobalComponentsLockExclusive();
    }

    //
    // Call the component's custom initializer if applicable.
    //

    Initialize = ComponentInitializeRoutines[Id];
    if (Initialize) {
        Result = Initialize(Component);
        if (FAILED(Result)) {
            if (Result != E_OUTOFMEMORY) {
                PH_ERROR(PerfectHashComponentInitialize, Result);
            }
            TlsContext->LastResult = Result;
            Unknown->Vtbl->Release(Unknown);
            AcquireGlobalComponentsLockExclusive();
            if (GlobalComponents.FirstComponent == Component) {
                GlobalComponents.FirstComponent = NULL;
            }
            ReleaseGlobalComponentsLockExclusive();
            Component = NULL;
        }
    }

    return Component;
}

COMPONENT_QUERY_INTERFACE ComponentQueryInterface;

_Use_decl_annotations_
HRESULT
ComponentQueryInterface(
    PCOMPONENT Component,
    REFIID InterfaceId,
    PVOID *Interface
    )
{
    BOOLEAN Match;
    PIUNKNOWN Unknown;
    PCOMPONENT NewComponent;
    PERFECT_HASH_INTERFACE_ID Id;

    //
    // Validate caller's pointer.
    //

    if (!ARGUMENT_PRESENT(Interface)) {
        return E_POINTER;
    }

    //
    // Clear the pointer up-front.
    //

    *Interface = NULL;

    //
    // Validate the IID and process the request.
    //

    Id = PerfectHashInterfaceGuidToId(InterfaceId);

    if (!IsValidPerfectHashInterfaceId(Id)) {
        return E_NOINTERFACE;
    }

    //
    // Fast-path: return self if ID matches current component ID or the request
    // is for IUnknown or IClassFactory (which all of our components support).
    //

    Match = (
        Id == Component->Id                         ||
        Id == PerfectHashIUnknownInterfaceId        ||
        Id == PerfectHashIClassFactoryInterfaceId
    );

    if (Match) {
        Unknown = &Component->Unknown;
        *Interface = Unknown;
        Unknown->Vtbl->AddRef(Unknown);
        return S_OK;
    }

    //
    // Create a new instance of the requested interface.
    //
    // (Should we support this, or should we force creation of new objects
    //  to go through CreateInstance()?)
    //

    NewComponent = CreateComponent(Id, NULL);
    if (!NewComponent) {
        return E_OUTOFMEMORY;
    }

    //
    // Update the caller's pointer and return success.
    //

    *Interface = NewComponent;

    return S_OK;
}

COMPONENT_ADD_REF ComponentAddRef;

_Use_decl_annotations_
ULONG
ComponentAddRef(
    PCOMPONENT Component
    )
{
    LONG Count;

    Count = InterlockedIncrement((PLONG)&Component->ReferenceCount);

    ASSERT(Count >= 1);

    return (ULONG)Count;
}

COMPONENT_RELEASE ComponentRelease;

_Use_decl_annotations_
ULONG
ComponentRelease(
    PCOMPONENT Component
    )
{
    LONG Count;
    PCOMPONENT_RUNDOWN Rundown;

    Count = InterlockedDecrement((PLONG)&Component->ReferenceCount);

    ASSERT(Count >= 0);

    if (Count > 0) {
        return Count;
    }

    ASSERT(InterlockedDecrement(&ComponentCount) >= 0);

    AcquireGlobalComponentsLockExclusive();

    if (GlobalComponents.FirstComponent != Component) {
        ReleaseGlobalComponentsLockExclusive();
    } else {

        PRTL Rtl;
        PALLOCATOR Allocator;

        Rtl = (PRTL)RtlInitOnceToPointer(GlobalComponents.Rtl.Ptr);
        Allocator = (PALLOCATOR)(
            RtlInitOnceToPointer(GlobalComponents.Allocator.Ptr)
        );

        GlobalComponents.Rtl.Ptr = NULL;
        GlobalComponents.Allocator.Ptr = NULL;
        GlobalComponents.FirstComponent = NULL;

        ReleaseGlobalComponentsLockExclusive();

        RELEASE(Rtl);
        RELEASE(Allocator);
    }

    //
    // Call the component's custom rundown routine if applicable.
    //

    Rundown = ComponentRundownRoutines[Component->Id];
    if (Rundown) {
        Rundown(Component);
    }

    //
    // Free the backing interface memory.
    //

    if (!HeapFree(GetProcessHeap(), 0, Component)) {
        SYS_ERROR(HeapFree);
    }

    return (ULONG)Count;
}

COMPONENT_CREATE_INSTANCE ComponentCreateInstance;

_Use_decl_annotations_
HRESULT
ComponentCreateInstance(
    PCOMPONENT Component,
    PIUNKNOWN OuterUnknown,
    REFIID InterfaceId,
    PVOID *Interface
    )
{
    PIUNKNOWN Unknown;
    HRESULT Result = S_OK;
    PERFECT_HASH_INTERFACE_ID Id;
    PCOMPONENT NewComponent = NULL;
    PPERFECT_HASH_TLS_CONTEXT TlsContext;
    PERFECT_HASH_TLS_CONTEXT LocalTlsContext;

    //
    // Validate parameters.
    //

    if (!ARGUMENT_PRESENT(Component)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Interface)) {
        return E_POINTER;
    } else {
        *Interface = NULL;
    }

    //
    // None of our interfaces support aggregation yet.
    //

    if (OuterUnknown != NULL) {
        return CLASS_E_NOAGGREGATION;
    }

    //
    // Validate the IID and process the request.
    //

    Id = PerfectHashInterfaceGuidToId(InterfaceId);

    if (!IsValidPerfectHashInterfaceId(Id)) {
        return E_NOINTERFACE;
    }

    //
    // Obtain the active TLS context, using our local stack-allocated one if
    // need be.
    //

    TlsContext = PerfectHashTlsGetOrSetContext(&LocalTlsContext);

    if (TlsContextTryCreateGlobalComponent(TlsContext, Id)) {

        SHORT Offset;

        //
        // Determine if there's a TLS override for this component.
        //

        Offset = ComponentInterfaceTlsContextOffsets[Id];
        NewComponent = *((PCOMPONENT *)RtlOffsetToPointer(TlsContext, Offset));

        if (!NewComponent) {

            //
            // No TLS override was set.  Create a new global component.
            //

            NewComponent = CreateGlobalComponent(Id, OuterUnknown);

            if (!NewComponent) {
                Result = TlsContext->LastResult;
                PH_ERROR(CreateGlobalComponent, Result);
                goto Error;
            }
        }

        //
        // If we get here, a TLS component was provided, or we created a new
        // global component.  Either way, we need to increment the reference
        // count to reflect the new reference being obtained by the caller.
        //

        Unknown = &NewComponent->Unknown;
        Unknown->Vtbl->AddRef(Unknown);

    } else {

        //
        // Create a new instance of the requested interface.
        //

        NewComponent = CreateComponent(Id, OuterUnknown);

        if (!NewComponent) {
            Result = TlsContext->LastResult;
            if (Result != E_OUTOFMEMORY) {
                PH_ERROR(CreateComponent, Result);
            }
            goto Error;
        }

    }

    //
    // If we get to here, NewComponent should be non-NULL.  Assert this, then
    // update the caller's pointer.
    //

    ASSERT(NewComponent);
    *Interface = NewComponent;

    goto End;

Error:

    ASSERT(!NewComponent);

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    PerfectHashTlsClearContextIfActive(&LocalTlsContext);

    return Result;
}

COMPONENT_LOCK_SERVER ComponentLockServer;

_Use_decl_annotations_
HRESULT
ComponentLockServer(
    PCOMPONENT Component,
    BOOL Lock
    )
{
    UNREFERENCED_PARAMETER(Component);

    if (Lock) {
        InterlockedIncrement(&ServerLockCount);
    } else {
        InterlockedDecrement(&ServerLockCount);
    }

    return S_OK;
}

//
// Standard DLL COM exports.
//

//
// We use PerfectHashDllGetClassObject() instead of DllGetClassObject() as the
// latter is defined in system headers but has incorrect SAL annotations on it.
// The SAL annotations on our internal DLL_GET_CLASS_OBJECT function are correct
// so we need to declare the function of that type in order to compile under max
// warnings.
//
// The PerfectHash.def file is responsible for exporting the correct name.
//

DLL_GET_CLASS_OBJECT PerfectHashDllGetClassObject;

_Use_decl_annotations_
HRESULT
PerfectHashDllGetClassObject(
    REFCLSID ClassId,
    REFIID InterfaceId,
    LPVOID *Interface
    )
{
    HRESULT Result;
    PICLASSFACTORY ClassFactory;
    PERFECT_HASH_INTERFACE_ID Id;
    PPERFECT_HASH_TLS_CONTEXT TlsContext;
    PERFECT_HASH_TLS_CONTEXT LocalTlsContext;

    if (!ARGUMENT_PRESENT(Interface)) {
        return E_POINTER;
    } else {
        *Interface = NULL;
    }

    //
    // Validate the CLSID.
    //

    if (!InlineIsEqualGUID(ClassId, &CLSID_PERFECT_HASH)) {
        return CLASS_E_CLASSNOTAVAILABLE;
    }

    TlsContext = PerfectHashTlsGetOrSetContext(&LocalTlsContext);

    //
    // Class ID was valid, proceed with class factory creation.
    //

    Id = PerfectHashIClassFactoryInterfaceId;
    ClassFactory = (PICLASSFACTORY)CreateComponent(Id, NULL);
    if (!ClassFactory) {
        Result = TlsContext->LastResult;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-value"
        _Analysis_assume_(Result < 0);
#pragma clang diagnostic pop

        goto End;
    }

    //
    // Dispatch to the class factory's QueryInterface() method.
    //

    Result = ClassFactory->Vtbl->QueryInterface(ClassFactory,
                                                InterfaceId,
                                                Interface);

    ClassFactory->Vtbl->Release(ClassFactory);

End:

    PerfectHashTlsClearContextIfActive(&LocalTlsContext);

    return Result;
}

HRESULT
PerfectHashDllCanUnloadNow(
    VOID
    )
{
    if (ComponentCount == 0 && ServerLockCount == 0) {
        return S_OK;
    }

    return S_FALSE;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
