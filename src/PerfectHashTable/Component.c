/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Component.c

Abstract:

    This module implements COM-related routines for the perfect hash
    table library.

--*/

#include "stdafx.h"

//
// Globals for capturing component counts and server locks.
//

volatile LONG ComponentCount = 0;
volatile LONG ServerLockCount = 0;

_Must_inspect_result_
PCOMPONENT
CreateComponent(
    PERFECT_HASH_TABLE_INTERFACE_ID Id,
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

    if (!IsValidPerfectHashTableInterfaceId(Id)) {
        return NULL;
    }

    HeapHandle = GetProcessHeap();

    AllocSize = ComponentSizes[Id];

    Component = (PCOMPONENT)HeapAlloc(HeapHandle, HEAP_ZERO_MEMORY, AllocSize);
    if (!Component) {
        return NULL;
    }

    Component->Id = Id;
    Component->SizeOfStruct = (ULONG)AllocSize;
    Component->OuterUnknown = OuterUnknown;

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

    __debugbreak();
    for (Index = 0; Index < NumberOfFunctions; Index++) {
        *DestFunction++ = *SourceFunction++;
    }

    //
    // AddRef() before we call the custom initializer, if applicable, such that
    // we can invoke the appropriate rundown functionality via Release() if
    // the initializer fails.  Ditto for incrementing the component count.
    //

    Unknown = &Component->Unknown;
    Unknown->Vtbl->AddRef(Unknown);
    InterlockedIncrement(&ComponentCount);

    //
    // Call the component's custom initializer if applicable.
    //

    Initialize = ComponentInitializeRoutines[Id];
    if (Initialize) {
        Result = Initialize(Component);
        if (FAILED(Result)) {
            Unknown->Vtbl->Release(Unknown);
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
    PERFECT_HASH_TABLE_INTERFACE_ID Id;

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

    Id = PerfectHashTableInterfaceGuidToId(InterfaceId);

    if (!IsValidPerfectHashTableInterfaceId(Id)) {
        return E_NOINTERFACE;
    }

    //
    // Fast-path: return self if ID matches current component ID or the request
    // is for IUnknown or IClassFactory (which all of our components support).
    //

    Match = (
        Id == Component->Id                             ||
        Id == PerfectHashTableUnknownInterfaceId        ||
        Id == PerfectHashTableClassFactoryInterfaceId
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

    //
    // Call the component's custom rundown routine if applicable.
    //

    Rundown = ComponentRundownRoutines[Component->Id];
    if (Rundown) {
        Rundown(Component);
    }

    return (ULONG)Count;
}

COMPONENT_CREATE_INSTANCE ComponentCreateInstance;

_Use_decl_annotations_
HRESULT
ComponentCreateInstance(
    PCOMPONENT Component,
    PIUNKNOWN UnknownOuter,
    REFIID InterfaceId,
    PVOID *Interface
    )
{
    PCOMPONENT NewComponent;
    PERFECT_HASH_TABLE_INTERFACE_ID Id;

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

    if (UnknownOuter != NULL) {
        return CLASS_E_NOAGGREGATION;
    }

    //
    // Validate the IID and process the request.
    //

    Id = PerfectHashTableInterfaceGuidToId(InterfaceId);

    if (!IsValidPerfectHashTableInterfaceId(Id)) {
        return E_NOINTERFACE;
    }

    //
    // Create a new instance of the requested interface.
    //

    NewComponent = CreateComponent(Id, UnknownOuter);
    if (!NewComponent) {
        return E_OUTOFMEMORY;
    }

    //
    // Update the caller's pointer and return success.
    //

    *Interface = NewComponent;

    return S_OK;
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
// We use PerfectHashTableDllGetClassObject() instead of DllGetClassObject() as
// the latter is defined in system headers but has incorrect SAL annotations on
// it.  The SAL annotations on our internal DLL_GET_CLASS_OBJECT function are
// correct so we need to declare the function of that type in order to compile
// under max warnings.
//
// The PerfectHashTable.def file is responsible for exporting the correct name.
//

DLL_GET_CLASS_OBJECT PerfectHashTableDllGetClassObject;

_Use_decl_annotations_
HRESULT
PerfectHashTableDllGetClassObject(
    REFCLSID ClassId,
    REFIID InterfaceId,
    LPVOID *Interface
    )
{
    HRESULT Result;
    PICLASSFACTORY ClassFactory;
    PERFECT_HASH_TABLE_INTERFACE_ID Id;

    if (!ARGUMENT_PRESENT(Interface)) {
        return E_POINTER;
    } else {
        *Interface = NULL;
    }

    //
    // Validate the CLSID.
    //

    if (!InlineIsEqualGUID(ClassId, &CLSID_PERFECT_HASH_TABLE)) {
        return CLASS_E_CLASSNOTAVAILABLE;
    }

    //
    // Class ID was valid, proceed with class factory creation.
    //

    Id = PerfectHashTableClassFactoryInterfaceId;
    ClassFactory = (PICLASSFACTORY)CreateComponent(Id, NULL);
    if (!ClassFactory) {
        return E_OUTOFMEMORY;
    }

    //
    // Dispatch to the class factory's QueryInterface() method.
    //

    Result = ClassFactory->Vtbl->QueryInterface(ClassFactory,
                                                InterfaceId,
                                                Interface);

    ClassFactory->Vtbl->Release(ClassFactory);

    return Result;
}

HRESULT
PerfectHashTableDllCanUnloadNow(
    VOID
    )
{
    if (ComponentCount == 0 && ServerLockCount == 0) {
        return S_OK;
    }

    return S_FALSE;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
