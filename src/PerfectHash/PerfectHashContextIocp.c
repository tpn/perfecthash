/*++

Copyright (c) 2018-2026 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashContextIocp.c

Abstract:

    This module implements the IOCP-backed perfect hash context component.
    The implementation is currently scaffolded to allow incremental bring-up
    of the IOCP runtime, NUMA-aware worker pools, and bulk/table create
    workflows.

--*/

#include "stdafx.h"
#include "PerfectHashIocpBufferPool.h"

//
// Forward decls for internal helpers.
//

static
HRESULT
PerfectHashContextIocpEnumerateNumaNodes(
    _In_ PPERFECT_HASH_CONTEXT_IOCP ContextIocp
    );

static
HRESULT
PerfectHashContextIocpCreateIoCompletionPorts(
    _In_ PPERFECT_HASH_CONTEXT_IOCP ContextIocp
    );

static
HRESULT
PerfectHashContextIocpCreateWorkerThreads(
    _In_ PPERFECT_HASH_CONTEXT_IOCP ContextIocp
    );

static
VOID
PerfectHashIocpApplyNodeAffinity(
    _In_ PPERFECT_HASH_IOCP_NODE Node
    );

static
VOID
PerfectHashIocpFreeBuffer(
    _In_ PRTL Rtl,
    _In_ PPERFECT_HASH_IOCP_BUFFER Buffer
    );

static
VOID
PerfectHashIocpRundownBufferList(
    _In_ PRTL Rtl,
    _In_ PGUARDED_LIST BufferList
    );

static
VOID
PerfectHashIocpRundownOversizePools(
    _In_ PALLOCATOR Allocator,
    _Inout_ PLIST_ENTRY ListHead,
    _Inout_opt_ PULONG PoolCount
    );

static
#ifdef PH_WINDOWS
DWORD
WINAPI
PerfectHashIocpWorkerThreadProc(
    _In_ PVOID Parameter
    );
#else
VOID
PerfectHashIocpWorkerThreadProc(
    _In_ PVOID Parameter
    );
#endif

static
VOID
PerfectHashIocpFreeBuffer(
    _In_ PRTL Rtl,
    _In_ PPERFECT_HASH_IOCP_BUFFER Buffer
    )
{
    BOOL Success;
    HRESULT Result;
    HANDLE ProcessHandle;
    PVOID BaseAddress;

    if (!ARGUMENT_PRESENT(Rtl) || !ARGUMENT_PRESENT(Buffer)) {
        return;
    }

    ProcessHandle = NULL;
    if (Buffer->OwnerPool) {
        ProcessHandle = Buffer->OwnerPool->ProcessHandle;
    }

    if (!ProcessHandle) {
        ProcessHandle = GetCurrentProcess();
    }

    if (Buffer->Flags & PERFECT_HASH_IOCP_BUFFER_FLAG_GUARD_PAGES) {
        BaseAddress = Buffer;
        Result = Rtl->Vtbl->DestroyBuffer(Rtl,
                                          ProcessHandle,
                                          &BaseAddress,
                                          Buffer->AllocationSize);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashIocpFreeBuffer_DestroyBuffer, Result);
        }
        return;
    }

    Success = VirtualFreeEx(ProcessHandle, Buffer, 0, MEM_RELEASE);
    if (!Success) {
        SYS_ERROR(VirtualFreeEx);
    }
}

static
VOID
PerfectHashIocpRundownBufferList(
    _In_ PRTL Rtl,
    _In_ PGUARDED_LIST BufferList
    )
{
    BOOLEAN NotEmpty;
    PLIST_ENTRY Entry;
    PPERFECT_HASH_IOCP_BUFFER Buffer;

    if (!ARGUMENT_PRESENT(Rtl) || !ARGUMENT_PRESENT(BufferList)) {
        return;
    }

    while (TRUE) {
        Entry = NULL;
        NotEmpty = BufferList->Vtbl->RemoveHeadEx(BufferList, &Entry);
        if (!NotEmpty) {
            break;
        }

        Buffer = CONTAINING_RECORD(Entry,
                                   PERFECT_HASH_IOCP_BUFFER,
                                   ListEntry);

        PerfectHashIocpFreeBuffer(Rtl, Buffer);
    }

    BufferList->Vtbl->Reset(BufferList);
}

static
VOID
PerfectHashIocpRundownOversizePools(
    _In_ PALLOCATOR Allocator,
    _Inout_ PLIST_ENTRY ListHead,
    _Inout_opt_ PULONG PoolCount
    )
{
    PLIST_ENTRY Entry;
    PPERFECT_HASH_IOCP_BUFFER_POOL Pool;

    if (!ARGUMENT_PRESENT(Allocator) || !ARGUMENT_PRESENT(ListHead)) {
        return;
    }

    while (!IsListEmpty(ListHead)) {
        Entry = RemoveHeadList(ListHead);
        Pool = CONTAINING_RECORD(Entry,
                                 PERFECT_HASH_IOCP_BUFFER_POOL,
                                 ListEntry);
        Allocator->Vtbl->FreePointer(Allocator, (PVOID *)&Pool);
    }

    InitializeListHead(ListHead);
    if (PoolCount) {
        *PoolCount = 0;
    }
}

PERFECT_HASH_CONTEXT_IOCP_INITIALIZE PerfectHashContextIocpInitialize;

_Use_decl_annotations_
HRESULT
PerfectHashContextIocpInitialize(
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp
    )
/*++

Routine Description:

    Initializes an IOCP-backed perfect hash table context.  This routine
    creates the Rtl and Allocator components and seeds default configuration
    values.  IOCP resources are brought online in later steps.

Arguments:

    ContextIocp - Supplies a pointer to a PERFECT_HASH_CONTEXT_IOCP structure
        for which initialization is to be performed.

Return Value:

    S_OK on success, an appropriate error code on failure.

--*/
{
    HRESULT Result;
    PRTL Rtl;

    if (!ARGUMENT_PRESENT(ContextIocp)) {
        return E_POINTER;
    }

    Result = ContextIocp->Vtbl->CreateInstance(ContextIocp,
                                               NULL,
                                               &IID_PERFECT_HASH_RTL,
                                               &ContextIocp->Rtl);

    if (FAILED(Result)) {
        return Result;
    }

    Result = ContextIocp->Vtbl->CreateInstance(ContextIocp,
                                               NULL,
                                               &IID_PERFECT_HASH_ALLOCATOR,
                                               &ContextIocp->Allocator);

    if (FAILED(Result)) {
        return Result;
    }

    Rtl = ContextIocp->Rtl;

    ContextIocp->IocpConcurrency = 0;
    ContextIocp->MaxWorkerThreads = 0;

    ContextIocp->NumaNodeCount = Rtl->CpuFeatures.NumaNodeCount;
    if (ContextIocp->NumaNodeCount == 0) {
        ContextIocp->NumaNodeCount = 1;
    }

    ContextIocp->NumaNodeMask = PERFECT_HASH_NUMA_NODE_MASK_ALL;
    ContextIocp->Flags.UseNumaNodeMask = FALSE;
    ContextIocp->State.Initialized = TRUE;
    ContextIocp->CompletionCallback = NULL;
    ContextIocp->CompletionContext = NULL;

    ContextIocp->StartedEvent = CreateEventW(NULL, TRUE, FALSE, NULL);
    if (!ContextIocp->StartedEvent) {
        return PH_E_SYSTEM_CALL_FAILED;
    }

    ContextIocp->ShutdownEvent = CreateEventW(NULL, TRUE, FALSE, NULL);
    if (!ContextIocp->ShutdownEvent) {
        return PH_E_SYSTEM_CALL_FAILED;
    }

    return S_OK;
}

PERFECT_HASH_CONTEXT_IOCP_RUNDOWN PerfectHashContextIocpRundown;

_Use_decl_annotations_
VOID
PerfectHashContextIocpRundown(
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp
    )
/*++

Routine Description:

    Rundowns an IOCP-backed perfect hash table context.

Arguments:

    ContextIocp - Supplies a pointer to a PERFECT_HASH_CONTEXT_IOCP structure
        to rundown.

Return Value:

    None.

--*/
{
    ULONG Index;
    PRTL Rtl;
    PALLOCATOR Allocator;

    if (!ARGUMENT_PRESENT(ContextIocp)) {
        return;
    }

    Rtl = ContextIocp->Rtl;
    Allocator = ContextIocp->Allocator;

    if (ContextIocp->State.Running) {
        PerfectHashContextIocpStop(ContextIocp);
    }

    if (ContextIocp->Nodes) {
        for (Index = 0; Index < ContextIocp->NodeCount; Index++) {
            PPERFECT_HASH_IOCP_NODE Node = &ContextIocp->Nodes[Index];
            ULONG ThreadIndex;

            if (Node->FileWorkBufferList && Rtl) {
                PerfectHashIocpRundownBufferList(Rtl,
                                                 Node->FileWorkBufferList);
            }

            if (Allocator) {
                if (Node->FileWorkBufferPools) {
                    Allocator->Vtbl->FreePointer(
                        Allocator,
                        (PVOID *)&Node->FileWorkBufferPools
                    );
                }

                PerfectHashIocpRundownOversizePools(
                    Allocator,
                    &Node->FileWorkOversizePools,
                    &Node->FileWorkOversizePoolCount
                );
            }

            Node->FileWorkBufferPoolCount = 0;

            RELEASE(Node->FileWorkBufferList);

            if (Node->WorkerThreads) {
                for (ThreadIndex = 0;
                     ThreadIndex < Node->WorkerThreadCount;
                     ThreadIndex++) {
                    if (Node->WorkerThreads[ThreadIndex]) {
                        CloseHandle(Node->WorkerThreads[ThreadIndex]);
                    }
                }
                if (Allocator) {
                    Allocator->Vtbl->FreePointer(
                        Allocator,
                        (PVOID *)&Node->WorkerThreads
                    );
                }
            }

            if (Node->IoCompletionPort) {
                CloseHandle(Node->IoCompletionPort);
            }
        }
        if (Allocator) {
            Allocator->Vtbl->FreePointer(
                Allocator,
                (PVOID *)&ContextIocp->Nodes
            );
        }
    }

    if (ContextIocp->ShutdownEvent) {
        CloseHandle(ContextIocp->ShutdownEvent);
        ContextIocp->ShutdownEvent = NULL;
    }

    if (ContextIocp->StartedEvent) {
        CloseHandle(ContextIocp->StartedEvent);
        ContextIocp->StartedEvent = NULL;
    }

    RELEASE(ContextIocp->BaseOutputDirectory);
    RELEASE(ContextIocp->Allocator);
    RELEASE(ContextIocp->Rtl);
}

PERFECT_HASH_CONTEXT_IOCP_START PerfectHashContextIocpStart;

_Use_decl_annotations_
HRESULT
PerfectHashContextIocpStart(
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp
    )
/*++

Routine Description:

    Starts the IOCP runtime for a PERFECT_HASH_CONTEXT_IOCP instance.  This
    enumerates NUMA nodes, creates one IOCP per node, and spawns worker threads
    with appropriate affinity.

Arguments:

    ContextIocp - Supplies a pointer to a PERFECT_HASH_CONTEXT_IOCP structure
        for which the runtime is to be started.

Return Value:

    S_OK on success, an appropriate error code on failure.

--*/
{
#ifndef PH_WINDOWS
    UNREFERENCED_PARAMETER(ContextIocp);
    return E_NOTIMPL;
#else
    HRESULT Result;

    if (!ARGUMENT_PRESENT(ContextIocp)) {
        return E_POINTER;
    }

    if (!TryAcquirePerfectHashContextIocpLockExclusive(ContextIocp)) {
        return PH_E_CONTEXT_LOCKED;
    }

    if (ContextIocp->State.Running) {
        ReleasePerfectHashContextIocpLockExclusive(ContextIocp);
        return S_FALSE;
    }

    Result = PerfectHashContextIocpEnumerateNumaNodes(ContextIocp);
    if (FAILED(Result)) {
        goto Error;
    }

    Result = PerfectHashContextIocpCreateIoCompletionPorts(ContextIocp);
    if (FAILED(Result)) {
        goto Error;
    }

    Result = PerfectHashContextIocpCreateWorkerThreads(ContextIocp);
    if (FAILED(Result)) {
        goto Error;
    }

    ContextIocp->State.Running = TRUE;
    if (ContextIocp->StartedEvent) {
        SetEvent(ContextIocp->StartedEvent);
    }

    Result = S_OK;
    goto End;

Error:

    PerfectHashContextIocpStop(ContextIocp);

End:

    ReleasePerfectHashContextIocpLockExclusive(ContextIocp);

    return Result;
#endif
}

PERFECT_HASH_CONTEXT_IOCP_STOP PerfectHashContextIocpStop;

_Use_decl_annotations_
HRESULT
PerfectHashContextIocpStop(
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp
    )
/*++

Routine Description:

    Stops the IOCP runtime for a PERFECT_HASH_CONTEXT_IOCP instance by posting
    shutdown completions to each worker thread and releasing resources.

Arguments:

    ContextIocp - Supplies a pointer to a PERFECT_HASH_CONTEXT_IOCP structure
        for which the runtime is to be stopped.

Return Value:

    S_OK on success, an appropriate error code on failure.

--*/
{
#ifndef PH_WINDOWS
    UNREFERENCED_PARAMETER(ContextIocp);
    return E_NOTIMPL;
#else
    ULONG NodeIndex;
    ULONG ThreadIndex;
    PRTL Rtl;
    PALLOCATOR Allocator;

    if (!ARGUMENT_PRESENT(ContextIocp)) {
        return E_POINTER;
    }

    if (!ContextIocp->Nodes) {
        return S_FALSE;
    }

    Rtl = ContextIocp->Rtl;
    Allocator = ContextIocp->Allocator;

    ContextIocp->State.Stopping = TRUE;
    if (ContextIocp->ShutdownEvent) {
        SetEvent(ContextIocp->ShutdownEvent);
    }

    for (NodeIndex = 0; NodeIndex < ContextIocp->NodeCount; NodeIndex++) {
        PPERFECT_HASH_IOCP_NODE Node = &ContextIocp->Nodes[NodeIndex];

        for (ThreadIndex = 0;
             ThreadIndex < Node->WorkerThreadCount;
             ThreadIndex++) {
            PostQueuedCompletionStatus(Node->IoCompletionPort,
                                       0,
                                       PERFECT_HASH_IOCP_SHUTDOWN_KEY,
                                       NULL);
        }
    }

    for (NodeIndex = 0; NodeIndex < ContextIocp->NodeCount; NodeIndex++) {
        PPERFECT_HASH_IOCP_NODE Node = &ContextIocp->Nodes[NodeIndex];

        for (ThreadIndex = 0;
             ThreadIndex < Node->WorkerThreadCount;
             ThreadIndex++) {
            if (Node->WorkerThreads[ThreadIndex]) {
                WaitForSingleObject(Node->WorkerThreads[ThreadIndex],
                                    INFINITE);
            }
        }
    }

    for (NodeIndex = 0; NodeIndex < ContextIocp->NodeCount; NodeIndex++) {
        PPERFECT_HASH_IOCP_NODE Node = &ContextIocp->Nodes[NodeIndex];

        if (Node->FileWorkBufferList && Rtl) {
            PerfectHashIocpRundownBufferList(Rtl,
                                             Node->FileWorkBufferList);
        }

        if (Allocator) {
            if (Node->FileWorkBufferPools) {
                Allocator->Vtbl->FreePointer(
                    Allocator,
                    (PVOID *)&Node->FileWorkBufferPools
                );
            }

            PerfectHashIocpRundownOversizePools(
                Allocator,
                &Node->FileWorkOversizePools,
                &Node->FileWorkOversizePoolCount
            );
        }

        Node->FileWorkBufferPoolCount = 0;

        RELEASE(Node->FileWorkBufferList);

        if (Node->WorkerThreads) {
            for (ThreadIndex = 0;
                 ThreadIndex < Node->WorkerThreadCount;
                 ThreadIndex++) {
                if (Node->WorkerThreads[ThreadIndex]) {
                    CloseHandle(Node->WorkerThreads[ThreadIndex]);
                }
            }
            if (ContextIocp->Allocator) {
                ContextIocp->Allocator->Vtbl->FreePointer(
                    ContextIocp->Allocator,
                    (PVOID *)&Node->WorkerThreads
                );
            }
        }

        if (Node->IoCompletionPort) {
            CloseHandle(Node->IoCompletionPort);
            Node->IoCompletionPort = NULL;
        }
    }

    if (ContextIocp->Allocator) {
        ContextIocp->Allocator->Vtbl->FreePointer(
            ContextIocp->Allocator,
            (PVOID *)&ContextIocp->Nodes
        );
    }

    ContextIocp->Nodes = NULL;
    ContextIocp->NodeCount = 0;
    ContextIocp->IoCompletionPortCount = 0;
    ContextIocp->TotalWorkerThreadCount = 0;

    ContextIocp->State.Stopped = TRUE;
    ContextIocp->State.Running = FALSE;

    return S_OK;
#endif
}

static
VOID
PerfectHashIocpApplyNodeAffinity(
    PPERFECT_HASH_IOCP_NODE Node
    )
{
#ifdef PH_WINDOWS
    GROUP_AFFINITY Previous;

    if (!SetThreadGroupAffinity(GetCurrentThread(),
                                &Node->GroupAffinity,
                                &Previous)) {
        if (Node->GroupAffinity.Group == 0) {
            SetThreadAffinityMask(GetCurrentThread(),
                                  Node->GroupAffinity.Mask);
        }
    }
#else
    UNREFERENCED_PARAMETER(Node);
#endif
}

#ifdef PH_WINDOWS
static
DWORD
WINAPI
PerfectHashIocpWorkerThreadProc(
    PVOID Parameter
    )
{
    BOOL Success;
    ULONG_PTR Key;
    DWORD NumberOfBytes;
    LPOVERLAPPED Overlapped;
    PPERFECT_HASH_IOCP_NODE Node = (PPERFECT_HASH_IOCP_NODE)Parameter;

    PerfectHashIocpApplyNodeAffinity(Node);

    while (TRUE) {
        Success = GetQueuedCompletionStatus(Node->IoCompletionPort,
                                            &NumberOfBytes,
                                            &Key,
                                            &Overlapped,
                                            INFINITE);

        if (Key == PERFECT_HASH_IOCP_SHUTDOWN_KEY) {
            break;
        }

        if (Overlapped) {
            PPERFECT_HASH_IOCP_WORK WorkItem =
                (PPERFECT_HASH_IOCP_WORK)Overlapped;

            if (WorkItem->Signature == PH_IOCP_WORK_SIGNATURE &&
                WorkItem->CompletionCallback) {
                WorkItem->CompletionCallback(Node->ContextIocp,
                                             Key,
                                             Overlapped,
                                             NumberOfBytes,
                                             Success);
                continue;
            }
        }

        if (Node->ContextIocp->CompletionCallback && Overlapped) {
            Node->ContextIocp->CompletionCallback(Node->ContextIocp,
                                                  Key,
                                                  Overlapped,
                                                  NumberOfBytes,
                                                  Success);
        }
    }

    return 0;
}
#else
static
VOID
PerfectHashIocpWorkerThreadProc(
    PVOID Parameter
    )
{
    UNREFERENCED_PARAMETER(Parameter);
}
#endif

static
HRESULT
PerfectHashContextIocpEnumerateNumaNodes(
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp
    )
{
#ifdef PH_WINDOWS
    BOOL Success;
    HRESULT Result;
    USHORT NodeId;
    ULONG HighestNode;
    USHORT HighestNodeShort;
    ULONG SelectedCount;
    ULONG TotalProcessors;
    ULONGLONG SelectedMask;
    PRTL Rtl;
    PALLOCATOR Allocator;
    PPERFECT_HASH_IOCP_NODE Nodes;

    if (ContextIocp->Nodes) {
        return S_OK;
    }

    Rtl = ContextIocp->Rtl;
    Allocator = ContextIocp->Allocator;
    if (!Rtl || !Allocator) {
        return E_UNEXPECTED;
    }

    if (!GetNumaHighestNodeNumber(&HighestNode)) {
        SYS_ERROR(GetNumaHighestNodeNumber);
        return PH_E_SYSTEM_CALL_FAILED;
    }

    if (HighestNode >= 64) {
        return E_NOTIMPL;
    }

    HighestNodeShort = (USHORT)HighestNode;

    SelectedMask = ContextIocp->Flags.UseNumaNodeMask ?
                   ContextIocp->NumaNodeMask :
                   PERFECT_HASH_NUMA_NODE_MASK_ALL;

    SelectedCount = 0;
    TotalProcessors = 0;

    for (NodeId = 0; NodeId <= HighestNodeShort; NodeId++) {
        GROUP_AFFINITY Affinity = { 0 };
        ULONG ProcessorCount;

        Success = GetNumaNodeProcessorMaskEx(NodeId, &Affinity);
        if (!Success) {
            continue;
        }

        if (SelectedMask != PERFECT_HASH_NUMA_NODE_MASK_ALL &&
            !(SelectedMask & (1ULL << NodeId))) {
            continue;
        }

        ProcessorCount = (ULONG)(
            Rtl->PopulationCountPointer((ULONG_PTR)Affinity.Mask)
        );

        if (ProcessorCount == 0) {
            continue;
        }

        SelectedCount++;
        TotalProcessors += ProcessorCount;
    }

    if (SelectedCount == 0 || TotalProcessors == 0) {
        return E_INVALIDARG;
    }

    Nodes = (PPERFECT_HASH_IOCP_NODE)(
        Allocator->Vtbl->Calloc(Allocator, SelectedCount, sizeof(*Nodes))
    );
    if (!Nodes) {
        return E_OUTOFMEMORY;
    }

    ContextIocp->Nodes = Nodes;
    ContextIocp->NodeCount = SelectedCount;
    ContextIocp->IoCompletionPortCount = SelectedCount;

    SelectedCount = 0;

    for (NodeId = 0; NodeId <= HighestNodeShort; NodeId++) {
        GROUP_AFFINITY Affinity = { 0 };
        ULONG ProcessorCount;
        PPERFECT_HASH_IOCP_NODE Node;

        Success = GetNumaNodeProcessorMaskEx(NodeId, &Affinity);
        if (!Success) {
            continue;
        }

        if (SelectedMask != PERFECT_HASH_NUMA_NODE_MASK_ALL &&
            !(SelectedMask & (1ULL << NodeId))) {
            continue;
        }

        ProcessorCount = (ULONG)(
            Rtl->PopulationCountPointer((ULONG_PTR)Affinity.Mask)
        );

        if (ProcessorCount == 0) {
            continue;
        }

        Node = &Nodes[SelectedCount++];
        Node->ContextIocp = ContextIocp;
        Node->NodeId = NodeId;
        Node->ProcessorCount = ProcessorCount;
        Node->GroupAffinity = Affinity;
        InitializeSRWLock(&Node->FileWorkBufferPoolLock);
        InitializeListHead(&Node->FileWorkOversizePools);
        Node->FileWorkBufferPoolCount = 0;
        Node->FileWorkOversizePoolCount = 0;

        Result = ContextIocp->Vtbl->CreateInstance(
            ContextIocp,
            NULL,
            &IID_PERFECT_HASH_GUARDED_LIST,
            &Node->FileWorkBufferList
        );

        if (FAILED(Result)) {
            return Result;
        }
    }

    //
    // Configure per-node IOCP concurrency and worker thread counts.
    //

    {
        ULONG Index;
        ULONG TotalWorkerThreads = 0;
        ULONG DefaultIocpConcurrency = ContextIocp->IocpConcurrency;
        ULONG DefaultMaxThreads = ContextIocp->MaxWorkerThreads;

        for (Index = 0; Index < ContextIocp->NodeCount; Index++) {
            PPERFECT_HASH_IOCP_NODE Node = &Nodes[Index];
            ULONG IocpConcurrency;
            ULONG MaxThreads;

            IocpConcurrency = DefaultIocpConcurrency;
            if (IocpConcurrency == 0 ||
                IocpConcurrency > Node->ProcessorCount) {
                IocpConcurrency = Node->ProcessorCount;
            }

            if (IocpConcurrency == 0) {
                IocpConcurrency = 1;
            }

            MaxThreads = DefaultMaxThreads;
            if (MaxThreads == 0) {
                if (DefaultIocpConcurrency != 0) {
                    MaxThreads = IocpConcurrency * 2;
                } else {
                    MaxThreads = Node->ProcessorCount;
                }
            }

            if (MaxThreads == 0) {
                MaxThreads = 1;
            }

            Node->IocpConcurrency = IocpConcurrency;
            Node->WorkerThreadCount = MaxThreads;
            TotalWorkerThreads += MaxThreads;
        }

        ContextIocp->TotalWorkerThreadCount = TotalWorkerThreads;
    }

    return S_OK;
#else
    UNREFERENCED_PARAMETER(ContextIocp);
    return E_NOTIMPL;
#endif
}

static
HRESULT
PerfectHashContextIocpCreateIoCompletionPorts(
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp
    )
{
#ifdef PH_WINDOWS
    ULONG Index;

    for (Index = 0; Index < ContextIocp->NodeCount; Index++) {
        PPERFECT_HASH_IOCP_NODE Node = &ContextIocp->Nodes[Index];

        if (Node->IoCompletionPort) {
            continue;
        }

        Node->IoCompletionPort = CreateIoCompletionPort(
            INVALID_HANDLE_VALUE,
            NULL,
            0,
            Node->IocpConcurrency
        );

        if (!Node->IoCompletionPort) {
            SYS_ERROR(CreateIoCompletionPort);
            return PH_E_SYSTEM_CALL_FAILED;
        }
    }

    return S_OK;
#else
    UNREFERENCED_PARAMETER(ContextIocp);
    return E_NOTIMPL;
#endif
}

static
HRESULT
PerfectHashContextIocpCreateWorkerThreads(
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp
    )
{
#ifdef PH_WINDOWS
    ULONG Index;
    PALLOCATOR Allocator;

    Allocator = ContextIocp->Allocator;

    for (Index = 0; Index < ContextIocp->NodeCount; Index++) {
        PPERFECT_HASH_IOCP_NODE Node = &ContextIocp->Nodes[Index];
        ULONG ThreadIndex;

        if (Node->WorkerThreadCount == 0) {
            continue;
        }

        if (Node->WorkerThreads) {
            continue;
        }

        Node->WorkerThreads = (PHANDLE)(
            Allocator->Vtbl->Calloc(Allocator,
                                    Node->WorkerThreadCount,
                                    sizeof(HANDLE))
        );
        if (!Node->WorkerThreads) {
            return E_OUTOFMEMORY;
        }

        for (ThreadIndex = 0;
             ThreadIndex < Node->WorkerThreadCount;
             ThreadIndex++) {
            HANDLE ThreadHandle;

            ThreadHandle = CreateThread(NULL,
                                        0,
                                        PerfectHashIocpWorkerThreadProc,
                                        Node,
                                        0,
                                        NULL);
            if (!ThreadHandle) {
                SYS_ERROR(CreateThread);
                return PH_E_SYSTEM_CALL_FAILED;
            }

            Node->WorkerThreads[ThreadIndex] = ThreadHandle;
        }
    }

    return S_OK;
#else
    UNREFERENCED_PARAMETER(ContextIocp);
    return E_NOTIMPL;
#endif
}

PERFECT_HASH_CONTEXT_IOCP_SET_MAXIMUM_CONCURRENCY
PerfectHashContextIocpSetMaximumConcurrency;

_Use_decl_annotations_
HRESULT
PerfectHashContextIocpSetMaximumConcurrency(
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp,
    ULONG MaximumConcurrency
    )
{
    if (!ARGUMENT_PRESENT(ContextIocp)) {
        return E_POINTER;
    }

    if (MaximumConcurrency == 0) {
        return E_INVALIDARG;
    }

    if (!TryAcquirePerfectHashContextIocpLockExclusive(ContextIocp)) {
        return PH_E_CONTEXT_LOCKED;
    }

    ContextIocp->IocpConcurrency = MaximumConcurrency;

    ReleasePerfectHashContextIocpLockExclusive(ContextIocp);

    return S_OK;
}

PERFECT_HASH_CONTEXT_IOCP_GET_MAXIMUM_CONCURRENCY
PerfectHashContextIocpGetMaximumConcurrency;

_Use_decl_annotations_
HRESULT
PerfectHashContextIocpGetMaximumConcurrency(
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp,
    PULONG MaximumConcurrency
    )
{
    if (!ARGUMENT_PRESENT(ContextIocp)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(MaximumConcurrency)) {
        return E_POINTER;
    }

    *MaximumConcurrency = ContextIocp->IocpConcurrency;
    return S_OK;
}

PERFECT_HASH_CONTEXT_IOCP_SET_MAXIMUM_THREADS
    PerfectHashContextIocpSetMaximumThreads;

_Use_decl_annotations_
HRESULT
PerfectHashContextIocpSetMaximumThreads(
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp,
    ULONG MaximumThreads
    )
{
    if (!ARGUMENT_PRESENT(ContextIocp)) {
        return E_POINTER;
    }

    if (MaximumThreads == 0) {
        return E_INVALIDARG;
    }

    if (!TryAcquirePerfectHashContextIocpLockExclusive(ContextIocp)) {
        return PH_E_CONTEXT_LOCKED;
    }

    ContextIocp->MaxWorkerThreads = MaximumThreads;

    ReleasePerfectHashContextIocpLockExclusive(ContextIocp);

    return S_OK;
}

PERFECT_HASH_CONTEXT_IOCP_GET_MAXIMUM_THREADS
    PerfectHashContextIocpGetMaximumThreads;

_Use_decl_annotations_
HRESULT
PerfectHashContextIocpGetMaximumThreads(
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp,
    PULONG MaximumThreads
    )
{
    if (!ARGUMENT_PRESENT(ContextIocp)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(MaximumThreads)) {
        return E_POINTER;
    }

    *MaximumThreads = ContextIocp->MaxWorkerThreads;
    return S_OK;
}

PERFECT_HASH_CONTEXT_IOCP_SET_NUMA_NODE_MASK
    PerfectHashContextIocpSetNumaNodeMask;

_Use_decl_annotations_
HRESULT
PerfectHashContextIocpSetNumaNodeMask(
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp,
    PERFECT_HASH_NUMA_NODE_MASK NumaNodeMask
    )
{
    if (!ARGUMENT_PRESENT(ContextIocp)) {
        return E_POINTER;
    }

    if (!TryAcquirePerfectHashContextIocpLockExclusive(ContextIocp)) {
        return PH_E_CONTEXT_LOCKED;
    }

    ContextIocp->NumaNodeMask = NumaNodeMask;
    ContextIocp->Flags.UseNumaNodeMask = TRUE;

    ReleasePerfectHashContextIocpLockExclusive(ContextIocp);

    return S_OK;
}

PERFECT_HASH_CONTEXT_IOCP_GET_NUMA_NODE_MASK
    PerfectHashContextIocpGetNumaNodeMask;

_Use_decl_annotations_
HRESULT
PerfectHashContextIocpGetNumaNodeMask(
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp,
    PPERFECT_HASH_NUMA_NODE_MASK NumaNodeMask
    )
{
    if (!ARGUMENT_PRESENT(ContextIocp)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(NumaNodeMask)) {
        return E_POINTER;
    }

    *NumaNodeMask = ContextIocp->NumaNodeMask;
    return S_OK;
}

PERFECT_HASH_CONTEXT_IOCP_SET_BASE_OUTPUT_DIRECTORY
    PerfectHashContextIocpSetBaseOutputDirectory;

_Use_decl_annotations_
HRESULT
PerfectHashContextIocpSetBaseOutputDirectory(
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp,
    PCUNICODE_STRING BaseOutputDirectory
    )
{
    HRESULT Result = S_OK;
    PPERFECT_HASH_PATH Path = NULL;
    PPERFECT_HASH_PATH_PARTS Parts = NULL;
    PPERFECT_HASH_DIRECTORY Directory;
    PERFECT_HASH_DIRECTORY_CREATE_FLAGS DirectoryCreateFlags = { 0 };

    if (!ARGUMENT_PRESENT(ContextIocp)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(BaseOutputDirectory)) {
        return E_POINTER;
    }

    if (!IsValidMinimumDirectoryNullTerminatedUnicodeString(
        BaseOutputDirectory)) {
        return E_INVALIDARG;
    }

    if (!TryAcquirePerfectHashContextIocpLockExclusive(ContextIocp)) {
        return PH_E_CONTEXT_LOCKED;
    }

    if (ContextIocp->BaseOutputDirectory) {
        ReleasePerfectHashContextIocpLockExclusive(ContextIocp);
        return PH_E_CONTEXT_BASE_OUTPUT_DIRECTORY_ALREADY_SET;
    }

    Result = ContextIocp->Vtbl->CreateInstance(ContextIocp,
                                               NULL,
                                               &IID_PERFECT_HASH_PATH,
                                               &Path);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashPathCreateInstance, Result);
        goto Error;
    }

    Result = Path->Vtbl->Copy(Path, BaseOutputDirectory, &Parts, NULL);
    if (FAILED(Result)) {
        PH_ERROR(PerfectHashPathCopy, Result);
        goto Error;
    }

    Result = ContextIocp->Vtbl->CreateInstance(ContextIocp,
                                               NULL,
                                               &IID_PERFECT_HASH_DIRECTORY,
                                               &ContextIocp->BaseOutputDirectory);

    if (FAILED(Result)) {
        PH_ERROR(CreateInstancePerfectHashDirectory, Result);
        goto Error;
    }

    Directory = ContextIocp->BaseOutputDirectory;

    Result = Directory->Vtbl->Create(Directory,
                                     Path,
                                     &DirectoryCreateFlags);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashDirectoryCreate, Result);
        goto Error;
    }

    goto End;

Error:

    if (Result == S_OK) {
        Result = PH_E_CONTEXT_SET_BASE_OUTPUT_DIRECTORY_FAILED;
    }

End:

    RELEASE(Path);
    ReleasePerfectHashContextIocpLockExclusive(ContextIocp);
    return Result;
}

PERFECT_HASH_CONTEXT_IOCP_GET_BASE_OUTPUT_DIRECTORY
    PerfectHashContextIocpGetBaseOutputDirectory;

_Use_decl_annotations_
HRESULT
PerfectHashContextIocpGetBaseOutputDirectory(
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp,
    PPERFECT_HASH_DIRECTORY *BaseOutputDirectoryPointer
    )
{
    if (!ARGUMENT_PRESENT(ContextIocp)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(BaseOutputDirectoryPointer)) {
        return E_POINTER;
    }

    *BaseOutputDirectoryPointer = ContextIocp->BaseOutputDirectory;

    return S_OK;
}

PERFECT_HASH_CONTEXT_IOCP_CREATE_TABLE_CONTEXT
    PerfectHashContextIocpCreateTableContext;

_Use_decl_annotations_
HRESULT
PerfectHashContextIocpCreateTableContext(
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp,
    PPERFECT_HASH_CONTEXT *ContextPointer
    )
/*++

Routine Description:

    Creates a PERFECT_HASH_CONTEXT instance suitable for IOCP-native
    workflows.  The TLS flag CreateContextWithoutThreadpool is set for the
    duration of the call to suppress threadpool initialization.

Arguments:

    ContextIocp - Supplies a pointer to the IOCP context.

    ContextPointer - Receives the newly created PERFECT_HASH_CONTEXT instance.

Return Value:

    S_OK on success, otherwise an error code.

--*/
{
    HRESULT Result;
    BOOLEAN LocalTlsActive = FALSE;
    PPERFECT_HASH_TLS_CONTEXT ActiveTls;
    PERFECT_HASH_TLS_CONTEXT LocalTlsContext = { 0 };
    PERFECT_HASH_TLS_CONTEXT_FLAGS SavedFlags = { 0 };

    if (!ARGUMENT_PRESENT(ContextIocp)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(ContextPointer)) {
        return E_POINTER;
    }

    *ContextPointer = NULL;

    ActiveTls = PerfectHashTlsGetContext();
    if (ActiveTls) {
        SavedFlags = ActiveTls->Flags;
        ActiveTls->Flags.CreateContextWithoutThreadpool = TRUE;
    } else {
        ActiveTls = PerfectHashTlsGetOrSetContext(&LocalTlsContext);
        ActiveTls->Flags.CreateContextWithoutThreadpool = TRUE;
        LocalTlsActive = TRUE;
    }

    Result = ContextIocp->Vtbl->CreateInstance(ContextIocp,
                                               NULL,
                                               &IID_PERFECT_HASH_CONTEXT,
                                               ContextPointer);

    if (SUCCEEDED(Result) && *ContextPointer) {
        SetContextUseOverlappedIo(*ContextPointer);
    }

    if (LocalTlsActive) {
        PerfectHashTlsClearContextIfActive(&LocalTlsContext);
    } else if (ActiveTls) {
        ActiveTls->Flags = SavedFlags;
    }

    return Result;
}

PERFECT_HASH_CONTEXT_IOCP_BULK_CREATE PerfectHashContextIocpBulkCreate;
PERFECT_HASH_CONTEXT_IOCP_BULK_CREATE_ARGVW PerfectHashContextIocpBulkCreateArgvW;
PERFECT_HASH_CONTEXT_IOCP_EXTRACT_BULK_CREATE_ARGS_FROM_ARGVW
    PerfectHashContextIocpExtractBulkCreateArgsFromArgvW;
PERFECT_HASH_CONTEXT_IOCP_TABLE_CREATE PerfectHashContextIocpTableCreate;
PERFECT_HASH_CONTEXT_IOCP_TABLE_CREATE_ARGVW
    PerfectHashContextIocpTableCreateArgvW;
PERFECT_HASH_CONTEXT_IOCP_EXTRACT_TABLE_CREATE_ARGS_FROM_ARGVW
    PerfectHashContextIocpExtractTableCreateArgsFromArgvW;
PERFECT_HASH_CONTEXT_IOCP_TABLE_CREATE_ARGVA
    PerfectHashContextIocpTableCreateArgvA;
PERFECT_HASH_CONTEXT_IOCP_BULK_CREATE_ARGVA
    PerfectHashContextIocpBulkCreateArgvA;

static
HRESULT
PerfectHashContextIocpInvokeLegacyContextArgvW(
    _In_ PPERFECT_HASH_CONTEXT_IOCP ContextIocp,
    _In_ ULONG NumberOfArguments,
    _In_ LPWSTR *ArgvW,
    _In_ LPWSTR CommandLineW,
    _In_ BOOLEAN IsBulkCreate
    )
{
    HRESULT Result;
    PPERFECT_HASH_CONTEXT Context;

    if (!ARGUMENT_PRESENT(ContextIocp)) {
        return E_POINTER;
    }

    Result = ContextIocp->Vtbl->CreateInstance(ContextIocp,
                                               NULL,
                                               &IID_PERFECT_HASH_CONTEXT,
                                               &Context);
    if (FAILED(Result)) {
        return Result;
    }

    {
        ULONG Concurrency;

        Concurrency = ContextIocp->MaxWorkerThreads;
        if (Concurrency == 0) {
            Concurrency = ContextIocp->IocpConcurrency;
        }

        if (Concurrency > 0) {
            Result = Context->Vtbl->SetMaximumConcurrency(
                Context,
                Concurrency
            );
            if (FAILED(Result)) {
                RELEASE(Context);
                return Result;
            }
        }
    }

    if (IsBulkCreate) {
        Result = Context->Vtbl->BulkCreateArgvW(Context,
                                                NumberOfArguments,
                                                ArgvW,
                                                CommandLineW);
    } else {
        Result = Context->Vtbl->TableCreateArgvW(Context,
                                                 NumberOfArguments,
                                                 ArgvW,
                                                 CommandLineW);
    }

    RELEASE(Context);
    return Result;
}

_Use_decl_annotations_
HRESULT
PerfectHashContextIocpBulkCreate(
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp,
    PCUNICODE_STRING KeysDirectory,
    PCUNICODE_STRING BaseOutputDirectory,
    PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    PPERFECT_HASH_CONTEXT_BULK_CREATE_FLAGS ContextBulkCreateFlags,
    PPERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags,
    PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags,
    PPERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags,
    PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    )
{
    UNREFERENCED_PARAMETER(ContextIocp);
    UNREFERENCED_PARAMETER(KeysDirectory);
    UNREFERENCED_PARAMETER(BaseOutputDirectory);
    UNREFERENCED_PARAMETER(AlgorithmId);
    UNREFERENCED_PARAMETER(HashFunctionId);
    UNREFERENCED_PARAMETER(MaskFunctionId);
    UNREFERENCED_PARAMETER(ContextBulkCreateFlags);
    UNREFERENCED_PARAMETER(KeysLoadFlags);
    UNREFERENCED_PARAMETER(TableCreateFlags);
    UNREFERENCED_PARAMETER(TableCompileFlags);
    UNREFERENCED_PARAMETER(TableCreateParameters);

    return E_NOTIMPL;
}

_Use_decl_annotations_
HRESULT
PerfectHashContextIocpBulkCreateArgvW(
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp,
    ULONG NumberOfArguments,
    LPWSTR *ArgvW,
    LPWSTR CommandLineW
    )
{
    return PerfectHashContextIocpInvokeLegacyContextArgvW(
        ContextIocp,
        NumberOfArguments,
        ArgvW,
        CommandLineW,
        TRUE
    );
}


_Use_decl_annotations_
HRESULT
PerfectHashContextIocpTableCreate(
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp,
    PCUNICODE_STRING KeysPath,
    PCUNICODE_STRING BaseOutputDirectory,
    PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    PPERFECT_HASH_CONTEXT_TABLE_CREATE_FLAGS ContextTableCreateFlags,
    PPERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags,
    PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags,
    PPERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags,
    PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    )
{
    UNREFERENCED_PARAMETER(ContextIocp);
    UNREFERENCED_PARAMETER(KeysPath);
    UNREFERENCED_PARAMETER(BaseOutputDirectory);
    UNREFERENCED_PARAMETER(AlgorithmId);
    UNREFERENCED_PARAMETER(HashFunctionId);
    UNREFERENCED_PARAMETER(MaskFunctionId);
    UNREFERENCED_PARAMETER(ContextTableCreateFlags);
    UNREFERENCED_PARAMETER(KeysLoadFlags);
    UNREFERENCED_PARAMETER(TableCreateFlags);
    UNREFERENCED_PARAMETER(TableCompileFlags);
    UNREFERENCED_PARAMETER(TableCreateParameters);

    return E_NOTIMPL;
}

_Use_decl_annotations_
HRESULT
PerfectHashContextIocpTableCreateArgvW(
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp,
    ULONG NumberOfArguments,
    LPWSTR *ArgvW,
    LPWSTR CommandLineW
    )
{
    return PerfectHashContextIocpInvokeLegacyContextArgvW(
        ContextIocp,
        NumberOfArguments,
        ArgvW,
        CommandLineW,
        FALSE
    );
}


_Use_decl_annotations_
HRESULT
PerfectHashContextIocpTableCreateArgvA(
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp,
    ULONG NumberOfArguments,
    LPSTR *ArgvA
    )
{
    UNREFERENCED_PARAMETER(ContextIocp);
    UNREFERENCED_PARAMETER(NumberOfArguments);
    UNREFERENCED_PARAMETER(ArgvA);

    return E_NOTIMPL;
}

_Use_decl_annotations_
HRESULT
PerfectHashContextIocpBulkCreateArgvA(
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp,
    ULONG NumberOfArguments,
    LPSTR *ArgvA
    )
{
    UNREFERENCED_PARAMETER(ContextIocp);
    UNREFERENCED_PARAMETER(NumberOfArguments);
    UNREFERENCED_PARAMETER(ArgvA);

    return E_NOTIMPL;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
