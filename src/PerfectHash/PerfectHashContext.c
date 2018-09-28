/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashContext.c

Abstract:

    This module implements the runtime context creation routine for the
    PerfectHash component.  The context encapsulates threadpool resources
    in order to support finding perfect hash table solutions in parallel.

    Routines are provided for context initialization and rundown, setting and
    getting the maximum concurrency associated with a context, setting and
    getting the output directory, and callback routines for the various
    threadpool functions.

--*/

#include "stdafx.h"

//
// Forward definition of various callbacks we implement in this module.
// These are needed up-front in order to create and register the various
// work and cleanup callbacks as part of context creation.
//
// (Annoyingly, winnt.h only defines PTP_WORK_CALLBACK, not the underlying
//  raw function type TP_WORK_CALLBACK, so, do that now.  Ditto for the
//  PTP_CLEANUP_GROUP_CANCEL_CALLBACK signature.)
//

typedef
VOID
(NTAPI TP_WORK_CALLBACK)(
    _Inout_     PTP_CALLBACK_INSTANCE Instance,
    _Inout_opt_ PVOID                 Context,
    _Inout_     PTP_WORK              Work
    );


typedef
VOID
(NTAPI TP_CLEANUP_GROUP_CANCEL_CALLBACK)(
    _Inout_opt_ PVOID ObjectContext,
    _Inout_opt_ PVOID CleanupContext
    );

//
// Forward definitions.
//

TP_WORK_CALLBACK MainWorkCallback;
TP_WORK_CALLBACK FileWorkCallback;
TP_WORK_CALLBACK ErrorWorkCallback;
TP_WORK_CALLBACK FinishedWorkCallback;
TP_CLEANUP_GROUP_CANCEL_CALLBACK CleanupCallback;

PERFECT_HASH_CONTEXT_INITIALIZE PerfectHashContextInitialize;

//
// Main context creation routine.
//

_Use_decl_annotations_
HRESULT
PerfectHashContextInitialize(
    PPERFECT_HASH_CONTEXT Context
    )
/*++

Routine Description:

    Initializes a perfect hash table context.  This involves creating the
    threadpool and associated resources for the context.

Arguments:

    Context - Supplies a pointer to a PERFECT_HASH_CONTEXT structure
        for which initialization is to be performed.

Return Value:

    S_OK on success, an appropriate error code on failure.

--*/
{
    PRTL Rtl;
    BYTE Index;
    BYTE NumberOfEvents;
    HRESULT Result = S_OK;
    ULONG LastError;
    PHANDLE Event;
    PCHAR Buffer;
    PCHAR BaseBuffer;
    PCHAR ExpectedBuffer;
    ULONG MaximumConcurrency;
    ULONG NumberOfProcessors;
    ULONG SizeOfNamesWideBuffer = 0;
    PWSTR NamesWideBuffer;
    PTP_POOL Threadpool;
    PALLOCATOR Allocator;
    ULARGE_INTEGER AllocSize;
    ULARGE_INTEGER ObjectNameArraySize;
    ULARGE_INTEGER ObjectNamePointersArraySize;
    PUNICODE_STRING Name;
    PPUNICODE_STRING Names;
    PPUNICODE_STRING Prefixes;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Context)) {
        return E_POINTER;
    }

    //
    // Create Rtl and Allocator components.
    //

    Result = Context->Vtbl->CreateInstance(Context,
                                           NULL,
                                           &IID_PERFECT_HASH_RTL,
                                           &Context->Rtl);

    if (FAILED(Result)) {
        goto Error;
    }

    Result = Context->Vtbl->CreateInstance(Context,
                                           NULL,
                                           &IID_PERFECT_HASH_ALLOCATOR,
                                           &Context->Allocator);

    if (FAILED(Result)) {
        goto Error;
    }

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Allocator = Context->Allocator;

    //
    // Calculate the size required by the array of UNICODE_STRING structures
    // that trail the context, then the array of addresses to those structures.
    //

    ObjectNameArraySize.QuadPart = (
        (NumberOfContextObjectPrefixes * sizeof(UNICODE_STRING))
    );

    ObjectNamePointersArraySize.QuadPart = (
        (NumberOfContextObjectPrefixes * sizeof(PUNICODE_STRING))
    );

    ASSERT(!ObjectNameArraySize.HighPart);
    ASSERT(!ObjectNamePointersArraySize.HighPart);

    AllocSize.QuadPart = (
        ObjectNameArraySize.QuadPart +
        ObjectNamePointersArraySize.QuadPart
    );

    //
    // Sanity check we haven't overflowed.
    //

    ASSERT(!AllocSize.HighPart);

    //
    // Allocate space for the object name buffer.
    //

    BaseBuffer = Buffer = (PCHAR)(
        Allocator->Vtbl->Calloc(
            Allocator,
            1,
            AllocSize.LowPart
        )
    );

    if (!Buffer) {
        SYS_ERROR(HeapAlloc);
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    //
    // Allocation of buffer was successful, continue with initialization.
    //

    Context->SizeOfStruct = sizeof(*Context);
    Context->State.AsULong = 0;
    Context->Flags.AsULong = 0;

    InitializeSRWLock(&Context->Lock);

    //
    // Carve the buffer we just allocated up into an array of UNICODE_STRING
    // structures that will be filled out by RtlCreateRandomObjectNames().
    //

    Context->ObjectNames = (PUNICODE_STRING)Buffer;

    Buffer += ObjectNameArraySize.LowPart;
    Context->ObjectNamesPointerArray = (PPUNICODE_STRING)Buffer;

    Buffer += ObjectNamePointersArraySize.LowPart;

    //
    // If our pointer arithmetic was correct, Buffer should match the base
    // address of the context plus the total allocation size at this point.
    // Assert this invariant now.
    //

    ExpectedBuffer = RtlOffsetToPointer(BaseBuffer, AllocSize.LowPart);
    ASSERT(Buffer == ExpectedBuffer);

    //
    // Wire up the pointer array to the object names.
    //

    Names = Context->ObjectNamesPointerArray;

    for (Index = 0; Index < NumberOfContextObjectPrefixes; Index++) {
        Names[Index] = Context->ObjectNames + Index;
    }

    //
    // Create the random object names for our underlying events.
    //

    Prefixes = (PPUNICODE_STRING)&ContextObjectPrefixes;

    Result = Rtl->Vtbl->CreateRandomObjectNames(
        Rtl,
        Allocator,
        Allocator,
        NumberOfContextObjectPrefixes,
        64,
        NULL,
        Context->ObjectNamesPointerArray,
        Prefixes,
        &SizeOfNamesWideBuffer,
        &NamesWideBuffer
    );

    if (FAILED(Result)) {
        PH_ERROR(RtlCreateRandomObjectNames, Result);
        goto Error;
    }

    //
    // Wire up the wide buffer pointer and containing size.
    //

    Context->ObjectNamesWideBuffer = NamesWideBuffer;
    Context->SizeOfObjectNamesWideBuffer = SizeOfNamesWideBuffer;
    Context->NumberOfObjects = NumberOfContextObjectPrefixes;

    //
    // Initialize the event pointer to the first handle, and the name pointer
    // to the first UNICODE_STRING pointer.  Obtain the number of events.
    //

    Event = (PHANDLE)&Context->FirstEvent;
    Name = &Context->ObjectNames[0];
    NumberOfEvents = GetNumberOfContextEvents(Context);

    for (Index = 0; Index < NumberOfEvents; Index++, Event++, Name++) {

        //
        // We want all of our events to be manual reset, such that they stay
        // signaled even after they've satisfied a wait.
        //

        BOOLEAN ManualReset = TRUE;

        *Event = CreateEventW(NULL,
                              ManualReset,
                              FALSE,
                              Name->Buffer);

        LastError = GetLastError();

        if (!*Event || LastError == ERROR_ALREADY_EXISTS) {

            //
            // As the event names are random, a last error that indicates the
            // name already exists is evident of a pretty serious problem.
            // Treat this as a fatal.
            //

            SYS_ERROR(CreateEventW);
            goto Error;
        }
    }

    //
    // Default the maximum concurrency to the number of processors.
    //

    NumberOfProcessors = GetMaximumProcessorCount(ALL_PROCESSOR_GROUPS);
    MaximumConcurrency = NumberOfProcessors;

    Context->MinimumConcurrency = MaximumConcurrency;
    Context->MaximumConcurrency = MaximumConcurrency;

    //
    // Create the Main threadpool structures.  This threadpool creates a fixed
    // number of threads equal to the maximum concurrency specified by the user
    // (e.g. min threads is set to the same value as max threads).
    //

    Threadpool = Context->MainThreadpool = CreateThreadpool(NULL);
    if (!Threadpool) {
        SYS_ERROR(CreateThreadpool);
        goto Error;
    }

    if (!SetThreadpoolThreadMinimum(Threadpool, MaximumConcurrency)) {
        SYS_ERROR(SetThreadpoolThreadMinimum);
        goto Error;
    }

    SetThreadpoolThreadMaximum(Threadpool, MaximumConcurrency);

    //
    // Initialize the Main threadpool environment, set its priority to
    // low, then associate it with the Main threadpool.
    //

    InitializeThreadpoolEnvironment(&Context->MainCallbackEnv);
    SetThreadpoolCallbackPriority(&Context->MainCallbackEnv,
                                  TP_CALLBACK_PRIORITY_LOW);
    SetThreadpoolCallbackPool(&Context->MainCallbackEnv,
                              Context->MainThreadpool);

    //
    // Create a cleanup group for the Main threadpool and register it.
    //

    Context->MainCleanupGroup = CreateThreadpoolCleanupGroup();
    if (!Context->MainCleanupGroup) {
        SYS_ERROR(CreateThreadpoolCleanupGroup);
        goto Error;
    }

    SetThreadpoolCallbackCleanupGroup(&Context->MainCallbackEnv,
                                      Context->MainCleanupGroup,
                                      CleanupCallback);

    //
    // Create a work object for the Main threadpool.
    //

    Context->MainWork = CreateThreadpoolWork(MainWorkCallback,
                                             Context,
                                             &Context->MainCallbackEnv);

    if (!Context->MainWork) {
        SYS_ERROR(CreateThreadpoolWork);
        goto Error;
    }

    //
    // Initialize the main work list head.
    //

    InitializeSListHead(&Context->MainWorkListHead);

    //
    // Create the File threadpool structures.  We currently use the default
    // number of system threads for this threadpool.  We don't have to clamp
    // it at the moment as we don't bulk-enqueue work items (that can cause
    // the threadpool machinery to think more threads need to be created).
    //

    Threadpool = Context->FileThreadpool = CreateThreadpool(NULL);
    if (!Threadpool) {
        SYS_ERROR(CreateThreadpool);
        goto Error;
    }

    //
    // Initialize the File threadpool environment and associate it with the
    // File threadpool.
    //

    InitializeThreadpoolEnvironment(&Context->FileCallbackEnv);
    SetThreadpoolCallbackPool(&Context->FileCallbackEnv,
                              Context->FileThreadpool);

    //
    // Create a cleanup group for the File threadpool and register it.
    //

    Context->FileCleanupGroup = CreateThreadpoolCleanupGroup();
    if (!Context->FileCleanupGroup) {
        SYS_ERROR(CreateThreadpoolCleanupGroup);
        goto Error;
    }

    SetThreadpoolCallbackCleanupGroup(&Context->FileCallbackEnv,
                                      Context->FileCleanupGroup,
                                      CleanupCallback);

    //
    // Create a work object for the File threadpool.
    //

    Context->FileWork = CreateThreadpoolWork(FileWorkCallback,
                                             Context,
                                             &Context->FileCallbackEnv);

    if (!Context->FileWork) {
        SYS_ERROR(CreateThreadpoolWork);
        goto Error;
    }

    //
    // Initialize the file work list head.
    //

    InitializeSListHead(&Context->FileWorkListHead);

    //
    // Create the Finished and Error threadpools and associated resources.
    // These are slightly easier as we only have 1 thread maximum for each
    // pool and no cleanup group is necessary.
    //

    Context->FinishedThreadpool = CreateThreadpool(NULL);
    if (!Context->FinishedThreadpool) {
        SYS_ERROR(CreateThreadpool);
        goto Error;
    }

    if (!SetThreadpoolThreadMinimum(Context->FinishedThreadpool, 1)) {
        SYS_ERROR(SetThreadpoolThreadMinimum);
        goto Error;
    }

    SetThreadpoolThreadMaximum(Context->FinishedThreadpool, 1);

    Context->FinishedWork = CreateThreadpoolWork(FinishedWorkCallback,
                                                 Context,
                                                 &Context->FinishedCallbackEnv);
    if (!Context->FinishedWork) {
        SYS_ERROR(CreateThreadpoolWork);
        goto Error;
    }

    //
    // Initialize the finished list head.
    //

    InitializeSListHead(&Context->FinishedWorkListHead);

    //
    // Create the Error threadpool.
    //

    Context->ErrorThreadpool = CreateThreadpool(NULL);
    if (!Context->ErrorThreadpool) {
        SYS_ERROR(CreateThreadpool);
        goto Error;
    }

    if (!SetThreadpoolThreadMinimum(Context->ErrorThreadpool, 1)) {
        SYS_ERROR(SetThreadpoolThreadMinimum);
        goto Error;
    }

    SetThreadpoolThreadMaximum(Context->ErrorThreadpool, 1);

    Context->ErrorWork = CreateThreadpoolWork(ErrorWorkCallback,
                                              Context,
                                              &Context->ErrorCallbackEnv);
    if (!Context->ErrorWork) {
        SYS_ERROR(CreateThreadpoolWork);
        goto Error;
    }

    //
    // We're done!  Indicate success and finish up.
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

PERFECT_HASH_CONTEXT_RUNDOWN PerfectHashContextRundown;

_Use_decl_annotations_
VOID
PerfectHashContextRundown(
    PPERFECT_HASH_CONTEXT Context
    )
/*++

Routine Description:

    Release all resources associated with a context.

Arguments:

    Context - Supplies a pointer to the PERFECT_HASH_CONTEXT to rundown.

Return Value:

    None.

--*/
{
    PRTL Rtl;
    BYTE Index;
    BYTE NumberOfEvents;
    PALLOCATOR Allocator;
    PHANDLE Event;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Context)) {
        return;
    }

    Rtl = Context->Rtl;
    Allocator = Context->Allocator;

    if (!Rtl || !Allocator) {
        return;
    }

    //
    // Sanity check the perfect hash structure size matches what we expect.
    //

    ASSERT(Context->SizeOfStruct == sizeof(*Context));

    //
    // Loop through all the events associated with the context and check if
    // they need to be closed.  (We do this instead of explicit calls to each
    // named event (e.g. CloseHandle(Context->ShutdownEvent)) as it means we
    // don't have to add new destructor code here every time we add an event.)
    //
    //

    Event = (PHANDLE)&Context->FirstEvent;
    NumberOfEvents = GetNumberOfContextEvents(Context);

    for (Index = 0; Index < NumberOfEvents; Index++, Event++) {

        if (*Event && *Event != INVALID_HANDLE_VALUE) {
            if (!CloseHandle(*Event)) {
                SYS_ERROR(CloseHandle);
            }
            *Event = NULL;
        }
    }

    if (Context->MainCleanupGroup) {

        CloseThreadpoolCleanupGroupMembers(Context->MainCleanupGroup,
                                           TRUE,
                                           NULL);
        Context->MainWork = NULL;
        CloseThreadpoolCleanupGroup(Context->MainCleanupGroup);
        Context->MainCleanupGroup = NULL;

    } else {

        //
        // Invariant check: Context->MainWork should never be set if
        // MainCleanupGroup is not set.
        //

        ASSERT(!Context->MainWork);
    }

    if (Context->MainThreadpool) {
        CloseThreadpool(Context->MainThreadpool);
        Context->MainThreadpool = NULL;
    }

    //
    // The Finished and Error threadpools do not have a cleanup group associated
    // with them, so we can close their work items directly, if applicable.
    //

    if (Context->FinishedWork) {
        CloseThreadpoolWork(Context->FinishedWork);
        Context->FinishedWork = NULL;
    }

    if (Context->FinishedThreadpool) {
        CloseThreadpool(Context->FinishedThreadpool);
        Context->FinishedThreadpool = NULL;
    }

    if (Context->ErrorWork) {
        CloseThreadpoolWork(Context->ErrorWork);
        Context->ErrorWork = NULL;
    }

    if (Context->ErrorThreadpool) {
        CloseThreadpool(Context->ErrorThreadpool);
        Context->ErrorThreadpool = NULL;
    }

    if (Context->ObjectNames) {
        Allocator->Vtbl->FreePointer(Allocator,
                                     &Context->ObjectNames);
    }

    Allocator->Vtbl->Release(Allocator);
    Rtl->Vtbl->Release(Rtl);
}

PERFECT_HASH_CONTEXT_RESET PerfectHashContext;

_Use_decl_annotations_
HRESULT
PerfectHashContextReset(
    PPERFECT_HASH_CONTEXT Context
    )
/*++

Routine Description:

    Resets a context; required after a context has created a table before it
    can create a new table.

Arguments:

    Context - Supplies a pointer to a PERFECT_HASH_CONTEXT structure
        for which the reset is to be performed.

Return Value:

    S_OK on success.

    E_POINTER if Context is NULL.

    E_UNEXPECTED for other errors.

--*/
{
    if (!ARGUMENT_PRESENT(Context)) {
        return E_POINTER;
    }

    Context->AlgorithmId = PerfectHashNullAlgorithmId;
    Context->HashFunctionId = PerfectHashNullHashFunctionId;
    Context->MaskFunctionId = PerfectHashNullMaskFunctionId;

    Context->AlgorithmContext = NULL;
    Context->HashFunctionContext = NULL;
    Context->SolvedContext = NULL;

    Context->HighestDeletedEdgesCount = 0;
    Context->ResizeTableThreshold = 0;
    Context->ResizeLimit = 0;
    Context->Attempts = 0;

    Context->SolveStartCycles.QuadPart = 0;
    Context->SolveStartCounter.QuadPart = 0;
    Context->SolveEndCycles.QuadPart = 0;
    Context->SolveEndCounter.QuadPart = 0;
    Context->SolveElapsedCycles.QuadPart = 0;
    Context->SolveElapsedMicroseconds.QuadPart = 0;

    Context->VerifyStartCycles.QuadPart = 0;
    Context->VerifyStartCounter.QuadPart = 0;
    Context->VerifyEndCycles.QuadPart = 0;
    Context->VerifyEndCounter.QuadPart = 0;
    Context->VerifyElapsedCycles.QuadPart = 0;
    Context->VerifyElapsedMicroseconds.QuadPart = 0;

    Context->FailedAttempts = 0;
    Context->FinishedCount = 0;

    return S_OK;
}

//
// Implement callback routines.
//

_Use_decl_annotations_
VOID
MainWorkCallback(
    PTP_CALLBACK_INSTANCE Instance,
    PVOID Ctx,
    PTP_WORK Work
    )
/*++

Routine Description:

    This is the callback routine for the Main threadpool's work.  It will be
    invoked by a thread in the Main group whenever SubmitThreadpoolWork()
    is called against Context->MainWork.  The caller is responsible for pushing
    a work item to Context->MainWorkListHead prior to submission.

    This routine pops an item off Context->MainWorkListHead, then calls the
    worker routine that was registered with the context.

Arguments:

    Instance - Supplies a pointer to the callback instance responsible for this
        threadpool callback invocation.

    Ctx - Supplies a pointer to the owning PERFECT_HASH_CONTEXT.

    Work - Supplies a pointer to the TP_WORK object for this routine.

Return Value:

    None.

--*/
{
    PSLIST_ENTRY ListEntry;
    PPERFECT_HASH_CONTEXT Context;

    UNREFERENCED_PARAMETER(Work);

    if (!ARGUMENT_PRESENT(Ctx)) {
        return;
    }

    //
    // Cast the Ctx variable into a suitable type, then pop a list entry off
    // the main work list head.
    //

    Context = (PPERFECT_HASH_CONTEXT)Ctx;
    ListEntry = InterlockedPopEntrySList(&Context->MainWorkListHead);

    if (!ListEntry) {

        //
        // A spurious work item was requested but no corresponding element was
        // pushed to the list.  This typically indicates API misuse.  We could
        // terminate here, however, that's pretty drastic, so let's just ignore
        // it.
        //

        return;
    }

    //
    // Dispatch the work item to the routine registered with the context.
    //

    Context->MainWorkCallback(Instance, Context, ListEntry);

    return;
}

_Use_decl_annotations_
VOID
FileWorkCallback(
    PTP_CALLBACK_INSTANCE Instance,
    PVOID Ctx,
    PTP_WORK Work
    )
/*++

Routine Description:

    This is the callback routine for the Main threadpool's file work.  It will
    be invoked by a thread in the Main group whenever SubmitThreadpoolWork()
    is called against Context->FileWork.  The caller is responsible for pushing
    a work item to Context->FileWorkListHead prior to submission.

    This routine pops an item off Context->FileWorkListHead, then calls the
    worker routine that was registered with the context.

Arguments:

    Instance - Supplies a pointer to the callback instance responsible for this
        threadpool callback invocation.

    Ctx - Supplies a pointer to the owning PERFECT_HASH_CONTEXT.

    Work - Supplies a pointer to the TP_WORK object for this routine.

Return Value:

    None.

--*/
{
    PSLIST_ENTRY ListEntry;
    PPERFECT_HASH_CONTEXT Context;

    UNREFERENCED_PARAMETER(Work);

    if (!ARGUMENT_PRESENT(Ctx)) {
        return;
    }

    //
    // Cast the Ctx variable into a suitable type, then pop a list entry off
    // the file work list head.
    //

    Context = (PPERFECT_HASH_CONTEXT)Ctx;
    ListEntry = InterlockedPopEntrySList(&Context->FileWorkListHead);

    if (!ListEntry) {

        //
        // A spurious work item was requested but no corresponding element was
        // pushed to the list.  This typically indicates API misuse.  We could
        // terminate here, however, that's pretty drastic, so let's just ignore
        // it.
        //

        return;
    }

    //
    // Dispatch the work item to the routine registered with the context.
    //

    Context->FileWorkCallback(Instance, Context, ListEntry);

    return;
}

_Use_decl_annotations_
VOID
FinishedWorkCallback(
    PTP_CALLBACK_INSTANCE Instance,
    PVOID Ctx,
    PTP_WORK Work
    )
/*++

Routine Description:

    This routine will be called when a Main thread successfully finds a perfect
    hash table solution.  It is responsible for signaling the ShutdownEvent and
    FinishedEvent.  It will also call CloseThreadpoolCleanupGroupMembers() to
    cancel pending callbacks.

Arguments:

    Instance - Supplies a pointer to the callback instance responsible for this
        threadpool callback invocation.

    Ctx - Supplies a pointer to the owning PERFECT_HASH_CONTEXT.

    Work - Supplies a pointer to the TP_WORK object for this routine.

Return Value:

    None.

--*/
{
    BOOL CancelPending = TRUE;
    PPERFECT_HASH_CONTEXT Context;

    UNREFERENCED_PARAMETER(Instance);
    UNREFERENCED_PARAMETER(Work);

    if (!ARGUMENT_PRESENT(Ctx)) {
        return;
    }

    //
    // Cast the Ctx variable into a suitable type.
    //

    Context = (PPERFECT_HASH_CONTEXT)Ctx;

    //
    // Wait for the main work group members.  This should block until all
    // the workers have returned.
    //

    WaitForThreadpoolWorkCallbacks(Context->MainWork, CancelPending);

    //
    // Signal the shutdown, succeeded and completed events.
    //

    SetEvent(Context->ShutdownEvent);
    SetEvent(Context->SucceededEvent);
    SetEvent(Context->CompletedEvent);

    return;
}

_Use_decl_annotations_
VOID
ErrorWorkCallback(
    PTP_CALLBACK_INSTANCE Instance,
    PVOID Ctx,
    PTP_WORK Work
    )
/*++

Routine Description:

    Not yet implemented.

Arguments:

    Instance - Supplies a pointer to the callback instance responsible for this
        threadpool callback invocation.

    Ctx - Supplies a pointer to the owning PERFECT_HASH_CONTEXT.

    Work - Supplies a pointer to the TP_WORK object for this routine.

Return Value:

    None.

--*/
{
    //
    // N.B. This routine has not been implemented yet.  Main algorithm worker
    //      threads cannot cause termination of the entire context.
    //

    UNREFERENCED_PARAMETER(Instance);
    UNREFERENCED_PARAMETER(Ctx);
    UNREFERENCED_PARAMETER(Work);

    return;
}

_Use_decl_annotations_
VOID
CleanupCallback(
    PVOID ObjectContext,
    PVOID CleanupContext
    )
/*++

Routine Description:

    This method is called when the Main threadpool context is closed via the
    CloseThreadpoolCleanupGroupMembers() routine, which will be issued by either
    the Finished or Error threadpool.

Arguments:

    ObjectContext - Supplies a pointer to the PERFECT_HASH_CONTEXT
        structure for which this cleanup was associated.

    CleanupContext - Optionally supplies per-cleanup context information at the
        time CloseThreadpoolCleanupGroupMembers() was called.  Not currently
        used.

Return Value:

    None.

--*/
{
    PPERFECT_HASH_CONTEXT Context;

    UNREFERENCED_PARAMETER(CleanupContext);

    Context = (PPERFECT_HASH_CONTEXT)ObjectContext;

    //
    // (This is placeholder scaffolding at the moment.  We don't do anything
    //  explicit in this callback currently.)
    //

    return;
}

PERFECT_HASH_CONTEXT_SET_MAXIMUM_CONCURRENCY
    PerfectHashContextSetMaximumConcurrency;

_Use_decl_annotations_
HRESULT
PerfectHashContextSetMaximumConcurrency(
    PPERFECT_HASH_CONTEXT Context,
    ULONG MaximumConcurrency
    )
/*++

Routine Description:

    Sets the maximum number of threads of the underlying main work threadpool
    used by the perfect hash table context.  This controls how many parallel
    worker threads will attempt to concurrently find a solution.

Arguments:

    Context - Supplies a pointer to the PERFECT_HASH_CONTEXT instance
        for which the maximum concurrency is to be set.

    MaximumConcurrency - Supplies the maximum concurrency.

Return Value:

    S_OK - Maximum concurrency was set successfully.

    E_POINTER - Context parameter is NULL.

    PH_E_CONTEXT_LOCKED - The context is locked.

    PH_E_SET_MAXIMUM_CONCURRENCY_FAILED - The call to
        SetThreadpoolThreadMinimum() failed.

    E_UNEXPECTED - Internal error.

--*/
{
    PRTL Rtl;
    HRESULT Result = S_OK;
    PTP_POOL Threadpool;

    if (!ARGUMENT_PRESENT(Context)) {
        return E_POINTER;
    }

    if (!Context->MainThreadpool) {
        return E_UNEXPECTED;
    }

    if (!TryAcquirePerfectHashContextLockExclusive(Context)) {
        return PH_E_CONTEXT_LOCKED;
    }

    Rtl = Context->Rtl;
    Threadpool = Context->MainThreadpool;

    SetThreadpoolThreadMaximum(Threadpool, MaximumConcurrency);
    Context->MaximumConcurrency = MaximumConcurrency;

    if (!SetThreadpoolThreadMinimum(Threadpool, MaximumConcurrency)) {
        SYS_ERROR(SetThreadpoolThreadMinimum);
        Result = PH_E_SET_MAXIMUM_CONCURRENCY_FAILED;
    } else {
        Context->MinimumConcurrency = MaximumConcurrency;
    }

    ReleasePerfectHashContextLockExclusive(Context);

    return Result;
}

PERFECT_HASH_CONTEXT_GET_MAXIMUM_CONCURRENCY
    PerfectHashContextGetMaximumConcurrency;

_Use_decl_annotations_
HRESULT
PerfectHashContextGetMaximumConcurrency(
    PPERFECT_HASH_CONTEXT Context,
    PULONG MaximumConcurrency
    )
/*++

Routine Description:

    Gets the maximum number of threads of the underlying main work threadpool
    used by the perfect hash table context.  This controls how many parallel
    worker threads will attempt to concurrently find a solution.

Arguments:

    Context - Supplies a pointer to the PERFECT_HASH_CONTEXT instance
        for which the maximum concurrency is to be obtained.

    MaximumConcurrency - Supplies the address of a variable that will receive
        the maximum concurrency configured for the given context.

Return Value:

    S_OK - Concurrency was successfully obtained.

    E_POINTER - Context or MaximumConcurrency parameters were NULL.

--*/
{
    if (!ARGUMENT_PRESENT(Context)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(MaximumConcurrency)) {
        return E_POINTER;
    }

    *MaximumConcurrency = Context->MaximumConcurrency;

    return S_OK;
}

PERFECT_HASH_CONTEXT_SET_OUTPUT_DIRECTORY PerfectHashContextSetOutputDirectory;

_Use_decl_annotations_
HRESULT
PerfectHashContextSetOutputDirectory(
    PPERFECT_HASH_CONTEXT Context,
    PCUNICODE_STRING OutputDirectory
    )
/*++

Routine Description:

    Sets the output directory for a given context.  All generated files will be
    saved to this directory.  This routine must be called before attempting to
    create any perfect hash tables.

Arguments:

    Context - Supplies the context for which the output directory is to be set.

    OutputDirectory - Supplies the output directory to set.

Return Value:

    S_OK - Output directory was set successfully.

    E_POINTER - Context or OutputDirectory parameters were NULL.

    E_INVALIDARG - OutputDirectory was not a valid directory string.

    E_UNEXPECTED - Internal error.

    PH_E_CONTEXT_LOCKED - The context is locked.

--*/
{
    PRTL Rtl;
    PWSTR Buffer;
    PALLOCATOR Allocator;
    HRESULT Result = S_OK;
    PUNICODE_STRING Directory;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Context)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(OutputDirectory)) {
        return E_POINTER;
    }

    if (!IsValidMinimumDirectoryNullTerminatedUnicodeString(OutputDirectory)) {
        return E_INVALIDARG;
    }

    if (!TryAcquirePerfectHashContextLockExclusive(Context)) {
        return PH_E_CONTEXT_LOCKED;
    }

    //
    // Argument validation complete.
    //

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Directory = &Context->OutputDirectory;
    Allocator = Context->Allocator;

    //
    // If an output directory is already set, free it.
    //

    if (Directory->Buffer) {
        Allocator->Vtbl->FreePointer(Allocator, &Directory->Buffer);
    }

    //
    // Allocate and copy the string.
    //

    Buffer = Allocator->Vtbl->Calloc(Allocator,
                                     1,
                                     OutputDirectory->MaximumLength);

    if (!Buffer) {
        SYS_ERROR(VirtualAlloc);
        goto Error;
    }

    CopyMemory(Buffer,
               OutputDirectory->Buffer,
               OutputDirectory->MaximumLength);

    Directory->Length = OutputDirectory->Length;
    Directory->MaximumLength = OutputDirectory->MaximumLength;
    Directory->Buffer = Buffer;

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = PH_E_CONTEXT_SET_OUTPUT_DIRECTORY_FAILED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    ReleasePerfectHashContextLockExclusive(Context);

    return Result;
}

PERFECT_HASH_CONTEXT_GET_OUTPUT_DIRECTORY PerfectHashContextGetOutputDirectory;

_Use_decl_annotations_
HRESULT
PerfectHashContextGetOutputDirectory(
    PPERFECT_HASH_CONTEXT Context,
    PCUNICODE_STRING *OutputDirectoryPointer
    )
/*++

Routine Description:

    Obtains a previously set output directory for a given context.

Arguments:

    Context - Supplies the context for which the output directory is to be
        obtained.

    OuputDirectory - Supplies the address of a variable that receives a pointer
        to the UNICODE_STRING structure representing the output directory.

Return Value:

    S_OK - Output directory obtained successfully.

    E_POINTER - Context or OutputDirectory parameters were NULL.

    PH_E_CONTEXT_LOCKED - The context was locked.

    PH_E_CONTEXT_OUTPUT_DIRECTORY_NOT_SET - No output directory has been set.

--*/
{
    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Context)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(OutputDirectoryPointer)) {
        return E_POINTER;
    }

    if (!TryAcquirePerfectHashContextLockExclusive(Context)) {
        return PH_E_CONTEXT_LOCKED;
    }

    if (!Context->OutputDirectory.Buffer) {
        ReleasePerfectHashContextLockExclusive(Context);
        return PH_E_CONTEXT_OUTPUT_DIRECTORY_NOT_SET;
    }

    *OutputDirectoryPointer = &Context->OutputDirectory;

    ReleasePerfectHashContextLockExclusive(Context);

    return S_OK;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :