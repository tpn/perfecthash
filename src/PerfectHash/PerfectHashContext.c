/*++

Copyright (c) 2018-2024 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashContext.c

Abstract:

    This is the main implementation module of the perfect hash context
    component for the perfect hash table library.  A context is used
    to encapsulate threadpool resources in order to support finding
    perfect hash table solutions in parallel.

    Routines are provided for context initialization and rundown, setting and
    getting the maximum concurrency associated with a context, setting and
    getting the base output directory, callback routines for the various
    threadpool functions, and initializing key size.

--*/

#include "stdafx.h"
#include "bsthreadpool.h"

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
(NTAPI TP_TIMER_CALLBACK)(
    _Inout_     PTP_CALLBACK_INSTANCE Instance,
    _Inout_opt_ PVOID                 Context,
    _Inout_     PTP_TIMER             Work
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
TP_WORK_CALLBACK ConsoleWorkCallback;
TP_WORK_CALLBACK FinishedWorkCallback;
TP_TIMER_CALLBACK SolveTimeoutCallback;
TP_CLEANUP_GROUP_CANCEL_CALLBACK CleanupCallback;

//
// Spin count for the best graph critical section.
//

#define BEST_GRAPH_CS_SPINCOUNT 4000

//
// Helper functions.
//

ULONG
GetMainThreadpoolConcurrency(
    _In_ ULONG MaximumConcurrency
    )
/*++

Routine Description:

    This routine returns the concurrency value that should be used to set the
    main work threadpool's minimum and maximum concurrency limits.  This is
    higher than the parallel graph solving concurrency, as we need to account
    for non-solver thread callbacks (such as the solve timeout and console
    thread) which are also associated with the main work threadpool.

    The value returned by this routine should only be passed to the threadpool
    routines SetThreadpoolThreadMinimum() and SetThreadpoolThreadMaximum(); no
    other component needs to know about this value.  (The concurrency values
    captured in Context->MinimumConcurrency and Context->MaximumConcurrency
    have very specific roles in CreatePerfectHashTableImplChm01(), for example,
    and must precisely reflect the number of parallel solver graphs to be used.)

    If SetThreadpoolThreadMaximum() was called with Context->MaximumConcurrency,
    then the solve timeout and console threads would never get a chance to run.

Arguments:

    MaximumConcurrency - Supplies the maximum graph solving concurrency value.

Return Value:

    The concurrency level to be passed to SetThreadpoolThreadMinimum() and
    SetThreadpoolThreadMaximum(), which will be greater than MaximumConcurrency,
    and will take into account any additional main work thread callbacks such
    as the solve timeout and console thread.

--*/
{
    //
    // Current list of non-solver thread callbacks we need to account for:
    //  - Solve timeout
    //  - Console thread
    //

    return MaximumConcurrency + 2;
}


//
// Main context creation routine.
//

PERFECT_HASH_CONTEXT_INITIALIZE PerfectHashContextInitialize;

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
    PACL Acl = NULL;
    BYTE Index;
    BYTE NumberOfEvents;
    BOOL Success;
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
    PSTRING ComputerName;
    DWORD ComputerNameLength;
    PTP_POOL Threadpool;
    PALLOCATOR Allocator;
    ULARGE_INTEGER AllocSize;
    ULONG ThreadpoolConcurrency;
    ULARGE_INTEGER ObjectNameArraySize;
    ULARGE_INTEGER ObjectNamePointersArraySize;
    PUNICODE_STRING Name;
    PPUNICODE_STRING Names;
    PPUNICODE_STRING Prefixes;
    SYSTEM_INFO SystemInfo;
    EXPLICIT_ACCESS_W ExplicitAccess;
    ULONG NumberOfPagesForConsoleBuffer;
    SECURITY_ATTRIBUTES SecurityAttributes;
    SECURITY_DESCRIPTOR SecurityDescriptor;
    PSECURITY_ATTRIBUTES Attributes;

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
    // Create guarded lists.
    //

    Result = Context->Vtbl->CreateInstance(Context,
                                           NULL,
                                           &IID_PERFECT_HASH_GUARDED_LIST,
                                           &Context->MainWorkList);

    if (FAILED(Result)) {
        goto Error;
    }

    Result = Context->Vtbl->CreateInstance(Context,
                                           NULL,
                                           &IID_PERFECT_HASH_GUARDED_LIST,
                                           &Context->FileWorkList);

    if (FAILED(Result)) {
        goto Error;
    }

    Result = Context->Vtbl->CreateInstance(Context,
                                           NULL,
                                           &IID_PERFECT_HASH_GUARDED_LIST,
                                           &Context->FinishedWorkList);

    if (FAILED(Result)) {
        goto Error;
    }

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Allocator = Context->Allocator;

    //
    // Initialize system allocation granularity.
    //

    GetSystemInfo(&SystemInfo);
    Context->SystemAllocationGranularity = SystemInfo.dwAllocationGranularity;

    //
    // Create an exclusive DACL for use with our events.
    //

#ifdef PH_WINDOWS
    Result = CreateExclusiveDaclForCurrentUser(Rtl,
                                               &SecurityAttributes,
                                               &SecurityDescriptor,
                                               &ExplicitAccess,
                                               &Acl);

    if (FAILED(Result)) {
        PH_ERROR(CreateExclusiveDaclForCurrentUser, Result);
        goto Error;
    }

    Attributes = &SecurityAttributes;
#else
    Attributes = NULL;
#endif // PH_WINDOWS

#ifdef PH_WINDOWS

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
#endif

    //
    // Allocation of buffer was successful, continue with initialization.
    //

    Context->SizeOfStruct = sizeof(*Context);
    Context->State.AsULong = 0;
    Context->Flags.AsULong = 0;

    InitializeSRWLock(&Context->Lock);

    if (!InitializeCriticalSectionAndSpinCount(
        &Context->BestGraphCriticalSection,
        BEST_GRAPH_CS_SPINCOUNT)) {

        //
        // This should never fail from Vista onward.
        //

        PH_RAISE(PH_E_SYSTEM_CALL_FAILED);
    }

#ifdef PH_WINDOWS

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
        128,
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

#endif

    //
    // Initialize the event pointer to the first handle, and the name pointer
    // to the first UNICODE_STRING pointer.  Obtain the number of events.
    //

    Event = (PHANDLE)&Context->FirstEvent;
    Name = &Context->ObjectNames[0];
    NumberOfEvents = GetNumberOfContextEvents(Context);

#ifdef PH_WINDOWS
#define GetObjectNameBuffer(O) (O->Buffer)
#else
#define GetObjectNameBuffer(O) (NULL)
#endif

    for (Index = 0; Index < NumberOfEvents; Index++, Event++, Name++) {

        //
        // We want all of our events to be manual reset, such that they stay
        // signaled even after they've satisfied a wait.
        //

        BOOLEAN ManualReset = TRUE;

        *Event = CreateEventW(Attributes,
                              ManualReset,
                              FALSE,
                              GetObjectNameBuffer(Name));

        LastError = GetLastError();

        if (!*Event || LastError == ERROR_ALREADY_EXISTS) {

            //
            // As the event names are random, a last error that indicates the
            // name already exists is evident of a pretty serious problem.
            // Treat this as a fatal.
            //

            SYS_ERROR(CreateEventW);
            Result = PH_E_SYSTEM_CALL_FAILED;
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

    ThreadpoolConcurrency = GetMainThreadpoolConcurrency(MaximumConcurrency);

#ifdef PH_WINDOWS

    //
    // Create the Main threadpool structures.  This threadpool creates a fixed
    // number of threads equal to the maximum concurrency specified by the user
    // (e.g. min threads is set to the same value as max threads).
    //

    Threadpool = Context->MainThreadpool = CreateThreadpool(NULL);
    if (!Threadpool) {
        SYS_ERROR(CreateThreadpool);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    if (!SetThreadpoolThreadMinimum(Threadpool, ThreadpoolConcurrency)) {
        SYS_ERROR(SetThreadpoolThreadMinimum);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    SetThreadpoolThreadMaximum(Threadpool, ThreadpoolConcurrency);

    //
    // Initialize the Main threadpool and environment.
    //

    InitializeThreadpoolEnvironment(&Context->MainCallbackEnv);
    SetThreadpoolCallbackPool(&Context->MainCallbackEnv,
                              Context->MainThreadpool);

    //
    // Create a cleanup group for the Main threadpool and register it.
    //

    Context->MainCleanupGroup = CreateThreadpoolCleanupGroup();
    if (!Context->MainCleanupGroup) {
        SYS_ERROR(CreateThreadpoolCleanupGroup);
        Result = PH_E_SYSTEM_CALL_FAILED;
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
        SYS_ERROR(CreateThreadpoolWork_MainWork);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // Create a work object for the console reader thread.
    //

    Context->ConsoleWork = CreateThreadpoolWork(ConsoleWorkCallback,
                                                Context,
                                                &Context->MainCallbackEnv);

    if (!Context->ConsoleWork) {
        SYS_ERROR(CreateThreadpoolWork_ConsoleWork);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // Create the timer object for the solve timeout.
    //

#ifdef PH_WINDOWS
    Context->SolveTimeout = CreateThreadpoolTimer(SolveTimeoutCallback,
                                                  Context,
                                                  &Context->MainCallbackEnv);
    if (!Context->SolveTimeout) {
        SYS_ERROR(CreateThreadpoolTimer_SolveTimeout);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }
#endif

    //
    // Create the File threadpool structures.  Automatically clamp the min/max
    // threads for this threadpool to the number of system processors.
    //

    Threadpool = Context->FileThreadpool = CreateThreadpool(NULL);
    if (!Threadpool) {
        SYS_ERROR(CreateThreadpool);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // N.B. When debugging, it can be helpful to set the max concurrency here
    //      to 1 so that only one thread will be hitting file work callbacks
    //      at any given time.  Change the #if 0 to #if 1 to toggle this
    //      behavior.
    //

#ifdef _DBG
#if 0
    MaximumConcurrency = 1;
#endif
#endif

    SetThreadpoolThreadMaximum(Threadpool, MaximumConcurrency);

    if (!SetThreadpoolThreadMinimum(Threadpool, MaximumConcurrency)) {
        SYS_ERROR(SetThreadpoolThreadMinimum);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    SetThreadpoolThreadMaximum(Threadpool, MaximumConcurrency);

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
        Result = PH_E_SYSTEM_CALL_FAILED;
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
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // Create the Finished and Error threadpools and associated resources.
    // These are slightly easier as we only have 1 thread maximum for each
    // pool and no cleanup group is necessary.
    //

    Context->FinishedThreadpool = CreateThreadpool(NULL);
    if (!Context->FinishedThreadpool) {
        SYS_ERROR(CreateThreadpool);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    InitializeThreadpoolEnvironment(&Context->FinishedCallbackEnv);
    SetThreadpoolCallbackPool(&Context->FinishedCallbackEnv,
                              Context->FinishedThreadpool);

    if (!SetThreadpoolThreadMinimum(Context->FinishedThreadpool, 1)) {
        SYS_ERROR(SetThreadpoolThreadMinimum);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    SetThreadpoolThreadMaximum(Context->FinishedThreadpool, 1);

    Context->FinishedWork = CreateThreadpoolWork(FinishedWorkCallback,
                                                 Context,
                                                 &Context->FinishedCallbackEnv);
    if (!Context->FinishedWork) {
        SYS_ERROR(CreateThreadpoolWork);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // Create the Error threadpool.
    //

    Context->ErrorThreadpool = CreateThreadpool(NULL);
    if (!Context->ErrorThreadpool) {
        SYS_ERROR(CreateThreadpool);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    InitializeThreadpoolEnvironment(&Context->ErrorCallbackEnv);
    SetThreadpoolCallbackPool(&Context->ErrorCallbackEnv,
                              Context->ErrorThreadpool);


    if (!SetThreadpoolThreadMinimum(Context->ErrorThreadpool, 1)) {
        SYS_ERROR(SetThreadpoolThreadMinimum);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    SetThreadpoolThreadMaximum(Context->ErrorThreadpool, 1);

    Context->ErrorWork = CreateThreadpoolWork(ErrorWorkCallback,
                                              Context,
                                              &Context->ErrorCallbackEnv);
    if (!Context->ErrorWork) {
        SYS_ERROR(CreateThreadpoolWork);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }
#else // PH_COMPAT


    Context->FileThreadpool = ThreadpoolInit(MaximumConcurrency);
    if (!Context->FileThreadpool) {
        SYS_ERROR(ThreadpoolInit_FileWork);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    Context->MainThreadpool = ThreadpoolInit(ThreadpoolConcurrency);
    if (!Context->MainThreadpool) {
        SYS_ERROR(ThreadpoolInit_Main);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

#endif

    //
    // Initialize the timestamp string.
    //

    Result = InitializeTimestampString((PCHAR)&Context->TimestampBuffer,
                                       sizeof(Context->TimestampBuffer),
                                       &Context->TimestampString,
                                       &Context->FileTime.AsFileTime,
                                       &Context->SystemTime);
    if (FAILED(Result)) {
        PH_ERROR(PerfectHashContextInitialize_InitTimestampString, Result);
        goto Error;
    }

    //
    // Wire up the ComputerName string and buffer, then get the computer name.
    //

    ComputerName = &Context->ComputerName;
    ComputerName->Length = 0;
    ComputerName->MaximumLength = sizeof(Context->ComputerNameBuffer);
    ComputerNameLength = (DWORD)ComputerName->MaximumLength;
    ComputerName->Buffer = (PCHAR)&Context->ComputerNameBuffer;

    Success = GetComputerNameA((PCHAR)ComputerName->Buffer,
                               &ComputerNameLength);
    if (!Success) {
        SYS_ERROR(GetComputerNameA);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }
    ASSERT(ComputerName->Length < MAX_COMPUTERNAME_LENGTH);
    ComputerName->Length = (USHORT)ComputerNameLength;

    //
    // Obtain stdin and stdout handles.
    //

    Context->InputHandle = GetStdHandle(STD_INPUT_HANDLE);
    if (Context->InputHandle == INVALID_HANDLE_VALUE) {
        SYS_ERROR(GetStdHandle_InvalidInputHandle);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    } else if (Context->InputHandle == NULL) {
        Result = PH_E_SYSTEM_CALL_FAILED;
        PH_ERROR(GetStdHandle_NoStdInputHandle, Result);
        goto Error;
    }

    Context->OutputHandle = GetStdHandle(STD_OUTPUT_HANDLE);
    if (Context->OutputHandle == INVALID_HANDLE_VALUE) {
        SYS_ERROR(GetStdHandle_InvalidOutputHandle);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    } else if (Context->OutputHandle == NULL) {
        Result = PH_E_SYSTEM_CALL_FAILED;
        PH_ERROR(GetStdHandle_NoStdOutputHandle, Result);
        goto Error;
    }

    //
    // Create a temporary buffer that can be used to construct console output.
    //

    NumberOfPagesForConsoleBuffer = 2;
    Context->ProcessHandle = NULL;
    Result = Rtl->Vtbl->CreateBuffer(Rtl,
                                     &Context->ProcessHandle,
                                     NumberOfPagesForConsoleBuffer,
                                     NULL,
                                     &Context->ConsoleBufferSizeInBytes,
                                     &Context->ConsoleBuffer);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashContextInitialize_CreateConsoleBuffer, Result);
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

#ifdef PH_WINDOWS
    if (Acl) {
        LocalFree(Acl);
        Acl = NULL;
    }
#endif

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
    HRESULT Result;

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
    // If the context needs a reset, do it now.  This ensures the guarded
    // lists are cleared prior to rundown below.
    //

    if (Context->State.NeedsReset) {
        Result = PerfectHashContextReset(Context);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashContextReset, Result);
        }
    }

#ifdef PH_WINDOWS
    Result = PerfectHashContextTryRundownCallbackTableValuesFile(Context);
    if (FAILED(Result)) {
        PH_ERROR(TryRundownCallbackTableValuesFile, Result);
    }

    if (Context->State.HasFunctionHooking != FALSE) {
        Context->ClearFunctionEntryCallback(NULL, NULL, NULL, NULL, NULL);
        if (!FreeLibrary(Context->CallbackModule)) {
            SYS_ERROR(FreeLibrary_CallbackModule);
        }
        if (!FreeLibrary(Context->FunctionHookModule)) {
            SYS_ERROR(FreeLibrary_FunctionHookModule);
        }
    }

    //
    // Close the low-memory resource notification handle.
    //

    if (Context->LowMemoryEvent) {
        if (!CloseEvent(Context->LowMemoryEvent)) {
            SYS_ERROR(CloseHandle);
        }
        Context->LowMemoryEvent = NULL;
    }
#endif

    DestroyCuRuntimeContext(&Context->CuRuntimeContext);

    //
    // Free the solving contexts.
    //

    Allocator->Vtbl->FreePointer(Allocator, &Context->CuSolveContexts);

    //
    // Loop through all the events associated with the context and check if
    // they need to be closed.  (We do this instead of explicit calls to each
    // named event (e.g. CloseHandle(Context->ShutdownEvent)) as it means we
    // don't have to add new destructor code here every time we add an event.)
    //

    Event = (PHANDLE)&Context->FirstEvent;
    NumberOfEvents = GetNumberOfContextEvents(Context);

    for (Index = 0; Index < NumberOfEvents; Index++, Event++) {

        if (*Event && *Event != INVALID_HANDLE_VALUE) {
            if (!CloseEvent(*Event)) {
                SYS_ERROR(CloseHandle);
            }
            *Event = NULL;
        }
    }

#ifdef PH_WINDOWS

    if (Context->MainCleanupGroup) {

        CloseThreadpoolCleanupGroupMembers(Context->MainCleanupGroup,
                                           TRUE,
                                           NULL);
        Context->MainWork = NULL;
        Context->SolveTimeout = NULL;
        CloseThreadpoolCleanupGroup(Context->MainCleanupGroup);
        Context->MainCleanupGroup = NULL;

    } else {

        //
        // Invariant check: MainWork and SolveTimeout should never be set if
        // MainCleanupGroup is not set.
        //

        ASSERT(!Context->MainWork);
        ASSERT(!Context->SolveTimeout);
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

#else // PH_COMPAT

    if (Context->MainThreadpool) {
        ThreadpoolDestroy(Context->MainThreadpool);
        Context->MainThreadpool = NULL;
    }

    if (Context->FileThreadpool) {
        ThreadpoolDestroy(Context->FileThreadpool);
        Context->FileThreadpool = NULL;
    }

#endif

    if (Context->ObjectNames) {
        Allocator->Vtbl->FreePointer(Allocator,
                                     &Context->ObjectNames);
    }

    if (Context->ConsoleBuffer != NULL) {
        Result = Rtl->Vtbl->DestroyBuffer(Rtl,
                                          Context->ProcessHandle,
                                          &Context->ConsoleBuffer,
                                          Context->ConsoleBufferSizeInBytes);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashContextRundown_DestroyConsoleBuffer, Result);
        }
    }

#define EXPAND_AS_RELEASE(Verb, VUpper, Name, Upper) RELEASE(Context->Name);

    CONTEXT_FILE_WORK_TABLE_ENTRY(EXPAND_AS_RELEASE);

    DeleteCriticalSection(&Context->BestGraphCriticalSection);

    RELEASE(Context->MainWorkList);
    RELEASE(Context->FileWorkList);
    RELEASE(Context->FinishedWorkList);
    RELEASE(Context->BulkCreateCsvFile);
    RELEASE(Context->BaseOutputDirectory);
#ifdef PH_WINDOWS
    RELEASE(Context->Cu);
#endif
    RELEASE(Context->Allocator);
    RELEASE(Context->Rtl);
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
    PRTL Rtl;

    if (!ARGUMENT_PRESENT(Context)) {
        return E_POINTER;
    }

    Rtl = Context->Rtl;

    Context->AlgorithmId = PerfectHashNullAlgorithmId;
    Context->HashFunctionId = PerfectHashNullHashFunctionId;
    Context->MaskFunctionId = PerfectHashNullMaskFunctionId;

    Context->AlgorithmContext = NULL;
    Context->HashFunctionContext = NULL;
    Context->SolvedContext = NULL;

    Context->Attempts = 0;
    Context->MinAttempts = 0;
    Context->MaxAttempts = 0;
    Context->ResizeLimit = 0;
    Context->InitialTableSize = 0;
    Context->StartMilliseconds = 0;
    Context->ResizeTableThreshold = 0;
    Context->MostRecentSolvedAttempt = 0;
    Context->TargetNumberOfSolutions = 0;
    Context->HighestDeletedEdgesCount = 0;
    Context->NumberOfTableResizeEvents = 0;
    Context->TotalNumberOfAttemptsWithSmallerTableSizes = 0;
    Context->ClosestWeCameToSolvingGraphWithSmallerTableSizes = 0;

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

    Context->BaselineAttempts = 0;
    Context->BaselineFileTime.AsULongLong = 0;

    Context->GraphRegisterSolvedTsxStarted = 0;
    Context->GraphRegisterSolvedTsxSuccess = 0;
    Context->GraphRegisterSolvedTsxFailed = 0;
    Context->GraphRegisterSolvedTsxRetry = 0;

    Context->ActiveSolvingLoops = 0;
    Context->RemainingSolverLoops = 0;
    Context->GraphMemoryFailures = 0;
    Context->LowMemoryObserved = 0;
    Context->State.AllGraphsFailedMemoryAllocation = FALSE;
    Context->State.SolveTimeoutExpired = FALSE;
    Context->State.FixedAttemptsReached = FALSE;
    Context->State.MaxAttemptsReached = FALSE;
    Context->State.BestCoverageTargetValueFound = FALSE;

    Context->CyclicGraphFailures = 0;
    Context->VertexCollisionFailures = 0;

    Context->MainWorkList->Vtbl->Reset(Context->MainWorkList);
    Context->FileWorkList->Vtbl->Reset(Context->FileWorkList);
    Context->FinishedWorkList->Vtbl->Reset(Context->FinishedWorkList);

    Context->KeysSubset = NULL;
    Context->UserSeeds = NULL;
    Context->SeedMasks = NULL;

    //
    // Suppress concurrency warnings.
    //

    _Benign_race_begin_
    Context->BestGraph = NULL;
    Context->SpareGraph = NULL;
    Context->NewBestGraphCount = 0;
    Context->FirstAttemptSolved = 0;
    Context->EqualBestGraphCount = 0;
    Context->RunningSolutionsFoundRatio = 0.0;
    ZeroArray(Context->BestGraphInfo);
    _Benign_race_end_

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
    invoked by a thread in the Main group whenever SubmitThreadpoolWork() is
    called against Context->MainWork.  The caller is responsible for appending
    a work item to Context->MainWorkList prior to submission.

    This routine removes the head item from the Context->MainWorkList, then
    calls the worker routine that was registered with the context.

Arguments:

    Instance - Supplies a pointer to the callback instance responsible for this
        threadpool callback invocation.

    Ctx - Supplies a pointer to the owning PERFECT_HASH_CONTEXT.

    Work - Supplies a pointer to the TP_WORK object for this routine.

Return Value:

    None.

--*/
{
    HRESULT Result;
    PGUARDED_LIST List;
    PLIST_ENTRY ListEntry = NULL;
    PPERFECT_HASH_CONTEXT Context;

    UNREFERENCED_PARAMETER(Work);

    if (!ARGUMENT_PRESENT(Ctx)) {
        PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
        return;
    }

    //
    // Cast the Ctx variable into a suitable type, then remove the head item
    // off the list.
    //

    Context = (PPERFECT_HASH_CONTEXT)Ctx;
    List = Context->MainWorkList;

    if (!RemoveHeadMainWork(Context, &ListEntry)) {

        //
        // A spurious work item was requested but no corresponding element was
        // pushed to the list.  This typically indicates API misuse.  We could
        // terminate here, however, that's pretty drastic, so let's just log it.
        //

        Result = PH_E_CONTEXT_MAIN_WORK_LIST_EMPTY;
        PH_ERROR(PerfectHashContextMainWorkCallback, Result);
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
ConsoleWorkCallback(
    PTP_CALLBACK_INSTANCE Instance,
    PVOID Ctx,
    PTP_WORK Work
    )
/*++

Routine Description:

    This is the callback routine for the read console thread.

Arguments:

    Instance - Supplies a pointer to the callback instance responsible for this
        threadpool callback invocation.

    Ctx - Supplies a pointer to the owning PERFECT_HASH_CONTEXT.

    Work - Supplies a pointer to the TP_WORK object for this routine.

Return Value:

    None.

--*/
{
    PPERFECT_HASH_CONTEXT Context;

    UNREFERENCED_PARAMETER(Work);
    UNREFERENCED_PARAMETER(Instance);

    if (!ARGUMENT_PRESENT(Ctx)) {
        PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
        return;
    }

    //
    // Cast the Ctx variable into a suitable type, then remove the head item
    // off the list.
    //

    Context = (PPERFECT_HASH_CONTEXT)Ctx;

    //
    // Dispatch the work item to the routine registered with the context.
    //

    Context->ConsoleWorkCallback(Context);

    return;
}

_Use_decl_annotations_
VOID
SolveTimeoutCallback(
    PTP_CALLBACK_INSTANCE Instance,
    PVOID Ctx,
    PTP_TIMER Timer
    )
/*++

Routine Description:

    This is the callback routine for the Main threadpool "solve timeout" timer.

Arguments:

    Instance - Supplies a pointer to the callback instance responsible for this
        threadpool callback invocation.

    Ctx - Supplies a pointer to the owning PERFECT_HASH_CONTEXT.

    Timer - Supplies a pointer to the TP_TIMER object for this routine.

Return Value:

    None.

--*/
{
    PPERFECT_HASH_CONTEXT Context;

    UNREFERENCED_PARAMETER(Instance);
    UNREFERENCED_PARAMETER(Timer);

    if (!ARGUMENT_PRESENT(Ctx)) {
        PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
    }

    Context = (PPERFECT_HASH_CONTEXT)Ctx;

    CONTEXT_END_TIMERS(Solve);
    SetStopSolving(Context);
    Context->State.SolveTimeoutExpired = TRUE;
    SubmitThreadpoolWork(Context->FinishedWork);

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
    is called against Context->FileWork.  The caller is responsible for
    appending a work item to Context->FileWorkList prior to submission.

    This routine removes the head item off Context->FileWorkList, then calls
    the worker routine that was registered with the context.

Arguments:

    Instance - Supplies a pointer to the callback instance responsible for this
        threadpool callback invocation.

    Ctx - Supplies a pointer to the owning PERFECT_HASH_CONTEXT.

    Work - Supplies a pointer to the TP_WORK object for this routine.

Return Value:

    None.

--*/
{
    HRESULT Result;
    PGUARDED_LIST List;
    PLIST_ENTRY ListEntry = NULL;
    PPERFECT_HASH_CONTEXT Context;

    UNREFERENCED_PARAMETER(Work);

    if (!ARGUMENT_PRESENT(Ctx)) {
        PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
        return;
    }

    //
    // Cast the Ctx variable into a suitable type, then pop a list entry off
    // the file work list head.
    //

    Context = (PPERFECT_HASH_CONTEXT)Ctx;
    List = Context->FileWorkList;

    if (!RemoveHeadFileWork(Context, &ListEntry)) {
        Result = PH_E_CONTEXT_FILE_WORK_LIST_EMPTY;
        PH_ERROR(PerfectHashContextFileWorkCallback, Result);
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

    This routine will be called when a graph solving attempt has finished.
    This will be triggered by either finding a satisfactory solution, or
    hitting solving limits (max number of attempts, max number of retries,
    etc).

Arguments:

    Instance - Supplies a pointer to the callback instance responsible for this
        threadpool callback invocation.

    Ctx - Supplies a pointer to the owning PERFECT_HASH_CONTEXT.

    Work - Supplies a pointer to the TP_WORK object for this routine.

Return Value:

    None.

--*/
{
    HRESULT Result;
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

    if (FindBestMemoryCoverage(Context)) {

        //
        // Acquire the best graph critical section then determine if best
        // graph is non-NULL.  If so, a solution has been found; verify
        // the finished count is greater than 0, then clear the context's
        // best graph field and push it onto the context's finished list.
        //

        EnterCriticalSection(&Context->BestGraphCriticalSection);

        if (Context->BestGraph) {

            PGRAPH BestGraph;
            PASSIGNED_MEMORY_COVERAGE BestCoverage;

            //
            // Invariant checks: finished count and new best graph count should
            // both be greater than 0.  If they're not, raise an exception.
            //

            if (Context->FinishedCount == 0) {
                Result = PH_E_INVARIANT_CHECK_FAILED;
                PH_ERROR(Graph_BestGraphButFinishedCountIsZero, Result);
                LeaveCriticalSection(&Context->BestGraphCriticalSection);
                PH_RAISE(Result);
            } else if (Context->NewBestGraphCount <= 0) {
                Result = PH_E_INVARIANT_CHECK_FAILED;
                PH_ERROR(Graph_BestGraphButNewBestGraphCountIsZero, Result);
                LeaveCriticalSection(&Context->BestGraphCriticalSection);
                PH_RAISE(Result);
            }

            //
            // Copy the best graph's coverage information to the table.
            //

            BestGraph = Context->BestGraph;
            BestCoverage = &BestGraph->AssignedMemoryCoverage;
            CopyCoverage(Context->Table->Coverage, BestCoverage);

        } else {

            //
            // Verify our finished count is 0.
            //
            // N.B. This invariant is less critical than the one above,
            //      and may need reviewing down the track, if we ever
            //      support the notion of finding solutions but none of
            //      them meet our criteria for "best" (i.e. they didn't
            //      hit a target number of empty free cache lines, for
            //      example).
            //

            if (Context->FinishedCount != 0) {
                Result = PH_E_INVARIANT_CHECK_FAILED;
                PH_ERROR(FinishWork_NoBestGraphButFinishedCountIsNotZero,
                         Result);
                LeaveCriticalSection(&Context->BestGraphCriticalSection);
                PH_RAISE(Result);
            }
        }

        LeaveCriticalSection(&Context->BestGraphCriticalSection);
    }

    //
    // If a solution has been found, signal the success event; otherwise,
    // signal the failed event.
    //

    if (Context->FinishedCount > 0) {
        SetEvent(Context->SucceededEvent);
    } else {
        SetEvent(Context->FailedEvent);
    }

    //
    // Signal the shutdown and completed events.
    //

    SetEvent(Context->ShutdownEvent);
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
    PTP_POOL Threadpool;
    HRESULT Result = S_OK;
    ULONG ThreadpoolConcurrency;

    if (!ARGUMENT_PRESENT(Context)) {
        return E_POINTER;
    }

    if (!TryAcquirePerfectHashContextLockExclusive(Context)) {
        return PH_E_CONTEXT_LOCKED;
    }

    Rtl = Context->Rtl;

    Context->MaximumConcurrency = MaximumConcurrency;
    Context->MinimumConcurrency = MaximumConcurrency;

#ifdef PH_WINDOWS
    if (!Context->MainThreadpool) {
        ReleasePerfectHashContextLockExclusive(Context);
        return E_UNEXPECTED;
    }

    ThreadpoolConcurrency = GetMainThreadpoolConcurrency(MaximumConcurrency);
    Threadpool = Context->MainThreadpool;
    SetThreadpoolThreadMaximum(Threadpool, ThreadpoolConcurrency);

    if (!SetThreadpoolThreadMinimum(Threadpool, ThreadpoolConcurrency)) {
        SYS_ERROR(SetThreadpoolThreadMinimum);
        Result = PH_E_SET_MAXIMUM_CONCURRENCY_FAILED;
    }
#endif

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

PERFECT_HASH_CONTEXT_SET_BASE_OUTPUT_DIRECTORY
    PerfectHashContextSetBaseOutputDirectory;

_Use_decl_annotations_
HRESULT
PerfectHashContextSetBaseOutputDirectory(
    PPERFECT_HASH_CONTEXT Context,
    PCUNICODE_STRING BaseOutputDirectory
    )
/*++

Routine Description:

    Sets the base output directory for a given context.  All generated files
    will be saved to this directory.  This routine must be called before
    attempting to create any perfect hash tables.

    N.B. This routine can only be called once for a given context.  If a
         different base output directory is desired, a new context must
         be created.

Arguments:

    Context - Supplies the context for which the bsae output directory is to
        be set.

    BaseOutputDirectory - Supplies the base output directory to set.

Return Value:

    S_OK - Base output directory was set successfully.

    E_POINTER - Context or BaseOutputDirectory parameters were NULL.

    E_INVALIDARG - BaseOutputDirectory was not a valid directory string.

    E_UNEXPECTED - Internal error.

    PH_E_CONTEXT_LOCKED - The context is locked.

    PH_E_CONTEXT_BASE_OUTPUT_DIRECTORY_ALREADY_SET - The base output directory
        has already been set.

--*/
{
    HRESULT Result = S_OK;
    PPERFECT_HASH_PATH Path = NULL;
    PPERFECT_HASH_PATH_PARTS Parts = NULL;
    PPERFECT_HASH_DIRECTORY Directory;
    PERFECT_HASH_DIRECTORY_CREATE_FLAGS DirectoryCreateFlags = { 0 };

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Context)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(BaseOutputDirectory)) {
        return E_POINTER;
    }

    if (!IsValidMinimumDirectoryNullTerminatedUnicodeString(
        BaseOutputDirectory)) {
        return E_INVALIDARG;
    }

    if (!TryAcquirePerfectHashContextLockExclusive(Context)) {
        return PH_E_CONTEXT_LOCKED;
    }

    if (Context->BaseOutputDirectory) {
        ReleasePerfectHashContextLockExclusive(Context);
        return PH_E_CONTEXT_BASE_OUTPUT_DIRECTORY_ALREADY_SET;
    }

    //
    // Argument validation complete.
    //

    //
    // Create a path instance.
    //

    Result = Context->Vtbl->CreateInstance(Context,
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

    //
    // Create a directory instance.
    //

    Result = Context->Vtbl->CreateInstance(Context,
                                           NULL,
                                           &IID_PERFECT_HASH_DIRECTORY,
                                           &Context->BaseOutputDirectory);

    if (FAILED(Result)) {
        PH_ERROR(CreateInstancePerfectHashDirectory, Result);
        goto Error;
    }

    Directory = Context->BaseOutputDirectory;

    //
    // Create the directory using the base output directory path.
    //

    Result = Directory->Vtbl->Create(Directory,
                                     Path,
                                     &DirectoryCreateFlags);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashDirectoryCreate, Result);
        goto Error;
    }

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = PH_E_CONTEXT_SET_BASE_OUTPUT_DIRECTORY_FAILED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    //
    // Release Path reference if applicable, unlock the context, and return.
    //

    RELEASE(Path);

    ReleasePerfectHashContextLockExclusive(Context);

    return Result;
}

PERFECT_HASH_CONTEXT_GET_BASE_OUTPUT_DIRECTORY
    PerfectHashContextGetBaseOutputDirectory;

_Use_decl_annotations_
HRESULT
PerfectHashContextGetBaseOutputDirectory(
    PPERFECT_HASH_CONTEXT Context,
    PPERFECT_HASH_DIRECTORY *BaseOutputDirectoryPointer
    )
/*++

Routine Description:

    Obtains a previously set output directory for a given context.

Arguments:

    Context - Supplies the context for which the output directory is to be
        obtained.

    BaseOuputDirectory - Supplies the address of a variable that receives a
        pointer to the PERFECT_HASH_DIRECTORY instance of a previously set
        base output directory.  Caller is responsible for calling Release().

Return Value:

    S_OK - Base output directory obtained successfully.

    E_POINTER - Context or BaseOutputDirectory parameters were NULL.

    PH_E_CONTEXT_LOCKED - The context was locked.

    PH_E_CONTEXT_BASE_OUTPUT_DIRECTORY_NOT_SET - No base output directory has
        been set.

--*/
{
    PPERFECT_HASH_DIRECTORY Directory;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Context)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(BaseOutputDirectoryPointer)) {
        return E_POINTER;
    }

    if (!TryAcquirePerfectHashContextLockExclusive(Context)) {
        return PH_E_CONTEXT_LOCKED;
    }

    if (!Context->BaseOutputDirectory) {
        ReleasePerfectHashContextLockExclusive(Context);
        return PH_E_CONTEXT_BASE_OUTPUT_DIRECTORY_NOT_SET;
    }

    //
    // Argument validation complete.  Add a reference to the directory and
    // update the caller's pointer, then return success.
    //

    Directory = Context->BaseOutputDirectory;
    Directory->Vtbl->AddRef(Directory);

    *BaseOutputDirectoryPointer = Directory;

    ReleasePerfectHashContextLockExclusive(Context);

    return S_OK;
}


PERFECT_HASH_CONTEXT_APPLY_THREADPOOL_PRIORITIES
    PerfectHashContextApplyThreadpoolPriorities;

_Use_decl_annotations_
VOID
PerfectHashContextApplyThreadpoolPriorities(
    PPERFECT_HASH_CONTEXT Context,
    PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    )
/*++

Routine Description:

    Enumerates the given table create parameters and applies any threadpool
    priorities found to the main work and file work threadpools.

Arguments:

    Context - Supplies a pointer to the PERFECT_HASH_CONTEXT instance
        for which the threadpool priorities are to be applied.

    TableCreateParameters - Supplies a pointer to the table create params.

Return Value:

    None.

--*/
{
    ULONG Index;
    ULONG Count;
    PPERFECT_HASH_TABLE_CREATE_PARAMETER Param;

    Count = TableCreateParameters->NumberOfElements;
    Param = TableCreateParameters->Params;

    //
    // Disable "enum not handled in switch statement" warning.
    //
    //      warning C4061: enumerator 'TableCreateParameterNullId' in switch
    //                     of enum 'PERFECT_HASH_TABLE_CREATE_PARAMETER_ID'
    //                     is not explicitly handled by a case label
    //

#pragma warning(push)
#pragma warning(disable: 4061)

    for (Index = 0; Index < Count; Index++, Param++) {

        switch (Param->Id) {

            case TableCreateParameterMainWorkThreadpoolPriorityId:
                SetThreadpoolCallbackPriority(&Context->MainCallbackEnv,
                                              Param->AsTpCallbackPriority);
                break;

            case TableCreateParameterFileWorkThreadpoolPriorityId:
                SetThreadpoolCallbackPriority(&Context->FileCallbackEnv,
                                              Param->AsTpCallbackPriority);
                break;

            default:
                break;
        }

#pragma warning(pop)

    }

    return;
}


PERFECT_HASH_CONTEXT_INITIALIZE_KEY_SIZE PerfectHashContextInitializeKeySize;

_Use_decl_annotations_
HRESULT
PerfectHashContextInitializeKeySize(
    PPERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags,
    PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters,
    PULONG KeySizeInBytes
    )
/*++

Routine Description:

    Initializes a key size based on either keys load flags or table create
    parameters.  If the flag TryInferKeySizeFromKeysFilename is passed, the
    KeySizeInBytes parameter will be set to 0.  Otherwise, an attempt will
    be made to search for the KeySizeInBytes table create parameter.  If that
    is not present, the default key size (4) will be used.

    This is an internal routine that is used by the table create and bulk
    create functions; it is assumed KeysLoadFlags and TableCreateParameters
    are valid (no validation is done).

Arguments:

    KeysLoadFlags - Supplies the keys load flags.

    TableCreateParameters - Optionally supplies a pointer to the table create
        parameters, if applicable.

    KeySizeInBytes - Receives a key size suitable for passing to the keys
        Load() function.

Return Value:

    S_OK on success, otherwise, an appropriate error code.

--*/
{
    HRESULT Result = S_OK;
    PPERFECT_HASH_TABLE_CREATE_PARAMETER Param;
    PERFECT_HASH_TABLE_CREATE_PARAMETER_ID ParamId;

    if (KeysLoadFlags->TryInferKeySizeFromKeysFilename) {
        *KeySizeInBytes = 0;
        return Result;
    }

    if (TableCreateParameters == NULL) {
        *KeySizeInBytes = DEFAULT_KEY_SIZE_IN_BYTES;
        return Result;
    }

    //
    // Query the table create parameters for the key size in bytes parameter;
    // if it's present, use it, otherwise, default to ULONG.
    //

    Param = NULL;
    ParamId = TableCreateParameterKeySizeInBytesId;
    Result = GetTableCreateParameterForId(TableCreateParameters,
                                          ParamId,
                                          &Param);

    if (FAILED(Result)) {
        PH_ERROR(GetTableCreateParameterForId, Result);
        return Result;
    }

    if (Result == S_OK) {
        *KeySizeInBytes = Param->AsULong;
    } else {

        //
        // No such parameter found; use the default.
        //

        ASSERT(Result == PH_S_TABLE_CREATE_PARAMETER_NOT_FOUND);
        Result = S_OK;
        *KeySizeInBytes = DEFAULT_KEY_SIZE_IN_BYTES;
    }

    return Result;
}

PERFECT_HASH_CONTEXT_INITIALIZE_RNG PerfectHashContextInitializeRng;

_Use_decl_annotations_
HRESULT
PerfectHashContextInitializeRng(
    PPERFECT_HASH_CONTEXT Context,
    PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags,
    PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    )
/*++

Routine Description:

    Enumerates the given table create parameters, validates any RNG-related
    values, then initializes the supporting RNG infrastructure.

Arguments:

    Context - Supplies a pointer to the PERFECT_HASH_CONTEXT instance for which
        RNG initialization is to be performed.

    TableCreateFlags - Supplies a pointer to the table create flags.

    TableCreateParameters - Supplies a pointer to the table create params.

Return Value:

    S_OK - RNG initialized successfully.

    Otherwise, an appropriate error code.

--*/
{
    PRTL Rtl;
    ULONG Index;
    ULONG Count;
    HRESULT Result;
    RNG_FLAGS Flags;
    BOOLEAN SawRngSeed;
    BOOLEAN UseRandomStartSeed;
    PPERFECT_HASH_TABLE_CREATE_PARAMETER Param;

    SawRngSeed = FALSE;
    Flags.AsULong = 0;

    Rtl = Context->Rtl;
    Count = TableCreateParameters->NumberOfElements;
    Param = TableCreateParameters->Params;
    UseRandomStartSeed = (TableCreateFlags->RngUseRandomStartSeed == TRUE);

    //
    // Disable "enum not handled in switch statement" warning.
    //
    //      warning C4061: enumerator 'TableCreateParameterNullId' in switch
    //                     of enum 'PERFECT_HASH_TABLE_CREATE_PARAMETER_ID'
    //                     is not explicitly handled by a case label
    //

#pragma warning(push)
#pragma warning(disable: 4061)

    for (Index = 0; Index < Count; Index++, Param++) {

        switch (Param->Id) {

            case TableCreateParameterRngId:
                Context->RngId = Param->AsRngId;
                break;

            case TableCreateParameterRngSeedId:
                Context->RngSeed = Param->AsULongLong;
                SawRngSeed = TRUE;
                break;

            case TableCreateParameterRngSubsequenceId:
                Context->RngSubsequence = Param->AsULongLong;
                break;

            case TableCreateParameterRngOffsetId:
                Context->RngOffset = Param->AsULongLong;
                break;

            default:
                break;
        }

#pragma warning(pop)

    }

    //
    // Validate --Rng.
    //

    if (!IsValidPerfectHashRngId(Context->RngId)) {
        Context->RngId = RNG_DEFAULT_ID;
    }

    Result = PerfectHashLookupNameForId(Rtl,
                                        PerfectHashRngEnumId,
                                        Context->RngId,
                                        &Context->RngName);
    if (FAILED(Result)) {
        PH_ERROR(PerfectHashContextInitializeRng_LookupNameForId, Result);
        goto Error;
    }

    //
    // Initialize the RNG seed.
    //

    if (UseRandomStartSeed != FALSE) {
        if (SawRngSeed != FALSE) {
            Result = PH_E_RNG_USE_RANDOM_START_SEED_CONFLICTS_WITH_RNG_SEED;
            goto Error;
        }
        Flags.UseRandomStartSeed = TRUE;
    } else if (SawRngSeed == FALSE) {
        Flags.UseDefaultStartSeed = TRUE;
        Context->RngSeed = RNG_DEFAULT_SEED;
    } else {

        //
        // Nothing else to do here; the user-supplied start seed will be used
        // automatically (and has already been captured into Context->RngSeed).
        //

        NOTHING;
    }

    //
    // Copy flags over.
    //

    Context->RngFlags.AsULong = Flags.AsULong;

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

#ifdef PH_WINDOWS
_Use_decl_annotations_
HRESULT
PerfectHashContextTryPrepareCallbackTableValuesFile (
    PPERFECT_HASH_CONTEXT Context,
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags
    )
/*++

Routine Description:

    Prepares a file to save a callback DLL's table values.

Arguments:

    Context - Supplies a pointer to the PERFECT_HASH_CONTEXT instance for which
        function hook callback DLL table values are to be persisted.

    TableCreateFlags - Supplies the table create flags.

Return Value:

    S_FALSE - User requested that table saving be disabled, or the callback
        DLL did not export the expected symbols.

    S_OK - File prepared successfully.

    Otherwise, an appropriate error code.

--*/
{
    PRTL Rtl;
    HRESULT Result;
    PCHAR Output;
    PWCHAR WideOutput;
    PCHAR Buffer = NULL;
    ULONG SizeInBytes;
    STRING TimestampA = { 0, };
    UNICODE_STRING SuffixW;
    UNICODE_STRING TimestampW;
    LARGE_INTEGER EndOfFile;
    PCUNICODE_STRING BaseName;
    ULONGLONG BufferSizeInBytes = 0;
    ULONGLONG RemainingBytes;
    ULONG NumberOfPagesForBuffer;
    PCUNICODE_STRING NewDirectory;
    PPERFECT_HASH_PATH Path = NULL;
    PPERFECT_HASH_FILE File = NULL;
    PPERFECT_HASH_PATH ExistingPath;
    PPERFECT_HASH_PATH DllPath = NULL;
    PPERFECT_HASH_PATH_PARTS DllParts = NULL;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Context)) {
        return E_POINTER;
    }

    //
    // Determine if we should proceed with preparing a table values file.
    //

    if (TableCreateFlags.DisableSavingCallbackTableValues != FALSE) {
        return S_FALSE;
    }

    if ((Context->CallbackModuleNumberOfTableValues == 0)   ||
        (Context->CallbackModuleTableValueSizeInBytes == 0) ||
        (Context->CallbackDllPath == NULL) ||
        (!IsValidUnicodeStringWithMinimumLengthInChars(Context->CallbackDllPath,
                                                       4))) {

        return S_FALSE;
    }

    ASSERT(Context->CallbackModuleTableValuesFile == NULL);

    ASSERT(Context->BaseOutputDirectory != NULL);
    ASSERT(Context->BaseOutputDirectory->Path != NULL);
    ASSERT(Context->BaseOutputDirectory->Path->FullPath.Buffer != NULL);

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;

    //
    // Create a path instance for the callback DLL path.
    //

    Result = Context->Vtbl->CreateInstance(Context,
                                           NULL,
                                           &IID_PERFECT_HASH_PATH,
                                           &DllPath);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashPathCreateInstance_CallbackDllPath, Result);
        goto Error;
    }

    Result = DllPath->Vtbl->Copy(DllPath,
                                 Context->CallbackDllPath,
                                 &DllParts,
                                 NULL);
    if (FAILED(Result)) {
        PH_ERROR(PerfectHashPathCopy_CallbackDllPath, Result);
        goto Error;
    }

    //
    // Create a path instance for the table values file.
    //

    Result = Context->Vtbl->CreateInstance(Context,
                                           NULL,
                                           &IID_PERFECT_HASH_PATH,
                                           &Path);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashPathCreateInstance_TableValuesFile, Result);
        goto Error;
    }

    BaseName = &DllParts->BaseName;
    ExistingPath = Context->BaseOutputDirectory->Path;
    NewDirectory = &Context->BaseOutputDirectory->Path->FullPath;

    //
    // Create a temporary buffer to use for path construction.
    //

    NumberOfPagesForBuffer = 1;
    Result = Rtl->Vtbl->CreateBuffer(Rtl,
                                     &Context->ProcessHandle,
                                     NumberOfPagesForBuffer,
                                     NULL,
                                     &BufferSizeInBytes,
                                     &Buffer);

    if (FAILED(Result)) {
        PH_ERROR(TryPrepareCallbackTableValuesFile_CreateBuffer, Result);
        goto Error;
    }

    SizeInBytes = RTL_TIMESTAMP_FORMAT_FILE_SUFFIX_LENGTH;
    Result = InitializeTimestampStringForFileSuffix(Buffer,
                                                    SizeInBytes,
                                                    &TimestampA,
                                                    &Context->SystemTime);
    if (FAILED(Result)) {
        PH_ERROR(TryPrepareCallbackTableValuesFile_InitTimestampFile, Result);
        goto Error;
    }

    //
    // Convert the char timestamp to a wide timestamp.
    //

    Output = Buffer;
    RemainingBytes = BufferSizeInBytes - TimestampA.Length;

    Output += TimestampA.Length;
    SuffixW.Length = 0;
    SuffixW.MaximumLength = 0;
    SuffixW.Buffer = WideOutput = (PWSTR)Output;

    WIDE_OUTPUT_UNICODE_STRING(WideOutput, &TableValuesSuffix);

    ASSERT(*WideOutput == L'\0');

    SuffixW.Length = (USHORT)(RtlPointerToOffset(SuffixW.Buffer, WideOutput));
    RemainingBytes -= SuffixW.Length;

    TimestampW.Buffer = WideOutput;
    TimestampW.Length = 0;
    TimestampW.MaximumLength = (USHORT)min(0xffff, RemainingBytes);

    Result = AppendStringToUnicodeStringFast(&TimestampA, &TimestampW);
    if (FAILED(Result)) {
        PH_ERROR(
            TryPrepareCallbackTableValuesFile_AppendStringToUnicodeString,
            Result
        );
        goto Error;
    }

    RemainingBytes -= TimestampW.Length;
    SuffixW.Length += TimestampW.Length;
    SuffixW.MaximumLength -= (USHORT)min(0xffff, RemainingBytes);

    //
    // Create the path name for the table values file.
    //

    Result = Path->Vtbl->Create(Path,
                                ExistingPath,
                                NewDirectory,           // NewDirectory
                                NULL,                   // DirectorySuffix
                                BaseName,               // NewBaseName
                                &SuffixW,               // BaseNameSuffix
                                &TableValuesExtension,  // NewExtension
                                NULL,                   // NewStreamName
                                NULL,                   // Parts
                                NULL);                  // Reserved

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashPathCreate_TableValuesFile, Result);
        goto Error;
    }

    EndOfFile.QuadPart = (
        (LONGLONG)Context->CallbackModuleNumberOfTableValues *
        (LONGLONG)Context->CallbackModuleTableValueSizeInBytes
    );

    //
    // Create the file instance.
    //

    Result = Context->Vtbl->CreateInstance(Context,
                                           NULL,
                                           &IID_PERFECT_HASH_FILE,
                                           &File);
    if (FAILED(Result)) {
        PH_ERROR(TryPrepareCallbackTableValuesFile_CreateFileInstance, Result);
        goto Error;
    }

    Result = File->Vtbl->Create(File, Path, &EndOfFile, NULL, NULL);
    if (FAILED(Result)) {
        PH_ERROR(TryPrepareCallbackTableValuesFile_CreateFile, Result);
        goto Error;
    }

    //
    // File created successfully.  Save to context then finish up.
    //

    Context->CallbackModuleTableValuesFile = File;
    Context->CallbackModuleTableValuesEndOfFile.QuadPart = EndOfFile.QuadPart;
    File->Vtbl->AddRef(File);

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

    //
    // Release the path and file objects, destroy the temporary buffer if
    // applicable, then return.
    //

    RELEASE(Path);
    RELEASE(File);

    if (Buffer != NULL) {
        ASSERT(BufferSizeInBytes != 0);
        Result = Rtl->Vtbl->DestroyBuffer(Rtl,
                                          Context->ProcessHandle,
                                          &Buffer,
                                          BufferSizeInBytes);
        if (FAILED(Result)) {
            PH_ERROR(TryPrepareCallbackTableValuesFile_DestroyBuffer, Result);
        }
    }

    return Result;
}

_Use_decl_annotations_
HRESULT
PerfectHashContextTryRundownCallbackTableValuesFile (
    PPERFECT_HASH_CONTEXT Context
    )
/*++

Routine Description:

    Runs down a table values file if applicable.

Arguments:

    Context - Supplies a pointer to the PERFECT_HASH_CONTEXT instance for which
        function hook callback DLL table value rundown is to be performed.

Return Value:

    S_FALSE - No table values file is active.

    S_OK - Rundown successful.

    Otherwise, an appropriate error code.

--*/
{
    PRTL Rtl;
    HRESULT Result;
    ULONGLONG SizeInBytes;
    PPERFECT_HASH_FILE File;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Context)) {
        return E_POINTER;
    }

    //
    // Determine if we should proceed with rundown.
    //

    File = Context->CallbackModuleTableValuesFile;
    if (File == NULL) {
        return S_FALSE;
    }

    ASSERT(Context->CallbackModuleTableValues != NULL);

    //
    // Copy the callback module's table values to the memory mapped file.
    //

    Rtl = Context->Rtl;

    SizeInBytes = Context->CallbackModuleTableValuesEndOfFile.QuadPart;
    CopyMemory(File->BaseAddress,
               Context->CallbackModuleTableValues,
               Context->CallbackModuleTableValuesEndOfFile.QuadPart);

    File->NumberOfBytesWritten.QuadPart = SizeInBytes;

    //
    // Close the file mapping.
    //

    Result = File->Vtbl->Close(File, NULL);
    if (FAILED(Result)) {
        PH_ERROR(TryRundownCallbackTableValuesFile_FileClose, Result);
    }

    //
    // And finally, release the reference.
    //

    RELEASE(Context->CallbackModuleTableValuesFile);

    return Result;
}

PERFECT_HASH_CONTEXT_INITIALIZE_FUNCTION_HOOK_CALLBACK_DLL
    PerfectHashContextInitializeFunctionHookCallbackDll;

_Use_decl_annotations_
HRESULT
PerfectHashContextInitializeFunctionHookCallbackDll(
    PPERFECT_HASH_CONTEXT Context,
    PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags,
    PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    )
/*++

Routine Description:

    Enumerates the given table create parameters, identifies if a function
    hook callback DLL has been requested, and if so, initializes the required
    infrastructure.

Arguments:

    Context - Supplies a pointer to the PERFECT_HASH_CONTEXT instance for which
        function hook callback DLL initialization is to be performed.

    TableCreateFlags - Supplies a pointer to the table create flags.

    TableCreateParameters - Supplies a pointer to the table create params.

Return Value:

    S_OK - No function hook callback DLL requested.

    PH_S_FUNCTION_HOOK_CALLBACK_DLL_INITIALIZED - Successfully initialized the
        requested function hook callback DLL.

    Otherwise, an appropriate error code.

--*/
{
    PRTL Rtl;
    PVOID Proc;
    ULONG Index;
    ULONG Count;
    HRESULT Result;
    PCSTRING Name = NULL;
    PUNICODE_STRING Path = NULL;
    HMODULE CallbackModule = NULL;
    HMODULE FunctionHookModule = NULL;
    PPERFECT_HASH_PATH CallbackDllPath = NULL;
    PPERFECT_HASH_TABLE_CREATE_PARAMETER Param;
    PFUNCTION_ENTRY_CALLBACK FunctionEntryCallback;
    PSET_FUNCTION_ENTRY_CALLBACK SetFunctionEntryCallback;
    PGET_FUNCTION_ENTRY_CALLBACK GetFunctionEntryCallback;
    PCLEAR_FUNCTION_ENTRY_CALLBACK ClearFunctionEntryCallback;
    PIS_FUNCTION_ENTRY_CALLBACK_ENABLED IsFunctionEntryCallbackEnabled;

    UNREFERENCED_PARAMETER(TableCreateFlags);

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Count = TableCreateParameters->NumberOfElements;
    Param = TableCreateParameters->Params;

    //
    // Disable "enum not handled in switch statement" warning.
    //
    //      warning C4061: enumerator 'TableCreateParameterNullId' in switch
    //                     of enum 'PERFECT_HASH_TABLE_CREATE_PARAMETER_ID'
    //                     is not explicitly handled by a case label
    //

#pragma warning(push)
#pragma warning(disable: 4061)

    for (Index = 0; Index < Count; Index++, Param++) {

        switch (Param->Id) {

            case TableCreateParameterFunctionHookCallbackDllPathId:
                Path = &Param->AsUnicodeString;
                break;

            case TableCreateParameterFunctionHookCallbackFunctionNameId:
                Name = &Param->AsString;
                break;

            case TableCreateParameterFunctionHookCallbackIgnoreRipId:
                Context->CallbackModuleIgnoreRip = Param->AsULong;
                break;

            default:
                break;
        }
    }

#pragma warning(pop)

    if (Path == NULL) {

        //
        // No function hook callback DLL specified, finish up.
        //

        Result = S_OK;
        goto End;
    }

    //
    // If no name has been requested, use the default.
    //

    if (Name == NULL) {
        Name = &FunctionHookCallbackDefaultFunctionNameA;
    }

    //
    // A DLL has been specified.  Attempt to load it.
    //

    CallbackModule = LoadLibraryW(Path->Buffer);
    if (!CallbackModule) {
        SYS_ERROR(LoadLibrary);
        Result = PH_E_FAILED_TO_LOAD_FUNCTION_HOOK_CALLBACK_DLL;
        goto Error;
    }
    Context->CallbackDllPath = Path;

    //
    // Attempt to resolve the callback function.
    //

    Proc = (PVOID)GetProcAddress(CallbackModule, Name->Buffer);
    if (!Proc) {
        SYS_ERROR(GetProcAddress);
        Result = PH_E_FAILED_TO_GET_ADDRESS_OF_FUNCTION_HOOK_CALLBACK;
        goto Error;
    }
    FunctionEntryCallback = (PFUNCTION_ENTRY_CALLBACK)Proc;

    //
    // Attempt to get the table values from the callback DLL.  If we can't,
    // that's fine, it's not considered a fatal error.
    //

    Proc = (PVOID)GetProcAddress(CallbackModule, "TableValues");
    if (Proc) {
        Context->CallbackModuleTableValues = (PVOID)Proc;
    }

    Proc = (PVOID)GetProcAddress(CallbackModule, "NumberOfTableValues");
    if (Proc) {
        Context->CallbackModuleNumberOfTableValues = *((SIZE_T *)Proc);
    }

    Proc = (PVOID)GetProcAddress(CallbackModule, "TableValueSizeInBytes");
    if (Proc) {
        Context->CallbackModuleTableValueSizeInBytes = *((SIZE_T *)Proc);
    }

    //
    // Successfully resolved the callback function.  Obtain a reference to the
    // FunctionHook.dll module, then resolve the public methods.
    //

    FunctionHookModule = LoadLibraryA("FunctionHook.dll");
    if (!FunctionHookModule) {
        SYS_ERROR(LoadLibrary);
        Result = PH_E_FAILED_TO_LOAD_FUNCTION_HOOK_DLL;
        goto Error;
    }

    Proc = (PVOID)GetProcAddress(FunctionHookModule,
                                 "SetFunctionEntryCallback");
    if (!Proc) {
        SYS_ERROR(GetProcAddress);
        Result = PH_E_FAILED_TO_GET_ADDRESS_OF_SET_FUNCTION_ENTRY_CALLBACK;
        goto Error;
    }
    SetFunctionEntryCallback = (PSET_FUNCTION_ENTRY_CALLBACK)Proc;

    Proc = (PVOID)GetProcAddress(FunctionHookModule,
                                 "GetFunctionEntryCallback");
    if (!Proc) {
        SYS_ERROR(GetProcAddress);
        Result = PH_E_FAILED_TO_GET_ADDRESS_OF_GET_FUNCTION_ENTRY_CALLBACK;
        goto Error;
    }
    GetFunctionEntryCallback = (PGET_FUNCTION_ENTRY_CALLBACK)Proc;

    Proc = (PVOID)GetProcAddress(FunctionHookModule,
                                 "ClearFunctionEntryCallback");
    if (!Proc) {
        SYS_ERROR(GetProcAddress);
        Result = PH_E_FAILED_TO_GET_ADDRESS_OF_CLEAR_FUNCTION_ENTRY_CALLBACK;
        goto Error;
    }
    ClearFunctionEntryCallback = (PCLEAR_FUNCTION_ENTRY_CALLBACK)Proc;

    Proc = (PVOID)GetProcAddress(FunctionHookModule,
                                 "IsFunctionEntryCallbackEnabled");
    if (!Proc) {
        SYS_ERROR(GetProcAddress);
        Result =
            PH_E_FAILED_TO_GET_ADDRESS_OF_IS_FUNCTION_ENTRY_CALLBACK_ENABLED;
        goto Error;
    }
    IsFunctionEntryCallbackEnabled = (PIS_FUNCTION_ENTRY_CALLBACK_ENABLED)Proc;

    //
    // Everything has been loaded successfully.  Save all of the module and
    // function pointers to the context, update the state, then as the final
    // step, set the function entry callback.
    //

    Context->CallbackModule = CallbackModule;
    Context->FunctionHookModule = FunctionHookModule;
    Context->FunctionEntryCallback = FunctionEntryCallback;
    Context->SetFunctionEntryCallback = SetFunctionEntryCallback;
    Context->ClearFunctionEntryCallback = ClearFunctionEntryCallback;
    Context->IsFunctionEntryCallbackEnabled = IsFunctionEntryCallbackEnabled;

    Context->State.HasFunctionHooking = TRUE;

    SetFunctionEntryCallback(FunctionEntryCallback,
                             CallbackModule,
                             PerfectHashModuleInfo.lpBaseOfDll,
                             PerfectHashModuleInfo.SizeOfImage,
                             Context->CallbackModuleIgnoreRip);

    //
    // Save the runtime values such that the hooking can be toggled on and off
    // at runtime via console input.
    //

    GetFunctionEntryCallback(&Context->CallbackFunction,
                             &Context->CallbackContext,
                             &Context->CallbackModuleBaseAddress,
                             &Context->CallbackModuleSizeInBytes,
                             &Context->CallbackModuleIgnoreRip);

    //
    // We're done!  Indicate success and finish up.
    //

    Result = PH_S_FUNCTION_HOOK_CALLBACK_DLL_INITIALIZED;
    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    if (CallbackModule) {
        if (!FreeLibrary(CallbackModule)) {
            SYS_ERROR(FreeLibrary_CallbackModule);
        }
        CallbackModule = NULL;
    }

    if (FunctionHookModule) {
        if (!FreeLibrary(FunctionHookModule)) {
            SYS_ERROR(FreeLibrary_FunctionHookModule);
        }
        FunctionHookModule = NULL;
    }

    //
    // Intentional follow-on to End.
    //

End:

    RELEASE(CallbackDllPath);

    return Result;
}

#endif // PH_WINDOWS

_Must_inspect_result_
HRESULT
NTAPI
PerfectHashContextInitializeLowMemoryMonitor(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ BOOLEAN MonitorLowMemory
    )
/*++

Routine Description:

    If we haven't already created a low-memory notification event, this
    routine will create one and register for low-memory notifications when
    `MonitorLowMemory` is TRUE.  Windows only.

Arguments:

    Context - Supplies an instance of PERFECT_HASH_CONTEXT.

    MonitorLowMemory - Supplies a boolean value that indicates whether or not
        low-memory conditions should be monitored.

Return Value:

    S_OK on success, otherwise an appropriate error code.

--*/
{
    HANDLE Handle;
    HRESULT Result = S_OK;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Context)) {
        return E_POINTER;
    }

#ifndef PH_WINDOWS
    return Result;
#else

    //
    // If the low-memory handle is already non-NULL, we're done.  We ignore
    // checking for the invariant of the handle not being NULL but the flag
    // being FALSE.
    //

    if (Context->LowMemoryEvent) {
        return Result;
    }

    //
    // If we've been asked to monitor low-memory conditions, create an
    // appropriate low-memory resource notification handle.  If not, just
    // create a dummy event handle that will never be signaled, as this
    // simplifies the main solving logic with regards to waiting on arrays
    // of event handles.
    //

    if (MonitorLowMemory) {
        MEMORY_RESOURCE_NOTIFICATION_TYPE Type;
        Type = LowMemoryResourceNotification;
        Handle = CreateMemoryResourceNotification(Type);
        if (!IsValidHandle(Handle)) {
            SYS_ERROR(CreateMemoryResourceNotification);
            Result = E_OUTOFMEMORY;
            goto Error;
        }
    } else {
        Handle = CreateEventW(NULL, TRUE, FALSE, NULL);
        if (!IsValidHandle(Handle)) {
            SYS_ERROR(CreateEventW);
            Result = E_OUTOFMEMORY;
            goto Error;
        }
    }

    Context->LowMemoryEvent = Handle;
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

#endif
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
