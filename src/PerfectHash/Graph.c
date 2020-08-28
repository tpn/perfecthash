/*++

Copyright (c) 2018-2020 Trent Nelson <trent@trent.me>

Module Name:

    Graph.c

Abstract:

    This module implements generic graph functionality.

--*/

#include "stdafx.h"
#include "PerfectHashEventsPrivate.h"

//
// Helper macro for graph event writing.
//

#define EVENT_WRITE_GRAPH(Name)   \
    EventWriteGraph##Name##Event( \
        NULL,                     \
        Edge,                     \
        NumberOfKeys,             \
        Key,                      \
        Result,                   \
        Cycles,                   \
        Microseconds,             \
        Graph->Seed1,             \
        Graph->Seed2,             \
        Graph->Seed3,             \
        Graph->Seed4,             \
        Graph->Seed5,             \
        Graph->Seed6,             \
        Graph->Seed7,             \
        Graph->Seed8              \
    )

//
// Forward decl.
//

GRAPH_ADD_KEYS GraphHashKeysThenAdd;
GRAPH_ADD_KEYS GraphAddKeysOriginalSeededHashRoutines;
GRAPH_VERIFY GraphVerifyOriginalSeededHashRoutines;
GRAPH_CALCULATE_ASSIGNED_MEMORY_COVERAGE
    GraphCalculateAssignedMemoryCoverage_AVX2;
GRAPH_ASSIGN GraphAssign2;

//
// COM scaffolding routines for initialization and rundown.
//

GRAPH_INITIALIZE GraphInitialize;

_Use_decl_annotations_
HRESULT
GraphInitialize(
    PGRAPH Graph
    )
/*++

Routine Description:

    Initializes a graph structure.  This is a relatively simple method that
    just primes the COM scaffolding.

Arguments:

    Graph - Supplies a pointer to a GRAPH structure for which initialization
        is to be performed.

Return Value:

    S_OK - Success.

    E_POINTER - Graph is NULL.

    E_UNEXPECTED - All other errors.

--*/
{
    PRTL Rtl;
    HRESULT Result = S_OK;
    PPERFECT_HASH_TLS_CONTEXT TlsContext;
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags;

    if (!ARGUMENT_PRESENT(Graph)) {
        return E_POINTER;
    }

    Graph->SizeOfStruct = sizeof(*Graph);

    //
    // Create Rtl and Allocator components.
    //

    Result = Graph->Vtbl->CreateInstance(Graph,
                                         NULL,
                                         &IID_PERFECT_HASH_RTL,
                                         &Graph->Rtl);

    if (FAILED(Result)) {
        goto Error;
    }

    Result = Graph->Vtbl->CreateInstance(Graph,
                                         NULL,
                                         &IID_PERFECT_HASH_ALLOCATOR,
                                         &Graph->Allocator);

    if (FAILED(Result)) {
        goto Error;
    }

    //
    // Load the table create flags from the TLS context.
    //

    TlsContext = PerfectHashTlsEnsureContext();
    TableCreateFlags.AsULong = TlsContext->TableCreateFlags.AsULong;

    //
    // Override vtbl methods based on table create flags.
    //

    if (TableCreateFlags.UseOriginalSeededHashRoutines != FALSE) {

        ASSERT(TableCreateFlags.HashAllKeysFirst == FALSE);
        Graph->Vtbl->AddKeys = GraphAddKeysOriginalSeededHashRoutines;
        Graph->Vtbl->Verify = GraphVerifyOriginalSeededHashRoutines;

    } else {

        ASSERT(Graph->Vtbl->AddKeys == GraphAddKeys);
        ASSERT(Graph->Vtbl->Verify == GraphVerify);

        if (TableCreateFlags.HashAllKeysFirst != FALSE) {
            Graph->Vtbl->AddKeys = GraphHashKeysThenAdd;
        }

    }

    //
    // Use the optimized AVX2 routine for calculating assigned memory coverage
    // if the CPU supports the instruction set.
    //

    Rtl = Graph->Rtl;
    if (Rtl->CpuFeatures.AVX2 != FALSE) {
        Graph->Vtbl->CalculateAssignedMemoryCoverage =
            GraphCalculateAssignedMemoryCoverage_AVX2;
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


GRAPH_RUNDOWN GraphRundown;

_Use_decl_annotations_
VOID
GraphRundown(
    PGRAPH Graph
    )
/*++

Routine Description:

    Release all resources associated with a graph.

Arguments:

    Graph - Supplies a pointer to a GRAPH structure for which rundown is to
        be performed.

Return Value:

    None.

--*/
{
    //
    // Sanity check structure size.
    //

    ASSERT(Graph->SizeOfStruct == sizeof(*Graph));

    //
    // Release applicable COM references.
    //

    RELEASE(Graph->Rtl);
    RELEASE(Graph->Allocator);

    //
    // Free the vertex pairs array if applicable.
    //

    if (Graph->VertexPairs != NULL) {
        if (!VirtualFree(Graph->VertexPairs, 0, MEM_RELEASE)) {
            SYS_ERROR(VirtualFree);
        }
    }

    return;
}

//
// Implement main vtbl routines.
//

GRAPH_SOLVE GraphSolve;

_Use_decl_annotations_
HRESULT
GraphSolve(
    PGRAPH Graph,
    PGRAPH *NewGraphPointer
    )
/*++

Routine Description:

    Add all keys to the hypergraph using the unique seeds to hash each key into
    two vertex values, connected by a "hyper-edge".  Determine if the graph is
    acyclic, if it is, we've "solved" the graph.  If not, we haven't.

Arguments:

    Graph - Supplies a pointer to the graph to be solved.

    NewGraphPointer - Supplies the address of a variable which will receive the
        address of a new graph instance to be used for solving if the routine
        returns PH_S_USE_NEW_GRAPH_FOR_SOLVING.

Return Value:

    PH_S_STOP_GRAPH_SOLVING - Stop graph solving.

    PH_S_GRAPH_SOLVING_STOPPED - Graph solving has been stopped.

    PH_S_CONTINUE_GRAPH_SOLVING - Continue graph solving.

    PH_S_USE_NEW_GRAPH_FOR_SOLVING - Continue graph solving but use the graph
        returned via the NewGraphPointer parameter.

--*/
{
    PKEY Keys;
    PEDGE Edges;
    PGRAPH_INFO Info;
    ULONG NumberOfKeys;
    HRESULT Result;
    LONGLONG FinishedCount;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_CONTEXT Context;
    PASSIGNED_MEMORY_COVERAGE Coverage;

    //
    // Initialize aliases.
    //

    Info = Graph->Info;
    Context = Info->Context;
    Table = Context->Table;
    NumberOfKeys = Table->Keys->NumberOfElements.LowPart;
    Edges = Keys = (PKEY)Table->Keys->KeyArrayBaseAddress;

    //
    // Attempt to add all the keys to the graph.
    //

    Result = Graph->Vtbl->AddKeys(Graph, NumberOfKeys, Keys);

    if (FAILED(Result)) {

        //
        // If the failure was due to a vertex collision, increment the counter
        // and jump to the failure handler (which results in the status code
        // PH_S_CONTINUE_GRAPH_SOLVING ultimately being returned).
        //
        // For any other reason, the error is considered fatal and graph solving
        // should stop.
        //

        if (Result == PH_E_GRAPH_VERTEX_COLLISION_FAILURE) {
            InterlockedIncrement64(&Context->VertexCollisionFailures);
            goto Failed;
        }

        PH_ERROR(GraphSolve_AddKeys, Result);
        Result = PH_S_STOP_GRAPH_SOLVING;
        goto End;
    }

    MAYBE_STOP_GRAPH_SOLVING(Graph);

    //
    // We've added all of the vertices to the graph.  Determine if the graph
    // is acyclic.
    //

    if (!IsGraphAcyclic(Graph)) {

        //
        // Failed to create an acyclic graph.
        //

        InterlockedIncrement64(&Context->CyclicGraphFailures);
        goto Failed;
    }

    //
    // We created an acyclic graph.
    //

    //
    // Increment the finished count.  If the context indicates "first solved
    // graph wins", and the value is 1, we're the winning thread, so continue
    // with graph assignment.  Otherwise, just return with the stop graph
    // solving code and let the other thread finish up (i.e. perform the
    // assignment step and then persist the result).
    //

    FinishedCount = InterlockedIncrement64(&Context->FinishedCount);

    if (FirstSolvedGraphWins(Context)) {

        if (FinishedCount != 1) {

            //
            // Some other thread beat us.  Nothing left to do.
            //

            return PH_S_GRAPH_SOLVING_STOPPED;
        }
    }

    //
    // Perform the assignment step.
    //

    if (Table->GraphImpl == 1) {
        GraphAssign(Graph);
    } else {
        ASSERT(Table->GraphImpl == 2);
        GraphAssign2(Graph);
    }

    //
    // If we're in "first graph wins" mode and we reach this point, we're the
    // winning thread, so, push the graph onto the finished list head, then
    // submit the relevant finished threadpool work item and return stop graph
    // solving.
    //

    if (FirstSolvedGraphWins(Context)) {
        CONTEXT_END_TIMERS(Solve);
        SetStopSolving(Context);
        if (WantsAssignedMemoryCoverage(Graph)) {
            Graph->Vtbl->CalculateAssignedMemoryCoverage(Graph);
            CopyCoverage(Context->Table->Coverage,
                         &Graph->AssignedMemoryCoverage);
        }
        InsertHeadFinishedWork(Context, &Graph->ListEntry);
        SubmitThreadpoolWork(Context->FinishedWork);
        return PH_S_STOP_GRAPH_SOLVING;
    }

    //
    // If we reach this mode, we're in "find best memory coverage" mode, so,
    // register the solved graph then continue solving.
    //

    ASSERT(FindBestMemoryCoverage(Context));

    //
    // Calculate memory coverage information if applicable.
    //

    if (WantsAssignedMemoryCoverage(Graph)) {
        Graph->Vtbl->CalculateAssignedMemoryCoverage(Graph);
    } else if (WantsAssignedMemoryCoverageForKeysSubset(Graph)) {
        Graph->Vtbl->CalculateAssignedMemoryCoverageForKeysSubset(Graph);
    }

    //
    // This is a bit hacky; the graph traversal depth is proving to be more
    // interesting than initially thought, such that we've recently added a
    // best coverage type predicate aimed at maximizing it, which means we
    // need to make the value available from the coverage struct in order for
    // the X-macro to work, which means we're unnecessarily duplicating the
    // value at the table and coverage level.  Not particularly elegant.
    //

    Coverage = &Graph->AssignedMemoryCoverage;
    Coverage->MaxGraphTraversalDepth = Graph->MaximumTraversalDepth;

    //
    // Ditto for total traversals, empty vertices and collisions.
    //

    Coverage->TotalGraphTraversals = Graph->TotalTraversals;
    Coverage->NumberOfEmptyVertices = Graph->NumberOfEmptyVertices;
    Coverage->NumberOfCollisionsDuringAssignment = Graph->Collisions;

    //
    // Register the solved graph.  We can return this result directly.
    //

    Result = Graph->Vtbl->RegisterSolved(Graph, NewGraphPointer);

    //
    // Intentional follow-on to End.
    //

End:

    return Result;

Failed:

    InterlockedIncrement64(&Context->FailedAttempts);

    return PH_S_CONTINUE_GRAPH_SOLVING;
}

GRAPH_ADD_KEYS GraphAddKeys;

_Use_decl_annotations_
HRESULT
GraphAddKeys(
    PGRAPH Graph,
    ULONG NumberOfKeys,
    PKEY Keys
    )
/*++

Routine Description:

    Add all keys to the hypergraph using the unique seeds to hash each key into
    two vertex values, connected by a "hyper-edge".  This implementation uses
    the newer "Ex" version of the seeded hash routines.

Arguments:

    Graph - Supplies a pointer to the graph for which the keys will be added.

    NumberOfKeys - Supplies the number of keys.

    Keys - Supplies the base address of the keys array.

Return Value:

    S_OK - Success.

    PH_E_GRAPH_VERTEX_COLLISION_FAILURE - The graph encountered two vertices
        that, when masked, were identical.

--*/
{
    KEY Key = 0;
    EDGE Edge;
    PEDGE Edges;
    ULONG Mask;
    ULARGE_INTEGER Hash;
    HRESULT Result;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_SEEDED_HASH_EX SeededHashEx;

    DECL_GRAPH_COUNTER_LOCAL_VARS();

    //
    // Initialize aliases.
    //

    Table = Graph->Context->Table;
    Mask = Table->HashMask;
    SeededHashEx = SeededHashExRoutines[Table->HashFunctionId];
    Edges = (PEDGE)Keys;

    //
    // Enumerate all keys in the input set, hash them into two unique vertices,
    // then add them to the hypergraph.
    //

    Result = S_OK;

    START_GRAPH_COUNTER();

    for (Edge = 0; Edge < NumberOfKeys; Edge++) {
        Key = *Edges++;

        Hash.QuadPart = SeededHashEx(Key, &Graph->FirstSeed, Mask);

        if (Hash.HighPart == Hash.LowPart) {
            Result = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
            break;
        }

        //
        // Add the edge to the graph connecting these two vertices.
        //

        GraphAddEdge(Graph, Edge, Hash.LowPart, Hash.HighPart);
    }

    STOP_GRAPH_COUNTER(AddKeys);

    EVENT_WRITE_GRAPH(AddKeys);

    return Result;
}

GRAPH_HASH_KEYS GraphHashKeys;

_Use_decl_annotations_
HRESULT
GraphHashKeys(
    PGRAPH Graph,
    ULONG NumberOfKeys,
    PKEY Keys
    )
/*++

Routine Description:

    This routine hashes all keys into vertices without adding the resulting
    vertices to the graph.  It is used by GraphHashKeysThenAdd().

Arguments:

    Graph - Supplies a pointer to the graph for which the hash values will be
        created.

    NumberOfKeys - Supplies the number of keys.

    Keys - Supplies the base address of the keys array.

Return Value:

    S_OK - Success.

    PH_E_GRAPH_VERTEX_COLLISION_FAILURE - The graph encountered two vertices
        that, when masked, were identical.

--*/
{
    KEY Key = 0;
    EDGE Edge;
    ULONG Mask;
    PEDGE Edges;
    BOOL Success;
    HRESULT Result;
    VERTEX_PAIR Hash;
    GRAPH_FLAGS Flags;
    ULONG OldProtection;
    PULONGLONG VertexPairs;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_SEEDED_HASH_EX SeededHashEx;

    DECL_GRAPH_COUNTER_LOCAL_VARS();

    //
    // Initialize aliases.
    //

    Result = S_OK;
    Table = Graph->Context->Table;
    Mask = Table->HashMask;
    SeededHashEx = SeededHashExRoutines[Table->HashFunctionId];
    Edges = (PEDGE)Keys;

    //
    // Sanity check we can enumerate over the vertex pair elements via a
    // ULONGLONG pointer.
    //

    C_ASSERT(sizeof(*VertexPairs) == sizeof(Graph->VertexPairs));
    VertexPairs = (PULONGLONG)Graph->VertexPairs;

    //
    // Enumerate all keys in the input set and hash them into the vertex arrays.
    //

    START_GRAPH_COUNTER();

    for (Edge = 0; Edge < NumberOfKeys; Edge++) {
        Key = *Edges++;

        Hash.AsULongLong = SeededHashEx(Key, &Graph->FirstSeed, Mask);

        if (Hash.Vertex1 == Hash.Vertex2) {
            Result = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
            break;
        }

        *VertexPairs++ = Hash.AsULongLong;
    }

    STOP_GRAPH_COUNTER(HashKeys);

    EVENT_WRITE_GRAPH(HashKeys);

    if (SUCCEEDED(Result)) {

        Flags.AsULong = Graph->Flags.AsULong;

        if (Flags.VertexPairsArrayIsWriteCombined) {

            //
            // Determine if write-combining needs to be removed from the vertex
            // pairs array.  If not, issue a memory barrier to ensure we've
            // got consistency before we start reading the vertices and adding
            // them to the graph.  (We don't do this when removing the page
            // protection as that'll implicitly have a memory barrier.)
            //

            if (!Flags.RemoveWriteCombineAfterSuccessfulHashKeys) {

                MemoryBarrier();

            } else {

                Success = VirtualProtect(Graph->VertexPairs,
                                         Graph->Info->VertexPairsSizeInBytes,
                                         PAGE_READONLY,
                                         &OldProtection);

                //
                // If the call was successful, clear the write-combine flag,
                // otherwise, error out.
                //

                if (Success) {
                    Graph->Flags.VertexPairsArrayIsWriteCombined = FALSE;
                } else {
                    SYS_ERROR(VirtualProtect);
                    Result = PH_E_SYSTEM_CALL_FAILED;
                }
            }
        }
    }

    return Result;
}

GRAPH_ADD_KEYS GraphHashKeysThenAdd;

_Use_decl_annotations_
HRESULT
GraphHashKeysThenAdd(
    PGRAPH Graph,
    ULONG NumberOfKeys,
    PKEY Keys
    )
/*++

Routine Description:

    This routine is a drop-in replacement for Graph->Vtbl->AddKeys (handled by
    GraphInitialize()), and is responsible for hashing all keys into vertices
    first, then adding all resulting vertices to the graph.  This differs from
    the normal GraphAddKeys() behavior, which hashes a key into two vertices
    and immediately adds them to the graph via GraphAddEdge().  (This routine
    loops over the keys twice; once to construct all the vertices, then again
    to add them all to the graph.)

    The motivation behind this routine is to separate out the action of hashing
    keys versus adding them to the graph to better analyze performance.

Arguments:

    Graph - Supplies a pointer to the graph for which the keys will be added.

    NumberOfKeys - Supplies the number of keys.

    Keys - Supplies the base address of the keys array.

Return Value:

    S_OK - Success.

    PH_E_GRAPH_VERTEX_COLLISION_FAILURE - The graph encountered two vertices
        that, when masked, were identical.

        N.B. Unlike GraphAddKeys(), when this code is returned, none of the
             vertices will have been added to the graph at this point (versus
             having the graph in a partially-constructed state).  This has no
             impact on the behavior of the graph solving, other than potentially
             being faster overall for graphs encountering a lot of collisions
             (because the overhead of writing to all the graph's First/Next
             arrays will have been avoided).

--*/
{
    EDGE Edge;
    HRESULT Result;
    VERTEX_PAIR VertexPair;
    PVERTEX_PAIR VertexPairs;

    DECL_GRAPH_COUNTER_LOCAL_VARS();

    //
    // Attempt to hash the keys first.
    //

    Result = GraphHashKeys(Graph, NumberOfKeys, Keys);
    if (FAILED(Result)) {
        return Result;
    }

    //
    // No vertex collisions were encountered.  All the vertex pairs have been
    // written to Graph->VertexPairs, indexed by Edge.  Loop through the number
    // of edges and add the vertices to the graph.
    //

    VertexPairs = Graph->VertexPairs;

    START_GRAPH_COUNTER();

    for (Edge = 0; Edge < NumberOfKeys; Edge++) {
        VertexPair = *(VertexPairs++);
        GraphAddEdge(Graph, Edge, VertexPair.Vertex1, VertexPair.Vertex2);
    }

    STOP_GRAPH_COUNTER(AddHashedKeys);

    EventWriteGraphAddHashedKeysEvent(NULL, NumberOfKeys, Cycles, Microseconds);

    return S_OK;
}

_Use_decl_annotations_
HRESULT
GraphAddKeysOriginalSeededHashRoutines(
    PGRAPH Graph,
    ULONG NumberOfKeys,
    PKEY Keys
    )
/*++

Routine Description:

    Add all keys to the hypergraph using the unique seeds to hash each key into
    two vertex values, connected by a "hyper-edge".

Arguments:

    Graph - Supplies a pointer to the graph for which the keys will be added.

    NumberOfKeys - Supplies the number of keys.

    Keys - Supplies the base address of the keys array.

Return Value:

    S_OK - Success.

    PH_E_GRAPH_VERTEX_COLLISION_FAILURE - The graph encountered two vertices
        that, when masked, were identical.

--*/
{
    KEY Key;
    EDGE Edge;
    PEDGE Edges;
    VERTEX Vertex1;
    VERTEX Vertex2;
    ULARGE_INTEGER Hash;
    HRESULT Result;
    PPERFECT_HASH_TABLE Table;

    DECL_GRAPH_COUNTER_LOCAL_VARS();

    Table = Graph->Context->Table;
    Edges = (PEDGE)Keys;

    //
    // Enumerate all keys in the input set, hash them into two unique vertices,
    // then add them to the hypergraph.
    //

    Result = S_OK;

    START_GRAPH_COUNTER();

    for (Edge = 0; Edge < NumberOfKeys; Edge++) {
        Key = *Edges++;

        if (FAILED(Table->Vtbl->SeededHash(Table,
                                           Key,
                                           Graph->NumberOfSeeds,
                                           &Graph->FirstSeed,
                                           &Hash.QuadPart))) {
            Result = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
            break;
        }

        ASSERT(Hash.HighPart != Hash.LowPart);

        //
        // Mask the individual vertices.
        //

        MASK_HASH(Hash.LowPart, &Vertex1);
        MASK_HASH(Hash.HighPart, &Vertex2);

        //
        // We can't have two vertices point to the same location.
        // Abort this graph attempt.
        //

        if (Vertex1 == Vertex2) {
            Result = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
            break;
        }

        //
        // Add the edge to the graph connecting these two vertices.
        //

        GraphAddEdge(Graph, Edge, Vertex1, Vertex2);
    }

    STOP_GRAPH_COUNTER(AddKeys);

Error:

    return Result;
}

_Use_decl_annotations_
HRESULT
GraphVerifyOriginalSeededHashRoutines(
    PGRAPH Graph
    )
/*++

Routine Description:

    Verify a solved graph is working correctly.  This walks through the entire
    original key set, captures the index that is returned when the key is hashed
    (i.e. simulates the Index() method), sets a bit in a bitmap for each index,
    verifying that we never see the same index twice, as this would indicate a
    collision, and then finally, verify that the number of set bits in the
    bitmap exactly equals the number of keys we saw.

    N.B. The original chm.c-style modulus-oriented solution fails to pass this
         step, which actually matches the experience I had with it during my
         initial evaluation.

Arguments:

    Graph - Supplies a pointer to the graph to be verified.

Return Value:

    S_OK - Graph was solved successfully.

    PH_S_GRAPH_VERIFICATION_SKIPPED - The verification step was skipped.

    E_POINTER - Graph was NULL.

    E_OUTOFMEMORY - Out of memory.

    E_UNEXPECTED - Internal error.

    PH_E_COLLISIONS_ENCOUNTERED_DURING_GRAPH_VERIFICATION - Collisions were
        detected during graph validation.

    PH_E_NUM_ASSIGNMENTS_NOT_EQUAL_TO_NUM_KEYS_DURING_GRAPH_VERIFICATION -
        The number of value assignments did not equal the number of keys
        during graph validation.

--*/
{
    PRTL Rtl;
    KEY Key;
    KEY PreviousKey;
    PKEY Keys;
    EDGE Edge;
    PEDGE Edges;
    ULONG Bit;
    ULONG Index;
    ULONG PrevIndex;
    PULONG Values = NULL;
    VERTEX Vertex1;
    VERTEX Vertex2;
    VERTEX MaskedLow;
    VERTEX MaskedHigh;
    VERTEX PrevVertex1;
    VERTEX PrevVertex2;
    VERTEX PrevMaskedLow;
    VERTEX PrevMaskedHigh;
    PVERTEX Assigned;
    PGRAPH_INFO Info;
    ULONG NumberOfKeys;
    ULONG NumberOfAssignments;
    ULONG Collisions = 0;
    LONGLONG Combined;
    LONGLONG PrevCombined;
    ULARGE_INTEGER Hash;
    ULARGE_INTEGER PrevHash;
    HRESULT Result = S_OK;
    PALLOCATOR Allocator;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_CONTEXT Context;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Graph)) {
        return E_POINTER;
    }

    if (SkipGraphVerification(Graph)) {
        return PH_S_GRAPH_VERIFICATION_SKIPPED;
    }

    //
    // Initialize aliases.
    //

    Info = Graph->Info;
    Context = Info->Context;
    Rtl = Context->Rtl;
    Table = Context->Table;
    Allocator = Graph->Allocator;
    NumberOfKeys = Graph->NumberOfKeys;
    Edges = Keys = (PKEY)Table->Keys->KeyArrayBaseAddress;
    Assigned = Graph->Assigned;

    //
    // Sanity check our assigned bitmap is clear.
    //

    NumberOfAssignments = Rtl->RtlNumberOfSetBits(&Graph->AssignedBitmap);
    ASSERT(NumberOfAssignments == 0);

    //
    // Allocate a values array if one is not present.
    //

    Values = Graph->Values;

    if (!Values) {
        Values = Graph->Values = (PULONG)(
            Allocator->Vtbl->Calloc(
                Allocator,
                Info->ValuesSizeInBytes,
                sizeof(*Graph->Values)
            )
        );
    }

    if (!Values) {
        return E_OUTOFMEMORY;
    }

    //
    // Enumerate all keys in the input set and verify they can be resolved
    // correctly from the assigned vertex array.
    //

    for (Edge = 0; Edge < NumberOfKeys; Edge++) {
        Key = *Edges++;

        //
        // Hash the key.
        //

        SEEDED_HASH(Key, &Hash.QuadPart);

        ASSERT(Hash.QuadPart);
        ASSERT(Hash.HighPart != Hash.LowPart);

        //
        // Mask the high and low parts of the hash.
        //

        MASK_HASH(Hash.LowPart, &MaskedLow);
        MASK_HASH(Hash.HighPart, &MaskedHigh);

        //
        // Extract the individual vertices.
        //

        Vertex1 = Assigned[MaskedLow];
        Vertex2 = Assigned[MaskedHigh];

        //
        // Mask the result.
        //

        Combined = (LONGLONG)Vertex1 + (LONGLONG)Vertex2;

        MASK_INDEX(Combined, &Index);

        Bit = Index;

        //
        // Make sure we haven't seen this bit before.
        //

        if (TestGraphBit(AssignedBitmap, Bit)) {

            //
            // We've seen this index before!  Get the key that previously
            // mapped to it.
            //

            PreviousKey = Values[Index];

            SEEDED_HASH(PreviousKey, &PrevHash.QuadPart);

            MASK_HASH(PrevHash.LowPart, &PrevMaskedLow);
            MASK_HASH(PrevHash.HighPart, &PrevMaskedHigh);

            PrevVertex1 = Assigned[MaskedLow];
            PrevVertex2 = Assigned[MaskedHigh];

            PrevCombined = (LONGLONG)PrevVertex1 + (LONGLONG)PrevVertex2;

            MASK_INDEX(PrevCombined, &PrevIndex);

            Collisions++;

        }

        //
        // Set the bit and store this key in the underlying values array.
        //

        SetGraphBit(AssignedBitmap, Bit);
        Values[Index] = Key;

    }

    if (Collisions) {
        Result = PH_E_COLLISIONS_ENCOUNTERED_DURING_GRAPH_VERIFICATION;
        goto Error;
    }

    NumberOfAssignments = Rtl->RtlNumberOfSetBits(&Graph->AssignedBitmap);

    if (NumberOfAssignments != NumberOfKeys) {
        Result =
           PH_E_NUM_ASSIGNMENTS_NOT_EQUAL_TO_NUM_KEYS_DURING_GRAPH_VERIFICATION;
        goto Error;
    }

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    if (Graph->Values) {
        Allocator->Vtbl->FreePointer(Allocator, &Graph->Values);
    }

    return Result;
}

_Use_decl_annotations_
HRESULT
GraphVerify(
    PGRAPH Graph
    )
/*++

Routine Description:

    Verify a solved graph is working correctly using the new "Ex" hash routines.

Arguments:

    Graph - Supplies a pointer to the graph to be verified.

Return Value:

    S_OK - Graph was solved successfully.

    PH_S_GRAPH_VERIFICATION_SKIPPED - The verification step was skipped.

    E_POINTER - Graph was NULL.

    E_OUTOFMEMORY - Out of memory.

    E_UNEXPECTED - Internal error.

    PH_E_COLLISIONS_ENCOUNTERED_DURING_GRAPH_VERIFICATION - Collisions were
        detected during graph validation.

    PH_E_NUM_ASSIGNMENTS_NOT_EQUAL_TO_NUM_KEYS_DURING_GRAPH_VERIFICATION -
        The number of value assignments did not equal the number of keys
        during graph validation.

--*/
{
    PRTL Rtl;
    KEY Key;
    KEY PreviousKey;
    PKEY Keys;
    EDGE Edge;
    PEDGE Edges;
    ULONG Bit;
    ULONG Index;
    ULONG PrevIndex;
    ULONG HashMask;
    ULONG IndexMask;
    PULONG Values = NULL;
    VERTEX Vertex1;
    VERTEX Vertex2;
    VERTEX PrevVertex1;
    VERTEX PrevVertex2;
    PVERTEX Assigned;
    PGRAPH_INFO Info;
    ULONG NumberOfKeys;
    ULONG NumberOfAssignments;
    ULONG Collisions = 0;
    ULARGE_INTEGER Hash;
    ULARGE_INTEGER PrevHash;
    HRESULT Result = S_OK;
    PALLOCATOR Allocator;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_CONTEXT Context;
    PPERFECT_HASH_TABLE_SEEDED_HASH_EX SeededHashEx;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Graph)) {
        return E_POINTER;
    }

    if (SkipGraphVerification(Graph)) {
        return PH_S_GRAPH_VERIFICATION_SKIPPED;
    }

    //
    // Initialize aliases.
    //

    Info = Graph->Info;
    Context = Info->Context;
    Rtl = Context->Rtl;
    Table = Context->Table;
    HashMask = Table->HashMask;
    IndexMask = Table->IndexMask;
    Allocator = Graph->Allocator;
    NumberOfKeys = Graph->NumberOfKeys;
    Edges = Keys = (PKEY)Table->Keys->KeyArrayBaseAddress;
    Assigned = Graph->Assigned;
    SeededHashEx = SeededHashExRoutines[Table->HashFunctionId];

    //
    // Sanity check our assigned bitmap is clear.
    //

    NumberOfAssignments = Rtl->RtlNumberOfSetBits(&Graph->AssignedBitmap);
    ASSERT(NumberOfAssignments == 0);

    //
    // Allocate a values array if one is not present.
    //

    Values = Graph->Values;

    if (!Values) {
        Values = Graph->Values = (PULONG)(
            Allocator->Vtbl->Calloc(
                Allocator,
                Info->ValuesSizeInBytes,
                sizeof(*Graph->Values)
            )
        );
    }

    if (!Values) {
        return E_OUTOFMEMORY;
    }

    //
    // Enumerate all keys in the input set and verify they can be resolved
    // correctly from the assigned vertex array.
    //

    for (Edge = 0; Edge < NumberOfKeys; Edge++) {
        Key = *Edges++;

        //
        // Hash the key.
        //

        Hash.QuadPart = SeededHashEx(Key, &Graph->FirstSeed, HashMask);

        ASSERT(Hash.QuadPart);
        ASSERT(Hash.HighPart != Hash.LowPart);

        //
        // Extract the individual vertices.
        //

        Vertex1 = Assigned[Hash.LowPart];
        Vertex2 = Assigned[Hash.HighPart];

        //
        // Calculate the index by adding the assigned values together.
        //

        Index = (ULONG)((Vertex1 + Vertex2) & IndexMask);

        Bit = Index;

        //
        // Make sure we haven't seen this bit before.
        //

        if (TestGraphBit(AssignedBitmap, Bit)) {

            //
            // We've seen this index before!  Get the key that previously
            // mapped to it.
            //

            PreviousKey = Values[Index];

            PrevHash.QuadPart = SeededHashEx(Key, &Graph->FirstSeed, HashMask);

            PrevVertex1 = Assigned[PrevHash.LowPart];
            PrevVertex2 = Assigned[PrevHash.HighPart];

            PrevIndex = (ULONG)((PrevVertex1 + PrevVertex2) & IndexMask);

            Collisions++;

        }

        //
        // Set the bit and store this key in the underlying values array.
        //

        SetGraphBit(AssignedBitmap, Bit);
        Values[Index] = Key;

    }

    if (Collisions) {
        Result = PH_E_COLLISIONS_ENCOUNTERED_DURING_GRAPH_VERIFICATION;
        goto Error;
    }

    NumberOfAssignments = Rtl->RtlNumberOfSetBits(&Graph->AssignedBitmap);

    if (NumberOfAssignments != NumberOfKeys) {
        Result =
           PH_E_NUM_ASSIGNMENTS_NOT_EQUAL_TO_NUM_KEYS_DURING_GRAPH_VERIFICATION;
        goto Error;
    }

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    if (Graph->Values) {
        Allocator->Vtbl->FreePointer(Allocator, &Graph->Values);
    }

    return Result;
}


VOID
VerifyMemoryCoverageInvariants(
    _In_ PGRAPH Graph,
    _In_ PASSIGNED_MEMORY_COVERAGE Coverage
    );

GRAPH_CALCULATE_ASSIGNED_MEMORY_COVERAGE GraphCalculateAssignedMemoryCoverage;

_Use_decl_annotations_
VOID
GraphCalculateAssignedMemoryCoverage(
    PGRAPH Graph
    )
/*++

Routine Description:

    Calculate the memory coverage of a solved, assigned graph.  This routine
    walks the entire assigned array (see comments at the start of Graph.h for
    more info about the role of the assigned array) and calculates how many
    cache lines, pages and large pages are empty vs used.  ("Used" means one
    or more assigned values were found.)

Arguments:

    Graph - Supplies a pointer to the graph for which memory coverage of the
        assigned array is to be calculated.

Return Value:

    None.

--*/
{
    BYTE Count;
    USHORT PageCount;
    ULONG LargePageCount;
    ULONG PageIndex;
    ULONG CacheLineIndex;
    ULONG LargePageIndex;
    ULONG NumberOfCacheLines;
    ULONG TotalBytesProcessed;
    ULONG PageSizeBytesProcessed;
    ULONG LargePageSizeBytesProcessed;
    BOOLEAN FoundFirst = FALSE;
    BOOLEAN IsLastCacheLine = FALSE;
    PASSIGNED_CACHE_LINE AssignedCacheLine;
    PASSIGNED_MEMORY_COVERAGE Coverage;

    ULONG Index;
    PASSIGNED Assigned;

    Coverage = &Graph->AssignedMemoryCoverage;
    Coverage->Attempt = Graph->Attempt;
    NumberOfCacheLines = Coverage->TotalNumberOfCacheLines;
    AssignedCacheLine = (PASSIGNED_CACHE_LINE)Graph->Assigned;

    PageIndex = 0;
    LargePageIndex = 0;
    TotalBytesProcessed = 0;
    PageSizeBytesProcessed = 0;
    LargePageSizeBytesProcessed = 0;

    //
    // Enumerate the assigned array in cache-line-sized strides.
    //

    for (CacheLineIndex = 0;
         CacheLineIndex < NumberOfCacheLines;
         CacheLineIndex++) {

        Count = 0;
        IsLastCacheLine = (CacheLineIndex == NumberOfCacheLines - 1);

        //
        // Point at the first element in this cache line.
        //

        Assigned = (PASSIGNED)(AssignedCacheLine[CacheLineIndex]);

        //
        // For each cache line, enumerate over each individual element, and,
        // if it is not NULL, increment the local count and total count.
        //

        for (Index = 0; Index < NUM_ASSIGNED_PER_CACHE_LINE; Index++) {
            if (*Assigned++) {
                Count++;
                Coverage->TotalNumberOfAssigned++;
            }
        }

        ASSERT(Count >= 0 && Count <= 16);
        Coverage->NumberOfAssignedPerCacheLineCounts[Count]++;

        //
        // Increment the empty or used counters depending on whether or not
        // any assigned elements were detected.
        //

        if (!Count) {

            Coverage->NumberOfEmptyCacheLines++;

        } else {

            Coverage->NumberOfUsedCacheLines++;

            if (!FoundFirst) {
                FoundFirst = TRUE;
                Coverage->FirstCacheLineUsed = CacheLineIndex;
                Coverage->FirstPageUsed = PageIndex;
                Coverage->FirstLargePageUsed = LargePageIndex;
                Coverage->LastCacheLineUsed = CacheLineIndex;
                Coverage->LastPageUsed = PageIndex;
                Coverage->LastLargePageUsed = LargePageIndex;
                Coverage->MaxAssignedPerCacheLineCount = Count;
            } else {
                Coverage->LastCacheLineUsed = CacheLineIndex;
                Coverage->LastPageUsed = PageIndex;
                Coverage->LastLargePageUsed = LargePageIndex;
                if (Coverage->MaxAssignedPerCacheLineCount < Count) {
                    Coverage->MaxAssignedPerCacheLineCount = Count;
                }
            }

        }

        //
        // Update histograms based on the count we just observed.
        //

        Coverage->NumberOfAssignedPerCacheLine[CacheLineIndex] = Count;
        Coverage->NumberOfAssignedPerLargePage[LargePageIndex] += Count;
        Coverage->NumberOfAssignedPerPage[PageIndex] += Count;

        TotalBytesProcessed += CACHE_LINE_SIZE;
        PageSizeBytesProcessed += CACHE_LINE_SIZE;
        LargePageSizeBytesProcessed += CACHE_LINE_SIZE;

        //
        // If we've hit a page boundary, or this is the last cache line we'll
        // be processing, finalize counts for this page.  Likewise for large
        // pages.
        //

        if (PageSizeBytesProcessed == PAGE_SIZE || IsLastCacheLine) {

            PageSizeBytesProcessed = 0;
            PageCount = Coverage->NumberOfAssignedPerPage[PageIndex];

            if (PageCount) {
                Coverage->NumberOfUsedPages++;
            } else {
                Coverage->NumberOfEmptyPages++;
            }

            PageIndex++;

            if (LargePageSizeBytesProcessed == LARGE_PAGE_SIZE ||
                IsLastCacheLine) {

                LargePageSizeBytesProcessed = 0;
                LargePageCount =
                    Coverage->NumberOfAssignedPerLargePage[LargePageIndex];

                if (LargePageCount) {
                    Coverage->NumberOfUsedLargePages++;
                } else {
                    Coverage->NumberOfEmptyLargePages++;
                }

                LargePageIndex++;
            }
        }
    }

    //
    // Enumeration of the assigned array complete; verify invariants.
    //

    VerifyMemoryCoverageInvariants(Graph, Coverage);

    return;
}


_Use_decl_annotations_
VOID
GraphCalculateAssignedMemoryCoverage_AVX2(
    PGRAPH Graph
    )
/*++

Routine Description:

    AVX2 implementation of GraphCalculateAssignedMemoryCoverage().

Arguments:

    Graph - Supplies a pointer to the graph for which memory coverage of the
        assigned array is to be calculated.

Return Value:

    None.

--*/
{
    BYTE Count;
    BYTE FirstCount;
    BYTE SecondCount;
    USHORT PageCount;
    ULONG LargePageCount;
    ULONG PageIndex;
    ULONG CacheLineIndex;
    ULONG LargePageIndex;
    ULONG NumberOfCacheLines;
    ULONG TotalBytesProcessed;
    ULONG PageSizeBytesProcessed;
    ULONG LargePageSizeBytesProcessed;
    BOOLEAN FoundFirst = FALSE;
    BOOLEAN IsLastCacheLine = FALSE;
    PASSIGNED_CACHE_LINE AssignedCacheLine;
    PASSIGNED_MEMORY_COVERAGE Coverage;

    ULONG Mask;
    PBYTE Assigned;
    YMMWORD NotZerosYmm;
    YMMWORD AssignedYmm;
    YMMWORD ShiftedYmm;
    const YMMWORD AllZeros = _mm256_set1_epi8(0);

    Coverage = &Graph->AssignedMemoryCoverage;
    Coverage->Attempt = Graph->Attempt;
    NumberOfCacheLines = Coverage->TotalNumberOfCacheLines;
    AssignedCacheLine = (PASSIGNED_CACHE_LINE)Graph->Assigned;

    PageIndex = 0;
    LargePageIndex = 0;
    TotalBytesProcessed = 0;
    PageSizeBytesProcessed = 0;
    LargePageSizeBytesProcessed = 0;

    //
    // Enumerate the assigned array in cache-line-sized strides.
    //

    for (CacheLineIndex = 0;
         CacheLineIndex < NumberOfCacheLines;
         CacheLineIndex++) {

        Count = 0;
        IsLastCacheLine = (CacheLineIndex == NumberOfCacheLines - 1);

        //
        // Load 32 bytes into a YMM register and compare it against a YMM
        // register that is all zeros.  Shift the resulting comparison result
        // right 24 bits, then generate a ULONG mask (we need the shift because
        // we have to use the intrinsic _mm256_movemask_epi8() as there's no
        // _mm256_movemask_epi32()).  The population count of the resulting
        // mask provides us with the number of non-zero ULONG elements within
        // that 32 byte chunk.  Update the counts and then repeat for the
        // second 32 byte chunk.
        //

        //
        // First 32 bytes of the cache line.
        //

        Assigned = (PBYTE)AssignedCacheLine;
        AssignedYmm = _mm256_stream_load_si256((PYMMWORD)Assigned);
        NotZerosYmm = _mm256_cmpgt_epi32(AssignedYmm, AllZeros);
        ShiftedYmm = _mm256_srli_epi32(NotZerosYmm, 24);
        Mask = _mm256_movemask_epi8(ShiftedYmm);
        FirstCount = (BYTE)__popcnt(Mask);
        ASSERT(FirstCount >= 0 && FirstCount <= 8);

        //
        // Second 32 bytes of the cache line.
        //

        Assigned += 32;
        AssignedYmm = _mm256_stream_load_si256((PYMMWORD)Assigned);
        NotZerosYmm = _mm256_cmpgt_epi32(AssignedYmm, AllZeros);
        ShiftedYmm = _mm256_srli_epi32(NotZerosYmm, 24);
        Mask = _mm256_movemask_epi8(ShiftedYmm);
        SecondCount = (BYTE)__popcnt(Mask);
        ASSERT(SecondCount >= 0 && SecondCount <= 8);

        Count = FirstCount + SecondCount;
        Coverage->TotalNumberOfAssigned += Count;

        ASSERT(Count >= 0 && Count <= 16);
        Coverage->NumberOfAssignedPerCacheLineCounts[Count]++;

        //
        // Advance the cache line pointer.
        //

        AssignedCacheLine++;

        //
        // Increment the empty or used counters depending on whether or not
        // any assigned elements were detected.
        //

        if (!Count) {

            Coverage->NumberOfEmptyCacheLines++;

        } else {

            Coverage->NumberOfUsedCacheLines++;

            if (!FoundFirst) {
                FoundFirst = TRUE;
                Coverage->FirstCacheLineUsed = CacheLineIndex;
                Coverage->FirstPageUsed = PageIndex;
                Coverage->FirstLargePageUsed = LargePageIndex;
                Coverage->LastCacheLineUsed = CacheLineIndex;
                Coverage->LastPageUsed = PageIndex;
                Coverage->LastLargePageUsed = LargePageIndex;
                Coverage->MaxAssignedPerCacheLineCount = Count;
            } else {
                Coverage->LastCacheLineUsed = CacheLineIndex;
                Coverage->LastPageUsed = PageIndex;
                Coverage->LastLargePageUsed = LargePageIndex;
                if (Coverage->MaxAssignedPerCacheLineCount < Count) {
                    Coverage->MaxAssignedPerCacheLineCount = Count;
                }
            }

        }

        //
        // Update histograms based on the count we just observed.
        //

        Coverage->NumberOfAssignedPerCacheLine[CacheLineIndex] = Count;
        Coverage->NumberOfAssignedPerLargePage[LargePageIndex] += Count;
        Coverage->NumberOfAssignedPerPage[PageIndex] += Count;

        TotalBytesProcessed += CACHE_LINE_SIZE;
        PageSizeBytesProcessed += CACHE_LINE_SIZE;
        LargePageSizeBytesProcessed += CACHE_LINE_SIZE;

        //
        // If we've hit a page boundary, or this is the last cache line we'll
        // be processing, finalize counts for this page.  Likewise for large
        // pages.
        //

        if (PageSizeBytesProcessed == PAGE_SIZE || IsLastCacheLine) {

            PageSizeBytesProcessed = 0;
            PageCount = Coverage->NumberOfAssignedPerPage[PageIndex];

            if (PageCount) {
                Coverage->NumberOfUsedPages++;
            } else {
                Coverage->NumberOfEmptyPages++;
            }

            PageIndex++;

            if (LargePageSizeBytesProcessed == LARGE_PAGE_SIZE ||
                IsLastCacheLine) {

                LargePageSizeBytesProcessed = 0;
                LargePageCount =
                    Coverage->NumberOfAssignedPerLargePage[LargePageIndex];

                if (LargePageCount) {
                    Coverage->NumberOfUsedLargePages++;
                } else {
                    Coverage->NumberOfEmptyLargePages++;
                }

                LargePageIndex++;
            }
        }
    }

    //
    // Enumeration of the assigned array complete; verify invariants.
    //

    VerifyMemoryCoverageInvariants(Graph, Coverage);

    return;
}


GRAPH_CALCULATE_ASSIGNED_MEMORY_COVERAGE_FOR_KEYS_SUBSET
    GraphCalculateAssignedMemoryCoverageForKeysSubset;

_Use_decl_annotations_
VOID
GraphCalculateAssignedMemoryCoverageForKeysSubset(
    PGRAPH Graph
    )
/*++

Routine Description:

    Calculate the memory coverage of a solved, assigned graph for a subset of
    keys.  This routine enumerates each key (edge) in the subset, recalculates
    the two hash values (vertices), then increments the relevant counts for
    where each vertex resides based on cache line, page and large page.  Then,
    these counts are enumerated and for those where a count greater than zero
    is observed, the coverage's NumberOfUsedPages, NumberOfUsedLargePages and
    NumberOfUsedCacheLines counts are incremented.  A histogram is also kept
    of cache line counts (NumberOfAssignedPerCacheLineCounts).

Arguments:

    Graph - Supplies a pointer to the graph for which memory coverage of the
        assigned array for a subset of keys is to be calculated.

Return Value:

    None.

--*/
{
    KEY Key;
    ULONG Index;
    ULONG Count;
    PULONG Value;
    VERTEX Vertex1;
    VERTEX Vertex2;
    ULONG_PTR Offset1;
    ULONG_PTR Offset2;
    ULONG PageIndex1;
    ULONG PageIndex2;
    ULONG LargePageIndex1;
    ULONG LargePageIndex2;
    ULONG CacheLineIndex1;
    ULONG CacheLineIndex2;
    PKEYS_SUBSET Subset;
    ULARGE_INTEGER Hash;
    HRESULT Result = S_OK;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_CONTEXT Context;
    PASSIGNED_MEMORY_COVERAGE Coverage;
    PASSIGNED_PAGE_COUNT PageCount;
    PASSIGNED_LARGE_PAGE_COUNT LargePageCount;
    PASSIGNED_CACHE_LINE_COUNT CacheLineCount;

    //
    // Initialize aliases.
    //

    Context = Graph->Context;
    Table = Context->Table;
    Subset = Context->KeysSubset;
    Value = Subset->Values;
    Coverage = &Graph->AssignedMemoryCoverage;
    Coverage->Attempt = Graph->Attempt;

    //
    // Walk the key subset, hash each key, identify the assigned array location,
    // increment the relevant counters.
    //

    for (Index = 0; Index < Subset->NumberOfValues; Index++) {
        Key = *Value++;

        //
        // Generate both hashes for the key.
        //

        SEEDED_HASH(Key, &Hash.QuadPart);

        //
        // Mask the individual vertices.
        //

        MASK_HASH(Hash.LowPart, &Vertex1);
        MASK_HASH(Hash.HighPart, &Vertex2);

        //
        // Invariant check: the vertices must differ by this point (see the
        // check in GraphSolve()).
        //

        ASSERT(Vertex1 != Vertex2);

        //
        // Process the first vertex.
        //

        Offset1 = ((ULONG_PTR)Vertex1) << ASSIGNED_SHIFT;

        PageIndex1 = (ULONG)(Offset1 >> PAGE_SHIFT);
        LargePageIndex1 = (ULONG)(Offset1 >> LARGE_PAGE_SHIFT);
        CacheLineIndex1 = (ULONG)(Offset1 >> CACHE_LINE_SHIFT);

        Coverage->NumberOfAssignedPerPage[PageIndex1]++;
        Coverage->NumberOfAssignedPerLargePage[LargePageIndex1]++;
        Coverage->NumberOfAssignedPerCacheLine[CacheLineIndex1]++;

        //
        // Process the second vertex.
        //

        Offset2 = ((ULONG_PTR)Vertex2) << ASSIGNED_SHIFT;

        PageIndex2 = (ULONG)(Offset2 >> PAGE_SHIFT);
        LargePageIndex2 = (ULONG)(Offset2 >> LARGE_PAGE_SHIFT);
        CacheLineIndex2 = (ULONG)(Offset2 >> CACHE_LINE_SHIFT);

        Coverage->NumberOfAssignedPerPage[PageIndex2]++;
        Coverage->NumberOfAssignedPerLargePage[LargePageIndex2]++;
        Coverage->NumberOfAssignedPerCacheLine[CacheLineIndex2]++;

        //
        // Increment relevant counters if the same large page, page or cache
        // line is shared between the vertices.
        //

        if (LargePageIndex1 == LargePageIndex2) {
            Coverage->NumberOfKeysWithVerticesMappingToSameLargePage++;
            if (PageIndex1 == PageIndex2) {
                Coverage->NumberOfKeysWithVerticesMappingToSamePage++;
                if (CacheLineIndex1 == CacheLineIndex2) {
                    Coverage->NumberOfKeysWithVerticesMappingToSameCacheLine++;
                }
            }
        }
    }

    //
    // Sum the counts captured above.
    //

    PageCount = Coverage->NumberOfAssignedPerPage;
    for (Index = 0; Index < Coverage->TotalNumberOfPages; Index++) {
        Count = *PageCount++;
        if (Count > 0) {
            Coverage->NumberOfUsedPages++;
        }
    }

    LargePageCount = Coverage->NumberOfAssignedPerLargePage;
    for (Index = 0; Index < Coverage->TotalNumberOfLargePages; Index++) {
        Count = *LargePageCount++;
        if (Count > 0) {
            Coverage->NumberOfUsedLargePages++;
        }
    }

    CacheLineCount = Coverage->NumberOfAssignedPerCacheLine;
    for (Index = 0; Index < Coverage->TotalNumberOfCacheLines; Index++) {
        Count = *CacheLineCount++;
        if (Count > 0) {
            Coverage->NumberOfUsedCacheLines++;
            if (Coverage->MaxAssignedPerCacheLineCount < Count) {
                Coverage->MaxAssignedPerCacheLineCount = Count;
            }
        }
        Coverage->NumberOfAssignedPerCacheLineCounts[Count]++;
    }

    return;

    //
    // We need the following Error: label in order to use the SEEDED_HASH()
    // and MASK_HASH() macros above.  As we've already solved the graph at
    // this point (and thus, all generated vertices were usable), we shouldn't
    // hit this point, so, PH_RAISE() if we do.
    //

Error:

    Result = PH_E_UNREACHABLE_CODE;
    PH_ERROR(GraphCalculateAssignedMemoryCoverageForKeysSubset_Error, Result);
    PH_RAISE(Result);

}


//
// Disable optimization for this routine to prevent grouping of the PH_RAISE()
// statements (i.e. such that you can't tell exactly what line triggered the
// exception).
//
// N.B. This would be better addressed by individual error codes for each
//      invariant failure.
//

#pragma optimize("", off)
VOID
VerifyMemoryCoverageInvariants(
    _In_ PGRAPH Graph,
    _In_ PASSIGNED_MEMORY_COVERAGE Coverage
    )
{

    //
    // Invariant check: the total number of assigned elements we observed
    // should be less than or equal to the number of vertices.
    //

    if (Coverage->TotalNumberOfAssigned > Graph->NumberOfVertices) {
        PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
    }

    //
    // Invariant check: the number of used plus number of empty should equal
    // the total for each element type.
    //

    if (Coverage->NumberOfUsedPages + Coverage->NumberOfEmptyPages !=
        Coverage->TotalNumberOfPages) {
        PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
    }

    if (Coverage->NumberOfUsedLargePages + Coverage->NumberOfEmptyLargePages !=
        Coverage->TotalNumberOfLargePages) {
        PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
    }

    if (Coverage->NumberOfUsedCacheLines + Coverage->NumberOfEmptyCacheLines !=
        Coverage->TotalNumberOfCacheLines) {
        PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
    }

    //
    // Invariant check: the last used element should be greater than or equal
    // to the first used element.
    //

    if (Coverage->LastPageUsed < Coverage->FirstPageUsed) {
        PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
    }

    if (Coverage->LastLargePageUsed < Coverage->FirstLargePageUsed) {
        PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
    }

    if (Coverage->LastCacheLineUsed < Coverage->FirstCacheLineUsed) {
        PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
    }
}
#pragma optimize("", on)

//
// Helper macro for emitting graph-found events.  Used exclusively by the
// GraphRegisterSolved() routine.
//

#define EVENT_WRITE_GRAPH_FOUND(Name)                             \
    EventWriteGraph##Name##(                                      \
        NULL,                                                     \
        Attempt,                                                  \
        ElapsedMilliseconds,                                      \
        (ULONG)CoverageType,                                      \
        CoverageValue,                                            \
        (StopGraphSolving != FALSE),                              \
        (FoundBestGraph != FALSE),                                \
        (FoundEqualBestGraph != FALSE),                           \
        EqualCount,                                               \
        Coverage->TotalNumberOfPages,                             \
        Coverage->TotalNumberOfLargePages,                        \
        Coverage->TotalNumberOfCacheLines,                        \
        Coverage->NumberOfUsedPages,                              \
        Coverage->NumberOfUsedLargePages,                         \
        Coverage->NumberOfUsedCacheLines,                         \
        Coverage->NumberOfEmptyPages,                             \
        Coverage->NumberOfEmptyLargePages,                        \
        Coverage->NumberOfEmptyCacheLines,                        \
        Coverage->FirstPageUsed,                                  \
        Coverage->FirstLargePageUsed,                             \
        Coverage->FirstCacheLineUsed,                             \
        Coverage->LastPageUsed,                                   \
        Coverage->LastLargePageUsed,                              \
        Coverage->LastCacheLineUsed,                              \
        Coverage->TotalNumberOfAssigned,                          \
        Coverage->NumberOfKeysWithVerticesMappingToSamePage,      \
        Coverage->NumberOfKeysWithVerticesMappingToSameLargePage, \
        Coverage->NumberOfKeysWithVerticesMappingToSameCacheLine, \
        Coverage->MaxGraphTraversalDepth,                         \
        Coverage->TotalGraphTraversals,                           \
        Graph->Seeds[0],                                          \
        Graph->Seeds[1],                                          \
        Graph->Seeds[2],                                          \
        Graph->Seeds[3],                                          \
        Graph->Seeds[4],                                          \
        Graph->Seeds[5],                                          \
        Graph->Seeds[6],                                          \
        Graph->Seeds[7],                                          \
        Coverage->NumberOfAssignedPerCacheLineCounts[0],          \
        Coverage->NumberOfAssignedPerCacheLineCounts[1],          \
        Coverage->NumberOfAssignedPerCacheLineCounts[2],          \
        Coverage->NumberOfAssignedPerCacheLineCounts[3],          \
        Coverage->NumberOfAssignedPerCacheLineCounts[4],          \
        Coverage->NumberOfAssignedPerCacheLineCounts[5],          \
        Coverage->NumberOfAssignedPerCacheLineCounts[6],          \
        Coverage->NumberOfAssignedPerCacheLineCounts[7],          \
        Coverage->NumberOfAssignedPerCacheLineCounts[8],          \
        Coverage->NumberOfAssignedPerCacheLineCounts[9],          \
        Coverage->NumberOfAssignedPerCacheLineCounts[10],         \
        Coverage->NumberOfAssignedPerCacheLineCounts[11],         \
        Coverage->NumberOfAssignedPerCacheLineCounts[12],         \
        Coverage->NumberOfAssignedPerCacheLineCounts[13],         \
        Coverage->NumberOfAssignedPerCacheLineCounts[14],         \
        Coverage->NumberOfAssignedPerCacheLineCounts[15],         \
        Coverage->NumberOfAssignedPerCacheLineCounts[16]          \
    )

GRAPH_REGISTER_SOLVED GraphRegisterSolved;

_Use_decl_annotations_
HRESULT
GraphRegisterSolved(
    PGRAPH Graph,
    PGRAPH *NewGraphPointer
    )
/*++

Routine Description:

    Attempts to register a solved graph with a context if the graph's memory
    coverage is the best that's been encountered so far.

Arguments:

    Graph - Supplies a pointer to the solved graph to register.

    NewGraphPointer - Supplies the address of a variable which will receive the
        address of a new graph instance to be used for solving if the routine
        returns PH_S_USE_NEW_GRAPH_FOR_SOLVING.

Return Value:

    PH_S_CONTINUE_GRAPH_SOLVING - Continue graph solving with the current graph.

    PH_S_USE_NEW_GRAPH_FOR_SOLVING - Continue graph solving but use the graph
        returned via the NewGraphPointer parameter.

    PH_S_GRAPH_SOLVING_STOPPED - The context indicated that graph solving was
        to stop (due to a best solution already being found, or a limit being
        hit, for example).  No graph registration is performed in this instance.

--*/
{
    HRESULT Result;
    BOOLEAN HasLimit = FALSE;
    BOOLEAN IsLowestComparator = FALSE;
    BOOLEAN FoundBestGraph = FALSE;
    BOOLEAN StopGraphSolving = FALSE;
    BOOLEAN FoundEqualBestGraph = FALSE;
    ULONG Index;
    ULONG EqualCount = 0;
    ULONG BestGraphIndex = 0;
    ULONG CoverageValue = 0;
    ULONG CoverageLimit = 0;
    LONG EqualBestGraphIndex = 0;
    LONGLONG Attempt;
    PGRAPH BestGraph;
    PGRAPH SpareGraph;
    PGRAPH PreviousBestGraph;
    ULONGLONG ElapsedMilliseconds;
    PBEST_GRAPH_INFO BestGraphInfo = NULL;
    PPERFECT_HASH_CONTEXT Context;
    PASSIGNED_MEMORY_COVERAGE Coverage;
    PASSIGNED_MEMORY_COVERAGE BestCoverage = NULL;
    PASSIGNED_MEMORY_COVERAGE PreviousBestCoverage;
    PERFECT_HASH_TABLE_BEST_COVERAGE_TYPE_ID CoverageType;

    //
    // Initialize aliases.
    //

    Context = Graph->Context;
    Coverage = &Graph->AssignedMemoryCoverage;
    CoverageType = Context->BestCoverageType;
    Attempt = Coverage->Attempt;
    ElapsedMilliseconds = GetTickCount64() - Context->StartMilliseconds;

    //
    // Indicate continue graph solving unless we find a best graph.
    //

    Result = PH_S_CONTINUE_GRAPH_SOLVING;

    //
    // Enter the best graph critical section.
    //

    EnterCriticalSection(&Context->BestGraphCriticalSection);

    //
    // If there is no best graph currently set, proceed with setting it to
    // our current graph, then use the spare graph to continue solving.
    //

    if (Context->BestGraph) {
        BestGraph = Context->BestGraph;
        ASSERT(Context->NewBestGraphCount > 0);
    } else {
        ASSERT(Context->NewBestGraphCount == 0);
        SpareGraph = Context->SpareGraph;
        ASSERT(SpareGraph != NULL);
        ASSERT(IsSpareGraph(SpareGraph));
        SpareGraph->Flags.IsSpare = FALSE;
        Context->SpareGraph = NULL;
        BestGraph = Context->BestGraph = Graph;
        *NewGraphPointer = SpareGraph;
        BestGraphIndex = Context->NewBestGraphCount++;
        Result = PH_S_USE_NEW_GRAPH_FOR_SOLVING;
        goto End;
    }

    //
    // There's an existing best graph set.  Verify spare graph is NULL, then
    // initialize aliases to the previous best.
    //

    ASSERT(Context->SpareGraph == NULL);
    PreviousBestGraph = Context->BestGraph;
    PreviousBestCoverage = &PreviousBestGraph->AssignedMemoryCoverage;

    //
    // Determine if this graph has the "best" memory coverage and update the
    // best graph accordingly if so.
    //

#define EXPAND_AS_DETERMINE_IF_BEST_GRAPH(Name, Comparison, Comparator) \
    case BestCoverageType##Comparison##Name##Id:                        \
        CoverageValue = Coverage->##Name;                               \
        if (Coverage->##Name Comparator PreviousBestCoverage->##Name) { \
            Context->BestGraph = Graph;                                 \
            *NewGraphPointer = PreviousBestGraph;                       \
            BestGraphIndex = Context->NewBestGraphCount++;              \
            Result = PH_S_USE_NEW_GRAPH_FOR_SOLVING;                    \
        } else if (Coverage->##Name == PreviousBestCoverage->##Name) {  \
            Context->EqualBestGraphCount++;                             \
            FoundEqualBestGraph = TRUE;                                 \
            EqualBestGraphIndex = Context->NewBestGraphCount - 1;       \
        }                                                               \
        break;

    switch (CoverageType) {

        case BestCoverageTypeNullId:
        case BestCoverageTypeInvalidId:
            PH_RAISE(PH_E_UNREACHABLE_CODE);
            break;

        BEST_COVERAGE_TYPE_TABLE_ENTRY(EXPAND_AS_DETERMINE_IF_BEST_GRAPH)

        default:
            Result = PH_E_INVALID_BEST_COVERAGE_TYPE_ID;
            break;
    }

    //
    // Intentional follow-on to End.
    //

End:

    StopGraphSolving = FALSE;
    FoundBestGraph = (Result == PH_S_USE_NEW_GRAPH_FOR_SOLVING);

    if (FoundEqualBestGraph) {

        //
        // If this graph was found to be equal to the current best graph, update
        // the existing best graph info's equal count.
        //

        ASSERT(!FoundBestGraph);
        ASSERT(EqualBestGraphIndex >= 0);

        BestGraphInfo = &Context->BestGraphInfo[EqualBestGraphIndex];
        EqualCount = ++BestGraphInfo->EqualCount;

        //
        // If we've hit the maximum number of equal graphs, we can stop solving.
        //

        if (Context->MaxNumberOfEqualBestGraphs > 0 &&
            Context->MaxNumberOfEqualBestGraphs <= EqualCount)
        {
            StopGraphSolving = TRUE;
        }


    } else if (FoundBestGraph) {

        BestGraph = Graph;

        //
        // If we're still within the limits for the maximum number of best
        // graphs (captured within our context), then use the relevant element
        // from that array.
        //

        if (BestGraphIndex < MAX_BEST_GRAPH_INFO) {

            BestGraphInfo = &Context->BestGraphInfo[BestGraphIndex];

            //
            // Invariant check: address of BestGraphInfo element in the array
            // should be less than the address of next element in the struct
            // (the LowMemoryEvent handle).
            //

            ASSERT((ULONG_PTR)(BestGraphInfo) <
                   (ULONG_PTR)(&Context->LowMemoryEvent));

            //
            // Initialize the pointer to the best graph info's copy of the
            // coverage structure; we can copy this over outside the critical
            // section.
            //

            BestCoverage = &BestGraphInfo->Coverage;

        } else {

            //
            // Nothing to do if we've exceeded the number of best graphs we
            // capture in the context.  (The information may still be emitted
            // via an ETW event.)
            //

            ASSERT(BestCoverage == NULL);
        }

        //
        // Capture the value used to determine that this graph was the best.
        //

#define EXPAND_AS_SAVE_BEST_GRAPH_VALUE(Name, Comparison, Comparator) \
    case BestCoverageType##Comparison##Name##Id:                      \
        BestGraphInfo->Value = Coverage->##Name;                      \
        break;

        switch (CoverageType) {

            case BestCoverageTypeNullId:
            case BestCoverageTypeInvalidId:
                PH_RAISE(PH_E_UNREACHABLE_CODE);
                break;

            BEST_COVERAGE_TYPE_TABLE_ENTRY(EXPAND_AS_SAVE_BEST_GRAPH_VALUE)

            default:
                Result = PH_E_INVALID_BEST_COVERAGE_TYPE_ID;
                break;
        }
    }

    //
    // Determine if we've found sufficient "best" graphs whilst we still have
    // the critical section acquired (as NewBestGraphCount is protected by it).
    //

    if (!StopGraphSolving) {
        StopGraphSolving = (
            (ULONGLONG)Context->NewBestGraphCount >=
            Context->BestCoverageAttempts
        );
    }

    //
    // Leave the critical section and complete processing.
    //

    LeaveCriticalSection(&Context->BestGraphCriticalSection);

    //
    // Any failure code at this point is a critical internal invariant failure.
    //

    if (FAILED(Result)) {
        PH_RAISE(Result);
    }

    //
    // If we found a new best graph, BestCoverage will be non-NULL.
    //

    if (BestCoverage != NULL) {

        //
        // Copy the coverage, attempt, elapsed milliseconds and seeds.
        //

        CopyCoverage(BestCoverage, Coverage);

        BestGraphInfo->Attempt = Attempt;
        BestGraphInfo->ElapsedMilliseconds = ElapsedMilliseconds;

        C_ASSERT(sizeof(BestGraphInfo->Seeds) == sizeof(Graph->Seeds));

        ASSERT(BestGraphInfo != NULL);

        for (Index = 0; Index < Graph->NumberOfSeeds; Index++) {
            BestGraphInfo->Seeds[Index] = Graph->Seeds[Index];
        }

    }

    //
    // We need to determine what type of comparator is being used (i.e. lowest
    // or highest), because depending on what we're using for comparison, we
    // may have hit the lowest or highest possible value, in which case, graph
    // solving can be stopped (even if we haven't hit the target specified by
    // --BestCoverageAttempts).  For example, if the best coverage type is
    // LowestNumberOfEmptyCacheLines, and the coverage value is 0, we'll never
    // beat this, so we can stop graph solving now.
    //
    // So, leverage another X-macro expansion to extract the comparator type and
    // coverage value.  We need to do this here, after the End: label, as the
    // very first graph being registered may have hit the limit (which we have
    // seen happen regularly in practice).
    //

    Coverage = &BestGraph->AssignedMemoryCoverage;

#define EXPAND_AS_DETERMINE_IF_LOWEST(Name, Comparison, Comparator) \
    case BestCoverageType##Comparison##Name##Id:                    \
        IsLowestComparator = (0 Comparator 1);                      \
        CoverageValue = Coverage->##Name;                           \
        break;

    switch (CoverageType) {

        case BestCoverageTypeNullId:
        case BestCoverageTypeInvalidId:
            PH_RAISE(PH_E_UNREACHABLE_CODE);
            break;

        BEST_COVERAGE_TYPE_TABLE_ENTRY(EXPAND_AS_DETERMINE_IF_LOWEST)

        default:
            Result = PH_E_INVALID_BEST_COVERAGE_TYPE_ID;
            break;
    }

    if (FAILED(Result)) {
        PH_RAISE(Result);
    }

    if (IsLowestComparator) {

        //
        // The comparator is "lowest"; if the best value we found was zero, then
        // indicate stop solving, as we'll never be able to get lower than this.
        //

        if (CoverageValue == 0) {
            StopGraphSolving = TRUE;
        }

    } else {

        //
        // For highest comparisons, things are a little trickier, as we need to
        // know what is the maximum value to compare things against.  This info
        // isn't available from the X-macro, nor is it applicable to all types,
        // so, the following switch construct extracts limits manually.
        //

        HasLimit = FALSE;

        //
        // Disable "enum not handled in switch statement" warning.
        //
        //      warning C4061: enumerator 'TableCreateParameterNullId' in switch
        //                     of enum 'PERFECT_HASH_TABLE_CREATE_PARAMETER_ID'
        //                     is not explicitly handled by a case label
        //

#pragma warning(push)
#pragma warning(disable: 4061)

        switch (CoverageType) {

            case BestCoverageTypeNullId:
            case BestCoverageTypeInvalidId:
                PH_RAISE(PH_E_UNREACHABLE_CODE);
                break;

            case BestCoverageTypeHighestNumberOfEmptyCacheLinesId:
                HasLimit = TRUE;
                CoverageLimit = Coverage->TotalNumberOfCacheLines;
                break;

            case BestCoverageTypeHighestMaxAssignedPerCacheLineCountId:
                HasLimit = TRUE;
                CoverageLimit = 16;
                break;

            default:
                break;
        }

#pragma warning(pop)

        if (HasLimit) {
            if (CoverageValue == CoverageLimit) {
                StopGraphSolving = TRUE;
            }
        }
    }

    //
    // Communicate back to the context that solving can stop if indicated.
    //

    if (StopGraphSolving) {

        SetStopSolving(Context);

        //
        // Stop the solve timers here.  (These are less useful when not in
        // "first graph wins" mode.)
        //

        CONTEXT_END_TIMERS(Solve);

        //
        // Submit the finished threadpool work regardless of whether or
        // not a graph was found.  The finished callback will set the
        // appropriate success or failure events after waiting for all
        // the graph contexts to finish and then assessing the context.
        //

        SubmitThreadpoolWork(Context->FinishedWork);

        //
        // Clear the caller's NewGraphPointer, as we're not going to be doing
        // any more graph solving.
        //

        *NewGraphPointer = NULL;

        //
        // Return graph solving stopped.
        //

        Result = PH_S_GRAPH_SOLVING_STOPPED;
    }

    //
    // Emit the relevant ETW event.  (We use different ETW events for graph
    // found, found equal best, and found new best, because they occur at very
    // different frequencies, and have separate ETW keywords (to facilitate
    // isolation of just the specific event you're interested in).
    //

    if (FoundBestGraph != FALSE) {
        EVENT_WRITE_GRAPH_FOUND(FoundNewBest);
    } else if (FoundEqualBestGraph != FALSE) {
        EVENT_WRITE_GRAPH_FOUND(FoundEqualBest);
    }

    EVENT_WRITE_GRAPH_FOUND(Found);

    return Result;
}


#ifdef PERFECTHASH_X64_TSX

//
// TSX version of above.
//

_No_competing_thread_begin_

GRAPH_REGISTER_SOLVED GraphRegisterSolvedTsx;

_Use_decl_annotations_
HRESULT
GraphRegisterSolvedTsx(
    PGRAPH Graph,
    PGRAPH *NewGraphPointer
    )
/*++

Routine Description:

    Attempts to register a solved graph with a context if the graph's memory
    coverage is the best that's been encountered so far.

Arguments:

    Graph - Supplies a pointer to the solved graph to register.

    NewGraphPointer - Supplies the address of a variable which will receive the
        address of a new graph instance to be used for solving if the routine
        returns PH_S_USE_NEW_GRAPH_FOR_SOLVING.

Return Value:

    PH_S_CONTINUE_GRAPH_SOLVING - Continue graph solving with the current graph.

    PH_S_USE_NEW_GRAPH_FOR_SOLVING - Continue graph solving but use the graph
        returned via the NewGraphPointer parameter.

    PH_S_GRAPH_SOLVING_STOPPED - The context indicated that graph solving was
        to stop (due to a best solution already being found, or a limit being
        hit, for example).  No graph registration is performed in this instance.

--*/
{
    ULONG Status;
    ULONG Retries = 0;
    ULONG Started = 0;
    HRESULT Result = PH_S_CONTINUE_GRAPH_SOLVING;
    PGRAPH PreviousBestGraph;
    PPERFECT_HASH_CONTEXT Context;
    PASSIGNED_MEMORY_COVERAGE Coverage;
    PASSIGNED_MEMORY_COVERAGE PreviousBestCoverage;
    PERFECT_HASH_TABLE_BEST_COVERAGE_TYPE_ID CoverageType;

    //
    // Initialize aliases.
    //

    Context = Graph->Context;
    Coverage = &Graph->AssignedMemoryCoverage;
    CoverageType = Context->BestCoverageType;

    if (!Context->BestGraph) {
        return GraphRegisterSolved(Graph, NewGraphPointer);
    }

Retry:

    Status = _xbegin();

    if (Status & _XABORT_RETRY) {
        Retries++;
        goto Retry;
    } else if (Status != _XBEGIN_STARTED) {
        InterlockedIncrement(&Context->GraphRegisterSolvedTsxFailed);
        return GraphRegisterSolved(Graph, NewGraphPointer);
    }

    Started++;

    //
    // There's an existing best graph set.
    //

    PreviousBestGraph = Context->BestGraph;
    PreviousBestCoverage = &PreviousBestGraph->AssignedMemoryCoverage;

    //
    // Determine if this graph has the "best" memory coverage and update the
    // best graph accordingly if so.
    //

#define EXPAND_AS_DETERMINE_IF_BEST_GRAPH(Name, Comparison, Comparator) \
    case BestCoverageType##Comparison##Name##Id:                        \
        if (Coverage->##Name Comparator PreviousBestCoverage->##Name) { \
            Context->BestGraph = Graph;                                 \
            *NewGraphPointer = PreviousBestGraph;                       \
            Context->NewBestGraphCount++;                               \
            Result = PH_S_USE_NEW_GRAPH_FOR_SOLVING;                    \
        } else if (Coverage->##Name == PreviousBestCoverage->##Name) {  \
            Context->EqualBestGraphCount++;                             \
        }                                                               \
        break;

    switch (CoverageType) {

        BEST_COVERAGE_TYPE_TABLE_ENTRY(EXPAND_AS_DETERMINE_IF_BEST_GRAPH)

        default:
            Result = PH_E_INVALID_BEST_COVERAGE_TYPE_ID;
            break;
    }

    _xend();

    if (Started > 0) {
        InterlockedAdd(&Context->GraphRegisterSolvedTsxStarted, Started);
    }

    if (Retries > 0) {
        InterlockedAdd(&Context->GraphRegisterSolvedTsxRetry, Retries);
    }

    InterlockedIncrement(&Context->GraphRegisterSolvedTsxSuccess);

    //
    // Any failure code at this point is a critical internal invariant failure.
    //

    if (FAILED(Result)) {
        PH_RAISE(Result);
    }

    return Result;
}

_No_competing_thread_end_

#endif // PERFECTHASH_X64_TSX


GRAPH_SET_INFO GraphSetInfo;

_Use_decl_annotations_
HRESULT
GraphSetInfo(
    PGRAPH Graph,
    PGRAPH_INFO Info
    )
/*++

Routine Description:

    Registers information about a graph with an individual graph instance.
    This routine is called once per unique graph info (that is, if a table
    resize event occurs it will be called again with the new graph info).
    The LoadInfo() routine will use the provided info for allocating or
    reallocating the necessary buffers required for graph solving.

    N.B. This routine is intended to be called from the "main" thread, whereas
         LoadInfo() is intended to be called as the first operation by graph
         solving worker threads.  Thus, this routine is pretty simple.

Arguments:

    Graph - Supplies a pointer to the graph instance.

    Info - Supplies a pointer to the graph info instance.

Return Value:

    S_OK - Success.

    E_POINTER - Graph or Info were NULL.

--*/
{
    if (!ARGUMENT_PRESENT(Graph)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Info)) {
        return E_POINTER;
    }

    Graph->Info = Info;
    Graph->Flags.IsInfoSet = TRUE;

    return S_OK;
}


GRAPH_ENTER_SOLVING_LOOP GraphEnterSolvingLoop;

_Use_decl_annotations_
HRESULT
GraphEnterSolvingLoop(
    PGRAPH Graph
    )
/*++

Routine Description:

    Enters the graph solving loop.

Arguments:

    Graph - Supplies a pointer to a graph instance.

Return Value:

    S_OK - Success.

    E_POINTER - Graph was NULL.

    E_OUTOFMEMORY - Out of memory.

    Non-exhaustive list of additional errors that may be returned:

    PH_E_GRAPH_NO_INFO_SET - No graph information was set.

    PH_E_NO_MORE_SEEDS - No more seed data is available.

--*/
{
    PGRAPH NewGraph;
    HRESULT Result = S_OK;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Graph)) {
        return E_POINTER;
    }

    //
    // Acquire the exclusive graph lock for the duration of the routine.  The
    // graph should never be locked at this point; if it is, consider it a
    // fatal error.
    //

    if (!TryAcquireGraphLockExclusive(Graph)) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(GraphEnterSolvingLoop_GraphLocked, Result);
        PH_RAISE(Result);
    }

    //
    // Load the graph info.
    //

    Result = Graph->Vtbl->LoadInfo(Graph);

    if (FAILED(Result)) {

        if (Result != E_OUTOFMEMORY) {

            //
            // Anything other than an out-of-memory indication from LoadInfo()
            // indicates an internal error somewhere; log the error, then raise.
            //

            PH_ERROR(GraphLoadInfo, Result);
            PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);

        }

        //
        // We failed to allocate sufficient memory for the graph.  Check for the
        // edge case where *all* threads failed to allocate memory, and set the
        // context state flag and FailedEvent accordingly.
        //

        if (InterlockedDecrement(&Graph->Context->GraphMemoryFailures) == 0) {
            Graph->Context->State.AllGraphsFailedMemoryAllocation = TRUE;
            SetStopSolving(Graph->Context);
            if (!SetEvent(Graph->Context->FailedEvent)) {
                SYS_ERROR(SetEvent);
                Result = PH_E_SYSTEM_CALL_FAILED;
            }
        }

        goto End;
    }

    //
    // Begin the solving loop.
    //

    while (Graph->Vtbl->ShouldWeContinueTryingToSolve(Graph)) {

        Result = Graph->Vtbl->LoadNewSeeds(Graph);
        if (FAILED(Result)) {

            //
            // N.B. This will need to be adjusted when we support the notion
            //      of no more seed data (PH_E_NO_MORE_SEEDS).
            //

            PH_ERROR(GraphLoadNewSeeds, Result);
            break;
        }

        Result = Graph->Vtbl->Reset(Graph);
        if (FAILED(Result)) {
            PH_ERROR(GraphReset, Result);
            break;
        } else if (Result != PH_S_CONTINUE_GRAPH_SOLVING) {
            break;
        }

        NewGraph = NULL;
        Result = Graph->Vtbl->Solve(Graph, &NewGraph);
        if (FAILED(Result)) {
            PH_ERROR(GraphSolve, Result);
            break;
        }

        if (Result == PH_S_STOP_GRAPH_SOLVING ||
            Result == PH_S_GRAPH_SOLVING_STOPPED) {
            if (NewGraph != NULL) {
                PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
            }
            break;
        }

        if (Result == PH_S_USE_NEW_GRAPH_FOR_SOLVING) {

            if (NewGraph == NULL) {
                PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
            }

            //
            // Acquire the new graph's lock and release the existing
            // graph's lock.
            //

            AcquireGraphLockExclusive(NewGraph);
            ReleaseGraphLockExclusive(Graph);

            if (!IsGraphInfoLoaded(NewGraph) ||
                NewGraph->LastLoadedNumberOfVertices <
                Graph->NumberOfVertices) {

                Result = NewGraph->Vtbl->LoadInfo(NewGraph);
                if (FAILED(Result)) {
                    PH_ERROR(GraphLoadInfo_NewGraph, Result);
                    goto End;
                }
            }

            Graph = NewGraph;
            continue;
        }

        //
        // Invariant check: result should always be PH_S_CONTINUE_GRAPH_SOLVING
        // at this point.
        //

        ASSERT(Result == PH_S_CONTINUE_GRAPH_SOLVING);

        //
        // Continue the loop and attempt another solve.
        //

    }

    //
    // Intentional follow-on to End.
    //

End:

    if (SUCCEEDED(Result)) {

        //
        // Normalize the success error codes (e.g. PH_S_STOP_GRAPH_SOLVING)
        // into a single S_OK return value.
        //

        Result = S_OK;
    }

    ReleaseGraphLockExclusive(Graph);

    return Result;
}


GRAPH_LOAD_INFO GraphLoadInfo;

_Use_decl_annotations_
HRESULT
GraphLoadInfo(
    PGRAPH Graph
    )
/*++

Routine Description:

    This routine is called by graph solving worker threads prior to attempting
    any solving; it is responsible for initializing the graph structure and
    allocating (or reallocating) the necessary buffers required for graph
    solving, using the sizes indicated by the info structure previously set
    by the main thread via SetInfo().

Arguments:

    Graph - Supplies a pointer to the graph instance.

Return Value:

    S_OK - Success.

    E_POINTER - Graph was NULL.

    E_OUTOFMEMORY - Out of memory.

    PH_E_GRAPH_NO_INFO_SET - No graph information has been set for this graph.

    PH_E_GRAPH_INFO_ALREADY_LOADED - Graph information has already been loaded
        for this graph.

--*/
{
    PRTL Rtl;
    HRESULT Result;
    PGRAPH_INFO Info;
    PGRAPH_INFO PrevInfo;
    PALLOCATOR Allocator;
    ULONG ProtectionFlags;
    PPERFECT_HASH_TABLE Table;
    SIZE_T VertexPairsSizeInBytes;
    PPERFECT_HASH_CONTEXT Context;
    BOOLEAN LargePagesForVertexPairs;
    PASSIGNED_MEMORY_COVERAGE Coverage;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;
    PRTL_TRY_LARGE_PAGE_VIRTUAL_ALLOC Alloc;
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Graph)) {
        return E_POINTER;
    }

    if (!IsGraphInfoSet(Graph)) {
        return PH_E_GRAPH_NO_INFO_SET;
    } else if (IsGraphInfoLoaded(Graph)) {
        return PH_E_GRAPH_INFO_ALREADY_LOADED;
    } else {
        Info = Graph->Info;
    }

    //
    // Sanity check the graph size is correct.
    //

    ASSERT(sizeof(*Graph) == Info->SizeOfGraphStruct);

    //
    // Initialize aliases.
    //

    Context = Info->Context;
    Rtl = Context->Rtl;
    PrevInfo = Info->PrevInfo;
    Allocator = Graph->Allocator;
    Table = Context->Table;
    TableInfoOnDisk = Table->TableInfoOnDisk;
    TableCreateFlags.AsULong = Table->TableCreateFlags.AsULong;

    //
    // Set the relevant graph fields based on the provided info.
    //

    Graph->Context = Context;
    Graph->NumberOfSeeds = Table->TableInfoOnDisk->NumberOfSeeds;
    Graph->NumberOfKeys = Table->Keys->NumberOfElements.LowPart;

    Graph->ThreadId = GetCurrentThreadId();
    Graph->ThreadAttempt = 0;

    Graph->EdgeMask = Table->IndexMask;
    Graph->VertexMask = Table->HashMask;
    Graph->EdgeModulus = Table->IndexModulus;
    Graph->VertexModulus = Table->HashModulus;
    Graph->MaskFunctionId = Info->Context->MaskFunctionId;

    Graph->Flags.Paranoid = IsParanoid(Table);

    CopyInline(&Graph->Dimensions,
               &Info->Dimensions,
               sizeof(Graph->Dimensions));

    Result = S_OK;

    //
    // Allocate (or reallocate) arrays.
    //

#define ALLOC_ARRAY(Name, Type)                       \
    if (!Graph->##Name) {                             \
        Graph->##Name = (Type)(                       \
            Allocator->Vtbl->AlignedMalloc(           \
                Allocator,                            \
                (ULONG_PTR)Info->##Name##SizeInBytes, \
                YMMWORD_ALIGNMENT                     \
            )                                         \
        );                                            \
    } else {                                          \
        Graph->##Name## = (Type)(                     \
            Allocator->Vtbl->AlignedReAlloc(          \
                Allocator,                            \
                Graph->##Name,                        \
                (ULONG_PTR)Info->##Name##SizeInBytes, \
                YMMWORD_ALIGNMENT                     \
            )                                         \
        );                                            \
    }                                                 \
    if (!Graph->##Name) {                             \
        Result = E_OUTOFMEMORY;                       \
        goto Error;                                   \
    }

    ALLOC_ARRAY(Next, PEDGE);
    ALLOC_ARRAY(Edges, PEDGE);
    ALLOC_ARRAY(First, PVERTEX);
    ALLOC_ARRAY(Order, PULONG);
    ALLOC_ARRAY(Assigned, PASSIGNED);

    //
    // If we're hashing all keys first, prepare the vertex pairs array if it
    // hasn't already been prepared.  (This array is sized off the number of
    // keys, which never changes upon subsequent table resize events, so it
    // never needs to be reallocated to a larger size (unlike the other arrays
    // above, which grow larger upon each resize event).)
    //

    if (TableCreateFlags.HashAllKeysFirst) {

        ASSERT(Info->VertexPairsSizeInBytes != 0);
        VertexPairsSizeInBytes = (SIZE_T)Info->VertexPairsSizeInBytes;

        if (Graph->VertexPairs == NULL) {

            LargePagesForVertexPairs = (BOOLEAN)(
                TableCreateFlags.TryLargePagesForVertexPairs != FALSE
            );

            ProtectionFlags = PAGE_READWRITE;

            if (Graph->Flags.WantsWriteCombiningForVertexPairsArray) {

                //
                // Large pages and write-combine are incompatible.  (This will
                // have been weeded out by IsValidTableCreateFlags(), so we can
                // just ASSERT() instead here.)
                //

                ASSERT(!LargePagesForVertexPairs);

                ProtectionFlags |= PAGE_WRITECOMBINE;
            }

            //
            // Proceed with allocation of the vertex pairs array.
            //

            Alloc = Rtl->Vtbl->TryLargePageVirtualAlloc;
            Graph->VertexPairs = Alloc(Rtl,
                                       NULL,
                                       VertexPairsSizeInBytes,
                                       MEM_RESERVE | MEM_COMMIT,
                                       ProtectionFlags,
                                       &LargePagesForVertexPairs);

            if (Graph->VertexPairs == NULL) {
                Result = E_OUTOFMEMORY;
                goto Error;
            }

            //
            // Update the graph flags indicating whether or not large pages
            // were used, and if write-combining is active.
            //

            Graph->Flags.VertexPairsArrayUsesLargePages =
                LargePagesForVertexPairs;

            Graph->Flags.VertexPairsArrayIsWriteCombined =
                Graph->Flags.WantsWriteCombiningForVertexPairsArray;

        }
    }

    //
    // Set the bitmap sizes and then allocate (or reallocate) the bitmap
    // buffers.
    //

    Graph->DeletedEdgesBitmap.SizeOfBitMap = Graph->TotalNumberOfEdges;
    Graph->VisitedVerticesBitmap.SizeOfBitMap = Graph->NumberOfVertices;
    Graph->AssignedBitmap.SizeOfBitMap = Graph->NumberOfVertices;
    Graph->IndexBitmap.SizeOfBitMap = Graph->NumberOfVertices;

#define ALLOC_BITMAP_BUFFER(Name)                          \
    if (!Graph->##Name##.Buffer) {                         \
        Graph->##Name##.Buffer = (PULONG)(                 \
            Allocator->Vtbl->Malloc(                       \
                Allocator,                                 \
                (ULONG_PTR)Info->##Name##BufferSizeInBytes \
            )                                              \
        );                                                 \
    } else {                                               \
        Graph->##Name##.Buffer = (PULONG)(                 \
            Allocator->Vtbl->ReAlloc(                      \
                Allocator,                                 \
                Graph->##Name##.Buffer,                    \
                (ULONG_PTR)Info->##Name##BufferSizeInBytes \
            )                                              \
        );                                                 \
    }                                                      \
    if (!Graph->##Name##.Buffer) {                         \
        Result = E_OUTOFMEMORY;                            \
        goto Error;                                        \
    }

    ALLOC_BITMAP_BUFFER(DeletedEdgesBitmap);
    ALLOC_BITMAP_BUFFER(VisitedVerticesBitmap);
    ALLOC_BITMAP_BUFFER(AssignedBitmap);
    ALLOC_BITMAP_BUFFER(IndexBitmap);

    //
    // Check to see if we're in "first graph wins" mode, and have also been
    // asked to skip memory coverage information.  If so, we can jump straight
    // to the end and finish up.
    //

    if (FirstSolvedGraphWinsAndSkipMemoryCoverage(Context)) {
        Graph->Flags.WantsAssignedMemoryCoverage = FALSE;
        goto End;
    }

    if (FirstSolvedGraphWins(Context)) {

        Graph->Flags.WantsAssignedMemoryCoverage = TRUE;

    } else {

        if (DoesBestCoverageTypeRequireKeysSubset(Context->BestCoverageType)) {
            Graph->Flags.WantsAssignedMemoryCoverageForKeysSubset = TRUE;
        } else {
            Graph->Flags.WantsAssignedMemoryCoverage = TRUE;
        }

    }

    //
    // Fill out the assigned memory coverage structure and allocate buffers.
    //

    Coverage = &Graph->AssignedMemoryCoverage;

    Coverage->TotalNumberOfPages = Info->AssignedArrayNumberOfPages;
    Coverage->TotalNumberOfLargePages = Info->AssignedArrayNumberOfLargePages;
    Coverage->TotalNumberOfCacheLines = Info->AssignedArrayNumberOfCacheLines;

#define ALLOC_ASSIGNED_ARRAY(Name, Type)               \
    if (!Coverage->##Name) {                           \
        Coverage->##Name = (PASSIGNED_##Type##_COUNT)( \
            Allocator->Vtbl->AlignedMalloc(            \
                Allocator,                             \
                (ULONG_PTR)Info->##Name##SizeInBytes,  \
                YMMWORD_ALIGNMENT                      \
            )                                          \
        );                                             \
    } else {                                           \
        Coverage->##Name = (PASSIGNED_##Type##_COUNT)( \
            Allocator->Vtbl->AlignedReAlloc(           \
                Allocator,                             \
                Coverage->##Name,                      \
                (ULONG_PTR)Info->##Name##SizeInBytes,  \
                YMMWORD_ALIGNMENT                      \
            )                                          \
        );                                             \
    }                                                  \
    if (!Coverage->##Name) {                           \
        Result = E_OUTOFMEMORY;                        \
        goto Error;                                    \
    }

    ALLOC_ASSIGNED_ARRAY(NumberOfAssignedPerPage, PAGE);
    ALLOC_ASSIGNED_ARRAY(NumberOfAssignedPerCacheLine, CACHE_LINE);

    //
    // The number of large pages consumed may not change between resize events;
    // avoid a realloc if unnecessary by checking the previous info's number of
    // large pages if applicable.
    //

#define ALLOC_ASSIGNED_LARGE_PAGE_ARRAY(Name)                                 \
    if (!Coverage->##Name) {                                                  \
        Coverage->##Name = (PASSIGNED_LARGE_PAGE_COUNT)(                      \
            Allocator->Vtbl->AlignedMalloc(                                   \
                Allocator,                                                    \
                (ULONG_PTR)Info->##Name##SizeInBytes,                         \
                YMMWORD_ALIGNMENT                                             \
            )                                                                 \
        );                                                                    \
    } else {                                                                  \
        BOOLEAN DoReAlloc = TRUE;                                             \
        if (PrevInfo) {                                                       \
            if (PrevInfo->##Name##SizeInBytes == Info->##Name##SizeInBytes) { \
                DoReAlloc = FALSE;                                            \
            }                                                                 \
        }                                                                     \
        if (DoReAlloc) {                                                      \
            Coverage->##Name = (PASSIGNED_LARGE_PAGE_COUNT)(                  \
                Allocator->Vtbl->AlignedReAlloc(                              \
                    Allocator,                                                \
                    Coverage->##Name,                                         \
                    (ULONG_PTR)Info->##Name##SizeInBytes,                     \
                    YMMWORD_ALIGNMENT                                         \
                )                                                             \
            );                                                                \
        }                                                                     \
    }                                                                         \
    if (!Coverage->##Name) {                                                  \
        Result = E_OUTOFMEMORY;                                               \
        goto Error;                                                           \
    }

    ALLOC_ASSIGNED_LARGE_PAGE_ARRAY(NumberOfAssignedPerLargePage);

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    if (SUCCEEDED(Result)) {
        Graph->Flags.IsInfoLoaded = TRUE;
        Graph->LastLoadedNumberOfVertices = Graph->NumberOfVertices;
    }

    return Result;
}


GRAPH_RESET GraphReset;

_Use_decl_annotations_
HRESULT
GraphReset(
    PGRAPH Graph
    )
/*++

Routine Description:

    Resets the state of a graph instance after a solving attempt, such that it
    can be used for a subsequent attempt.

Arguments:

    Graph - Supplies a pointer to the graph instance to reset.

Return Value:

    PH_S_CONTINUE_GRAPH_SOLVING - Graph was successfully reset and graph solving
        should continue.

    PH_S_GRAPH_SOLVING_STOPPED - Graph solving has been stopped.  The graph is
        not reset and solving should not continue.

    PH_S_TABLE_RESIZE_IMMINENT - The reset was not performed as a table resize
        is imminent (and thus, attempts at solving this current graph can be
        stopped).

    PH_E_SYSTEM_CALL_FAILED - A system call failed.

--*/
{
    PRTL Rtl;
    BOOL Success;
    PGRAPH_INFO Info;
    HRESULT Result = PH_S_CONTINUE_GRAPH_SOLVING;
    ULONG OldProtection;
    ULONG ProtectionFlags;
    ULONG TotalNumberOfPages;
    ULONG TotalNumberOfLargePages;
    ULONG TotalNumberOfCacheLines;
    PPERFECT_HASH_CONTEXT Context;
    SIZE_T VertexPairsSizeInBytes;
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags;
    PASSIGNED_MEMORY_COVERAGE Coverage;
    PASSIGNED_PAGE_COUNT NumberOfAssignedPerPage;
    PASSIGNED_LARGE_PAGE_COUNT NumberOfAssignedPerLargePage;
    PASSIGNED_CACHE_LINE_COUNT NumberOfAssignedPerCacheLine;

    //
    // Initialize aliases.
    //

    Context = Graph->Context;
    Info = Graph->Info;
    Rtl = Context->Rtl;
    TableCreateFlags.AsULong = Context->Table->TableCreateFlags.AsULong;

    MAYBE_STOP_GRAPH_SOLVING(Graph);

    //
    // Increment the thread attempt counter, and interlocked-increment the
    // global context counter.  If the global attempt is equal to the resize
    // table threshold, signal the event to try a larger table size and return
    // with the error code indicating a table resize is imminent.
    //

    ++Graph->ThreadAttempt;

    Graph->Attempt = InterlockedIncrement64(&Context->Attempts);

    if (!Context->FinishedCount &&
        Graph->Attempt - 1 == Context->ResizeTableThreshold) {

        if (!SetEvent(Context->TryLargerTableSizeEvent)) {
            SYS_ERROR(SetEvent);
            Result = PH_E_SYSTEM_CALL_FAILED;
            goto Error;
        }
        SetStopSolving(Context);
        return PH_S_TABLE_RESIZE_IMMINENT;
    }

    //
    // Clear the bitmap buffers.
    //

#define ZERO_BITMAP_BUFFER(Name)                           \
    ASSERT(0 == Info->##Name##BufferSizeInBytes -          \
           ((Info->##Name##BufferSizeInBytes >> 3) << 3)); \
    Rtl->RtlZeroMemory((PDWORD64)Graph->##Name##.Buffer,   \
                       Info->##Name##BufferSizeInBytes)


    ZERO_BITMAP_BUFFER(DeletedEdgesBitmap);
    ZERO_BITMAP_BUFFER(VisitedVerticesBitmap);
    ZERO_BITMAP_BUFFER(AssignedBitmap);
    ZERO_BITMAP_BUFFER(IndexBitmap);

    //
    // "Empty" all of the nodes.
    //

#define EMPTY_ARRAY(Name)                            \
    ASSERT(0 == Info->##Name##SizeInBytes -          \
           ((Info->##Name##SizeInBytes >> 3) << 3)); \
    Rtl->RtlFillMemory((PDWORD64)Graph->##Name,      \
                       Info->##Name##SizeInBytes,    \
                       (BYTE)~0)

    EMPTY_ARRAY(Next);
    EMPTY_ARRAY(First);
    EMPTY_ARRAY(Edges);

    //
    // The Order and Assigned arrays get zeroed.
    //

#define ZERO_ARRAY(Name)                             \
    ASSERT(0 == Info->##Name##SizeInBytes -          \
           ((Info->##Name##SizeInBytes >> 3) << 3)); \
    Rtl->RtlZeroMemory((PDWORD64)Graph->##Name##,    \
                       Info->##Name##SizeInBytes)

    ZERO_ARRAY(Order);
    ZERO_ARRAY(Assigned);

    Graph->OrderIndex = Graph->NumberOfKeys;

    if (TableCreateFlags.HashAllKeysFirst) {

        ASSERT(Graph->VertexPairs != NULL);

        //
        // If this is not the first time Reset() has been called for this graph
        // instance, the vertex pairs array's page protection may be set to
        // PAGE_READONLY.  This will occur if the previous graph attempt was
        // able to hash all keys without collision, but detected a cyclic graph,
        // and thus, wasn't a successful solve.  Or it was a successful solve,
        // became the best graph for a while, but then was beaten by another,
        // better graph solving attempt, and thus, was thrown back into the
        // solving mix (when in find best graph mode).
        //
        // We can detect this situation by determining if the write-combining
        // behavior is requested but the array is not currently indicating as
        // write-combined.
        //
        // N.B. We don't need to clear the individual vertex pair array elements
        //      like we do with the first/next/edge arrays as they have no state
        //      associated with the notion of being visited or not.  (Whereas we
        //      need to set the first/next/edge arrays to -1 before solving.)
        //

        if (Graph->Flags.WantsWriteCombiningForVertexPairsArray &&
            !Graph->Flags.VertexPairsArrayIsWriteCombined) {

            //
            // Restore the write-combine (and read/write) page protection so
            // that the vertex pairs can be subsequently written to without
            // trapping.
            //

            ASSERT(Info->VertexPairsSizeInBytes != 0);
            VertexPairsSizeInBytes = (SIZE_T)Info->VertexPairsSizeInBytes;
            ProtectionFlags = PAGE_READWRITE | PAGE_WRITECOMBINE;

            Success = VirtualProtect(Graph->VertexPairs,
                                     VertexPairsSizeInBytes,
                                     ProtectionFlags,
                                     &OldProtection);

            if (Success) {
                Graph->Flags.VertexPairsArrayIsWriteCombined = TRUE;
            } else {
                SYS_ERROR(VirtualProtect);
                Result = PH_E_SYSTEM_CALL_FAILED;
                goto End;
            }
        }
    }

    //
    // Clear any remaining values.
    //

    Graph->Collisions = 0;
    Graph->NumberOfEmptyVertices = 0;
    Graph->DeletedEdgeCount = 0;
    Graph->VisitedVerticesCount = 0;

    Graph->TraversalDepth = 0;
    Graph->TotalTraversals = 0;
    Graph->MaximumTraversalDepth = 0;

    Graph->Flags.Shrinking = FALSE;
    Graph->Flags.IsAcyclic = FALSE;

    RESET_GRAPH_COUNTER(AddKeys);
    RESET_GRAPH_COUNTER(HashKeys);
    RESET_GRAPH_COUNTER(AddHashedKeys);
    RESET_GRAPH_COUNTER(Assign);

    //
    // Avoid the overhead of resetting the memory coverage if we're in "first
    // graph wins" mode and have been requested to skip memory coverage.
    //

    if (FirstSolvedGraphWinsAndSkipMemoryCoverage(Context)) {
        goto End;
    }

    //
    // Clear the assigned memory coverage counts and arrays.
    //

    Coverage = &Graph->AssignedMemoryCoverage;

    //
    // Capture the totals and pointers prior to zeroing the struct.
    //

    TotalNumberOfPages = Coverage->TotalNumberOfPages;
    TotalNumberOfLargePages = Coverage->TotalNumberOfLargePages;
    TotalNumberOfCacheLines = Coverage->TotalNumberOfCacheLines;

    NumberOfAssignedPerPage = Coverage->NumberOfAssignedPerPage;
    NumberOfAssignedPerLargePage = Coverage->NumberOfAssignedPerLargePage;
    NumberOfAssignedPerCacheLine = Coverage->NumberOfAssignedPerCacheLine;

    ZeroStructPointer(Coverage);

    //
    // Restore the totals and pointers.
    //

    Coverage->TotalNumberOfPages = TotalNumberOfPages;
    Coverage->TotalNumberOfLargePages = TotalNumberOfLargePages;
    Coverage->TotalNumberOfCacheLines = TotalNumberOfCacheLines;

    Coverage->NumberOfAssignedPerPage = NumberOfAssignedPerPage;
    Coverage->NumberOfAssignedPerLargePage = NumberOfAssignedPerLargePage;
    Coverage->NumberOfAssignedPerCacheLine = NumberOfAssignedPerCacheLine;

#define ZERO_ASSIGNED_ARRAY(Name) \
    ZeroMemory(Coverage->##Name, Info->##Name##SizeInBytes)

    ZERO_ASSIGNED_ARRAY(NumberOfAssignedPerPage);
    ZERO_ASSIGNED_ARRAY(NumberOfAssignedPerLargePage);
    ZERO_ASSIGNED_ARRAY(NumberOfAssignedPerCacheLine);

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (SUCCEEDED(Result)) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}


GRAPH_LOAD_NEW_SEEDS GraphLoadNewSeeds;

_Use_decl_annotations_
HRESULT
GraphLoadNewSeeds(
    PGRAPH Graph
    )
/*++

Routine Description:

    Loads new seed data for a graph instance.  This is called prior to each
    solving attempt.

Arguments:

    Graph - Supplies a pointer to the graph instance for which the new seed
        data will be loaded.

Return Value:

    S_OK - Success.

    E_POINTER - Graph was NULL.

    PH_E_NO_MORE_SEEDS - No more seed data is available.  (Not currently
        returned for this implementation.)

    PH_E_SPARE_GRAPH - Graph is indicated as the spare graph.

--*/
{
    PRTL Rtl;
    HRESULT Result;
    ULONG SizeInBytes;
    PPERFECT_HASH_CONTEXT Context;
    PPERFECT_HASH_TABLE_CREATE_PARAMETERS Params;

    if (!ARGUMENT_PRESENT(Graph)) {
        return E_POINTER;
    }

    if (IsSpareGraph(Graph)) {
        return PH_E_SPARE_GRAPH;
    }

    SizeInBytes = Graph->NumberOfSeeds * sizeof(Graph->FirstSeed);

    Rtl = Graph->Rtl;

    Result = Rtl->Vtbl->GenerateRandomBytes(Rtl,
                                            SizeInBytes,
                                            (PBYTE)&Graph->FirstSeed);

    if (FAILED(Result)) {
        return Result;
    }

    Context = Graph->Context;
    Params = Context->Table->TableCreateParameters;

    //
    // Determine if we have seed masks counts first.
    //

    if (Params->Flags.HasSeedMaskCounts != FALSE) {

        Result = GraphApplyWeightedSeedMasks(Graph,
                                             Context->Seed3Byte1MaskCounts);
        if (FAILED(Result)) {
            PH_ERROR(GraphApplyWeightedSeedMasks_Seed3Byte1, Result);
            goto End;
        }

        Result = GraphApplyWeightedSeedMasks(Graph,
                                             Context->Seed3Byte2MaskCounts);
        if (FAILED(Result)) {
            PH_ERROR(GraphApplyWeightedSeedMasks_Seed3Byte2, Result);
            goto End;
        }

        //
        // Normalize the return code back to S_OK.
        //

        ASSERT(Result == S_OK || Result == S_FALSE);
        Result = S_OK;

    } else {

        //
        // Apply user seeds and seed masks if applicable, then return.
        //

        if (Context->UserSeeds) {
            Result = GraphApplyUserSeeds(Graph);
            if (FAILED(Result)) {
                PH_ERROR(GraphApplyUserSeeds, Result);
                goto End;
            }
        }

        if (Context->SeedMasks) {
            Result = GraphApplySeedMasks(Graph);
            if (FAILED(Result)) {
                PH_ERROR(GraphApplySeedMasks, Result);
                goto End;
            }
        }

    }

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}


GRAPH_APPLY_USER_SEEDS GraphApplyUserSeeds;

_Use_decl_annotations_
HRESULT
GraphApplyUserSeeds(
    PGRAPH Graph
    )
/*++

Routine Description:

    Applies user-provided seeds to the graph, if applicable.

Arguments:

    Graph - Supplies a pointer to the graph instance to which the user seed
        data will be applied, if applicable.

Return Value:

    S_OK - User seeds were successfully applied.

    S_FALSE - No user seeds present.

    E_POINTER - Graph was NULL.

    PH_E_SPARE_GRAPH - Graph is indicated as the spare graph (and it's not a
        CUDA graph).

    PH_E_INVALID_USER_SEEDS_ELEMENT_SIZE - The individual value size indicated
        by the user seed value array is invalid (i.e. not sizeof(ULONG)).

--*/
{
    HRESULT Result = S_OK;
    ULONG Index;
    PULONG Value;
    PULONG Seed;
    PULONG Seeds;
    PVALUE_ARRAY ValueArray;
    PPERFECT_HASH_CONTEXT Context;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Graph)) {
        return E_POINTER;
    }

    if (IsSpareGraph(Graph) && !IsCuGraph(Graph)) {
        return PH_E_SPARE_GRAPH;
    }

    Context = Graph->Context;
    ValueArray = Context->UserSeeds;

    if (!ValueArray) {

        //
        // No user seeds were provided (i.e. no --Seeds command line param).
        //

        return S_FALSE;
    }

    //
    // Ensure the value array size matches our graph seed size.
    //

    if (ValueArray->ValueSizeInBytes != sizeof(Graph->FirstSeed)) {
        return PH_E_INVALID_USER_SEEDS_ELEMENT_SIZE;
    }

    //
    // Validation complete.  The caller has provided valid seed data; loop
    // through it and apply any non-zero members at the relative offset.
    //

    Seeds = &Graph->FirstSeed;

    for (Index = 0, Value = ValueArray->Values;
         Index < Graph->NumberOfSeeds && Index < ValueArray->NumberOfValues;
         Index++, Value++) {

        if (*Value != 0) {

            //
            // Non-zero seed value detected; overwrite the applicable seed
            // slot with the caller-provided value.
            //

            Seed = Seeds + Index;
            *Seed = *Value;
        }
    }

    return Result;
}


GRAPH_APPLY_SEED_MASKS GraphApplySeedMasks;

_Use_decl_annotations_
HRESULT
GraphApplySeedMasks(
    PGRAPH Graph
    )
/*++

Routine Description:

    Applies masks to seeds, if applicable.

Arguments:

    Graph - Supplies a pointer to the graph instance to which the seed masks
        will be applied, if applicable.

Return Value:

    S_OK - User seeds were successfully applied.

    S_FALSE - No seed masks present.

    E_POINTER - Graph was NULL.

    PH_E_SPARE_GRAPH - Graph is indicated as the spare graph.

--*/
{
    ULONG Index;
    LONG Mask;
    ULONG NewSeed;
    PULONG Seed;
    PULONG Seeds;
    const LONG *Masks;
    PCSEED_MASKS SeedMasks;
    PPERFECT_HASH_CONTEXT Context;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Graph)) {
        return E_POINTER;
    }

    if (IsSpareGraph(Graph)) {
        return PH_E_SPARE_GRAPH;
    }

    Context = Graph->Context;
    SeedMasks = Context->SeedMasks;

    if (!SeedMasks) {

        //
        // No seed masks are available for this hash routine.
        //

        return S_FALSE;
    }

    //
    // Validation complete.  Loop through the masks and apply those with a value
    // greater than zero to the seed at the corresponding offset.
    //

    Seeds = &Graph->FirstSeed;
    Masks = &Context->SeedMasks->Mask1;

    for (Index = 0; Index < Graph->NumberOfSeeds; Index++) {

        Mask = *Masks++;

        if (Mask != -1 && Mask != 0) {

            //
            // Valid mask found, apply it to the seed data at this slot.
            //

            Seed = Seeds + Index;
            NewSeed = *Seed & Mask;
            *Seed = NewSeed;
        }
    }

    return S_OK;
}

GRAPH_APPLY_WEIGHTED_SEED_MASKS GraphApplyWeightedSeedMasks;

_Use_decl_annotations_
HRESULT
GraphApplyWeightedSeedMasks(
    PGRAPH Graph,
    PCSEED_MASK_COUNTS SeedMaskCounts
    )
/*++

Routine Description:

    Generates seed data based on weighted mask counts.  The SeedMaskCounts
    parameter indicates the target seed number and byte number.  This routine
    generates a random float between the interval [0.0, 1.0), then bisects the
    array of cumulative weight counts, identifying an insertion point to the
    right.  The insertion point value is written to the target seed's byte.

Arguments:

    Graph - Supplies a pointer to the graph instance.

    SeedMaskCounts - Optionally supplies a pointer to a seed mask count struct.

Return Value:

    S_OK - Weighted seed mask successfully applied.

    S_FALSE - No seed mask counts present.

    E_POINTER - Graph was NULL.

    PH_E_SPARE_GRAPH - Graph is indicated as the spare graph.

--*/
{
    PRTL Rtl;
    PULONG Seed;
    PBYTE Byte;
    HRESULT Result;
    BYTE Low;
    BYTE High;
    BYTE Middle;
    ULONG Target;
    ULONG Cumulative;
    DOUBLE Random;
    DOUBLE TargetDouble;
    ULARGE_INTEGER Large;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Graph)) {
        return E_POINTER;
    }

    if (IsSpareGraph(Graph)) {
        return PH_E_SPARE_GRAPH;
    }

    if (!SeedMaskCounts) {

        //
        // No seed mask counts were provided.
        //

        return S_FALSE;
    }

    //
    // Validation complete.  Generate two 32-bit random ULONGs, then normalize
    // into a [0,1) interval.
    //

    Rtl = Graph->Rtl;
    Large.QuadPart = 0;
    Result = Rtl->Vtbl->GenerateRandomBytes(Rtl,
                                            sizeof(Large),
                                            (PBYTE)&Large);

    if (FAILED(Result)) {
        return Result;
    }

    //
    // Derived from:
    // http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/CODES/mt19937ar.c
    //

    Large.HighPart >>= 5;
    Large.LowPart >>= 6;

    Random = (
        ((DOUBLE)Large.HighPart * (DOUBLE)67108864.0 + (DOUBLE)Large.LowPart) *
        ((DOUBLE)1.0/(DOUBLE)9007199254740992.0)
    );
    TargetDouble = Random * (DOUBLE)SeedMaskCounts->Total;
    Target = (ULONG)TargetDouble;

    //
    // Bisect the cumulative counts to find an appropriate insertion point to
    // the right.
    //

    Low = 0;
    High = 32;

    while (Low < High) {

        Middle = (Low + High) / 2;

        Cumulative = SeedMaskCounts->Cumulative[Middle];

        if (Target < Cumulative) {

            //
            // Our random value is less than the current middle point of the
            // cumulative counts.  Adjust the high marker to be the current
            // middle.
            //

            High = Middle;

        } else {

            //
            // Our random value is greater than or equal to the current middle
            // point of the cumulative counts.  Adjust the low marker to middle
            // plus 1.
            //

            Low = Middle + 1;
        }
    }

    //
    // Low now represents the 0-based insertion point into the cumulative
    // counts array.  This can be used directly as the seed's byte value.
    // The seed number and byte number fields in the seed mask counts struct
    // are all 1-based, which is why we subtract 1 from each below (for the
    // 0-based array offsets).
    //

    Seed = &Graph->FirstSeed;
    Seed += (SeedMaskCounts->SeedNumber - 1);
    Byte = (PBYTE)Seed;
    Byte += (SeedMaskCounts->ByteNumber - 1);

    //
    // Write the byte value and return success.
    //

    *Byte = Low;

    return S_OK;
}

GRAPH_SHOULD_WE_CONTINUE_TRYING_TO_SOLVE GraphShouldWeContinueTryingToSolve;

_Use_decl_annotations_
BOOLEAN
GraphShouldWeContinueTryingToSolve(
    PGRAPH Graph
    )
/*++

Routine Description:

    Determines if graph solving should continue.  This routine is intended to
    be called periodically by the graph solving loop, particularly before and
    after large pieces of work are completed (such as graph assignment).  If
    this routine returns FALSE, the caller should stop what they're doing and
    return the PH_S_STOP_GRAPH_SOLVING return code.

Arguments:

    Graph - Supplies a pointer to a graph instance.

Return Value:

    TRUE if solving should continue, FALSE otherwise.

--*/
{
    PPERFECT_HASH_CONTEXT Context;

    Context = Graph->Context;

    //
    // If ctrl-C has been pressed, set stop solving.  Continue solving unless
    // stop solving indicates otherwise.
    //

    if (CtrlCPressed) {
        SetStopSolving(Context);
    }

    return (StopSolving(Context) != FALSE ? FALSE : TRUE);
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
