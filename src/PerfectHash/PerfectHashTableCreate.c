/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTableCreate.c

Abstract:

    This module implements the Create() routine for the perfect hash table
    component.

--*/

#include "stdafx.h"

//
// Cap the maximum key set size we're willing to process to 10 million.
//

#define MAXIMUM_NUMBER_OF_KEYS 10000000

//
// Define the threshold for how many attempts need to be made at finding a
// perfect hash solution before we double our number of vertices and try again.
//
// With a 2-part hypergraph, solutions are found on average in sqrt(3) attempts.
// By attempt 18, there's a 99.9% chance we will have found a solution.
//

#define GRAPH_SOLVING_ATTEMPTS_THRESHOLD 18

//
// Define a limit for how many times the table resizing will be attempted before
// giving up.  For large table sizes and large concurrency values, note that we
// may hit memory limits before we hit this resize limit.
//

#define GRAPH_SOLVING_RESIZE_TABLE_LIMIT 5

//
// Forward decls.
//

typedef
_Check_return_
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PERFECT_HASH_TABLE_VALIDATE_CREATE_PARAMETERS)(
    _In_ PPERFECT_HASH_TABLE Table
    );
typedef PERFECT_HASH_TABLE_VALIDATE_CREATE_PARAMETERS
      *PPERFECT_HASH_TABLE_VALIDATE_CREATE_PARAMETERS;

PERFECT_HASH_TABLE_VALIDATE_CREATE_PARAMETERS
    PerfectHashTableValidateCreateParameters;

//
// Begin method implementations.
//

PERFECT_HASH_TABLE_CREATE PerfectHashTableCreate;

_Use_decl_annotations_
HRESULT
PerfectHashTableCreate(
    PPERFECT_HASH_TABLE Table,
    PPERFECT_HASH_CONTEXT Context,
    PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    PPERFECT_HASH_KEYS Keys,
    PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlagsPointer,
    ULONG NumberOfTableCreateParameters,
    PPERFECT_HASH_TABLE_CREATE_PARAMETER TableCreateParameters
    )
/*++

Routine Description:

    Creates a perfect hash table.

Arguments:

    Table - Supplies a perfect hash table instance for which the creation
        operation will be performed.

    Context - Supplies a pointer to an initialized PERFECT_HASH_CONTEXT
        structure that can be used by the underlying algorithm in order to
        search for perfect hash solutions in parallel.  The context must not
        be locked; this routine will acquire the context's lock exclusively
        for its duration.

    AlgorithmId - Supplies the algorithm to use.

    MaskFunctionId - Supplies the type of masking to use.  The algorithm and
        hash function must both support the requested masking type.

    HashFunctionId - Supplies the hash function to use.

    Keys - Supplies a pointer to a PERFECT_HASH_KEYS interface.

    TableCreateFlags - Optionally supplies a pointer to a context create
        table flags structure that can be used to customize table creation.

    NumberOfTableCreateParameters - Optionally supplies the number of elements
        in the TableCreateParameters array.

    TableCreateParameters - Optionally supplies an array of additional
        parameters that can be used to further customize table creation
        behavior.

Return Value:

    S_OK - Table created successfully.

    The following error codes may also be returned.  Note that this list is not
    guaranteed to be exhaustive; that is, error codes other than the ones listed
    below may also be returned.

    E_POINTER - Table, Context or Keys parameters were NULL.

    E_UNEXPECTED - Internal error.

    E_OUTOFMEMORY - Out of memory.

    PH_E_INVALID_ALGORITHM_ID - Invalid algorithm ID.

    PH_E_INVALID_HASH_FUNCTION_ID - Invalid hash function ID.

    PH_E_INVALID_MASK_FUNCTION_ID - Invalid mask function ID.

    PH_E_INVALID_TABLE_CREATE_FLAGS - Invalid table creation flags.

    PH_E_TABLE_LOCKED - The table is locked.

    PH_E_CONTEXT_LOCKED - The context is locked.

    PH_E_KEYS_FILE_NAME_NOT_VALID_C_IDENTIFIER - The file name component of
        the keys path contained characters that prevented it from being
        considered a valid C identifier.  As this name is used to generate
        various C structures, it must be valid.

    PH_E_BASE_NAME_INVALID_C_IDENTIFIER - The table base name component was
        not a valid C identifier.

    PH_E_CONTEXT_BASE_OUTPUT_DIRECTORY_NOT_SET - The base output directory
        has not been set for the context.

    PH_E_NO_INDEX_IMPL_C_STRING_FOUND - No C representation of the Index()
        implementation routine was found for the given algorithm, hash function
        and masking type.

    PH_E_NUM_TABLE_CREATE_PARMS_IS_ZERO_BUT_PARAM_POINTER_NOT_NULL - The number
        of table create params is zero but the parameters pointer is not null.

--*/
{
    PRTL Rtl;
    PALLOCATOR Allocator;
    HRESULT Result = S_OK;
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    } else {
        Rtl = Table->Rtl;
        Allocator = Table->Allocator;
    }

    if (!ARGUMENT_PRESENT(Context)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Keys)) {
        return E_POINTER;
    }

    if (Keys->NumberOfElements.QuadPart > MAXIMUM_NUMBER_OF_KEYS) {
        return PH_E_TOO_MANY_KEYS;
    }

    VALIDATE_FLAGS(TableCreate, TABLE_CREATE);

    if (!IsValidPerfectHashAlgorithmId(AlgorithmId)) {
        return PH_E_INVALID_ALGORITHM_ID;
    }

    if (!IsValidPerfectHashHashFunctionId(HashFunctionId)) {
        return PH_E_INVALID_HASH_FUNCTION_ID;
    }

    if (!IsValidPerfectHashMaskFunctionId(MaskFunctionId)) {
        return PH_E_INVALID_MASK_FUNCTION_ID;
    }

    if (!TryAcquirePerfectHashContextLockExclusive(Context)) {
        return PH_E_CONTEXT_LOCKED;
    }

    if (!TryAcquirePerfectHashTableLockExclusive(Table)) {
        ReleasePerfectHashContextLockExclusive(Context);
        return PH_E_TABLE_LOCKED;
    }

    if (Context->State.NeedsReset) {
        Result = PerfectHashContextReset(Context);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashContextReset, Result);
            goto Error;
        }
    }

    //
    // Invariant checks: no table should be associated with the context at this
    // point and vice versa, and no keys should be set.
    //

    if (Context->Table) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(PerfectHashTableCreate, Result);
        goto Error;
    }

    if (Table->Context) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(PerfectHashTableCreate, Result);
        goto Error;
    }

    if (Table->Keys) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(PerfectHashTableCreate, Result);
        goto Error;
    }


    //
    // Argument validation and invariant checks complete, continue with table
    // creation.  Take references to the relevant context and keys instances.
    //

    Context->Vtbl->AddRef(Context);
    Table->Context = Context;
    Context->Table = Table;

    Keys->Vtbl->AddRef(Keys);
    Table->Keys = Keys;

    //
    // Copy create flags.
    //

    Table->TableCreateFlags.AsULong = TableCreateFlags.AsULong;

    //
    // Our main enumeration IDs get replicated in both structures.
    //

    Table->AlgorithmId = Context->AlgorithmId = AlgorithmId;
    Table->MaskFunctionId = Context->MaskFunctionId = MaskFunctionId;
    Table->HashFunctionId = Context->HashFunctionId = HashFunctionId;

    //
    // Complete initialization of the table's now that the algo/hash/mask IDs
    // have been set.
    //

    CompletePerfectHashTableInitialization(Table);

    if (!Table->IndexImplString) {
        Result = PH_E_NO_INDEX_IMPL_C_STRING_FOUND;
        goto Error;
    }

    //
    // Validate the table create parameters.
    //

    Table->NumberOfTableCreateParameters = NumberOfTableCreateParameters;
    Table->TableCreateParameters = TableCreateParameters;

    Result = PerfectHashTableValidateCreateParameters(Table);
    if (FAILED(Result)) {
        PH_ERROR(ValidateTableCreateParameters, Result);
        goto Error;
    }

    //
    // Dispatch remaining creation work to the algorithm-specific routine.
    //

    Result = CreationRoutines[AlgorithmId](Table);

    if (Table->OutputDirectory) {
        Table->OutputDirectory->Vtbl->Close(Table->OutputDirectory);
    }

    if (FAILED(Result)) {
        goto Error;
    }

    //
    // Successfully created the table.  Create the values array if applicable,
    // then update flags and state.
    //

    if (!IsTableCreateOnly(Table)) {
        Result = PerfectHashTableCreateValuesArray(Table, 0);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashTableCreateValuesArray, Result);
            goto Error;
        }
    }

    Table->Flags.Created = TRUE;
    Table->Flags.Loaded = FALSE;
    Table->State.Valid = TRUE;

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    RELEASE(Table->Context);
    RELEASE(Table->Keys);

    Context->Table = NULL;
    Context->State.NeedsReset = TRUE;

    ReleasePerfectHashContextLockExclusive(Context);
    ReleasePerfectHashTableLockExclusive(Table);

    return Result;
}


_Use_decl_annotations_
HRESULT
PerfectHashTableValidateCreateParameters(
    PPERFECT_HASH_TABLE Table
    )
/*++

Routine Description:

    Validates the table creation parameters for a given table.

Arguments:

    Table - Supplies a pointer to the table.

Return Value:

    S_OK - Parameters validated successfully.

    Otherwise, an appropriate error code.

--*/
{
    ULONG Index;
    HRESULT Result = S_OK;
    BOOLEAN SawResizeLimit = FALSE;
    BOOLEAN SawResizeThreshold = FALSE;
    PPERFECT_HASH_CONTEXT Context;
    PPERFECT_HASH_TABLE_CREATE_PARAMETER Param;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    Context = Table->Context;

    if (Table->NumberOfTableCreateParameters == 0) {
        if (Table->TableCreateParameters != NULL) {
            return PH_E_NUM_TABLE_CREATE_PARAMS_IS_ZERO_BUT_PARAMS_POINTER_NOT_NULL;
        }
    }

    for (Index = 0, Param = Table->TableCreateParameters;
         Index < Table->NumberOfTableCreateParameters;
         Index++, Param++) {

        switch (Param->Id) {

            case TableCreateParameterAttemptsBeforeTableResizeId:
                Context->ResizeTableThreshold = Param->AsULong;
                SawResizeThreshold = TRUE;
                break;

            case TableCreateParameterMaxNumberOfTableResizesId:
                Context->ResizeLimit = Param->AsULong;
                SawResizeLimit = TRUE;
                break;

            case TableCreateParameterBestCoverageNumAttemptsId:
                Context->BestCoverageAttempts = Param->AsULongLong;
                break;

            case TableCreateParameterBestCoverageTypeId:
                Context->BestCoverageType = Param->AsBestCoverageType;
                break;

            default:
                Result = PH_E_INVALID_TABLE_CREATE_PARAMETER_ID;
                goto Error;

        }
    }

    //
    // If find best graph is indicated in the table create flags, make sure
    // we saw appropriate table create parameters.  Otherwise, set the default
    // "first graph wins" mode.
    //

    //
    // N.B. There's a bit of an impedance mismatch (or leaky abstraction
    //      depending on which way you look at it) at the moment with the
    //      notion of "find best graph" (which is a publicly exposed flag)
    //      and the corresponding "find best memory coverage" behavior,
    //      which is an internal implementation detail.
    //

    if (Table->TableCreateFlags.FindBestGraph) {

        if (!Context->BestCoverageAttempts ||
            !Context->BestCoverageType ||
            !IsValidBestCoverageType(Context->BestCoverageType)) {

            Result = PH_E_INVALID_TABLE_CREATE_PARAMETERS_FOR_FIND_BEST_GRAPH;
            goto Error;
        }

        SetFindBestMemoryCoverage(Context);

    } else {

        //
        // Set the default solving mode.
        //

        SetFirstSolvedGraphWins(Context);

    }

    //
    // If no resize threshold or resize limit has been set, use the defaults.
    //

    if (!SawResizeThreshold) {
        Context->ResizeTableThreshold = GRAPH_SOLVING_ATTEMPTS_THRESHOLD;
    }

    if (!SawResizeLimit) {
        Context->ResizeLimit = GRAPH_SOLVING_RESIZE_TABLE_LIMIT;
    }


    //
    // Validation complete, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = PH_E_TABLE_CREATE_PARAMETER_VALIDATION_FAILED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    return Result;

}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
