/*++

Copyright (c) 2018-2020 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTableCreate.c

Abstract:

    This module implements the Create() routine for the perfect hash table
    component.

--*/

#include "stdafx.h"

//
// Cap the maximum key set size we're willing to process to 2 billion.
//

#define MAXIMUM_NUMBER_OF_KEYS 2000000000

//
// Define the threshold for how many attempts need to be made at finding a
// perfect hash solution before we double our number of vertices and try again.
//

#define GRAPH_SOLVING_ATTEMPTS_THRESHOLD 100000000000

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
_Must_inspect_result_
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
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    PPERFECT_HASH_KEYS Keys,
    PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlagsPointer,
    PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
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

    HashFunctionId - Supplies the hash function to use.

    MaskFunctionId - Supplies the type of masking to use.  The algorithm and
        hash function must both support the requested masking type.

    Keys - Supplies a pointer to a PERFECT_HASH_KEYS interface.

    TableCreateFlags - Optionally supplies a pointer to a context create
        table flags structure that can be used to customize table creation.

    TableCreateParameters - Optionally supplies a pointer to a table create
        parameters structure that can be used to further customize table
        creation behavior.

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

    PH_E_INVALID_SEED_MASKS_STRUCTURE_SIZE - The size of the seed masks
        structure is too small (that is, the given graph indicates that it is
        using more seeds than can be currently captured by the SEED_MASKS
        structure).

--*/
{
    PRTL Rtl;
    PALLOCATOR Allocator;
    HRESULT Result = S_OK;
    HRESULT CloseResult;
    HRESULT CreateValuesResult;
    ULONG NumberOfSeeds;
    ULONG NumberOfMasks;
    PCSEED_MASKS SeedMasks;
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags;
    PPERFECT_HASH_FILE TableSizeFile = NULL;
    PULARGE_INTEGER RequestedNumberOfTableElements;
    LARGE_INTEGER EmptyEndOfFile = { 0 };
    PLARGE_INTEGER EndOfFile;

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
    // Allocate sufficient space for the assigned memory coverage structure.
    //

    Table->Coverage = Allocator->Vtbl->Calloc(Allocator,
                                              1,
                                              sizeof(*Table->Coverage));

    if (!Table->Coverage) {
        Result = E_OUTOFMEMORY;
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
    // Enable non-temporal AVX2 routines here if requested.  This is a bit
    // hacky, as we're fiddling with vtbl pointers in Rtl which we really
    // shouldn't have visibility into; but eh, it works and won't break
    // anything.
    //

#if defined(_M_AMD64) || defined(_M_X64)
    if (UseNonTemporalAvx2Routines(Table) &&
        Rtl->CpuFeatures.AVX2 != FALSE) {

        Rtl->Vtbl->CopyPages = RtlCopyPagesNonTemporal_AVX2;
        Rtl->Vtbl->FillPages = RtlFillPagesNonTemporal_AVX2;
    }
#endif

    //
    // Our main enumeration IDs get replicated in both structures.
    //

    Table->AlgorithmId = Context->AlgorithmId = AlgorithmId;
    Table->MaskFunctionId = Context->MaskFunctionId = MaskFunctionId;
    Table->HashFunctionId = Context->HashFunctionId = HashFunctionId;

    //
    // If this routine uses seed masks, make the context aware.
    //

    SeedMasks = HashRoutineSeedMasks[HashFunctionId];

    if (IsValidSeedMasks(SeedMasks)) {

        //
        // Verify the size of the masks structure is appropriate for the number
        // of seeds required by this hash function.
        //

        NumberOfSeeds = HashRoutineNumberOfSeeds[HashFunctionId];
        NumberOfMasks = sizeof(*SeedMasks) / sizeof(SeedMasks->Mask1);
        if (NumberOfSeeds > NumberOfMasks) {
            Result = PH_E_INVALID_SEED_MASKS_STRUCTURE_SIZE;
            goto Error;
        }

        //
        // Validation complete; set the seed masks for this run.
        //

        Context->SeedMasks = SeedMasks;
    }

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

    Table->TableCreateParameters = TableCreateParameters;

    Result = PerfectHashTableValidateCreateParameters(Table);
    if (FAILED(Result)) {
        PH_ERROR(ValidateTableCreateParameters, Result);
        goto Error;
    }

    //
    // Attempt to load previous table size.
    //

    RequestedNumberOfTableElements = &Table->RequestedNumberOfTableElements;
    Result = PerfectHashKeysLoadTableSize(Keys,
                                          AlgorithmId,
                                          HashFunctionId,
                                          MaskFunctionId,
                                          &TableSizeFile,
                                          RequestedNumberOfTableElements);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashKeysLoadTableSize, Result);
        goto Error;
    }

    //
    // If we haven't been asked to use the previous table size, reset the
    // table's requested number of elements back to 0.
    //

    if (TableCreateFlags.UsePreviousTableSize == FALSE) {
        Table->RequestedNumberOfTableElements.QuadPart = 0;
    }

    //
    // Dispatch remaining creation work to the algorithm-specific routine.
    //

    Result = CreationRoutines[AlgorithmId](Table);

    if (Table->OutputDirectory) {
        CloseResult = Table->OutputDirectory->Vtbl->Close(Table->OutputDirectory);
        if (FAILED(CloseResult)) {

            //
            // N.B. We don't 'goto Error' here at the end of this block like
            //      we normally do as we want the table size file close logic
            //      below to also run.
            //

            PH_ERROR(PerfectHashDirectoryClose, CloseResult);
            Result = CloseResult;
        }
    }

    //
    // Use an empty (0) end of file if we're ignoring previous table size, or
    // the table creation was not successful.  This will result in the
    // underlying file being deleted by the Close() call below.  Otherwise,
    // write the size back to the file.
    //

    if ((!TableCreateFlags.UsePreviousTableSize) || Result != S_OK) {

        EndOfFile = &EmptyEndOfFile;

    } else {

        EndOfFile = NULL;

        RequestedNumberOfTableElements =
            (PULARGE_INTEGER)TableSizeFile->BaseAddress;

        //
        // We can't use Table->TableInfoOnDisk->NumberOfTableElements.QuadPart
        // here, as that won't be valid if the table was created with the flag
        // 'CreateOnly'.  Use Table->HashSize instead, as that is always filled
        // out.  (As a side note, this highlights some of the less-than-ideal
        // quirks regarding our internal structure design and field naming.)
        //

        RequestedNumberOfTableElements->QuadPart = Table->HashSize;

        //
        // Invariant checks: the number of table elements should be greater than
        // zero, and, if non-modulus masking is active, should be a power of 2.
        //

        if (RequestedNumberOfTableElements->QuadPart == 0) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(PerfectHashTableCreate_NumTableElemsIsZero, Result);
            goto Error;
        }

        if (!IsModulusMasking(MaskFunctionId) &&
            !IsPowerOfTwo(RequestedNumberOfTableElements->QuadPart)) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(PerfectHashTableCreate_NumTableElemsNotPow2, Result);
            goto Error;
        }

        TableSizeFile->NumberOfBytesWritten.QuadPart = sizeof(ULARGE_INTEGER);
    }

    CloseResult = TableSizeFile->Vtbl->Close(TableSizeFile, EndOfFile);
    if (FAILED(CloseResult)) {
        PH_ERROR(PerfectHashTableCreate_TableSizeFileClose, CloseResult);
        Result = CloseResult;
        goto Error;
    }

    if (Result != S_OK) {
        goto Error;
    }

    //
    // Successfully created the table.  Create the values array if applicable,
    // then update flags and state.
    //

    if (!IsTableCreateOnly(Table)) {
        CreateValuesResult = PerfectHashTableCreateValuesArray(Table, 0);
        if (FAILED(CreateValuesResult)) {
            if (CreateValuesResult != E_OUTOFMEMORY) {
                PH_ERROR(PerfectHashTableCreateValuesArray, CreateValuesResult);
                Result = CreateValuesResult;
            } else {
                Result = PH_I_TABLE_CREATED_BUT_VALUES_ARRAY_ALLOC_FAILED;
            }
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

    RELEASE(TableSizeFile);
    RELEASE(Table->Context);
    RELEASE(Table->Keys);

    Context->Table = NULL;
    Context->State.NeedsReset = TRUE;

    if (Result == E_OUTOFMEMORY) {

        //
        // Convert the out-of-memory error code into our equivalent info code.
        //

        Result = PH_I_OUT_OF_MEMORY;
    }

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
    PRTL Rtl;
    ULONG Index;
    ULONG Count;
    HRESULT Result = S_OK;
    BOOLEAN SawResizeLimit = FALSE;
    BOOLEAN SawResizeThreshold = FALSE;
    PPERFECT_HASH_CONTEXT Context;
    PPERFECT_HASH_TABLE_CREATE_PARAMETER Param;
    PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParams;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    Rtl = Table->Rtl;
    Context = Table->Context;
    TableCreateParams = Table->TableCreateParameters;

    Count = TableCreateParams->NumberOfElements;
    Param = TableCreateParams->Params;

    if (Count == 0) {
        if (Param != NULL) {
            return PH_E_INVALID_TABLE_CREATE_PARAMETERS;
        }
    } else {
        if (Param == NULL) {
            return PH_E_INVALID_TABLE_CREATE_PARAMETERS;
        }
    }

    for (Index = 0; Index < Count; Index++, Param++) {

        switch (Param->Id) {

            case TableCreateParameterAttemptsBeforeTableResizeId:
                Context->ResizeTableThreshold = Param->AsULong;
                SawResizeThreshold = TRUE;
                break;

            case TableCreateParameterMaxNumberOfTableResizesId:
                Context->ResizeLimit = Param->AsULong;
                SawResizeLimit = TRUE;
                break;

            case TableCreateParameterBestCoverageAttemptsId:
                Context->BestCoverageAttempts = Param->AsULongLong;
                break;

            case TableCreateParameterBestCoverageTypeId:
                Context->BestCoverageType = Param->AsBestCoverageType;
                break;

            case TableCreateParameterKeysSubsetId:
                Context->KeysSubset = &Param->AsKeysSubset;
                break;

            case TableCreateParameterMainWorkThreadpoolPriorityId:
            case TableCreateParameterFileWorkThreadpoolPriorityId:

                //
                // These two parameters are handled earlier by the context
                // (see PerfectHashContextApplyThreadpoolPriorities()).
                //

                break;

            case TableCreateParameterSeedsId:
                Context->UserSeeds = &Param->AsValueArray;
                break;

            case TableCreateParameterKeySizeInBytesId:

                //
                // We don't need to do anything for this parameter; it will have
                // been consumed much earlier in the pipeline.
                //

                break;

            case TableCreateParameterValueSizeInBytesId:
                Table->ValueSizeInBytes =
                    (ULONG)Rtl->RoundUpPowerOfTwo32(Param->AsULong);
                if (Table->ValueSizeInBytes == 4) {
                    Table->ValueType = LongType;
                } else if (Table->ValueSizeInBytes == 8) {
                    Table->ValueType = LongLongType;
                } else {
                    Result = PH_E_INVALID_VALUE_SIZE_IN_BYTES_PARAMETER;
                    goto Error;
                }
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
            !IsValidPerfectHashBestCoverageTypeId(Context->BestCoverageType)) {

            Result = PH_E_INVALID_TABLE_CREATE_PARAMETERS_FOR_FIND_BEST_GRAPH;
            goto Error;
        }

        SetFindBestMemoryCoverage(Context);

        if (DoesBestCoverageTypeRequireKeysSubset(Context->BestCoverageType)) {
            if (!Context->KeysSubset) {
                Result = PH_E_BEST_COVERAGE_TYPE_REQUIRES_KEYS_SUBSET;
                goto Error;
            }
            Context->State.BestMemoryCoverageForKeysSubset = TRUE;
        }

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
    // Use our default global NT-style C type names array for now.
    //

    Table->CTypeNames = (PCSTRING)&NtTypeNames;

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
