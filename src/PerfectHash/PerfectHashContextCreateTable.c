/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashContextCreateTable.c

Abstract:

    This module implements the perfect hash table creation routine.

--*/

#include "stdafx.h"

PERFECT_HASH_CONTEXT_CREATE_TABLE PerfectHashContextCreateTable;

_Use_decl_annotations_
HRESULT
PerfectHashContextCreateTable(
    PPERFECT_HASH_CONTEXT Context,
    PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    PPERFECT_HASH_KEYS Keys,
    PPERFECT_HASH_CONTEXT_CREATE_TABLE_FLAGS ContextCreateTableFlagsPointer,
    PVOID *TablePointer
    )
/*++

Routine Description:

    Creates and initializes a PERFECT_HASH_TABLE structure from a given set
    of keys, using the requested algorithm.

Arguments:

    Context - Supplies a pointer to an initialized PERFECT_HASH_CONTEXT
        structure that can be used by the underlying algorithm in order to
        search for perfect hash solutions in parallel.

    AlgorithmId - Supplies the algorithm to use.

    MaskFunctionId - Supplies the type of masking to use.  The algorithm and
        hash function must both support the requested masking type.

    HashFunctionId - Supplies the hash function to use.

    Keys - Supplies a pointer to a PERFECT_HASH_KEYS interface.

    ContextCreateTableFlags - Optionally supplies a pointer to a context create
        table flags structure that can be used to customize table creation.

    Table - Supplies a variable that receives the address of a perfect hash
        table structure if the routine was successful.  Caller is responsible
        for releasing the table when it is no longer required via the vtbl
        (e.g. Table->Vtbl->Release(Table)).

Return Value:

    S_OK - Table was created successfully.

    The following error codes may also be returned.  Note that this list is not
    guaranteed to be exhaustive; that is, error codes other than the ones listed
    below may also be returned.

    E_POINTER - Context, Keys or Table parameters were NULL.

    E_INVALIDARG - Path was not valid.

    E_UNEXPECTED - General error.

    E_OUTOFMEMORY - Out of memory.

    PH_E_INVALID_ALGORITHM_ID - Invalid algorithm ID.

    PH_E_INVALID_HASH_FUNCTION_ID - Invalid hash function ID.

    PH_E_INVALID_MASK_FUNCTION_ID - Invalid mask function ID.

    PH_E_INVALID_CONTEXT_CREATE_TABLE_FLAGS - Invalid flags value pointed to by
        CreateTableFlags parameter.

    PH_E_CONTEXT_LOCKED - The context is locked.  Only one table creation can
        be active at any given moment.

    PH_E_KEYS_FILE_NAME_NOT_VALID_C_IDENTIFIER - The file name component of
        the keys path contained characters that prevented it from being
        considered a valid C identifier.  As this name is used to generate
        various C structures, it must be valid.

    PH_E_CONTEXT_OUTPUT_DIRECTORY_NOT_SET - No output directory has been set
        for the context.  Call SetOutputDirectory() first.

--*/
{
    PRTL Rtl;
    HRESULT Result = S_OK;
    PALLOCATOR Allocator;
    PUNICODE_STRING OutputDirectory;
    PPERFECT_HASH_TABLE Table = NULL;
    PERFECT_HASH_CONTEXT_CREATE_TABLE_FLAGS ContextCreateTableFlags;

    if (ARGUMENT_PRESENT(TablePointer)) {
        *TablePointer = NULL;
    } else {
        return E_POINTER;
    }

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Context)) {

        return E_POINTER;

    } else {

        //
        // Initialize aliases.
        //

        Allocator = Context->Allocator;
        Rtl = Context->Rtl;

    }

    VALIDATE_FLAGS(ContextCreateTable, CONTEXT_CREATE_TABLE);

    if (!IsValidPerfectHashAlgorithmId(AlgorithmId)) {
        return PH_E_INVALID_ALGORITHM_ID;
    }

    if (!IsValidPerfectHashHashFunctionId(HashFunctionId)) {
        return PH_E_INVALID_HASH_FUNCTION_ID;
    }

    if (!IsValidPerfectHashMaskFunctionId(MaskFunctionId)) {
        return PH_E_INVALID_MASK_FUNCTION_ID;
    }

    if (!ARGUMENT_PRESENT(Keys)) {

        return E_POINTER;

    } else {

        //
        // Ensure the number of keys is within our maximum tested limit.
        //

        if (Keys->NumberOfElements.QuadPart > MAXIMUM_NUMBER_OF_KEYS) {
            return PH_E_TOO_MANY_KEYS;
        }
    }

    if (!TryAcquirePerfectHashContextLockExclusive(Context)) {
        return PH_E_CONTEXT_LOCKED;
    }

    if (!Context->OutputDirectory.Buffer) {
        ReleasePerfectHashContextLockExclusive(Context);
        return PH_E_CONTEXT_OUTPUT_DIRECTORY_NOT_SET;
    } else {
        OutputDirectory = &Context->OutputDirectory;
    }

    if (Context->State.NeedsReset) {
        Result = PerfectHashContextReset(Context);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashContextReset, Result);
            ReleasePerfectHashContextLockExclusive(Context);
            return PH_E_CONTEXT_RESET_FAILED;
        }
    }

    //
    // No table should be associated with the context at this point.
    //

    if (Context->Table) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(PerfectHashContextCreateTable, Result);
        ReleasePerfectHashContextLockExclusive(Context);
        return Result;
    }

    //
    // Create a new table instance.
    //

    Result = Context->Vtbl->CreateInstance(Context,
                                           NULL,
                                           &IID_PERFECT_HASH_TABLE,
                                           &Table);

    if (FAILED(Result)) {
        goto Error;
    }

    Context->Table = Table;

    Keys->Vtbl->AddRef(Keys);
    Table->Keys = Keys;

    Context->Vtbl->AddRef(Context);
    Table->Context = Context;
    Table->OutputDirectory = OutputDirectory;

    //
    // Our main enumeration IDs get replicated in both structures.
    //

    Table->AlgorithmId = Context->AlgorithmId = AlgorithmId;
    Table->MaskFunctionId = Context->MaskFunctionId = MaskFunctionId;
    Table->HashFunctionId = Context->HashFunctionId = HashFunctionId;

    //
    // Complete initialization of the table's vtbl now that the hash/mask IDs
    // have been set.
    //

    CompletePerfectHashTableVtblInitialization(Table);

    //
    // Common initialization is complete, dispatch remaining work to the
    // algorithm's creation routine.
    //

    Result = CreationRoutines[AlgorithmId](Table);
    if (FAILED(Result)) {
        goto Error;
    }

    //
    // We're done!  Jump to the end to finish up.
    //

    Table->State.Valid = TRUE;
    Table->Flags.Created = TRUE;

    if (ARGUMENT_PRESENT(TablePointer)) {
        *TablePointer = Table;
    }

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    if (Table) {
        Table->Vtbl->Release(Table);
        Table = NULL;
    }

    //
    // Intentional follow-on to End.
    //

End:

    Context->Table = NULL;
    Context->State.NeedsReset = TRUE;

    //
    // Update the caller's pointer to the table.  This will be NULL if an error
    // occurred, which is fine.
    //

    ReleasePerfectHashContextLockExclusive(Context);

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
