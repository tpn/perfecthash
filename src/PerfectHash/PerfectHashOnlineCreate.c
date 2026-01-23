/*++

Copyright (c) 2025 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashOnlineCreate.c

Abstract:

    This module implements the online table creation routine that builds a
    perfect hash table from an in-memory key array.

--*/

#include "stdafx.h"

PERFECT_HASH_ONLINE_CREATE_TABLE_FROM_KEYS PerfectHashOnlineCreateTableFromKeys;

_Use_decl_annotations_
HRESULT
PerfectHashOnlineCreateTableFromKeys(
    PPERFECT_HASH_ONLINE Online,
    PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    ULONG KeySizeInBytes,
    ULONGLONG NumberOfKeys,
    PVOID KeysArray,
    PPERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlagsPointer,
    PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlagsPointer,
    PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters,
    PPERFECT_HASH_TABLE *TablePointer
    )
/*++

Routine Description:

    Creates a perfect hash table from an in-memory key array.

Arguments:

    Online - Supplies a pointer to the PERFECT_HASH_ONLINE interface.

    AlgorithmId - Supplies the algorithm ID to use.

    HashFunctionId - Supplies the hash function ID to use.

    MaskFunctionId - Supplies the mask function ID to use.

    KeySizeInBytes - Supplies the size of each key element, in bytes.

    NumberOfKeys - Supplies the number of keys in the array.

    KeysArray - Supplies the base address of the keys array.

    KeysLoadFlags - Optionally supplies a pointer to a keys load flags
        structure.

    TableCreateFlags - Optionally supplies a pointer to a table create flags
        structure.

    TableCreateParameters - Optionally supplies a pointer to a table create
        parameters structure.

    TablePointer - Receives a pointer to the created table on success.

Return Value:

    S_OK on success, an appropriate error code on failure.

--*/
{
    HRESULT Result = S_OK;
    PPERFECT_HASH_KEYS Keys = NULL;
    PPERFECT_HASH_TABLE Table = NULL;
    PPERFECT_HASH_CONTEXT Context = NULL;
    BOOLEAN TlsContextSet = FALSE;
    BOOLEAN AllocatorOverrideSet = FALSE;
    BOOLEAN PrevDisableGlobalAllocator = FALSE;
    PALLOCATOR PrevAllocator = NULL;
    PALLOCATOR AllocatorOverride = NULL;
    PPERFECT_HASH_TLS_CONTEXT TlsContext;
    PERFECT_HASH_TLS_CONTEXT LocalTlsContext = { 0 };
    PERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags;
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags;
    static const PERFECT_HASH_TABLE_CREATE_PARAMETERS
        DefaultTableCreateParameters = {
            sizeof(PERFECT_HASH_TABLE_CREATE_PARAMETERS),
            0,
            NULL,
            NULL,
            { 0 },
            0
        };

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Online)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(TablePointer)) {
        return E_POINTER;
    }

    *TablePointer = NULL;

    if (!ARGUMENT_PRESENT(KeysArray)) {
        return E_POINTER;
    }

    if (NumberOfKeys == 0) {
        return E_INVALIDARG;
    }

    if (!IsValidPerfectHashAlgorithmId(AlgorithmId)) {
        return PH_E_INVALID_ALGORITHM_ID;
    }

    if (!IsValidPerfectHashHashFunctionId(HashFunctionId)) {
        return PH_E_INVALID_HASH_FUNCTION_ID;
    }

    if (!IsValidPerfectHashMaskFunctionId(MaskFunctionId)) {
        return PH_E_INVALID_MASK_FUNCTION_ID;
    }

    if (KeySizeInBytes != sizeof(ULONG) &&
        KeySizeInBytes != sizeof(ULONGLONG)) {
        return PH_E_INVALID_KEY_SIZE;
    }

    if (ARGUMENT_PRESENT(KeysLoadFlagsPointer)) {
        Result = IsValidKeysLoadFlags(KeysLoadFlagsPointer);
        if (FAILED(Result)) {
            return PH_E_INVALID_KEYS_LOAD_FLAGS;
        }
        KeysLoadFlags.AsULong = KeysLoadFlagsPointer->AsULong;
    } else {
        KeysLoadFlags.AsULong = 0;
    }

    if (ARGUMENT_PRESENT(TableCreateFlagsPointer)) {
        Result = IsValidTableCreateFlags(TableCreateFlagsPointer);
        if (FAILED(Result)) {
            return Result;
        }
        TableCreateFlags.AsULongLong = TableCreateFlagsPointer->AsULongLong;
    } else {
        TableCreateFlags.AsULongLong = 0;
        TableCreateFlags.Quiet = TRUE;
    }

    //
    // Online creation never writes output files.
    //

    TableCreateFlags.NoFileIo = TRUE;

    if (!ARGUMENT_PRESENT(TableCreateParameters)) {
        TableCreateParameters = (PPERFECT_HASH_TABLE_CREATE_PARAMETERS)
            &DefaultTableCreateParameters;
    }

    //
    // Establish TLS context early so we can honor an allocator override.
    //

    TlsContext = PerfectHashTlsGetOrSetContext(&LocalTlsContext);

    if (TableCreateParameters->Allocator) {
        AllocatorOverride = TableCreateParameters->Allocator;
        PrevAllocator = TlsContext->Allocator;
        PrevDisableGlobalAllocator =
            TlsContext->Flags.DisableGlobalAllocatorComponent;
        TlsContext->Allocator = AllocatorOverride;
        TlsContext->Flags.DisableGlobalAllocatorComponent = FALSE;
        AllocatorOverrideSet = TRUE;
    }

    //
    // Create keys, context, and table instances.
    //

    Result = Online->Vtbl->CreateInstance(Online,
                                          NULL,
                                          &IID_PERFECT_HASH_KEYS,
                                          PPV(&Keys));

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashKeysCreateInstance, Result);
        goto Error;
    }

    Result = Online->Vtbl->CreateInstance(Online,
                                          NULL,
                                          &IID_PERFECT_HASH_CONTEXT,
                                          PPV(&Context));

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashContextCreateInstance, Result);
        goto Error;
    }

    Result = Online->Vtbl->CreateInstance(Online,
                                          NULL,
                                          &IID_PERFECT_HASH_TABLE,
                                          PPV(&Table));

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashTableCreateInstance, Result);
        goto Error;
    }

    //
    // Initialize context state required for graph solving.
    //

    PerfectHashContextApplyThreadpoolPriorities(Context,
                                                TableCreateParameters);

    Result = PerfectHashContextInitializeRng(Context,
                                             &TableCreateFlags,
                                             TableCreateParameters);
    if (FAILED(Result)) {
        PH_ERROR(PerfectHashContextInitializeRng, Result);
        goto Error;
    }

    Result = PerfectHashContextInitializeLowMemoryMonitor(Context, FALSE);
    if (FAILED(Result)) {
        PH_ERROR(PerfectHashContextInitializeLowMemoryMonitor, Result);
        goto Error;
    }

    TlsContext->Context = Context;
    TlsContextSet = TRUE;

    //
    // Load keys from the in-memory array.
    //

    Result = PerfectHashKeysLoadFromArray(Keys,
                                          &KeysLoadFlags,
                                          KeySizeInBytes,
                                          NumberOfKeys,
                                          KeysArray);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashKeysLoadFromArray, Result);
        goto Error;
    }

    //
    // Create the table.
    //

    Result = Table->Vtbl->Create(Table,
                                 Context,
                                 AlgorithmId,
                                 HashFunctionId,
                                 MaskFunctionId,
                                 Keys,
                                 &TableCreateFlags,
                                 TableCreateParameters);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashTableCreate, Result);
        goto Error;
    }

    if (AllocatorOverrideSet) {
        TlsContext->Allocator = PrevAllocator;
        TlsContext->Flags.DisableGlobalAllocatorComponent =
            PrevDisableGlobalAllocator;
        AllocatorOverrideSet = FALSE;
    }

    if (TlsContextSet) {
        PerfectHashTlsClearContextIfActive(&LocalTlsContext);
        TlsContextSet = FALSE;
    }

    //
    // Release our references to Keys and Context; the Table retains them.
    //

    RELEASE(Keys);
    RELEASE(Context);

    *TablePointer = Table;
    return S_OK;

Error:

    if (AllocatorOverrideSet) {
        TlsContext->Allocator = PrevAllocator;
        TlsContext->Flags.DisableGlobalAllocatorComponent =
            PrevDisableGlobalAllocator;
        AllocatorOverrideSet = FALSE;
    }

    if (TlsContextSet) {
        PerfectHashTlsClearContextIfActive(&LocalTlsContext);
        TlsContextSet = FALSE;
    }

    RELEASE(Table);
    RELEASE(Keys);
    RELEASE(Context);

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
