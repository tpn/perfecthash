/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTable.c

Abstract:

    This module implements the initialization and rundown routines for the
    PERFECT_HASH_TABLE instance, initialize paths routine, and the get flags
    routine.

--*/

#include "stdafx.h"

PERFECT_HASH_TABLE_INITIALIZE PerfectHashTableInitialize;

_Use_decl_annotations_
HRESULT
PerfectHashTableInitialize(
    PPERFECT_HASH_TABLE Table
    )
/*++

Routine Description:

    Initializes a perfect hash table.  This is a relatively simple method that
    just primes the COM scaffolding; the bulk of the work is done when either
    creating a new table (PerfectHashTableContext->Vtbl->CreateTable) or when
    loading an existing table (PerfectHashTable->Vtbl->Load).

Arguments:

    Table - Supplies a pointer to a PERFECT_HASH_TABLE structure for which
        initialization is to be performed.

Return Value:

    S_OK on success.  E_POINTER if Table is NULL.

--*/
{
    HRESULT Result = S_OK;

    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    Table->SizeOfStruct = sizeof(*Table);

    //
    // Create Rtl and Allocator components.
    //

    Result = Table->Vtbl->CreateInstance(Table,
                                         NULL,
                                         &IID_PERFECT_HASH_RTL,
                                         &Table->Rtl);

    if (FAILED(Result)) {
        goto Error;
    }

    Result = Table->Vtbl->CreateInstance(Table,
                                         NULL,
                                         &IID_PERFECT_HASH_ALLOCATOR,
                                         &Table->Allocator);

    if (FAILED(Result)) {
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

PERFECT_HASH_TABLE_RUNDOWN PerfectHashTableRundown;

_Use_decl_annotations_
VOID
PerfectHashTableRundown(
    PPERFECT_HASH_TABLE Table
    )
/*++

Routine Description:

    Release all resources associated with a perfect hash table.

Arguments:

    Table - Supplies a pointer to a PERFECT_HASH_TABLE structure for which
        rundown is to be performed.

Return Value:

    None.

--*/
{
    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Table)) {
        return;
    }

    //
    // Sanity check structure size.
    //

    ASSERT(Table->SizeOfStruct == sizeof(*Table));

    //
    // Free the memory used for the values array, if applicable.
    //

    if (Table->ValuesBaseAddress) {
        if (!VirtualFree(Table->ValuesBaseAddress, 0, MEM_RELEASE)) {
            SYS_ERROR(VirtualFree);
        }
        Table->ValuesBaseAddress = NULL;
    }

    //
    // Release applicable COM references.
    //

    RELEASE(Table->TableFile);
    RELEASE(Table->InfoStream);
    RELEASE(Table->CHeaderFile);
    RELEASE(Table->CSourceFile);
    RELEASE(Table->CSourceKeysFile);
    RELEASE(Table->CSourceTableDataFile);
    RELEASE(Table->Context);
    RELEASE(Table->Keys);
    RELEASE(Table->Rtl);
    RELEASE(Table->Allocator);

    return;
}


PERFECT_HASH_TABLE_GET_FLAGS PerfectHashTableGetFlags;

_Use_decl_annotations_
HRESULT
PerfectHashTableGetFlags(
    PPERFECT_HASH_TABLE Table,
    ULONG SizeOfFlags,
    PPERFECT_HASH_TABLE_FLAGS Flags
    )
/*++

Routine Description:

    Returns the flags associated with a loaded table instance.

Arguments:

    Table - Supplies a pointer to a PERFECT_HASH_TABLE structure for which the
        flags are to be obtained.

    SizeOfFlags - Supplies the size of the structure pointed to by the Flags
        parameter, in bytes.

    Flags - Supplies the address of a variable that receives the flags.

Return Value:

    S_OK - Success.

    E_POINTER - Table or Flags is NULL.

    E_INVALIDARG - SizeOfFlags does not match the size of the flags structure.

    PH_E_TABLE_NOT_LOADED - No file has been loaded yet.

    PH_E_TABLE_LOCKED - The table is locked.

--*/
{
    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Flags)) {
        return E_POINTER;
    }

    if (SizeOfFlags != sizeof(*Flags)) {
        return E_INVALIDARG;
    }

    if (!TryAcquirePerfectHashTableLockExclusive(Table)) {
        return PH_E_TABLE_LOCKED;
    }

    if (!Table->Flags.Loaded) {
        ReleasePerfectHashTableLockExclusive(Table);
        return PH_E_TABLE_NOT_LOADED;
    }

    Flags->AsULong = Table->Flags.AsULong;

    ReleasePerfectHashTableLockExclusive(Table);

    return S_OK;
}

PERFECT_HASH_TABLE_CREATE_PATH PerfectHashTableCreatePath;

_Use_decl_annotations_
HRESULT
PerfectHashTableCreatePath(
    PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    PCUNICODE_STRING NewDirectory,
    PCUNICODE_STRING NewBaseName,
    PCUNICODE_STRING AdditionalSuffix,
    PPERFECT_HASH_PATH ExistingPath,
    PPERFECT_HASH_PATH *PathPointer,
    PPERFECT_HASH_PATH_PARTS *PartsPointer
    )
/*++

Routine Description:

    Creates a new path instance from the given parameters.

Arguments:

    AlgorithmId - Supplies the algorithm to use.

    MaskFunctionId - Supplies the type of masking to use.

    HashFunctionId - Supplies the hash function to use.

    NewDirectory - Optionally supplies a fully-qualified path to use as the
        directory.

    NewBaseName - Optionally supplies a new base name to use for the table.  If
        NULL, the base name of the existing path will be used.

    AdditionalSuffix - Optionally supplies an additional suffix to add to the
        base name.

    NewExtension - Optionally supplies a new file extension to use.

    NewStreamName - Optionally supplies a new stream name to use.

    ExistingPath - Supplies a pointer to an existing path to use as a template
        for path creation.

    Path - Receives a newly created path instance on success.  Caller is
        responsible for releasing the instance when finished with it.

    Parts - Optionally receives the path parts.

Return Value:

    S_OK - Created path successfully.

    E_POINTER - Path or ExistingPath parameters were NULL.

    E_INVALIDARG - One or more parameters were invalid.

    E_OUTOFMEMORY - Out of memory.

    E_UNEXPECTED - Internal error.

    PH_E_BASE_NAME_INVALID_C_IDENTIFIER - The base name is not a valid C
        identifier.

--*/
{
    PRTL Rtl;
    HRESULT Result = S_OK;
    PPERFECT_HASH_PATH Path;
    UNICODE_STRING TableSuffix;
    WCHAR TableSuffixBuffer[256];
    LONG_INTEGER TableSuffixLength;

    if (!ARGUMENT_PRESENT(PathPointer)) {
        return E_POINTER;
    }

    //
    // Initialize the table suffix from the algorithm, hash function and mask.
    //

    TableSuffix.Buffer = (PWSTR)TableSuffixBuffer;
    TableSuffix.Length = 0;
    TableSuffix.MaximumLength = sizeof(TableSuffixBuffer);

    Result = InitializeTableSuffix(&TableSuffix,
                                   AlgorithmId,
                                   HashFunctionId,
                                   MaskFunctionId,
                                   AdditionalSuffix);

    if (FAILED(Result)) {
        PH_ERROR(InitializeTableSuffix, Result);
        goto Error;
    }

    //
    // Create a new Path instance.
    //

    Result = ExistingPath->Vtbl->CreateInstance(ExistingPath,
                                                NULL,
                                                &IID_PERFECT_HASH_PATH,
                                                &Path);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashPathCreateInstance, Result);
        goto Error;
    }

    //
    // Construct a new path from the relevant arguments.
    //

    Result = Path->Vtbl->Create(Path,
                                ExistingPath,
                                NewDirectory,
                                NewBaseName,
                                &TableSuffix,
                                NewExtension,
                                NewStreamName,
                                Parts,
                                NULL);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashPathCreate, Result);
        goto Error;
    }

    //
    // Update the caller's pointer and return.
    //

    *PathPointer = Path;
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

PERFECT_HASH_TABLE_GET_FILE PerfectHashTableGetFile;

_Use_decl_annotations_
HRESULT
PerfectHashTableGetFile(
    PPERFECT_HASH_FILE Table,
    PCPERFECT_HASH_FILE *File
    )
/*++

Routine Description:

    Obtains the file instance for a given perfect hash table.

Arguments:

    Table - Supplies a pointer to a PERFECT_HASH_TABLE structure for which the
        file is to be obtained.

    File - Supplies the address of a variable that receives a pointer to the
        file instance.  The caller must release this reference when finished
        with it via File->Vtbl->Release(File).

Return Value:

    S_OK - Success.

    E_POINTER - Table or File parameters were NULL.

    PH_E_TABLE_LOCKED - The file is locked exclusively.

    PH_E_INVALID_TABLE - Table is invalid.

--*/
{

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(File)) {
        return E_POINTER;
    }

    //
    // Clear the caller's pointer up-front.
    //

    *File = NULL;

    if (!TryAcquirePerfectHashTableLockShared(Table)) {
        return PH_E_TABLE_LOCKED;
    }

    if (!IsValidTable(Table)) {
        ReleasePerfectHashTableLockShared(Table);
        return PH_E_INVALID_TABLE;
    }

    //
    // Argument validation complete.  Add a reference to the path and update
    // the caller's pointer, then return success.
    //

    Table->File->Vtbl->AddRef(Table->File);
    *File = Table->File;

    ReleasePerfectHashTableLockShared(Table);

    return S_OK;
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
