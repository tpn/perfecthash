/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashKeysLoadTableSize.c

Abstract:

    This module implements routines related to the persistence of table size
    information for a given keys instance and algorithm, hash and mask type.
    Routines are provided for loading table size data and initializing an
    appropriate suffix.

--*/

#include "stdafx.h"

#ifdef PH_ONLINE_ONLY

PERFECT_HASH_KEYS_LOAD_TABLE_SIZE PerfectHashKeysLoadTableSize;

_Use_decl_annotations_
HRESULT
PerfectHashKeysLoadTableSize(
    PPERFECT_HASH_KEYS Keys,
    PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    PPERFECT_HASH_FILE *TableSizeFilePointer,
    PULARGE_INTEGER RequestedNumberOfTableElements
    )
{
    UNREFERENCED_PARAMETER(Keys);
    UNREFERENCED_PARAMETER(AlgorithmId);
    UNREFERENCED_PARAMETER(HashFunctionId);
    UNREFERENCED_PARAMETER(MaskFunctionId);
    UNREFERENCED_PARAMETER(TableSizeFilePointer);
    UNREFERENCED_PARAMETER(RequestedNumberOfTableElements);

    return PH_E_NOT_IMPLEMENTED;
}

#else

#define KEYS_TABLE_SIZE_FILE_SIZE sizeof(ULONGLONG)

//
// Forward decls.
//

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI PERFECT_HASH_KEYS_TABLE_SIZE_INITIALIZE_SUFFIX)(
    _In_ PPERFECT_HASH_KEYS Keys,
    _Inout_ PUNICODE_STRING Suffix,
    _In_ PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    _In_ PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    _In_ PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId
    );
typedef PERFECT_HASH_KEYS_TABLE_SIZE_INITIALIZE_SUFFIX
      *PPERFECT_HASH_KEYS_TABLE_SIZE_INITIALIZE_SUFFIX;

extern PERFECT_HASH_KEYS_TABLE_SIZE_INITIALIZE_SUFFIX
    PerfectHashKeysTableSizeInitializeSuffix;

//
// Begin method implementations.
//

PERFECT_HASH_KEYS_LOAD_TABLE_SIZE PerfectHashKeysLoadTableSize;

_Use_decl_annotations_
HRESULT
PerfectHashKeysLoadTableSize(
    PPERFECT_HASH_KEYS Keys,
    PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    PPERFECT_HASH_FILE *FilePointer,
    PULARGE_INTEGER RequestedNumberOfTableElements
    )
/*++

Routine Description:

    For a given algorithm, hash function and masking type, loads the table size
    associated with a keys file.  If a perfect hash table has previously been
    created, the table size will reflect the previous size that was used to
    successfully create a table.

    The motivation behind this routine is to reduce table creation time by
    appropriately sizing the number of edges and vertices up front, which
    avoids the need to perform any table resize events.

    N.B. The terms table size and "requested number of table elements" are used
         interchangeably.

Arguments:

    Keys - Supplies a pointer to loaded and unlocked keys instance from which
        the table size is to be loaded.

    AlgorithmId - Supplies the algorithm to use.

    HashFunctionId - Supplies the hash function to use.

    MaskFunctionId - Supplies the type of masking to use.

    FilePointer - Receives a pointer to a file instance backing the table size.
        The caller is responsible for writing the final table size to this
        file instance (i.e. via the memory mapped File->BaseAddress), then
        calling Close() and Release() once it is no longer required.  That is,
        the caller is solely responsible for the lifetime of the returned
        instance.

    RequestedNumberOfTableElements - Receives the table size, or requested
        number of table elements, for the given keys instance and algo, hash,
        mask combination.  If no previous value has been recorded, this will
        be 0.

Return Value:

    S_OK - Success.

    N.B. Not an exhaustive list of error codes.

    E_POINTER - Keys, FilePointer or RequestedNumberOfTableElements were NULL.

    PH_E_INVALID_ALGORITHM_ID - Invalid algorithm ID.

    PH_E_INVALID_HASH_FUNCTION_ID - Invalid hash function ID.

    PH_E_INVALID_MASK_FUNCTION_ID - Invalid mask function ID.

    PH_E_KEYS_LOCKED - The keys are locked.

    PH_E_KEYS_NOT_LOADED - The keys file has not been loaded.

    PH_E_SYSTEM_CALL_FAILED - A system call has failed.

--*/
{
    PRTL Rtl;
    HRESULT Result = S_OK;
    PPERFECT_HASH_FILE File = NULL;
    PPERFECT_HASH_PATH Path = NULL;
    LARGE_INTEGER EndOfFile;
    PERFECT_HASH_FILE_CREATE_FLAGS FileCreateFlags;
    PERFECT_HASH_PATH_CREATE_FLAGS PathCreateFlags;
    UNICODE_STRING Suffix;
    WCHAR SuffixBuffer[KEYS_TABLE_SIZE_SUFFIX_BUFFER_SIZE_IN_CHARS];

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Keys)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(FilePointer)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(RequestedNumberOfTableElements)) {
        return E_POINTER;
    }

    if (!IsValidPerfectHashAlgorithmId(AlgorithmId)) {
        return E_INVALIDARG;
    }

    if (!IsValidPerfectHashHashFunctionId(HashFunctionId)) {
        return E_INVALIDARG;
    }

    if (!IsValidPerfectHashMaskFunctionId(MaskFunctionId)) {
        return E_INVALIDARG;
    }

    if (!TryAcquirePerfectHashKeysLockExclusive(Keys)) {
        return PH_E_KEYS_LOCKED;
    }

    if (!IsLoadedKeys(Keys)) {
        ReleasePerfectHashKeysLockExclusive(Keys);
        return PH_E_KEYS_NOT_LOADED;
    }

    //
    // Argument validation complete.  Continue with loading.
    //

    //
    // Create a path instance.
    //

    Result = Keys->Vtbl->CreateInstance(Keys,
                                        NULL,
                                        &IID_PERFECT_HASH_PATH,
                                        PPV(&Path));

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashPathCreateInstance, Result);
        goto Error;
    }

    //
    // Initialize the table size suffix then create the path.  E.g.
    //
    // Keys path:
    //
    //      C:\Temp\keys\KernelBase_2415.keys
    //
    // Table size path:
    //
    //      C:\Temp\keys\KernelBase_2415.keys:Chm01_Crc32Rotate_And.TableSize
    //
    //

    Rtl = Keys->Rtl;
    ZeroArray(SuffixBuffer);

    Suffix.Buffer = (PWSTR)SuffixBuffer;
    Suffix.Length = 0;
    Suffix.MaximumLength = sizeof(SuffixBuffer);

    Result = PerfectHashKeysTableSizeInitializeSuffix(Keys,
                                                      &Suffix,
                                                      AlgorithmId,
                                                      HashFunctionId,
                                                      MaskFunctionId);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashKeysTableSizeInitializeSuffix, Result);
        goto Error;
    }

    PathCreateFlags.AsULong = 0;
    PathCreateFlags.DisableCharReplacement = TRUE;

    Result = Path->Vtbl->Create(Path,
                                Keys->File->Path,
                                NULL,               // NewDirectory
                                NULL,               // DirectorySuffix
                                NULL,               // NewBaseName
                                NULL,               // BaseNameSuffix
                                NULL,               // NewExtension
                                &Suffix,            // NewStreamName
                                NULL,               // Parts
                                &PathCreateFlags);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashPathCreate, Result);
        goto Error;
    }

    //
    // Create a file instance.
    //

    Result = Keys->Vtbl->CreateInstance(Keys,
                                        NULL,
                                        &IID_PERFECT_HASH_FILE,
                                        PPV(&File));

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashFileCreateInstance, Result);
        goto Error;
    }

    //
    // Initialize end of file and file create flags, then create the file.
    //

    EndOfFile.QuadPart = KEYS_TABLE_SIZE_FILE_SIZE;

    FileCreateFlags.AsULong = 0;
    FileCreateFlags.NoTruncate = TRUE;

    Result = File->Vtbl->Create(File,
                                Path,
                                &EndOfFile,
                                NULL,
                                &FileCreateFlags);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashKeysLoadTableSize_FileCreate, Result);
        goto Error;
    }

    RequestedNumberOfTableElements->QuadPart = *((PULONGLONG)File->BaseAddress);

    //
    // Invariant check: if the number of table elements is greater than zero
    // and non-modulus masking is active, verify it is a power of 2.
    //

    if (RequestedNumberOfTableElements->QuadPart > 0 &&
        !IsModulusMasking(MaskFunctionId) &&
        !IsPowerOfTwo(RequestedNumberOfTableElements->QuadPart)) {

        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(PerfectHashKeysLoadTableSize_NumTableElemsNotPow2, Result);
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

    RELEASE(File);

    //
    // Intentional follow-on to End.
    //

End:

    RELEASE(Path);

    //
    // Update the caller's pointer.  File may be NULL here, which is okay.
    //

    *FilePointer = File;

    ReleasePerfectHashKeysLockExclusive(Keys);

    return Result;
}


extern PERFECT_HASH_KEYS_TABLE_SIZE_INITIALIZE_SUFFIX
    PerfectHashKeysTableSizeInitializeSuffix;

_Use_decl_annotations_
HRESULT
PerfectHashKeysTableSizeInitializeSuffix(
    PPERFECT_HASH_KEYS Keys,
    PUNICODE_STRING Suffix,
    PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId
    )
/*++

Routine Description:

    Helper routine for constructing the keys table size suffix.

Arguments:

    Keys - Supplies a pointer to a keys instance.

    Suffix - Supplies a pointer to an empty, initialized UNICODE_STRING
        structure that will receive the suffix.  That is, Length must be
        0, MaximumLength must reflect the total size in bytes of the Buffer,
        and Buffer must not be NULL.  If the routine is successful, Length
        will be updated and Buffer will be written to.

    AlgorithmId - Supplies the algorithm to use.

    HashFunctionId - Supplies the hash function to use.

    MaskFunctionId - Supplies the type of masking to use.

Return Value:

    S_OK - Initialized suffix successfully.

    PH_E_STRING_BUFFER_OVERFLOW - Suffix was too small.

    PH_E_INVARIANT_CHECK_FAILED - Internal error.

--*/
{
    PRTL Rtl;
    PWSTR Dest;
    USHORT Count;
    USHORT Offset = 0;
    HRESULT Result = S_OK;
    ULONG_PTR ExpectedDest;
    LONG_INTEGER SuffixLength;
    PUNICODE_STRING AlgorithmName;
    PUNICODE_STRING HashFunctionName;
    PUNICODE_STRING MaskFunctionName;
    PCUNICODE_STRING AdditionalSuffix = &KeysTableSizeSuffix;

    Rtl = Keys->Rtl;

    //
    // AlgorithmName
    //

    AlgorithmName = (PUNICODE_STRING)AlgorithmNames[AlgorithmId];
    SuffixLength.LongPart = AlgorithmName->Length;

    //
    // HashFunctionName
    //

    HashFunctionName = (PUNICODE_STRING)HashFunctionNames[HashFunctionId];
    SuffixLength.LongPart += (
        sizeof(L'_') +
        HashFunctionName->Length
    );

    //
    // MaskFunctionName
    //

    MaskFunctionName = (PUNICODE_STRING)MaskFunctionNames[MaskFunctionId];
    SuffixLength.LongPart += (
        sizeof(L'_') +
        MaskFunctionName->Length
    );

    SuffixLength.LongPart += AdditionalSuffix->Length;

    if (SuffixLength.HighPart) {
        Result = PH_E_STRING_BUFFER_OVERFLOW;
        PH_ERROR(InitializeKeysTableSizeSuffix_SuffixLength, Result);
        goto Error;
    }

    if ((ULONG)Suffix->MaximumLength <
        SuffixLength.LongPart + sizeof(WCHAR)) {
        Result = PH_E_STRING_BUFFER_OVERFLOW;
        PH_ERROR(InitializeKeysTableSizeSuffix_SuffixMaxLength, Result);
        goto Error;
    }

    Dest = Suffix->Buffer;

    //
    // AlgorithmName
    //

    Offset = (USHORT)RtlPointerToOffset(Suffix->Buffer, Dest);
    Count = AlgorithmName->Length / sizeof(WCHAR);
    CopyInline(Dest, AlgorithmName->Buffer, AlgorithmName->Length);
    Dest += Count;

    //
    // HashFunctionName
    //

    *Dest++ = L'_';
    Count = HashFunctionName->Length / sizeof(WCHAR);
    CopyInline(Dest, HashFunctionName->Buffer, HashFunctionName->Length);
    Dest += Count;

    //
    // MaskFunctionName
    //

    *Dest++ = L'_';
    Count = MaskFunctionName->Length / sizeof(WCHAR);
    CopyInline(Dest, MaskFunctionName->Buffer, MaskFunctionName->Length);
    Dest += Count;

    //
    // AdditionalSuffix
    //

    Count = AdditionalSuffix->Length / sizeof(WCHAR);
    CopyInline(Dest, AdditionalSuffix->Buffer, AdditionalSuffix->Length);
    Dest += Count;

    ExpectedDest = (ULONG_PTR)(
        RtlOffsetToPointer(
            Suffix->Buffer,
            SuffixLength.LowPart
        )
    );

    if ((ULONG_PTR)Dest != ExpectedDest) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(InitializeKeysTableSizeSuffix_FinalDestCheck, Result);
        goto Error;
    }

    Suffix->Length = SuffixLength.LowPart;

    *Dest++ = L'\0';

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

    return Result;
}

#endif // PH_ONLINE_ONLY

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
