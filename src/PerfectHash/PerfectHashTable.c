/*++

Copyright (c) 2018-2022. Trent Nelson <trent@trent.me>

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
                                         PPV(&Table->Rtl));

    if (FAILED(Result)) {
        goto Error;
    }

    Result = Table->Vtbl->CreateInstance(Table,
                                         NULL,
                                         &IID_PERFECT_HASH_ALLOCATOR,
                                         PPV(&Table->Allocator));

    if (FAILED(Result)) {
        goto Error;
    }

    //
    // Initialize the timestamp string.
    //

    Result = InitializeTimestampString((PCHAR)&Table->TimestampBuffer,
                                       sizeof(Table->TimestampBuffer),
                                       &Table->TimestampString,
                                       &Table->FileTime.AsFileTime,
                                       &Table->SystemTime);
    if (FAILED(Result)) {
        PH_ERROR(PerfectHashTableInitialize_InitTimestampString, Result);
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
    PALLOCATOR Allocator;

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

    Allocator = Table->Allocator;

    //
    // Free the memory used for the values array, if applicable.
    //

    if (Table->ValuesBaseAddress) {
        if (!VirtualFree(Table->ValuesBaseAddress,
                         VFS(Table->ValuesArraySizeInBytes),
                         MEM_RELEASE)) {
            SYS_ERROR(VirtualFree);
            PH_RAISE(E_UNEXPECTED);
        }
        Table->ValuesBaseAddress = NULL;
    }

    //
    // Free the copy of the coverage information from the best graph, if
    // applicable.
    //

    if (Table->Coverage) {
        Allocator->Vtbl->FreePointer(Allocator, &Table->Coverage);
    }

    //
    // Free the post-create in-memory allocation for the table info and table
    // data, if applicable.
    //

    if (Table->Flags.Created && !IsTableCreateOnly(Table)) {
        if (Table->TableInfoOnDisk && WasTableInfoOnDiskHeapAllocated(Table)) {
            Allocator->Vtbl->FreePointer(Allocator,
                                         PPV(&Table->TableInfoOnDisk));
        }
        if (Table->TableDataBaseAddress && WasTableDataHeapAllocated(Table)) {
            if (!VirtualFree(Table->TableDataBaseAddress,
                             VFS(Table->TableDataSizeInBytes),
                             MEM_RELEASE)) {
                SYS_ERROR(VirtualFree);
            }
            Table->TableDataBaseAddress = NULL;
        }
    }

    //
    // Invariant check: the context files should always be NULL.
    //

#define EXPAND_AS_ASSERT_NULL(Verb, VUpper, Name, Upper) \
    ASSERT(Table->Name == NULL);

    CONTEXT_FILE_WORK_TABLE_ENTRY(EXPAND_AS_ASSERT_NULL);

    //
    // Release applicable COM references.
    //

#define EXPAND_AS_RELEASE(          \
    Verb, VUpper, Name, Upper,      \
    EofType, EofValue,              \
    Suffix, Extension, Stream, Base \
)                                   \
    RELEASE(Table->Name);

    FILE_WORK_TABLE_ENTRY(EXPAND_AS_RELEASE);

    RELEASE(Table->OutputPath);
    RELEASE(Table->OutputDirectory);
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

PERFECT_HASH_TABLE_INITIALIZE_TABLE_SUFFIX
    PerfectHashTableInitializeTableSuffix;

_Use_decl_annotations_
HRESULT
PerfectHashTableInitializeTableSuffix(
    PPERFECT_HASH_TABLE Table,
    PUNICODE_STRING Suffix,
    PULONG NumberOfResizeEvents,
    PULARGE_INTEGER NumberOfTableElements,
    PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    PCUNICODE_STRING AdditionalSuffix,
    PUSHORT AlgorithmOffset
    )
/*++

Routine Description:

    Helper routine for constructing the base name suffix for a perfect hash
    table from a given set of parameters.  The suffix is constructed from
    parameters in the order they appear in the function signature, assuming
    they are valid.

Arguments:

    Table - Supplies a pointer to a table instance.

    Suffix - Supplies a pointer to an empty, initialized UNICODE_STRING
        structure that will receive the table suffix.  That is, Length must be
        0, MaximumLength must reflect the total size in bytes of the Buffer, and
        Buffer must not be NULL.  If the routine is successful, Length will be
        updated and Buffer will be written to.

    NumberOfResizeEvents - Optionally supplies the number of table resize
        events that have occurred for this table.

    NumberOfTableElements - Optionally supplies the number of table elements.
        This value will be converted into a base-10 string representation if
        present.

    AlgorithmId - Optionally supplies the algorithm to use.

    HashFunctionId - Optionally supplies the hash function to use.

    MaskFunctionId - Optionally supplies the type of masking to use.

    AdditionalSuffix - Optionally supplies an additional suffix to append to
        the buffer.

    AlgorithmOffset - Receives the byte offset of the algorithm name, relative
        to Suffix->Buffer, excluding the leading '_'.  E.g. given the suffix
        "_Chm01_Crc32Rotate_And", the algorithm offset will be the position of
        the 'C' in "_Chm01...".

Return Value:

    S_OK - Initialized suffix successfully.

    E_INVALIDARG - AdditionalSuffix, if non-NULL, was not valid.

    PH_E_STRING_BUFFER_OVERFLOW - Suffix was too small.

    PH_E_INVARIANT_CHECK_FAILED - Internal error.

--*/
{
    PRTL Rtl;
    PWSTR Dest;
    USHORT Count;
    USHORT Offset = 0;
    BOOLEAN Success;
    HRESULT Result = S_OK;
    ULONG_PTR ExpectedDest;
    BYTE NumberOfResizeDigits = 0;
    BYTE NumberOfElementsDigits = 0;
    LONG_INTEGER TableSuffixLength = { 0 };
    BOOLEAN IncludeNumberOfResizes;
    BOOLEAN IncludeNumberOfElements;
    PUNICODE_STRING AlgorithmName = NULL;
    PUNICODE_STRING HashFunctionName = NULL;
    PUNICODE_STRING MaskFunctionName = NULL;

    Rtl = Table->Rtl;

    IncludeNumberOfResizes = (
        IncludeNumberOfTableResizeEventsInOutputPath(Table) &&
        ARGUMENT_PRESENT(NumberOfResizeEvents)
    );

    if (IncludeNumberOfResizes) {
        NumberOfResizeDigits = (
            CountNumberOfDigitsInline(*NumberOfResizeEvents)
        );
        TableSuffixLength.LongPart += (
            sizeof(L'_') +
            (NumberOfResizeDigits * sizeof(WCHAR))
        );
    }

    IncludeNumberOfElements = (
        IncludeNumberOfTableElementsInOutputPath(Table) &&
        ARGUMENT_PRESENT(NumberOfTableElements)
    );

    if (IncludeNumberOfElements) {
        NumberOfElementsDigits = (
            CountNumberOfLongLongDigitsInline(NumberOfTableElements->QuadPart)
        );
        TableSuffixLength.LongPart += (
            sizeof(L'_') +
            (NumberOfElementsDigits * sizeof(WCHAR))
        );
    }

    if (IsValidPerfectHashAlgorithmId(AlgorithmId)) {
        AlgorithmName = (PUNICODE_STRING)AlgorithmNames[AlgorithmId];
        TableSuffixLength.LongPart += (
            sizeof(L'_') +
            AlgorithmName->Length
        );
    }

    if (IsValidPerfectHashHashFunctionId(HashFunctionId)) {
        HashFunctionName = (PUNICODE_STRING)HashFunctionNames[HashFunctionId];
        TableSuffixLength.LongPart += (
            sizeof(L'_') +
            HashFunctionName->Length
        );
    }

    if (IsValidPerfectHashMaskFunctionId(MaskFunctionId)) {
        MaskFunctionName = (PUNICODE_STRING)MaskFunctionNames[MaskFunctionId];
        TableSuffixLength.LongPart += (
            sizeof(L'_') +
            MaskFunctionName->Length
        );
    }

    if (ARGUMENT_PRESENT(AdditionalSuffix)) {
        if (!IsValidUnicodeString(AdditionalSuffix)) {
            Result = E_INVALIDARG;
            PH_ERROR(InitializeTableSuffix_AdditionalSuffix, Result);
            goto Error;
        }
        TableSuffixLength.LongPart += (
            sizeof(L'_') +
            AdditionalSuffix->Length
        );
    }

    if (TableSuffixLength.HighPart) {
        Result = PH_E_STRING_BUFFER_OVERFLOW;
        PH_ERROR(InitializeTableSuffix_TableSuffixLength, Result);
        goto Error;
    }

    if ((ULONG)Suffix->MaximumLength <
        TableSuffixLength.LongPart + sizeof(WCHAR)) {
        Result = PH_E_STRING_BUFFER_OVERFLOW;
        PH_ERROR(InitializeTableSuffix_SuffixMaxLength, Result);
        goto Error;
    }

    Dest = Suffix->Buffer;

    if (NumberOfResizeDigits) {
        *Dest++ = L'_';
        Suffix->Length = sizeof(L'_');

        Success = (
            AppendIntegerToUnicodeString(
                Suffix,
                *NumberOfResizeEvents,
                NumberOfResizeDigits,
                L'\0'
            )
        );

        if (!Success) {
            Result = PH_E_STRING_BUFFER_OVERFLOW;
            PH_ERROR(InitializeTableSuffix_AppendIntegerResize, Result);
            goto Error;
        }

        Dest += NumberOfResizeDigits;
    }

    if (NumberOfElementsDigits) {
        *Dest++ = L'_';
        Suffix->Length += sizeof(L'_');

        Success = (
            AppendLongLongIntegerToUnicodeString(
                Suffix,
                NumberOfTableElements->QuadPart,
                NumberOfElementsDigits,
                L'\0'
            )
        );

        if (!Success) {
            Result = PH_E_STRING_BUFFER_OVERFLOW;
            PH_ERROR(InitializeTableSuffix_AppendIntegerElements, Result);
            goto Error;
        }

        Dest += NumberOfElementsDigits;
    }

    if (AlgorithmName) {
        *Dest++ = L'_';
        Offset = (USHORT)RtlPointerToOffset(Suffix->Buffer, Dest);
        Count = AlgorithmName->Length / sizeof(WCHAR);
        CopyInline(Dest, AlgorithmName->Buffer, AlgorithmName->Length);
        Dest += Count;
    }

    if (HashFunctionName) {
        *Dest++ = L'_';
        Count = HashFunctionName->Length / sizeof(WCHAR);
        CopyInline(Dest, HashFunctionName->Buffer, HashFunctionName->Length);
        Dest += Count;
    }

    if (MaskFunctionName) {
        *Dest++ = L'_';
        Count = MaskFunctionName->Length / sizeof(WCHAR);
        CopyInline(Dest, MaskFunctionName->Buffer, MaskFunctionName->Length);
        Dest += Count;
    }

    if (AdditionalSuffix) {
        *Dest++ = L'_';
        Count = AdditionalSuffix->Length / sizeof(WCHAR);
        CopyInline(Dest, AdditionalSuffix->Buffer, AdditionalSuffix->Length);
        Dest += Count;
    }

    ExpectedDest = (ULONG_PTR)(
        RtlOffsetToPointer(
            Suffix->Buffer,
            TableSuffixLength.LowPart
        )
    );

    if ((ULONG_PTR)Dest != ExpectedDest) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(PerfectHashTableInitializeTableSuffix_DestCheck, Result);
        goto Error;
    }

    Suffix->Length = TableSuffixLength.LowPart;

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

    *AlgorithmOffset = Offset;

    return Result;
}

PERFECT_HASH_TABLE_CREATE_PATH PerfectHashTableCreatePath;

_Use_decl_annotations_
HRESULT
PerfectHashTableCreatePath(
    PPERFECT_HASH_TABLE Table,
    PPERFECT_HASH_PATH ExistingPath,
    PULONG NumberOfResizeEvents,
    PULARGE_INTEGER NumberOfTableElements,
    PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    PCUNICODE_STRING NewDirectory,
    PCUNICODE_STRING NewBaseName,
    PCUNICODE_STRING AdditionalSuffix,
    PCUNICODE_STRING NewExtension,
    PCUNICODE_STRING NewStreamName,
    PPERFECT_HASH_PATH *PathPointer,
    PPERFECT_HASH_PATH_PARTS *PartsPointer
    )
/*++

Routine Description:

    Creates a new path instance from the given parameters.

Arguments:

    Table - Supplies a pointer to the table instance.

    ExistingPath - Supplies a pointer to an existing path to use as a template
        for path creation.

    NumberOfResizeEvents - Optionally supplies the number of table resize
        events that have occurred for this table.

    NumberOfTableElements - Optionally supplies the number of table elements.
        This value will be converted into a base-10 string representation if
        present.

    AlgorithmId - Optionally supplies the algorithm to use.

    HashFunctionId - Optionally supplies the hash function to use.

    MaskFunctionId - Optionally supplies the type of masking to use.

    NewDirectory - Optionally supplies a fully-qualified path to use as the
        directory.

    NewBaseName - Optionally supplies a new base name to use for the table.  If
        NULL, the base name of the existing path will be used.

    AdditionalSuffix - Optionally supplies an additional suffix to add to the
        base name.

    NewExtension - Optionally supplies a new file extension to use.

    NewStreamName - Optionally supplies a new stream name to use.

    Path - Receives a newly created path instance on success.  Caller is
        responsible for releasing the instance when finished with it.

    Parts - Optionally receives the path parts.

Return Value:

    S_OK - Created path successfully.

    E_POINTER - Path or ExistingPath parameters were NULL.

    E_INVALIDARG - One or more parameters were invalid.

    E_OUTOFMEMORY - Out of memory.

    E_UNEXPECTED - Internal error.

--*/
{
    PRTL Rtl;
    HRESULT Result = S_OK;
    USHORT AlgorithmOffset = 0;
    USHORT AdditionalSuffixALength = 0;
    PPERFECT_HASH_PATH Path = NULL;
    PPERFECT_HASH_PATH_PARTS Parts = NULL;
    UNICODE_STRING TableSuffix;
    WCHAR TableSuffixBuffer[TABLE_SUFFIX_BUFFER_SIZE_IN_CHARS];

    if (!ARGUMENT_PRESENT(PathPointer)) {
        return E_POINTER;
    } else {
        *PathPointer = NULL;
    }

    if (ARGUMENT_PRESENT(PartsPointer)) {
        *PartsPointer = NULL;
    }

    if (ARGUMENT_PRESENT(AdditionalSuffix)) {

        //
        // Capture the length in bytes for an ASCII representation of the
        // additional suffix length; the + 1 accounts for the leading '_'
        // that will automatically be added by the suffix initialization
        // routine below.
        //

        AdditionalSuffixALength = (AdditionalSuffix->Length / sizeof(WCHAR))+1;
    }

    Rtl = Table->Rtl;
    ZeroArray(TableSuffixBuffer);

    //
    // Initialize the table suffix.
    //

    TableSuffix.Buffer = (PWSTR)TableSuffixBuffer;
    TableSuffix.Length = 0;
    TableSuffix.MaximumLength = sizeof(TableSuffixBuffer);

    Result = (
        PerfectHashTableInitializeTableSuffix(
            Table,
            &TableSuffix,
            NumberOfResizeEvents,
            NumberOfTableElements,
            AlgorithmId,
            HashFunctionId,
            MaskFunctionId,
            AdditionalSuffix,
            &AlgorithmOffset
        )
    );

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashTableInitializeTableSuffix, Result);
        goto Error;
    }

    //
    // Create a new Path instance.
    //

    Result = Table->Vtbl->CreateInstance(Table,
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
                                NULL,           // DirectorySuffix
                                NewBaseName,
                                &TableSuffix,
                                NewExtension,
                                NewStreamName,
                                &Parts,
                                NULL);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashPathCreate, Result);
        goto Error;
    }

    //
    // If an ASCII base name was successfully extracted, we can update the
    // table name's length now to exclude the additional suffix we may have
    // added above.
    //

    if (Path->BaseNameA.Length > 0) {

        //
        // N.B. AdditionalSuffixALength may be 0 here, which is ok.
        //

        ASSERT(Path->BaseNameA.Buffer);
        ASSERT(Path->TableNameA.Buffer);
        ASSERT(Path->BaseNameA.Buffer == Path->TableNameA.Buffer);

        ASSERT(Path->TableNameA.Length != 0);
        ASSERT(Path->TableNameA.Length <= Path->TableNameA.MaximumLength);
        ASSERT(Path->TableNameA.Length > AdditionalSuffixALength);

        Path->TableNameA.Length -= AdditionalSuffixALength;
        Path->TableNameA.MaximumLength = Path->TableNameA.Length;

        //
        // As above, but for the uppercase variants.
        //

        ASSERT(Path->BaseNameUpperA.Buffer);
        ASSERT(Path->TableNameUpperA.Buffer);
        ASSERT(Path->BaseNameUpperA.Buffer == Path->TableNameUpperA.Buffer);

        ASSERT(Path->TableNameUpperA.Length != 0);
        ASSERT(Path->TableNameUpperA.Length <=
               Path->TableNameUpperA.MaximumLength);
        ASSERT(Path->TableNameUpperA.Length > AdditionalSuffixALength);

        Path->TableNameUpperA.Length -= AdditionalSuffixALength;
        Path->TableNameUpperA.MaximumLength = Path->TableNameUpperA.Length;

        //
        // Convert the algorithm offset into the byte offset of the first
        // character of the algorithm name relative to the base/table name
        // ASCII buffers initialized above.  We shift right to convert from
        // wchar to char, then add the existing path length.
        //

        Path->AdditionalSuffixAOffset = (
            (AlgorithmOffset / sizeof(WCHAR)) + ExistingPath->BaseNameA.Length
        );
    }

    //
    // Update the caller's pointer and return.
    //

    *PathPointer = Path;

    if (ARGUMENT_PRESENT(PartsPointer)) {
        *PartsPointer = Parts;
    }

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
    PPERFECT_HASH_TABLE Table,
    PPERFECT_HASH_FILE *File
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

    Table->TableFile->Vtbl->AddRef(Table->TableFile);
    *File = Table->TableFile;

    ReleasePerfectHashTableLockShared(Table);

    return S_OK;
}

PERFECT_HASH_TABLE_CREATE_VALUES_ARRAY PerfectHashTableCreateValuesArray;

_Use_decl_annotations_
HRESULT
PerfectHashTableCreateValuesArray(
    PPERFECT_HASH_TABLE Table,
    ULONG ValueSizeInBytes
    )
/*++

Routine Description:

    Creates the values array for a given table.

Arguments:

    Table - Supplies a pointer to a table instance for which the values array
        is to be created.

    ValueSizeInBytes - Optionally supplies a custom size, in bytes, of an
        individual value element.  Currently, only sizeof(ULONG) is supported.

Return Value:

    S_OK - Success.

    E_POINTER - Table was NULL.

    E_OUTOFMEMORY - Out of memory.

    PH_E_INVALID_VALUE_SIZE - Invalid value size.

--*/
{
    PRTL Rtl;
    PVOID BaseAddress;
    HRESULT Result = S_OK;
    ULONGLONG ArrayAllocSize;
    BOOLEAN LargePagesForValues;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    if (IsTableCreateOnly(Table)) {
        PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
    }

    if (ValueSizeInBytes != 0) {
        if (ValueSizeInBytes != sizeof(ULONG)) {
            return PH_E_INVALID_VALUE_SIZE;
        }
    } else {
        ValueSizeInBytes = sizeof(ULONG);
    }

    //
    // The values array base address should be NULL at this point.
    //

    if (Table->ValuesBaseAddress) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(PerfectHashTableCreateValuesArray_ValuesBaseAddress, Result);
        goto Error;
    }

    //
    // Initialize aliases.
    //

    Rtl = Table->Rtl;

    //
    // Allocate an array for the table values (i.e. the things stored when the
    // Insert(Key, Value) routine is called).  The dimensions will be the same
    // as the number of table elements * key size, and can be indexed directly
    // by the result of the Index() routine.
    //

    LargePagesForValues = (
        Table->TableCreateFlags.TryLargePagesForValuesArray == TRUE
    );

    ArrayAllocSize = (
        Table->TableInfoOnDisk->NumberOfTableElements.QuadPart *
        (ULONGLONG)ValueSizeInBytes
    );

    BaseAddress = Rtl->Vtbl->TryLargePageVirtualAlloc(Rtl,
                                                      NULL,
                                                      ArrayAllocSize,
                                                      MEM_RESERVE | MEM_COMMIT,
                                                      PAGE_READWRITE,
                                                      &LargePagesForValues);

    Table->ValuesBaseAddress = BaseAddress;
    Table->ValuesArraySizeInBytes = ArrayAllocSize;

    if (!BaseAddress) {
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    //
    // Update flags with large page result for values array.
    //

    Table->Flags.ValuesArrayUsesLargePages = LargePagesForValues;

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

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
