/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashPath.c

Abstract:

    This is the module for the PERFECT_HASH_PATH component of the perfect
    hash table library.  Routines are provided for initialization, rundown,
    extracting parts (internal), copying, creating, resetting, and getting
    parts of a path (public).

--*/

#include "stdafx.h"

PERFECT_HASH_PATH_INITIALIZE PerfectHashPathInitialize;

_Use_decl_annotations_
HRESULT
PerfectHashPathInitialize(
    PPERFECT_HASH_PATH Path
    )
/*++

Routine Description:

    Initializes a perfect hash path structure.  This is a relatively simple
    method that just primes the COM scaffolding; the bulk of the work is done
    when copying or creating a path.

Arguments:

    Path - Supplies a pointer to a PERFECT_HASH_PATH structure for which
        initialization is to be performed.

Return Value:

    S_OK - Success.

    E_POINTER - Path is NULL.

    E_UNEXPECTED - All other errors.

--*/
{
    HRESULT Result = S_OK;

    if (!ARGUMENT_PRESENT(Path)) {
        return E_POINTER;
    }

    Path->SizeOfStruct = sizeof(*Path);

    //
    // Create Rtl and Allocator components.
    //

    Result = Path->Vtbl->CreateInstance(Path,
                                        NULL,
                                        &IID_PERFECT_HASH_RTL,
                                        &Path->Rtl);

    if (FAILED(Result)) {
        goto Error;
    }

    Result = Path->Vtbl->CreateInstance(Path,
                                        NULL,
                                        &IID_PERFECT_HASH_ALLOCATOR,
                                        &Path->Allocator);

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

PERFECT_HASH_PATH_RESET PerfectHashPathReset;

_Use_decl_annotations_
HRESULT
PerfectHashPathReset(
    PPERFECT_HASH_PATH Path
    )
/*++

Routine Description:

    Resets a path instance that has been initialized via Copy() or Create().
    This frees the underlying string buffer used for FullPath.Buffer, and zeros
    all UNICODE_STRING structures for the path parts.

    Requires the exclusive path lock to be held.

Arguments:

    Path - Supplies a pointer to a PERFECT_HASH_PATH structure for which the
        reset is to be performed.

Return Value:

    S_OK - Path reset successfully.

    E_POINTER - Path was NULL.

--*/
{
    PRTL Rtl;
    PALLOCATOR Allocator;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Path)) {
        return E_POINTER;
    }

    //
    // Initialize aliases.
    //

    Rtl = Path->Rtl;
    Allocator = Path->Allocator;

    //
    // Free the full path's buffer.
    //

    Allocator = Path->Allocator;
    Allocator->Vtbl->FreeUnicodeStringBuffer(Allocator, &Path->FullPath);

    //
    // Zero the remaining string structures.
    //

    ZeroStruct(Path->Parts);

    //
    // Clear the state and flags.
    //

    ClearPathState(Path);
    ClearPathFlags(Path);

    //
    // Indicate success and return.
    //

    return S_OK;
}

PERFECT_HASH_PATH_RUNDOWN PerfectHashPathRundown;

_Use_decl_annotations_
VOID
PerfectHashPathRundown(
    PPERFECT_HASH_PATH Path
    )
/*++

Routine Description:

    Release all resources associated with a perfect hash file.

Arguments:

    Path - Supplies a pointer to a PERFECT_HASH_PATH structure for which
        rundown is to be performed.

Return Value:

    None.

--*/
{
    HRESULT Result;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Path)) {
        return;
    }

    AcquirePerfectHashPathLockExclusive(Path);

    //
    // Sanity check structure size.
    //

    ASSERT(Path->SizeOfStruct == sizeof(*Path));

    //
    // Reset the path.
    //

    Result = Path->Vtbl->Reset(Path);
    if (FAILED(Result)) {
        PH_ERROR(PerfectHashPathReset, Result);
    }

    //
    // Release COM references.
    //

    RELEASE(Path->Allocator);
    RELEASE(Path->Rtl);

    //
    // Release lock and return.
    //

    ReleasePerfectHashPathLockExclusive(Path);

    return;
}

PERFECT_HASH_PATH_EXTRACT_PARTS PerfectHashPathExtractParts;

_Use_decl_annotations_
HRESULT
PerfectHashPathExtractParts(
    PPERFECT_HASH_PATH Path
    )
/*++

Routine Description:

    Extracts all relevant path parts from Path->FullPath and fills in their
    associated UNICODE_STRING structures.  Requires Path->Lock to be held
    exclusively.

Arguments:

    Path - Supplies the path instance for which parts are to be extracted.

Return Value:

    S_OK - Parts extracted success.

    E_POINTER - Path was NULL.

    PH_E_PATH_PARTS_EXTRACTION_FAILED - Failed to extract the path into parts.

--*/
{
    PRTL Rtl;
    USHORT Index;
    USHORT Count;
    PWSTR Start;
    PWSTR End;
    PWSTR Char;
    WCHAR Wide;
    HRESULT Result = S_OK;
    BOOLEAN Found = FALSE;
    BOOLEAN Valid;
    HRESULT ResetResult;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Path)) {
        return E_POINTER;
    }

    //
    // Argument validation complete.
    //

    //
    // Initialize aliases.
    //

    Rtl = Path->Rtl;

    ASSERT(IsValidMinimumDirectoryNullTerminatedUnicodeString(&Path->FullPath));

    Start = Path->FullPath.Buffer;
    End = (PWSTR)RtlOffsetToPointer(Start, Path->FullPath.Length);

    //
    // Reverse through the string and find the first dot.
    //

    Count = Path->FullPath.Length >> 1;

    for (Index = 0, Char = End; Index < Count; Index++, Char--) {
        if (*Char == L'.') {
            Found = TRUE;
            break;
        }
    }

    if (!Found) {
        goto Error;
    }

    ASSERT(*Char == L'.');
    Char++;
    Path->Extension.Buffer = Char;
    Path->Extension.Length = (USHORT)RtlPointerToOffset(End, Char);
    Path->Extension.MaximumLength = Path->Extension.Length;

    //
    // Advance through the string and try see if there's a colon, indicating
    // an NTFS stream.
    //

    Found = FALSE;

    while (Char != End) {
        if (*Char == L':') {
            Found = TRUE;
            break;
        }
        Char++;
    }

    ASSERT(!Path->StreamName.Buffer);

    if (Found) {
        ASSERT(*Char == L':');
        Char++;
        Path->StreamName.Buffer = Char;
        Path->StreamName.Length = (USHORT)RtlPointerToOffset(End, Char);
        Path->StreamName.MaximumLength = Path->StreamName.Length;
        ASSERT(Path->StreamName.Buffer[Path->StreamName.Length >> 1] == L'\0');

        //
        // Remove the stream name from the extension.
        //

        Path->Extension.Length -= Path->StreamName.Length;
        Path->Extension.MaximumLength = Path->Extension.Length;
        ASSERT(Path->Extension.Buffer[Path->Extension.Length >> 1] == L':');
    }

    //
    // Reset the Char pointer back to the dot, then reverse backward looking
    // for the first directory slash.
    //

    Found = FALSE;
    Char = Path->Extension.Buffer;
    while (Char != Start) {
        if (*Char == L'\\') {
            Found = TRUE;
            break;
        }
        Char++;
    }

    if (!Found) {
        goto Error;
    }

    //
    // Fill out the base name, file name and directory strings now that we know
    // where the first directory separator occurs.
    //

    ASSERT(*Char == L'\\');
    Char++;
    Path->BaseName.Buffer = Char;
    Path->BaseName.Length = (USHORT)(
        RtlPointerToOffset(
            Path->Extension.Buffer,
            Path->BaseName.Buffer
        )
    );
    Path->BaseName.MaximumLength = Path->BaseName.Length;
    ASSERT(Path->BaseName.Buffer[Path->BaseName.Length >> 1] == L'.');

    Path->FileName.Buffer = Char;
    Path->FileName.Length = (USHORT)(
        RtlPointerToOffset(
            Path->FileName.Buffer,
            End
        )
    );
    Path->FileName.MaximumLength = Path->FileName.Length;
    if (Path->StreamName.Buffer) {
        ASSERT(Path->FileName.Buffer[Path->FileName.Length >> 1] == L':');
    } else {
        ASSERT(Path->FileName.Buffer[Path->FileName.Length >> 1] == L'\0');
    }

    Path->Directory.Buffer = Start;
    Path->Directory.Length = (USHORT)(
        RtlPointerToOffset(
            Path->Directory.Buffer,
            Path->FileName.Buffer
        )
    ) - sizeof(WCHAR);
    Path->Directory.MaximumLength = Path->Directory.Length;
    ASSERT(Path->Directory.Buffer[Path->Directory.Length >> 1] == L'\\');

    //
    // Extract the drive depending on where the first colon lives.
    //

    if (Path->Directory.Buffer[1] == L':') {

        Path->Drive.Length = sizeof(WCHAR);
        Path->Drive.MaximumLength = Path->Drive.Length;
        Path->Drive.Buffer = Start;

    } else {

        //
        // N.B. This won't work for network drives.
        //

        Found = FALSE;
        Char = Start;
        while (Char != Path->FileName.Buffer) {
            if (*Char == L':') {
                Found = TRUE;
                break;
            }
            Char++;
        }

        ASSERT(!Path->Drive.Buffer);

        if (Found) {
            ASSERT(*Char == L':');
            Char--;
            Path->Drive.Buffer = Char;
            Path->Drive.Length = sizeof(WCHAR);
            Path->Drive.MaximumLength = Path->Drive.Length;
        }
    }

    //
    // Verify the base name is a valid C identifier.  (Handle the first char
    // separately, ensuring it doesn't start with a number.)
    //

    Char = Path->BaseName.Buffer;

    Wide = *Char++;

    Valid = (
        Wide == L'_' || (
            (Wide >= L'a' && Wide <= L'z') ||
            (Wide >= L'A' && Wide <= L'Z')
        )
    );

    if (Valid) {

        //
        // Copy the first character, then verify the remaining characters in the
        // base name are valid for a C identifier.
        //

        Path->BaseNameA.Buffer[0] = (CHAR)Wide;

        Count = (Path->BaseName.Length >> 1) - 1;
        ASSERT(Path->BaseName.Buffer[Count] != L'.');
        ASSERT(Path->BaseName.Buffer[Count + 1] == L'.');

        for (Index = 1; Index < Count; Index++) {

            Wide = *Char++;

            //
            // Replace hyphens and spaces with underscores.
            //

            if (Wide == L'-' || Wide == L' ') {
                Wide = L'_';
            }

            Valid = (
                Wide == L'_' || (
                    (Wide >= L'a' && Wide <= L'z') ||
                    (Wide >= L'A' && Wide <= L'Z') ||
                    (Wide >= L'0' && Wide <= L'9')
                )
            );

            if (!Valid) {
                break;
            }

            //
            // Copy a char representation into the buffer.
            //

            Path->BaseNameA.Buffer[Index] = (CHAR)Wide;
        }
    }

    if (!Valid) {

        //
        // Base name was not a valid C identifier.  Clear the BaseNameA
        // representation.
        //

        ZeroMemory(Path->BaseNameA.Buffer, Path->BaseNameA.MaximumLength);
        ZeroStruct(Path->BaseNameA);

    } else {

        //
        // If we get here, we've found a valid C identifier in the file's base
        // name.  Toggle the relevant flag and NULL-terminate the ASCII
        // representation.
        //

        Path->Flags.BaseNameIsValidCIdentifier = TRUE;

        ASSERT(Path->BaseNameA.Length + 1 <= Path->BaseNameA.MaximumLength);
        Path->BaseNameA.Buffer[Path->BaseNameA.Length + 1] = '\0';
    }

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = PH_E_PATH_PARTS_EXTRACTION_FAILED;
    }

    //
    // Clear all intermediate paths we may have extracted.
    //

    ResetResult = Path->Vtbl->Reset(Path);
    if (FAILED(ResetResult)) {
        PH_ERROR(PerfectHashPathReset, ResetResult);
    }

    //
    // Intentional follow-on to End.
    //

End:

    ReleasePerfectHashPathLockExclusive(Path);

    return Result;
}

PERFECT_HASH_PATH_COPY PerfectHashPathCopy;

_Use_decl_annotations_
HRESULT
PerfectHashPathCopy(
    PPERFECT_HASH_PATH Path,
    PCUNICODE_STRING Source,
    PCPERFECT_HASH_PATH_PARTS *Parts,
    PVOID Reserved
    )
/*++

Routine Description:

    Constructs a new path instance by deep-copying the given source path.

Arguments:

    Path - Supplies the path instance.

    Source - Supplies a pointer to the source path to copy.

    Parts - Optionally receives a pointer to the individual path parts
        structure if the routine was successful.

Return Value:

    S_OK - Path copied successfully.

    E_POINTER - One or more parameters were NULL.

    E_OUTOFMEMORY - Out of memory.

    E_INVALIDARG - Source path is invalid.

    PH_E_PATH_PARTS_EXTRACTION_FAILED - Failed to extract the path into parts.

--*/
{
    PRTL Rtl;
    PVOID BaseAddress;
    HRESULT ResetResult;
    HRESULT Result = S_OK;
    PALLOCATOR Allocator;

    UNREFERENCED_PARAMETER(Reserved);

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Path)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Source)) {
        return E_POINTER;
    }

    if (!IsValidMinimumDirectoryNullTerminatedUnicodeString(Source)) {
        return E_INVALIDARG;
    }

    if (!TryAcquirePerfectHashPathLockExclusive(Path)) {
        return PH_E_PATH_LOCKED;
    }

    if (IsPathSet(Path)) {
        ReleasePerfectHashPathLockExclusive(Path);
        return PH_E_PATH_ALREADY_SET;
    }

    //
    // Argument validation complete.
    //

    //
    // Initialize aliases.
    //

    Rtl = Path->Rtl;
    Allocator = Path->Allocator;

    BaseAddress = Allocator->Vtbl->Calloc(Allocator,
                                          1,
                                          Source->MaximumLength);

    if (!BaseAddress) {
        SYS_ERROR(VirtualAlloc);
        return E_OUTOFMEMORY;
    }

    Path->FullPath.Buffer = (PWSTR)BaseAddress;
    Path->FullPath.Length = Source->Length;
    Path->FullPath.MaximumLength = Source->MaximumLength;

    CopyMemory(Path->FullPath.Buffer,
               Source->Buffer,
               Path->FullPath.Length);

    Result = Path->Vtbl->ExtractParts(Path);
    if (FAILED(Result)) {
        PH_ERROR(PerfectHashPathExtractParts, Result);
        goto Error;
    }

    //
    // Update the caller's pointer if applicable.
    //

    if (ARGUMENT_PRESENT(Parts)) {
        *Parts = &Path->Parts;
    }

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    ResetResult = Path->Vtbl->Reset(Path);
    if (FAILED(ResetResult)) {
        PH_ERROR(PerfectHashPathReset, ResetResult);
    }

    //
    // Intentional follow-on to End.
    //

End:

    ReleasePerfectHashPathLockExclusive(Path);

    return Result;
}

PERFECT_HASH_PATH_CREATE PerfectHashPathCreate;

_Use_decl_annotations_
HRESULT
PerfectHashPathCreate(
    PPERFECT_HASH_PATH Path,
    PPERFECT_HASH_PATH ExistingPath,
    PCUNICODE_STRING NewDirectory,
    PCUNICODE_STRING NewBaseName,
    PCUNICODE_STRING BaseNameSuffix,
    PCUNICODE_STRING NewExtension,
    PCUNICODE_STRING NewStreamName,
    PCPERFECT_HASH_PATH_PARTS *Parts,
    PVOID Reserved
    )
/*++

Routine Description:

    Creates a new path structure, using an existing path as a base template,
    optionally replacing the directory, adding a base name suffix, changing
    the extension, and adding a stream name.

Arguments:

    Path - Supplies the path instance.

    ExistingPath - Supplies a pointer to an existing path to use as a template.

    NewDirectory - Optionally supplies a new directory.

    NewBaseName - Optionally supplies a new base name.

    BaseNameSuffix - Optionally supplies a new base name suffix to append to
        NewBaseName (if non-NULL), otherwise the existing path's base name.

    NewExtension - Optionally supplies a new extension to use for the file.

    NewStreamName - Optionally supplies a new NTFS stream name to use for the
        file.

    Parts - Optionally receives a pointer to the individual path parts
        structure if the routine was successful.

Return Value:

    S_OK - Path created successfully.

    E_POINTER - Path or ExistingPath were NULL pointers.

    E_UNEXPECTED - Internal error.

    E_OUTOFMEMORY - Out of memory.

    E_INVALIDARG - One or more parameters were invalid.

    PH_E_PATH_ALREADY_SET - The provided path instance has already had a path
        set via Copy() or Create().

    PH_E_PATH_LOCKED - Path is locked.

    PH_E_EXISTING_PATH_LOCKED - ExistingPath is locked.

    PH_E_EXISTING_PATH_NO_PATH_SET - The provided existing path instance did
        not have a path set.

    PH_E_PATH_PARTS_EXTRACTION_FAILED - Failed to extract the path into parts.

    PH_E_STRING_BUFFER_OVERFLOW - The new path string buffer exceeded limits.

    PH_E_INVARIANT_CHECK_FAILED - An internal invariant check failed.

--*/
{
    PRTL Rtl;
    USHORT Count;
    USHORT FullPathLength;
    USHORT FullPathMaximumLength;
    PWSTR Dest;
    PVOID BaseAddress;
    PCHAR ExpectedDest;
    HRESULT ResetResult;
    HRESULT Result = S_OK;
    BOOLEAN HasStream = FALSE;
    PALLOCATOR Allocator;
    PCUNICODE_STRING Source;
    ULONG_INTEGER AllocSize = { 0 };
    ULONG_INTEGER AlignedAllocSize;
    ULONG_INTEGER BaseNameALength;
    ULONG_INTEGER BaseNameAMaximumLength;

    UNREFERENCED_PARAMETER(Reserved);

    //
    // Validate arguments and calculate the string buffer allocation size where
    // applicable.
    //

    if (!ARGUMENT_PRESENT(Path)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(ExistingPath)) {
        return E_POINTER;
    }

    if (!TryAcquirePerfectHashPathLockShared(ExistingPath)) {
        return PH_E_EXISTING_PATH_LOCKED;
    }

    if (!IsPathSet(ExistingPath)) {
        ReleasePerfectHashPathLockShared(ExistingPath);
        return PH_E_EXISTING_PATH_NO_PATH_SET;
    }

    if (!TryAcquirePerfectHashPathLockExclusive(Path)) {
        ReleasePerfectHashPathLockShared(ExistingPath);
        return PH_E_PATH_LOCKED;
    }

    //
    // N.B. Both locks have been acquired at this stage.  Error handling code
    //      should jump to the 'Error:' block herein to ensure both locks are
    //      released correctly.
    //

    if (IsPathSet(Path)) {
        Result = PH_E_PATH_ALREADY_SET;
        goto Error;
    }

    if (ARGUMENT_PRESENT(NewDirectory)) {
        if (!IsValidMinimumDirectoryNullTerminatedUnicodeString(NewDirectory)) {
            Result = E_INVALIDARG;
            goto Error;
        } else {
            AllocSize.LongPart = NewDirectory->Length;
        }
    } else {
        AllocSize.LongPart = ExistingPath->Directory.Length;
    }

    if (ARGUMENT_PRESENT(NewBaseName)) {
        if (!IsValidUnicodeString(NewBaseName)) {
            Result = E_INVALIDARG;
            goto Error;
        }
        AllocSize.LongPart += NewBaseName->Length;
        BaseNameALength.LongPart = NewBaseName->Length >> 1;
    } else {
        AllocSize.LongPart += ExistingPath->BaseName.Length;
        BaseNameALength.LongPart = ExistingPath->BaseName.Length >> 1;
    }

    if (ARGUMENT_PRESENT(BaseNameSuffix)) {
        if (!IsValidUnicodeString(BaseNameSuffix)) {
            Result = E_INVALIDARG;
            goto Error;
        } else {
            AllocSize.LongPart += BaseNameSuffix->Length;
            BaseNameALength.LongPart += BaseNameSuffix->Length >> 1;
        }
    }

    if (ARGUMENT_PRESENT(NewExtension)) {
        if (!IsValidUnicodeString(NewExtension)) {
            Result = E_INVALIDARG;
            goto Error;
        } else {
            AllocSize.LongPart += NewExtension->Length;
        }
    } else {
        AllocSize.LongPart += ExistingPath->Extension.Length;
    }

    //
    // Account for the period preceeding the extension, which isn't included
    // in the Extension.Length part.
    //

    AllocSize.LongPart += sizeof(WCHAR);

    //
    // Continue with argument validation.
    //

    if (ARGUMENT_PRESENT(NewStreamName)) {
        if (!IsValidUnicodeString(NewStreamName)) {
            Result = E_INVALIDARG;
            goto Error;
        } else {
            HasStream = TRUE;
            AllocSize.LongPart += NewStreamName->Length;
        }
    } else if (IsValidUnicodeString(&ExistingPath->StreamName)) {
        HasStream = TRUE;
        AllocSize.LongPart += ExistingPath->StreamName.Length;
    }

    //
    // If a stream is present, account for the colon.
    //

    if (HasStream) {
        AllocSize.LongPart += sizeof(WCHAR);
    }

    //
    // Account for the trailing NULL and verify we haven't overflowed USHORT.
    //

    AllocSize.LongPart += sizeof(WCHAR);

    if (AllocSize.HighPart) {
        Result = PH_E_STRING_BUFFER_OVERFLOW;
        goto Error;
    }

    //
    // Align up to an 8-byte boundary and verify we haven't overflowed USHORT.
    //

    AlignedAllocSize.LongPart = ALIGN_UP(AllocSize.LongPart, 8);

    if (AlignedAllocSize.HighPart) {
        Result = PH_E_STRING_BUFFER_OVERFLOW;
        goto Error;
    }

    //
    // Capture the lengths associated with the full path at this point, prior
    // to factoring in space required for the ASCII representation of the base
    // name.
    //
    // N.B. FullPathLength includes the trailing NULL.
    //

    FullPathLength = AllocSize.LowPart;
    FullPathMaximumLength = AlignedAllocSize.LowPart;

    //
    // Align the base name ASCII representation length up to an 8-byte boundary
    // and verify we haven't overflowed.  We add 1 to account for the trailing
    // NULL (which *isn't* included in BaseNameA.Length).
    //

    BaseNameAMaximumLength.LongPart = ALIGN_UP(BaseNameALength.LongPart + 1, 8);
    if (!BaseNameAMaximumLength.HighPart) {
        Result = PH_E_STRING_BUFFER_OVERFLOW;
        goto Error;
    }

    //
    // Add the base name buffer length into the total aligned alloc length.
    //

    AlignedAllocSize.LongPart += BaseNameAMaximumLength.LowPart;

    //
    // Argument validation complete.  Initialize aliases then attempt to
    // allocate a string buffer of sufficient size.
    //

    Rtl = Path->Rtl;
    Allocator = Path->Allocator;

    BaseAddress = Allocator->Vtbl->Calloc(Allocator,
                                          1,
                                          AlignedAllocSize.LongPart);

    if (!BaseAddress) {
        SYS_ERROR(VirtualAlloc);
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    //
    // Buffer allocation was successful.  Proceed with path construction based
    // on the arguments we validated earlier.
    //

    Path->FullPath.Buffer = (PWSTR)BaseAddress;
    Path->FullPath.Length = FullPathLength;
    Path->FullPath.MaximumLength = FullPathMaximumLength;

    Dest = Path->FullPath.Buffer;

    if (ARGUMENT_PRESENT(NewDirectory)) {
        Source = NewDirectory;
    } else {
        Source = &ExistingPath->Directory;
    }

    Count = Source->Length >> 1;

    //
    // Copy the directory.
    //

    CopyMemory(Dest, Source->Buffer, Source->Length);
    Dest += Count;
    ASSERT(*Dest == L'\0');
    ASSERT(*(Dest - 1) != L'\\');

    *Dest++ = L'\\';

    //
    // Copy the base name.
    //

    if (ARGUMENT_PRESENT(NewBaseName)) {
        Source = NewBaseName;
    } else {
        Source = &ExistingPath->BaseName;
    }

    Count = Source->Length >> 1;

    CopyMemory(Dest, Source->Buffer, Source->Length);
    Dest += Count;
    ASSERT(*Dest == L'\0');
    ASSERT(*(Dest - 1) == L'\\');

    //
    // Copy the base name suffix if applicable.
    //

    if (ARGUMENT_PRESENT(BaseNameSuffix)) {
        Source = BaseNameSuffix;
        Count = Source->Length >> 1;
        CopyMemory(Dest, Source->Buffer, Source->Length);
        Dest += Count;
        ASSERT(*Dest == L'\0');
    }

    //
    // Write the period prior to the extension.
    //

    *Dest++ = L'.';

    if (ARGUMENT_PRESENT(NewExtension)) {
        Source = NewExtension;
    } else {
        Source = &ExistingPath->Extension;
    }

    Count = Source->Length >> 1;
    CopyMemory(Dest, Source->Buffer, Source->Length);
    Dest += Count;
    ASSERT(*Dest == L'\0');

    //
    // Copy the NTFS stream name if applicable.
    //

    if (HasStream) {

        if (ARGUMENT_PRESENT(NewStreamName)) {
            Source = NewStreamName;
        } else {
            Source = &ExistingPath->StreamName;
        }

        *Dest++ = L':';

        Count = Source->Length >> 1;

        CopyMemory(Dest, Source->Buffer, Source->Length);
        Dest += Count;
    }

    //
    // Write the terminating NULL.
    //

    *Dest++ = L'\0';

    //
    // Verify the Dest pointer matches where we expect it to.
    //

    ExpectedDest = RtlOffsetToPointer(BaseAddress, AllocSize.LowPart);
    if ((ULONG_PTR)Dest != (ULONG_PTR)ExpectedDest) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        goto Error;
    }

    //
    // Align the destination pointer up to an 8-byte boundary and verify it
    // matches the expected aligned alloc size.
    //

    Dest = (PWSTR)ALIGN_UP(Dest, 8);
    ExpectedDest = RtlOffsetToPointer(BaseAddress, AlignedAllocSize.LongPart);
    if ((ULONG_PTR)Dest != (ULONG_PTR)ExpectedDest) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        goto Error;
    }

    //
    // Initialize the BaseNameA string to point to the area after the full path
    // string buffer.
    //

    Path->BaseNameA.Buffer = (PSTR)Dest;
    Path->BaseNameA.Length = BaseNameALength.LowPart;
    Path->BaseNameA.MaximumLength = BaseNameALength.LowPart;

    //
    // We've finished constructing our new path's FullPath.Buffer.  Final step
    // is to extract the individual parts into the relevant UNICODE_STRING
    // structures.
    //

    Result = Path->Vtbl->ExtractParts(Path);
    if (FAILED(Result)) {
        PH_ERROR(PerfectHashPathExtractParts, Result);
        goto Error;
    }

    //
    // Update the caller's pointer if applicable.
    //

    if (ARGUMENT_PRESENT(Parts)) {
        *Parts = &Path->Parts;
    }

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    ResetResult = Path->Vtbl->Reset(Path);
    if (FAILED(ResetResult)) {
        PH_ERROR(PerfectHashPathReset, ResetResult);
    }

    //
    // Intentional follow-on to End.
    //

End:

    ReleasePerfectHashPathLockShared(ExistingPath);
    ReleasePerfectHashPathLockExclusive(Path);

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
