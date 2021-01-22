/*++

Copyright (c) 2018-2020 Trent Nelson <trent@trent.me>

Module Name:

    RtlRandom.c

Abstract:

    This module implements functionality related to random data.  Routines are
    provided for generating random bytes, and creating random object names.

--*/

#include "stdafx.h"
#include "PerfectHashEventsPrivate.h"

RTL_GENERATE_RANDOM_BYTES RtlGenerateRandomBytes;

_Use_decl_annotations_
HRESULT
RtlGenerateRandomBytes(
    PRTL Rtl,
    ULONG SizeOfBufferInBytes,
    PBYTE Buffer
    )
/*++

Routine Description:

    This routine writes random bytes into a given buffer using the system's
    random data generation facilities.

Arguments:

    Rtl - Supplies a pointer to an RTL instance.

    SizeOfBufferInBytes - Supplies the size of the Buffer parameter, in bytes.

    Buffer - Supplies the address for which random bytes will be written.

Return Value:

    S_OK - Success.

    E_POINTER - Rtl or Buffer is NULL.

    E_INVALIDARG - SizeOfBufferInBytes is 0.

    PH_E_FAILED_TO_GENERATE_RANDOM_BYTES - System routine failed to generate
        random bytes.

--*/
{
    HRESULT Result;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Rtl)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Buffer)) {
        return E_POINTER;
    }

    if (SizeOfBufferInBytes == 0) {
        return E_INVALIDARG;
    }

    //
    // Argument validation complete.  Continue.
    //

    EVENT_WRITE_RTL_RANDOM_BYTES_START(SizeOfBufferInBytes);

    ASSERT(Rtl->CryptProv != 0);
    if (!CryptGenRandom(Rtl->CryptProv, SizeOfBufferInBytes, Buffer)) {
        SYS_ERROR(CryptGenRandom);
        Result = PH_E_FAILED_TO_GENERATE_RANDOM_BYTES;
    } else {
        Result = S_OK;
    }

    EVENT_WRITE_RTL_RANDOM_BYTES_STOP(SizeOfBufferInBytes, Result);

    return Result;
}


RTL_CREATE_RANDOM_OBJECT_NAMES RtlCreateRandomObjectNames;

_Use_decl_annotations_
HRESULT
RtlCreateRandomObjectNames(
    PRTL Rtl,
    PALLOCATOR TemporaryAllocator,
    PALLOCATOR WideBufferAllocator,
    USHORT NumberOfNames,
    USHORT LengthOfNameInChars,
    PUNICODE_STRING NamespacePrefix,
    PPUNICODE_STRING NamesArrayPointer,
    PPUNICODE_STRING PrefixArrayPointer,
    PULONG SizeOfWideBufferInBytes,
    PWSTR *WideBufferPointer
    )
/*++

Routine Description:

    This routine writes Base64-encoded random data to an existing buffer in a
    format suitable for subsequent use of UNICODE_STRING-based system names for
    things like events or shared memory handles.

Arguments:

    Rtl - Supplies a pointer to an initialized RTL structure.

    TemporaryAllocator - Supplies a pointer to an initialized ALLOCATOR struct
        that this routine will use for temporary allocations.  (Any temporarily
        allocated memory will be freed before the routine returns, regardless
        of success/error.)

    WideBufferAllocator - Supplies a pointer to an initialized ALLOCATOR struct
        that this routine will use to allocate the final wide character buffer
        that contains the base64-encoded random data.  This data will then have
        the object namespace+prefix and trailing NULL characters overlaid on
        top of it.  (That is, the UNICODE_STRING structures pointed to by the
        NamesArray will have their Buffer addresses point within this buffer
        space.)  The caller is responsible for freeing this address (which will
        be received via the output parameter WideBufferPointer).

    NumberOfNames - Supplies the number of names that will be carved out of the
        provided WideBuffer by the caller.  This parameter is used in concert
        with the LengthOfNameInChars parameter to ensure the buffer is laid out
        in the correct format.

    LengthOfNameInChars - Supplies the desired length of each name string in
        characters.  This length is assumed to include the trailing NULL and
        the prefix -- that is, the space required to contain the prefix and
        trailing NULL will be subtracted from this parameter.  For optimal
        layout, this parameter should be a power of 2 -- with 64 and 128 being
        good default values.

    NamespacePrefix - Optionally supplies a pointer to a UNICODE_STRING to
        use as the namespace (prefix) for each string.  If NULL, this value
        defaults L"Local\\".  (If L"Global\\" is used, the caller is responsible
        for ensuring the SeCreateGlobalPrivilege privilege is enabled.)

    NamesArrayPointer - Supplies a pointer to the first element of an array of
        of pointers to UNICODE_STRING structures that will be filled out with
        the details of the corresponding object name.  Sufficient space should
        be allocated such that the array contains sizeof(UNICODE_STRING) *
        NumberOfNames in space.

    PrefixArrayPointer - Optionally supplies a pointer to the first element of
        an array of pointers to UNICODE_STRING structures which can be used to
        further customize the name of the object after the namespace but before
        the random character data.  If a NULL pointer resides at a given array
        element, it is assumed no prefix is desired for this element.

    SizeOfWideBufferInBytes - Receives the size in bytes of the buffer allocated
        to store the object names' wide character data.

    WideBufferPointer - Receives the base address of the object names wide char
        buffer.


Return Value:

    S_OK on success, an appropriate error code on error.

--*/
{
    BOOL Success;
    USHORT Index;
    USHORT Count;
    HRESULT Result = S_OK;
    LONG PrefixLengthInChars;
    LONG NumberOfWideBase64CharsToCopy;
    LONG CharsRemaining;
    LONG CharsUsed;
    LONG RandomCharsUsed;
    LONG FinalCharCount;
    ULONG CryptFlags;
    ULONG SizeOfBinaryBufferInBytes;
    ULONG SizeOfWideBase64BufferInBytes = 0;
    ULONG OldLengthOfWideBase64BufferInChars;
    ULONG LengthOfWideBase64BufferInChars;
    PBYTE BinaryBuffer = NULL;
    PWCHAR Dest;
    PWCHAR WideBase64Buffer = NULL;
    PUNICODE_STRING String;
    PUNICODE_STRING Prefix;
    PPUNICODE_STRING Prefixes;
    PUNICODE_STRING Namespace;
    UNICODE_STRING LocalNamespace = RTL_CONSTANT_STRING(L"Local\\");

    //
    // Validate arguments.
    //

    if (ARGUMENT_PRESENT(NamespacePrefix)) {
        if (!IsValidUnicodeStringWithMinimumLengthInChars(NamespacePrefix,7)) {
            return E_INVALIDARG;
        }
        Namespace = NamespacePrefix;
    } else {
        Namespace = &LocalNamespace;
    }

    //
    // Namespace length should be (far) less than the desired name length.
    //

    if (Namespace->Length >= (LengthOfNameInChars << 1)) {
        return E_UNEXPECTED;
    }

    //
    // If the namespace ends with a trailing NULL, omit it by reducing the
    // length by one wide character.  Then, verify the final character is a
    // slash.
    //

    if (Namespace->Buffer[(Namespace->Length >> 1) - 1] == L'\0') {
        Namespace->Length -= sizeof(WCHAR);
    }

    if (Namespace->Buffer[(Namespace->Length >> 1) - 1] != L'\\') {
        return E_UNEXPECTED;
    }

    if (ARGUMENT_PRESENT(PrefixArrayPointer)) {
        Prefixes = PrefixArrayPointer;
    } else {
        Prefixes = NULL;
    }

    //
    // Allocate a buffer for the initial binary data; we generate more random
    // data than we need here, but it's easier than trying to get everything
    // exact up-front (i.e. base64->binary size conversions).
    //
    // N.B. We use Allocator->Vtbl->Malloc() instead of the Calloc() here as the
    //      existing memory data will contribute as a seed value.
    //

    SizeOfBinaryBufferInBytes = NumberOfNames * LengthOfNameInChars;

    BinaryBuffer = (PBYTE)(
        TemporaryAllocator->Vtbl->Malloc(
            TemporaryAllocator,
            SizeOfBinaryBufferInBytes
        )
    );

    if (!BinaryBuffer) {
        SYS_ERROR(HeapAlloc);
        return E_OUTOFMEMORY;
    }

    //
    // Allocate a wide character buffer for the base64-encoded binary data that
    // is double the length of the binary buffer -- this is simpler than trying
    // to get the exact conversion right.
    //

    SizeOfWideBase64BufferInBytes = ALIGN_UP_POINTER(
        (ULONG_PTR)NumberOfNames *
        (ULONG_PTR)LengthOfNameInChars *
        sizeof(WCHAR) *
        2
    );

    WideBase64Buffer = (PWCHAR)(
        WideBufferAllocator->Vtbl->Calloc(
            WideBufferAllocator,
            1,
            SizeOfWideBase64BufferInBytes
        )
    );

    if (!WideBase64Buffer) {
        SYS_ERROR(HeapAlloc);
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    //
    // We successfully allocated space for our two buffers.  Fill the first one
    // with random data now.
    //

    Result = Rtl->Vtbl->GenerateRandomBytes(Rtl,
                                            SizeOfBinaryBufferInBytes,
                                            (PBYTE)BinaryBuffer);
    if (FAILED(Result)) {
        SYS_ERROR(CryptGenRandom);
        goto Error;
    }

    //
    // Convert the entire binary data buffer into base64-encoded wide character
    // representation.  We calculate the number of wide characters the buffer
    // can receive by shifting the byte size right by one, then copy that value
    // into a second variable that CryptBinaryToStringW() can overwrite with
    // the actual length converted.
    //

    OldLengthOfWideBase64BufferInChars = SizeOfWideBase64BufferInBytes >> 1;
    LengthOfWideBase64BufferInChars = OldLengthOfWideBase64BufferInChars;

    CryptFlags = CRYPT_STRING_BASE64 | CRYPT_STRING_NOCRLF;
    Success = Rtl->CryptBinaryToStringW(BinaryBuffer,
                                        SizeOfBinaryBufferInBytes,
                                        CryptFlags,
                                        WideBase64Buffer,
                                        &LengthOfWideBase64BufferInChars);

    if (!Success) {
        SYS_ERROR(CryptBinaryToStringW);
        goto Error;
    }

    //
    // Conversion of the binary data into base64-encoded data was successful,
    // so we can free the binary buffer now.
    //

    TemporaryAllocator->Vtbl->FreePointer(TemporaryAllocator, &BinaryBuffer);

    //
    // Loop through the array of pointers to UNICODE_STRING structures and fill
    // each one out, adding the namespace and prefixes accordingly.
    //

    RandomCharsUsed = 0;

    for (Index = 0; Index < NumberOfNames; Index++) {

        //
        // Resolve the next unicode string pointer.
        //

        String = *(NamesArrayPointer + Index);

        //
        // Reset counters.  CharsUsed has one subtracted from it in addition
        // to the namespace length to account for the trailing NULL.
        //

        CharsUsed = (Namespace->Length >> 1) + 1;
        CharsRemaining = (LONG)LengthOfNameInChars - CharsUsed;

        if (Prefixes && ((Prefix = *(Prefixes + Index)) != NULL)) {

            //
            // Omit any trailing NULLs from the custom prefix provided by the
            // caller, then subtract the prefix length from the remaining bytes
            // and verify we've got a sensible number left.
            //

            PrefixLengthInChars = Prefix->Length >> 1;
            if (Prefix->Buffer[PrefixLengthInChars - 1] == L'\0') {
                PrefixLengthInChars -= 1;
            }

            CharsUsed += PrefixLengthInChars;
            CharsRemaining -= PrefixLengthInChars;
        } else {
            Prefix = NULL;
            PrefixLengthInChars = 0;
        }

        if (CharsRemaining <= 0) {
            Result = PH_E_CREATE_RANDOM_OBJECT_NAMES_LENGTH_OF_NAME_TOO_SHORT;
            PH_ERROR(RtlCreateRandomObjectNames_CharsRemaining, Result);
            goto Error;
        }

        //
        // Final sanity check that the lengths add up.
        //

        NumberOfWideBase64CharsToCopy = CharsRemaining;

        FinalCharCount = (
            (Namespace->Length >> 1) +
            PrefixLengthInChars +
            NumberOfWideBase64CharsToCopy +
            1
        );

        if (FinalCharCount != LengthOfNameInChars) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(RtlCreateRandomObjectNames_FinalCharCount, Result);
            goto Error;
        }

        //
        // Everything checks out, fill out the unicode string details and copy
        // the relevant parts over: namespace, optional prefix and then finally
        // the random characters.
        //

        String->Length = (USHORT)(FinalCharCount << 1) - (USHORT)sizeof(WCHAR);
        String->MaximumLength = String->Length + sizeof(WCHAR);

        Dest = String->Buffer = (WideBase64Buffer + RandomCharsUsed);

        //
        // Copy the namespace and optionally prefix into the initial part of
        // the random character data buffer, then NULL terminate the name and
        // update counters.
        //

        Count = Namespace->Length >> 1;
        CopyMemory(Dest, Namespace->Buffer, Namespace->Length);
        Dest += Count;
        RandomCharsUsed += Count;

        if (Prefix) {
            Count = (USHORT)PrefixLengthInChars;
            CopyMemory(Dest, Prefix->Buffer, Prefix->Length);
            Dest += Count;
            RandomCharsUsed += Count;
        }

        Count = (USHORT)NumberOfWideBase64CharsToCopy + 1;
        Dest += Count - 1;
        *Dest = L'\0';
        RandomCharsUsed += Count;
    }

    //
    // We're done, indicate success and finish up.
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

    if (BinaryBuffer) {
        TemporaryAllocator->Vtbl->FreePointer(
            TemporaryAllocator,
            &BinaryBuffer
        );
    }

    *WideBufferPointer = WideBase64Buffer;
    *SizeOfWideBufferInBytes = SizeOfWideBase64BufferInBytes;

    return Result;
}

_Use_decl_annotations_
HRESULT
RtlCreateSingleRandomObjectName(
    PRTL Rtl,
    PALLOCATOR TemporaryAllocator,
    PALLOCATOR WideBufferAllocator,
    PCUNICODE_STRING Prefix,
    PUNICODE_STRING Name
    )
/*++

Routine Description:

    This is a convenience routine that simplifies the task of creating a single,
    optionally-prefixed, random object name.  Behind the scenes, it calls the
    procedure Rtl->Vtbl->CreateRandomObjectNames().

Arguments:

    Rtl - Supplies a pointer to an initialized RTL structure.

    TemporaryAllocator - Supplies a pointer to an initialized ALLOCATOR struct
        that this routine will use for temporary allocations.  (Any temporarily
        allocated memory will be freed before the routine returns, regardless
        of success/error.)

    WideBufferAllocator - Supplies a pointer to an initialized ALLOCATOR struct
        that this routine will use to allocate the final wide character buffer
        that contains the base64-encoded random data.  This data will then have
        the prefix and trailing NULL characters overlaid on top of it.  (That
        is, the UNICODE_STRING structure pointed to by the Name parameter will
        have its Buffer address point within this buffer.)  The caller is
        responsible for freeing this address (Name->Buffer).

    Prefix - Optionally supplies the address of a UNICODE_STRING structure to
        use as the prefix for the object name.  This will be appended after
        the namespace name and before the random base64-encoded data.

    Name - Supplies the address of a UNICODE_STRING structure that will receive
        the details of the newly-created object name.  The caller is responsible
        for freeing the address at Name->Buffer via Allocator.

Return Value:

    S_OK on success, an appropriate error code on error.

--*/
{
    HRESULT Result;
    ULONG SizeOfBuffer = 0;
    PWSTR WideBuffer;
    PUNICODE_STRING Names[1];
    PCUNICODE_STRING Prefixes[1];

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Name)) {
        return E_INVALIDARG;
    }

    //
    // Initialize the arrays.
    //

    Names[0] = Name;
    Prefixes[0] = Prefix;

    Result = Rtl->Vtbl->CreateRandomObjectNames(Rtl,
                                                TemporaryAllocator,
                                                WideBufferAllocator,
                                                1,
                                                64,
                                                NULL,
                                                (PPUNICODE_STRING)&Names,
                                                (PPUNICODE_STRING)&Prefixes,
                                                &SizeOfBuffer,
                                                &WideBuffer);


    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
