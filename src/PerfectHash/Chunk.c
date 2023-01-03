/*++

Copyright (c) 2018-2023. Trent Nelson <trent@trent.me>

Module Name:

    Chunk.c

Abstract:

    This module implements the ProcessChunks() routine.

--*/

#include "stdafx.h"

PROCESS_CHUNKS ProcessChunks;

_Use_decl_annotations_
HRESULT
ProcessChunks(
    PRTL Rtl,
    PCCHUNK Chunks,
    ULONG NumberOfChunks,
    PCCHUNK_VALUES Values,
    ULONG NumberOfConditionals,
    PBOOLEAN Conditionals,
    PCHAR *BufferPointer
    )
/*++

Routine Description:

    Processes an array of chunks and writes the output to the given string
    buffer.

Arguments:

    Rtl - Supplies a pointer to an initialized RTL instance.

    Chunks - Supplies an array of chunks to process.

    NumberOfChunks - Supplies the number of chunks in the Chunks array.

    Values - Supplies the chunk values to use for substitution as applicable.

    NumberOfConditionals - Supplies the number of conditional chunk ops in the
        array of chunks.

    Conditionals - Supplies the array of boolean conditions that dictate if
        a conditional chunk is written.

    BufferPointer - Supplies a pointer to the buffer that will receive the
        processed chunk strings.  This pointer will also be updated to point
        at the new end of buffer after this routine completes successfully.

Return Value:

    S_OK - Chunks processed successfully.

    E_POINTER - One or more pointer parameters were NULL.

    E_INVALIDARG - NumberOfChunks was 0 or *BufferPointer was NULL.

    PH_E_INVALID_CHUNK_OP - Invalid chunk op encountered.

--*/
{
    ULONG Index;
    CHUNK_OP NewOp;
    ULONG NumberOfConditionalsSeen = 0;
    PCCHUNK Chunk;
    HRESULT Result = S_OK;
    PCSTRING String;
    PCHAR Output;

    UNREFERENCED_PARAMETER(Rtl);

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Chunks)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Values)) {
        return E_POINTER;
    }

    if (NumberOfChunks == 0) {
        return E_INVALIDARG;
    }

    if (!ARGUMENT_PRESENT(BufferPointer)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(*BufferPointer)) {
        return E_INVALIDARG;
    }

    //
    // Argument validation complete.
    //

    Output = *BufferPointer;

    for (Index = 0, Chunk = Chunks; Index < NumberOfChunks; Index++, Chunk++) {

        if (!IsValidChunkOp(Chunk->Op)) {
            Result = PH_E_INVALID_CHUNK_OP;
            goto Error;
        }

        if (Chunk->Op == ChunkOpRaw) {
            String = &Chunk->RawString;
        } else if (Chunk->Op == ChunkOpStringPointer) {
            String = Chunk->StringPointer;
        } else if (IsConditionalChunkOp(Chunk->Op, &NewOp)) {
            if (++NumberOfConditionalsSeen > NumberOfConditionals) {
                Result = PH_E_INVALID_NUMBER_OF_CONDITIONALS;
                goto Error;
            }
            if (Conditionals[NumberOfConditionalsSeen-1] != FALSE) {
                if (NewOp == ChunkOpRaw) {
                    String = &Chunk->RawString;
                } else if (Chunk->Op == ChunkOpStringPointer) {
                    String = Chunk->StringPointer;
                } else {
                    String = *(&Values->First + NewOp);
                }
            } else {
                continue;
            }
        } else {
            String = *(&Values->First + Chunk->Op);
        }

        if (IsEmptyString(String)) {
            continue;
        }

        if (!IsValidString(String)) {
            Result = PH_E_INVALID_CHUNK_STRING;
            goto Error;
        }

        CopyMemoryInline(Output, String->Buffer, String->Length);

        Output += String->Length;
    }

    if (NumberOfConditionals != NumberOfConditionalsSeen) {
        Result = PH_E_NUMBER_OF_CONDITIONALS_MISMATCHED;
        goto Error;
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

    *BufferPointer = Output;

    return Result;
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
