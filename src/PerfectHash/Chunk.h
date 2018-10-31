/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Chunk.h

Abstract:

    This is the private header file for the Chunk module of the perfect hash
    library.  Chunk is the generic term we use to describe constructing a text
    output string from an array of "chunks", with support for injecting values
    at runtime as necessary.

--*/

#include "stdafx.h"

//
// Define the type of operations that can be performed on a chunk.
//

typedef enum _CHUNK_OP {
    ChunkOpRaw = 0,
    ChunkOpStringPointer,
    ChunkOpInsertProjectGuid,
    ChunkOpInsertRootNamespace,
    ChunkOpInsertProjectName,
    ChunkOpInsertBaseName,
    ChunkOpInsertTableName,
    ChunkOpInsertTargetName,
    ChunkOpInsertTargetExt,
    ChunkOpInsertConfigurationType,
    ChunkOpInsertFileSuffix,
    ChunkOpInsertTargetPrefix,
    ChunkOpInsertTargetSuffix,
    ChunkOpInvalid,
    ChunkOpLast = ChunkOpInvalid - 1,
} CHUNK_OP;

FORCEINLINE
BOOLEAN
IsValidChunkOp(
    _In_ CHUNK_OP Op
    )
{
    return Op >= ChunkOpRaw && Op <= ChunkOpLast;
}

//
// Define the values used at runtime in corresponding with the matching chunk
// op above.
//

typedef struct _CHUNK_VALUES {

    //
    // We want to be able to index into this structure directly via the chunk
    // op enum, e.g.:
    //
    //      CHUNK Chunk = { ChunkOpInsertProjectName, };
    //      PCSTRING Target;
    //
    //      Target = *(&Values->First + Chunk->Op);
    //      ASSERT(Target == Values->ProjectName);
    //
    // Thus, "dummy" members are added for the initial enum values that don't
    // map to a member in this structure.
    //

    union {
        PCSTRING RawOpDummy;
        PCSTRING First;
    };
    PCSTRING StringPointerDummy;

    PCSTRING ProjectGuid;
    PCSTRING RootNamespace;
    PCSTRING ProjectName;
    PCSTRING BaseName;
    PCSTRING TableName;
    PCSTRING TargetName;
    PCSTRING TargetExt;
    PCSTRING ConfigurationType;
    PCSTRING FileSuffix;
    PCSTRING TargetPrefix;
    PCSTRING TargetSuffix;
} CHUNK_VALUES;
typedef CHUNK_VALUES *PCHUNK_VALUES;
typedef const CHUNK_VALUES *PCCHUNK_VALUES;

//
// Disable warning C4820:
//      '<anonymous-tag>': '4' bytes padding added after data member ...
//

#pragma warning(push)
#pragma warning(disable: 4820)
typedef struct _CHUNK {
    CHUNK_OP Op;
    union {
        STRING RawString;
        PCSTRING StringPointer;
    };
} CHUNK;
typedef CHUNK *PCHUNK;
typedef const CHUNK *PCCHUNK;
#pragma warning(pop)

typedef
_Check_return_
_Success_(return >= 0)
HRESULT
(NTAPI PROCESS_CHUNKS)(
    _In_ PRTL Rtl,
    _In_reads_(NumberOfChunks) PCCHUNK Chunks,
    _In_ ULONG NumberOfChunks,
    _In_ PCCHUNK_VALUES Values,
    _Inout_ PCHAR *BufferPointer
    );
typedef PROCESS_CHUNKS *PPROCESS_CHUNKS;

extern PROCESS_CHUNKS ProcessChunks;

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
