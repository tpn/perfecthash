/*++

Copyright (c) 2020 Trent Nelson <trent@trent.me>

Module Name:

    Graph.cuh

Abstract:

    CUDA graph implementation.

--*/

#pragma once

#define MAX_NUMBER_OF_SEEDS 8

#define IsEmpty(Value) ((ULONG)Value == EMPTY)
#define IsNeighborEmpty(Neighbor) ((ULONG)Neighbor == EMPTY)

//
// When a solution has been found and the assignment step begins, the initial
// value assigned to a vertex is govered by the following macro.
//

#define INITIAL_ASSIGNMENT_VALUE 0

//
// Define helper macros for referring to seed constants stored in local
// variables by their uppercase names.  This allows easy copy-and-pasting of
// the algorithm "guts" between the "compiled" perfect hash table routines in
// ../CompiledPerfectHashTable and the SeededHashEx() implementations here.
//

#define SEED1 Seed1
#define SEED2 Seed2
#define SEED3 Seed3
#define SEED4 Seed4
#define SEED5 Seed5
#define SEED6 Seed6
#define SEED7 Seed7
#define SEED8 Seed8

#define SEED3_BYTE1 Seed3.Byte1
#define SEED3_BYTE2 Seed3.Byte2
#define SEED3_BYTE3 Seed3.Byte3
#define SEED3_BYTE4 Seed3.Byte4

#define SEED6_BYTE1 Seed6.Byte1
#define SEED6_BYTE2 Seed6.Byte2
#define SEED6_BYTE3 Seed6.Byte3
#define SEED6_BYTE4 Seed6.Byte4


//
// Helper defines.
//

#ifndef NOTHING
#define NOTHING
#endif

#define BREAKPOINT __brkpt
#define TRAP __trap

#define ASSERT(Condition)                     \
    if (!(Condition)) {                       \
        asm("trap;");                         \
        Result = PH_E_INVARIANT_CHECK_FAILED; \
        goto End;                             \
    }

#define CU_RESULT cudaError_t
#define CU_STREAM cudaStream_t
#define CU_EVENT cudaEvent_t

typedef CU_STREAM *PCU_STREAM;
typedef CU_EVENT *PCU_EVENT;

#define FindBestGraph(Graph) ((Graph)->Flags.FindBestGraph != FALSE)
#define FirstSolvedGraphWins(Graph) ((Graph)->Flags.FindBestGraph == FALSE)

#define PH_ERROR(Name, Result)           \
    PerfectHashPrintError(#Name,         \
                          __FILE__,      \
                          __LINE__,      \
                          (ULONG)Result)


#define CU_ERROR(Name, CuResult)      \
    PerfectHashPrintCuError(#Name,    \
                            __FILE__, \
                            __LINE__, \
                            CuResult)

#define CU_CHECK(CuResult, Name)                   \
    if (CU_FAILED(CuResult)) {                     \
        CU_ERROR(Name, CuResult);                  \
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED; \
        goto Error;                                \
    }

#define CU_MEMSET(Buffer, Value, Size, Stream)               \
    CuResult = cudaMemsetAsync(Buffer, Value, Size, Stream); \
    CU_CHECK(CuResult, cudaMemsetAsync)

#define CU_ZERO(Buffer, Size, Stream) \
    CU_MEMSET(Buffer, 0, Size, Stream)


#define CREATE_STREAM(Target)                                            \
    CuResult = cudaStreamCreateWithFlags(Target, cudaStreamNonBlocking); \
    CU_CHECK(CuResult, cudaStreamCreateWithFlags)

typedef struct _CU_KERNEL_STREAMS {
    union {
        CU_STREAM Reset;
        CU_STREAM FirstStream;
    };

    CU_STREAM IsAcyclic;
    CU_STREAM SortVertexPairs;
    CU_STREAM Assign;
    CU_STREAM CalculateMemoryCoverage;

    union {
        CU_STREAM Solve;
        CU_STREAM LastStream;
    };
} CU_KERNEL_STREAMS;
typedef CU_KERNEL_STREAMS *PCU_KERNEL_STREAMS;

typedef struct _CU_KERNEL_CONTEXT {
    CU_KERNEL_STREAMS Streams;

    union {
        curandStatePhilox4_32_10_t Philox4;
    } RngState;

} CU_KERNEL_CONTEXT;
typedef CU_KERNEL_CONTEXT *PCU_KERNEL_CONTEXT;

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab filetype=cuda formatoptions=croql   :
