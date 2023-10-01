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

#undef ASSERT
#define ASSERT(Condition)                     \
    if (!(Condition)) {                       \
        asm("trap;");                         \
        Result = PH_E_INVARIANT_CHECK_FAILED; \
        goto End;                             \
    }


#define CU_RESULT cudaError_t
#define CU_STREAM cudaStream_t
#define CU_EVENT cudaEvent_t

//typedef CU_STREAM *PCU_STREAM;
//typedef CU_EVENT *PCU_EVENT;

#define CU_SUCCEEDED(Result) (Result == CUDA_SUCCESS)
#define CU_FAILED(Result) (Result != CUDA_SUCCESS)

#undef FindBestGraph
#define FindBestGraph(Graph) ((Graph)->Flags.FindBestGraph != FALSE)

#undef FirstSolvedGraphWins
#define FirstSolvedGraphWins(Graph) ((Graph)->Flags.FindBestGraph == FALSE)

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

    CU_STREAM AddKeys;
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

#define PerfectHashPrintCuError(FunctionName, FileName, LineNumber, Error) \
    printf("%s:%d %s failed with error 0x%x: %s: %s.\n",                   \
           FileName,                                                       \
           LineNumber,                                                     \
           FunctionName,                                                   \
           Error,                                                          \
           cudaGetErrorName((CU_RESULT)Error),                             \
           cudaGetErrorString((CU_RESULT)Error))

#define PerfectHashPrintError(FunctionName, FileName, LineNumber, Result) \
    printf("%s:%d %s failed with error 0x%x.\n",                          \
           FileName,                                                      \
           LineNumber,                                                    \
           FunctionName,                                                  \
           Result)

#undef PH_ERROR
#define PH_ERROR(Name, Result)           \
    PerfectHashPrintError(#Name,         \
                          __FILE__,      \
                          __LINE__,      \
                          (ULONG)Result)


#undef CU_ERROR
#define CU_ERROR(Name, CuResult)      \
    PerfectHashPrintCuError(#Name,    \
                            __FILE__, \
                            __LINE__, \
                            CuResult)



//
// Assigned 8
//

typedef BYTE EDGE8;
typedef BYTE ASSIGNED8;
typedef ASSIGNED8 *PASSIGNED8;

typedef BYTE KEY8;
typedef CHAR ORDER8;
typedef BYTE EDGE8;
typedef BYTE VERTEX8;
typedef BYTE DEGREE8;

typedef EDGE8 *PEDGE8;
typedef VERTEX8 *PVERTEX8;
typedef DEGREE8 *PDEGREE8;

//
// 64-bit types.
//

typedef uint64_t KEY64;
typedef uint64_t VERTEX64;
typedef uint64_t DEGREE64;
typedef uint64_t ORDER64;
typedef KEY64 *PKEY64;
typedef uint64_t EDGE64;
typedef EDGE64 *PEDGE64;
typedef VERTEX64 *PVERTEX64;
typedef DEGREE64 *PDEGREE64;


template<typename VertexTypeT, typename PairTypeT>

class VERTEX_PAIR_CU {
public:
    //VERTEX_PAIR_CU() : Vertex1{0}, Vertex2{0} {}

    using VertexType = VertexTypeT;
    using PairType = PairTypeT;
    struct {
        union {
            struct {
                union {
                    VertexType Vertex1;
                    VertexType LowPart;
                };
                union {
                    VertexType Vertex2;
                    VertexType HighPart;
                };
            };
            union {
                PairType AsPair;
                PairType CombinedPart;
            };
        };
    };

    __host__ __device__ __forceinline__
    bool
    operator< (
        const VERTEX_PAIR_CU& Right
        ) const
    {
        if (Vertex1 < Right.Vertex1) {
            return true;
        } else if (Vertex1 == Right.Vertex1) {
            return (Vertex2 < Right.Vertex2);
        } else {
            return false;
        }
    }

    __host__ __device__ __forceinline__
    bool
    operator< (
        const VERTEX_PAIR_CU* Right
        ) const
    {
        return (*this < *Right);
    }

    __host__ __device__ __forceinline__
    bool
    operator== (
        const VERTEX_PAIR_CU& Right
        ) const
    {
        return (Vertex1 == Right.Vertex1) && (Vertex2 == Right.Vertex2);
    }

    __host__ __device__ __forceinline__
    bool
    operator== (
        const VERTEX_PAIR_CU* Right
        ) const
    {
        return (*this == *Right);
    }

};
template<typename VertexType, typename PairType>
using PVERTEX_PAIR_CU = VERTEX_PAIR_CU<VertexType, PairType> *;

using VERTEX_PAIR_CU8 = VERTEX_PAIR_CU<VERTEX8, USHORT>;
using VERTEX_PAIR_CU16 = VERTEX_PAIR_CU<VERTEX16, ULONG>;
using VERTEX_PAIR_CU32 = VERTEX_PAIR_CU<VERTEX, ULONGLONG>;
//using VERTEX_PAIR_CU64 = VERTEX_PAIR_CU<VERTEX64, __int128>;

typedef struct _EDGE83 {
    union {
        struct {
            VERTEX8 Vertex1;
            VERTEX8 Vertex2;
        };
        VERTEX_PAIR_CU8 AsVertex8Pair;
        USHORT AsUShort;
    };
} EDGE83, *PEDGE83;

typedef struct _VERTEX83 {

    //
    // The degree of connections for this vertex.
    //

    DEGREE8 Degree;

    //
    // All edges for this vertex; an incidence list constructed via XOR'ing all
    // edges together (aka "the XOR-trick").
    //

    EDGE8 Edges;

} VERTEX83, *PVERTEX83;


template<typename ResultType, typename KeyType, typename VertexType>
using HashFunctionType = ResultType(*)(KeyType, PULONG Seeds, VertexType);

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
auto
GetHashFunctionForId(
    _In_ PERFECT_HASH_HASH_FUNCTION_ID Id
    )
{
    switch (Id) {
        case PerfectHashHashJenkinsFunctionId:
            return PerfectHashTableSeededHashExCppJenkins<
                typename ResultType,
                typename KeyType,
                typename VertexType>;

        case PerfectHashHashMultiplyShiftRFunctionId:
            return PerfectHashTableSeededHashExCppMultiplyShiftR<
                typename ResultType,
                typename KeyType,
                typename VertexType>;

        default:
            return PerfectHashTableSeededHashExCppNull<
                typename ResultType,
                typename KeyType,
                typename VertexType>;
    }
}

template<
    typename KeyTypeT,
    typename VertexTypeT,
    typename VertexPairTypeT,
    typename Edge3TypeT,
    typename Vertex3Type,
    typename EdgeTypeT,
    typename OrderTypeT,
    typename AssignedTypeT,
    typename ValueTypeT,
    typename CountTypeT//,
    //CountTypeT MaxCountT = std::numeric_limits<CountTypeT>::max()
    >
struct GRAPH_CU : GRAPH {
    using KeyType = KeyTypeT;
    using VertexType = VertexTypeT;
    using VertexPairType = VertexPairTypeT;
    using AssignedType = AssignedTypeT;
    using OrderType = OrderTypeT;
    using EdgeType = EdgeTypeT;
    using IndexType = EdgeTypeT;
    using Edge3Type = Edge3TypeT;
    using ValueType = ValueTypeT;
    using CountType = CountTypeT;
    //using MaxCount = std::integral_constant<CountTypeT, MaxCountT>;
};

using GRAPH8 =
    GRAPH_CU<
        uint32_t,          // KeyType
        VERTEX8,           // VertexType
        VERTEX_PAIR_CU8,   // VertexPairType
        EDGE83,            // Edge3Type
        VERTEX83,          // Vertex3Type
        EDGE8,             // EdgeType
        ORDER8,            // OrderType
        ASSIGNED8,         // AssignedType
        int8_t,            // ValueType
        uint8_t>;          // CountType
        //0xf>;            // MaxCount
using PGRAPH8 = GRAPH8*;

using GRAPH16 =
    GRAPH_CU<
        uint32_t,           // KeyType
        VERTEX16,           // VertexType
        VERTEX_PAIR_CU16,   // VertexPairType
        EDGE163,            // Edge3Type
        VERTEX163,          // Vertex3Type
        EDGE16,             // EdgeType
        ORDER16,            // OrderType
        ASSIGNED16,         // AssignedType
        int16_t,            // ValueType
        uint8_t>;           // CountType
        //0xf>;             // MaxCount
using PGRAPH16 = GRAPH16*;

using GRAPH32 =
    GRAPH_CU<
        uint32_t,           // KeyType
        VERTEX,             // VertexType
        VERTEX_PAIR_CU32,   // VertexPairType
        EDGE3,              // Edge3Type
        VERTEX3,            // Vertex3Type
        EDGE,               // EdgeType
        ORDER,              // OrderType
        ASSIGNED,           // AssignedType
        int32_t,            // ValueType
        uint8_t>;           // CountType
        //0xf>;             // MaxCount
using PGRAPH32 = GRAPH32*;

#if 0
using GRAPH64 =
    GRAPH_CU<
        uint64_t,           // KeyType
        VERTEX,             // VertexType
        VERTEX_PAIR_CU64,   // VertexPairType
        EDGE3,              // Edge3Type
        VERTEX3,            // Vertex3Type
        EDGE,               // EdgeType
        ORDER,              // OrderType
        ASSIGNED,           // AssignedType
        int64_t,            // ValueType
        uint8_t,            // CountType
        0xf>;               // MaxCount
using PGRAPH64 = GRAPH64*;
#endif


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab filetype=cuda formatoptions=croql   :
