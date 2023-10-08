/*++

Copyright (c) 2020 Trent Nelson <trent@trent.me>

Module Name:

    Graph.cuh

Abstract:

    CUDA graph implementation.

--*/

#pragma once

#include <limits>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

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

#define CUDA_CALL(F)                                   \
    if ((F) != cudaSuccess) {                          \
        printf("Error %s at %s:%d\n",                  \
               cudaGetErrorString(cudaGetLastError()), \
               __FILE__,                               \
               __LINE__-1);                            \
        exit(-1);                                      \
    }

#define CUDA_CHECK()                                   \
    if ((cudaPeekAtLastError()) != cudaSuccess) {      \
        printf("Error %s at %s:%d\n",                  \
               cudaGetErrorString(cudaGetLastError()), \
               __FILE__,                               \
               __LINE__-1);                            \
        exit(-1);                                      \
    }


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

template <typename Type>
struct ATOMIC_VERTEX {
    cuda::std::atomic<Type> Degree;
    cuda::std::atomic<Type> Edges;
};

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
                ResultType,
                KeyType,
                VertexType>;

        case PerfectHashHashMultiplyShiftRFunctionId:
            return PerfectHashTableSeededHashExCppMultiplyShiftR<
                ResultType,
                KeyType,
                VertexType>;

        default:
            return PerfectHashTableSeededHashExCppNull<
                ResultType,
                KeyType,
                VertexType>;
    }
}

template<
    typename KeyTypeT,
    typename VertexTypeT,
    typename VertexPairTypeT,
    typename Edge3TypeT,
    typename Vertex3TypeT,
    typename AtomicVertex3TypeT,
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
    using Edge3Type = Edge3TypeT;
    using Vertex3Type = Vertex3TypeT;
    using AtomicVertex3Type = AtomicVertex3TypeT;
    using EdgeType = EdgeTypeT;
    using OrderType = OrderTypeT;
    using AssignedType = AssignedTypeT;
    using IndexType = EdgeTypeT;
    using ValueType = ValueTypeT;
    using CountType = CountTypeT;
    using AtomicOrderType = cuda::std::atomic<OrderType>;
    //using MaxCount = std::integral_constant<CountTypeT, MaxCountT>;

    __host__ __device__ __forceinline__
    AtomicOrderType*
    GetAtomicOrderIndex() { return (AtomicOrderType*)&OrderIndex; }
};

using GRAPH8 =
    GRAPH_CU<
        uint32_t,               // KeyType
        VERTEX8,                // VertexType
        VERTEX_PAIR_CU8,        // VertexPairType
        EDGE83,                 // Edge3Type
        VERTEX83,               // Vertex3Type
        ATOMIC_VERTEX<EDGE8>,   // AtomicVertex3Type
        EDGE8,                  // EdgeType
        ORDER8,                 // OrderType
        ASSIGNED8,              // AssignedType
        int8_t,                 // ValueType
        uint8_t>;               // CountType
        //0xf>;                 // MaxCount
using PGRAPH8 = GRAPH8*;

using GRAPH16 =
    GRAPH_CU<
        uint32_t,              // KeyType
        VERTEX16,              // VertexType
        VERTEX_PAIR_CU16,      // VertexPairType
        EDGE163,               // Edge3Type
        VERTEX163,             // Vertex3Type
        ATOMIC_VERTEX<EDGE16>, // AtomicVertex3Type
        EDGE16,                // EdgeType
        ORDER16,               // OrderType
        ASSIGNED16,            // AssignedType
        int16_t,               // ValueType
        uint8_t>;              // CountType
        //0xf>;                // MaxCount
using PGRAPH16 = GRAPH16*;

using GRAPH32 =
    GRAPH_CU<
        uint32_t,              // KeyType
        VERTEX,                // VertexType
        VERTEX_PAIR_CU32,      // VertexPairType
        EDGE3,                 // Edge3Type
        VERTEX3,               // Vertex3Type
        ATOMIC_VERTEX<EDGE>,   // AtomicVertex3Type
        EDGE,                  // EdgeType
        ORDER,                 // OrderType
        ASSIGNED,              // AssignedType
        int32_t,               // ValueType
        uint8_t>;              // CountType
        //0xf>;                // MaxCount
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

struct LOCK {
    cuda::std::atomic<bool> Value;

    __device__ __forceinline__
    void
    Lock() {
        while (Value.exchange(true, cuda::std::memory_order_acquire)) {
            // Spin.
        }
    }

    __device__ __forceinline__
    void
    Unlock() {
        Value.store(false, cuda::std::memory_order_release);
    }
};
typedef LOCK *PLOCK;

template <typename T>
__forceinline__
__device__
T
AtomicAggIncCG(T *Address)
{
    T Prev;
    cg::coalesced_group Group = cg::coalesced_threads();
    if (Group.thread_rank() == 0) {
        auto Size = Group.size();
        Prev = atomicAdd((decltype(Size) *)Address, Size);
    }
    Prev = Group.thread_rank() + Group.shfl(Prev, 0);
    return Prev;
}

template <typename T>
__forceinline__
__device__
T
AtomicAggSubCG(T *Address)
{
    T Prev;
    cg::coalesced_group Group = cg::coalesced_threads();
    if (Group.thread_rank() == 0) {
        auto Size = Group.size();
        Prev = atomicSub((decltype(Size) *)Address, Size);
    }
    Prev = Group.thread_rank() - Group.shfl(Prev, 0);
    return Prev;
}

template <typename T>
__forceinline__
__device__
void
AtomicAggIncCGV(T *Address)
{
    cg::coalesced_group Group = cg::coalesced_threads();
    if (Group.thread_rank() == 0) {
        auto Size = Group.size();
        atomicAdd((decltype(Size) *)Address, Size);
    }
    return;
}

template <typename T>
__forceinline__
__device__
void
AtomicAggSubCGV(T *Address)
{
    cg::coalesced_group Group = cg::coalesced_threads();
    if (Group.thread_rank() == 0) {
        auto Size = Group.size();
        atomicSub((decltype(Size) *)Address, Size);
    }
    return;
}

template <typename T>
__device__
bool
TestBit(
    const uint32_t* Address,
    T Bit
    )
{
    const T Index = static_cast<T>(Bit / (sizeof(T) * 8));
    const T Shift = static_cast<T>(Bit & ((sizeof(T) * 8) - 1));
    return ((Address[Index] >> Shift) & 1);
}

template <typename T>
__device__
void
SetBit(
    const uint32_t* Address,
    T Bit
    )
{
    const T Index = static_cast<T>(Bit / (sizeof(T) * 8));
    const T Shift = static_cast<T>(Bit & ((sizeof(T) * 8) - 1));
    Address[Index] |= (static_cast<T>(1) << Shift);
}

template <typename T>
__device__
T
AtomicSetBit(
    uint32_t* Address,
    T Bit)
{
    static_assert(
        sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
        "T must be either uint16_t or uint32_t or uint64_t"
    );

    const T Index = static_cast<T>(Bit / (sizeof(T) * 8));
    const T Shift = static_cast<T>(Bit & ((sizeof(T) * 8) - 1));
    const T Value = (static_cast<T>(1) << Shift);

    using AtomicType = typename std::conditional<
        sizeof(T) == 2,
        uint32_t,
        typename std::conditional<
            sizeof(T) == 4,
            uint32_t,
            uint64_t
        >::type
    >::type;

    T* Target = reinterpret_cast<T*>(&Address[Index * sizeof(T)]);

    if ((((ptrdiff_t)Target) & 0x3) != 0) {
        printf("Target is not 4-byte aligned: %p\n", Target);
        return 0;
    }

    return static_cast<T>(
        atomicOr(reinterpret_cast<AtomicType*>(Target),
                 static_cast<AtomicType>(Value))
    );
}

template<uint8_t MaxCount = std::numeric_limits<uint8_t>::max()>
__forceinline__
__device__
uint8_t
AtomicAggIncMultiCG(uint8_t *BaseAddress, uint8_t Offset)
{
    uint32_t PrevLong;
    uint32_t CompLong;
    uint32_t NextLong;
    uint8_t PrevByte;
    uint8_t NextByte;
    uint8_t Remaining;
    uint32_t constexpr ByteSelection[] = { 0x3214, 0x3240, 0x3410, 0x4210 };
    const uint8_t Index = Offset & 0x3;
    const uint8_t Bucket = Offset >> 2;
    const uint32_t Selector = ByteSelection[Index];
    uint32_t *Base = reinterpret_cast<uint32_t*>(BaseAddress);
    uint32_t *Address = Base + Bucket;
    ULONG_BYTES PrevBytes;

    cg::coalesced_group Group = cg::coalesced_threads();
    auto LabeledGroup = cg::labeled_partition(Group, Offset);
    if (LabeledGroup.thread_rank() == 0) {
        do {
            PrevLong = PrevBytes.AsULong = *Address;
            PrevByte = PrevBytes.Bytes[Index];
            Remaining = MaxCount - PrevByte;
            if (Remaining == 0) {

                //
                // The counter has already been saturated, nothing more to do.
                //

                NextByte = PrevByte;
                break;

            } else if (Remaining <= LabeledGroup.size()) {

                //
                // The counter will be saturated after this increment.
                //

                NextByte = MaxCount;

            } else {

                //
                // We have sufficient room left in the counter to add the
                // group size in its entirety without overflowing.
                //

                NextByte = PrevByte + LabeledGroup.size();
            }

            NextLong = __byte_perm(PrevLong, NextByte, Selector);
            CompLong = __byte_perm(PrevLong, PrevByte, Selector);
            if (CompLong != PrevLong) {
                printf("CompLong != PrevLong: %08x != %08x\n", CompLong, PrevLong);
                continue;
            }

            PrevBytes.AsULong = atomicCAS(Address, PrevLong, NextLong);

        } while (PrevBytes.AsULong != PrevLong);
    }

    return NextByte;
}

template <typename T>
__device__
constexpr uint8_t CalculateShift() {
    if constexpr (sizeof(T) == 1) {
        return 2;
    } else if constexpr (sizeof(T) == 2) {
        return 1;
    } else if constexpr (sizeof(T) == 4) {
        return 0;
    } else {
        static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4,
                      "Unsupported size for OffsetType");
        return 0;
    }
}


template<typename OffsetType,
         typename CountType = uint8_t,
         typename AtomicType = uint32_t,
         CountType MaxCount = std::numeric_limits<CountType>::max()>
__forceinline__
__device__
typename std::enable_if<
    std::disjunction<
        std::is_same<OffsetType, uint8_t>,
        std::is_same<OffsetType, uint16_t>,
        std::is_same<OffsetType, uint32_t>,
    >::value,
    CountType
>::type
AtomicAggIncMultiCGEx(CountType *BaseAddress, OffsetType Offset)
{
    AtomicType PrevAtomic;
    AtomicType NextAtomic;
    CountType PrevCount;
    CountType NextCount;
    CountType Remaining;
    const CountType constexpr CountMask = ~0;
    const AtomicType constexpr Selection[] = { 0x3214, 0x3240, 0x3410, 0x4210 };
    const AtomicType Mask = ~0x3;
    const AtomicType Bucket = ((Offset >> 2) & Mask);
    const uint8_t Index = static_cast<uint8_t>(Offset & 0x3);
    const uint32_t Selector = Selection[Index];
    uintptr_t Base = reinterpret_cast<uintptr_t>(BaseAddress);
    Base += Offset * sizeof(CountType);
    Base &= ~0x3;
    AtomicType *Address = reinterpret_cast<AtomicType*>(Base);

    cg::coalesced_group Group = cg::coalesced_threads();
    auto LabeledGroup = cg::labeled_partition(Group, Offset);
    if (LabeledGroup.thread_rank() == 0) {
        do {
            PrevAtomic = *Address;
            PrevCount = (PrevAtomic >> (Index * 8)) & CountMask;
            Remaining = MaxCount - PrevCount;
            if (Remaining == 0) {

                //
                // The counter has already been saturated, nothing more to do.
                //

                NextCount = PrevCount;
                break;

            } else if (Remaining <= LabeledGroup.size()) {

                //
                // The counter will be saturated after this increment.
                //

                NextCount = MaxCount;

            } else {

                //
                // We have sufficient room left in the counter to add the
                // group size in its entirety without overflowing.
                //

                NextCount = PrevCount + LabeledGroup.size();
            }

            NextAtomic = __byte_perm(PrevAtomic, NextCount, Selector);

        } while (atomicCAS(Address, PrevAtomic, NextAtomic) != PrevAtomic);
    }

    NextCount = LabeledGroup.shfl(NextCount, 0) - (LabeledGroup.size() - 1);
    Remaining = MaxCount - NextCount;
    if (Remaining == 0) {

        //
        // The counter has already been saturated, nothing more to do.
        //

        NOTHING;

    } else if (Remaining <= LabeledGroup.thread_rank()) {

        //
        // The counter will be saturated after this increment.
        //

        NextCount = MaxCount;

    } else {

        //
        // We have sufficient room left in the counter to add this thread rank
        // without overflowing.
        //

        NextCount = NextCount + LabeledGroup.thread_rank();
    }

    return NextCount;
}

template<typename OffsetType,
         typename ValueType = uint8_t,
         typename AtomicType = uint32_t>
__forceinline__
__device__
typename std::enable_if<
    std::disjunction<
        std::is_same<OffsetType, uint8_t>,
        std::is_same<OffsetType, uint16_t>,
        std::is_same<OffsetType, uint32_t>,
    >::value,
    void
>::type
AtomicAggXORMultiCGEx(
    ValueType *BaseAddress,
    OffsetType Offset,
    ValueType ValueToXOR)
{
    AtomicType PrevAtomic;
    AtomicType NextAtomic;
    ValueType PrevValue;
    const ValueType constexpr ValueMask = ~0;
    const AtomicType constexpr Selection[] = { 0x3214, 0x3240, 0x3410, 0x4210 };
    const AtomicType Mask = ~0x3;
    const AtomicType Bucket = ((Offset >> 2) & Mask);
    const uint8_t Index = static_cast<uint8_t>(Offset & 0x3);
    const uint32_t Selector = Selection[Index];
    uintptr_t Base = reinterpret_cast<uintptr_t>(BaseAddress);
    Base += Offset * sizeof(ValueType);
    Base &= ~0x3;
    AtomicType *Address = reinterpret_cast<AtomicType*>(Base);

    cg::coalesced_group Group = cg::coalesced_threads();
    auto LabeledGroup = cg::labeled_partition(Group, Offset);

    //
    // XOR reduction across all threads in the labeled group.
    //

    ValueType CumulativeXOR =
        LabeledGroup.reduce(ValueToXOR, cg::bit_xor<ValueType>());

    if (LabeledGroup.thread_rank() == 0) {

        //
        // Atomically XOR the cumulative XOR value with the current value.
        //

        do {
            PrevAtomic = *Address;
            PrevValue = (PrevAtomic >> (Index * 8)) & ValueMask;
            PrevValue ^= CumulativeXOR;
            NextAtomic = __byte_perm(PrevAtomic, PrevValue, Selector);
        } while (atomicCAS(Address, PrevAtomic, NextAtomic) != PrevAtomic);
    }
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab filetype=cuda formatoptions=croql   :
