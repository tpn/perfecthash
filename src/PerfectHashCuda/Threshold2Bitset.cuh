/*++

Copyright (c) 2026 Trent Nelson <trent@trent.me>

Module Name:

    Threshold2Bitset.cuh

Abstract:

    CUDA helpers for dense threshold-2 histograms represented by two bitsets.

--*/

#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include <type_traits>

enum THRESHOLD2_BITSET_STATE : uint8_t {
    Threshold2BitsetZero = 0,
    Threshold2BitsetOne = 1,
    Threshold2BitsetMany = 2,
};

template <typename WordType>
struct Threshold2BitsetWordTraits
{
    static_assert(
        std::is_unsigned<WordType>::value,
        "WordType must be an unsigned integer type"
    );

    static_assert(
        sizeof(WordType) == sizeof(uint32_t) ||
        sizeof(WordType) == sizeof(uint64_t),
        "WordType must be a 32-bit or 64-bit unsigned integer type"
    );

    static constexpr uint32_t BitsPerWord = sizeof(WordType) * 8u;
    static constexpr uint32_t WordIndexShift = BitsPerWord == 32u ? 5u : 6u;
    static constexpr uint32_t WordBitMask = BitsPerWord - 1u;
};

template <typename OffsetType>
using Threshold2BitsetUnsignedOffsetType =
    typename std::make_unsigned<
        typename std::remove_cv<OffsetType>::type
    >::type;

template <typename OffsetType>
__host__
__device__
constexpr
Threshold2BitsetUnsignedOffsetType<OffsetType>
Threshold2BitsetNormalizeOffset(
    OffsetType Offset
    )
{
    static_assert(
        std::is_integral<OffsetType>::value,
        "OffsetType must be an integral type"
    );

    static_assert(
        !std::is_same<typename std::remove_cv<OffsetType>::type, bool>::value,
        "OffsetType must not be bool"
    );

    return static_cast<Threshold2BitsetUnsignedOffsetType<OffsetType>>(Offset);
}

template <typename WordType,
          typename CountType>
__host__
__device__
constexpr
size_t
Threshold2BitsetWordCount(
    CountType NumberOfBins
    )
{
    using Traits = Threshold2BitsetWordTraits<WordType>;
    const auto Count = Threshold2BitsetNormalizeOffset(NumberOfBins);
    return static_cast<size_t>(
        (Count + Traits::BitsPerWord - 1u) >> Traits::WordIndexShift
    );
}

template <typename WordType,
          typename OffsetType>
__host__
__device__
constexpr
size_t
Threshold2BitsetWordIndex(
    OffsetType Offset
    )
{
    using Traits = Threshold2BitsetWordTraits<WordType>;
    const auto NormalizedOffset = Threshold2BitsetNormalizeOffset(Offset);
    return static_cast<size_t>(
        NormalizedOffset >> Traits::WordIndexShift
    );
}

template <typename WordType,
          typename OffsetType>
__host__
__device__
constexpr
WordType
Threshold2BitsetMask(
    OffsetType Offset
    )
{
    using Traits = Threshold2BitsetWordTraits<WordType>;
    const auto NormalizedOffset = Threshold2BitsetNormalizeOffset(Offset);
    const uint32_t Shift = static_cast<uint32_t>(
        NormalizedOffset & Traits::WordBitMask
    );
    return static_cast<WordType>(static_cast<WordType>(1u) << Shift);
}

template <typename WordType,
          typename OffsetType>
__host__
__device__
THRESHOLD2_BITSET_STATE
Threshold2BitsetReadState(
    const WordType *Seen,
    const WordType *Many,
    OffsetType Offset
    )
{
    const size_t WordIndex = Threshold2BitsetWordIndex<WordType>(Offset);
    const WordType Mask = Threshold2BitsetMask<WordType>(Offset);

    if ((Many[WordIndex] & Mask) != 0) {
        return Threshold2BitsetMany;
    } else if ((Seen[WordIndex] & Mask) != 0) {
        return Threshold2BitsetOne;
    } else {
        return Threshold2BitsetZero;
    }
}

template <typename WordType>
__device__
__forceinline__
WordType
Threshold2BitsetAtomicOr(
    WordType *Address,
    WordType Mask
    )
{
    static_assert(
        sizeof(WordType) == sizeof(uint32_t) ||
        sizeof(WordType) == sizeof(uint64_t),
        "WordType must be a 32-bit or 64-bit unsigned integer type"
    );

    if constexpr (sizeof(WordType) == sizeof(uint32_t)) {
        return static_cast<WordType>(
            atomicOr(
                reinterpret_cast<unsigned int *>(Address),
                static_cast<unsigned int>(Mask)
            )
        );
    } else {
        return static_cast<WordType>(
            atomicOr(
                reinterpret_cast<unsigned long long *>(Address),
                static_cast<unsigned long long>(Mask)
            )
        );
    }
}

template <typename WordType,
          bool SetSeenWhenMany>
__device__
__forceinline__
THRESHOLD2_BITSET_STATE
Threshold2BitsetMarkMany(
    WordType *Seen,
    WordType *Many,
    WordType Mask
    )
{
    if constexpr (SetSeenWhenMany) {
        Threshold2BitsetAtomicOr(Seen, Mask);
    }

    Threshold2BitsetAtomicOr(Many, Mask);
    return Threshold2BitsetMany;
}

template <typename WordType,
          bool SetSeenWhenMany>
__device__
__forceinline__
THRESHOLD2_BITSET_STATE
Threshold2BitsetMarkOne(
    WordType *Seen,
    WordType *Many,
    WordType Mask
    )
{
    const WordType Previous = Threshold2BitsetAtomicOr(Seen, Mask);
    if ((Previous & Mask) != 0) {
        return Threshold2BitsetMarkMany<WordType, SetSeenWhenMany>(
            Seen,
            Many,
            Mask
        );
    }

    return Threshold2BitsetOne;
}

__device__
__forceinline__
uint32_t
Threshold2BitsetLaneId()
{
    uint32_t LaneId;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(LaneId));
    return LaneId;
}

template <typename OffsetType,
          typename WordType = uint32_t,
          bool SetSeenWhenMany = false,
          bool SkipIfAlreadyMany = true>
__device__
__forceinline__
THRESHOLD2_BITSET_STATE
AtomicAggIncThreshold2Bitset(
    WordType *Seen,
    WordType *Many,
    OffsetType Offset
    )
{
    static_assert(
        sizeof(Threshold2BitsetUnsignedOffsetType<OffsetType>) <= sizeof(uint64_t),
        "OffsetType must fit in 64 bits"
    );

    const size_t WordIndex = Threshold2BitsetWordIndex<WordType>(Offset);
    const WordType Mask = Threshold2BitsetMask<WordType>(Offset);
    WordType *SeenAddress = Seen + WordIndex;
    WordType *ManyAddress = Many + WordIndex;

    if constexpr (SkipIfAlreadyMany) {
        if ((*ManyAddress & Mask) != 0) {
            return Threshold2BitsetMany;
        }
    }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
    return Threshold2BitsetMarkOne<WordType, SetSeenWhenMany>(
        SeenAddress,
        ManyAddress,
        Mask
    );
#else
    using UnsignedOffsetType = Threshold2BitsetUnsignedOffsetType<OffsetType>;
    const UnsignedOffsetType NormalizedOffset =
        Threshold2BitsetNormalizeOffset(Offset);

    if constexpr (sizeof(UnsignedOffsetType) <= sizeof(uint32_t)) {
        const uint32_t ActiveMask = __activemask();
        const uint32_t MatchValue = static_cast<uint32_t>(NormalizedOffset);
        const uint32_t PeerMask = __match_any_sync(ActiveMask, MatchValue);
        const uint32_t LaneId = Threshold2BitsetLaneId();
        const uint32_t LeaderLane = static_cast<uint32_t>(__ffs(PeerMask) - 1);
        const uint32_t PeerCount = static_cast<uint32_t>(__popc(PeerMask));

        if (PeerCount > 1u) {
            if (LaneId == LeaderLane) {
                Threshold2BitsetMarkMany<WordType, SetSeenWhenMany>(
                    SeenAddress,
                    ManyAddress,
                    Mask
                );
            }
            return Threshold2BitsetMany;
        } else {
            return Threshold2BitsetMarkOne<WordType, SetSeenWhenMany>(
                SeenAddress,
                ManyAddress,
                Mask
            );
        }
    } else {
        return Threshold2BitsetMarkOne<WordType, SetSeenWhenMany>(
            SeenAddress,
            ManyAddress,
            Mask
        );
    }
#endif
}
