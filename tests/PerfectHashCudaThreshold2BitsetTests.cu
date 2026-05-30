/*++

Copyright (c) 2026 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashCudaThreshold2BitsetTests.cu

Abstract:

    Tests for CUDA threshold-2 bitset histogram helpers.

--*/

#include <PerfectHashCuda/Threshold2Bitset.cuh>
#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>

#include <vector>

static_assert(
    Threshold2BitsetWordCount<uint32_t>(0u) == 0u,
    "Zero bins must require zero words"
);

static_assert(
    Threshold2BitsetWordCount<uint32_t>(1u) == 1u,
    "One bin must require one word"
);

static_assert(
    Threshold2BitsetWordCount<uint32_t>(32u) == 1u,
    "Thirty-two bins must require one 32-bit word"
);

static_assert(
    Threshold2BitsetWordCount<uint32_t>(33u) == 2u,
    "Thirty-three bins must require two 32-bit words"
);

static_assert(
    Threshold2BitsetWordCount<uint64_t>(64u) == 1u,
    "Sixty-four bins must require one 64-bit word"
);

static_assert(
    Threshold2BitsetWordCount<uint64_t>(65u) == 2u,
    "Sixty-five bins must require two 64-bit words"
);

#define CUDA_CHECK(Call)                                                     \
    do {                                                                     \
        cudaError_t CudaCheckError = (Call);                                 \
        if (CudaCheckError != cudaSuccess) {                                 \
            fprintf(stderr,                                                   \
                    "%s:%d: CUDA error %s: %s\n",                          \
                    __FILE__,                                                \
                    __LINE__,                                                \
                    #Call,                                                   \
                    cudaGetErrorString(CudaCheckError));                     \
            return 1;                                                        \
        }                                                                    \
    } while (0)

template <typename WordType,
          bool SetSeenWhenMany,
          bool SkipIfAlreadyMany>
__global__
void
Threshold2BitsetKernel(
    const uint32_t *Bins,
    size_t NumberOfInputs,
    WordType *Seen,
    WordType *Many
    )
{
    const size_t ThreadIndex = (
        (static_cast<size_t>(blockIdx.x) * blockDim.x) + threadIdx.x
    );
    const size_t Stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (size_t Index = ThreadIndex;
         Index < NumberOfInputs;
         Index += Stride) {

        AtomicAggIncThreshold2Bitset<
            uint32_t,
            WordType,
            SetSeenWhenMany,
            SkipIfAlreadyMany
        >(Seen, Many, Bins[Index]);
    }
}

template <typename WordType>
THRESHOLD2_BITSET_STATE
ExpectedState(
    const std::vector<uint32_t>& Counts,
    uint32_t Bin
    )
{
    if (Counts[Bin] >= 2u) {
        return Threshold2BitsetMany;
    } else if (Counts[Bin] == 1u) {
        return Threshold2BitsetOne;
    } else {
        return Threshold2BitsetZero;
    }
}

template <typename WordType,
          bool SetSeenWhenMany = false,
          bool SkipIfAlreadyMany = true>
int
RunCase(
    const char *Name,
    uint32_t NumberOfBins,
    const std::vector<uint32_t>& Inputs
    )
{
    const size_t NumberOfWords =
        Threshold2BitsetWordCount<WordType>(NumberOfBins);
    const size_t InputsSizeInBytes = Inputs.size() * sizeof(uint32_t);
    const size_t BitsetSizeInBytes = NumberOfWords * sizeof(WordType);
    uint32_t *DeviceInputs = nullptr;
    WordType *DeviceSeen = nullptr;
    WordType *DeviceMany = nullptr;
    std::vector<WordType> HostSeen(NumberOfWords);
    std::vector<WordType> HostMany(NumberOfWords);
    std::vector<uint32_t> Counts(NumberOfBins);

    for (uint32_t Bin : Inputs) {
        if (Bin >= NumberOfBins) {
            fprintf(stderr,
                    "%s: input bin %u is outside bin count %u\n",
                    Name,
                    Bin,
                    NumberOfBins);
            return 1;
        }

        if (Counts[Bin] < 2u) {
            Counts[Bin]++;
        }
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&DeviceInputs),
                          InputsSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&DeviceSeen),
                          BitsetSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&DeviceMany),
                          BitsetSizeInBytes));
    CUDA_CHECK(cudaMemcpy(DeviceInputs,
                          Inputs.data(),
                          InputsSizeInBytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(DeviceSeen, 0, BitsetSizeInBytes));
    CUDA_CHECK(cudaMemset(DeviceMany, 0, BitsetSizeInBytes));

    Threshold2BitsetKernel<WordType, SetSeenWhenMany, SkipIfAlreadyMany>
        <<<4, 128>>>(DeviceInputs, Inputs.size(), DeviceSeen, DeviceMany);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(HostSeen.data(),
                          DeviceSeen,
                          BitsetSizeInBytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(HostMany.data(),
                          DeviceMany,
                          BitsetSizeInBytes,
                          cudaMemcpyDeviceToHost));

    for (uint32_t Bin = 0; Bin < NumberOfBins; Bin++) {
        const THRESHOLD2_BITSET_STATE Actual =
            Threshold2BitsetReadState(HostSeen.data(), HostMany.data(), Bin);
        const THRESHOLD2_BITSET_STATE Expected =
            ExpectedState<WordType>(Counts, Bin);

        if (Actual != Expected) {
            fprintf(stderr,
                    "%s: bin %u expected state %u, got %u\n",
                    Name,
                    Bin,
                    static_cast<unsigned>(Expected),
                    static_cast<unsigned>(Actual));
            return 1;
        }
    }

    if constexpr (SetSeenWhenMany) {
        for (uint32_t Bin = 0; Bin < NumberOfBins; Bin++) {
            if (Counts[Bin] < 2u) {
                continue;
            }

            const size_t WordIndex =
                Threshold2BitsetWordIndex<WordType>(Bin);
            const WordType Mask = Threshold2BitsetMask<WordType>(Bin);

            if ((HostSeen[WordIndex] & Mask) == 0) {
                fprintf(stderr,
                        "%s: bin %u is many but seen bit was not set\n",
                        Name,
                        Bin);
                return 1;
            }
        }
    }

    CUDA_CHECK(cudaFree(DeviceInputs));
    CUDA_CHECK(cudaFree(DeviceSeen));
    CUDA_CHECK(cudaFree(DeviceMany));

    return 0;
}

int
main()
{
    int DeviceCount = 0;
    cudaError_t Error = cudaGetDeviceCount(&DeviceCount);

    if (Error == cudaErrorNoDevice || DeviceCount == 0) {
        printf("No CUDA devices found; skipping threshold-2 bitset tests.\n");
        return 0;
    }

    CUDA_CHECK(Error);

    const std::vector<uint32_t> UniqueAndBoundaryBins = {
        0u, 1u, 2u, 31u, 32u, 33u, 63u, 64u, 65u, 96u, 127u
    };

    const std::vector<uint32_t> SameWarpDuplicateBins = {
        7u, 7u, 8u, 9u, 10u
    };

    std::vector<uint32_t> CrossWarpDuplicateBins;
    CrossWarpDuplicateBins.reserve(256);
    for (uint32_t Index = 0; Index < 256u; Index++) {
        CrossWarpDuplicateBins.push_back(Index % 4u == 0u ? 42u : Index);
    }

    const std::vector<uint32_t> MixedBins = {
        0u, 31u, 31u, 32u, 33u, 63u, 63u, 63u,
        64u, 65u, 95u, 96u, 96u, 127u, 128u, 191u, 191u
    };

    if (RunCase<uint32_t>("uint32 unique-boundary",
                          192u,
                          UniqueAndBoundaryBins) != 0) {
        return 1;
    }

    if (RunCase<uint32_t>("uint32 same-warp duplicate",
                          64u,
                          SameWarpDuplicateBins) != 0) {
        return 1;
    }

    if (RunCase<uint32_t>("uint32 cross-warp duplicate",
                          256u,
                          CrossWarpDuplicateBins) != 0) {
        return 1;
    }

    if (RunCase<uint32_t, true, true>("uint32 set-seen-when-many",
                                      192u,
                                      MixedBins) != 0) {
        return 1;
    }

    if (RunCase<uint32_t, false, false>("uint32 no-skip saturated",
                                        192u,
                                        MixedBins) != 0) {
        return 1;
    }

    if (RunCase<uint64_t>("uint64 unique-boundary",
                          192u,
                          UniqueAndBoundaryBins) != 0) {
        return 1;
    }

    if (RunCase<uint64_t>("uint64 same-warp duplicate",
                          64u,
                          SameWarpDuplicateBins) != 0) {
        return 1;
    }

    if (RunCase<uint64_t>("uint64 mixed",
                          192u,
                          MixedBins) != 0) {
        return 1;
    }

    return 0;
}
