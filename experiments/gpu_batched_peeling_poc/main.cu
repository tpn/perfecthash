#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

enum class StorageMode {
    Auto,
    Bits16,
    Bits32,
};

struct Options {
    uint32_t Edges = 2048;
    uint32_t Batch = 256;
    uint32_t Threads = 256;
    uint64_t KeySeed = 0x123456789abcdef0ull;
    uint64_t GraphSeed = 0x0f1e2d3c4b5a6978ull;
    std::string KeysFile;
    StorageMode Storage = StorageMode::Auto;
    bool Verbose = false;
};

template<typename StorageT>
struct FrontierItemT {
    uint32_t Graph;
    StorageT Vertex;
    StorageT Edge;
};

struct CpuResult {
    bool Success = false;
    bool Verified = false;
    bool Invalid = false;
    uint32_t Peeled = 0;
};

struct ExperimentResult {
    uint32_t KeyCount = 0;
    uint32_t EdgeCapacity = 0;
    uint32_t Vertices = 0;
    uint32_t Batch = 0;
    uint32_t Rounds = 0;
    uint32_t GpuSuccess = 0;
    uint32_t CpuSuccess = 0;
    uint32_t Mismatches = 0;
    uint32_t CpuVerifyIssues = 0;
    float GpuMilliseconds = 0.0f;
    double CpuMilliseconds = 0.0;
    uint32_t StorageBits = 0;
};

template<typename T>
constexpr T
MaxValue()
{
    return std::numeric_limits<T>::max();
}

inline void
CheckCuda(cudaError_t Error, const char *Message)
{
    if (Error != cudaSuccess) {
        std::cerr << Message << ": " << cudaGetErrorString(Error) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

std::vector<uint64_t>
LoadKeysFromFile(const std::string &Path)
{
    std::ifstream File(Path, std::ios::binary | std::ios::ate);
    if (!File) {
        std::cerr << "Failed to open keys file: " << Path << "\n";
        std::exit(EXIT_FAILURE);
    }

    std::streamsize Size = File.tellg();
    if (Size <= 0 || (Size % static_cast<std::streamsize>(sizeof(uint32_t))) != 0) {
        std::cerr << "Invalid keys file size: " << Path << "\n";
        std::exit(EXIT_FAILURE);
    }

    std::vector<uint32_t> Raw(static_cast<size_t>(Size / sizeof(uint32_t)));
    File.seekg(0, std::ios::beg);
    if (!File.read(reinterpret_cast<char *>(Raw.data()), Size)) {
        std::cerr << "Failed to read keys file: " << Path << "\n";
        std::exit(EXIT_FAILURE);
    }

    return std::vector<uint64_t>(Raw.begin(), Raw.end());
}

const char *
StorageModeToString(StorageMode Mode)
{
    switch (Mode) {
        case StorageMode::Auto:
            return "auto";
        case StorageMode::Bits16:
            return "16";
        case StorageMode::Bits32:
            return "32";
        default:
            return "unknown";
    }
}

Options
ParseOptions(int argc, char **argv)
{
    Options Opts;

    for (int Index = 1; Index < argc; ++Index) {
        std::string Arg(argv[Index]);

        auto RequireValue = [&](const char *Name) -> std::string {
            if (Index + 1 >= argc) {
                std::cerr << "Missing value for " << Name << "\n";
                std::exit(EXIT_FAILURE);
            }
            return argv[++Index];
        };

        if (Arg == "--edges") {
            Opts.Edges = static_cast<uint32_t>(std::stoul(RequireValue("--edges")));
        } else if (Arg == "--keys-file") {
            Opts.KeysFile = RequireValue("--keys-file");
        } else if (Arg == "--batch") {
            Opts.Batch = static_cast<uint32_t>(std::stoul(RequireValue("--batch")));
        } else if (Arg == "--threads") {
            Opts.Threads = static_cast<uint32_t>(std::stoul(RequireValue("--threads")));
        } else if (Arg == "--key-seed") {
            Opts.KeySeed = std::stoull(RequireValue("--key-seed"), nullptr, 0);
        } else if (Arg == "--graph-seed") {
            Opts.GraphSeed = std::stoull(RequireValue("--graph-seed"), nullptr, 0);
        } else if (Arg == "--storage-bits") {
            auto Value = RequireValue("--storage-bits");
            if (Value == "auto") {
                Opts.Storage = StorageMode::Auto;
            } else if (Value == "16") {
                Opts.Storage = StorageMode::Bits16;
            } else if (Value == "32") {
                Opts.Storage = StorageMode::Bits32;
            } else {
                std::cerr << "Invalid --storage-bits value: " << Value << "\n";
                std::exit(EXIT_FAILURE);
            }
        } else if (Arg == "--verbose") {
            Opts.Verbose = true;
        } else if (Arg == "--help" || Arg == "-h") {
            std::cout
                << "Usage: gpu_batched_peeling_poc [options]\n"
                << "  --edges <n>         Number of logical keys for generated input\n"
                << "  --keys-file <p>     Load 32-bit keys from a .keys file\n"
                << "  --batch <n>         Number of graph attempts in the batch\n"
                << "  --threads <n>       Threads per block for build/collect/peel kernels\n"
                << "  --storage-bits <x>  auto, 16, or 32\n"
                << "  --key-seed <x>      Base seed for generated keys\n"
                << "  --graph-seed <x>    Base seed for per-graph hash seeds\n"
                << "  --verbose           Print per-graph mismatch details\n";
            std::exit(EXIT_SUCCESS);
        } else {
            std::cerr << "Unknown argument: " << Arg << "\n";
            std::exit(EXIT_FAILURE);
        }
    }

    if (Opts.Edges == 0 || Opts.Batch == 0 || Opts.Threads == 0) {
        std::cerr << "Edges, batch, and threads must be non-zero.\n";
        std::exit(EXIT_FAILURE);
    }

    return Opts;
}

inline uint32_t
NextPowerOfTwo(uint32_t Value)
{
    if (Value <= 1) {
        return 1;
    }

    --Value;
    Value |= Value >> 1;
    Value |= Value >> 2;
    Value |= Value >> 4;
    Value |= Value >> 8;
    Value |= Value >> 16;
    return Value + 1;
}

__host__ __device__ inline uint64_t
SplitMix64(uint64_t Value)
{
    Value += 0x9e3779b97f4a7c15ull;
    Value = (Value ^ (Value >> 30)) * 0xbf58476d1ce4e5b9ull;
    Value = (Value ^ (Value >> 27)) * 0x94d049bb133111ebull;
    return Value ^ (Value >> 31);
}

__host__ __device__ inline uint32_t
HashVertex(uint64_t Key, uint64_t Seed, uint32_t VertexMask)
{
    return static_cast<uint32_t>(SplitMix64(Key ^ Seed) & VertexMask);
}

template<typename StorageT>
__global__ void
BuildGraphsKernel(uint32_t Edges,
                  uint32_t Vertices,
                  uint32_t Batch,
                  const uint64_t *Keys,
                  const uint64_t *Seeds1,
                  const uint64_t *Seeds2,
                  StorageT *EdgeU,
                  StorageT *EdgeV,
                  uint32_t *Degree,
                  uint32_t *XorEdge,
                  uint32_t *InvalidGraphs)
{
    uint64_t Global = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t Total = static_cast<uint64_t>(Edges) * Batch;
    uint32_t VertexMask = Vertices - 1;

    while (Global < Total) {
        uint32_t Graph = static_cast<uint32_t>(Global / Edges);
        uint32_t Edge = static_cast<uint32_t>(Global % Edges);
        uint32_t EdgeIndex = Graph * Edges + Edge;
        uint32_t VertexBase = Graph * Vertices;

        uint64_t Key = Keys[Edge];
        uint32_t U = HashVertex(Key, Seeds1[Graph], VertexMask);
        uint32_t V = HashVertex(Key, Seeds2[Graph], VertexMask);

        EdgeU[EdgeIndex] = static_cast<StorageT>(U);
        EdgeV[EdgeIndex] = static_cast<StorageT>(V);

        if (U == V) {
            atomicExch(&InvalidGraphs[Graph], 1u);
            Global += blockDim.x * gridDim.x;
            continue;
        }

        atomicAdd(&Degree[VertexBase + U], 1u);
        atomicAdd(&Degree[VertexBase + V], 1u);
        atomicXor(&XorEdge[VertexBase + U], Edge);
        atomicXor(&XorEdge[VertexBase + V], Edge);

        Global += blockDim.x * gridDim.x;
    }
}

template<typename StorageT>
__global__ void
CollectFrontierKernel(uint32_t Vertices,
                      uint32_t Batch,
                      const uint32_t *Degree,
                      const uint32_t *XorEdge,
                      const uint32_t *InvalidGraphs,
                      FrontierItemT<StorageT> *Frontier,
                      uint32_t *FrontierCount)
{
    uint64_t Global = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t Total = static_cast<uint64_t>(Vertices) * Batch;

    while (Global < Total) {
        uint32_t Graph = static_cast<uint32_t>(Global / Vertices);
        uint32_t Vertex = static_cast<uint32_t>(Global % Vertices);

        if (InvalidGraphs[Graph] == 0 && Degree[Global] == 1) {
            uint32_t Position = atomicAdd(FrontierCount, 1u);
            Frontier[Position].Graph = Graph;
            Frontier[Position].Vertex = static_cast<StorageT>(Vertex);
            Frontier[Position].Edge = static_cast<StorageT>(XorEdge[Global]);
        }

        Global += blockDim.x * gridDim.x;
    }
}

template<typename StorageT>
__global__ void
PeelFrontierKernel(uint32_t Edges,
                   uint32_t Vertices,
                   uint32_t FrontierCount,
                   const FrontierItemT<StorageT> *Frontier,
                   const StorageT *EdgeU,
                   const StorageT *EdgeV,
                   uint32_t *Degree,
                   uint32_t *XorEdge,
                   uint32_t *EdgePeeled,
                   StorageT *OwnerVertex,
                   StorageT *PeelOrder,
                   uint32_t *PeeledCount)
{
    uint64_t Global = blockIdx.x * blockDim.x + threadIdx.x;

    while (Global < FrontierCount) {
        FrontierItemT<StorageT> Item = Frontier[Global];
        uint32_t Edge = static_cast<uint32_t>(Item.Edge);
        uint32_t EdgeIndex = Item.Graph * Edges + Edge;

        if (atomicCAS(&EdgePeeled[EdgeIndex], 0u, 1u) == 0u) {
            uint32_t Order = atomicAdd(&PeeledCount[Item.Graph], 1u);
            uint32_t VertexBase = Item.Graph * Vertices;
            uint32_t U = static_cast<uint32_t>(EdgeU[EdgeIndex]);
            uint32_t V = static_cast<uint32_t>(EdgeV[EdgeIndex]);

            OwnerVertex[EdgeIndex] = Item.Vertex;
            PeelOrder[Item.Graph * Edges + Order] = static_cast<StorageT>(Edge);

            atomicSub(&Degree[VertexBase + U], 1u);
            atomicXor(&XorEdge[VertexBase + U], Edge);
            atomicSub(&Degree[VertexBase + V], 1u);
            atomicXor(&XorEdge[VertexBase + V], Edge);
        }

        Global += blockDim.x * gridDim.x;
    }
}

template<typename StorageT>
__global__ void
AssignGraphsKernel(uint32_t Edges,
                   uint32_t Vertices,
                   uint32_t EdgeMask,
                   const uint32_t *InvalidGraphs,
                   const StorageT *EdgeU,
                   const StorageT *EdgeV,
                   const StorageT *OwnerVertex,
                   const StorageT *PeelOrder,
                   const uint32_t *PeeledCount,
                   StorageT *Assigned)
{
    uint32_t Graph = blockIdx.x;

    if (threadIdx.x != 0) {
        return;
    }

    if (InvalidGraphs[Graph] != 0 || PeeledCount[Graph] != Edges) {
        return;
    }

    uint32_t VertexBase = Graph * Vertices;
    uint32_t EdgeBase = Graph * Edges;

    for (int64_t Index = static_cast<int64_t>(Edges) - 1; Index >= 0; --Index) {
        uint32_t Edge = static_cast<uint32_t>(PeelOrder[EdgeBase + static_cast<uint32_t>(Index)]);
        uint32_t Owner = static_cast<uint32_t>(OwnerVertex[EdgeBase + Edge]);
        uint32_t U = static_cast<uint32_t>(EdgeU[EdgeBase + Edge]);
        uint32_t V = static_cast<uint32_t>(EdgeV[EdgeBase + Edge]);
        uint32_t Other = (Owner == U) ? V : U;
        uint32_t OtherAssigned = static_cast<uint32_t>(Assigned[VertexBase + Other]);
        uint32_t Value = (Edge - OtherAssigned) & EdgeMask;

        Assigned[VertexBase + Owner] = static_cast<StorageT>(Value);
    }
}

template<typename StorageT>
__global__ void
VerifyGraphsKernel(uint32_t Edges,
                   uint32_t Vertices,
                   uint32_t Batch,
                   uint32_t EdgeMask,
                   const uint32_t *InvalidGraphs,
                   const StorageT *EdgeU,
                   const StorageT *EdgeV,
                   const StorageT *Assigned,
                   const uint32_t *PeeledCount,
                   uint32_t *VerifyFailures)
{
    uint64_t Global = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t Total = static_cast<uint64_t>(Edges) * Batch;

    while (Global < Total) {
        uint32_t Graph = static_cast<uint32_t>(Global / Edges);
        uint32_t Edge = static_cast<uint32_t>(Global % Edges);

        if (InvalidGraphs[Graph] != 0 || PeeledCount[Graph] != Edges) {
            atomicAdd(&VerifyFailures[Graph], 1u);
            Global += blockDim.x * gridDim.x;
            continue;
        }

        uint32_t EdgeBase = Graph * Edges;
        uint32_t VertexBase = Graph * Vertices;
        uint32_t U = static_cast<uint32_t>(EdgeU[EdgeBase + Edge]);
        uint32_t V = static_cast<uint32_t>(EdgeV[EdgeBase + Edge]);
        uint32_t Index = (
            static_cast<uint32_t>(Assigned[VertexBase + U]) +
            static_cast<uint32_t>(Assigned[VertexBase + V])
        ) & EdgeMask;

        if (Index != Edge) {
            atomicAdd(&VerifyFailures[Graph], 1u);
        }

        Global += blockDim.x * gridDim.x;
    }
}

template<typename StorageT>
CpuResult
RunCpuReference(uint32_t Graph,
                uint32_t Edges,
                uint32_t Vertices,
                uint32_t EdgeMask,
                const std::vector<uint64_t> &Keys,
                const std::vector<uint64_t> &Seeds1,
                const std::vector<uint64_t> &Seeds2)
{
    CpuResult Result;
    std::vector<uint32_t> Degree(Vertices, 0);
    std::vector<uint32_t> XorEdge(Vertices, 0);
    std::vector<StorageT> EdgeU(Edges, 0);
    std::vector<StorageT> EdgeV(Edges, 0);
    std::vector<StorageT> Owner(Edges, MaxValue<StorageT>());
    std::vector<StorageT> Order;
    std::vector<StorageT> Assigned(Vertices, 0);
    std::vector<uint8_t> Peeled(Edges, 0);
    std::vector<uint32_t> Queue;

    Order.reserve(Edges);
    Queue.reserve(Vertices);

    for (uint32_t Edge = 0; Edge < Edges; ++Edge) {
        uint32_t U = HashVertex(Keys[Edge], Seeds1[Graph], Vertices - 1);
        uint32_t V = HashVertex(Keys[Edge], Seeds2[Graph], Vertices - 1);

        EdgeU[Edge] = static_cast<StorageT>(U);
        EdgeV[Edge] = static_cast<StorageT>(V);

        if (U == V) {
            Result.Invalid = true;
            return Result;
        }

        ++Degree[U];
        ++Degree[V];
        XorEdge[U] ^= Edge;
        XorEdge[V] ^= Edge;
    }

    for (uint32_t Vertex = 0; Vertex < Vertices; ++Vertex) {
        if (Degree[Vertex] == 1) {
            Queue.push_back(Vertex);
        }
    }

    for (size_t Head = 0; Head < Queue.size(); ++Head) {
        uint32_t Vertex = Queue[Head];

        if (Degree[Vertex] != 1) {
            continue;
        }

        uint32_t Edge = XorEdge[Vertex];
        if (Peeled[Edge] != 0) {
            continue;
        }

        Peeled[Edge] = 1;
        Owner[Edge] = static_cast<StorageT>(Vertex);
        Order.push_back(static_cast<StorageT>(Edge));

        uint32_t U = static_cast<uint32_t>(EdgeU[Edge]);
        uint32_t V = static_cast<uint32_t>(EdgeV[Edge]);

        if (Degree[U] > 0) {
            --Degree[U];
            XorEdge[U] ^= Edge;
            if (Degree[U] == 1) {
                Queue.push_back(U);
            }
        }

        if (Degree[V] > 0) {
            --Degree[V];
            XorEdge[V] ^= Edge;
            if (Degree[V] == 1) {
                Queue.push_back(V);
            }
        }
    }

    Result.Peeled = static_cast<uint32_t>(Order.size());
    Result.Success = (Order.size() == Edges);
    if (!Result.Success) {
        return Result;
    }

    for (int64_t Index = static_cast<int64_t>(Order.size()) - 1; Index >= 0; --Index) {
        uint32_t Edge = static_cast<uint32_t>(Order[static_cast<size_t>(Index)]);
        uint32_t OwnerVertex = static_cast<uint32_t>(Owner[Edge]);
        uint32_t U = static_cast<uint32_t>(EdgeU[Edge]);
        uint32_t V = static_cast<uint32_t>(EdgeV[Edge]);
        uint32_t Other = (OwnerVertex == U) ? V : U;
        uint32_t Value = (Edge - static_cast<uint32_t>(Assigned[Other])) & EdgeMask;

        Assigned[OwnerVertex] = static_cast<StorageT>(Value);
    }

    Result.Verified = true;

    for (uint32_t Edge = 0; Edge < Edges; ++Edge) {
        uint32_t Index = (
            static_cast<uint32_t>(Assigned[static_cast<uint32_t>(EdgeU[Edge])]) +
            static_cast<uint32_t>(Assigned[static_cast<uint32_t>(EdgeV[Edge])])
        ) & EdgeMask;
        if (Index != Edge) {
            Result.Verified = false;
            break;
        }
    }

    return Result;
}

template<typename StorageT>
ExperimentResult
RunExperiment(const Options &Opts,
              const std::vector<uint64_t> &Keys,
              const std::vector<uint64_t> &Seeds1,
              const std::vector<uint64_t> &Seeds2,
              const std::string &RequestedKeys)
{
    using FrontierItem = FrontierItemT<StorageT>;

    ExperimentResult Result;

    uint32_t Batch = Opts.Batch;
    uint32_t KeyCount = static_cast<uint32_t>(Keys.size());
    uint32_t EdgeCapacity = NextPowerOfTwo(KeyCount);
    uint32_t Vertices = NextPowerOfTwo(EdgeCapacity + 1);
    uint32_t EdgeMask = EdgeCapacity - 1;

    Result.KeyCount = KeyCount;
    Result.EdgeCapacity = EdgeCapacity;
    Result.Vertices = Vertices;
    Result.Batch = Batch;
    Result.StorageBits = static_cast<uint32_t>(sizeof(StorageT) * 8);

    uint64_t TotalEdges = static_cast<uint64_t>(KeyCount) * Batch;
    uint64_t TotalVertices = static_cast<uint64_t>(Vertices) * Batch;
    uint64_t FrontierCapacity = TotalVertices;

    uint64_t *DKeys = nullptr;
    uint64_t *DSeeds1 = nullptr;
    uint64_t *DSeeds2 = nullptr;
    StorageT *DEdgeU = nullptr;
    StorageT *DEdgeV = nullptr;
    uint32_t *DDegree = nullptr;
    uint32_t *DXorEdge = nullptr;
    uint32_t *DInvalidGraphs = nullptr;
    uint32_t *DEdgePeeled = nullptr;
    StorageT *DOwnerVertex = nullptr;
    StorageT *DPeelOrder = nullptr;
    uint32_t *DPeeledCount = nullptr;
    StorageT *DAssigned = nullptr;
    uint32_t *DVerifyFailures = nullptr;
    FrontierItem *DFrontier = nullptr;
    uint32_t *DFrontierCount = nullptr;

    CheckCuda(cudaMalloc(&DKeys, Keys.size() * sizeof(Keys[0])), "cudaMalloc(DKeys)");
    CheckCuda(cudaMalloc(&DSeeds1, Seeds1.size() * sizeof(Seeds1[0])), "cudaMalloc(DSeeds1)");
    CheckCuda(cudaMalloc(&DSeeds2, Seeds2.size() * sizeof(Seeds2[0])), "cudaMalloc(DSeeds2)");
    CheckCuda(cudaMalloc(&DEdgeU, TotalEdges * sizeof(StorageT)), "cudaMalloc(DEdgeU)");
    CheckCuda(cudaMalloc(&DEdgeV, TotalEdges * sizeof(StorageT)), "cudaMalloc(DEdgeV)");
    CheckCuda(cudaMalloc(&DDegree, TotalVertices * sizeof(uint32_t)), "cudaMalloc(DDegree)");
    CheckCuda(cudaMalloc(&DXorEdge, TotalVertices * sizeof(uint32_t)), "cudaMalloc(DXorEdge)");
    CheckCuda(cudaMalloc(&DInvalidGraphs, Batch * sizeof(uint32_t)), "cudaMalloc(DInvalidGraphs)");
    CheckCuda(cudaMalloc(&DEdgePeeled, TotalEdges * sizeof(uint32_t)), "cudaMalloc(DEdgePeeled)");
    CheckCuda(cudaMalloc(&DOwnerVertex, TotalEdges * sizeof(StorageT)), "cudaMalloc(DOwnerVertex)");
    CheckCuda(cudaMalloc(&DPeelOrder, TotalEdges * sizeof(StorageT)), "cudaMalloc(DPeelOrder)");
    CheckCuda(cudaMalloc(&DPeeledCount, Batch * sizeof(uint32_t)), "cudaMalloc(DPeeledCount)");
    CheckCuda(cudaMalloc(&DAssigned, TotalVertices * sizeof(StorageT)), "cudaMalloc(DAssigned)");
    CheckCuda(cudaMalloc(&DVerifyFailures, Batch * sizeof(uint32_t)), "cudaMalloc(DVerifyFailures)");
    CheckCuda(cudaMalloc(&DFrontier, FrontierCapacity * sizeof(FrontierItem)), "cudaMalloc(DFrontier)");
    CheckCuda(cudaMalloc(&DFrontierCount, sizeof(uint32_t)), "cudaMalloc(DFrontierCount)");

    CheckCuda(cudaMemcpy(DKeys, Keys.data(), Keys.size() * sizeof(Keys[0]), cudaMemcpyHostToDevice),
              "cudaMemcpy(DKeys)");
    CheckCuda(cudaMemcpy(DSeeds1, Seeds1.data(), Seeds1.size() * sizeof(Seeds1[0]), cudaMemcpyHostToDevice),
              "cudaMemcpy(DSeeds1)");
    CheckCuda(cudaMemcpy(DSeeds2, Seeds2.data(), Seeds2.size() * sizeof(Seeds2[0]), cudaMemcpyHostToDevice),
              "cudaMemcpy(DSeeds2)");

    CheckCuda(cudaMemset(DEdgeU, 0, TotalEdges * sizeof(StorageT)), "cudaMemset(DEdgeU)");
    CheckCuda(cudaMemset(DEdgeV, 0, TotalEdges * sizeof(StorageT)), "cudaMemset(DEdgeV)");
    CheckCuda(cudaMemset(DDegree, 0, TotalVertices * sizeof(uint32_t)), "cudaMemset(DDegree)");
    CheckCuda(cudaMemset(DXorEdge, 0, TotalVertices * sizeof(uint32_t)), "cudaMemset(DXorEdge)");
    CheckCuda(cudaMemset(DInvalidGraphs, 0, Batch * sizeof(uint32_t)), "cudaMemset(DInvalidGraphs)");
    CheckCuda(cudaMemset(DEdgePeeled, 0, TotalEdges * sizeof(uint32_t)), "cudaMemset(DEdgePeeled)");
    CheckCuda(cudaMemset(DOwnerVertex, 0xff, TotalEdges * sizeof(StorageT)), "cudaMemset(DOwnerVertex)");
    CheckCuda(cudaMemset(DPeelOrder, 0xff, TotalEdges * sizeof(StorageT)), "cudaMemset(DPeelOrder)");
    CheckCuda(cudaMemset(DPeeledCount, 0, Batch * sizeof(uint32_t)), "cudaMemset(DPeeledCount)");
    CheckCuda(cudaMemset(DAssigned, 0, TotalVertices * sizeof(StorageT)), "cudaMemset(DAssigned)");
    CheckCuda(cudaMemset(DVerifyFailures, 0, Batch * sizeof(uint32_t)), "cudaMemset(DVerifyFailures)");

    cudaEvent_t Start = nullptr;
    cudaEvent_t Stop = nullptr;
    CheckCuda(cudaEventCreate(&Start), "cudaEventCreate(Start)");
    CheckCuda(cudaEventCreate(&Stop), "cudaEventCreate(Stop)");

    uint32_t BuildBlocks = static_cast<uint32_t>((TotalEdges + Opts.Threads - 1) / Opts.Threads);
    uint32_t VertexBlocks = static_cast<uint32_t>((TotalVertices + Opts.Threads - 1) / Opts.Threads);

    CheckCuda(cudaEventRecord(Start), "cudaEventRecord(Start)");

    BuildGraphsKernel<StorageT><<<BuildBlocks, Opts.Threads>>>(KeyCount,
                                                               Vertices,
                                                               Batch,
                                                               DKeys,
                                                               DSeeds1,
                                                               DSeeds2,
                                                               DEdgeU,
                                                               DEdgeV,
                                                               DDegree,
                                                               DXorEdge,
                                                               DInvalidGraphs);
    CheckCuda(cudaGetLastError(), "BuildGraphsKernel launch");

    uint32_t FrontierCount = 0;

    for (;;) {
        CheckCuda(cudaMemset(DFrontierCount, 0, sizeof(uint32_t)), "cudaMemset(DFrontierCount)");

        CollectFrontierKernel<StorageT><<<VertexBlocks, Opts.Threads>>>(Vertices,
                                                                        Batch,
                                                                        DDegree,
                                                                        DXorEdge,
                                                                        DInvalidGraphs,
                                                                        DFrontier,
                                                                        DFrontierCount);
        CheckCuda(cudaGetLastError(), "CollectFrontierKernel launch");
        CheckCuda(cudaMemcpy(&FrontierCount, DFrontierCount, sizeof(uint32_t), cudaMemcpyDeviceToHost),
                  "cudaMemcpy(FrontierCount)");

        if (FrontierCount == 0) {
            break;
        }

        ++Result.Rounds;

        uint32_t PeelBlocks = static_cast<uint32_t>((FrontierCount + Opts.Threads - 1) / Opts.Threads);
        PeelFrontierKernel<StorageT><<<PeelBlocks, Opts.Threads>>>(KeyCount,
                                                                   Vertices,
                                                                   FrontierCount,
                                                                   DFrontier,
                                                                   DEdgeU,
                                                                   DEdgeV,
                                                                   DDegree,
                                                                   DXorEdge,
                                                                   DEdgePeeled,
                                                                   DOwnerVertex,
                                                                   DPeelOrder,
                                                                   DPeeledCount);
        CheckCuda(cudaGetLastError(), "PeelFrontierKernel launch");
    }

    AssignGraphsKernel<StorageT><<<Batch, 1>>>(KeyCount,
                                               Vertices,
                                               EdgeMask,
                                               DInvalidGraphs,
                                               DEdgeU,
                                               DEdgeV,
                                               DOwnerVertex,
                                               DPeelOrder,
                                               DPeeledCount,
                                               DAssigned);
    CheckCuda(cudaGetLastError(), "AssignGraphsKernel launch");

    uint32_t VerifyBlocks = static_cast<uint32_t>((TotalEdges + Opts.Threads - 1) / Opts.Threads);
    VerifyGraphsKernel<StorageT><<<VerifyBlocks, Opts.Threads>>>(KeyCount,
                                                                 Vertices,
                                                                 Batch,
                                                                 EdgeMask,
                                                                 DInvalidGraphs,
                                                                 DEdgeU,
                                                                 DEdgeV,
                                                                 DAssigned,
                                                                 DPeeledCount,
                                                                 DVerifyFailures);
    CheckCuda(cudaGetLastError(), "VerifyGraphsKernel launch");

    CheckCuda(cudaEventRecord(Stop), "cudaEventRecord(Stop)");
    CheckCuda(cudaEventSynchronize(Stop), "cudaEventSynchronize(Stop)");
    CheckCuda(cudaEventElapsedTime(&Result.GpuMilliseconds, Start, Stop), "cudaEventElapsedTime");

    std::vector<uint32_t> InvalidGraphs(Batch);
    std::vector<uint32_t> PeeledCount(Batch);
    std::vector<uint32_t> VerifyFailures(Batch);

    CheckCuda(cudaMemcpy(InvalidGraphs.data(), DInvalidGraphs, Batch * sizeof(uint32_t), cudaMemcpyDeviceToHost),
              "cudaMemcpy(InvalidGraphs)");
    CheckCuda(cudaMemcpy(PeeledCount.data(), DPeeledCount, Batch * sizeof(uint32_t), cudaMemcpyDeviceToHost),
              "cudaMemcpy(PeeledCount)");
    CheckCuda(cudaMemcpy(VerifyFailures.data(), DVerifyFailures, Batch * sizeof(uint32_t), cudaMemcpyDeviceToHost),
              "cudaMemcpy(VerifyFailures)");

    auto CpuStart = std::chrono::steady_clock::now();
    std::vector<CpuResult> CpuResults(Batch);

    for (uint32_t Graph = 0; Graph < Batch; ++Graph) {
        CpuResults[Graph] = RunCpuReference<StorageT>(Graph,
                                                      KeyCount,
                                                      Vertices,
                                                      EdgeMask,
                                                      Keys,
                                                      Seeds1,
                                                      Seeds2);
    }

    auto CpuStop = std::chrono::steady_clock::now();
    Result.CpuMilliseconds =
        std::chrono::duration<double, std::milli>(CpuStop - CpuStart).count();

    for (uint32_t Graph = 0; Graph < Batch; ++Graph) {
        bool ThisGpuSuccess = (InvalidGraphs[Graph] == 0 &&
                               PeeledCount[Graph] == KeyCount &&
                               VerifyFailures[Graph] == 0);
        bool ThisCpuSuccess = (CpuResults[Graph].Success && CpuResults[Graph].Verified);

        if (ThisGpuSuccess) {
            ++Result.GpuSuccess;
        }
        if (ThisCpuSuccess) {
            ++Result.CpuSuccess;
        }
        if (ThisGpuSuccess != ThisCpuSuccess) {
            ++Result.Mismatches;
            if (Opts.Verbose) {
                std::cout << "Mismatch graph=" << Graph
                          << " gpu_success=" << ThisGpuSuccess
                          << " cpu_success=" << ThisCpuSuccess
                          << " invalid=" << InvalidGraphs[Graph]
                          << " peeled_gpu=" << PeeledCount[Graph]
                          << " peeled_cpu=" << CpuResults[Graph].Peeled
                          << " verify_failures=" << VerifyFailures[Graph]
                          << "\n";
            }
        }
        if (CpuResults[Graph].Success && !CpuResults[Graph].Verified) {
            ++Result.CpuVerifyIssues;
        }
    }

    std::cout
        << "GPU Batched Peeling POC\n"
        << "  Keys file:          "
        << (Opts.KeysFile.empty() ? "<generated>" : Opts.KeysFile) << "\n"
        << "  Requested keys:     " << RequestedKeys << "\n"
        << "  Actual keys:        " << KeyCount << "\n"
        << "  Edge capacity:      " << EdgeCapacity << "\n"
        << "  Vertices:           " << Vertices << "\n"
        << "  Batch size:         " << Batch << "\n"
        << "  Storage bits:       " << Result.StorageBits << "\n"
        << "  Storage mode:       " << StorageModeToString(Opts.Storage) << "\n"
        << "  Peel rounds:        " << Result.Rounds << "\n"
        << "  GPU success:        " << Result.GpuSuccess << "/" << Batch << "\n"
        << "  CPU success:        " << Result.CpuSuccess << "/" << Batch << "\n"
        << "  Success mismatches: " << Result.Mismatches << "\n"
        << "  CPU verify issues:  " << Result.CpuVerifyIssues << "\n"
        << std::fixed << std::setprecision(3)
        << "  GPU time (ms):      " << Result.GpuMilliseconds << "\n"
        << "  CPU time (ms):      " << Result.CpuMilliseconds << "\n";

    CheckCuda(cudaFree(DFrontierCount), "cudaFree(DFrontierCount)");
    CheckCuda(cudaFree(DFrontier), "cudaFree(DFrontier)");
    CheckCuda(cudaFree(DVerifyFailures), "cudaFree(DVerifyFailures)");
    CheckCuda(cudaFree(DAssigned), "cudaFree(DAssigned)");
    CheckCuda(cudaFree(DPeeledCount), "cudaFree(DPeeledCount)");
    CheckCuda(cudaFree(DPeelOrder), "cudaFree(DPeelOrder)");
    CheckCuda(cudaFree(DOwnerVertex), "cudaFree(DOwnerVertex)");
    CheckCuda(cudaFree(DEdgePeeled), "cudaFree(DEdgePeeled)");
    CheckCuda(cudaFree(DInvalidGraphs), "cudaFree(DInvalidGraphs)");
    CheckCuda(cudaFree(DXorEdge), "cudaFree(DXorEdge)");
    CheckCuda(cudaFree(DDegree), "cudaFree(DDegree)");
    CheckCuda(cudaFree(DEdgeV), "cudaFree(DEdgeV)");
    CheckCuda(cudaFree(DEdgeU), "cudaFree(DEdgeU)");
    CheckCuda(cudaFree(DSeeds2), "cudaFree(DSeeds2)");
    CheckCuda(cudaFree(DSeeds1), "cudaFree(DSeeds1)");
    CheckCuda(cudaFree(DKeys), "cudaFree(DKeys)");
    CheckCuda(cudaEventDestroy(Stop), "cudaEventDestroy(Stop)");
    CheckCuda(cudaEventDestroy(Start), "cudaEventDestroy(Start)");

    return Result;
}

template<typename StorageT>
bool
SupportsStorage(uint32_t EdgeCapacity, uint32_t Vertices)
{
    constexpr uint64_t Max = static_cast<uint64_t>(std::numeric_limits<StorageT>::max());
    return (static_cast<uint64_t>(EdgeCapacity - 1) <= Max &&
            static_cast<uint64_t>(Vertices - 1) <= Max);
}

} // namespace

int
main(int argc, char **argv)
{
    Options Opts = ParseOptions(argc, argv);

    std::vector<uint64_t> Keys;
    std::string RequestedKeys = std::to_string(Opts.Edges);

    if (!Opts.KeysFile.empty()) {
        Keys = LoadKeysFromFile(Opts.KeysFile);
        RequestedKeys = "<n/a>";
    } else {
        Keys.resize(Opts.Edges);
        for (uint32_t Edge = 0; Edge < Opts.Edges; ++Edge) {
            Keys[Edge] = SplitMix64(Opts.KeySeed + Edge);
        }
    }

    if (Keys.empty()) {
        std::cerr << "No keys available.\n";
        return EXIT_FAILURE;
    }

    uint32_t KeyCount = static_cast<uint32_t>(Keys.size());
    uint32_t EdgeCapacity = NextPowerOfTwo(KeyCount);
    uint32_t Vertices = NextPowerOfTwo(EdgeCapacity + 1);

    std::vector<uint64_t> Seeds1(Opts.Batch);
    std::vector<uint64_t> Seeds2(Opts.Batch);
    for (uint32_t Graph = 0; Graph < Opts.Batch; ++Graph) {
        Seeds1[Graph] = SplitMix64(Opts.GraphSeed + (Graph * 2ull));
        Seeds2[Graph] = SplitMix64(Opts.GraphSeed + (Graph * 2ull) + 1ull);
    }

    StorageMode Selected = Opts.Storage;
    if (Selected == StorageMode::Auto) {
        if (SupportsStorage<uint16_t>(EdgeCapacity, Vertices)) {
            Selected = StorageMode::Bits16;
        } else {
            Selected = StorageMode::Bits32;
        }
    }

    if (Selected == StorageMode::Bits16 && !SupportsStorage<uint16_t>(EdgeCapacity, Vertices)) {
        std::cerr << "16-bit storage does not support edge capacity " << EdgeCapacity
                  << " and vertex count " << Vertices << ".\n";
        return EXIT_FAILURE;
    }

    ExperimentResult Result;
    if (Selected == StorageMode::Bits16) {
        Result = RunExperiment<uint16_t>(Opts, Keys, Seeds1, Seeds2, RequestedKeys);
    } else {
        Result = RunExperiment<uint32_t>(Opts, Keys, Seeds1, Seeds2, RequestedKeys);
    }

    return (Result.Mismatches == 0 && Result.CpuVerifyIssues == 0) ?
        EXIT_SUCCESS :
        EXIT_FAILURE;
}
