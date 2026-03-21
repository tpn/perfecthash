#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr uint32_t INVALID_U32 = 0xffffffffu;

struct Options {
    uint32_t Edges = 2048;
    uint32_t Batch = 256;
    uint32_t Threads = 256;
    uint64_t KeySeed = 0x123456789abcdef0ull;
    uint64_t GraphSeed = 0x0f1e2d3c4b5a6978ull;
    bool Verbose = false;
};

struct FrontierItem {
    uint32_t Graph;
    uint32_t Vertex;
    uint32_t Edge;
};

struct CpuResult {
    bool Success = false;
    bool Verified = false;
    bool Invalid = false;
    uint32_t Peeled = 0;
};

inline void
CheckCuda(cudaError_t Error, const char *Message)
{
    if (Error != cudaSuccess) {
        std::cerr << Message << ": " << cudaGetErrorString(Error) << "\n";
        std::exit(EXIT_FAILURE);
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
        } else if (Arg == "--batch") {
            Opts.Batch = static_cast<uint32_t>(std::stoul(RequireValue("--batch")));
        } else if (Arg == "--threads") {
            Opts.Threads = static_cast<uint32_t>(std::stoul(RequireValue("--threads")));
        } else if (Arg == "--key-seed") {
            Opts.KeySeed = std::stoull(RequireValue("--key-seed"), nullptr, 0);
        } else if (Arg == "--graph-seed") {
            Opts.GraphSeed = std::stoull(RequireValue("--graph-seed"), nullptr, 0);
        } else if (Arg == "--verbose") {
            Opts.Verbose = true;
        } else if (Arg == "--help" || Arg == "-h") {
            std::cout
                << "Usage: gpu_batched_peeling_poc [options]\n"
                << "  --edges <n>       Number of edges/keys (rounded up to power of two)\n"
                << "  --batch <n>       Number of graph attempts in the batch\n"
                << "  --threads <n>     Threads per block for build/collect/peel kernels\n"
                << "  --key-seed <x>    Base seed for generated keys\n"
                << "  --graph-seed <x>  Base seed for per-graph hash seeds\n"
                << "  --verbose         Print per-graph mismatch details\n";
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

__global__ void
BuildGraphsKernel(uint32_t Edges,
                  uint32_t Vertices,
                  uint32_t Batch,
                  const uint64_t *Keys,
                  const uint64_t *Seeds1,
                  const uint64_t *Seeds2,
                  uint32_t *EdgeU,
                  uint32_t *EdgeV,
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

        EdgeU[EdgeIndex] = U;
        EdgeV[EdgeIndex] = V;

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

__global__ void
CollectFrontierKernel(uint32_t Vertices,
                      uint32_t Batch,
                      const uint32_t *Degree,
                      const uint32_t *XorEdge,
                      const uint32_t *InvalidGraphs,
                      FrontierItem *Frontier,
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
            Frontier[Position].Vertex = Vertex;
            Frontier[Position].Edge = XorEdge[Global];
        }

        Global += blockDim.x * gridDim.x;
    }
}

__global__ void
PeelFrontierKernel(uint32_t Edges,
                   uint32_t Vertices,
                   uint32_t FrontierCount,
                   const FrontierItem *Frontier,
                   const uint32_t *EdgeU,
                   const uint32_t *EdgeV,
                   uint32_t *Degree,
                   uint32_t *XorEdge,
                   uint32_t *EdgePeeled,
                   uint32_t *OwnerVertex,
                   uint32_t *PeelOrder,
                   uint32_t *PeeledCount)
{
    uint64_t Global = blockIdx.x * blockDim.x + threadIdx.x;

    while (Global < FrontierCount) {
        FrontierItem Item = Frontier[Global];
        uint32_t EdgeIndex = Item.Graph * Edges + Item.Edge;

        if (atomicCAS(&EdgePeeled[EdgeIndex], 0u, 1u) == 0u) {
            uint32_t Order = atomicAdd(&PeeledCount[Item.Graph], 1u);
            uint32_t VertexBase = Item.Graph * Vertices;
            uint32_t U = EdgeU[EdgeIndex];
            uint32_t V = EdgeV[EdgeIndex];

            OwnerVertex[EdgeIndex] = Item.Vertex;
            PeelOrder[Item.Graph * Edges + Order] = Item.Edge;

            atomicSub(&Degree[VertexBase + U], 1u);
            atomicXor(&XorEdge[VertexBase + U], Item.Edge);
            atomicSub(&Degree[VertexBase + V], 1u);
            atomicXor(&XorEdge[VertexBase + V], Item.Edge);
        }

        Global += blockDim.x * gridDim.x;
    }
}

__global__ void
AssignGraphsKernel(uint32_t Edges,
                   uint32_t Vertices,
                   uint32_t EdgeMask,
                   const uint32_t *InvalidGraphs,
                   const uint32_t *EdgeU,
                   const uint32_t *EdgeV,
                   const uint32_t *OwnerVertex,
                   const uint32_t *PeelOrder,
                   const uint32_t *PeeledCount,
                   uint32_t *Assigned)
{
    uint32_t Graph = blockIdx.x;

    if (Graph >= gridDim.x || threadIdx.x != 0) {
        return;
    }

    if (InvalidGraphs[Graph] != 0 || PeeledCount[Graph] != Edges) {
        return;
    }

    uint32_t VertexBase = Graph * Vertices;
    uint32_t EdgeBase = Graph * Edges;

    for (int64_t Index = static_cast<int64_t>(Edges) - 1; Index >= 0; --Index) {
        uint32_t Edge = PeelOrder[EdgeBase + static_cast<uint32_t>(Index)];
        uint32_t Owner = OwnerVertex[EdgeBase + Edge];
        uint32_t U = EdgeU[EdgeBase + Edge];
        uint32_t V = EdgeV[EdgeBase + Edge];
        uint32_t Other = (Owner == U) ? V : U;
        uint32_t OtherAssigned = Assigned[VertexBase + Other];

        Assigned[VertexBase + Owner] = (Edge - OtherAssigned) & EdgeMask;
    }
}

__global__ void
VerifyGraphsKernel(uint32_t Edges,
                   uint32_t Vertices,
                   uint32_t Batch,
                   uint32_t EdgeMask,
                   const uint32_t *InvalidGraphs,
                   const uint32_t *EdgeU,
                   const uint32_t *EdgeV,
                   const uint32_t *Assigned,
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
        uint32_t U = EdgeU[EdgeBase + Edge];
        uint32_t V = EdgeV[EdgeBase + Edge];
        uint32_t Index = (Assigned[VertexBase + U] +
                          Assigned[VertexBase + V]) & EdgeMask;

        if (Index != Edge) {
            atomicAdd(&VerifyFailures[Graph], 1u);
        }

        Global += blockDim.x * gridDim.x;
    }
}

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
    std::vector<uint32_t> EdgeU(Edges, 0);
    std::vector<uint32_t> EdgeV(Edges, 0);
    std::vector<uint32_t> Owner(Edges, INVALID_U32);
    std::vector<uint32_t> Order;
    std::vector<uint32_t> Assigned(Vertices, 0);
    std::vector<uint8_t> Peeled(Edges, 0);
    std::vector<uint32_t> Queue;

    Order.reserve(Edges);
    Queue.reserve(Vertices);

    for (uint32_t Edge = 0; Edge < Edges; ++Edge) {
        uint32_t U = HashVertex(Keys[Edge], Seeds1[Graph], Vertices - 1);
        uint32_t V = HashVertex(Keys[Edge], Seeds2[Graph], Vertices - 1);

        EdgeU[Edge] = U;
        EdgeV[Edge] = V;

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
        Owner[Edge] = Vertex;
        Order.push_back(Edge);

        uint32_t U = EdgeU[Edge];
        uint32_t V = EdgeV[Edge];

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
        uint32_t Edge = Order[static_cast<size_t>(Index)];
        uint32_t OwnerVertex = Owner[Edge];
        uint32_t U = EdgeU[Edge];
        uint32_t V = EdgeV[Edge];
        uint32_t Other = (OwnerVertex == U) ? V : U;
        Assigned[OwnerVertex] = (Edge - Assigned[Other]) & EdgeMask;
    }

    Result.Verified = true;

    for (uint32_t Edge = 0; Edge < Edges; ++Edge) {
        uint32_t Index = (Assigned[EdgeU[Edge]] + Assigned[EdgeV[Edge]]) & EdgeMask;
        if (Index != Edge) {
            Result.Verified = false;
            break;
        }
    }

    return Result;
}

} // namespace

int
main(int argc, char **argv)
{
    Options Opts = ParseOptions(argc, argv);

    uint32_t Edges = NextPowerOfTwo(Opts.Edges);
    uint32_t Vertices = NextPowerOfTwo(Edges + 1);
    uint32_t Batch = Opts.Batch;
    uint32_t EdgeMask = Edges - 1;

    std::vector<uint64_t> Keys(Edges);
    std::vector<uint64_t> Seeds1(Batch);
    std::vector<uint64_t> Seeds2(Batch);

    for (uint32_t Edge = 0; Edge < Edges; ++Edge) {
        Keys[Edge] = SplitMix64(Opts.KeySeed + Edge);
    }

    for (uint32_t Graph = 0; Graph < Batch; ++Graph) {
        Seeds1[Graph] = SplitMix64(Opts.GraphSeed + (Graph * 2ull));
        Seeds2[Graph] = SplitMix64(Opts.GraphSeed + (Graph * 2ull) + 1ull);
    }

    uint64_t TotalEdges = static_cast<uint64_t>(Edges) * Batch;
    uint64_t TotalVertices = static_cast<uint64_t>(Vertices) * Batch;
    uint64_t FrontierCapacity = TotalVertices;

    uint64_t *DKeys = nullptr;
    uint64_t *DSeeds1 = nullptr;
    uint64_t *DSeeds2 = nullptr;
    uint32_t *DEdgeU = nullptr;
    uint32_t *DEdgeV = nullptr;
    uint32_t *DDegree = nullptr;
    uint32_t *DXorEdge = nullptr;
    uint32_t *DInvalidGraphs = nullptr;
    uint32_t *DEdgePeeled = nullptr;
    uint32_t *DOwnerVertex = nullptr;
    uint32_t *DPeelOrder = nullptr;
    uint32_t *DPeeledCount = nullptr;
    uint32_t *DAssigned = nullptr;
    uint32_t *DVerifyFailures = nullptr;
    FrontierItem *DFrontier = nullptr;
    uint32_t *DFrontierCount = nullptr;

    CheckCuda(cudaMalloc(&DKeys, Keys.size() * sizeof(Keys[0])), "cudaMalloc(DKeys)");
    CheckCuda(cudaMalloc(&DSeeds1, Seeds1.size() * sizeof(Seeds1[0])), "cudaMalloc(DSeeds1)");
    CheckCuda(cudaMalloc(&DSeeds2, Seeds2.size() * sizeof(Seeds2[0])), "cudaMalloc(DSeeds2)");
    CheckCuda(cudaMalloc(&DEdgeU, TotalEdges * sizeof(uint32_t)), "cudaMalloc(DEdgeU)");
    CheckCuda(cudaMalloc(&DEdgeV, TotalEdges * sizeof(uint32_t)), "cudaMalloc(DEdgeV)");
    CheckCuda(cudaMalloc(&DDegree, TotalVertices * sizeof(uint32_t)), "cudaMalloc(DDegree)");
    CheckCuda(cudaMalloc(&DXorEdge, TotalVertices * sizeof(uint32_t)), "cudaMalloc(DXorEdge)");
    CheckCuda(cudaMalloc(&DInvalidGraphs, Batch * sizeof(uint32_t)), "cudaMalloc(DInvalidGraphs)");
    CheckCuda(cudaMalloc(&DEdgePeeled, TotalEdges * sizeof(uint32_t)), "cudaMalloc(DEdgePeeled)");
    CheckCuda(cudaMalloc(&DOwnerVertex, TotalEdges * sizeof(uint32_t)), "cudaMalloc(DOwnerVertex)");
    CheckCuda(cudaMalloc(&DPeelOrder, TotalEdges * sizeof(uint32_t)), "cudaMalloc(DPeelOrder)");
    CheckCuda(cudaMalloc(&DPeeledCount, Batch * sizeof(uint32_t)), "cudaMalloc(DPeeledCount)");
    CheckCuda(cudaMalloc(&DAssigned, TotalVertices * sizeof(uint32_t)), "cudaMalloc(DAssigned)");
    CheckCuda(cudaMalloc(&DVerifyFailures, Batch * sizeof(uint32_t)), "cudaMalloc(DVerifyFailures)");
    CheckCuda(cudaMalloc(&DFrontier, FrontierCapacity * sizeof(FrontierItem)), "cudaMalloc(DFrontier)");
    CheckCuda(cudaMalloc(&DFrontierCount, sizeof(uint32_t)), "cudaMalloc(DFrontierCount)");

    CheckCuda(cudaMemcpy(DKeys, Keys.data(), Keys.size() * sizeof(Keys[0]), cudaMemcpyHostToDevice),
              "cudaMemcpy(DKeys)");
    CheckCuda(cudaMemcpy(DSeeds1, Seeds1.data(), Seeds1.size() * sizeof(Seeds1[0]), cudaMemcpyHostToDevice),
              "cudaMemcpy(DSeeds1)");
    CheckCuda(cudaMemcpy(DSeeds2, Seeds2.data(), Seeds2.size() * sizeof(Seeds2[0]), cudaMemcpyHostToDevice),
              "cudaMemcpy(DSeeds2)");

    CheckCuda(cudaMemset(DEdgeU, 0, TotalEdges * sizeof(uint32_t)), "cudaMemset(DEdgeU)");
    CheckCuda(cudaMemset(DEdgeV, 0, TotalEdges * sizeof(uint32_t)), "cudaMemset(DEdgeV)");
    CheckCuda(cudaMemset(DDegree, 0, TotalVertices * sizeof(uint32_t)), "cudaMemset(DDegree)");
    CheckCuda(cudaMemset(DXorEdge, 0, TotalVertices * sizeof(uint32_t)), "cudaMemset(DXorEdge)");
    CheckCuda(cudaMemset(DInvalidGraphs, 0, Batch * sizeof(uint32_t)), "cudaMemset(DInvalidGraphs)");
    CheckCuda(cudaMemset(DEdgePeeled, 0, TotalEdges * sizeof(uint32_t)), "cudaMemset(DEdgePeeled)");
    CheckCuda(cudaMemset(DOwnerVertex, 0xff, TotalEdges * sizeof(uint32_t)), "cudaMemset(DOwnerVertex)");
    CheckCuda(cudaMemset(DPeelOrder, 0xff, TotalEdges * sizeof(uint32_t)), "cudaMemset(DPeelOrder)");
    CheckCuda(cudaMemset(DPeeledCount, 0, Batch * sizeof(uint32_t)), "cudaMemset(DPeeledCount)");
    CheckCuda(cudaMemset(DAssigned, 0, TotalVertices * sizeof(uint32_t)), "cudaMemset(DAssigned)");
    CheckCuda(cudaMemset(DVerifyFailures, 0, Batch * sizeof(uint32_t)), "cudaMemset(DVerifyFailures)");

    cudaEvent_t Start = nullptr;
    cudaEvent_t Stop = nullptr;
    CheckCuda(cudaEventCreate(&Start), "cudaEventCreate(Start)");
    CheckCuda(cudaEventCreate(&Stop), "cudaEventCreate(Stop)");

    uint32_t BuildBlocks = static_cast<uint32_t>((TotalEdges + Opts.Threads - 1) / Opts.Threads);
    uint32_t VertexBlocks = static_cast<uint32_t>((TotalVertices + Opts.Threads - 1) / Opts.Threads);

    CheckCuda(cudaEventRecord(Start), "cudaEventRecord(Start)");

    BuildGraphsKernel<<<BuildBlocks, Opts.Threads>>>(Edges,
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

    uint32_t Rounds = 0;
    uint32_t FrontierCount = 0;

    for (;;) {
        CheckCuda(cudaMemset(DFrontierCount, 0, sizeof(uint32_t)), "cudaMemset(DFrontierCount)");

        CollectFrontierKernel<<<VertexBlocks, Opts.Threads>>>(Vertices,
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

        ++Rounds;

        uint32_t PeelBlocks = static_cast<uint32_t>((FrontierCount + Opts.Threads - 1) / Opts.Threads);
        PeelFrontierKernel<<<PeelBlocks, Opts.Threads>>>(Edges,
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

    AssignGraphsKernel<<<Batch, 1>>>(Edges,
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

    dim3 VerifyGrid(static_cast<uint32_t>((TotalEdges + Opts.Threads - 1) / Opts.Threads), Batch);
    VerifyGraphsKernel<<<VerifyGrid.x, Opts.Threads>>>(Edges,
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

    float GpuMilliseconds = 0.0f;
    CheckCuda(cudaEventElapsedTime(&GpuMilliseconds, Start, Stop), "cudaEventElapsedTime");

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
        CpuResults[Graph] = RunCpuReference(Graph,
                                            Edges,
                                            Vertices,
                                            EdgeMask,
                                            Keys,
                                            Seeds1,
                                            Seeds2);
    }

    auto CpuStop = std::chrono::steady_clock::now();
    double CpuMilliseconds = std::chrono::duration<double, std::milli>(CpuStop - CpuStart).count();

    uint32_t GpuSuccess = 0;
    uint32_t CpuSuccess = 0;
    uint32_t Mismatches = 0;
    uint32_t VerifiedMismatch = 0;

    for (uint32_t Graph = 0; Graph < Batch; ++Graph) {
        bool ThisGpuSuccess = (InvalidGraphs[Graph] == 0 &&
                               PeeledCount[Graph] == Edges &&
                               VerifyFailures[Graph] == 0);
        bool ThisCpuSuccess = (CpuResults[Graph].Success && CpuResults[Graph].Verified);

        if (ThisGpuSuccess) {
            ++GpuSuccess;
        }
        if (ThisCpuSuccess) {
            ++CpuSuccess;
        }
        if (ThisGpuSuccess != ThisCpuSuccess) {
            ++Mismatches;
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
            ++VerifiedMismatch;
        }
    }

    std::cout
        << "GPU Batched Peeling POC\n"
        << "  Requested edges:    " << Opts.Edges << "\n"
        << "  Actual edges:       " << Edges << "\n"
        << "  Vertices:           " << Vertices << "\n"
        << "  Batch size:         " << Batch << "\n"
        << "  Peel rounds:        " << Rounds << "\n"
        << "  GPU success:        " << GpuSuccess << "/" << Batch << "\n"
        << "  CPU success:        " << CpuSuccess << "/" << Batch << "\n"
        << "  Success mismatches: " << Mismatches << "\n"
        << "  CPU verify issues:  " << VerifiedMismatch << "\n"
        << std::fixed << std::setprecision(3)
        << "  GPU time (ms):      " << GpuMilliseconds << "\n"
        << "  CPU time (ms):      " << CpuMilliseconds << "\n";

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

    return (Mismatches == 0 && VerifiedMismatch == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
