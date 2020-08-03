/*++

Copyright (c) 2020 Trent Nelson <trent@trent.me>

Module Name:

    Graph.cu

Abstract:

    CUDA graph implementation.

--*/

extern "C" {

#include <PerfectHashCuda.h>
#include "../PerfectHash/Graph.h"

#include "Graph.cuh"

#include <cuda_device_runtime_api.h>

#if 0
#define CU_RESULT CUresult
#define CU_STREAM CUstream
#define CU_EVENT CUevent
#else
#define CU_RESULT cudaError_t
#define CU_STREAM cudaStream_t
#define CU_EVENT cudaEvent_t
#endif

//
// Shared memory.
//

extern SHARED ULONG SharedRaw[];

GLOBAL
VOID
HashAllMultiplyShiftR(
    _In_reads_(NumberOfKeys) PKEY Keys,
    _In_ ULONG NumberOfKeys,
    _Out_writes_(NumberOfKeys) PVERTEX_PAIR VertexPairs,
    _In_ PULONG Seeds,
    _Out_ PHRESULT GlobalResult
    )
{
    KEY Key;
    ULONG Index;
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Vertex1;
    ULONG Vertex2;
    PHRESULT BlockResult;
    PINT2 Output = (PINT2)VertexPairs;
    PGRAPH_SHARED Shared = (PGRAPH_SHARED)SharedRaw;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3.AsULong = Seeds[2];

    //
    // If this is thread 0 in the block, initialize the shared memory and set
    // the global result to S_OK.
    //

    if (ThreadIndex.x == 0) {

        Shared->HashKeysBlockResults = (PHRESULT)(
            RtlOffsetToPointer(
                SharedRaw,
                sizeof(GRAPH_SHARED)
            )
        );


        *GlobalResult = S_OK;
    }

    __syncthreads();

    BlockResult = &Shared->HashKeysBlockResults[BlockIndex.x];

    FOR_EACH_1D(Index, NumberOfKeys) {

        //
        // Block-level fast-path exit if we've already detected a vertex
        // collision.
        //

        if (*BlockResult != S_OK) {
            goto End;
        }

        Key = Keys[Index];

        Vertex1 = ((Key * SEED1) >> SEED3_BYTE1);
        Vertex2 = ((Key * SEED2) >> SEED3_BYTE2);

        if (Vertex1 == Vertex2) {

            //
            // Set the block-level and global-level results to indicate
            // collision, then jump to the end.
            //

            *BlockResult = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
            *GlobalResult = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
            goto End;
        }

        //
        // Store the vertex pairs.
        //

        Output[Index].x = Vertex1;
        Output[Index].y = Vertex2;
    }

End:
    __syncthreads();
    return;
}

DEVICE
BOOLEAN
ShouldWeContinueSolving(
    _In_ PGRAPH Graph
    )
{
    return FALSE;
}

GLOBAL
VOID
PerfectHashCudaEnterSolvingLoop(
    _In_ PGRAPH Graph
    )
/*++

Routine Description:

    This is the main entry point for the CUDA graph solving implementation.
    This kernel is intended to be called with a single thread.  It launches
    child kernels dynamically.

Arguments:

    Graph - Supplies a pointer to a GRAPH structure for which solving is to be
        performed.

Return Value:

    N.B. The return value is provided to the caller via Graph->CuKernelResult.

    S_OK - Success.

--*/
{
    PKEY Keys;
    ULONG NumberOfKeys;
    ULONG BlocksPerGrid;
    ULONG ThreadsPerBlock;
    ULONG SharedMemoryInBytes;
    HRESULT Result;
    HRESULT KernelResult = S_OK;
    CU_STREAM HashKeysStream;
    CU_RESULT CuResult;
    PGRAPH NewGraph;
    PGRAPH_VTBL Vtvl;

    //
    // Abort if the kernel is called with more than one thread.
    //

    if (ThreadIndex.x > 0 || BlockIndex.x > 0) {
        KernelResult = PH_E_CU_KERNEL_SOLVE_LOOP_INVALID_DIMENSIONS;
        goto End;
    }

    //
    // Initialize aliases.
    //

    Keys = (PKEY)Graph->DeviceKeys;
    NumberOfKeys = Graph->NumberOfKeys;

    //
    // Initialize the graph's vtbl.
    //

    Vtbl = Graph->Vtbl;
    Vtbl->ShouldWeContinueTryingToSolve = GraphCuShouldWeContinueTryingToSolve;
    Vtbl->Solve = GraphCuSolve;
    Vtbl->LoadNewSeeds = GraphCuLoadNewSeeds;
    Vtbl->Reset = GraphCuResult;

    //
    // Begin the solving loop.
    //

    while (Graph->Vtbl->ShouldWeContinueTryingToSolve(Graph)) {

        Result = Graph->Vtbl->LoadNewSeeds(Graph);
        if (FAILED(Result)) {

            //
            // N.B. This will need to be adjusted when we support the notion
            //      of no more seed data (PH_E_NO_MORE_SEEDS).
            //

            PH_ERROR(GraphLoadNewSeeds, Result);
            break;
        }

        Result = Graph->Vtbl->Reset(Graph);
        if (FAILED(Result)) {
            PH_ERROR(GraphReset, Result);
            break;
        } else if (Result != PH_S_CONTINUE_GRAPH_SOLVING) {
            break;
        }

        NewGraph = NULL;
        Result = Graph->Vtbl->Solve(Graph, &NewGraph);
        if (FAILED(Result)) {
            PH_ERROR(GraphSolve, Result);
            break;
        }

        if (Result == PH_S_STOP_GRAPH_SOLVING ||
            Result == PH_S_GRAPH_SOLVING_STOPPED) {
            if (NewGraph != NULL) {
                PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
            }
            break;
        }

        if (Result == PH_S_USE_NEW_GRAPH_FOR_SOLVING) {

            if (NewGraph == NULL) {
                PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
            }

            //
            // Acquire the new graph's lock and release the existing
            // graph's lock.
            //

            AcquireGraphLockExclusive(NewGraph);
            ReleaseGraphLockExclusive(Graph);

            if (!IsGraphInfoLoaded(NewGraph) ||
                NewGraph->LastLoadedNumberOfVertices <
                Graph->NumberOfVertices) {

                Result = NewGraph->Vtbl->LoadInfo(NewGraph);
                if (FAILED(Result)) {
                    PH_ERROR(GraphLoadInfo_NewGraph, Result);
                    goto End;
                }
            }

            Graph = NewGraph;
            continue;
        }

        //
        // Invariant check: result should always be PH_S_CONTINUE_GRAPH_SOLVING
        // at this point.
        //

        ASSERT(Result == PH_S_CONTINUE_GRAPH_SOLVING);

        //
        // Continue the loop and attempt another solve.
        //

    }

    //Graph->Seeds[0] = 2344307159;
    //Graph->Seeds[1] = 2331343182;
    Graph->Seeds[0] = 83;
    Graph->Seeds[1] = 5;
    //Graph->Seeds[2] = 2827;
    Graph->Seeds[2] = 0x0101;

    //printf("Entered solving loop.\n");
    printf("Entered Solving Loop! Graph: %p, Keys: %p\n", Graph, Keys);
    printf("NumberOfKeys: %u\n", NumberOfKeys);
    printf("Key[1]: %u\n", Keys[1]);

    printf("threadIdx:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("blockIdx:(%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z);

    printf("blockDim:(%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
    printf("gridDim:(%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);

    BlocksPerGrid = Graph->CuBlocksPerGrid;
    ThreadsPerBlock = Graph->CuThreadsPerBlock;

    CuResult = cudaStreamCreateWithFlags(&HashKeysStream,
                                         cudaStreamNonBlocking);
    if (CU_FAILED(CuResult)) {
        printf("cuStreamCreate() failed: %x\n", CuResult);
        KernelResult = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto End;
    }

    Graph->CuHashKeysResult = E_FAIL;

    SharedMemoryInBytes = (

        //
        // Account for the GRAPH_SHARED structure.
        //

        sizeof(GRAPH_SHARED) +

        //
        // Account for the array of result codes (one per block) for HashKeys.
        //

        (sizeof(HRESULT) * BlocksPerGrid)

    );

    HashAllMultiplyShiftR<<<
        BlocksPerGrid,
        ThreadsPerBlock,
        SharedMemoryInBytes,
        HashKeysStream
    >>>(
        Keys,
        NumberOfKeys,
        Graph->VertexPairs,
        Graph->Seeds,
        &Graph->CuHashKeysResult
    );

    CuResult = cudaDeviceSynchronize();
    if (CU_FAILED(CuResult)) {
        printf("cudaDeviceSynchronize() failed: %x\n", CuResult);
        KernelResult = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto End;
    }

    KernelResult = Graph->CuHashKeysResult;

    //printf("HashResult: %x\n", HashResult);

    //ClockBlock(1000);
    //Graph->CuKernelResult = HashResult;
End:
    printf("Leaving Solving Loop!  Result: %x\n", KernelResult);
    Graph->CuKernelResult = KernelResult;
}

} // extern "C"

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
