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
        ULONG GlobalIndex;

        GlobalIndex = BlockIndex.x * BlockDim.x + ThreadIndex.x;

        Shared->HashKeysBlockResults = (PHRESULT)(
            RtlOffsetToPointer(
                SharedRaw,
                sizeof(GRAPH_SHARED)
            )
        );
        *GlobalResult = S_OK;
        printf("Shared %p, offset: %p, global: %p\n",
               Shared,
               Shared->HashKeysBlockResults,
               GlobalResult);

        printf("GlobalIndex: %d\n", GlobalIndex);
        printf("threadIdx:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
        printf("blockIdx:(%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z);

        printf("blockDim:(%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
        printf("gridDim:(%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);
    }

    __syncthreads();

    BlockResult = &Shared->HashKeysBlockResults[BlockIndex.x];

    FOR_EACH_1D(Index, NumberOfKeys) {

        if (*BlockResult != S_OK) {
            goto End;
        }

        Key = Keys[Index];

        Vertex1 = ((Key * SEED1) >> SEED3_BYTE1);
        Vertex2 = ((Key * SEED2) >> SEED3_BYTE2);

        if (Vertex1 == Vertex2) {
            printf("Vertex collision! %d == %d\n", Vertex1, Vertex2);
            *BlockResult = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
            *GlobalResult = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
            goto End;
        } else {
            printf("Vertex good %d != %d\n", Vertex1, Vertex2);
        }

        Output[Index].x = Vertex1;
        Output[Index].y = Vertex2;
    }

End:
    __syncthreads();
    return;
}

GLOBAL
VOID
Hello(int i)
{
    printf("Hello! %d\n", i);
}

GLOBAL
VOID
PerfectHashCudaEnterSolvingLoop(
    _In_ PGRAPH Graph
    )
{
    PKEY Keys;
    ULONG NumberOfKeys;
    ULONG BlocksPerGrid;
    ULONG ThreadsPerBlock;
    ULONG SharedMemoryInBytes;
    HRESULT KernelResult = S_OK;
    //HRESULT HashResult = S_OK;
    CU_STREAM HashKeysStream;
    //CU_EVENT HashKeys;
    CU_RESULT CuResult;

    Keys = (PKEY)Graph->DeviceKeys;
    NumberOfKeys = Graph->NumberOfKeys;

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

}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
