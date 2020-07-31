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
    _In_ PULONG Seeds
    )
{
    ULONG Index;
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG Key;
    PINT2 Output = (PINT2)VertexPairs;
    PHRESULT Result;
    PGRAPH_SHARED Shared = (PGRAPH_SHARED)SharedRaw;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3.AsULong = Seeds[2];

    //Result = &Shared->HashKeysBlockResults[BlockIndex.x];

    FOR_EACH_1D(Index, NumberOfKeys) {

        /*
        if (*Result != S_OK) {
            goto End;
        }
        */

        Key = Keys[Index];

        Vertex1 = ((Key * SEED1) >> SEED3_BYTE1);
        Vertex2 = ((Key * SEED2) >> SEED3_BYTE2);

        /*
        if (Vertex1 == Vertex2) {
            //*Result = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
            goto End;
        }
        */

        Output[Index].x = Vertex1;
        Output[Index].y = Vertex2;
    }

    /*
End:
    __syncthreads();
    return;
    */
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
    HRESULT HashResult = S_OK;
    PGRAPH_SHARED Shared = (PGRAPH_SHARED)SharedRaw;

    Shared->HashKeysBlockResults = (PHRESULT)(
        RtlOffsetToPointer(
            SharedRaw,
            sizeof(GRAPH_SHARED)
        )
    );

    printf("Shared %p, offset: %p\n", Shared, Shared->HashKeysBlockResults);

    Keys = (PKEY)Graph->DeviceKeys;
    NumberOfKeys = Graph->NumberOfKeys;

    Graph->Seeds[0] = 2344307159;
    Graph->Seeds[1] = 2331343182;
    Graph->Seeds[2] = 2827;

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

    Hello<<<1, 1>>>(1);

    HashAllMultiplyShiftR<<<BlocksPerGrid, ThreadsPerBlock>>>(
        Keys,
        NumberOfKeys,
        Graph->VertexPairs,
        Graph->Seeds
    );

    printf("HashResult: %x\n", HashResult);

    //ClockBlock(1000);
    Graph->CuKernelResult = HashResult;
    printf("Leaving Solving Loop!\n");
}

}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
