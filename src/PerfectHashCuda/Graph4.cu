/*++

Copyright (c) 2023 Trent Nelson <trent@trent.me>

Module Name:

    Graph4.cu

Abstract:

    CUDA graph implementation.

--*/

#define PH_CU

#include <PerfectHash.h>

EXTERN_C_BEGIN
#include "../PerfectHash/CuDeviceAttributes.h"
//#include "../PerfectHash/Cu.h"
#include "../PerfectHash/Graph.h"

#include <cuda.h>
#include <cuda_device_runtime_api.h>
EXTERN_C_END

#include <curand_kernel.h>

#include "Graph.cuh"

#include <stdio.h>

//
// Define helper macros.
//

#define EMPTY ((VERTEX)-1)
#define IsEmpty(Value) ((ULONG)Value == EMPTY)

//
// When a solution has been found and the assignment step begins, the initial
// value assigned to a vertex is govered by the following macro.
//

#define INITIAL_ASSIGNMENT_VALUE 0

//
// Shared memory.
//

extern SHARED ULONG SharedRaw[];

EXTERN_C
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
    HRESULT Result = S_OK;
    //PGRAPH NewGraph;

    //
    // Abort if the kernel is called with more than one thread.
    //

    if (GridDim.x > 1  || GridDim.y > 1  || GridDim.z > 1 ||
        BlockDim.x > 1 || BlockDim.y > 1 || BlockDim.z > 1)
    {
        Result = PH_E_CU_KERNEL_SOLVE_LOOP_INVALID_DIMENSIONS;
        goto End;
    }

    if (Graph->SizeOfStruct != sizeof(GRAPH)) {
        printf("%u != %u!\n", (ULONG)Graph->SizeOfStruct, (ULONG)sizeof(GRAPH));
        return;
    }

    printf("Entered solving loop!  sizeof(GRAPH): %d\n", (LONG)sizeof(GRAPH));

#if 0
    if (Graph->CuKernelContext == NULL) {
        Result = GraphCuCreateKernelContext(Graph);
        if (FAILED(Result)) {
            PH_ERROR(GraphCuCreateKernelContext, Result);
            return;
        }
        printf("Created context successfully.\n");
    }

    //
    // Begin the solving loop.
    //

    GraphCuTrySolve(Graph);

    //
    // We're done, finish up.
    //

    goto End;
#endif

#if 0
Error:

    if (SUCCEEDED(Result)) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

#endif
End:
    Graph->CuKernelResult = Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab filetype=cuda formatoptions=croql   :
