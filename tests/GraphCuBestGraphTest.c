#include "stdafx.h"
#include <stdio.h>

static PERFECT_HASH_CONTEXT Context;
static PH_CU_SOLVE_CONTEXT SolveContext;
static GRAPH Cpu, CpuSpare, Gpu, GpuSpare;

int main(void)
{
    Cpu.Context = CpuSpare.Context = Gpu.Context = GpuSpare.Context = &Context;
    Gpu.Flags.IsCuGraph = GpuSpare.Flags.IsCuGraph = TRUE;
    Gpu.CuSolveContext = GpuSpare.CuSolveContext = &SolveContext;
    Gpu.CuHostGraph = GpuSpare.CuHostGraph = &Gpu;
    Gpu.CuHostSpareGraph = GpuSpare.CuHostSpareGraph = &GpuSpare;
    CpuSpare.Flags.IsSpare = GpuSpare.Flags.IsSpare = TRUE;
    Context.SpareGraph = &CpuSpare;

    // The CPU wins first, so its worker continues on the pooled CPU spare.
    if (GraphGetReplacementForNewBest(&Cpu, NULL) != &CpuSpare ||
        Context.SpareGraph != NULL || CpuSpare.Flags.IsSpare) {
        fprintf(stderr, "The CPU worker did not claim its spare graph.\n");
        return 1;
    }

    // CUDA displaces the CPU winner. The CPU graph returns to the pool and
    // the CUDA worker takes its own sibling, preserving stream ownership.
    if (GraphGetReplacementForNewBest(&Gpu, &Cpu) != &GpuSpare ||
        Context.SpareGraph != &Cpu || GpuSpare.Flags.IsSpare) {
        fprintf(stderr, "The CUDA worker did not retain its private graph pair.\n");
        return 1;
    }

    // The CPU wins again and must reclaim the pooled CPU graph.
    if (GraphGetReplacementForNewBest(&CpuSpare, &Gpu) != &Cpu ||
        Context.SpareGraph != NULL) {
        fprintf(stderr, "The displaced CUDA winner crossed into the CPU worker.\n");
        return 1;
    }

    // A CUDA worker currently solving on its spare rotates back to its primary.
    if (GraphGetReplacementForNewBest(&GpuSpare, &CpuSpare) != &Gpu ||
        Context.SpareGraph != &CpuSpare) {
        fprintf(stderr, "The CUDA worker could not resume on its primary graph.\n");
        return 1;
    }
    return 0;
}
