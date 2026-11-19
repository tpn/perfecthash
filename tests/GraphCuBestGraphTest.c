#include "stdafx.h"
#include <stdio.h>

static PERFECT_HASH_CONTEXT Context;
static PH_CU_SOLVE_CONTEXT SolveContext, OtherSolveContext;
static GRAPH Cpu, CpuSpare, Gpu, GpuSpare;
static GRAPH OtherGpu, OtherGpuSpare;

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

    // A different CUDA worker must continue on its own stream and graph pair.
    OtherGpu.Context = OtherGpuSpare.Context = &Context;
    OtherGpu.Flags.IsCuGraph = OtherGpuSpare.Flags.IsCuGraph = TRUE;
    OtherGpu.CuSolveContext = OtherGpuSpare.CuSolveContext = &OtherSolveContext;
    OtherGpu.CuHostGraph = OtherGpuSpare.CuHostGraph = &OtherGpu;
    OtherGpu.CuHostSpareGraph = OtherGpuSpare.CuHostSpareGraph = &OtherGpuSpare;
    OtherGpuSpare.Flags.IsSpare = TRUE;
    if (GraphGetReplacementForNewBest(&OtherGpu, &GpuSpare) != &OtherGpuSpare ||
        GraphGetReplacementForNewBest(&Gpu, &OtherGpu) != &GpuSpare ||
        Context.SpareGraph != &CpuSpare) {
        fprintf(stderr, "A CUDA worker acquired another worker's graph.\n");
        return 1;
    }
    return 0;
}
