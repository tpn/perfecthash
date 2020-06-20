/*++

Copyright (c) 2020 Trent Nelson <trent@trent.me>

Module Name:

    Random.cpp

Abstract:

    Temporary stub for MTG32 setup routines.

--*/

#define EXTERN_C extern "C"
#define EXTERN_C_BEGIN EXTERN_C {
#define EXTERN_C_END }

#include <curand_kernel.h>

EXTERN_C
int
MakeMTG32Constants(void)
{
    return 0;
}

EXTERN_C
int
MakeMTG32KernelState(void)
{
    return 0;
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
