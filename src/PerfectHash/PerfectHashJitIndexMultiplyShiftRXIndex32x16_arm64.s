//++
//
// Copyright (c) 2026 Trent Nelson <trent@trent.me>
//
// Module Name:
//
//   PerfectHashJitIndexMultiplyShiftRXIndex32x16_arm64.s
//
// Abstract:
//
//   This module implements the MultiplyShiftRX Index32x16() routine as a
//   position-independent blob suitable for RawDog JIT patching.
//
//--

        .text
        .p2align 4

        .globl PerfectHashJitIndexMultiplyShiftRXIndex32x16_arm64

//+++
//
// VOID
// PerfectHashJitIndexMultiplyShiftRXIndex32x16_arm64(
//     _In_ ULONG Key1,
//     _In_ ULONG Key2,
//     _In_ ULONG Key3,
//     _In_ ULONG Key4,
//     _In_ ULONG Key5,
//     _In_ ULONG Key6,
//     _In_ ULONG Key7,
//     _In_ ULONG Key8,
//     _In_ ULONG Key9,
//     _In_ ULONG Key10,
//     _In_ ULONG Key11,
//     _In_ ULONG Key12,
//     _In_ ULONG Key13,
//     _In_ ULONG Key14,
//     _In_ ULONG Key15,
//     _In_ ULONG Key16,
//     _Out_ PULONG Index1,
//     _Out_ PULONG Index2,
//     _Out_ PULONG Index3,
//     _Out_ PULONG Index4,
//     _Out_ PULONG Index5,
//     _Out_ PULONG Index6,
//     _Out_ PULONG Index7,
//     _Out_ PULONG Index8,
//     _Out_ PULONG Index9,
//     _Out_ PULONG Index10,
//     _Out_ PULONG Index11,
//     _Out_ PULONG Index12,
//     _Out_ PULONG Index13,
//     _Out_ PULONG Index14,
//     _Out_ PULONG Index15,
//     _Out_ PULONG Index16
//     );
//
// Routine Description:
//
//   This routine implements the MultiplyShiftRX index functionality for sixteen
//   keys.  It is designed to be patched in-place by replacing the sentinel
//   values in the embedded data block that follows the routine.
//
//--

PerfectHashJitIndexMultiplyShiftRXIndex32x16_arm64:

#include "PerfectHashJitIndex32x16Arm64.inc"

        ldr     x10, RawDogAssigned
        ldr     w11, RawDogSeed3Byte1
        ldr     w14, RawDogSeed1
        ldr     w15, RawDogSeed2
        ldr     w6, RawDogIndexMask

        add     x9, x29, #16                 // Key base.
        add     x17, x29, #(80 + 8 * STACK_KEY_SLOT_SIZE) // Output pointer base.
        mov     w8, wzr

1:
        ldr     w0, [x9, w8, uxtw #2]

        mov     w1, w0
        mul     w1, w1, w14
        lsrv    w1, w1, w11

        mov     w2, w0
        mul     w2, w2, w15
        lsrv    w2, w2, w11

        ldr     w1, [x10, w1, uxtw #2]
        ldr     w2, [x10, w2, uxtw #2]

        add     w0, w1, w2
        and     w0, w0, w6

        ldr     x12, [x17, w8, uxtw #3]
        str     w0, [x12]

        add     w8, w8, #1
        cmp     w8, #16
        b.lo    1b

        ldp     x29, x30, [sp], #80
        ret

        .p2align 3
RawDogAssigned:
        .quad 0xA1A1A1A1A1A1A1A1

RawDogSeed1:
        .quad 0xB1B1B1B1B1B1B1B1

RawDogSeed2:
        .quad 0xC1C1C1C1C1C1C1C1

RawDogSeed3Byte1:
        .quad 0xD1D1D1D1D1D1D1D1

RawDogIndexMask:
        .quad 0x2121212121212121