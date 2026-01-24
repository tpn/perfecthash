//++
//
// Copyright (c) 2026 Trent Nelson <trent@trent.me>
//
// Module Name:
//
//   PerfectHashJitIndexMulshrolate1RXIndex32x8_arm64.s
//
// Abstract:
//
//   This module implements the Mulshrolate1RX Index32x8() routine as a
//   position-independent blob suitable for RawDog JIT patching.
//
//--

        .text
        .p2align 4

        .globl PerfectHashJitIndexMulshrolate1RXIndex32x8_arm64

//+++
//
// VOID
// PerfectHashJitIndexMulshrolate1RXIndex32x8_arm64(
//     _In_ ULONG Key1,
//     _In_ ULONG Key2,
//     _In_ ULONG Key3,
//     _In_ ULONG Key4,
//     _In_ ULONG Key5,
//     _In_ ULONG Key6,
//     _In_ ULONG Key7,
//     _In_ ULONG Key8,
//     _Out_ PULONG Index1,
//     _Out_ PULONG Index2,
//     _Out_ PULONG Index3,
//     _Out_ PULONG Index4,
//     _Out_ PULONG Index5,
//     _Out_ PULONG Index6,
//     _Out_ PULONG Index7,
//     _Out_ PULONG Index8
//     );
//
// Routine Description:
//
//   This routine implements the Mulshrolate1RX index functionality for eight
//   keys.  It is designed to be patched in-place by replacing the sentinel
//   values in the embedded data block that follows the routine.
//
//--

PerfectHashJitIndexMulshrolate1RXIndex32x8_arm64:

        stp     x29, x30, [sp, #-48]!
        mov     x29, sp

        str     w0, [x29, #16]
        str     w1, [x29, #20]
        str     w2, [x29, #24]
        str     w3, [x29, #28]
        str     w4, [x29, #32]
        str     w5, [x29, #36]
        str     w6, [x29, #40]
        str     w7, [x29, #44]

        ldr     x10, RawDogAssigned
        ldr     w11, RawDogSeed3Byte1
        ldr     w3, RawDogSeed3Byte2
        ldr     w14, RawDogSeed1
        ldr     w15, RawDogSeed2
        ldr     w16, RawDogIndexMask

        add     x9, x29, #16                 // Key base.
        add     x17, x29, #48                // Output pointer base.
        mov     w8, wzr

1:
        ldr     w0, [x9, w8, uxtw #2]

        mov     w1, w0
        mul     w1, w1, w14
        rorv    w1, w1, w3
        lsrv    w1, w1, w11

        mov     w2, w0
        mul     w2, w2, w15
        lsrv    w2, w2, w11

        ldr     w1, [x10, w1, uxtw #2]
        ldr     w2, [x10, w2, uxtw #2]

        add     w0, w1, w2
        and     w0, w0, w16

        ldr     x12, [x17, w8, uxtw #3]
        str     w0, [x12]

        add     w8, w8, #1
        cmp     w8, #8
        b.lo    1b

        ldp     x29, x30, [sp], #48
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

RawDogSeed3Byte2:
        .quad 0xE1E1E1E1E1E1E1E1

RawDogSeed3Byte3:
        .quad 0xD2D2D2D2D2D2D2D2

RawDogIndexMask:
        .quad 0x2121212121212121
