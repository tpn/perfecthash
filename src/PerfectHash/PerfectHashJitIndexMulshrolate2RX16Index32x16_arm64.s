//++
//
// Copyright (c) 2026 Trent Nelson <trent@trent.me>
//
// Module Name:
//
//   PerfectHashJitIndexMulshrolate2RX16Index32x16_arm64.s
//
// Abstract:
//
//   This module implements the Mulshrolate2RX Index32x16() routine for 16-bit
//   assigned tables as a position-independent blob suitable for RawDog JIT
//   patching.
//
//--

        .text
        .p2align 4

        .globl PerfectHashJitIndexMulshrolate2RX16Index32x16_arm64

//+++
//
// VOID
// PerfectHashJitIndexMulshrolate2RX16Index32x16_arm64(
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
//   This routine implements the Mulshrolate2RX index functionality for sixteen
//   keys against 16-bit assigned table data.  It is designed to be patched
//   in-place by replacing the sentinel values in the embedded data block that
//   follows the routine.
//
//--

PerfectHashJitIndexMulshrolate2RX16Index32x16_arm64:

        stp     x29, x30, [sp, #-80]!
        mov     x29, sp

        str     w0, [x29, #16]
        str     w1, [x29, #20]
        str     w2, [x29, #24]
        str     w3, [x29, #28]
        str     w4, [x29, #32]
        str     w5, [x29, #36]
        str     w6, [x29, #40]
        str     w7, [x29, #44]

        add     x12, x29, #80                // Stack args base.
        ldr     w0, [x12, #0]
        str     w0, [x29, #48]
        ldr     w0, [x12, #4]
        str     w0, [x29, #52]
        ldr     w0, [x12, #8]
        str     w0, [x29, #56]
        ldr     w0, [x12, #12]
        str     w0, [x29, #60]
        ldr     w0, [x12, #16]
        str     w0, [x29, #64]
        ldr     w0, [x12, #20]
        str     w0, [x29, #68]
        ldr     w0, [x12, #24]
        str     w0, [x29, #72]
        ldr     w0, [x12, #28]
        str     w0, [x29, #76]

        ldr     x10, RawDogAssigned
        ldr     w11, RawDogSeed3Byte1
        ldr     w3, RawDogSeed3Byte2
        ldr     w4, RawDogSeed3Byte3
        ldr     w14, RawDogSeed1
        ldr     w15, RawDogSeed2
        ldr     w16, RawDogIndexMask

        add     x9, x29, #16                 // Key base.
        add     x17, x29, #112               // Output pointer base.
        mov     w8, wzr

1:
        ldr     w0, [x9, w8, uxtw #2]

        mov     w1, w0
        mul     w1, w1, w14
        rorv    w1, w1, w3
        lsrv    w1, w1, w11

        mov     w2, w0
        mul     w2, w2, w15
        rorv    w2, w2, w4
        lsrv    w2, w2, w11

        ldrh    w1, [x10, w1, uxtw #1]
        ldrh    w2, [x10, w2, uxtw #1]

        add     w0, w1, w2
        and     w0, w0, w16

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

RawDogSeed3Byte2:
        .quad 0xE1E1E1E1E1E1E1E1

RawDogSeed3Byte3:
        .quad 0xD2D2D2D2D2D2D2D2

RawDogIndexMask:
        .quad 0x2121212121212121