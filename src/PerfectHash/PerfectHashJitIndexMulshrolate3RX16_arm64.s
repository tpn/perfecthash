//++
//
// Copyright (c) 2026 Trent Nelson <trent@trent.me>
//
// Module Name:
//
//   PerfectHashJitIndexMulshrolate3RX16_arm64.s
//
// Abstract:
//
//   This module implements the Mulshrolate3RX Index() routine for 16-bit
//   assigned tables as a position-independent blob suitable for RawDog JIT
//   patching.
//
//--

        .text
        .p2align 4

        .globl PerfectHashJitIndexMulshrolate3RX16_arm64

//+++
//
// ULONG
// PerfectHashJitIndexMulshrolate3RX16_arm64(
//     _In_ ULONG Key
//     );
//
// Routine Description:
//
//   This routine implements the Mulshrolate3RX index functionality for tables
//   using 16-bit assigned elements.  It is designed to be patched in-place by
//   replacing the sentinel values in the embedded data block that follows the
//   routine.
//
// Arguments:
//
//   Key (w0) - Supplies the key for which an index is to be obtained.
//
// Return Value:
//
//   The index corresponding to the given key.
//
//--

PerfectHashJitIndexMulshrolate3RX16_arm64:

        ldr     x10, RawDogAssigned           // Load assigned base address.

        ldr     w11, RawDogSeed3Byte1

        mov     w1, w0                        // Copy key into w1.
        ldr     w2, RawDogSeed1               // Load seed1.
        mul     w1, w1, w2                    // Vertex1 = Key * Seed1.
        ldr     w3, RawDogSeed3Byte2
        rorv    w1, w1, w3                    // Vertex1 = ror(Vertex1, Seed3_Byte2).
        ldr     w4, RawDogSeed4
        mul     w1, w1, w4                    // Vertex1 *= Seed4.
        lsrv    w1, w1, w11                   // Vertex1 >>= Seed3_Byte1.

        mov     w2, w0                        // Copy key into w2.
        ldr     w5, RawDogSeed2               // Load seed2.
        mul     w2, w2, w5                    // Vertex2 = Key * Seed2.
        ldr     w6, RawDogSeed3Byte3
        rorv    w2, w2, w6                    // Vertex2 = ror(Vertex2, Seed3_Byte3).
        lsrv    w2, w2, w11                   // Vertex2 >>= Seed3_Byte1.

        ldrh    w1, [x10, w1, uxtw #1]        // Load vertex1.
        ldrh    w2, [x10, w2, uxtw #1]        // Load vertex2.

        add     w0, w1, w2                    // Add vertices.
        ldr     w7, RawDogIndexMask
        and     w0, w0, w7

        ret                                   // Return.

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

RawDogSeed4:
        .quad 0xB2B2B2B2B2B2B2B2

RawDogIndexMask:
        .quad 0x2121212121212121
