// Copyright (c) 2026 Trent Nelson <trent@trent.me>
//
// Mulshrolate4RX RawDog index for 1, 4, 8, or 16 keys.  The batched
// entry points receive all keys by value, followed by one output pointer
// per key.  The scalar entry point returns the index in w0.

        .text
        .p2align 4
        .globl RAWDOG_FUNCTION
RAWDOG_FUNCTION:

#if RAWDOG_KEY_COUNT == 4
#define FRAME_SIZE 64
        stp     x29, x30, [sp, #-FRAME_SIZE]!
        mov     x29, sp
        stp     w0, w1, [x29, #16]
        stp     w2, w3, [x29, #24]
        stp     x4, x5, [x29, #32]
        stp     x6, x7, [x29, #48]
        add     x8, x29, #32
#elif RAWDOG_KEY_COUNT == 8
#define FRAME_SIZE 48
        stp     x29, x30, [sp, #-FRAME_SIZE]!
        mov     x29, sp
        stp     w0, w1, [x29, #16]
        stp     w2, w3, [x29, #24]
        stp     w4, w5, [x29, #32]
        stp     w6, w7, [x29, #40]
        add     x8, x29, #FRAME_SIZE
#elif RAWDOG_KEY_COUNT == 16
#define FRAME_SIZE 80
#include "PerfectHashJitIndex32x16Arm64.inc"
        add     x8, x29, #(FRAME_SIZE + 8 * STACK_KEY_SLOT_SIZE)
#endif

        ldr     x10, RawDogAssigned
        ldr     w11, RawDogSeed3Byte1
        ldr     w3, RawDogSeed3Byte2
        ldr     w4, RawDogSeed3Byte3
        ldr     w14, RawDogSeed1
        ldr     w15, RawDogSeed2
        ldr     w16, RawDogSeed4
        ldr     w6, RawDogSeed5
        ldr     w17, RawDogIndexMask

#if RAWDOG_KEY_COUNT > 1
        add     x9, x29, #16            // Start of packed keys.
        mov     w7, #RAWDOG_KEY_COUNT
1:
        ldr     w0, [x9], #4
#endif
        mul     w1, w0, w14
        mul     w2, w0, w15
        rorv    w1, w1, w3
        rorv    w2, w2, w4
        mul     w1, w1, w16
        mul     w2, w2, w6
        lsrv    w1, w1, w11
        lsrv    w2, w2, w11
#if RAWDOG_ASSIGNED_BITS == 16
        ldrh    w1, [x10, w1, uxtw #1]
        ldrh    w2, [x10, w2, uxtw #1]
#else
        ldr     w1, [x10, w1, uxtw #2]
        ldr     w2, [x10, w2, uxtw #2]
#endif
        add     w0, w1, w2
        and     w0, w0, w17

#if RAWDOG_KEY_COUNT > 1
        ldr     x12, [x8], #8           // Next output pointer.
        str     w0, [x12]
        subs    w7, w7, #1
        b.ne    1b
        ldp     x29, x30, [sp], #FRAME_SIZE
#endif
        ret

        .p2align 3
RawDogAssigned:   .quad 0xA1A1A1A1A1A1A1A1
RawDogSeed1:      .quad 0xB1B1B1B1B1B1B1B1
RawDogSeed2:      .quad 0xC1C1C1C1C1C1C1C1
RawDogSeed3Byte1: .quad 0xD1D1D1D1D1D1D1D1
RawDogSeed3Byte2: .quad 0xE1E1E1E1E1E1E1E1
RawDogSeed3Byte3: .quad 0xD2D2D2D2D2D2D2D2
RawDogSeed4:      .quad 0xB2B2B2B2B2B2B2B2
RawDogSeed5:      .quad 0xB3B3B3B3B3B3B3B3
RawDogIndexMask:  .quad 0x2121212121212121
