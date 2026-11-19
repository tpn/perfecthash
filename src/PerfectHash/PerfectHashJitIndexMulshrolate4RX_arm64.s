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
#elif RAWDOG_KEY_COUNT == 8 && !defined(RAWDOG_NEON_X8)
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

#if RAWDOG_KEY_COUNT == 8 && defined(RAWDOG_NEON_X8)
        // Each argument contributes its low 32 bits, irrespective of the
        // unspecified upper bits of registers at entry.
        bfi     x0, x1, #32, #32
        bfi     x4, x5, #32, #32
        bfi     x2, x3, #32, #32
        bfi     x6, x7, #32, #32
        fmov    d0, x0
        fmov    d1, x4
        mov     v0.d[1], x2
        mov     v1.d[1], x6
        ldr     q4, RawDogSeed1         // Seed1, Seed2 in lanes 0, 2.
        ldr     q5, RawDogSeed4         // Seed4, Seed5 in lanes 0, 2.
        ldr     x12, RawDogAssigned
        ldr     w13, RawDogIndexMask

        mul     v2.4s, v0.4s, v4.s[0]   // First vertices (keys 0-3, 4-7).
        mul     v3.4s, v1.4s, v4.s[0]
        mul     v0.4s, v0.4s, v4.s[2]   // Second vertices.
        mul     v1.4s, v1.4s, v4.s[2]

        // SHL followed by SRI rotates each lane. The JIT patches the shift
        // immediates from the table's seeds before making the code executable.
RawDogRotateFirst:
        shl     v16.4s, v2.4s, #0
        shl     v17.4s, v3.4s, #0
        sri     v16.4s, v2.4s, #32
        sri     v17.4s, v3.4s, #32
RawDogRotateSecond:
        shl     v18.4s, v0.4s, #0
        shl     v19.4s, v1.4s, #0
        sri     v18.4s, v0.4s, #32
        sri     v19.4s, v1.4s, #32
        mul     v2.4s, v16.4s, v5.s[0]
        mul     v3.4s, v17.4s, v5.s[0]
        mul     v0.4s, v18.4s, v5.s[2]
        mul     v1.4s, v19.4s, v5.s[2]
RawDogFinalShift:
        ushr    v2.4s, v2.4s, #32
        ushr    v3.4s, v3.4s, #32
        ushr    v0.4s, v0.4s, #32
        ushr    v1.4s, v1.4s, #32

        // NEON has no indexed gather. Issue the independent assigned loads
        // for each group before summing and writing to its output pointers.
#if RAWDOG_ASSIGNED_BITS == 16
#define LOAD_ASSIGNED ldrh
#define ASSIGNED_SCALE 1
#else
#define LOAD_ASSIGNED ldr
#define ASSIGNED_SCALE 2
#endif
        .macro lookup_four first, second, offset
        umov    w0, \first\().s[0]
        umov    w1, \second\().s[0]
        umov    w2, \first\().s[1]
        umov    w3, \second\().s[1]
        umov    w4, \first\().s[2]
        umov    w5, \second\().s[2]
        umov    w6, \first\().s[3]
        umov    w7, \second\().s[3]
        LOAD_ASSIGNED w0, [x12, w0, uxtw #ASSIGNED_SCALE]
        LOAD_ASSIGNED w1, [x12, w1, uxtw #ASSIGNED_SCALE]
        LOAD_ASSIGNED w2, [x12, w2, uxtw #ASSIGNED_SCALE]
        LOAD_ASSIGNED w3, [x12, w3, uxtw #ASSIGNED_SCALE]
        LOAD_ASSIGNED w4, [x12, w4, uxtw #ASSIGNED_SCALE]
        LOAD_ASSIGNED w5, [x12, w5, uxtw #ASSIGNED_SCALE]
        LOAD_ASSIGNED w6, [x12, w6, uxtw #ASSIGNED_SCALE]
        LOAD_ASSIGNED w7, [x12, w7, uxtw #ASSIGNED_SCALE]
        ldp     x9, x10, [sp, #\offset]
        ldp     x11, x14, [sp, #(\offset + 16)]
        add     w0, w0, w1
        add     w2, w2, w3
        add     w4, w4, w5
        add     w6, w6, w7
        and     w0, w0, w13
        and     w2, w2, w13
        and     w4, w4, w13
        and     w6, w6, w13
        str     w0, [x9]
        str     w2, [x10]
        str     w4, [x11]
        str     w6, [x14]
        .endm

        lookup_four v2, v0, 0
        lookup_four v3, v1, 32
#else
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
#endif
        ret

        // All variants share the same sentinel list, including the NEON
        // kernel that consumes Seed3's bytes as instruction immediates.
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

#if RAWDOG_KEY_COUNT == 8 && defined(RAWDOG_NEON_X8)
        // Byte offsets of the three four-instruction groups to specialize.
NeonShiftOffsets:
        .long RawDogRotateFirst - RAWDOG_FUNCTION
        .long RawDogRotateSecond - RAWDOG_FUNCTION
        .long RawDogFinalShift - RAWDOG_FUNCTION
#endif
