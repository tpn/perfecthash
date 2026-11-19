// Pack the sixteen incoming 32-bit keys into contiguous local storage.
// Darwin packs stack arguments at natural alignment. AAPCS64 and Windows
// allocate an eight-byte stack slot for each of the final eight keys.
#ifdef __APPLE__
#define STACK_KEY_SLOT_SIZE 4
#else
#define STACK_KEY_SLOT_SIZE 8
#endif

        stp     x29, x30, [sp, #-80]!
        mov     x29, sp
        stp     w0, w1, [x29, #16]
        stp     w2, w3, [x29, #24]
        stp     w4, w5, [x29, #32]
        stp     w6, w7, [x29, #40]

        add     x12, x29, #80
        ldr     w0, [x12, #(0 * STACK_KEY_SLOT_SIZE)]
        ldr     w1, [x12, #(1 * STACK_KEY_SLOT_SIZE)]
        ldr     w2, [x12, #(2 * STACK_KEY_SLOT_SIZE)]
        ldr     w3, [x12, #(3 * STACK_KEY_SLOT_SIZE)]
        ldr     w4, [x12, #(4 * STACK_KEY_SLOT_SIZE)]
        ldr     w5, [x12, #(5 * STACK_KEY_SLOT_SIZE)]
        ldr     w6, [x12, #(6 * STACK_KEY_SLOT_SIZE)]
        ldr     w7, [x12, #(7 * STACK_KEY_SLOT_SIZE)]
        stp     w0, w1, [x29, #48]
        stp     w2, w3, [x29, #56]
        stp     w4, w5, [x29, #64]
        stp     w6, w7, [x29, #72]
