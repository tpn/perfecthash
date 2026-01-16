; PerfectHash memset (NASM).

%include "PerfectHashNasm.inc"

default rel

section .text

global memset

; void *memset(void *dest, int c, size_t n)
memset:
    mov     rax, PH_ARG1
    mov     r10, PH_ARG1
    mov     rcx, PH_ARG3
    movzx   r11d, PH_ARG2B
    test    rcx, rcx
    jz      .done

.loop:
    mov     byte [r10], r11b
    inc     r10
    dec     rcx
    jnz     .loop

.done:
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
