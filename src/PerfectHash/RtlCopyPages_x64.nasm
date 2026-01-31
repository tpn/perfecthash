; RtlCopyPages x64 assembly routines (NASM).

%include "PerfectHashNasm.inc"

default rel

section .text

global RtlCopyPagesNonTemporal_AVX2
global RtlCopyPages_AVX2

; HRESULT
; RtlCopyPagesNonTemporal_AVX2(
;     _In_ PRTL Rtl,
;     _In_ PCHAR Destination,
;     _In_ PCHAR Source,
;     _In_ ULONG NumberOfPages
;     );
RtlCopyPagesNonTemporal_AVX2:
    test    PH_ARG4, PH_ARG4
    jz      .ret

    mov     r10, -PAGE_SIZE
    mov     r11, 128
    sub     PH_ARG2, r10
    sub     PH_ARG3, r10
    mov     rax, r10
    jmp     .loop

    align   16
.loop:
    prefetchnta [PH_ARG3 + rax + 64 * 0]
    prefetchnta [PH_ARG3 + rax + 64 * 1]

    vmovntdqa ymm0, [PH_ARG3 + rax + 32 * 0]
    vmovntdqa ymm1, [PH_ARG3 + rax + 32 * 1]
    vmovntdqa ymm2, [PH_ARG3 + rax + 32 * 2]
    vmovntdqa ymm3, [PH_ARG3 + rax + 32 * 3]

    vmovntdq  [PH_ARG2 + rax + 32 * 0], ymm0
    vmovntdq  [PH_ARG2 + rax + 32 * 1], ymm1
    vmovntdq  [PH_ARG2 + rax + 32 * 2], ymm2
    vmovntdq  [PH_ARG2 + rax + 32 * 3], ymm3

    add     rax, r11
    jnz     .loop

    sub     PH_ARG4, 1
    jz      .done

    sub     PH_ARG2, r10
    sub     PH_ARG3, r10
    mov     rax, r10
    jmp     .loop

.done:
    sfence
    xor     eax, eax
.ret:
    ret

; HRESULT
; RtlCopyPages_AVX2(
;     _In_ PRTL Rtl,
;     _In_ PCHAR Destination,
;     _In_ PCHAR Source,
;     _In_ ULONG NumberOfPages
;     );
RtlCopyPages_AVX2:
    test    PH_ARG4, PH_ARG4
    jz      .ret2

    mov     r10, -PAGE_SIZE
    mov     r11, 128
    sub     PH_ARG2, r10
    sub     PH_ARG3, r10
    mov     rax, r10
    jmp     .loop2

    align   16
.loop2:
    prefetchnta [PH_ARG3 + rax + 64 * 0]
    prefetchnta [PH_ARG3 + rax + 64 * 1]

    vmovdqa  ymm0, [PH_ARG3 + rax + 32 * 0]
    vmovdqa  ymm1, [PH_ARG3 + rax + 32 * 1]
    vmovdqa  ymm2, [PH_ARG3 + rax + 32 * 2]
    vmovdqa  ymm3, [PH_ARG3 + rax + 32 * 3]

    vmovdqa  [PH_ARG2 + rax + 32 * 0], ymm0
    vmovdqa  [PH_ARG2 + rax + 32 * 1], ymm1
    vmovdqa  [PH_ARG2 + rax + 32 * 2], ymm2
    vmovdqa  [PH_ARG2 + rax + 32 * 3], ymm3

    add     rax, r11
    jnz     .loop2

    sub     PH_ARG4, 1
    jz      .done2

    sub     PH_ARG2, r10
    sub     PH_ARG3, r10
    mov     rax, r10
    jmp     .loop2

.done2:
    sfence
    xor     eax, eax
.ret2:
    ret

section .note.GNU-stack noalloc noexec nowrite progbits