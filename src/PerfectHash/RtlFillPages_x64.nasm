; RtlFillPages x64 assembly routines (NASM).

%include "PerfectHashNasm.inc"

default rel

section .text

global RtlFillPagesNonTemporal_AVX2
global RtlFillPages_AVX2

; HRESULT
; RtlFillPagesNonTemporal_AVX2(
;     _In_ PRTL Rtl,
;     _Out_writes_bytes_all_(NumberOfPages << PAGE_SHIFT) PCHAR Dest,
;     _In_ BYTE Byte,
;     _In_ ULONG NumberOfPages
;     );
RtlFillPagesNonTemporal_AVX2:
    test    PH_ARG4, PH_ARG4
    jz      .ret

    movd            xmm0, PH_ARG3D
    vpbroadcastb    ymm1, xmm0

    mov     r10, -PAGE_SIZE
    sub     PH_ARG2, r10
    mov     rax, r10

    align   16
.loop:
    vmovntdq [PH_ARG2 + rax + 32 * 0], ymm1
    vmovntdq [PH_ARG2 + rax + 32 * 1], ymm1
    vmovntdq [PH_ARG2 + rax + 32 * 2], ymm1
    vmovntdq [PH_ARG2 + rax + 32 * 3], ymm1

    add     rax, 32 * 4
    jnz     .loop

    sub     PH_ARG4, 1
    jz      .done

    sub     PH_ARG2, r10
    mov     rax, r10
    jmp     .loop

.done:
    sfence
    xor     eax, eax
.ret:
    ret

; HRESULT
; RtlFillPages_AVX2(
;     _In_ PRTL Rtl,
;     _Out_writes_bytes_all_(NumberOfPages << PAGE_SHIFT) PCHAR Dest,
;     _In_ BYTE Byte,
;     _In_ ULONG NumberOfPages
;     );
RtlFillPages_AVX2:
    test    PH_ARG4, PH_ARG4
    jz      .ret2

    movd            xmm0, PH_ARG3D
    vpbroadcastb    ymm1, xmm0

    mov     r10, -PAGE_SIZE
    sub     PH_ARG2, r10
    mov     rax, r10

    align   16
.loop2:
    vmovdqa [PH_ARG2 + rax + 32 * 0], ymm1
    vmovdqa [PH_ARG2 + rax + 32 * 1], ymm1
    vmovdqa [PH_ARG2 + rax + 32 * 2], ymm1
    vmovdqa [PH_ARG2 + rax + 32 * 3], ymm1

    add     rax, 32 * 4
    jnz     .loop2

    sub     PH_ARG4, 1
    jz      .done2

    sub     PH_ARG2, r10
    mov     rax, r10
    jmp     .loop2

.done2:
    sfence
    xor     eax, eax
.ret2:
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
