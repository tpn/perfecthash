;++
;
; Generated NASM RawDog JIT blob: PerfectHashJitIndexMultiplyShiftR16Index32x8_x64
;
;--

        bits 64
        default rel

        section .text

        RAWDOG_IMM8_TABLE_MAGIC equ 0x8C4B2A1D9F573E61

        global PerfectHashJitIndexMultiplyShiftR16Index32x8_x64

PerfectHashJitIndexMultiplyShiftR16Index32x8_x64:

        ;IACA_VC_START

        mov     r11, rsp

        vmovd   xmm0, edi
        vpinsrd xmm0, xmm0, esi, 1
        vpinsrd xmm0, xmm0, edx, 2
        vpinsrd xmm0, xmm0, ecx, 3

        vmovd   xmm1, r8d
        vpinsrd xmm1, xmm1, r9d, 1
        vpinsrd xmm1, xmm1, dword [r11 + 0x8], 2
        vpinsrd xmm1, xmm1, dword [r11 + 0x10], 3

        vinserti128 ymm1, ymm0, xmm1, 1

        mov     r10, [rel RawDogAssigned]
        vpbroadcastd ymm2, dword [rel RawDogSeed1]
        vpbroadcastd ymm3, dword [rel RawDogSeed2]

        vpmulld ymm4, ymm1, ymm2
        vpmulld ymm5, ymm1, ymm3

        vpsrld  ymm4, ymm4, 0x2
Seed3Byte1ImmOffset equ $-1-$$
        vpsrld  ymm5, ymm5, 0x3
Seed3Byte2ImmOffset equ $-1-$$

        vpbroadcastd ymm2, dword [rel RawDogHashMask]
        vpand   ymm4, ymm4, ymm2
        vpand   ymm5, ymm5, ymm2

        mov     r9d, dword [rel RawDogIndexMask]

        vpextrd eax, xmm4, 0
        vpextrd ecx, xmm5, 0
        movzx   eax, word [r10 + rax * 2]
        movzx   ecx, word [r10 + rcx * 2]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0x18]
        mov     dword [rdx], eax

        vpextrd eax, xmm4, 1
        vpextrd ecx, xmm5, 1
        movzx   eax, word [r10 + rax * 2]
        movzx   ecx, word [r10 + rcx * 2]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0x20]
        mov     dword [rdx], eax

        vpextrd eax, xmm4, 2
        vpextrd ecx, xmm5, 2
        movzx   eax, word [r10 + rax * 2]
        movzx   ecx, word [r10 + rcx * 2]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0x28]
        mov     dword [rdx], eax

        vpextrd eax, xmm4, 3
        vpextrd ecx, xmm5, 3
        movzx   eax, word [r10 + rax * 2]
        movzx   ecx, word [r10 + rcx * 2]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0x30]
        mov     dword [rdx], eax

        vextracti128 xmm0, ymm4, 1
        vextracti128 xmm1, ymm5, 1

        vpextrd eax, xmm0, 0
        vpextrd ecx, xmm1, 0
        movzx   eax, word [r10 + rax * 2]
        movzx   ecx, word [r10 + rcx * 2]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0x38]
        mov     dword [rdx], eax

        vpextrd eax, xmm0, 1
        vpextrd ecx, xmm1, 1
        movzx   eax, word [r10 + rax * 2]
        movzx   ecx, word [r10 + rcx * 2]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0x40]
        mov     dword [rdx], eax

        vpextrd eax, xmm0, 2
        vpextrd ecx, xmm1, 2
        movzx   eax, word [r10 + rax * 2]
        movzx   ecx, word [r10 + rcx * 2]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0x48]
        mov     dword [rdx], eax

        vpextrd eax, xmm0, 3
        vpextrd ecx, xmm1, 3
        movzx   eax, word [r10 + rax * 2]
        movzx   ecx, word [r10 + rcx * 2]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0x50]
        mov     dword [rdx], eax

        ;IACA_VC_END

        ret

        align 8
RawDogImm8PatchTable:
        dq RAWDOG_IMM8_TABLE_MAGIC
        dd 2
        dd 0
        dd Seed3Byte1ImmOffset
        dd Seed3Byte2ImmOffset

        align 8
RawDogAssigned:
        dq 0xA1A1A1A1A1A1A1A1

RawDogSeed1:
        dq 0xB1B1B1B1B1B1B1B1

RawDogSeed2:
        dq 0xC1C1C1C1C1C1C1C1

RawDogSeed3Byte1:
        dq 0xD1D1D1D1D1D1D1D1

RawDogSeed3Byte2:
        dq 0xE1E1E1E1E1E1E1E1

RawDogHashMask:
        dq 0xF1F1F1F1F1F1F1F1

RawDogIndexMask:
        dq 0x2121212121212121

; vim:set tw=80 ts=8 sw=4 sts=4 et syntax=nasm fo=croql comments=\:;
