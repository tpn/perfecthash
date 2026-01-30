;++
;
; Generated NASM RawDog JIT blob: PerfectHashJitIndexMultiplyShiftRIndex32x4Avx2_x64
;
;--

        bits 64
        default rel

        section .text

        RAWDOG_IMM8_TABLE_MAGIC equ 0x8C4B2A1D9F573E61

        global PerfectHashJitIndexMultiplyShiftRIndex32x4Avx2_x64

PerfectHashJitIndexMultiplyShiftRIndex32x4Avx2_x64:

        ;IACA_VC_START

        vmovd   xmm0, edi
        vpinsrd xmm0, xmm0, esi, 1
        vpinsrd xmm0, xmm0, edx, 2
        vpinsrd xmm1, xmm0, ecx, 3

        mov     r10, 0xA1A1A1A1A1A1A1A1
        vpbroadcastd xmm2, dword [rel RawDogMsrSeed1]
        vpbroadcastd xmm3, dword [rel RawDogMsrSeed2]

        vpmulld xmm4, xmm1, xmm2
        vpmulld xmm5, xmm1, xmm3

        vpsrld  xmm4, xmm4, 0x2
Seed3Byte1ImmOffset equ $-1-$$

        vpsrld  xmm5, xmm5, 0x3
Seed3Byte2ImmOffset equ $-1-$$

        vpbroadcastd xmm2, dword [rel RawDogMsrHashMask]
        vpand   xmm4, xmm4, xmm2
        vpand   xmm5, xmm5, xmm2

        mov     r11d, 0x06172839

        vpextrd eax, xmm4, 0
        vpextrd ecx, xmm5, 0
        mov     eax, dword [r10 + rax * 4]
        mov     ecx, dword [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r11d
        mov     dword [r8], eax

        vpextrd eax, xmm4, 1
        vpextrd ecx, xmm5, 1
        mov     eax, dword [r10 + rax * 4]
        mov     ecx, dword [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r11d
        mov     dword [r9], eax

        vpextrd eax, xmm4, 2
        vpextrd ecx, xmm5, 2
        mov     eax, dword [r10 + rax * 4]
        mov     ecx, dword [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r11d
        mov     rdx, qword [rsp + 0x8]
        mov     dword [rdx], eax

        vpextrd eax, xmm4, 3
        vpextrd ecx, xmm5, 3
        mov     eax, dword [r10 + rax * 4]
        mov     ecx, dword [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r11d
        mov     rdx, qword [rsp + 0x10]
        mov     dword [rdx], eax

        ;IACA_VC_END

        ret

        align 4
RawDogMsrSeed1:    dd 0xB1C2D3E4
RawDogMsrSeed2:    dd 0xC2D3E4F5
RawDogMsrHashMask: dd 0xF5061728

        align 8
RawDogImm8PatchTable:
        dq RAWDOG_IMM8_TABLE_MAGIC
        dd 2
        dd 0
        dd Seed3Byte1ImmOffset
        dd Seed3Byte2ImmOffset

; vim:set tw=80 ts=8 sw=4 sts=4 et syntax=nasm fo=croql comments=\:;
