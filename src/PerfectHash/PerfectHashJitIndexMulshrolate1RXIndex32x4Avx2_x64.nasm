;++
;
; Generated NASM RawDog JIT blob: PerfectHashJitIndexMulshrolate1RXIndex32x4Avx2_x64
;
;--

        bits 64
        default rel

        section .text

        global PerfectHashJitIndexMulshrolate1RXIndex32x4Avx2_x64

PerfectHashJitIndexMulshrolate1RXIndex32x4Avx2_x64:

        ;IACA_VC_START

        mov     r11, rsp
        sub     rsp, 0x30

        mov     dword [rsp + 0x0], edi
        mov     dword [rsp + 0x4], esi
        mov     dword [rsp + 0x8], edx
        mov     dword [rsp + 0xc], ecx

        mov     qword [rsp + 0x10], r8
        mov     qword [rsp + 0x18], r9
        mov     rax, qword [r11 + 0x8]
        mov     qword [rsp + 0x20], rax
        mov     rax, qword [r11 + 0x10]
        mov     qword [rsp + 0x28], rax

        vmovdqu xmm0, [rsp + 0x0]

        mov     r10, [rel RawDogAssigned]
        mov     eax, dword [rel RawDogSeed1]
        vmovd   xmm1, eax
        vpshufd xmm1, xmm1, 0

        mov     eax, dword [rel RawDogSeed2]
        vmovd   xmm2, eax
        vpshufd xmm2, xmm2, 0

        vpmulld xmm3, xmm0, xmm1
        vpmulld xmm4, xmm0, xmm2

        mov     ecx, dword [rel RawDogSeed3Byte2]
        and     ecx, 31
        mov     edx, 32
        sub     edx, ecx
        vmovd   xmm1, ecx
        vmovd   xmm2, edx
        vpsrld  xmm0, xmm3, xmm1
        vpslld  xmm1, xmm3, xmm2
        vpor    xmm3, xmm0, xmm1

        mov     ecx, dword [rel RawDogSeed3Byte1]
        and     ecx, 31
        vmovd   xmm1, ecx
        vpsrld  xmm3, xmm3, xmm1
        vpsrld  xmm4, xmm4, xmm1

        mov     r9d, dword [rel RawDogIndexMask]

        vpextrd eax, xmm3, 0
        vpextrd ecx, xmm4, 0
        mov     eax, dword [r10 + rax * 4]
        mov     ecx, dword [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [rsp + 0x10]
        mov     dword [rdx], eax

        vpextrd eax, xmm3, 1
        vpextrd ecx, xmm4, 1
        mov     eax, dword [r10 + rax * 4]
        mov     ecx, dword [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [rsp + 0x18]
        mov     dword [rdx], eax

        vpextrd eax, xmm3, 2
        vpextrd ecx, xmm4, 2
        mov     eax, dword [r10 + rax * 4]
        mov     ecx, dword [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [rsp + 0x20]
        mov     dword [rdx], eax

        vpextrd eax, xmm3, 3
        vpextrd ecx, xmm4, 3
        mov     eax, dword [r10 + rax * 4]
        mov     ecx, dword [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [rsp + 0x28]
        mov     dword [rdx], eax

        vzeroupper
        add     rsp, 0x30

        ;IACA_VC_END

        ret

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

RawDogIndexMask:
        dq 0x2121212121212121

; vim:set tw=80 ts=8 sw=4 sts=4 et syntax=nasm fo=croql comments=\:;
