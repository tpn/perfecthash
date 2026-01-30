;+
;
; Generated NASM RawDog JIT blob: PerfectHashJitIndexMulshrolate4RXIndex32x16_x64
;
;--

        bits 64
        default rel

        section .text

        global PerfectHashJitIndexMulshrolate4RXIndex32x16_x64

        align 16
PerfectHashJitIndexMulshrolate4RXIndex32x16_x64:

        ;IACA_VC_START

        mov     r11, rsp                       ; Save stack base.
        sub     rsp, 0xC0                      ; Reserve local storage.

        mov     dword [rsp + 0x00], edi        ; Store keys 1-4.
        mov     dword [rsp + 0x04], esi
        mov     dword [rsp + 0x08], edx
        mov     dword [rsp + 0x0c], ecx
        mov     dword [rsp + 0x10], r8d        ; Store keys 5-6.
        mov     dword [rsp + 0x14], r9d
        mov     eax, dword [r11 + 0x08]        ; Load keys 7-16.
        mov     dword [rsp + 0x18], eax
        mov     eax, dword [r11 + 0x10]
        mov     dword [rsp + 0x1c], eax
        mov     eax, dword [r11 + 0x18]
        mov     dword [rsp + 0x20], eax
        mov     eax, dword [r11 + 0x20]
        mov     dword [rsp + 0x24], eax
        mov     eax, dword [r11 + 0x28]
        mov     dword [rsp + 0x28], eax
        mov     eax, dword [r11 + 0x30]
        mov     dword [rsp + 0x2c], eax
        mov     eax, dword [r11 + 0x38]
        mov     dword [rsp + 0x30], eax
        mov     eax, dword [r11 + 0x40]
        mov     dword [rsp + 0x34], eax
        mov     eax, dword [r11 + 0x48]
        mov     dword [rsp + 0x38], eax
        mov     eax, dword [r11 + 0x50]
        mov     dword [rsp + 0x3c], eax

        vmovdqu32 zmm0, [rsp + 0x00]           ; Load keys.

        mov     r10, [rel RawDogAssigned]
        vpbroadcastd zmm1, dword [rel RawDogSeed1]
        vpbroadcastd zmm2, dword [rel RawDogSeed2]
        vpbroadcastd zmm14, dword [rel RawDogSeed4]
        vpbroadcastd zmm15, dword [rel RawDogSeed5]

        vpmulld zmm3, zmm0, zmm1               ; Vertex1 = Key * Seed1.
        vpmulld zmm4, zmm0, zmm2               ; Vertex2 = Key * Seed2.

        mov     eax, dword [rel RawDogSeed3Byte1]
        mov     ecx, dword [rel RawDogSeed3Byte2]
        mov     edx, dword [rel RawDogSeed3Byte3]
        and     eax, 31
        and     ecx, 31
        and     edx, 31

        mov     r8d, 32
        sub     r8d, ecx                       ; 32 - Seed3_Byte2.
        mov     r9d, 32
        sub     r9d, edx                       ; 32 - Seed3_Byte3.

        vmovd   xmm5, eax                      ; Seed3_Byte1.
        vmovd   xmm6, ecx                      ; Seed3_Byte2.
        vmovd   xmm7, r8d                      ; 32 - Seed3_Byte2.
        vmovd   xmm8, edx                      ; Seed3_Byte3.
        vmovd   xmm9, r9d                      ; 32 - Seed3_Byte3.

        vpsrld  zmm12, zmm3, xmm6              ; ror(Vertex1, Seed3_Byte2).
        vpslld  zmm13, zmm3, xmm7
        vpord   zmm3, zmm12, zmm13
        vpmulld zmm3, zmm3, zmm14              ; Vertex1 *= Seed4.
        vpsrld  zmm3, zmm3, xmm5               ; Vertex1 >>= Seed3_Byte1.

        vpsrld  zmm12, zmm4, xmm8              ; ror(Vertex2, Seed3_Byte3).
        vpslld  zmm13, zmm4, xmm9
        vpord   zmm4, zmm12, zmm13
        vpmulld zmm4, zmm4, zmm15              ; Vertex2 *= Seed5.
        vpsrld  zmm4, zmm4, xmm5               ; Vertex2 >>= Seed3_Byte1.

        vmovdqu32 [rsp + 0x40], zmm3
        vmovdqu32 [rsp + 0x80], zmm4

        mov     r9d, dword [rel RawDogIndexMask]

        mov     eax, dword [rsp + 0x40]
        mov     ecx, dword [rsp + 0x80]
        mov     eax, dword [r10 + rax * 4]
        mov     ecx, dword [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0x58]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x44]
        mov     ecx, dword [rsp + 0x84]
        mov     eax, dword [r10 + rax * 4]
        mov     ecx, dword [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0x60]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x48]
        mov     ecx, dword [rsp + 0x88]
        mov     eax, dword [r10 + rax * 4]
        mov     ecx, dword [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0x68]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x4c]
        mov     ecx, dword [rsp + 0x8c]
        mov     eax, dword [r10 + rax * 4]
        mov     ecx, dword [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0x70]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x50]
        mov     ecx, dword [rsp + 0x90]
        mov     eax, dword [r10 + rax * 4]
        mov     ecx, dword [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0x78]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x54]
        mov     ecx, dword [rsp + 0x94]
        mov     eax, dword [r10 + rax * 4]
        mov     ecx, dword [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0x80]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x58]
        mov     ecx, dword [rsp + 0x98]
        mov     eax, dword [r10 + rax * 4]
        mov     ecx, dword [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0x88]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x5c]
        mov     ecx, dword [rsp + 0x9c]
        mov     eax, dword [r10 + rax * 4]
        mov     ecx, dword [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0x90]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x60]
        mov     ecx, dword [rsp + 0xa0]
        mov     eax, dword [r10 + rax * 4]
        mov     ecx, dword [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0x98]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x64]
        mov     ecx, dword [rsp + 0xa4]
        mov     eax, dword [r10 + rax * 4]
        mov     ecx, dword [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0xa0]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x68]
        mov     ecx, dword [rsp + 0xa8]
        mov     eax, dword [r10 + rax * 4]
        mov     ecx, dword [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0xa8]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x6c]
        mov     ecx, dword [rsp + 0xac]
        mov     eax, dword [r10 + rax * 4]
        mov     ecx, dword [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0xb0]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x70]
        mov     ecx, dword [rsp + 0xb0]
        mov     eax, dword [r10 + rax * 4]
        mov     ecx, dword [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0xb8]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x74]
        mov     ecx, dword [rsp + 0xb4]
        mov     eax, dword [r10 + rax * 4]
        mov     ecx, dword [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0xc0]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x78]
        mov     ecx, dword [rsp + 0xb8]
        mov     eax, dword [r10 + rax * 4]
        mov     ecx, dword [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0xc8]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x7c]
        mov     ecx, dword [rsp + 0xbc]
        mov     eax, dword [r10 + rax * 4]
        mov     ecx, dword [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0xd0]
        mov     dword [rdx], eax

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

RawDogSeed3Byte3:
        dq 0xD2D2D2D2D2D2D2D2

RawDogSeed4:
        dq 0xB2B2B2B2B2B2B2B2

RawDogSeed5:
        dq 0xB3B3B3B3B3B3B3B3

RawDogIndexMask:
        dq 0x2121212121212121

; vim:set tw=80 ts=8 sw=4 sts=4 et syntax=nasm fo=croql comments=\:;
