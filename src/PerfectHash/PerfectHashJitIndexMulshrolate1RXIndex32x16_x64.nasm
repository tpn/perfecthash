;++
;
; Generated NASM RawDog JIT blob: PerfectHashJitIndexMulshrolate1RXIndex32x16_x64
;
;--

        bits 64
        default rel

        section .text

        global PerfectHashJitIndexMulshrolate1RXIndex32x16_x64

PerfectHashJitIndexMulshrolate1RXIndex32x16_x64:

        ;IACA_VC_START

        mov     r11, rsp
        sub     rsp, 0xc0

        mov     dword [rsp + 0x0], edi
        mov     dword [rsp + 0x4], esi
        mov     dword [rsp + 0x8], edx
        mov     dword [rsp + 0xc], ecx
        mov     dword [rsp + 0x10], r8d
        mov     dword [rsp + 0x14], r9d
        mov     eax, dword [r11 + 0x8]
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

        mov     rax, qword [r11 + 0x58]
        mov     qword [rsp + 0x40], rax
        mov     rax, qword [r11 + 0x60]
        mov     qword [rsp + 0x48], rax
        mov     rax, qword [r11 + 0x68]
        mov     qword [rsp + 0x50], rax
        mov     rax, qword [r11 + 0x70]
        mov     qword [rsp + 0x58], rax
        mov     rax, qword [r11 + 0x78]
        mov     qword [rsp + 0x60], rax
        mov     rax, qword [r11 + 0x80]
        mov     qword [rsp + 0x68], rax
        mov     rax, qword [r11 + 0x88]
        mov     qword [rsp + 0x70], rax
        mov     rax, qword [r11 + 0x90]
        mov     qword [rsp + 0x78], rax
        mov     rax, qword [r11 + 0x98]
        mov     qword [rsp + 0x80], rax
        mov     rax, qword [r11 + 0xa0]
        mov     qword [rsp + 0x88], rax
        mov     rax, qword [r11 + 0xa8]
        mov     qword [rsp + 0x90], rax
        mov     rax, qword [r11 + 0xb0]
        mov     qword [rsp + 0x98], rax
        mov     rax, qword [r11 + 0xb8]
        mov     qword [rsp + 0xa0], rax
        mov     rax, qword [r11 + 0xc0]
        mov     qword [rsp + 0xa8], rax
        mov     rax, qword [r11 + 0xc8]
        mov     qword [rsp + 0xb0], rax
        mov     rax, qword [r11 + 0xd0]
        mov     qword [rsp + 0xb8], rax

        mov     r10, [rel RawDogAssigned]
        mov     r8d, dword [rel RawDogSeed3Byte1]

        mov     eax, dword [rsp + 0x0]
        imul    eax, dword [rel RawDogSeed1]
        mov     ecx, dword [rel RawDogSeed3Byte2]
        ror     eax, cl
        mov     ecx, r8d
        shr     eax, cl
        mov     edx, dword [rsp + 0x0]
        imul    edx, dword [rel RawDogSeed2]
        mov     ecx, r8d
        shr     edx, cl
        mov     eax, dword [r10 + rax * 4]
        mov     edx, dword [r10 + rdx * 4]
        add     eax, edx
        and     eax, dword [rel RawDogIndexMask]
        mov     rdx, qword [rsp + 0x40]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x4]
        imul    eax, dword [rel RawDogSeed1]
        mov     ecx, dword [rel RawDogSeed3Byte2]
        ror     eax, cl
        mov     ecx, r8d
        shr     eax, cl
        mov     edx, dword [rsp + 0x4]
        imul    edx, dword [rel RawDogSeed2]
        mov     ecx, r8d
        shr     edx, cl
        mov     eax, dword [r10 + rax * 4]
        mov     edx, dword [r10 + rdx * 4]
        add     eax, edx
        and     eax, dword [rel RawDogIndexMask]
        mov     rdx, qword [rsp + 0x48]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x8]
        imul    eax, dword [rel RawDogSeed1]
        mov     ecx, dword [rel RawDogSeed3Byte2]
        ror     eax, cl
        mov     ecx, r8d
        shr     eax, cl
        mov     edx, dword [rsp + 0x8]
        imul    edx, dword [rel RawDogSeed2]
        mov     ecx, r8d
        shr     edx, cl
        mov     eax, dword [r10 + rax * 4]
        mov     edx, dword [r10 + rdx * 4]
        add     eax, edx
        and     eax, dword [rel RawDogIndexMask]
        mov     rdx, qword [rsp + 0x50]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0xc]
        imul    eax, dword [rel RawDogSeed1]
        mov     ecx, dword [rel RawDogSeed3Byte2]
        ror     eax, cl
        mov     ecx, r8d
        shr     eax, cl
        mov     edx, dword [rsp + 0xc]
        imul    edx, dword [rel RawDogSeed2]
        mov     ecx, r8d
        shr     edx, cl
        mov     eax, dword [r10 + rax * 4]
        mov     edx, dword [r10 + rdx * 4]
        add     eax, edx
        and     eax, dword [rel RawDogIndexMask]
        mov     rdx, qword [rsp + 0x58]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x10]
        imul    eax, dword [rel RawDogSeed1]
        mov     ecx, dword [rel RawDogSeed3Byte2]
        ror     eax, cl
        mov     ecx, r8d
        shr     eax, cl
        mov     edx, dword [rsp + 0x10]
        imul    edx, dword [rel RawDogSeed2]
        mov     ecx, r8d
        shr     edx, cl
        mov     eax, dword [r10 + rax * 4]
        mov     edx, dword [r10 + rdx * 4]
        add     eax, edx
        and     eax, dword [rel RawDogIndexMask]
        mov     rdx, qword [rsp + 0x60]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x14]
        imul    eax, dword [rel RawDogSeed1]
        mov     ecx, dword [rel RawDogSeed3Byte2]
        ror     eax, cl
        mov     ecx, r8d
        shr     eax, cl
        mov     edx, dword [rsp + 0x14]
        imul    edx, dword [rel RawDogSeed2]
        mov     ecx, r8d
        shr     edx, cl
        mov     eax, dword [r10 + rax * 4]
        mov     edx, dword [r10 + rdx * 4]
        add     eax, edx
        and     eax, dword [rel RawDogIndexMask]
        mov     rdx, qword [rsp + 0x68]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x18]
        imul    eax, dword [rel RawDogSeed1]
        mov     ecx, dword [rel RawDogSeed3Byte2]
        ror     eax, cl
        mov     ecx, r8d
        shr     eax, cl
        mov     edx, dword [rsp + 0x18]
        imul    edx, dword [rel RawDogSeed2]
        mov     ecx, r8d
        shr     edx, cl
        mov     eax, dword [r10 + rax * 4]
        mov     edx, dword [r10 + rdx * 4]
        add     eax, edx
        and     eax, dword [rel RawDogIndexMask]
        mov     rdx, qword [rsp + 0x70]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x1c]
        imul    eax, dword [rel RawDogSeed1]
        mov     ecx, dword [rel RawDogSeed3Byte2]
        ror     eax, cl
        mov     ecx, r8d
        shr     eax, cl
        mov     edx, dword [rsp + 0x1c]
        imul    edx, dword [rel RawDogSeed2]
        mov     ecx, r8d
        shr     edx, cl
        mov     eax, dword [r10 + rax * 4]
        mov     edx, dword [r10 + rdx * 4]
        add     eax, edx
        and     eax, dword [rel RawDogIndexMask]
        mov     rdx, qword [rsp + 0x78]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x20]
        imul    eax, dword [rel RawDogSeed1]
        mov     ecx, dword [rel RawDogSeed3Byte2]
        ror     eax, cl
        mov     ecx, r8d
        shr     eax, cl
        mov     edx, dword [rsp + 0x20]
        imul    edx, dword [rel RawDogSeed2]
        mov     ecx, r8d
        shr     edx, cl
        mov     eax, dword [r10 + rax * 4]
        mov     edx, dword [r10 + rdx * 4]
        add     eax, edx
        and     eax, dword [rel RawDogIndexMask]
        mov     rdx, qword [rsp + 0x80]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x24]
        imul    eax, dword [rel RawDogSeed1]
        mov     ecx, dword [rel RawDogSeed3Byte2]
        ror     eax, cl
        mov     ecx, r8d
        shr     eax, cl
        mov     edx, dword [rsp + 0x24]
        imul    edx, dword [rel RawDogSeed2]
        mov     ecx, r8d
        shr     edx, cl
        mov     eax, dword [r10 + rax * 4]
        mov     edx, dword [r10 + rdx * 4]
        add     eax, edx
        and     eax, dword [rel RawDogIndexMask]
        mov     rdx, qword [rsp + 0x88]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x28]
        imul    eax, dword [rel RawDogSeed1]
        mov     ecx, dword [rel RawDogSeed3Byte2]
        ror     eax, cl
        mov     ecx, r8d
        shr     eax, cl
        mov     edx, dword [rsp + 0x28]
        imul    edx, dword [rel RawDogSeed2]
        mov     ecx, r8d
        shr     edx, cl
        mov     eax, dword [r10 + rax * 4]
        mov     edx, dword [r10 + rdx * 4]
        add     eax, edx
        and     eax, dword [rel RawDogIndexMask]
        mov     rdx, qword [rsp + 0x90]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x2c]
        imul    eax, dword [rel RawDogSeed1]
        mov     ecx, dword [rel RawDogSeed3Byte2]
        ror     eax, cl
        mov     ecx, r8d
        shr     eax, cl
        mov     edx, dword [rsp + 0x2c]
        imul    edx, dword [rel RawDogSeed2]
        mov     ecx, r8d
        shr     edx, cl
        mov     eax, dword [r10 + rax * 4]
        mov     edx, dword [r10 + rdx * 4]
        add     eax, edx
        and     eax, dword [rel RawDogIndexMask]
        mov     rdx, qword [rsp + 0x98]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x30]
        imul    eax, dword [rel RawDogSeed1]
        mov     ecx, dword [rel RawDogSeed3Byte2]
        ror     eax, cl
        mov     ecx, r8d
        shr     eax, cl
        mov     edx, dword [rsp + 0x30]
        imul    edx, dword [rel RawDogSeed2]
        mov     ecx, r8d
        shr     edx, cl
        mov     eax, dword [r10 + rax * 4]
        mov     edx, dword [r10 + rdx * 4]
        add     eax, edx
        and     eax, dword [rel RawDogIndexMask]
        mov     rdx, qword [rsp + 0xa0]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x34]
        imul    eax, dword [rel RawDogSeed1]
        mov     ecx, dword [rel RawDogSeed3Byte2]
        ror     eax, cl
        mov     ecx, r8d
        shr     eax, cl
        mov     edx, dword [rsp + 0x34]
        imul    edx, dword [rel RawDogSeed2]
        mov     ecx, r8d
        shr     edx, cl
        mov     eax, dword [r10 + rax * 4]
        mov     edx, dword [r10 + rdx * 4]
        add     eax, edx
        and     eax, dword [rel RawDogIndexMask]
        mov     rdx, qword [rsp + 0xa8]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x38]
        imul    eax, dword [rel RawDogSeed1]
        mov     ecx, dword [rel RawDogSeed3Byte2]
        ror     eax, cl
        mov     ecx, r8d
        shr     eax, cl
        mov     edx, dword [rsp + 0x38]
        imul    edx, dword [rel RawDogSeed2]
        mov     ecx, r8d
        shr     edx, cl
        mov     eax, dword [r10 + rax * 4]
        mov     edx, dword [r10 + rdx * 4]
        add     eax, edx
        and     eax, dword [rel RawDogIndexMask]
        mov     rdx, qword [rsp + 0xb0]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x3c]
        imul    eax, dword [rel RawDogSeed1]
        mov     ecx, dword [rel RawDogSeed3Byte2]
        ror     eax, cl
        mov     ecx, r8d
        shr     eax, cl
        mov     edx, dword [rsp + 0x3c]
        imul    edx, dword [rel RawDogSeed2]
        mov     ecx, r8d
        shr     edx, cl
        mov     eax, dword [r10 + rax * 4]
        mov     edx, dword [r10 + rdx * 4]
        add     eax, edx
        and     eax, dword [rel RawDogIndexMask]
        mov     rdx, qword [rsp + 0xb8]
        mov     dword [rdx], eax

        add     rsp, 0xc0

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
