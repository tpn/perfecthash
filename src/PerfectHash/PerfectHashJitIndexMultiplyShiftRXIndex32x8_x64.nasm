;++
;
; Generated NASM RawDog JIT blob: PerfectHashJitIndexMultiplyShiftRXIndex32x8_x64
;
;--

        bits 64
        default rel

        section .text

        global PerfectHashJitIndexMultiplyShiftRXIndex32x8_x64

PerfectHashJitIndexMultiplyShiftRXIndex32x8_x64:

        ;IACA_VC_START

        mov     r11, rsp
        sub     rsp, 0x60

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

        mov     rax, qword [r11 + 0x18]
        mov     qword [rsp + 0x20], rax
        mov     rax, qword [r11 + 0x20]
        mov     qword [rsp + 0x28], rax
        mov     rax, qword [r11 + 0x28]
        mov     qword [rsp + 0x30], rax
        mov     rax, qword [r11 + 0x30]
        mov     qword [rsp + 0x38], rax
        mov     rax, qword [r11 + 0x38]
        mov     qword [rsp + 0x40], rax
        mov     rax, qword [r11 + 0x40]
        mov     qword [rsp + 0x48], rax
        mov     rax, qword [r11 + 0x48]
        mov     qword [rsp + 0x50], rax
        mov     rax, qword [r11 + 0x50]
        mov     qword [rsp + 0x58], rax

        mov     r10, [rel RawDogAssigned]
        mov     r8d, dword [rel RawDogSeed3Byte1]

        mov     eax, dword [rsp + 0x0]
        imul    eax, dword [rel RawDogSeed1]
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
        mov     rdx, qword [rsp + 0x20]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x4]
        imul    eax, dword [rel RawDogSeed1]
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
        mov     rdx, qword [rsp + 0x28]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x8]
        imul    eax, dword [rel RawDogSeed1]
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
        mov     rdx, qword [rsp + 0x30]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0xc]
        imul    eax, dword [rel RawDogSeed1]
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
        mov     rdx, qword [rsp + 0x38]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x10]
        imul    eax, dword [rel RawDogSeed1]
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
        mov     rdx, qword [rsp + 0x40]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x14]
        imul    eax, dword [rel RawDogSeed1]
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
        mov     rdx, qword [rsp + 0x48]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x18]
        imul    eax, dword [rel RawDogSeed1]
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
        mov     rdx, qword [rsp + 0x50]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x1c]
        imul    eax, dword [rel RawDogSeed1]
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
        mov     rdx, qword [rsp + 0x58]
        mov     dword [rdx], eax

        add     rsp, 0x60

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

RawDogIndexMask:
        dq 0x2121212121212121

; vim:set tw=80 ts=8 sw=4 sts=4 et syntax=nasm fo=croql comments=\:;
