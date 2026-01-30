;++
;
; Generated NASM RawDog JIT blob: PerfectHashJitIndexMulshrolate2RX16Index32x4_x64
;
;--

        bits 64
        default rel

        section .text

        global PerfectHashJitIndexMulshrolate2RX16Index32x4_x64

PerfectHashJitIndexMulshrolate2RX16Index32x4_x64:

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
        mov     ecx, dword [rel RawDogSeed3Byte3]
        ror     edx, cl
        mov     ecx, r8d
        shr     edx, cl
        movzx   eax, word [r10 + rax * 2]
        movzx   edx, word [r10 + rdx * 2]
        add     eax, edx
        and     eax, dword [rel RawDogIndexMask]
        mov     rdx, qword [rsp + 0x10]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x4]
        imul    eax, dword [rel RawDogSeed1]
        mov     ecx, dword [rel RawDogSeed3Byte2]
        ror     eax, cl
        mov     ecx, r8d
        shr     eax, cl
        mov     edx, dword [rsp + 0x4]
        imul    edx, dword [rel RawDogSeed2]
        mov     ecx, dword [rel RawDogSeed3Byte3]
        ror     edx, cl
        mov     ecx, r8d
        shr     edx, cl
        movzx   eax, word [r10 + rax * 2]
        movzx   edx, word [r10 + rdx * 2]
        add     eax, edx
        and     eax, dword [rel RawDogIndexMask]
        mov     rdx, qword [rsp + 0x18]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x8]
        imul    eax, dword [rel RawDogSeed1]
        mov     ecx, dword [rel RawDogSeed3Byte2]
        ror     eax, cl
        mov     ecx, r8d
        shr     eax, cl
        mov     edx, dword [rsp + 0x8]
        imul    edx, dword [rel RawDogSeed2]
        mov     ecx, dword [rel RawDogSeed3Byte3]
        ror     edx, cl
        mov     ecx, r8d
        shr     edx, cl
        movzx   eax, word [r10 + rax * 2]
        movzx   edx, word [r10 + rdx * 2]
        add     eax, edx
        and     eax, dword [rel RawDogIndexMask]
        mov     rdx, qword [rsp + 0x20]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0xc]
        imul    eax, dword [rel RawDogSeed1]
        mov     ecx, dword [rel RawDogSeed3Byte2]
        ror     eax, cl
        mov     ecx, r8d
        shr     eax, cl
        mov     edx, dword [rsp + 0xc]
        imul    edx, dword [rel RawDogSeed2]
        mov     ecx, dword [rel RawDogSeed3Byte3]
        ror     edx, cl
        mov     ecx, r8d
        shr     edx, cl
        movzx   eax, word [r10 + rax * 2]
        movzx   edx, word [r10 + rdx * 2]
        add     eax, edx
        and     eax, dword [rel RawDogIndexMask]
        mov     rdx, qword [rsp + 0x28]
        mov     dword [rdx], eax

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

RawDogSeed3Byte3:
        dq 0xD2D2D2D2D2D2D2D2

RawDogIndexMask:
        dq 0x2121212121212121

; vim:set tw=80 ts=8 sw=4 sts=4 et syntax=nasm fo=croql comments=\:;
