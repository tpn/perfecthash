;++
;
; Generated NASM RawDog JIT blob: PerfectHashJitIndexMultiplyShiftRX16_x64
;
;--

        bits 64
        default rel

        section .text

        global PerfectHashJitIndexMultiplyShiftRX16_x64

PerfectHashJitIndexMultiplyShiftRX16_x64:

        ;IACA_VC_START

        mov     r10, [rel RawDogAssigned]
        mov     r8d, dword [rel RawDogSeed3Byte1]

        mov     eax, edi
        imul    eax, dword [rel RawDogSeed1]
        mov     ecx, r8d
        shr     eax, cl

        mov     edx, edi
        imul    edx, dword [rel RawDogSeed2]
        mov     ecx, r8d
        shr     edx, cl

        movzx   eax, word [r10 + rax * 2]
        movzx   edx, word [r10 + rdx * 2]
        add     eax, edx
        and     eax, dword [rel RawDogIndexMask]

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