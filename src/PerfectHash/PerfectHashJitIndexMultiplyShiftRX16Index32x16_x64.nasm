;++
;
; Generated NASM RawDog JIT blob: PerfectHashJitIndexMultiplyShiftRX16Index32x16_x64
;
;--

        bits 64
        default rel

        section .text

        RAWDOG_IMM8_TABLE_MAGIC equ 0x8C4B2A1D9F573E61

        global PerfectHashJitIndexMultiplyShiftRX16Index32x16_x64

        align 16
PerfectHashJitIndexMultiplyShiftRX16Index32x16_x64:

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

        vpmulld zmm3, zmm0, zmm1               ; Vertex1 = Key * Seed1.
        vpmulld zmm4, zmm0, zmm2               ; Vertex2 = Key * Seed2.

        vpsrld  zmm3, zmm3, 0x2
Seed3Byte1ImmOffset equ $-1-$$
        vpsrld  zmm4, zmm4, 0x2
Seed3Byte2ImmOffset equ $-1-$$

        vmovdqu32 [rsp + 0x40], zmm3
        vmovdqu32 [rsp + 0x80], zmm4

        mov     r9d, dword [rel RawDogIndexMask]

        mov     eax, dword [rsp + 0x40]
        mov     ecx, dword [rsp + 0x80]
        movzx   eax, word [r10 + rax * 2]
        movzx   ecx, word [r10 + rcx * 2]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0x58]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x44]
        mov     ecx, dword [rsp + 0x84]
        movzx   eax, word [r10 + rax * 2]
        movzx   ecx, word [r10 + rcx * 2]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0x60]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x48]
        mov     ecx, dword [rsp + 0x88]
        movzx   eax, word [r10 + rax * 2]
        movzx   ecx, word [r10 + rcx * 2]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0x68]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x4c]
        mov     ecx, dword [rsp + 0x8c]
        movzx   eax, word [r10 + rax * 2]
        movzx   ecx, word [r10 + rcx * 2]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0x70]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x50]
        mov     ecx, dword [rsp + 0x90]
        movzx   eax, word [r10 + rax * 2]
        movzx   ecx, word [r10 + rcx * 2]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0x78]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x54]
        mov     ecx, dword [rsp + 0x94]
        movzx   eax, word [r10 + rax * 2]
        movzx   ecx, word [r10 + rcx * 2]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0x80]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x58]
        mov     ecx, dword [rsp + 0x98]
        movzx   eax, word [r10 + rax * 2]
        movzx   ecx, word [r10 + rcx * 2]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0x88]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x5c]
        mov     ecx, dword [rsp + 0x9c]
        movzx   eax, word [r10 + rax * 2]
        movzx   ecx, word [r10 + rcx * 2]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0x90]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x60]
        mov     ecx, dword [rsp + 0xa0]
        movzx   eax, word [r10 + rax * 2]
        movzx   ecx, word [r10 + rcx * 2]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0x98]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x64]
        mov     ecx, dword [rsp + 0xa4]
        movzx   eax, word [r10 + rax * 2]
        movzx   ecx, word [r10 + rcx * 2]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0xa0]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x68]
        mov     ecx, dword [rsp + 0xa8]
        movzx   eax, word [r10 + rax * 2]
        movzx   ecx, word [r10 + rcx * 2]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0xa8]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x6c]
        mov     ecx, dword [rsp + 0xac]
        movzx   eax, word [r10 + rax * 2]
        movzx   ecx, word [r10 + rcx * 2]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0xb0]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x70]
        mov     ecx, dword [rsp + 0xb0]
        movzx   eax, word [r10 + rax * 2]
        movzx   ecx, word [r10 + rcx * 2]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0xb8]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x74]
        mov     ecx, dword [rsp + 0xb4]
        movzx   eax, word [r10 + rax * 2]
        movzx   ecx, word [r10 + rcx * 2]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0xc0]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x78]
        mov     ecx, dword [rsp + 0xb8]
        movzx   eax, word [r10 + rax * 2]
        movzx   ecx, word [r10 + rcx * 2]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0xc8]
        mov     dword [rdx], eax

        mov     eax, dword [rsp + 0x7c]
        mov     ecx, dword [rsp + 0xbc]
        movzx   eax, word [r10 + rax * 2]
        movzx   ecx, word [r10 + rcx * 2]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword [r11 + 0xd0]
        mov     dword [rdx], eax

        add     rsp, 0xC0

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

RawDogIndexMask:
        dq 0x2121212121212121

; vim:set tw=80 ts=8 sw=4 sts=4 et syntax=nasm fo=croql comments=\:;
