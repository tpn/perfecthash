;++
;
; Copyright (c) 2025 Trent Nelson <trent@trent.me>
;
; Module Name:
;
;   PerfectHashJitIndexMultiplyShiftR_x64.nasm
;
; Abstract:
;
;   This module implements the MultiplyShiftR Index() routine as a position-
;   independent blob suitable for RawDog JIT patching.
;
;--

        bits 64
        default rel

        section .text

        RAWDOG_IMM8_TABLE_MAGIC equ 0x8C4B2A1D9F573E61

        global PerfectHashJitIndexMultiplyShiftR_x64

;+++
;
; ULONG
; PerfectHashJitIndexMultiplyShiftR_x64(
;     _In_ ULONG Key
;     );
;
; Routine Description:
;
;   This routine implements the MultiplyShiftR index functionality.  It is
;   designed to be patched in-place by replacing the sentinel values in the
;   embedded data block that follows the routine.
;
; Arguments:
;
;   Key (edi) - Supplies the key for which an index is to be obtained.
;
; Return Value:
;
;   The index corresponding to the given key.
;
;--

        align 16
PerfectHashJitIndexMultiplyShiftR_x64:

        ;IACA_VC_START

        mov     r10, 0xA1A1A1A1A1A1A1A1         ; Load assigned base address.

        imul    eax, edi, 0xB1C2D3E4           ; Vertex1 = Key * Seed1.
        shr     eax, 0x2                       ; Vertex1 >>= Seed3_Byte1.
Seed3Byte1ImmOffset equ $-1-$$

        imul    edx, edi, 0xC2D3E4F5           ; Vertex2 = Key * Seed2.
        shr     edx, 0x3                       ; Vertex2 >>= Seed3_Byte2.
Seed3Byte2ImmOffset equ $-1-$$

        mov     r11d, 0xF5061728
        and     eax, r11d                      ; Mask vertex1.
        and     edx, r11d                      ; Mask vertex2.

        mov     eax, dword [r10 + rax * 4]     ; Load vertex1.
        add     eax, dword [r10 + rdx * 4]     ; Add vertex2.
        and     eax, 0x06172839

        ;IACA_VC_END

        ret                                    ; Return.

        align 8
RawDogImm8PatchTable:
        dq RAWDOG_IMM8_TABLE_MAGIC
        dd 2
        dd 0
        dd Seed3Byte1ImmOffset
        dd Seed3Byte2ImmOffset

; vim:set tw=80 ts=8 sw=4 sts=4 et syntax=nasm fo=croql comments=\:;
