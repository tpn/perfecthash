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

        mov     r10, [rel RawDogAssigned]      ; Load assigned base address.

        mov     eax, edi                       ; Copy key into eax.
        imul    eax, dword [rel RawDogSeed1]   ; Vertex1 = Key * Seed1.
        mov     ecx, dword [rel RawDogSeed3Byte1]
        shr     eax, cl                        ; Vertex1 >>= Seed3_Byte1.

        mov     edx, edi                       ; Copy key into edx.
        imul    edx, dword [rel RawDogSeed2]   ; Vertex2 = Key * Seed2.
        mov     ecx, dword [rel RawDogSeed3Byte2]
        shr     edx, cl                        ; Vertex2 >>= Seed3_Byte2.

        mov     r11d, dword [rel RawDogHashMask]
        and     eax, r11d                      ; Mask vertex1.
        and     edx, r11d                      ; Mask vertex2.

        mov     eax, dword [r10 + rax * 4]     ; Load vertex1.
        mov     edx, dword [r10 + rdx * 4]     ; Load vertex2.

        add     eax, edx                       ; Add vertices.
        and     eax, dword [rel RawDogIndexMask]

        ;IACA_VC_END

        ret                                    ; Return.

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
