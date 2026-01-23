;++
;
; Copyright (c) 2025 Trent Nelson <trent@trent.me>
;
; Module Name:
;
;   PerfectHashJitIndexMulshrolate3RX16_x64.nasm
;
; Abstract:
;
;   This module implements the Mulshrolate3RX Index() routine (16-bit assigned)
;   as a position-independent blob suitable for RawDog JIT patching.
;
;--

        bits 64
        default rel

        section .text

        global PerfectHashJitIndexMulshrolate3RX16_x64

;+++
;
; ULONG
; PerfectHashJitIndexMulshrolate3RX16_x64(
;     _In_ ULONG Key
;     );
;
; Routine Description:
;
;   This routine implements the Mulshrolate3RX index functionality for tables
;   using 16-bit assigned elements.  It is designed to be patched in-place by
;   replacing the sentinel values in the embedded data block that follows the
;   routine.
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
PerfectHashJitIndexMulshrolate3RX16_x64:

        ;IACA_VC_START

        mov     r10, [rel RawDogAssigned]      ; Load assigned base address.

        mov     r11d, dword [rel RawDogSeed3Byte1]

        mov     eax, edi                       ; Copy key into eax.
        imul    eax, dword [rel RawDogSeed1]   ; Vertex1 = Key * Seed1.
        mov     ecx, dword [rel RawDogSeed3Byte2]
        ror     eax, cl                        ; Vertex1 = ror(Vertex1, Seed3_Byte2).
        imul    eax, dword [rel RawDogSeed4]   ; Vertex1 *= Seed4.
        mov     ecx, r11d
        shr     eax, cl                        ; Vertex1 >>= Seed3_Byte1.

        mov     edx, edi                       ; Copy key into edx.
        imul    edx, dword [rel RawDogSeed2]   ; Vertex2 = Key * Seed2.
        mov     ecx, dword [rel RawDogSeed3Byte3]
        ror     edx, cl                        ; Vertex2 = ror(Vertex2, Seed3_Byte3).
        mov     ecx, r11d
        shr     edx, cl                        ; Vertex2 >>= Seed3_Byte1.

        movzx   eax, word [r10 + rax * 2]      ; Load vertex1.
        movzx   edx, word [r10 + rdx * 2]      ; Load vertex2.

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

RawDogSeed3Byte3:
        dq 0xD2D2D2D2D2D2D2D2

RawDogSeed4:
        dq 0xB2B2B2B2B2B2B2B2

RawDogIndexMask:
        dq 0x2121212121212121

; vim:set tw=80 ts=8 sw=4 sts=4 et syntax=nasm fo=croql comments=\:;
