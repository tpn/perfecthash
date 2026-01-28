        title "Perfect Hash RawDog Mulshrolate3RX16 x64"

;++
;
; Copyright (c) 2025 Trent Nelson <trent@trent.me>
;
; Module Name:
;
;   PerfectHashJitIndexMulshrolate3RX16_x64.asm
;
; Abstract:
;
;   This module implements the Mulshrolate3RX Index() routine for 16-bit
;   assigned tables as a position-independent blob suitable for RawDog JIT
;   patching.
;
;--

include PerfectHash.inc

;++
;
; ULONG
; PerfectHashJitIndexMulshrolate3RX16_x64(
;     _In_ ULONG Key
;     );
;
; Routine Description:
;
;   This routine implements the Mulshrolate3RX index functionality against
;   16-bit assigned table data.  It is designed to be patched in-place by
;   replacing the sentinel values embedded in the instruction stream.
;
; Arguments:
;
;   Key (ecx) - Supplies the key for which an index is to be obtained.
;
; Return Value:
;
;   The index corresponding to the given key.
;
;--

        LEAF_ENTRY PerfectHashJitIndexMulshrolate3RX16_x64, _TEXT$00, NoPad

        ;IACA_VC_START

        mov     r10, 0A1A1A1A1A1A1A1A1h             ; Assigned base address.
        mov     r8,  0B1B1B1B1B1B1B1B1h             ; Seed1.
        mov     r9,  0C1C1C1C1C1C1C1C1h             ; Seed2.
        mov     r11, 0D1D1D1D1D1D1D1D1h             ; Seed3 byte 1.

        mov     eax, ecx                           ; Copy key into eax.
        mov     edx, ecx                           ; Copy key into edx.
        imul    eax, r8d                           ; Vertex1 = Key * Seed1.
        mov     r8,  0E1E1E1E1E1E1E1E1h             ; Seed3 byte 2.
        mov     ecx, r8d
        ror     eax, cl                            ; Vertex1 = ror(Vertex1).
        mov     r8,  0B2B2B2B2B2B2B2B2h             ; Seed4.
        imul    eax, r8d                           ; Vertex1 *= Seed4.
        mov     ecx, r11d
        shr     eax, cl                            ; Vertex1 >>= Seed3_Byte1.

        imul    edx, r9d                           ; Vertex2 = Key * Seed2.
        mov     r9,  0D2D2D2D2D2D2D2D2h             ; Seed3 byte 3.
        mov     ecx, r9d
        ror     edx, cl                            ; Vertex2 = ror(Vertex2).
        mov     ecx, r11d
        shr     edx, cl                            ; Vertex2 >>= Seed3_Byte1.

        movzx   eax, word ptr [r10 + rax * 2]      ; Load vertex1.
        movzx   edx, word ptr [r10 + rdx * 2]      ; Load vertex2.

        add     eax, edx                           ; Add vertices.
        mov     r11, 02121212121212121h            ; Index mask.
        and     eax, r11d

        ;IACA_VC_END

        ret                                        ; Return.

        LEAF_END PerfectHashJitIndexMulshrolate3RX16_x64, _TEXT$00

; vim:set tw=80 ts=8 sw=4 sts=4 et syntax=masm fo=croql comments=\:;           :

end
