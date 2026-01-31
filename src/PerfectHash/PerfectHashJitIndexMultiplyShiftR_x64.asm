        title "Perfect Hash RawDog MultiplyShiftR x64"

;++
;
; Copyright (c) 2025 Trent Nelson <trent@trent.me>
;
; Module Name:
;
;   PerfectHashJitIndexMultiplyShiftR_x64.asm
;
; Abstract:
;
;   This module implements the MultiplyShiftR Index() routine for 32-bit
;   assigned tables as a position-independent blob suitable for RawDog JIT
;   patching.
;
;--

include PerfectHash.inc

;++
;
; ULONG
; PerfectHashJitIndexMultiplyShiftR_x64(
;     _In_ ULONG Key
;     );
;
; Routine Description:
;
;   This routine implements the MultiplyShiftR index functionality against
;   32-bit assigned table data.  It is designed to be patched in-place by
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

        LEAF_ENTRY PerfectHashJitIndexMultiplyShiftR_x64, _TEXT$00, NoPad

        ;IACA_VC_START

        mov     r10, 0A1A1A1A1A1A1A1A1h             ; Assigned base address.

        mov     edx, ecx                           ; Copy key into edx.
        imul    eax, ecx, 0B1C2D3E4h               ; Vertex1 = Key * Seed1.
        mov     ecx, 0D31BEF20h
        shr     eax, cl                            ; Vertex1 >>= Seed3_Byte1.

        imul    edx, edx, 0C2D3E4F5h               ; Vertex2 = Key * Seed2.
        mov     ecx, 0E4F50617h
        shr     edx, cl                            ; Vertex2 >>= Seed3_Byte2.

        mov     r11d, 0F5061728h                   ; Hash mask.
        and     eax, r11d                          ; Mask vertex1.
        and     edx, r11d                          ; Mask vertex2.

        mov     eax, dword ptr [r10 + rax * 4]     ; Load vertex1.
        add     eax, dword ptr [r10 + rdx * 4]     ; Add vertex2.
        and     eax, 06172839h

        ;IACA_VC_END

        ret                                        ; Return.

        LEAF_END PerfectHashJitIndexMultiplyShiftR_x64, _TEXT$00

; vim:set tw=80 ts=8 sw=4 sts=4 et syntax=masm fo=croql comments=\:;           :

end