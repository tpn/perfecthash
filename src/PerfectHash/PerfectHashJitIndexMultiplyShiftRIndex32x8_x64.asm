        title "Perfect Hash RawDog MultiplyShiftR Index32x8 x64"

;++
;
; Copyright (c) 2026 Trent Nelson <trent@trent.me>
;
; Module Name:
;
;   PerfectHashJitIndexMultiplyShiftRIndex32x8_x64.asm
;
; Abstract:
;
;   This module implements the MultiplyShiftR Index32x8() routine as a
;   position-independent blob suitable for RawDog JIT patching.
;
;--

include PerfectHash.inc

KEYS_OFFSET            equ 000h
SEED1_OFFSET           equ 020h
SEED2_OFFSET           equ 028h
SEED3BYTE1_OFFSET      equ 030h
SEED3BYTE2_OFFSET      equ 038h
HASHMASK_OFFSET        equ 040h
INDEXMASK_OFFSET       equ 048h
LOCAL_STACK_SIZE       equ 060h

;++
;
; VOID
; PerfectHashJitIndexMultiplyShiftRIndex32x8_x64(
;     _In_ ULONG Key1,
;     _In_ ULONG Key2,
;     _In_ ULONG Key3,
;     _In_ ULONG Key4,
;     _In_ ULONG Key5,
;     _In_ ULONG Key6,
;     _In_ ULONG Key7,
;     _In_ ULONG Key8,
;     _Out_ PULONG Index1,
;     _Out_ PULONG Index2,
;     _Out_ PULONG Index3,
;     _Out_ PULONG Index4,
;     _Out_ PULONG Index5,
;     _Out_ PULONG Index6,
;     _Out_ PULONG Index7,
;     _Out_ PULONG Index8
;     );
;
; Routine Description:
;
;   This routine implements the MultiplyShiftR index functionality for eight
;   keys.  It is designed to be patched in-place by replacing the sentinel
;   values embedded in the instruction stream.
;
; Arguments:
;
;   Key1-Key4 (ecx, edx, r8d, r9d) - Supplies the first four keys.
;   Key5-Key8 ([rsp+0x28]..[rsp+0x40]) - Supplies the remaining keys.
;   Index1-Index8 ([rsp+0x48]..[rsp+0x80]) - Receives the resulting indices.
;
; Return Value:
;
;   None.
;
;--

        LEAF_ENTRY PerfectHashJitIndexMultiplyShiftRIndex32x8_x64, _TEXT$00, NoPad

        ;IACA_VC_START

        mov     r11, rsp                           ; Save stack base.
        sub     rsp, LOCAL_STACK_SIZE              ; Reserve local storage.

        mov     dword ptr [rsp + 00h], ecx         ; Store keys 1-4.
        mov     dword ptr [rsp + 04h], edx
        mov     dword ptr [rsp + 08h], r8d
        mov     dword ptr [rsp + 0Ch], r9d
        mov     eax, dword ptr [r11 + 028h]        ; Store keys 5-8.
        mov     dword ptr [rsp + 10h], eax
        mov     eax, dword ptr [r11 + 030h]
        mov     dword ptr [rsp + 14h], eax
        mov     eax, dword ptr [r11 + 038h]
        mov     dword ptr [rsp + 18h], eax
        mov     eax, dword ptr [r11 + 040h]
        mov     dword ptr [rsp + 1Ch], eax

        mov     r10, 0A1A1A1A1A1A1A1A1h             ; Assigned base address.

        mov     rax, 0B1B1B1B1B1B1B1B1h             ; Seed1.
        mov     qword ptr [rsp + SEED1_OFFSET], rax
        mov     rax, 0C1C1C1C1C1C1C1C1h             ; Seed2.
        mov     qword ptr [rsp + SEED2_OFFSET], rax
        mov     rax, 0D1D1D1D1D1D1D1D1h             ; Seed3 byte 1.
        mov     qword ptr [rsp + SEED3BYTE1_OFFSET], rax
        mov     rax, 0E1E1E1E1E1E1E1E1h             ; Seed3 byte 2.
        mov     qword ptr [rsp + SEED3BYTE2_OFFSET], rax
        mov     rax, 0F1F1F1F1F1F1F1F1h             ; Hash mask.
        mov     qword ptr [rsp + HASHMASK_OFFSET], rax
        mov     rax, 02121212121212121h            ; Index mask.
        mov     qword ptr [rsp + INDEXMASK_OFFSET], rax

        xor     r8d, r8d                           ; Initialize index.

Index32x8Loop:
        mov     eax, dword ptr [rsp + r8 * 4]      ; Load key.
        mov     edx, eax
        mov     ecx, dword ptr [rsp + SEED1_OFFSET]
        imul    eax, ecx                           ; Vertex1 = Key * Seed1.
        mov     ecx, dword ptr [rsp + SEED3BYTE1_OFFSET]
        shr     eax, cl                            ; Vertex1 >>= Seed3_Byte1.

        mov     ecx, dword ptr [rsp + SEED2_OFFSET]
        imul    edx, ecx                           ; Vertex2 = Key * Seed2.
        mov     ecx, dword ptr [rsp + SEED3BYTE2_OFFSET]
        shr     edx, cl                            ; Vertex2 >>= Seed3_Byte2.

        mov     ecx, dword ptr [rsp + HASHMASK_OFFSET]
        and     eax, ecx                           ; Mask vertex1.
        and     edx, ecx                           ; Mask vertex2.

        mov     eax, dword ptr [r10 + rax * 4]     ; Load vertex1.
        mov     edx, dword ptr [r10 + rdx * 4]     ; Load vertex2.

        add     eax, edx                           ; Add vertices.
        mov     ecx, dword ptr [rsp + INDEXMASK_OFFSET]
        and     eax, ecx                           ; Mask the index.

        mov     r9, qword ptr [r11 + r8 * 8 + 048h]
        mov     dword ptr [r9], eax

        inc     r8d
        cmp     r8d, 8
        jl      Index32x8Loop

        add     rsp, LOCAL_STACK_SIZE
        ret                                        ; Return.

        ;IACA_VC_END

        LEAF_END PerfectHashJitIndexMultiplyShiftRIndex32x8_x64, _TEXT$00

; vim:set tw=80 ts=8 sw=4 sts=4 et syntax=masm fo=croql comments=\:;           :

end
