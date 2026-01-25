        title "Perfect Hash RawDog MultiplyShiftR Index32x4 Avx2 x64"

;++
;
; Copyright (c) 2026 Trent Nelson <trent@trent.me>
;
; Module Name:
;
;   PerfectHashJitIndexMultiplyShiftRIndex32x4Avx2_x64.asm
;
; Abstract:
;
;   This module implements the MultiplyShiftR Index32x4() routine using AVX2
;   as a position-independent blob suitable for RawDog JIT patching.
;
;--

include PerfectHash.inc

KEYS_OFFSET            equ 000h
VERTEX1_OFFSET         equ 010h
VERTEX2_OFFSET         equ 020h
LOCAL_STACK_SIZE       equ 030h
OUTPUT_OFFSET          equ 028h

;++
;
; VOID
; PerfectHashJitIndexMultiplyShiftRIndex32x4Avx2_x64(
;     _In_ ULONG Key1,
;     _In_ ULONG Key2,
;     _In_ ULONG Key3,
;     _In_ ULONG Key4,
;     _Out_ PULONG Index1,
;     _Out_ PULONG Index2,
;     _Out_ PULONG Index3,
;     _Out_ PULONG Index4
;     );
;
; Routine Description:
;
;   This routine implements the MultiplyShiftR index functionality for four
;   keys using AVX2 instructions.  It is designed to be patched in-place by
;   replacing the sentinel values embedded in the instruction stream.
;
; Arguments:
;
;   Key1-Key4 (ecx, edx, r8d, r9d) - Supplies the first four keys.
;   Index1-Index4 ([rsp+0x28]..[rsp+0x40]) - Receives the resulting indices.
;
; Return Value:
;
;   None.
;
;--

        LEAF_ENTRY PerfectHashJitIndexMultiplyShiftRIndex32x4Avx2_x64, _TEXT$00, NoPad

        ;IACA_VC_START

        mov     r11, rsp                           ; Save stack base.
        sub     rsp, LOCAL_STACK_SIZE              ; Reserve local storage.

        mov     dword ptr [rsp + 00h], ecx         ; Store keys 1-4.
        mov     dword ptr [rsp + 04h], edx
        mov     dword ptr [rsp + 08h], r8d
        mov     dword ptr [rsp + 0Ch], r9d

        vmovdqu xmm0, xmmword ptr [rsp + KEYS_OFFSET]

        mov     r10, 0A1A1A1A1A1A1A1A1h             ; Assigned base address.

        mov     rax, 0B1B1B1B1B1B1B1B1h             ; Seed1.
        vmovd   xmm1, eax
        vpshufd xmm1, xmm1, 0

        mov     rax, 0C1C1C1C1C1C1C1C1h             ; Seed2.
        vmovd   xmm2, eax
        vpshufd xmm2, xmm2, 0

        vpmulld xmm3, xmm0, xmm1                   ; Vertex1 = Key * Seed1.
        vpmulld xmm4, xmm0, xmm2                   ; Vertex2 = Key * Seed2.

        mov     rax, 0D1D1D1D1D1D1D1D1h             ; Seed3 byte 1.
        mov     ecx, eax
        and     ecx, 31
        vmovd   xmm1, ecx
        vpsrld  xmm3, xmm3, xmm1

        mov     rax, 0E1E1E1E1E1E1E1E1h             ; Seed3 byte 2.
        mov     ecx, eax
        and     ecx, 31
        vmovd   xmm2, ecx
        vpsrld  xmm4, xmm4, xmm2

        mov     rax, 0F1F1F1F1F1F1F1F1h             ; Hash mask.
        vmovd   xmm1, eax
        vpshufd xmm1, xmm1, 0
        vpand   xmm3, xmm3, xmm1                   ; Mask vertex1.
        vpand   xmm4, xmm4, xmm1                   ; Mask vertex2.

        vmovdqu xmmword ptr [rsp + VERTEX1_OFFSET], xmm3
        vmovdqu xmmword ptr [rsp + VERTEX2_OFFSET], xmm4

        mov     rax, 02121212121212121h            ; Index mask.
        mov     r9d, eax

        mov     eax, dword ptr [rsp + VERTEX1_OFFSET + 00h]
        mov     ecx, dword ptr [rsp + VERTEX2_OFFSET + 00h]
        mov     eax, dword ptr [r10 + rax * 4]
        mov     ecx, dword ptr [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword ptr [r11 + OUTPUT_OFFSET + 00h]
        mov     dword ptr [rdx], eax

        mov     eax, dword ptr [rsp + VERTEX1_OFFSET + 04h]
        mov     ecx, dword ptr [rsp + VERTEX2_OFFSET + 04h]
        mov     eax, dword ptr [r10 + rax * 4]
        mov     ecx, dword ptr [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword ptr [r11 + OUTPUT_OFFSET + 08h]
        mov     dword ptr [rdx], eax

        mov     eax, dword ptr [rsp + VERTEX1_OFFSET + 08h]
        mov     ecx, dword ptr [rsp + VERTEX2_OFFSET + 08h]
        mov     eax, dword ptr [r10 + rax * 4]
        mov     ecx, dword ptr [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword ptr [r11 + OUTPUT_OFFSET + 10h]
        mov     dword ptr [rdx], eax

        mov     eax, dword ptr [rsp + VERTEX1_OFFSET + 0Ch]
        mov     ecx, dword ptr [rsp + VERTEX2_OFFSET + 0Ch]
        mov     eax, dword ptr [r10 + rax * 4]
        mov     ecx, dword ptr [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword ptr [r11 + OUTPUT_OFFSET + 18h]
        mov     dword ptr [rdx], eax

        vzeroupper

        add     rsp, LOCAL_STACK_SIZE
        ret                                        ; Return.

        ;IACA_VC_END

        LEAF_END PerfectHashJitIndexMultiplyShiftRIndex32x4Avx2_x64, _TEXT$00

; vim:set tw=80 ts=8 sw=4 sts=4 et syntax=masm fo=croql comments=\:;           :

end
