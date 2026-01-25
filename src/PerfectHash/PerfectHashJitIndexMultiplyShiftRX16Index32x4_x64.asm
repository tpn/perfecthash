        title "Perfect Hash RawDog MultiplyShiftRX16 Index32x4 x64"

;++
;
; Copyright (c) 2026 Trent Nelson <trent@trent.me>
;
; Module Name:
;
;   PerfectHashJitIndexMultiplyShiftRX16Index32x4_x64.asm
;
; Abstract:
;
;   This module implements the MultiplyShiftRX Index32x4() routine for 16-bit
;   assigned tables as a position-independent blob suitable for RawDog JIT
;   patching.
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
; PerfectHashJitIndexMultiplyShiftRX16Index32x4_x64(
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
;   This routine implements the MultiplyShiftRX index functionality for four
;   keys against 16-bit assigned table data.  It is designed to be patched
;   in-place by replacing the sentinel values embedded in the instruction
;   stream.
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

        LEAF_ENTRY PerfectHashJitIndexMultiplyShiftRX16Index32x4_x64, _TEXT$00, NoPad

        ;IACA_VC_START

        mov     r11, rsp                           ; Save stack base.
        sub     rsp, LOCAL_STACK_SIZE              ; Reserve local storage.

        mov     dword ptr [rsp + 00h], ecx         ; Store keys 1-4.
        mov     dword ptr [rsp + 04h], edx
        mov     dword ptr [rsp + 08h], r8d
        mov     dword ptr [rsp + 0Ch], r9d

        movdqu  xmm0, xmmword ptr [rsp + KEYS_OFFSET]

        mov     r10, 0A1A1A1A1A1A1A1A1h             ; Assigned base address.

        mov     rax, 0B1B1B1B1B1B1B1B1h             ; Seed1.
        movd    xmm1, eax
        pshufd  xmm1, xmm1, 0

        mov     rax, 0C1C1C1C1C1C1C1C1h             ; Seed2.
        movd    xmm2, eax
        pshufd  xmm2, xmm2, 0

        movdqa  xmm3, xmm0
        pmulld  xmm3, xmm1                         ; Vertex1 = Key * Seed1.
        movdqa  xmm4, xmm0
        pmulld  xmm4, xmm2                         ; Vertex2 = Key * Seed2.

        mov     rax, 0D1D1D1D1D1D1D1D1h             ; Seed3 byte 1.
        mov     ecx, eax
        and     ecx, 31
        movd    xmm1, ecx
        psrld   xmm3, xmm1

        psrld   xmm4, xmm1

        movdqu  xmmword ptr [rsp + VERTEX1_OFFSET], xmm3
        movdqu  xmmword ptr [rsp + VERTEX2_OFFSET], xmm4

        mov     rax, 02121212121212121h            ; Index mask.
        mov     r9d, eax

        mov     eax, dword ptr [rsp + VERTEX1_OFFSET + 00h]
        mov     ecx, dword ptr [rsp + VERTEX2_OFFSET + 00h]
        movzx   eax, word ptr [r10 + rax * 2]
        movzx   ecx, word ptr [r10 + rcx * 2]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword ptr [r11 + OUTPUT_OFFSET + 00h]
        mov     dword ptr [rdx], eax

        mov     eax, dword ptr [rsp + VERTEX1_OFFSET + 04h]
        mov     ecx, dword ptr [rsp + VERTEX2_OFFSET + 04h]
        movzx   eax, word ptr [r10 + rax * 2]
        movzx   ecx, word ptr [r10 + rcx * 2]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword ptr [r11 + OUTPUT_OFFSET + 08h]
        mov     dword ptr [rdx], eax

        mov     eax, dword ptr [rsp + VERTEX1_OFFSET + 08h]
        mov     ecx, dword ptr [rsp + VERTEX2_OFFSET + 08h]
        movzx   eax, word ptr [r10 + rax * 2]
        movzx   ecx, word ptr [r10 + rcx * 2]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword ptr [r11 + OUTPUT_OFFSET + 10h]
        mov     dword ptr [rdx], eax

        mov     eax, dword ptr [rsp + VERTEX1_OFFSET + 0Ch]
        mov     ecx, dword ptr [rsp + VERTEX2_OFFSET + 0Ch]
        movzx   eax, word ptr [r10 + rax * 2]
        movzx   ecx, word ptr [r10 + rcx * 2]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword ptr [r11 + OUTPUT_OFFSET + 18h]
        mov     dword ptr [rdx], eax

        add     rsp, LOCAL_STACK_SIZE
        ret                                        ; Return.

        ;IACA_VC_END

        LEAF_END PerfectHashJitIndexMultiplyShiftRX16Index32x4_x64, _TEXT$00

; vim:set tw=80 ts=8 sw=4 sts=4 et syntax=masm fo=croql comments=\:;           :

end
