        title "Perfect Hash RawDog Mulshrolate2RX Avx2 v3 x64"

;++
;
; Copyright (c) 2026 Trent Nelson <trent@trent.me>
;
; Module Name:
;
;   PerfectHashJitIndexMulshrolate2RXAvx2_v3_x64.asm
;
; Abstract:
;
;   This module implements the Mulshrolate2RX Index32x8() routine using AVX2
;   as a position-independent blob suitable for RawDog JIT patching.
;
;--

include PerfectHash.inc

KEYS_OFFSET            equ 000h
VERTEX1_OFFSET         equ 020h
VERTEX2_OFFSET         equ 040h
LOCAL_STACK_SIZE       equ 060h
OUTPUT_OFFSET          equ 048h

;++
;
; VOID
; PerfectHashJitIndexMulshrolate2RXAvx2_x64(
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
;   This routine implements the Mulshrolate2RX index functionality for eight
;   keys using AVX2 instructions.  It is designed to be patched in-place by
;   replacing the sentinel values embedded in the instruction stream.
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

        LEAF_ENTRY PerfectHashJitIndexMulshrolate2RXAvx2_x64, _TEXT$00, NoPad

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

        vmovdqu ymm0, ymmword ptr [rsp + KEYS_OFFSET]

        mov     r10, 0A1A1A1A1A1A1A1A1h             ; Assigned base address.

        mov     rax, 0B1B1B1B1B1B1B1B1h             ; Seed1.
        vmovd   xmm1, eax
        vpbroadcastd ymm1, xmm1

        mov     rax, 0C1C1C1C1C1C1C1C1h             ; Seed2.
        vmovd   xmm2, eax
        vpbroadcastd ymm2, xmm2

        vpmulld ymm3, ymm0, ymm1                   ; Vertex1 = Key * Seed1.
        vpmulld ymm4, ymm0, ymm2                   ; Vertex2 = Key * Seed2.

        mov     rax, 0E1E1E1E1E1E1E1E1h             ; Seed3 byte 2.
        mov     ecx, eax
        and     ecx, 31
        mov     edx, 32
        sub     edx, ecx
        vmovd   xmm1, ecx
        vmovd   xmm2, edx
        vpsrld  ymm0, ymm3, xmm1
        vpslld  ymm1, ymm3, xmm2
        vpor    ymm3, ymm0, ymm1

        mov     rax, 0D2D2D2D2D2D2D2D2h             ; Seed3 byte 3.
        mov     ecx, eax
        and     ecx, 31
        mov     edx, 32
        sub     edx, ecx
        vmovd   xmm1, ecx
        vmovd   xmm2, edx
        vpsrld  ymm0, ymm4, xmm1
        vpslld  ymm1, ymm4, xmm2
        vpor    ymm4, ymm0, ymm1

        mov     rax, 0D1D1D1D1D1D1D1D1h             ; Seed3 byte 1.
        mov     ecx, eax
        and     ecx, 31
        vmovd   xmm1, ecx

        vpsrld  ymm3, ymm3, xmm1
        vpsrld  ymm4, ymm4, xmm1

        vmovdqu ymmword ptr [rsp + VERTEX1_OFFSET], ymm3
        vmovdqu ymmword ptr [rsp + VERTEX2_OFFSET], ymm4

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

        mov     eax, dword ptr [rsp + VERTEX1_OFFSET + 10h]
        mov     ecx, dword ptr [rsp + VERTEX2_OFFSET + 10h]
        mov     eax, dword ptr [r10 + rax * 4]
        mov     ecx, dword ptr [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword ptr [r11 + OUTPUT_OFFSET + 20h]
        mov     dword ptr [rdx], eax

        mov     eax, dword ptr [rsp + VERTEX1_OFFSET + 14h]
        mov     ecx, dword ptr [rsp + VERTEX2_OFFSET + 14h]
        mov     eax, dword ptr [r10 + rax * 4]
        mov     ecx, dword ptr [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword ptr [r11 + OUTPUT_OFFSET + 28h]
        mov     dword ptr [rdx], eax

        mov     eax, dword ptr [rsp + VERTEX1_OFFSET + 18h]
        mov     ecx, dword ptr [rsp + VERTEX2_OFFSET + 18h]
        mov     eax, dword ptr [r10 + rax * 4]
        mov     ecx, dword ptr [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword ptr [r11 + OUTPUT_OFFSET + 30h]
        mov     dword ptr [rdx], eax

        mov     eax, dword ptr [rsp + VERTEX1_OFFSET + 1Ch]
        mov     ecx, dword ptr [rsp + VERTEX2_OFFSET + 1Ch]
        mov     eax, dword ptr [r10 + rax * 4]
        mov     ecx, dword ptr [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword ptr [r11 + OUTPUT_OFFSET + 38h]
        mov     dword ptr [rdx], eax

        vzeroupper

        add     rsp, LOCAL_STACK_SIZE
        ret                                        ; Return.

        ;IACA_VC_END

        LEAF_END PerfectHashJitIndexMulshrolate2RXAvx2_x64, _TEXT$00

; vim:set tw=80 ts=8 sw=4 sts=4 et syntax=masm fo=croql comments=\:;           :

end
