        title "Perfect Hash RawDog Mulshrolate1RX Index32x16 x64"

;++
;
; Copyright (c) 2026 Trent Nelson <trent@trent.me>
;
; Module Name:
;
;   PerfectHashJitIndexMulshrolate1RXIndex32x16_x64.asm
;
; Abstract:
;
;   This module implements the Mulshrolate1RX Index32x16() routine as a
;   position-independent blob suitable for RawDog JIT patching.
;
;--

include PerfectHash.inc

KEYS_OFFSET            equ 000h
VERTEX1_OFFSET         equ 040h
VERTEX2_OFFSET         equ 080h
LOCAL_STACK_SIZE       equ 0C0h
OUTPUT_OFFSET          equ 088h

;++
;
; VOID
; PerfectHashJitIndexMulshrolate1RXIndex32x16_x64(
;     _In_ ULONG Key1,
;     _In_ ULONG Key2,
;     _In_ ULONG Key3,
;     _In_ ULONG Key4,
;     _In_ ULONG Key5,
;     _In_ ULONG Key6,
;     _In_ ULONG Key7,
;     _In_ ULONG Key8,
;     _In_ ULONG Key9,
;     _In_ ULONG Key10,
;     _In_ ULONG Key11,
;     _In_ ULONG Key12,
;     _In_ ULONG Key13,
;     _In_ ULONG Key14,
;     _In_ ULONG Key15,
;     _In_ ULONG Key16,
;     _Out_ PULONG Index1,
;     _Out_ PULONG Index2,
;     _Out_ PULONG Index3,
;     _Out_ PULONG Index4,
;     _Out_ PULONG Index5,
;     _Out_ PULONG Index6,
;     _Out_ PULONG Index7,
;     _Out_ PULONG Index8,
;     _Out_ PULONG Index9,
;     _Out_ PULONG Index10,
;     _Out_ PULONG Index11,
;     _Out_ PULONG Index12,
;     _Out_ PULONG Index13,
;     _Out_ PULONG Index14,
;     _Out_ PULONG Index15,
;     _Out_ PULONG Index16
;     );
;
; Routine Description:
;
;   This routine implements the Mulshrolate1RX index functionality for sixteen
;   keys.  It is designed to be patched in-place by replacing the sentinel
;   values embedded in the instruction stream.
;
; Arguments:
;
;   Key1-Key4 (ecx, edx, r8d, r9d) - Supplies the first four keys.
;   Key5-Key16 ([rsp+0x28]..[rsp+0x80]) - Supplies the remaining keys.
;   Index1-Index16 ([rsp+0x88]..[rsp+0x100]) - Receives the resulting indices.
;
; Return Value:
;
;   None.
;
;--

        LEAF_ENTRY PerfectHashJitIndexMulshrolate1RXIndex32x16_x64, _TEXT$00, NoPad

        ;IACA_VC_START

        mov     r11, rsp                           ; Save stack base.
        sub     rsp, LOCAL_STACK_SIZE              ; Reserve local storage.

        mov     dword ptr [rsp + 00h], ecx         ; Store keys 1-4.
        mov     dword ptr [rsp + 04h], edx
        mov     dword ptr [rsp + 08h], r8d
        mov     dword ptr [rsp + 0Ch], r9d
        mov     eax, dword ptr [r11 + 028h]        ; Store keys 5-16.
        mov     dword ptr [rsp + 10h], eax
        mov     eax, dword ptr [r11 + 030h]
        mov     dword ptr [rsp + 14h], eax
        mov     eax, dword ptr [r11 + 038h]
        mov     dword ptr [rsp + 18h], eax
        mov     eax, dword ptr [r11 + 040h]
        mov     dword ptr [rsp + 1Ch], eax
        mov     eax, dword ptr [r11 + 048h]
        mov     dword ptr [rsp + 20h], eax
        mov     eax, dword ptr [r11 + 050h]
        mov     dword ptr [rsp + 24h], eax
        mov     eax, dword ptr [r11 + 058h]
        mov     dword ptr [rsp + 28h], eax
        mov     eax, dword ptr [r11 + 060h]
        mov     dword ptr [rsp + 2Ch], eax
        mov     eax, dword ptr [r11 + 068h]
        mov     dword ptr [rsp + 30h], eax
        mov     eax, dword ptr [r11 + 070h]
        mov     dword ptr [rsp + 34h], eax
        mov     eax, dword ptr [r11 + 078h]
        mov     dword ptr [rsp + 38h], eax
        mov     eax, dword ptr [r11 + 080h]
        mov     dword ptr [rsp + 3Ch], eax

        vmovdqu32 zmm0, zmmword ptr [rsp + KEYS_OFFSET]

        mov     r10, 0A1A1A1A1A1A1A1A1h             ; Assigned base address.

        mov     rax, 0B1B1B1B1B1B1B1B1h             ; Seed1.
        vmovd   xmm1, eax
        vpbroadcastd zmm1, xmm1

        mov     rax, 0C1C1C1C1C1C1C1C1h             ; Seed2.
        vmovd   xmm2, eax
        vpbroadcastd zmm2, xmm2

        vpmulld zmm3, zmm0, zmm1                   ; Vertex1 = Key * Seed1.
        vpmulld zmm4, zmm0, zmm2                   ; Vertex2 = Key * Seed2.

        mov     rax, 0E1E1E1E1E1E1E1E1h             ; Seed3 byte 2.
        mov     ecx, eax
        and     ecx, 31
        mov     edx, 32
        sub     edx, ecx
        vmovd   xmm1, ecx
        vmovd   xmm2, edx
        vpsrld  zmm0, zmm3, xmm1
        vpslld  zmm1, zmm3, xmm2
        vpor    zmm3, zmm0, zmm1

        mov     rax, 0D1D1D1D1D1D1D1D1h             ; Seed3 byte 1.
        mov     ecx, eax
        and     ecx, 31
        vmovd   xmm1, ecx

        vpsrld  zmm3, zmm3, xmm1
        vpsrld  zmm4, zmm4, xmm1

        vmovdqu32 zmmword ptr [rsp + VERTEX1_OFFSET], zmm3
        vmovdqu32 zmmword ptr [rsp + VERTEX2_OFFSET], zmm4

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

        mov     eax, dword ptr [rsp + VERTEX1_OFFSET + 20h]
        mov     ecx, dword ptr [rsp + VERTEX2_OFFSET + 20h]
        mov     eax, dword ptr [r10 + rax * 4]
        mov     ecx, dword ptr [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword ptr [r11 + OUTPUT_OFFSET + 40h]
        mov     dword ptr [rdx], eax

        mov     eax, dword ptr [rsp + VERTEX1_OFFSET + 24h]
        mov     ecx, dword ptr [rsp + VERTEX2_OFFSET + 24h]
        mov     eax, dword ptr [r10 + rax * 4]
        mov     ecx, dword ptr [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword ptr [r11 + OUTPUT_OFFSET + 48h]
        mov     dword ptr [rdx], eax

        mov     eax, dword ptr [rsp + VERTEX1_OFFSET + 28h]
        mov     ecx, dword ptr [rsp + VERTEX2_OFFSET + 28h]
        mov     eax, dword ptr [r10 + rax * 4]
        mov     ecx, dword ptr [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword ptr [r11 + OUTPUT_OFFSET + 50h]
        mov     dword ptr [rdx], eax

        mov     eax, dword ptr [rsp + VERTEX1_OFFSET + 2Ch]
        mov     ecx, dword ptr [rsp + VERTEX2_OFFSET + 2Ch]
        mov     eax, dword ptr [r10 + rax * 4]
        mov     ecx, dword ptr [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword ptr [r11 + OUTPUT_OFFSET + 58h]
        mov     dword ptr [rdx], eax

        mov     eax, dword ptr [rsp + VERTEX1_OFFSET + 30h]
        mov     ecx, dword ptr [rsp + VERTEX2_OFFSET + 30h]
        mov     eax, dword ptr [r10 + rax * 4]
        mov     ecx, dword ptr [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword ptr [r11 + OUTPUT_OFFSET + 60h]
        mov     dword ptr [rdx], eax

        mov     eax, dword ptr [rsp + VERTEX1_OFFSET + 34h]
        mov     ecx, dword ptr [rsp + VERTEX2_OFFSET + 34h]
        mov     eax, dword ptr [r10 + rax * 4]
        mov     ecx, dword ptr [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword ptr [r11 + OUTPUT_OFFSET + 68h]
        mov     dword ptr [rdx], eax

        mov     eax, dword ptr [rsp + VERTEX1_OFFSET + 38h]
        mov     ecx, dword ptr [rsp + VERTEX2_OFFSET + 38h]
        mov     eax, dword ptr [r10 + rax * 4]
        mov     ecx, dword ptr [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword ptr [r11 + OUTPUT_OFFSET + 70h]
        mov     dword ptr [rdx], eax

        mov     eax, dword ptr [rsp + VERTEX1_OFFSET + 3Ch]
        mov     ecx, dword ptr [rsp + VERTEX2_OFFSET + 3Ch]
        mov     eax, dword ptr [r10 + rax * 4]
        mov     ecx, dword ptr [r10 + rcx * 4]
        add     eax, ecx
        and     eax, r9d
        mov     rdx, qword ptr [r11 + OUTPUT_OFFSET + 78h]
        mov     dword ptr [rdx], eax

        vzeroupper

        add     rsp, LOCAL_STACK_SIZE
        ret                                        ; Return.

        ;IACA_VC_END

        LEAF_END PerfectHashJitIndexMulshrolate1RXIndex32x16_x64, _TEXT$00

; vim:set tw=80 ts=8 sw=4 sts=4 et syntax=masm fo=croql comments=\:;           :

end
