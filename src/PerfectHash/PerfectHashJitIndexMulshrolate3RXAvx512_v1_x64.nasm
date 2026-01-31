;++
;
; Copyright (c) 2025 Trent Nelson <trent@trent.me>
;
; Module Name:
;
;   PerfectHashJitIndexMulshrolate3RXAvx512_v1_x64.nasm
;
; Abstract:
;
;   This module implements the Mulshrolate3RX Index32x16() routine using
;   AVX-512 as a position-independent blob suitable for RawDog JIT patching.
;
;--

        bits 64
        default rel

        section .text

        global PerfectHashJitIndexMulshrolate3RXAvx512_x64

;+++
;
; VOID
; PerfectHashJitIndexMulshrolate3RXAvx512_x64(
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
;   This routine implements the Mulshrolate3RX Index32x16() functionality.  It
;   is designed to be patched in-place by replacing the sentinel values in the
;   embedded data block that follows the routine.
;
; Arguments:
;
;   Key1-Key6 (edi, esi, edx, ecx, r8d, r9d) - Supplies the first six keys.
;   Key7-Key16 ([rsp+0x08] .. [rsp+0x50]) - Supplies the remaining keys.
;   Index1-Index16 ([rsp+0x58] .. [rsp+0xd0]) - Receives the resulting indices.
;
; Return Value:
;
;   None.
;
;--

        align 16
PerfectHashJitIndexMulshrolate3RXAvx512_x64:

        ;IACA_VC_START

        mov     r11, rsp                       ; Save stack base.
        sub     rsp, 0x100                     ; Reserve local storage.

        mov     dword [rsp + 0x00], edi        ; Store keys 1-4.
        mov     dword [rsp + 0x04], esi
        mov     dword [rsp + 0x08], edx
        mov     dword [rsp + 0x0c], ecx
        mov     dword [rsp + 0x10], r8d        ; Store keys 5-6.
        mov     dword [rsp + 0x14], r9d
        mov     eax, dword [r11 + 0x08]        ; Load keys 7-16.
        mov     dword [rsp + 0x18], eax
        mov     eax, dword [r11 + 0x10]
        mov     dword [rsp + 0x1c], eax
        mov     eax, dword [r11 + 0x18]
        mov     dword [rsp + 0x20], eax
        mov     eax, dword [r11 + 0x20]
        mov     dword [rsp + 0x24], eax
        mov     eax, dword [r11 + 0x28]
        mov     dword [rsp + 0x28], eax
        mov     eax, dword [r11 + 0x30]
        mov     dword [rsp + 0x2c], eax
        mov     eax, dword [r11 + 0x38]
        mov     dword [rsp + 0x30], eax
        mov     eax, dword [r11 + 0x40]
        mov     dword [rsp + 0x34], eax
        mov     eax, dword [r11 + 0x48]
        mov     dword [rsp + 0x38], eax
        mov     eax, dword [r11 + 0x50]
        mov     dword [rsp + 0x3c], eax

        vmovdqu32 zmm0, [rsp + 0x00]           ; Load keys.

        vpbroadcastd zmm1, dword [rel RawDogSeed1]
        vpbroadcastd zmm2, dword [rel RawDogSeed2]
        vpmulld zmm3, zmm0, zmm1               ; Vertex1 = Key * Seed1.
        vpmulld zmm4, zmm0, zmm2               ; Vertex2 = Key * Seed2.

        vpbroadcastd zmm0, dword [rel RawDogSeed4]

        mov     eax, dword [rel RawDogSeed3Byte1]
        mov     ecx, dword [rel RawDogSeed3Byte2]
        mov     edx, dword [rel RawDogSeed3Byte3]
        and     eax, 31
        and     ecx, 31
        and     edx, 31

        mov     r8d, 32
        sub     r8d, ecx                       ; 32 - Seed3_Byte2.
        mov     r9d, 32
        sub     r9d, edx                       ; 32 - Seed3_Byte3.

        vmovd   xmm5, eax                      ; Seed3_Byte1.
        vmovd   xmm6, ecx                      ; Seed3_Byte2.
        vmovd   xmm7, r8d                      ; 32 - Seed3_Byte2.
        vmovd   xmm8, edx                      ; Seed3_Byte3.
        vmovd   xmm9, r9d                      ; 32 - Seed3_Byte3.

        vpsrld  zmm10, zmm3, xmm6              ; ror(Vertex1, Seed3_Byte2).
        vpslld  zmm11, zmm3, xmm7
        vpord   zmm3, zmm10, zmm11
        vpmulld zmm3, zmm3, zmm0               ; Vertex1 *= Seed4.
        vpsrld  zmm3, zmm3, xmm5               ; Vertex1 >>= Seed3_Byte1.

        vpsrld  zmm10, zmm4, xmm8              ; ror(Vertex2, Seed3_Byte3).
        vpslld  zmm11, zmm4, xmm9
        vpord   zmm4, zmm10, zmm11
        vpsrld  zmm4, zmm4, xmm5               ; Vertex2 >>= Seed3_Byte1.

        mov     eax, 0xffff
        kmovw   k1, eax

        mov     r10, [rel RawDogAssigned]
        vpgatherdd zmm13{k1}, [r10 + zmm3 * 4]
        kmovw   k1, eax                        ; Reset gather mask.
        vpgatherdd zmm14{k1}, [r10 + zmm4 * 4]

        vpaddd  zmm13, zmm13, zmm14            ; Vertex1 + Vertex2.
        vpbroadcastd zmm15, dword [rel RawDogIndexMask]
        vpandd  zmm13, zmm13, zmm15

        vmovdqu32 [rsp + 0x80], zmm13          ; Store indices.

        mov     eax, dword [rsp + 0x80]
        mov     r10, [r11 + 0x58]
        mov     dword [r10], eax
        mov     eax, dword [rsp + 0x84]
        mov     r10, [r11 + 0x60]
        mov     dword [r10], eax
        mov     eax, dword [rsp + 0x88]
        mov     r10, [r11 + 0x68]
        mov     dword [r10], eax
        mov     eax, dword [rsp + 0x8c]
        mov     r10, [r11 + 0x70]
        mov     dword [r10], eax
        mov     eax, dword [rsp + 0x90]
        mov     r10, [r11 + 0x78]
        mov     dword [r10], eax
        mov     eax, dword [rsp + 0x94]
        mov     r10, [r11 + 0x80]
        mov     dword [r10], eax
        mov     eax, dword [rsp + 0x98]
        mov     r10, [r11 + 0x88]
        mov     dword [r10], eax
        mov     eax, dword [rsp + 0x9c]
        mov     r10, [r11 + 0x90]
        mov     dword [r10], eax
        mov     eax, dword [rsp + 0xa0]
        mov     r10, [r11 + 0x98]
        mov     dword [r10], eax
        mov     eax, dword [rsp + 0xa4]
        mov     r10, [r11 + 0xa0]
        mov     dword [r10], eax
        mov     eax, dword [rsp + 0xa8]
        mov     r10, [r11 + 0xa8]
        mov     dword [r10], eax
        mov     eax, dword [rsp + 0xac]
        mov     r10, [r11 + 0xb0]
        mov     dword [r10], eax
        mov     eax, dword [rsp + 0xb0]
        mov     r10, [r11 + 0xb8]
        mov     dword [r10], eax
        mov     eax, dword [rsp + 0xb4]
        mov     r10, [r11 + 0xc0]
        mov     dword [r10], eax
        mov     eax, dword [rsp + 0xb8]
        mov     r10, [r11 + 0xc8]
        mov     dword [r10], eax
        mov     eax, dword [rsp + 0xbc]
        mov     r10, [r11 + 0xd0]
        mov     dword [r10], eax

        vzeroupper

        add     rsp, 0x100
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

