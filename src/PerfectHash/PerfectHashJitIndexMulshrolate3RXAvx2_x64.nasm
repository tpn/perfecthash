;++
;
; Copyright (c) 2025 Trent Nelson <trent@trent.me>
;
; Module Name:
;
;   PerfectHashJitIndexMulshrolate3RXAvx2_x64.nasm
;
; Abstract:
;
;   This module implements the Mulshrolate3RX Index32x8() routine using AVX2
;   as a position-independent blob suitable for RawDog JIT patching.
;
;--

        bits 64
        default rel

        section .text

        global PerfectHashJitIndexMulshrolate3RXAvx2_x64

;+++
;
; VOID
; PerfectHashJitIndexMulshrolate3RXAvx2_x64(
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
;   This routine implements the Mulshrolate3RX Index32x8() functionality.  It
;   is designed to be patched in-place by replacing the sentinel values in the
;   embedded data block that follows the routine.
;
; Arguments:
;
;   Key1-Key6 (edi, esi, edx, ecx, r8d, r9d) - Supplies the first six keys.
;   Key7-Key8 ([rsp+0x08], [rsp+0x10]) - Supplies the remaining keys.
;   Index1-Index8 ([rsp+0x18] .. [rsp+0x50]) - Receives the resulting indices.
;
; Return Value:
;
;   None.
;
;--

        align 16
PerfectHashJitIndexMulshrolate3RXAvx2_x64:

        ;IACA_VC_START

        mov     r11, rsp                       ; Save stack base.
        sub     rsp, 0x80                      ; Reserve local storage.

        mov     dword [rsp + 0x00], edi        ; Store keys 1-4.
        mov     dword [rsp + 0x04], esi
        mov     dword [rsp + 0x08], edx
        mov     dword [rsp + 0x0c], ecx
        mov     dword [rsp + 0x10], r8d        ; Store keys 5-6.
        mov     dword [rsp + 0x14], r9d
        mov     eax, dword [r11 + 0x08]        ; Load key7.
        mov     dword [rsp + 0x18], eax
        mov     eax, dword [r11 + 0x10]        ; Load key8.
        mov     dword [rsp + 0x1c], eax

        vmovdqu ymm0, [rsp + 0x00]             ; Load keys.

        vpbroadcastd ymm1, dword [rel RawDogSeed1]
        vpbroadcastd ymm2, dword [rel RawDogSeed2]
        vpbroadcastd ymm15, dword [rel RawDogSeed4]

        vpmulld ymm3, ymm0, ymm1               ; Vertex1 = Key * Seed1.
        vpmulld ymm4, ymm0, ymm2               ; Vertex2 = Key * Seed2.

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

        vpsrld  ymm10, ymm3, xmm6              ; ror(Vertex1, Seed3_Byte2).
        vpslld  ymm11, ymm3, xmm7
        vpor    ymm3, ymm10, ymm11
        vpmulld ymm3, ymm3, ymm15              ; Vertex1 *= Seed4.
        vpsrld  ymm3, ymm3, xmm5               ; Vertex1 >>= Seed3_Byte1.

        vpsrld  ymm10, ymm4, xmm8              ; ror(Vertex2, Seed3_Byte3).
        vpslld  ymm11, ymm4, xmm9
        vpor    ymm4, ymm10, ymm11
        vpsrld  ymm4, ymm4, xmm5               ; Vertex2 >>= Seed3_Byte1.

        mov     r10, [rel RawDogAssigned]
        vpcmpeqd ymm12, ymm12, ymm12           ; Gather mask = all ones.
        vpgatherdd ymm13, [r10 + ymm3 * 4], ymm12
        vpcmpeqd ymm12, ymm12, ymm12           ; Reset gather mask.
        vpgatherdd ymm14, [r10 + ymm4 * 4], ymm12

        vpaddd  ymm13, ymm13, ymm14            ; Vertex1 + Vertex2.
        vpbroadcastd ymm15, dword [rel RawDogIndexMask]
        vpand   ymm13, ymm13, ymm15

        vmovdqu [rsp + 0x40], ymm13            ; Store indices.

        mov     eax, dword [rsp + 0x40]
        mov     r10, [r11 + 0x18]
        mov     dword [r10], eax
        mov     eax, dword [rsp + 0x44]
        mov     r10, [r11 + 0x20]
        mov     dword [r10], eax
        mov     eax, dword [rsp + 0x48]
        mov     r10, [r11 + 0x28]
        mov     dword [r10], eax
        mov     eax, dword [rsp + 0x4c]
        mov     r10, [r11 + 0x30]
        mov     dword [r10], eax
        mov     eax, dword [rsp + 0x50]
        mov     r10, [r11 + 0x38]
        mov     dword [r10], eax
        mov     eax, dword [rsp + 0x54]
        mov     r10, [r11 + 0x40]
        mov     dword [r10], eax
        mov     eax, dword [rsp + 0x58]
        mov     r10, [r11 + 0x48]
        mov     dword [r10], eax
        mov     eax, dword [rsp + 0x5c]
        mov     r10, [r11 + 0x50]
        mov     dword [r10], eax

        vzeroupper

        add     rsp, 0x80
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
