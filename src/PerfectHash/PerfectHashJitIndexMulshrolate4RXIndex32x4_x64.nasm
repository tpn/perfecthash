; Four-key AVX kernel for the System V AMD64 ABI.  The four input keys
; arrive in edi, esi, edx, ecx.  The output pointers follow in r8, r9 and
; the first two stack arguments.

        bits 64
        default rel
        section .text

%ifdef ASSIGNED16
        global PerfectHashJitIndexMulshrolate4RX16Index32x4_x64
PerfectHashJitIndexMulshrolate4RX16Index32x4_x64:
%else
        global PerfectHashJitIndexMulshrolate4RXIndex32x4_x64
PerfectHashJitIndexMulshrolate4RXIndex32x4_x64:
%endif
        vmovd   xmm0, edi
        vpinsrd xmm0, xmm0, esi, 1
        vpinsrd xmm0, xmm0, edx, 2
        vpinsrd xmm0, xmm0, ecx, 3

%macro hash_vertex 4                    ; vertex, seed, rotation, multiplier
        vpshufd %1, [rel %2], 0
        vpmulld %1, xmm0, %1
        mov     ecx, [rel %3]
        and     ecx, 31
        vmovd   xmm3, ecx
        mov     edx, 32
        sub     edx, ecx
        vmovd   xmm4, edx
        vpsrld  xmm5, %1, xmm3
        vpslld  %1, %1, xmm4
        vpor    %1, %1, xmm5
        vpshufd xmm3, [rel %4], 0
        vpmulld %1, %1, xmm3
%endmacro
        hash_vertex xmm1, RawDogSeed1, RawDogSeed3Byte2, RawDogSeed4
        hash_vertex xmm2, RawDogSeed2, RawDogSeed3Byte3, RawDogSeed5
        mov     ecx, [rel RawDogSeed3Byte1]
        and     ecx, 31
        vmovd   xmm3, ecx
        vpsrld  xmm1, xmm1, xmm3
        vpsrld  xmm2, xmm2, xmm3

        mov     r10, [rel RawDogAssigned]
        mov     r11d, [rel RawDogIndexMask]
%assign lane 0
%rep 4
        vpextrd eax, xmm1, lane
        vpextrd edx, xmm2, lane
%ifdef ASSIGNED16
        movzx   eax, word [r10 + rax * 2]
        movzx   edx, word [r10 + rdx * 2]
%else
        mov     eax, [r10 + rax * 4]
        mov     edx, [r10 + rdx * 4]
%endif
        add     eax, edx
        and     eax, r11d
%if lane = 0
        mov     [r8], eax
%elif lane = 1
        mov     [r9], eax
%else
        mov     rcx, [rsp + (lane - 1) * 8]
        mov     [rcx], eax
%endif
%assign lane lane + 1
%endrep
        ret

        align 8
RawDogAssigned:   dq 0xA1A1A1A1A1A1A1A1
RawDogSeed1:      dq 0xB1B1B1B1B1B1B1B1
RawDogSeed2:      dq 0xC1C1C1C1C1C1C1C1
RawDogSeed3Byte1: dq 0xD1D1D1D1D1D1D1D1
RawDogSeed3Byte2: dq 0xE1E1E1E1E1E1E1E1
RawDogSeed3Byte3: dq 0xD2D2D2D2D2D2D2D2
RawDogSeed4:      dq 0xB2B2B2B2B2B2B2B2
RawDogSeed5:      dq 0xB3B3B3B3B3B3B3B3
RawDogIndexMask:  dq 0x2121212121212121
