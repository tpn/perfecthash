; PerfectHashTable FastIndexEx x64 routines (NASM).

%include "PerfectHashNasm.inc"

default rel

section .text

global FastIndexEx_x64_01
global FastIndexEx_x64_02
global FastIndexEx_x64_03

global PatchInstructionSeed1Begin
global PatchInstructionSeed1End
global PatchInstructionSeed2Begin
global PatchInstructionSeed2End
global PatchInstructionSeed3Begin
global PatchInstructionSeed3End
global PatchInstructionHashMaskBegin
global PatchInstructionHashMaskEnd
global PatchInstructionIndexMaskBegin
global PatchInstructionIndexMaskEnd

; ULONG
; FastIndexEx_x64_01(
;     _In_ PPERFECT_HASH_TABLE Table,
;     _In_ ULONG Key
;     );
FastIndexEx_x64_01:
    mov     rcx, PH_ARG1
    mov     rdx, PH_ARG2

    mov     eax, edx
    mov     r10, [rcx + PH_TABLE_DATA]
    mov     r11d, edx

PatchInstructionSeed1Begin:
    mov     ecx, Seed1
PatchInstructionSeed1End:

    crc32   ecx, edx

PatchInstructionSeed3Begin:
    xor     eax, Seed3
PatchInstructionSeed3End:

PatchInstructionSeed2Begin:
    mov     r9d, Seed2
PatchInstructionSeed2End:

    rol     edx, 0fh
    crc32   edx, r9d

    crc32   edx, eax

PatchInstructionHashMaskBegin:
    mov     eax, HashMask
PatchInstructionHashMaskEnd:

    and     ecx, eax
    and     edx, eax

    mov     rax, r10
    mov     ecx, dword [rax + rcx * 4]
    mov     eax, dword [rax + rdx * 4]

    add     rax, rcx
PatchInstructionIndexMaskBegin:
    and     eax, IndexMask
PatchInstructionIndexMaskEnd:

    ret

section .note.GNU-stack noalloc noexec nowrite progbits

; ULONG
; FastIndexEx_x64_02(
;     _In_ PULONG Assigned,
;     _In_ ULONG Key
;     );
FastIndexEx_x64_02:
    mov     rcx, PH_ARG1
    mov     rdx, PH_ARG2

    mov     eax, edx
    mov     r8d, Seed1
    mov     r11d, HashMask
    mov     r9d, Seed2
    mov     r10d, Seed3

    crc32   r8d, edx
    and     r8d, r11d
    mov     eax, dword [rcx + r8 * 4]

    xor     r10d, edx
    rol     edx, 0fh
    crc32   edx, r9d
    crc32   edx, r10d

    and     edx, r11d
    mov     edx, dword [rcx + rdx * 4]

    add     rax, rdx
    and     eax, IndexMask

    ret

; ULONG
; FastIndexEx_x64_03(
;     _In_ PULONG Assigned,
;     _In_ ULONG Key
;     );
FastIndexEx_x64_03:
    mov     rcx, PH_ARG1
    mov     rdx, PH_ARG2

    mov     eax, edx
    mov     r8d, Seed1
    mov     r11d, HashMask
    mov     r9d, Seed2
    mov     r10d, Seed3

    crc32   r8d, edx
    and     r8d, r11d
    mov     eax, dword [rcx + r8 * 4]

    xor     r10d, edx
    rol     edx, 0fh
    crc32   edx, r9d
    crc32   edx, r10d

    and     edx, r11d
    mov     edx, dword [rcx + rdx * 4]

    add     rax, rdx
    and     eax, IndexMask

    ret
