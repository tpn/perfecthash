        title "Perfect Hash Table AMD64 Index Routines"

;++
;
; Copyright (c) 2018 Trent Nelson <trent@trent.me>
;
; Module Name:
;
;   PerfectHashTableFastIndexEx_x64_*.asm
;
; Abstract:
;
;   This module implements the FastIndexEx() routine.
;
;--

include PerfectHashTable.inc

;
; Forward declaration of addresses we'll be patching.
;

altentry PatchInstructionSeed1Begin
altentry PatchInstructionSeed1End
altentry PatchInstructionSeed2Begin
altentry PatchInstructionSeed2End
altentry PatchInstructionSeed3Begin
altentry PatchInstructionSeed3End
altentry PatchInstructionHashMaskBegin
altentry PatchInstructionHashMaskEnd
altentry PatchInstructionIndexMaskBegin
altentry PatchInstructionIndexMaskEnd

;++
;
; ULONG
; FastIndexEx(
;     _In_ PPERFECT_HASH_TABLE Table,
;     _In_ ULONG Key
;     );
;
; Routine Description:
;
;   This routine implements the CRC32-Rotate-XOR index functionality.
;
; Arguments:
;
;   Table (rcx) - Supplies a pointer to the perfect hash table for which the
;       index operation is to be performed.
;
;   Key (rdx) - Supplies the key for which an index is to be obtained.
;
; Return Value:
;
;   The index corresponding to the given key.
;
;--

        LEAF_ENTRY FastIndexEx_x64_01, _TEXT$00

        IACA_VC_START

        mov     eax, edx                        ; Copy Key into rax.
        mov     r10, Table.Data[rcx]            ; Copy Assigned into r10.
        mov     r11d, edx                       ; Copy Key to r11d.

ALTERNATE_ENTRY PatchInstructionSeed1Begin
        mov     ecx, Seed1                      ; Copy Seed1 into rcx.
ALTERNATE_ENTRY PatchInstructionSeed1End

        crc32   ecx, edx                        ; Calc CRC32 of Seed1 + Key

ALTERNATE_ENTRY PatchInstructionSeed3Begin
        xor     eax, Seed3                      ; Calc Seed3 ^ Key.
ALTERNATE_ENTRY PatchInstructionSeed3End

ALTERNATE_ENTRY PatchInstructionSeed2Begin
        mov     r9d, Seed2                      ; Load second seed into r9.
ALTERNATE_ENTRY PatchInstructionSeed2End

        rol     edx, 0fh                        ; Rotate Key left 15 bits.
        crc32   edx, r9d                        ; Calc CRC32 of Seed2 + Key.

        crc32   edx, eax                        ; Calc CRC32 of (Seed3 ^ Key)
                                                ;   and CRC32(Seed2, Key).

;
; ecx now holds the first hash value, edx the second.  Load the hash mask into
; rax, then use it to mask both ecx and edx (using AND), generating our two
; offsets into the assigned array.  From there, we load the corresponding
; vertex values, then combine them into a final value, which we then mask via
; the index mask.
;

ALTERNATE_ENTRY PatchInstructionHashMaskBegin
        mov     eax, HashMask                   ; Load hash mask into rax.
ALTERNATE_ENTRY PatchInstructionHashMaskEnd

        and     ecx, eax                        ; Mask hash value 1.
        and     edx, eax                        ; Mask hash value 2.

        mov     rax, r10                        ; Load base address into rax.
        mov     ecx, dword ptr [rax + rcx * 4]  ; Load vertex 1.
        mov     eax, dword ptr [rax + rdx * 4]  ; Load vertex 2.

        add     rax, rcx                        ; Add the vertices together.
ALTERNATE_ENTRY PatchInstructionIndexMaskBegin
        and     eax, IndexMask                  ; Mask the return index.
ALTERNATE_ENTRY PatchInstructionIndexMaskEnd

        IACA_VC_END

        ret                                     ; Return.

        LEAF_END FastIndexEx_x64_01, _TEXT$00

;++
;
; ULONG
; FastIndexEx(
;     _In_ PULONG Assigned,
;     _In_ ULONG Key
;     );
;
; Routine Description:
;
;   This routine implements the CRC32-Rotate-XOR index functionality.
;
; Arguments:
;
;   Assigned (rcx) - Supplies the base address of the assigned array for which
;       the vertex values are to serve as an index.
;
;   Key (rdx) - Supplies the key for which an index is to be obtained.
;
; Return Value:
;
;   The index corresponding to the given key.
;
;--

        LEAF_ENTRY FastIndexEx_x64_02, _TEXT$00

        ;IACA_VC_START

        mov     eax, edx                        ; Copy Key into rax.
        mov     r8d, Seed1                      ; Copy Seed1 into r8d.
        mov     r11d, HashMask                  ; Load hash mask into r11d.
        mov     r9d, Seed2                      ; Copy Seed2 into r9d.
        mov     r10d, Seed3                     ; Copy Seed3 into r10d.

        crc32   r8d, edx                        ; Calc CRC32 of Seed1 & Key.
        and     r8d, r11d                       ; Mask hash value 1.
        mov     eax, dword ptr [rcx + r8 * 4]   ; Load vertex 1.

        xor     r10d, edx                       ; Calc Seed3 ^ Key.
        rol     edx, 0fh                        ; Rotate Key left 15 bites.
        crc32   edx, r9d                        ; Calc CRC of Seed2 & (Key<<15).
        crc32   edx, r10d                       ; Calc final CRC32.

        and     edx, r11d                       ; Mask hash value 2.
        mov     edx, dword ptr [rcx + rdx * 4]  ; Load vertex 2.

        add     rax, rdx                        ; Add the two vertices.
        and     rax, IndexMask                  ; Mask the index.

        ;IACA_VC_END

        ret                                     ; Return.

        LEAF_END FastIndexEx_x64_02, _TEXT$00

;++
;
; ULONG
; FastIndexEx(
;     _In_ PULONG Assigned,
;     _In_ ULONG Key
;     );
;
; Routine Description:
;
;   This routine implements the CRC32-Rotate-XOR index functionality.
;
; Arguments:
;
;   Assigned (rcx) - Supplies the base address of the assigned array for which
;       the vertex values are to serve as an index.
;
;   Key (rdx) - Supplies the key for which an index is to be obtained.
;
; Return Value:
;
;   The index corresponding to the given key.
;
;--

        LEAF_ENTRY FastIndexEx_x64_03, _TEXT$00

        ;IACA_VC_START

        mov     eax, edx                        ; Copy Key into rax.
        mov     r8d, Seed1                      ; Copy Seed1 into r8d.
        mov     r11d, HashMask                  ; Load hash mask into r11d.
        mov     r9d, Seed2                      ; Copy Seed2 into r9d.
        mov     r10d, Seed3                     ; Copy Seed3 into r10d.

        crc32   r8d, edx                        ; Calc CRC32 of Seed1 & Key.
        and     r8d, r11d                       ; Mask hash value 1.
        mov     eax, dword ptr [rcx + r8 * 4]   ; Load vertex 1.

        xor     r10d, edx                       ; Calc Seed3 ^ Key.
        rol     edx, 0fh                        ; Rotate Key left 15 bites.
        crc32   edx, r9d                        ; Calc CRC of Seed2 & (Key<<15).
        crc32   edx, r10d                       ; Calc final CRC32.

        and     edx, r11d                       ; Mask hash value 2.
        mov     edx, dword ptr [rcx + rdx * 4]  ; Load vertex 2.

        add     rax, rdx                        ; Add the two vertices.
        and     rax, IndexMask                  ; Mask the index.

        ;IACA_VC_END

        ret                                     ; Return.

        LEAF_END FastIndexEx_x64_03, _TEXT$00


; vim:set tw=80 ts=8 sw=4 sts=4 et syntax=masm fo=croql comments=\:;           :

end
