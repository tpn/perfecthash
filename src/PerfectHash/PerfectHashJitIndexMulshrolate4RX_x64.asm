        title "Perfect Hash RawDog Mulshrolate4RX x64"

; Windows x64 index kernels.  CMake assembles this file once per key count
; and assigned element width.  Each sentinel occurs once in the emitted blob.

include PerfectHash.inc

LoadAssignedPair MACRO
    IF ASSIGNED_BITS EQ 16
        movzx   eax, word ptr [r10 + rax * 2]
        movzx   edx, word ptr [r10 + rdx * 2]
    ELSE
        mov     eax, dword ptr [r10 + rax * 4]
        mov     edx, dword ptr [r10 + rdx * 4]
    ENDIF
ENDM

IF KEY_COUNT NE 1

; The home space and the by-value key slots provide two vectors of scratch
; space.  Pack the keys in ascending order to keep unread stack keys intact.
; The output pointers follow the scratch space.  RSP remains at entry.
VECTOR_BYTES equ KEY_COUNT * 4
FIRST_VECTOR equ 8
SECOND_VECTOR equ FIRST_VECTOR + VECTOR_BYTES

IF KEY_COUNT EQ 4
    keysVector  textequ <xmm0>
    firstVertex textequ <xmm1>
    lastVertex  textequ <xmm2>
    seedVector  textequ <xmm3>
    temporary   textequ <xmm5>
    moveVector  textequ <vmovdqu>
    vectorMem   textequ <xmmword ptr>
ELSEIF KEY_COUNT EQ 8
    keysVector  textequ <ymm0>
    firstVertex textequ <ymm1>
    lastVertex  textequ <ymm2>
    seedVector  textequ <ymm3>
    temporary   textequ <ymm5>
    moveVector  textequ <vmovdqu>
    vectorMem   textequ <ymmword ptr>
ELSE
    keysVector  textequ <zmm0>
    firstVertex textequ <zmm1>
    lastVertex  textequ <zmm2>
    seedVector  textequ <zmm3>
    moveVector  textequ <vmovdqu32>
    vectorMem   textequ <zmmword ptr>
ENDIF

BroadcastSeed MACRO sentinel
        mov     rax, sentinel
        vmovd   xmm3, eax
    IF KEY_COUNT EQ 4
        vpshufd xmm3, xmm3, 0
    ELSE
        vpbroadcastd seedVector, xmm3
    ENDIF
ENDM

RotateVertex MACRO vertex, sentinel
    IF KEY_COUNT EQ 16
        BroadcastSeed sentinel
        vprorvd vertex, vertex, seedVector
    ELSE
        mov     rax, sentinel
        and     eax, 31
        vmovd   xmm3, eax
        mov     edx, 32
        sub     edx, eax
        vmovd   xmm4, edx
        vpsrld  temporary, vertex, xmm3
        vpslld  vertex, vertex, xmm4
        vpor    vertex, vertex, temporary
    ENDIF
ENDM
ENDIF

        LEAF_ENTRY ENTRY_NAME, _TEXT$00, NoPad

        mov     r10, 0A1A1A1A1A1A1A1A1h         ; Assigned base address.
IF KEY_COUNT EQ 1
        mov     eax, ecx
        mov     edx, ecx
        mov     r8,  0B1B1B1B1B1B1B1B1h         ; Seed1.
        imul    eax, r8d
        mov     rcx, 0E1E1E1E1E1E1E1E1h         ; Seed3 byte 2.
        ror     eax, cl
        mov     r8,  0B2B2B2B2B2B2B2B2h         ; Seed4.
        imul    eax, r8d

        mov     r8,  0C1C1C1C1C1C1C1C1h         ; Seed2.
        imul    edx, r8d
        mov     rcx, 0D2D2D2D2D2D2D2D2h         ; Seed3 byte 3.
        ror     edx, cl
        mov     r8,  0B3B3B3B3B3B3B3B3h         ; Seed5.
        imul    edx, r8d

        mov     rcx, 0D1D1D1D1D1D1D1D1h         ; Seed3 byte 1.
        shr     eax, cl
        shr     edx, cl
        LoadAssignedPair
        add     eax, edx
        mov     r9,  02121212121212121h         ; Index mask.
        and     eax, r9d
ELSE
        mov     dword ptr [rsp + FIRST_VECTOR + 0], ecx
        mov     dword ptr [rsp + FIRST_VECTOR + 4], edx
        mov     dword ptr [rsp + FIRST_VECTOR + 8], r8d
        mov     dword ptr [rsp + FIRST_VECTOR + 12], r9d
lane = 4
        REPEAT KEY_COUNT - 4
            mov     eax, dword ptr [rsp + 8 + 8 * lane]
            mov     dword ptr [rsp + FIRST_VECTOR + 4 * lane], eax
lane = lane + 1
        ENDM
        moveVector keysVector, vectorMem [rsp + FIRST_VECTOR]

        BroadcastSeed 0B1B1B1B1B1B1B1B1h        ; Seed1.
        vpmulld firstVertex, keysVector, seedVector
        BroadcastSeed 0C1C1C1C1C1C1C1C1h        ; Seed2.
        vpmulld lastVertex, keysVector, seedVector
        RotateVertex firstVertex, 0E1E1E1E1E1E1E1E1h
        RotateVertex lastVertex,  0D2D2D2D2D2D2D2D2h
        BroadcastSeed 0B2B2B2B2B2B2B2B2h        ; Seed4.
        vpmulld firstVertex, firstVertex, seedVector
        BroadcastSeed 0B3B3B3B3B3B3B3B3h        ; Seed5.
        vpmulld lastVertex, lastVertex, seedVector

        mov     rax, 0D1D1D1D1D1D1D1D1h         ; Seed3 byte 1.
        and     eax, 31
        vmovd   xmm3, eax
        vpsrld  firstVertex, firstVertex, xmm3
        vpsrld  lastVertex, lastVertex, xmm3
        moveVector vectorMem [rsp + FIRST_VECTOR], firstVertex
        moveVector vectorMem [rsp + SECOND_VECTOR], lastVertex

        mov     r9, 02121212121212121h          ; Index mask.
lane = 0
        REPEAT KEY_COUNT
            mov     eax, dword ptr [rsp + FIRST_VECTOR + 4 * lane]
            mov     edx, dword ptr [rsp + SECOND_VECTOR + 4 * lane]
            LoadAssignedPair
            add     eax, edx
            and     eax, r9d
            mov     rcx, [rsp + 8 + 8 * (KEY_COUNT + lane)]
            mov     [rcx], eax
lane = lane + 1
        ENDM

        vzeroupper
ENDIF
        ret
        LEAF_END ENTRY_NAME, _TEXT$00

end
