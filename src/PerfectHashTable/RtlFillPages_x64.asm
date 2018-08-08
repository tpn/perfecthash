        title "Fill Pages AVX2 Assembly Routine"

;++
;
; Copyright (c) 2017 Trent Nelson <trent@trent.me>
;
; Module Name:
;
;   FillPages.asm
;
; Abstract:
;
;   This module implements various routines for filling pages of memory with
;   a byte pattern.
;
;--

include Asm.inc

;++
;
; VOID
; FillPagesNonTemporalAvx2_v1(
;     _Out_writes_bytes_all_(NumberOfPages << PAGE_SHIFT) PCHAR Dest,
;     _In_ BYTE Byte,
;     _In_ ULONG NumberOfPages
;     );
;
; Routine Description:
;
;   This routine fills one or more pages of memory with a given byte pattern
;   using non-temporal hints.
;
; Arguments:
;
;   Destination (rcx) - Supplies the address of the target memory.
;
;   Byte (rdx) - Supplies the byte to fill the pages of memory with.
;
;   NumberOfPages (r8) - Supplies the number of pages to copy.
;
; Return Value:
;
;   None.
;
;--

        LEAF_ENTRY FillPagesNonTemporalAvx2_v1, _TEXT$00

;
; Verify the NumberOfPages is greater than zero.
;

        test    r8, r8                      ; Test NumberOfPages against self.
        jz      short Fpx05                 ; Number of pages is 0, return.
        jmp     short Fpx06                 ; Number of pages >= 1, continue.

Fpx05:  ret

;
; Broadcast the byte value in rdx to a YMM register (by way of an XMM reg).
;

Fpx06:  movd            xmm0, rdx
        vpbroadcastb    ymm1, xmm0

;
; This routine uses the following pattern for processing pages (inspired by
; KeCopyPage()): initialize a counter to -PAGE_SIZE, add PAGE_SIZE to both
; pointer targets, use [<base_reg> + <counter_reg> + constant * 0..3] for the
; four successive prefetch/load/store instructions.
;
; Thus, we move -PAGE_SIZE (0xffffffff`fffff000) to r10 once, up front.  When
; we advance the rcx destination pointer, we sub r10 from each value, and before
; entering each loop, we reset rax to r10, then increment it until it hits zero.
; Thus, the two instructions after the mov r10 line below can be seen at
; multiple points in this routine.
;
; See also: CopyPages implementations.
;

Fpx07:  mov     r10, -PAGE_SIZE
        sub     rcx, r10                    ; Add page size to Destination ptr.
        mov     rax, r10                    ; Initialize counter register.

;
; Copy the YMM byte register into the destination memory a page at a time.
;

        align   16

Fpx10:  vmovntdq ymmword ptr [rcx + rax + 32 * 0], ymm1     ; Copy   0 -  31.
        vmovntdq ymmword ptr [rcx + rax + 32 * 1], ymm1     ; Copy  32 -  63.
        vmovntdq ymmword ptr [rcx + rax + 32 * 2], ymm1     ; Copy  64 -  95.
        vmovntdq ymmword ptr [rcx + rax + 32 * 3], ymm1     ; Copy  96 - 127.

;
; Increment the rax counter; multiply the number of registers in flight (4)
; by the register size (32 bytes).  Jump back to the start of the copy loop
; if we haven't copied 4096 bytes yet.
;

        add     rax, 32 * 4             ; Increment counter register.
        jnz     short Fpx10             ; Repeat copy whilst bytes copied != 4k.

;
; Decrement our NumberOfPages counter (r8).  If zero, we've copied all pages,
; and can jump to the end (Fpx80).
;
; N.B. `sub r8, 1` is used instead of `dec r8` because the latter introduces
;      a false dependency on the flags as it doesn't clear CF.
;

        sub     r8, 1                       ; --NumberOfPages.
        jz      short Fpx80                 ; No more pages, finalize.

;
; There are pages remaining.  Update our destination pointer by a page size
; again, and initialize rax to -PAGE_SIZE, then jump back to the start.
;

        sub     rcx, r10                    ; Add page size to Destination ptr.
        mov     rax, r10                    ; Reinitialize counter register.
        jmp     short Fpx10                 ; Jump back to start.

;
; Force a memory barrier and return.
;

Fpx80:  sfence

Fpx90:  ret

        LEAF_END FillPagesNonTemporalAvx2_v1, _TEXT$00

; vim:set tw=80 ts=8 sw=4 sts=4 et syntax=masm fo=croql comments=:;            :

end
