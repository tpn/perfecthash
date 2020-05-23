        title "Perfect Hash Memset Hack"

;++
;
; Copyright (c) 2020 Trent Nelson <trent@trent.me>
;
; Module Name:
;
;   memset_x64.asm
;
; Abstract:
;
;   This is a hack for getting around quirky memset issues.
;
;--

include PerfectHash.inc

extern _memset:proc

    LEAF_ENTRY memset, _TEXT$00
    jmp _memset
    LEAF_END memset, _TEXT$00

; vim:set tw=80 ts=8 sw=4 sts=4 et syntax=masm fo=croql comments=\:;           :

end
