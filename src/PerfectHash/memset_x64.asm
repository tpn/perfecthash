        title "Perfect Hash Memset"

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
;   The ETW event support results in the compiler injecting memset() calls,
;   but we don't have a memset() because we don't link against a C runtime
;   library.  So, implement a quick hacky memset in assembly using good ol'
;   rep stosb.
;
;--

include PerfectHash.inc

        NESTED_ENTRY memset, _TEXT$00
        mov         qword ptr [rsp + 8], rdi        ; Home rdi.
        END_PROLOGUE
        mov         r9, rcx
        mov         rdi, rcx
        movzx       eax, dl
        mov         rcx, r8
        rep stos    byte ptr [rdi]

;
; Begin epilogue.
;
        mov         rdi, qword ptr [rsp + 8]        ; Restore rdi.
        mov         rax, r9
        ret
        NESTED_END memset, _TEXT$00

; vim:set tw=80 ts=8 sw=4 sts=4 et syntax=masm fo=croql comments=\:;           :

end
