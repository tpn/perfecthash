        title "FunctionHook _penter"

;++
;
; Copyright (c) 2022-2023 Trent Nelson <trent@trent.me>
;
; Module Name:
;
;   _penter_x64.asm
;
; Abstract:
;
;   This module implements _penter routines that will call a configured callback
;   routine with the appropriate RIP address.  It is used to evaluate runtime
;   performance of compiled perfect hash tables that have been generated from
;   PerfectHash.dll binaries compiled with /Gh (hook functions with _penter).
;
;--

include PerfectHash.inc

;
; Externs defined in FunctionHook.c.
;

EXTERN FunctionEntryCallback:PTR PROC
EXTERN FunctionEntryCallbackContext:PTR PVOID
EXTERN HookedModuleBaseAddress:PVOID
EXTERN HookedModuleSizeInBytes:ULONG
EXTERN HookedModuleIgnoreRip:ULONG

;
; Locals struct.
;

Locals struct

    ;
    ; Define home parameter space.
    ;

    CalleeHomeRcx           dq      ?
    CalleeHomeRdx           dq      ?
    CalleeHomeR8            dq      ?
    CalleeHomeR9            dq      ?

    ;
    ; Saved volatile registers.
    ;

    SavedRax                dq      ?
    SavedRcx                dq      ?
    SavedRdx                dq      ?
    SavedR8                 dq      ?
    SavedR9                 dq      ?
    SavedR10                dq      ?
    SavedR11                dq      ?
    SavedXmm0               xmmword ?

    Padding1                dq      ?
    Padding2                dq      ?

    ReturnAddress           dq      ?

Locals ends

;
; Exclude the return address onward from the frame calculation size.
;

LOCALS_SIZE equ ((sizeof Locals) + (Locals.ReturnAddress - (sizeof Locals)))

;++
;
; VOID
; _penter_fast(
;     VOID
;     );
;
; Routine Description:
;
;   "Fast" implementation of _penter, which only persists volatile registers
;   as and when needed.
;
;   N.B. Hooking _penter is one of the rare occasions where volatile registers
;        need to be persisted; if we didn't do this, optimized builds would
;        crash in weird and wonderful ways.  This is because the compiler knows
;        everything related to register usage when calling internal routines,
;        and will regularly rely on volatile registers having their values
;        persist between function calls because it *knows*, for example, no
;        function in calls from a given point is going to write to RDX.
;
;        _penter breaks this assumption.  Or rather, we break this assumption
;        because we need to use volatile registers in order to carry out the
;        _penter tasks.  So, we *always* persist any volatile register we mutate
;        in this routine.  As an optimization, we only persist those we mutate,
;        prior to mutating them.  This avoids unnecessary ops if no callback is
;        configured, for example.
;
; Arguments:
;
;   None.
;
; Return Value:
;
;   None.
;
;--

        NESTED_ENTRY _penter_fast, _TEXT$00
        alloc_stack LOCALS_SIZE                     ; Allocate stack space.
        END_PROLOGUE

        mov     Locals.SavedR10[rsp], r10           ; Save r10.

;
; Check if the extern pointer HookedModuleBaseAddress has been set.  If it
; hasn't, fast-path exit.
;

        mov     r10, HookedModuleBaseAddress        ; Load module info ptr.
        test    r10, r10                            ; Is null?
        jz      Pe99                                ; Yes, exit.

;
; Save rax and rcx prior to use.  Next jump target for cleanup is Pe97.
;

        mov     Locals.SavedRax[rsp], rax           ; Save rax.
        mov     Locals.SavedRcx[rsp], rcx           ; Save rcx.

;
; Load the RIP from the return address.
;
;   rcx: Return RIP
;   r10: Base module address
;   eax: Module size in bytes
;   rdx: Normalized return RIP (i.e. return RIP - base module address)
;

        lea     rax, Locals.ReturnAddress[rsp]      ; Load RSP address.
        mov     rcx, [rax]                          ; Load return RIP.
        sub     rcx, 5                              ; Sub the call _penter len.

        cmp     rcx, r10                            ; Is RIP < base address?
        jl      Pe97                                ; Yes, out of bounds, exit.

;
; Save rdx prior to use.  Next jump target for cleanup is Pe95.
;

        mov     Locals.SavedRdx[rsp], rdx

;
; Load the RIP into rdx and then subtract the base image address.  Compare this
; value to the image size: if it's greater than, then we're out of bounds (i.e.
; not being called from a module we know about), so exit.
;

        xor     rax, rax                            ; Clear rax.
        mov     eax, HookedModuleSizeInBytes        ; Load image size.
        mov     rdx, rcx                            ; Load RIP into rdx.
        sub     rdx, r10                            ; Sub base address.
        cmp     rdx, rax                            ; Is RIP > base + size?
        jge     Pe95                                ; Yes, out of bounds, exit.

;
; Save r11 prior to use.  Next jump target for cleanup is Pe93.
;

        mov     Locals.SavedR11[rsp], r11

;
; Load the HookedModuleIgnoreRip into r11, and, if it's not zero, check to see
; if it matches our relative RIP in rdx.  If it does, then we've been asked to
; ignore this RIP, so jump to the end.
;

        xor     r11, r11                            ; Clear r11.
        mov     r11d, HookedModuleIgnoreRip         ; Load ignored RIP.
        test    r11d, r11d                          ; Is 0?
        jz      @F                                  ; Yes, skip check.
        cmp     edx, r11d                           ; Does RIP match ignored?
        je      Pe93                                ; Yes, exit.

;
; The normalized return RIP is in rdx.  Final step: load the callback pointer,
; and ensure it's not null.
;

@@:     mov     r10, FunctionEntryCallback          ; Load callback ptr.
        test    r10, r10                            ; Is null?
        jz      Pe93                                ; Yes, exit.

;
; All of our preconditions have been satisified.  Load the normalized return
; RIP into rcx (1st param), context into rdx (2nd param), save the remaining
; volatile registers, then call the callback.
;

        mov     rcx, rdx                            ; Load RIP into 1st param.
        mov     rdx, FunctionEntryCallbackContext   ; Load context ptr 2nd param.

;
; Save the remaining volatile registers prior to invoking the callback.
;

        mov     Locals.SavedR8[rsp],  r8
        mov     Locals.SavedR9[rsp],  r9
        vmovdqu xmmword ptr Locals.SavedXmm0[rsp], xmm0

;
; Invoke the callback.
;

        xor     r8, r8                              ; Clear r8.
        xor     r9, r9                              ; Clear r9.
        call    r10                                 ; Call the function callback.

;
; Restore volatile registers in the order we used them.
;

        mov     r8,  Locals.SavedR8[rsp]
        mov     r9,  Locals.SavedR9[rsp]
        vmovdqu xmm0, xmmword ptr Locals.SavedXmm0[rsp]

Pe93:   mov     r11, Locals.SavedR11[rsp]

Pe95:   mov     rdx, Locals.SavedRdx[rsp]

Pe97:   mov     rax, Locals.SavedRax[rsp]
        mov     rcx, Locals.SavedRcx[rsp]

Pe99:   mov     r10, Locals.SavedR10[rsp]

;
; Begin epilogue.
;

        add     rsp, LOCALS_SIZE                    ; Deallocate stack space.
        ret
        NESTED_END _penter_fast, _TEXT$00

;++
;
; VOID
; _penter_slow(
;     VOID
;     );
;
; Routine Description:
;
;   "Slow" implementation of _penter, which saves all applicable volatile
;   registers at the start of the routine, regardless of whether or not
;   a callback function is active.
;
;   This routine, despite being slower than _penter_fast, can still be useful
;   as it lowers the gap between _penter overhead when a callback is active
;   versus not.
;
;   For example, using _penter fast, solving HologramWorld-31016.keys with
;   Chm01 MultiplyShiftR And 24 (on a 24-core box), I get about 21,000 attempts
;   per second with the Index() callback active, and ~34,000 attempts with the
;   callback disabled.  (A non _penter build clocks in at nearly 40,000 attempts
;   per second.)
;
;   A slow _penter build clocks in at about 24,000 attempts per second when
;   no callback is active, and about 21,000 attempts per second when the Index()
;   routine is active as a callback.  So we can see that our Index() routine is
;   responsible for only ~3,000 attempts per second, not ~13,000 attempts per
;   second if we use the fast routine.
;
; Arguments:
;
;   None.
;
; Return Value:
;
;   None.
;
;--

        NESTED_ENTRY _penter_slow, _TEXT$00
        alloc_stack LOCALS_SIZE                     ; Allocate stack space.
        END_PROLOGUE

;
; Save all registers up front.
;

        mov     Locals.SavedRax[rsp], rax
        mov     Locals.SavedRcx[rsp], rcx
        mov     Locals.SavedRdx[rsp], rdx
        mov     Locals.SavedR8[rsp],  r8
        mov     Locals.SavedR9[rsp],  r9
        mov     Locals.SavedR10[rsp], r10
        mov     Locals.SavedR11[rsp], r11
        vmovdqu xmmword ptr Locals.SavedXmm0[rsp], xmm0

;
; Check if the extern pointer HookedModuleBaseAddress has been set.  If it
; hasn't, fast-path exit.
;

        mov     r10, HookedModuleBaseAddress        ; Load module info ptr.
        test    r10, r10                            ; Is null?
        jz      Ps90                                ; Yes, exit.

;
; Load the RIP from the return address.
;
;   rcx: Return RIP
;   r10: Base module address
;   eax: Module size in bytes
;   rdx: Normalized return RIP (i.e. return RIP - base module address)
;

        lea     rax, Locals.ReturnAddress[rsp]      ; Load RSP address.
        mov     rcx, [rax]                          ; Load return RIP.
        sub     rcx, 5                              ; Sub the call _penter len.

        cmp     rcx, r10                            ; Is RIP < base address?
        jl      Ps90                                ; Yes, out of bounds, exit.

;
; Load the RIP into rdx and then subtract the base image address.  Compare this
; value to the image size: if it's greater than, then we're out of bounds (i.e.
; not being called from a module we know about), so exit.
;

        xor     rax, rax                            ; Clear rax.
        mov     eax, HookedModuleSizeInBytes        ; Load image size.
        mov     rdx, rcx                            ; Load RIP into rdx.
        sub     rdx, r10                            ; Sub base address.
        cmp     rdx, rax                            ; Is RIP > base + size?
        jge     Ps90                                ; Yes, out of bounds, exit.

;
; Load the HookedModuleIgnoreRip into r11, and, if it's not zero, check to see
; if it matches our relative RIP in rdx.  If it does, then we've been asked to
; ignore this RIP, so jump to the end.
;

        xor     r11, r11                            ; Clear r11.
        mov     r11d, HookedModuleIgnoreRip         ; Load ignored RIP.
        test    r11d, r11d                          ; Is 0?
        jz      @F                                  ; Yes, skip check.
        cmp     edx, r11d                           ; Does RIP match ignored?
        je      Ps90                                ; Yes, exit.

;
; The normalized return RIP is in rdx.  Final step: load the callback pointer,
; and ensure it's not null.
;

@@:     mov     r10, FunctionEntryCallback          ; Load callback ptr.
        test    r10, r10                            ; Is null?
        jz      Ps90                                ; Yes, exit.

;
; All of our preconditions have been satisified.  Load the normalized return
; RIP into rcx (1st param), context into rdx (2nd param), save the remaining
; volatile registers, then call the callback.
;

        mov     rcx, rdx                            ; Load RIP into 1st param.
        mov     rdx, FunctionEntryCallbackContext   ; Load context ptr 2nd param.

;
; Invoke the callback.
;

        xor     r8, r8                              ; Clear r8.
        xor     r9, r9                              ; Clear r9.
        call    r10                                 ; Call the function callback.

;
; Restore volatile registers.
;

Ps90:   mov     rax, Locals.SavedRax[rsp]
        mov     rcx, Locals.SavedRcx[rsp]
        mov     rdx, Locals.SavedRdx[rsp]
        mov     r8,  Locals.SavedR8[rsp]
        mov     r9,  Locals.SavedR9[rsp]
        mov     r10, Locals.SavedR10[rsp]
        mov     r11, Locals.SavedR11[rsp]
        vmovdqu xmm0, xmmword ptr Locals.SavedXmm0[rsp]

;
; Begin epilogue.
;

        add     rsp, LOCALS_SIZE                    ; Deallocate stack space.
        ret
        NESTED_END _penter_slow, _TEXT$00


; vim:set tw=80 ts=8 sw=4 sts=4 et syntax=masm fo=croql comments=\:;           :

end
