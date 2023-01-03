        title "PerfectHash Assembly Include File"
        option nokeyword:<Length>

;++
;
; Copyright (c) Trent Nelson, 2016-2022.
;
; Module Name:
;
;   PerfectHash.inc
;
; Abstract:
;
;   This is an assembly include module for the PerfectHash library.
;
;--

include ksamd64.inc

IACA_VC_START macro Name

        mov     byte ptr gs:[06fh], 06fh

        endm

IACA_VC_END macro Name

        mov     byte ptr gs:[0deh], 0deh

        endm


;
; Declare constants we'll be patching in the final instruction stream.
;

Seed1       equ 011111111h
Seed2       equ 022222222h
Seed3       equ 033333333h
Seed4       equ 044444444h

HashMask    equ 0aaaaaaaah
IndexMask   equ 0bbbbbbbbh

;
; Helper typedefs.
;

CHAR typedef byte
UCHAR typedef byte
PCHAR typedef ptr byte
USHORT typedef word
WCHAR typedef word
PVOID typedef qword
LONG typedef dword
ULONG typedef dword
ULONG64 typedef qword
ULONGLONG typedef qword
PULONG typedef ptr ULONG
PULONGLONG typedef ptr ULONGLONG
PUCHAR typedef ptr UCHAR

PARAMS struct
    ReturnAddress dq ?
    union
        HomeRcx dq ?
        Param1  dq ?
    ends

    union
        HomeRdx dq ?
        Param2  dq ?
    ends

    union
        HomeR8  dq ?
        Param3  dq ?
    ends

    union
        HomeR9  dq ?
        Param4  dq ?
    ends
PARAMS ends

HOME_PARAMS struct
    ReturnAddress           dq      ?       ; 8     32      40      (28h)
    HomeRcx                 dq      ?       ; 8     24      32      (20h)
    HomeRdx                 dq      ?       ; 8     16      24      (18h)
    HomeR8                  dq      ?       ; 8     8       16      (10h)
    HomeR9                  dq      ?       ; 8     0       8       (08h)
HOME_PARAMS ends

Home typedef HOME_PARAMS

PERFECT_HASH_TABLE struct
    Vtbl dq ?
    Data dq ?
    Values dq ?
    SizeOfStruct dd ?
    Flags dd ?
PERFECT_HASH_TABLE ends

Table typedef PERFECT_HASH_TABLE

;
; ETW glue.
;

TRACEHANDLE typedef ULONG64

MCGEN_TRACE_CONTEXT struct
    RegistrationHandle      TRACEHANDLE ?
    Logger                  TRACEHANDLE ?
    MatchAnyKeyword         ULONGLONG ?
    MatchAllKeyword         ULONGLONG ?
    Flags                   ULONG ?
    IsEnabled               ULONG ?
    Level                   UCHAR ?
    Reserve                 UCHAR ?
    EnableBitsCount         USHORT ?
    EnableBitMask           PULONG ?
    EnableKeyWords          PULONGLONG ?
    EnableLevel             PUCHAR ?
MCGEN_TRACE_CONTEXT ends

EVENT_DATA_DESCRIPTOR struct
    Pointer         dq ?
    SizeInBytes     dd ?

    union
        Reserved        dw ?
        struct
            EventType   db ?
            Reserved1   db ?
            Reserved2   dw ?
        ends
    ends
EVENT_DATA_DESCRIPTOR ends

EventDataDescriptor typedef EVENT_DATA_DESCRIPTOR

_MODULEINFO struct
    BaseAddress     PVOID   ?
    SizeOfImage     DWORD   ?
    EntryPoint      PVOID   ?
_MODULEINFO ends

ModuleInfo typedef _MODULEINFO

; vim:set tw=80 ts=8 sw=4 sts=4 expandtab syntax=masm                          :

