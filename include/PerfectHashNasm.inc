; PerfectHash NASM include file (x64).

%ifndef PERFECTHASH_NASM_INC
%define PERFECTHASH_NASM_INC 1

BITS 64

%ifidn __OUTPUT_FORMAT__, win64
%define PH_WIN64 1
%endif

%ifdef PH_WIN64
%define PH_ARG1 rcx
%define PH_ARG1D ecx
%define PH_ARG1B cl
%define PH_ARG2 rdx
%define PH_ARG2D edx
%define PH_ARG2B dl
%define PH_ARG3 r8
%define PH_ARG3D r8d
%define PH_ARG3B r8b
%define PH_ARG4 r9
%define PH_ARG4D r9d
%define PH_ARG4B r9b
%else
%define PH_ARG1 rdi
%define PH_ARG1D edi
%define PH_ARG1B dil
%define PH_ARG2 rsi
%define PH_ARG2D esi
%define PH_ARG2B sil
%define PH_ARG3 rdx
%define PH_ARG3D edx
%define PH_ARG3B dl
%define PH_ARG4 rcx
%define PH_ARG4D ecx
%define PH_ARG4B cl
%endif

%define PAGE_SHIFT 12
%define PAGE_SIZE (1 << PAGE_SHIFT)

%define Seed1     0x11111111
%define Seed2     0x22222222
%define Seed3     0x33333333
%define Seed4     0x44444444

%define HashMask  0xaaaaaaaa
%define IndexMask 0xbbbbbbbb

%define PH_TABLE_VTBL          0
%define PH_TABLE_DATA          8
%define PH_TABLE_VALUES        16
%define PH_TABLE_SIZEOFSTRUCT  24
%define PH_TABLE_FLAGS         28

%endif
