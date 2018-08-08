/*++

    XXX: copied from the tracer project.

Copyright (c) 2017 Trent Nelson <trent@trent.me>

Module Name:

    SymbolLoader.c

Abstract:

    This module implements the symbol loading routines for the Rtl component.
    Routines are provided for loading an array of symbol names from a single
    module, and loading an array of symbol names from multiple modules.

    This functionality is geared toward the pattern of using structures to
    capture function pointers, as is common in both the Rtl component and
    other Tracer components.  For example::

        #define _NTDLL_FUNCTIONS_HEAD                           \
            PRTL_SUFFIX_STRING RtlSuffixString;                 \
            PRTL_LOOKUP_FUNCTION_ENTRY RtlLookupFunctionEntry;

        typedef struct _NTDLL_FUNCTIONS_HEAD {
            _NTDLL_FUNCTIONS_HEAD
        } NTDLL_FUNCTIONS_HEAD, *PNTDLL_FUNCTIONS_HEAD;

        typedef struct _FOO {

            union {
                NTDLL_FUNCTIONS_HEAD NtdllFunctions;
                struct {
                    _NTDLL_FUNCTIONS_HEAD
                };
            };

        } FOO, *PFOO;

    The following code could then be used to resolve the inner ntdll functions
    structure:

        BOOL Success;
        FOO Foo;
        RTL_BITMAP FailedBitmap;
        ULONG BitmapBuffer = 0;
        ULONG NumberOfResolvedSymbols;

        HMODULE NtdllModule = LoadLibraryA("ntdll");

        PSTR Names[] = {
            "RtlSuffixString",
            "RtlLookupFunctionEntry",
        };

        FailedBitmap.SizeOfBitmap = ARRAYSIZE(Names);
        FailedBitmap.Buffer = &BitmapBuffer;

        Success = LoadSymbols(Names,
                              ARRAYSIZE(Names),
                              (PULONG_PTR)&Foo->NtdllFunctions,
                              sizeof(Foo->NtdllFunctions) / sizeof(ULONG_PTR),
                              NtdllModule,
                              &FailedBitmap,
                              &NumberOfResolvedSymbols);

    This will resolve the function pointers embedded within the structure,
    allowing them to be used as normal, e.g.:

        Foo->RtlSuffixString(...);

    N.B. The pattern of #defining a _HEAD macro containing all of the function
         pointer type definitions is used for all Tracer components.  It has the
         following advantages:

            - Programmatic generation of the boilerplate glue to declare the
              PSTR Names[] array and call LoadSymbols().

            - Allows for the anonymous union inner struct pattern, where the
              entire structure can be referred to directly (Foo->NtdllFunctions)
              but function pointers are "passed-through" and can be accessed via
              Foo->RtlSuffixString() instead of Foo->NtdllFunctions->Rtl...
              This is preferred solely for aesthetic reasons.

         The disadvantage of this approach is that ctags/cscope/IntelliSense
         and various other code tools don't handle the inline _HEAD macro
         as nicely as explicitly listing out each member, e.g.:

            typedef struct _FOO {

                union {

                    NTDLL_FUNCTIONS NtdllFunctions;

                    //
                    // Inline NTDLL_FUNCTIONS.
                    //

                    struct {
                        PRTL_SUFFIX_STRING RtlSuffixString;
                        PRTL_LOOKUP_FUNCTION_ENTRY RtlLookupFunctionEntry;
                    };
                };

            } FOO, *PFOO;

         The long term plan for dealing with this disadvantage is to extend
         the automation glue to detect comments such as "Inline <structname>"
         and automatically sync the next anonymous struct with the contents
         of the given struct.  This would also obviate the need to use a
         the #define _HEAD ... pattern.

--*/

#include "stdafx.h"

//
// Suppress spectre warnings due to /Wall.
//

#pragma warning(push)
#pragma warning(disable : 4619)
#pragma warning(disable : 5045)

LOAD_SYMBOLS LoadSymbols;

_Use_decl_annotations_
BOOLEAN
LoadSymbols(
    CONST PCSZ *SymbolNameArray,
    ULONG NumberOfSymbolNames,
    PULONG_PTR SymbolAddressArray,
    ULONG NumberOfSymbolAddresses,
    HMODULE Module,
    PRTL_BITMAP FailedSymbols,
    PULONG NumberOfResolvedSymbolsPointer
    )
/*++

Routine Description:

    This routine is used to dynamically resolve an array of symbol names
    from a single module (DLL) via GetProcAddress(), storing the resulting
    addresses in symbol address array.  If a symbol name cannot be resolved,
    its corresponding failure bit is set in the failed symbol bitmap.

    See also: LoadSymbolsFromMultipleModules().

Arguments:

    SymbolNameArray - Supplies a pointer to an array of NULL-terminated C
        strings, with each string representing a name to look up (that is,
        call GetProcAddress() on).

    NumberOfSymbolNames - Supplies the number of elements in the parameter
        SymbolNameArray.  This must match NumberOfSymbolAddresses, otherwise,
        FALSE is returned.

    SymbolAddressArray - Supplies the address of an array of variables that will
        receive the address corresponding to each symbol in the names array, or
        NULL if symbol lookup failed.

    NumberOfSymbolAddresses - Supplies the number of elements in the parameter
        SymbolAddressArray.  This must match NumberOfSymbolNames, otherwise,
        FALSE is returned.

    Module - Supplies the HMODULE to use in the GetProcAddress() call.

    FailedSymbols - Supplies a pointer to an RTL_BITMAP structure, whose
        SizeOfBitmap field is equal to or greater than NumberOfSymbolNames.
        The entire bitmap buffer will be zeroed up-front and then a bit will
        be set for any failed lookup attempts.

        N.B. As the names and addresses arrays are 0-based, the array index is
             obtained by the failed bit position by subtracting 1.  E.g. if the
             first bit set is 5, that corresponds to Names[4].

    NumberOfResolvedSymbolsPointer - Supplies the address of a variable that
        will receive the number of successfully resolved symbols.

Return Value:

    If all parameters are successfully validated, TRUE will be returned
    (regardless of the results of the actual symbol loading attempts).
    FALSE is returned if any invalid parameters are detected.

    If a symbol cannot be resolved, the symbol's corresponding bit will be set
    in the FailedBitmap, and the corresponding pointer will be set to NULL in
    the SymbolAddressesArray array.

--*/
{
    ULONG Index;
    ULONG NumberOfElements;
    ULONG NumberOfResolvedSymbols;
    ULONG NumberOfBitmapBytesToZero;
    PCSZ Name;
    FARPROC Proc;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(NumberOfResolvedSymbolsPointer)) {
        return FALSE;
    }

    //
    // Clear the caller's pointer up-front.
    //

    *NumberOfResolvedSymbolsPointer = 0;

    if (!ARGUMENT_PRESENT(SymbolNameArray)) {
        return FALSE;
    }

    if (!ARGUMENT_PRESENT(SymbolAddressArray)) {
        return FALSE;
    }

    if (!ARGUMENT_PRESENT(FailedSymbols)) {
        return FALSE;
    }

    //
    // Initialize NumberOfElements to the NumberOfSymbolNames, then make sure
    // it equals NumberOfSymbolAddresses, and that FailedBitmap->SizeOfBitMap-1
    // is at least greater than or equal to this amount.
    //

    NumberOfElements = NumberOfSymbolNames;

    if (NumberOfElements != NumberOfSymbolAddresses ||
        NumberOfElements > FailedSymbols->SizeOfBitMap-1) {
        return FALSE;
    }

    //
    // Arguments are valid.  Zero the entire failed bitmap buffer (aligned up
    // to an 8-bit boundary, then shifted right three times to convert to the
    // number of bytes consumed by the bitmap).
    //

    NumberOfBitmapBytesToZero = ALIGN_UP(FailedSymbols->SizeOfBitMap, 8);
    NumberOfBitmapBytesToZero >>= 3;
    SecureZeroMemory(FailedSymbols->Buffer, NumberOfBitmapBytesToZero);

    //
    // The most common situation for this routine will be for most if not all
    // symbols resolving.  We default the NumberOfResolvedSymbols to the total
    // number of symbols, then subtract 1 each time a symbol fails.
    //

    NumberOfResolvedSymbols = NumberOfElements;

    for (Index = 0; Index < NumberOfElements; Index++) {

        //
        // Resolve symbol name and look up the symbol via GetProcAddress().
        //

        Name = *(SymbolNameArray + Index);
        Proc = GetProcAddress(Module, Name);

        if (!Proc) {

            //
            // No symbol was resolved.  Set the failed bit corresponding to this
            // symbol.  The + 1 accounts for the fact that the index is 0-based
            // but the bitmap is 1-based.
            //

            FastSetBit(FailedSymbols, Index + 1);

            //
            // Decrement the counter of successfully resolved symbols.
            //

            NumberOfResolvedSymbols--;

        }

        //
        // Suppress spectre warning.
        //

#pragma warning(suppress: 5045)

        //
        // Save the results to the target array (even if Proc was NULL).
        //

        *(SymbolAddressArray + Index) = (ULONG_PTR)Proc;

    }

    //
    // Update the caller's pointer with the count.
    //

    *NumberOfResolvedSymbolsPointer = NumberOfResolvedSymbols;

    return TRUE;
}

//
// Define the maximum number of modules that can be present in the module array
// passed to LoadSymbolsFromMultipleModules().  The NumberOfModules parameter
// will be checked that it does not exceed this value.
//

#define MAX_NUMBER_OF_MODULES 64

LOAD_SYMBOLS_FROM_MULTIPLE_MODULES LoadSymbolsFromMultipleModules;

_Use_decl_annotations_
BOOLEAN
LoadSymbolsFromMultipleModules(
    CONST PCSZ *SymbolNameArray,
    ULONG NumberOfSymbolNames,
    PULONG_PTR SymbolAddressArray,
    ULONG NumberOfSymbolAddresses,
    HMODULE *ModuleArray,
    USHORT NumberOfModules,
    PRTL_BITMAP FailedSymbols,
    PULONG NumberOfResolvedSymbolsPointer
    )
/*++

Routine Description:

    This routine is used to dynamically resolve an array of symbol names
    against an array of modules via GetProcAddress(), storing the resulting
    addresses in symbol address array.  If a symbol name cannot be resolved in
    any of the given modules, its corresponding failure bit is set in the
    failed symbol bitmap, also provided by the caller.

    For the most common case of resolving symbols against a single module,
    use LoadSymbols().

Arguments:

    SymbolNameArray - Supplies a pointer to an array of NULL-terminated C
        strings, with each string representing a name to look up (that is,
        call GetProcAddress() on).

    NumberOfSymbolNames - Supplies the number of elements in the parameter
        SymbolNameArray.  This must match NumberOfSymbolAddresses, otherwise,
        FALSE is returned.

    SymbolAddressArray - Supplies the address of an array of variables that will
        receive the address corresponding to each symbol in the names array, or
        NULL if symbol lookup failed.

    NumberOfSymbolAddresses - Supplies the number of elements in the parameter
        SymbolAddressArray.  This must match NumberOfSymbolNames, otherwise,
        FALSE is returned.

    ModuleArray - Supplies the address to an array of HMODULE variables.  The
        modules in this array are enumerated up to NumberOfModules whilst trying
        to resolve a given symbol name.

    NumberOfModules - Supplies the number of elements in the ModuleArray array.
        The value must be >= 1 and <= MAX_NUMBER_OF_MODULES, otherwise, FALSE
        is returned.

    FailedSymbols - Supplies a pointer to an RTL_BITMAP structure, whose
        SizeOfBitmap field is equal to or greater than NumberOfSymbolNames.
        The entire bitmap buffer will be zeroed up-front and then a bit will
        be set for any failed lookup attempts.

        N.B. As the names and addresses arrays are 0-based, the array index is
             obtained from the failed bit position by subtracting 1.  E.g. if
             the first bit set is 5, that corresponds to Names[4].

    NumberOfResolvedSymbolsPointer - Supplies the address of a variable that
        will receive the number of successfully resolved symbols.

Return Value:

    If all parameters are successfully validated, TRUE will be returned
    (regardless of the results of the actual symbol loading attempts).
    FALSE is returned if any invalid parameters are detected.

    If a symbol cannot be resolved in any of the modules provided, the symbol's
    corresponding bit will be set in the FailedBitmap, and the corresponding
    pointer in the SymbolAddressesArray array will be set to NULL.

--*/
{
    ULONG Index;
    ULONG ModuleIndex;
    ULONG NumberOfElements;
    ULONG NumberOfResolvedSymbols;
    ULONG NumberOfBitmapBytesToZero;
    PCSZ Name;
    FARPROC Proc;
    HMODULE Module;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(NumberOfResolvedSymbolsPointer)) {
        return FALSE;
    }

    //
    // Clear the caller's pointer up-front.
    //

    *NumberOfResolvedSymbolsPointer = 0;

    if (!ARGUMENT_PRESENT(SymbolNameArray)) {
        return FALSE;
    }

    if (!ARGUMENT_PRESENT(SymbolAddressArray)) {
        return FALSE;
    }

    if (!ARGUMENT_PRESENT(ModuleArray)) {
        return FALSE;
    }

    if (!ARGUMENT_PRESENT(FailedSymbols)) {
        return FALSE;
    }

    if (NumberOfModules == 0 || NumberOfModules > MAX_NUMBER_OF_MODULES) {
        return FALSE;
    }

    //
    // Initialize NumberOfElements to the NumberOfSymbolNames, then make sure
    // it equals NumberOfSymbolAddresses, and that FailedBitmap->SizeOfBitMap-1
    // is at least greater than or equal to this amount.
    //

    NumberOfElements = NumberOfSymbolNames;

    if (NumberOfElements != NumberOfSymbolAddresses ||
        NumberOfElements > FailedSymbols->SizeOfBitMap-1) {
        return FALSE;
    }

    //
    // Arguments are valid.  Zero the entire failed bitmap buffer (aligned up
    // to an 8-bit boundary, then shifted right three times to convert to the
    // number of bytes consumed by the bitmap).
    //

    NumberOfBitmapBytesToZero = ALIGN_UP(FailedSymbols->SizeOfBitMap, 8);
    NumberOfBitmapBytesToZero >>= 3;
    SecureZeroMemory(FailedSymbols->Buffer, NumberOfBitmapBytesToZero);

    //
    // The most common situation for this routine will be for most if not all
    // symbols resolving.  We default the NumberOfResolvedSymbols to the total
    // number of symbols, then subtract 1 each time a symbol fails.
    //

    NumberOfResolvedSymbols = NumberOfElements;

    for (Index = 0; Index < NumberOfElements; Index++) {

        //
        // Resolve symbol name and clear the Proc pointer.
        //

        Name = *(SymbolNameArray + Index);
        Proc = NULL;

        //
        // Inner loop over module array.
        //

        for (ModuleIndex = 0; ModuleIndex < NumberOfModules; ModuleIndex++) {

            //
            // Resolve the module.
            //

            //
            // Suppress spectre warning.
            //

#pragma warning(suppress: 5045)

            Module = *(ModuleArray + ModuleIndex);

            //
            // Look up the symbol via GetProcAddress().
            //

            Proc = GetProcAddress(Module, Name);

            //
            // If the symbol wasn't resolved, continue the loop.
            //

            if (!Proc) {
                continue;
            }

            break;
        }

        if (!Proc) {

            //
            // No symbol was resolved.  Set the failed bit corresponding to this
            // symbol.  The + 1 accounts for the fact that the index is 0-based
            // but the bitmap is 1-based.
            //

            FastSetBit(FailedSymbols, Index + 1);

            //
            // Decrement the counter of successfully resolved symbols.
            //

            NumberOfResolvedSymbols--;

        }

        //
        // Save the results to the target array (even if Proc was NULL).
        //

        *(SymbolAddressArray + Index) = (ULONG_PTR)Proc;

    }

    //
    // Update the caller's pointer with the count.
    //

    *NumberOfResolvedSymbolsPointer = NumberOfResolvedSymbols;

    return TRUE;
}

#if 0

BOOL
TestLoadSymbols(VOID)
{
    BOOL Success;

    PSTR Names[] = {
        "RtlNumberOfClearBits",
        "RtlNumberOfSetBits",
        "Dummy",
        "PfxInsertPrefix",
    };

    HMODULE Module = LoadLibraryA("ntdll");

    struct {
        PRTL_NUMBER_OF_CLEAR_BITS RtlNumberOfClearBits;
        PRTL_NUMBER_OF_SET_BITS RtlNumberOfSetBits;
        PVOID Dummy;
        PPFX_INSERT_PREFIX PfxInsertPrefix;
    } Functions;

    ULONG NumberOfResolvedSymbols;
    RTL_BITMAP FailedBitmap;
    ULONG BitmapBuffer = ((ULONG)-1);

    //
    // Wire up the failed bitmap.
    //

    FailedBitmap.SizeOfBitMap = ARRAYSIZE(Names);
    FailedBitmap.Buffer = &BitmapBuffer;

    Success = LoadSymbols(Names,
                          ARRAYSIZE(Names),
                          (PULONG_PTR)&Functions,
                          sizeof(Functions) / sizeof(ULONG_PTR),
                          Module,
                          &FailedBitmap,
                          FALSE,
                          &NumberOfResolvedSymbols);

    if (!Success) {
        __debugbreak();
    }

    if (NumberOfResolvedSymbols != 3) {
        __debugbreak();
        Success = FALSE;
        goto End;
    }

    if (Functions.RtlNumberOfSetBits(&FailedBitmap) != 1) {
        __debugbreak();
        Success = FALSE;
        goto End;
    }

    if (Functions.RtlNumberOfClearBits(&FailedBitmap) != 3) {
        __debugbreak();
        Success = FALSE;
        goto End;
    }

End:
    return Success;
}

BOOL
TestLoadSymbolsFromMultipleModules(VOID)
{
    USHORT Index;
    BOOL Success;

    PSTR Names[] = {
        "RtlNumberOfClearBits",
        "RtlNumberOfSetBits",
        "Dummy",
        "PfxInsertPrefix",
        "bsearch",
        "RtlPrefetchMemoryNonTemporal",
        "RtlEnumerateGenericTableAvl",
    };

    HMODULE Modules[] = {
        LoadLibraryA("kernel32"),
        LoadLibraryA("ntdll"),
        LoadLibraryA("ntoskrnl.exe"),
    };

    struct {
        PRTL_NUMBER_OF_CLEAR_BITS RtlNumberOfClearBits;
        PRTL_NUMBER_OF_SET_BITS RtlNumberOfSetBits;
        PVOID Dummy;
        PPFX_INSERT_PREFIX PfxInsertPrefix;
        PBSEARCH bsearch;
        PRTL_PREFETCH_MEMORY_NON_TEMPORAL RtlPrefetchMemoryNonTemporal;
        PRTL_ENUMERATE_GENERIC_TABLE_AVL RtlEnumerateGenericTableAvl;
    } Functions;

    ULONG NumberOfResolvedSymbols;
    RTL_BITMAP FailedBitmap;
    ULONG BitmapBuffer = ((ULONG)-1);

    //
    // Wire up the failed bitmap.
    //

    FailedBitmap.SizeOfBitMap = ARRAYSIZE(Names);
    FailedBitmap.Buffer = &BitmapBuffer;

    Success = LoadSymbolsFromMultipleModules(
        Names,
        ARRAYSIZE(Names),
        (PULONG_PTR)&Functions,
        sizeof(Functions) / sizeof(ULONG_PTR),
        Modules,
        ARRAYSIZE(Modules),
        &FailedBitmap,
        FALSE,
        &NumberOfResolvedSymbols
    );

    if (!Success) {
        __debugbreak();
    }

    if (NumberOfResolvedSymbols != 6) {
        __debugbreak();
        Success = FALSE;
        goto End;
    }

    if (Functions.RtlNumberOfSetBits(&FailedBitmap) != 1) {
        __debugbreak();
        Success = FALSE;
        goto End;
    }

    if (Functions.RtlNumberOfClearBits(&FailedBitmap) != 6) {
        __debugbreak();
        Success = FALSE;
        goto End;
    }

    for (Index = 0; Index < ARRAYSIZE(Modules); Index++) {
        FreeLibrary(Modules[Index]);
    }

End:
    return Success;
}

#endif

#pragma warning(pop)

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
