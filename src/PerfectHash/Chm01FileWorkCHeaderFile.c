/*++

Copyright (c) 2018-2023 Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWorkCHeaderFile.c

Abstract:

    This module implements the prepare and save file work callback routines for
    the C header file as part of the CHM v1 algorithm implementation for the
    perfect hash library.

    Example C header file (e.g. KernelBase_2415_8196_Chm01_Crc32Rotate_And.h),
    where T## -> table name, U## -> uppercase table name:

    Prepare step:

        - Table-specific macros and constants:
            - CPH_TABLENAME
            - CPH_TABLENAME_UPPER
            - U##_NUMBER_OF_KEYS
            - U##_SEEDS
            - U##_SEED[1..N]_BYTE[1..4]
            - U##_SEED[12|23|34|45|56|67]
            - T##_Seeds[]
            - T##_Seed1 .. SeedN
            - T##_HashMask
            - T##_IndexMask
            - T##_TableData
            - T##_TableValues
            - T##_NumberOfKeys

        - #include <CompiledPerfectHashMacroGlue.h>

    Save step:

        - Table-specific values (e.g. literal hash mask, index mask etc).

        - #define CPH_INLINE_ROUTINES

        - Contents of ../CompiledPerfectHashTableRoutinesPre.c

        - Contents of specific Index() implementation for the given algo,
          hash and mask (as specified by Table->IndexImplString). E.g. the
          contents of ../CompiledPerfectHashTableChm01IndexCrc32RotateAnd.c.

        - Contents of ../CompiledPerfectHashTableRoutines.c

        - Contents of ../CompiledPerfectHashTableRoutinesPost.c

        - DEFINE_TABLE_ROUTINES();
          DEFINE_TEST_AND_BENCHMARK_ROUTINES();

--*/

#include "stdafx.h"

extern const STRING CompiledPerfectHashTableRoutinesPreCSourceRawCString;
extern const STRING CompiledPerfectHashTableRoutinesCSourceRawCString;
extern const STRING CompiledPerfectHashTableRoutinesPostCSourceRawCString;

_Use_decl_annotations_
HRESULT
PrepareCHeaderFileChm01(
    PPERFECT_HASH_CONTEXT Context,
    PFILE_WORK_ITEM Item
    )
{
    PRTL Rtl;
    PCHAR Base;
    PCHAR Output;
    ULONG Count;
    ULONG NumberOfSeeds;
    ULONGLONG Index;
    PCSTRING Name;
    PCSTRING Upper;
    PPERFECT_HASH_KEYS Keys;
    PPERFECT_HASH_PATH Path;
    PPERFECT_HASH_FILE File;
    PPERFECT_HASH_TABLE Table;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Table = Context->Table;
    Keys = Table->Keys;
    File = *Item->FilePointer;
    Path = GetActivePath(File);
    Name = &Path->TableNameA;
    Upper = &Path->TableNameUpperA;
    TableInfoOnDisk = Table->TableInfoOnDisk;
    NumberOfSeeds = TableInfoOnDisk->NumberOfSeeds;

    Base = (PCHAR)File->BaseAddress;
    Output = Base;

    //
    // Write the keys.
    //

    OUTPUT_RAW("//\n// Compiled Perfect Hash Table C Header File.  "
               "Auto-generated.\n//\n\n"
               "#pragma once\n\n");

    OUTPUT_OPEN_EXTERN_C_SCOPE();

    OUTPUT_INCLUDE_STDAFX_H();

    OUTPUT_RAW("#ifndef CPH_TABLENAME\n"
               "#define CPH_TABLENAME ");
    OUTPUT_STRING(Name);

    OUTPUT_RAW("\n#endif\n\n"
               "#ifndef CPH_TABLENAME_UPPER\n"
               "#define CPH_TABLENAME_UPPER ");
    OUTPUT_STRING(Upper);

    OUTPUT_RAW("\n#endif\n\n#define ");
    OUTPUT_STRING(Upper);
    OUTPUT_RAW("_NUMBER_OF_KEYS ");
    OUTPUT_INT(Keys->NumberOfKeys.QuadPart);

    OUTPUT_RAW("\n\nextern const CPHSEED ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_Seeds[");
    OUTPUT_INT(NumberOfSeeds);
    OUTPUT_RAW("];\n");

    OUTPUT_RAW("\n\nextern const BYTE ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_NumberOfSeeds;\n");

    for (Index = 0, Count = 1; Index < NumberOfSeeds; Index++, Count++) {
        OUTPUT_RAW("extern const CPHSEED ");
        OUTPUT_STRING(Name);
        OUTPUT_RAW("_Seed");
        OUTPUT_INT(Count);
        OUTPUT_RAW(";\n");
    }

    if (NumberOfSeeds >= 2) {
        OUTPUT_RAW("extern const CPHSEED64 ");
        OUTPUT_STRING(Name);
        OUTPUT_RAW("_Seed12");
        OUTPUT_RAW(";\n");
    }

    if (NumberOfSeeds >= 5) {
        OUTPUT_RAW("extern const CPHSEED64 ");
        OUTPUT_STRING(Name);
        OUTPUT_RAW("_Seed34");
        OUTPUT_RAW(";\n");
    }

    OUTPUT_RAW("\nextern const CPHDKEY ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_HashMask;\n");

    OUTPUT_RAW("extern const CPHDKEY ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_IndexMask;\n\n");

    OUTPUT_RAW("extern const ");
    OUTPUT_STRING(Table->TableDataArrayTypeName);
    OUTPUT_RAW(" ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_TableData[];\n");

    OUTPUT_RAW("extern CPHVALUE ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_TableValues[];\n");

    OUTPUT_RAW("extern const ULONG ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_TableValueSizeInBytes;\n");

    OUTPUT_RAW("extern const ULONG ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_NumberOfTableValues;\n\n");

    OUTPUT_RAW("extern const CPHKEY ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_Keys[];\n");

    OUTPUT_RAW("extern const CPHDKEY ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_DownsizedKeys[];\n");

    OUTPUT_RAW("extern const ULONG ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_NumberOfKeys;\n");

    OUTPUT_RAW("extern const ULONG ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_KeySizeInBytes;\n");

    OUTPUT_RAW("extern const ULONG ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_OriginalKeySizeInBytes;\n");

    OUTPUT_RAW("extern const ULONG ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_DownsizedKeySizeInBytes;\n\n");

    if (Table->TableCreateFlags.IncludeKeysInCompiledDll != FALSE) {
        OUTPUT_RAW("#define CPH_HAS_KEYS 1\n\n");
    }

    //
    // If key downsizing has occurred, output the bitmap that was used.
    //

    if (KeysWereDownsized(Keys)) {

        OUTPUT_RAW("#define ");
        OUTPUT_STRING(Upper);
        OUTPUT_RAW("_KEY_DOWNSIZE_BITMAP 0x");
        OUTPUT_HEX64_RAW(Keys->DownsizeBitmap);
        OUTPUT_RAW("\n#define ");
        OUTPUT_STRING(Upper);
        OUTPUT_RAW("_DOWNSIZE_KEY(Key) ((CPHDKEY)_pext_u64(Key, 0x");
        OUTPUT_HEX64_RAW(Keys->DownsizeBitmap);
        OUTPUT_RAW("))\n");

        //
        // Write the left and right key rotation macros.
        //

        OUTPUT_RAW("#define ");
        OUTPUT_STRING(Upper);
        OUTPUT_RAW("_ROTATE_KEY_LEFT _rotl64\n");

        OUTPUT_RAW("#define ");
        OUTPUT_STRING(Upper);
        OUTPUT_RAW("_ROTATE_KEY_RIGHT _rotr64\n\n");

    } else {

        //
        // No downsizing occurred; output a dummy key downsize macro.
        //

        OUTPUT_RAW("#define ");
        OUTPUT_STRING(Upper);
        OUTPUT_RAW("_DOWNSIZE_KEY(Key) (Key)\n");

        //
        // Write the left and right key rotation macros.
        //

        OUTPUT_RAW("#define ");
        OUTPUT_STRING(Upper);
        OUTPUT_RAW("_ROTATE_KEY_LEFT _rotl\n");

        OUTPUT_RAW("#define ");
        OUTPUT_STRING(Upper);
        OUTPUT_RAW("_ROTATE_KEY_RIGHT _rotr\n\n");
    }

    OUTPUT_RAW("#include <CompiledPerfectHashMacroGlue.h>\n");

    OUTPUT_RAW("\n\n//\n// (End of preparation phase.)\n//\n\n");

    File->NumberOfBytesWritten.QuadPart = RtlPointerToOffset(Base, Output);

    return S_OK;
}

_Use_decl_annotations_
HRESULT
SaveCHeaderFileChm01(
    PPERFECT_HASH_CONTEXT Context,
    PFILE_WORK_ITEM Item
    )
{
    PRTL Rtl;
    PCHAR Base;
    PCHAR Output;
    ULONG Count;
    PULONG Seeds;
    PGRAPH Graph;
    PULONG Source;
    ULONG NumberOfSeeds;
    ULONG_BYTES Seed;
    PCSTRING Name;
    PCSTRING Upper;
    STRING Algo = { 0 };
    ULONGLONG Index;
    HRESULT Result = S_OK;
    PPERFECT_HASH_FILE File;
    PPERFECT_HASH_PATH Path;
    PPERFECT_HASH_TABLE Table;
    PTABLE_INFO_ON_DISK TableInfo;
    ULONGLONG NumberOfElements;
    ULONGLONG TotalNumberOfElements;

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Table = Context->Table;
    File = *Item->FilePointer;
    Path = GetActivePath(File);
    Name = &Path->TableNameA;
    Upper = &Path->TableNameUpperA;
    TableInfo = Table->TableInfoOnDisk;
    TotalNumberOfElements = TableInfo->NumberOfTableElements.QuadPart;
    NumberOfElements = TotalNumberOfElements >> 1;
    Graph = (PGRAPH)Context->SolvedContext;
    NumberOfSeeds = Graph->NumberOfSeeds;
    Source = Graph->Assigned;

    Algo.Buffer = (PSTR)(
        RtlOffsetToPointer(
            Path->TableNameUpperA.Buffer,
            Path->AdditionalSuffixAOffset
        )
    );

    Algo.Length = (
        Path->TableNameUpperA.Length -
        (USHORT)RtlPointerToOffset(Path->TableNameUpperA.Buffer, Algo.Buffer)
    );
    Algo.MaximumLength = Algo.Length;

    //
    // Pick up the offset from where we left off.
    //

    ASSERT(File->NumberOfBytesWritten.QuadPart > 0);

    Output = (PSTR)(
        RtlOffsetToPointer(
            File->BaseAddress,
            File->NumberOfBytesWritten.QuadPart
        )
    );

    //
    // Write seeds.
    //

    Seeds = &Graph->FirstSeed;

    for (Index = 0, Count = 1; Index < NumberOfSeeds; Index++, Count++) {

        //
        // Resolve the seed, then write the value in full.
        //

        Seed.AsULong = *Seeds++;
        OUTPUT_RAW("#define ");
        OUTPUT_STRING(Upper);
        OUTPUT_RAW("_SEED");
        OUTPUT_INT(Count);
        *Output++ = ' ';
        *Output++ = '0';
        *Output++ = 'x';
        OUTPUT_HEX_RAW(Seed.AsULong);
        *Output++ = '\n';

        //
        // Isolate each byte.
        //

#define WRITE_SEED_BYTE(ByteNumber)              \
        OUTPUT_RAW("#define ");                  \
        OUTPUT_STRING(Upper);                    \
        OUTPUT_RAW("_SEED");                     \
        OUTPUT_INT(Count);                       \
        OUTPUT_RAW("_BYTE" # ByteNumber);        \
        *Output++ = ' ';                         \
        *Output++ = '0';                         \
        *Output++ = 'x';                         \
        OUTPUT_HEX_RAW(Seed.Byte ## ByteNumber); \
        *Output++ = '\n';

        WRITE_SEED_BYTE(1);
        WRITE_SEED_BYTE(2);
        WRITE_SEED_BYTE(3);
        WRITE_SEED_BYTE(4);

        //
        // Isolate each word.
        //

#define WRITE_SEED_WORD(WordNumber)               \
        OUTPUT_RAW("#define ");                   \
        OUTPUT_STRING(Upper);                     \
        OUTPUT_RAW("_SEED");                      \
        OUTPUT_INT(Count);                        \
        OUTPUT_RAW("_WORD" # WordNumber);         \
        *Output++ = ' ';                          \
        *Output++ = '0';                          \
        *Output++ = 'x';                          \
        OUTPUT_HEX_RAW(Seed.UWord ## WordNumber); \
        *Output++ = '\n';

        WRITE_SEED_WORD(1);
        WRITE_SEED_WORD(2);
    }

    if (NumberOfSeeds >= 2) {
        OUTPUT_RAW("#define ");
        OUTPUT_STRING(Upper);
        OUTPUT_RAW("_SEED12 ");
        *Output++ = '0';
        *Output++ = 'x';
        OUTPUT_HEX_RAW(Graph->Seeds[1]);
        OUTPUT_HEX_RAW(Graph->Seeds[0]);
        *Output++ = '\n';
    }

    if (NumberOfSeeds >= 5) {
        OUTPUT_RAW("#define ");
        OUTPUT_STRING(Upper);
        OUTPUT_RAW("_SEED45 ");
        *Output++ = '0';
        *Output++ = 'x';
        OUTPUT_HEX_RAW(Graph->Seeds[4]);
        OUTPUT_HEX_RAW(Graph->Seeds[3]);
        *Output++ = '\n';
    }

    //
    // Write masks.
    //

    OUTPUT_RAW("\n#define ");
    OUTPUT_STRING(Upper);
    OUTPUT_RAW("_HASH_MASK 0x");
    OUTPUT_HEX_RAW(TableInfo->HashMask);
    OUTPUT_RAW("\n#define ");
    OUTPUT_STRING(Upper);
    OUTPUT_RAW("_INDEX_MASK 0x");
    OUTPUT_HEX_RAW(TableInfo->IndexMask);
    OUTPUT_RAW("\n\n");

    //
    // Write inline routines.
    //

    OUTPUT_PRAGMA_WARNING_DISABLE_UNREFERENCED_INLINE();

    OUTPUT_RAW("#define CPH_INLINE_ROUTINES\n");

    OUTPUT_STRING(&CompiledPerfectHashTableRoutinesPreCSourceRawCString);

    OUTPUT_STRING(Table->IndexImplString);

    OUTPUT_STRING(&CompiledPerfectHashTableRoutinesCSourceRawCString);

    OUTPUT_PRAGMA_WARNING_POP();

    OUTPUT_STRING(&CompiledPerfectHashTableRoutinesPostCSourceRawCString);

    OUTPUT_RAW("\nDEFINE_TABLE_ROUTINES();\n"
               "\nDEFINE_TEST_AND_BENCHMARKING_ROUTINES();\n\n");

    //
    // Close the extern C scope opened up during the prepare phase, then close
    // the underlying file.
    //

    OUTPUT_CLOSE_EXTERN_C_SCOPE();

    //
    // Update number of bytes written.
    //

    Base = (PCHAR)File->BaseAddress;
    File->NumberOfBytesWritten.QuadPart = RtlPointerToOffset(Base, Output);

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
