/*++

Copyright (c) 2024 Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWorkCppUnityFile.c

Abstract:

    This module implements the prepare and save file work callback routines
    for the C++ Unity source file as part of the CHM v1 algorithm
    implementation for the perfect hash library.  Unity in this context refers
    to an amalgamation of all the C++ source files that make up the compiled
    perfect hash table implementation.

    This module was based on:
        - Chm01FileWorkCHeaderFile.c

    Prepare step:

        - Contents of CompiledPerfectHashTableTypesPre.h.

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

        - Contents of ../../include/CompiledPerfectHashMacroGlue.h.

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
#include "CompiledPerfectHashTableBenchmarkIndexInline_CSource_RawCString.h"

extern const STRING no_sal2CHeaderRawCString;
extern const STRING CompiledPerfectHashCHeaderRawCString;
extern const STRING CompiledPerfectHashMacroGlueCHeaderRawCString;
extern const STRING CompiledPerfectHashTableSupportCHeaderRawCString;
extern const STRING CompiledPerfectHashTableSupportCSourceRawCString;
extern const STRING CompiledPerfectHashTableTypesPreCHeaderRawCString;
extern const STRING CompiledPerfectHashTableRoutinesCSourceRawCString;
extern const STRING CompiledPerfectHashTableTypesPostCHeaderRawCString;
extern const STRING CompiledPerfectHashTableRoutinesPreCSourceRawCString;
extern const STRING CompiledPerfectHashTableRoutinesPostCSourceRawCString;
extern const STRING CompiledPerfectHashTableBenchmarkIndexExeCSourceRawCString;

_Use_decl_annotations_
HRESULT
PrepareCppSourceUnityFileChm01(
    PPERFECT_HASH_CONTEXT Context,
    PFILE_WORK_ITEM Item
    )
{
    PRTL Rtl;
    PCHAR Base;
    PCHAR Output;
    ULONG Count;
    PULONG Long;
    ULONG NumberOfSeeds;
    ULONGLONG Index;
    ULONGLONG NumberOfKeys;
    PCSTRING Name;
    PCSTRING Upper;
    PPERFECT_HASH_KEYS Keys;
    PPERFECT_HASH_PATH Path;
    PPERFECT_HASH_FILE File;
    PPERFECT_HASH_TABLE Table;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;
    ULONGLONG NumberOfElements;
    ULONGLONG TotalNumberOfElements;
    HRESULT Result = S_OK;
    const ULONG Indent = 0x20202020;

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
    NumberOfKeys = Keys->NumberOfKeys.QuadPart;
    TableInfoOnDisk = Table->TableInfoOnDisk;
    NumberOfSeeds = TableInfoOnDisk->NumberOfSeeds;
    TotalNumberOfElements = TableInfoOnDisk->NumberOfTableElements.QuadPart;
    NumberOfElements = TotalNumberOfElements >> 1;

    Base = (PCHAR)File->BaseAddress;
    Output = Base;

    //
    // Write the keys.
    //

    OUTPUT_RAW("//\n// Compiled Perfect Hash Table C++ Unity File.  "
               "Auto-generated.\n"
               "// https://github.com/tpn/perfecthash\n"
               "//\n// Command line:\n"
               "// ");

    OUTPUT_WSTR_FAST(Context->CommandLineW);

    OUTPUT_RAW("\n//\n\n");

    OUTPUT_RAW("#define PH_UNITY\n");
    OUTPUT_RAW("#ifndef _WIN32\n");

    //
    // Include raw no_sal2.h
    //

    OUTPUT_STRING(&no_sal2CHeaderRawCString);

    OUTPUT_RAW("#endif\n");

    //
    // OUTPUT_INCLUDE_STDAFX_H():
    //    #include "stdafx.h":
    //      _Types.h
    //          PreTypes
    //          Raw names
    //          PostTypes
    //      <CompiledPerfectHash.h>
    //      C Header file.
    //


    if (IsIndexOnly(Table)) {
        OUTPUT_RAW("#define CPH_INDEX_ONLY 1\n\n");
    }

    OUTPUT_STRING(&CompiledPerfectHashTableTypesPreCHeaderRawCString);

    //
    // Write the CPHKEY, CPHDKEY, CPHVALUE, and CPHINDEX types.
    //

    OUTPUT_RAW("typedef ");
    OUTPUT_STRING(Table->OriginalKeySizeTypeName);
    OUTPUT_RAW(" CPHKEY;\n");

    OUTPUT_RAW("typedef ");
    OUTPUT_STRING(Table->KeySizeTypeName);
    OUTPUT_RAW(" CPHDKEY;\n");

    OUTPUT_RAW("typedef ");
    OUTPUT_STRING(Table->ValueTypeName);
    OUTPUT_RAW(" CPHVALUE;\n");

    OUTPUT_RAW("typedef ");
    OUTPUT_STRING(Table->SeedTypeName);
    OUTPUT_RAW(" CPHSEED;\n");

    OUTPUT_RAW("typedef ");
    OUTPUT_STRING(Table->IndexTypeName);
    OUTPUT_RAW(" CPHINDEX;\n\n");

    //
    // Write the explicit 32-bit and 64-bit versions of above.
    //

    OUTPUT_RAW("typedef unsigned int CPHKEY32;\n");
    OUTPUT_RAW("typedef unsigned int CPHDKEY32;\n");
    OUTPUT_RAW("typedef unsigned int CPHVALUE32;\n");
    OUTPUT_RAW("typedef unsigned int CPHINDEX32;\n");
    OUTPUT_RAW("typedef unsigned int CPHSEED32;\n");

    OUTPUT_RAW("typedef unsigned long long CPHKEY64;\n");
    OUTPUT_RAW("typedef unsigned long long CPHDKEY64;\n");
    OUTPUT_RAW("typedef unsigned long long CPHVALUE64;\n");
    OUTPUT_RAW("typedef unsigned long long CPHINDEX64;\n");
    OUTPUT_RAW("typedef unsigned long long CPHSEED64;\n");

    //
    // Write the post glue.
    //

    OUTPUT_STRING(&CompiledPerfectHashTableTypesPostCHeaderRawCString);

    //
    // Write the CompiledPerfectHash.h contents.
    //

    OUTPUT_STRING(&CompiledPerfectHashCHeaderRawCString);

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

    OUTPUT_RAW("\n\n");

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
        OUTPUT_RAW("_DOWNSIZE_KEY(Key) ((CPHDKEY)ExtractBits64(Key, 0x");
        OUTPUT_HEX64_RAW(Keys->DownsizeBitmap);
        OUTPUT_RAW("))\n");

        //
        // Write the left and right key rotation macros.
        //

        OUTPUT_RAW("#define ");
        OUTPUT_STRING(Upper);
        OUTPUT_RAW("_ROTATE_KEY_LEFT RotateLeft64\n");

        OUTPUT_RAW("#define ");
        OUTPUT_STRING(Upper);
        OUTPUT_RAW("_ROTATE_KEY_RIGHT RotateRight64\n\n");

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
        OUTPUT_RAW("_ROTATE_KEY_LEFT RotateLeft32\n");

        OUTPUT_RAW("#define ");
        OUTPUT_STRING(Upper);
        OUTPUT_RAW("_ROTATE_KEY_RIGHT RotateRight32\n\n");
    }

    OUTPUT_STRING(&CompiledPerfectHashMacroGlueCHeaderRawCString);

    //
    // Table values.
    //

    OUTPUT_RAW("#ifndef CPH_INDEX_ONLY\n\n");

    OUTPUT_RAW("static constexpr uint32_t ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_TableValueSizeInBytes = ");
    OUTPUT_INT(Table->ValueSizeInBytes == 0 ?
               sizeof(ULONG) : Table->ValueSizeInBytes);
    OUTPUT_RAW(";\n");

    OUTPUT_RAW("static constexpr uint32_t ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_NumberOfTableValues = ");
    OUTPUT_INT(NumberOfElements);
    OUTPUT_RAW(";\n\n");

    OUTPUT_RAW("#ifdef _WIN32\n"
               "#pragma data_seg(\".cphval\")\n"
               "#endif\n");

    OUTPUT_RAW("CPHVALUE ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_TableValues[");
    OUTPUT_INT(NumberOfElements);
    OUTPUT_RAW("] = { 0, };\n"
               "#ifdef _WIN32\n"
               "#pragma data_seg()\n"
               "#pragma comment(linker, "
               "\"/section:.cphval,rw");
    if (UseRwsSectionForTableValues(Table)) {
        *Output++ = 's';
    }
    OUTPUT_RAW("\")\n#endif\n#endif\n\n");

    //
    // Write the keys.
    //

    OUTPUT_RAW("#ifdef _WIN32\n#pragma const_seg(\".cphkeys\")\n#endif\n");

    OUTPUT_RAW("const ULONG ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_NumberOfKeys = ");
    OUTPUT_INT(NumberOfKeys);
    OUTPUT_RAW(";\n");

    OUTPUT_RAW("const ULONG ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_KeySizeInBytes = ");
    OUTPUT_INT(Keys->OriginalKeySizeInBytes);
    OUTPUT_RAW(";\n");

    OUTPUT_RAW("const ULONG ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_OriginalKeySizeInBytes = ");
    OUTPUT_INT(Keys->OriginalKeySizeInBytes);
    OUTPUT_RAW(";\n");

    OUTPUT_RAW("const ULONG ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_DownsizedKeySizeInBytes = ");
    OUTPUT_INT(4);
    OUTPUT_RAW(";\n");

    OUTPUT_RAW("const CPHKEY ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_Keys[");
    OUTPUT_INT(NumberOfKeys);
    OUTPUT_RAW("] = {\n");

    if (Keys->OriginalKeySizeType == LongType) {

        ULONG Key;
        PULONG SourceKeys;

        SourceKeys = (PULONG)Keys->KeyArrayBaseAddress;

        for (Index = 0, Count = 0; Index < NumberOfKeys; Index++) {

            if (Count == 0) {
                INDENT();
            }

            Key = *SourceKeys++;

            OUTPUT_HEX(Key);

            *Output++ = ',';

            if (++Count == 4) {
                Count = 0;
                *Output++ = '\n';
            } else {
                *Output++ = ' ';
            }
        }

    } else if (Keys->OriginalKeySizeType == LongLongType) {

        ULONGLONG Key;
        PULONGLONG SourceKeys;

        SourceKeys = (PULONGLONG)Keys->File->BaseAddress;

        for (Index = 0, Count = 0; Index < NumberOfKeys; Index++) {

            if (Count == 0) {
                INDENT();
            }

            Key = *SourceKeys++;

            OUTPUT_HEX64(Key);

            *Output++ = ',';

            if (++Count == 4) {
                Count = 0;
                *Output++ = '\n';
            } else {
                *Output++ = ' ';
            }
        }

    } else {

        Result = PH_E_UNREACHABLE_CODE;
        PH_ERROR(PrepareCSourceKeysFileChm01_UnknownKeyType, Result);
        PH_RAISE(Result);

    }

    //
    // If the last character written was a trailing space, replace
    // it with a newline.
    //

    if (*(Output - 1) == ' ') {
        *(Output - 1) = '\n';
    }

    OUTPUT_RAW("};\n#ifdef _WIN32\n#pragma const_seg()\n#endif\n");


    OUTPUT_RAW("\n\n//\n// (End of preparation phase.)\n//\n\n");

    File->NumberOfBytesWritten.QuadPart = RtlPointerToOffset(Base, Output);

    return S_OK;
}

_Use_decl_annotations_
HRESULT
SaveCppSourceUnityFileChm01(
    PPERFECT_HASH_CONTEXT Context,
    PFILE_WORK_ITEM Item
    )
{
    PRTL Rtl;
    PCHAR Base;
    PCHAR Output;
    ULONG Count;
    PULONG Long;
    PULONG Seeds;
    PGRAPH Graph;
    ULONG Value;
    USHORT Value16;
    PULONG Source;
    PUSHORT Source16;
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
    const ULONG Indent = 0x20202020;

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
    Source16 = Graph->Assigned16;

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
    // Write table data.
    //

    OUTPUT_RAW("#ifdef _WIN32\n#pragma const_seg(\".cphdata\")\n#endif\n");

    OUTPUT_RAW("static constexpr const ");
    OUTPUT_STRING(Table->TableDataArrayTypeName);
    OUTPUT_RAW(" ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_TableData[");
    OUTPUT_INT(TotalNumberOfElements);
    OUTPUT_RAW("] = {\n\n    //\n    // 1st half.\n    //\n\n");

    if (!IsUsingAssigned16(Graph)) {

        for (Index = 0, Count = 0; Index < TotalNumberOfElements; Index++) {

            if (Count == 0) {
                INDENT();
            }

            Value = *Source++;

            OUTPUT_HEX(Value);

            *Output++ = ',';

            if (++Count == 4) {
                Count = 0;
                *Output++ = '\n';
            } else {
                *Output++ = ' ';
            }

            if (Index == NumberOfElements-1) {
                OUTPUT_RAW("\n    //\n    // 2nd half.\n    //\n\n");
            }
        }

    } else {

        for (Index = 0, Count = 0; Index < TotalNumberOfElements; Index++) {

            if (Count == 0) {
                INDENT();
            }

            Value16 = *Source16++;

            OUTPUT_HEX(Value16);

            *Output++ = ',';

            if (++Count == 4) {
                Count = 0;
                *Output++ = '\n';
            } else {
                *Output++ = ' ';
            }

            if (Index == NumberOfElements-1) {
                OUTPUT_RAW("\n    //\n    // 2nd half.\n    //\n\n");
            }
        }

    }

    //
    // If the last character written was a trailing space, replace
    // it with a newline.
    //

    if (*(Output - 1) == ' ') {
        *(Output - 1) = '\n';
    }

    OUTPUT_RAW("};\n");


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

    OUTPUT_STRING(&CompiledPerfectHashTableSupportCHeaderRawCString);
    OUTPUT_STRING(&CompiledPerfectHashTableSupportCSourceRawCString);

    OUTPUT_STRING(&CompiledPerfectHashTableBenchmarkIndexInlineCSourceRawCString);

    OUTPUT_STRING(&CompiledPerfectHashTableBenchmarkIndexExeCSourceRawCString);

    //
    // Update number of bytes written.
    //

    Base = (PCHAR)File->BaseAddress;
    File->NumberOfBytesWritten.QuadPart = RtlPointerToOffset(Base, Output);

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
