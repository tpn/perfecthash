/*++

Copyright (c) 2026 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashOnlineJit.c

Abstract:

    Minimal C API wrapper for creating and querying 32-bit perfect hash
    tables with online mode and either RawDog JIT or LLVM JIT backends.

--*/

#include "stdafx.h"
#include <PerfectHash/PerfectHashOnlineJit.h>

#include <stdint.h>
#include <stdlib.h>

extern DLL_GET_CLASS_OBJECT PerfectHashDllGetClassObject;

struct PH_ONLINE_JIT_CONTEXT {
    PICLASSFACTORY ClassFactory;
    PPERFECT_HASH_ONLINE Online;
};

struct PH_ONLINE_JIT_TABLE {
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_JIT_INTERFACE JitInterface;
};

static
HRESULT
PhReleaseOnlineJitInterface(
    _Inout_ PH_ONLINE_JIT_TABLE *Table
    )
{
    if (!ARGUMENT_PRESENT(Table)) {
        return E_INVALIDARG;
    }

    if (Table->JitInterface) {
        Table->JitInterface->Vtbl->Release(Table->JitInterface);
        Table->JitInterface = NULL;
    }

    return S_OK;
}

static
HRESULT
PhEnsureOnlineJitInterface(
    _Inout_ PH_ONLINE_JIT_TABLE *Table
    )
{
    HRESULT Result;

    if (!ARGUMENT_PRESENT(Table) ||
        !ARGUMENT_PRESENT(Table->Table)) {
        return E_INVALIDARG;
    }

    if (ARGUMENT_PRESENT(Table->JitInterface)) {
        return S_OK;
    }

    Result = Table->Table->Vtbl->QueryInterface(
        Table->Table,
#ifdef __cplusplus
        IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
        &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
        (void **)&Table->JitInterface
    );

    return Result;
}

static
HRESULT
PhMapOnlineJitHashFunction(
    _In_ PH_ONLINE_JIT_HASH_FUNCTION HashFunction,
    _Out_ PPERFECT_HASH_HASH_FUNCTION_ID HashFunctionId
    )
{
    if (!ARGUMENT_PRESENT(HashFunctionId)) {
        return E_POINTER;
    }

    switch (HashFunction) {
        case PhOnlineJitHashMultiplyShiftR:
            *HashFunctionId = PerfectHashHashMultiplyShiftRFunctionId;
            break;
        case PhOnlineJitHashMultiplyShiftLR:
            *HashFunctionId = PerfectHashHashMultiplyShiftLRFunctionId;
            break;
        case PhOnlineJitHashMultiplyShiftRMultiply:
            *HashFunctionId = PerfectHashHashMultiplyShiftRMultiplyFunctionId;
            break;
        case PhOnlineJitHashMultiplyShiftR2:
            *HashFunctionId = PerfectHashHashMultiplyShiftR2FunctionId;
            break;
        case PhOnlineJitHashMultiplyShiftRX:
            *HashFunctionId = PerfectHashHashMultiplyShiftRXFunctionId;
            break;
        case PhOnlineJitHashMulshrolate1RX:
            *HashFunctionId = PerfectHashHashMulshrolate1RXFunctionId;
            break;
        case PhOnlineJitHashMulshrolate2RX:
            *HashFunctionId = PerfectHashHashMulshrolate2RXFunctionId;
            break;
        case PhOnlineJitHashMulshrolate3RX:
            *HashFunctionId = PerfectHashHashMulshrolate3RXFunctionId;
            break;
        case PhOnlineJitHashMulshrolate4RX:
            *HashFunctionId = PerfectHashHashMulshrolate4RXFunctionId;
            break;
        default:
            return E_INVALIDARG;
    }

    return S_OK;
}

static
HRESULT
PhMapPerfectHashHashFunctionToOnlineJit(
    _In_ PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    _Out_ PH_ONLINE_JIT_HASH_FUNCTION *HashFunction
    )
{
    if (!ARGUMENT_PRESENT(HashFunction)) {
        return E_POINTER;
    }

    if (HashFunctionId == PerfectHashHashMultiplyShiftRFunctionId) {
        *HashFunction = PhOnlineJitHashMultiplyShiftR;
    } else if (HashFunctionId == PerfectHashHashMultiplyShiftLRFunctionId) {
        *HashFunction = PhOnlineJitHashMultiplyShiftLR;
    } else if (HashFunctionId == PerfectHashHashMultiplyShiftRMultiplyFunctionId) {
        *HashFunction = PhOnlineJitHashMultiplyShiftRMultiply;
    } else if (HashFunctionId == PerfectHashHashMultiplyShiftR2FunctionId) {
        *HashFunction = PhOnlineJitHashMultiplyShiftR2;
    } else if (HashFunctionId == PerfectHashHashMultiplyShiftRXFunctionId) {
        *HashFunction = PhOnlineJitHashMultiplyShiftRX;
    } else if (HashFunctionId == PerfectHashHashMulshrolate1RXFunctionId) {
        *HashFunction = PhOnlineJitHashMulshrolate1RX;
    } else if (HashFunctionId == PerfectHashHashMulshrolate2RXFunctionId) {
        *HashFunction = PhOnlineJitHashMulshrolate2RX;
    } else if (HashFunctionId == PerfectHashHashMulshrolate3RXFunctionId) {
        *HashFunction = PhOnlineJitHashMulshrolate3RX;
    } else if (HashFunctionId == PerfectHashHashMulshrolate4RXFunctionId) {
        *HashFunction = PhOnlineJitHashMulshrolate4RX;
    } else {
        return E_INVALIDARG;
    }

    return S_OK;
}

static
HRESULT
PhApplyOnlineJitVectorWidth(
    _In_ ULONG VectorWidth,
    _Inout_ PPERFECT_HASH_TABLE_COMPILE_FLAGS CompileFlags
    )
{
    if (!ARGUMENT_PRESENT(CompileFlags)) {
        return E_POINTER;
    }

    switch (VectorWidth) {
        case 0:
        case 1:
            break;
        case 2:
            CompileFlags->JitIndex32x2 = TRUE;
            break;
        case 4:
            CompileFlags->JitIndex32x4 = TRUE;
            break;
        case 8:
            CompileFlags->JitIndex32x8 = TRUE;
            break;
        case 16:
            CompileFlags->JitIndex32x16 = TRUE;
            break;
        default:
            return E_INVALIDARG;
    }

    return S_OK;
}

static
HRESULT
PhCompileOnlineJitBackend(
    _In_ PH_ONLINE_JIT_CONTEXT *Context,
    _In_ PH_ONLINE_JIT_TABLE *Table,
    _In_ PH_ONLINE_JIT_BACKEND Backend,
    _In_ ULONG VectorWidth,
    _In_ PH_ONLINE_JIT_MAX_ISA JitMaxIsa
    )
{
    HRESULT Result;
    PERFECT_HASH_TABLE_COMPILE_FLAGS CompileFlags = {0};
    PTABLE_INFO_ON_DISK TableInfo;

    if (!ARGUMENT_PRESENT(Context) ||
        !ARGUMENT_PRESENT(Table) ||
        !ARGUMENT_PRESENT(Table->Table) ||
        !ARGUMENT_PRESENT(Table->Table->TableInfoOnDisk)) {
        return E_INVALIDARG;
    }

    TableInfo = Table->Table->TableInfoOnDisk;

    CompileFlags.Jit = TRUE;

    switch (Backend) {
        case PhOnlineJitBackendRawDogJit:
            CompileFlags.JitBackendRawDog = TRUE;
            Result = PhApplyOnlineJitVectorWidth(VectorWidth, &CompileFlags);
            if (FAILED(Result)) {
                return Result;
            }
            break;

        case PhOnlineJitBackendLlvmJit:
            CompileFlags.JitBackendLlvm = TRUE;
            Result = PhApplyOnlineJitVectorWidth(VectorWidth,
                                                 &CompileFlags);
            if (FAILED(Result)) {
                return Result;
            }
            break;

        case PhOnlineJitBackendAuto:
            return E_INVALIDARG;

        default:
            return E_INVALIDARG;
    }

    CompileFlags.JitMaxIsa = (ULONG)JitMaxIsa;

    return Context->Online->Vtbl->CompileTable(Context->Online,
                                               Table->Table,
                                               &CompileFlags);
}

static
HRESULT
PhEstimateCudaSourceBytes(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ ULONG Flags,
    _Out_ PULONGLONG AllocationSizePointer
    )
{
    PTABLE_INFO_ON_DISK TableInfo;
    ULONGLONG NumberOfTableElements;
    ULONGLONG TableElementBytes;
    ULONGLONG ValuesPerLine;
    ULONGLONG CharsPerValue;
    ULONGLONG NumberOfLines;
    ULONGLONG BaseSize;
    ULONGLONG RequiredSize;

    if (!ARGUMENT_PRESENT(Table) ||
        !ARGUMENT_PRESENT(Table->TableInfoOnDisk) ||
        !ARGUMENT_PRESENT(AllocationSizePointer)) {
        return E_POINTER;
    }

    TableInfo = Table->TableInfoOnDisk;

    NumberOfTableElements = TableInfo->NumberOfTableElements.QuadPart;

    BaseSize = (Table->Context && Table->Context->SystemAllocationGranularity)
        ? Table->Context->SystemAllocationGranularity
        : 65536;
    if ((Flags & PH_ONLINE_JIT_CUDA_SOURCE_FLAG_OMIT_TABLE_DATA) != 0) {
        TableElementBytes = 0;
    } else {
        if (TableInfo->AssignedElementSizeInBytes == sizeof(USHORT)) {
            ValuesPerLine = 8;
            CharsPerValue = 8;
        } else {
            ValuesPerLine = 4;
            CharsPerValue = 12;
        }
        NumberOfLines = (NumberOfTableElements + ValuesPerLine - 1) / ValuesPerLine;
        TableElementBytes =
            (NumberOfTableElements * CharsPerValue) +
            (NumberOfLines * 4) +
            32;
    }

    RequiredSize = (
        BaseSize +
        TableElementBytes +
        (256 * 1024)
    );

    *AllocationSizePointer = ALIGN_UP(RequiredSize, BaseSize);

    return S_OK;
}

#define PH_ONLINE_CUDA_INDENT() do {      \
    *Output++ = ' ';                      \
    *Output++ = ' ';                      \
    *Output++ = ' ';                      \
    *Output++ = ' ';                      \
} while (0)

static
HRESULT
PhEmitOnlineCudaSource(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ PCSTRING Name,
    _In_ ULONG Flags,
    _Out_ PCHAR Base,
    _Out_ PULONGLONG NumberOfBytesWrittenPointer
    )
{
    PCHAR Output;
    ULONG Count;
    ULONGLONG Index;
    ULONG Seed1;
    ULONG Seed2;
    ULONG Seed3;
    ULONG Seed4;
    ULONG Seed5;
    ULONG Seed3Byte1;
    ULONG Seed3Byte2;
    ULONG Seed3Byte3;
    ULONG Seed3Byte4;
    PULONG Source;
    PUSHORT Source16;
    BOOLEAN UsingAssigned16;
    STRING DecimalString;
    CHAR NumberOfKeysBuffer[32];
    CHAR NumberOfTableElementsBuffer[32];
    HRESULT Result = S_OK;
    PTABLE_INFO_ON_DISK TableInfo;

    if (!ARGUMENT_PRESENT(Table) ||
        !ARGUMENT_PRESENT(Table->TableInfoOnDisk) ||
        !ARGUMENT_PRESENT(Name) ||
        !ARGUMENT_PRESENT(Name->Buffer) ||
        !ARGUMENT_PRESENT(Base) ||
        !ARGUMENT_PRESENT(NumberOfBytesWrittenPointer)) {
        return E_POINTER;
    }

    TableInfo = Table->TableInfoOnDisk;
    UsingAssigned16 = (TableInfo->AssignedElementSizeInBytes == sizeof(USHORT));
    Seed1 = (TableInfo->NumberOfSeeds >= 1) ? TableInfo->Seed1 : 0;
    Seed2 = (TableInfo->NumberOfSeeds >= 2) ? TableInfo->Seed2 : 0;
    Seed3 = (TableInfo->NumberOfSeeds >= 3) ? TableInfo->Seed3 : 0;
    Seed4 = (TableInfo->NumberOfSeeds >= 4) ? TableInfo->Seed4 : 0;
    Seed5 = (TableInfo->NumberOfSeeds >= 5) ? TableInfo->Seed5 : 0;
    Seed3Byte1 = (Seed3 & 0xff);
    Seed3Byte2 = ((Seed3 >> 8) & 0xff);
    Seed3Byte3 = ((Seed3 >> 16) & 0xff);
    Seed3Byte4 = ((Seed3 >> 24) & 0xff);

    Output = Base;
    DecimalString.Buffer = NULL;
    DecimalString.Length = 0;
    DecimalString.MaximumLength = sizeof(NumberOfKeysBuffer);

    OUTPUT_RAW("// Auto-generated by Perfect Hash online JIT.\n");
    OUTPUT_RAW("#ifndef PERFECTHASH_ONLINE_JIT_NAMESPACE_NAME\n");
    OUTPUT_RAW("#define PERFECTHASH_ONLINE_JIT_NAMESPACE_NAME ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("\n#endif\n");
    OUTPUT_RAW("#ifndef PERFECTHASH_ONLINE_JIT_INDEX_KERNEL_NAME\n");
    OUTPUT_RAW("#define PERFECTHASH_ONLINE_JIT_INDEX_KERNEL_NAME index_kernel\n");
    OUTPUT_RAW("#endif\n\n");
    OUTPUT_RAW("#include <stddef.h>\n");
    OUTPUT_RAW("#include <stdint.h>\n\n");

    OUTPUT_RAW("namespace perfecthash::generated::"
               "PERFECTHASH_ONLINE_JIT_NAMESPACE_NAME {\n\n");

    OUTPUT_RAW("inline constexpr uint32_t algorithm_id = ");
    OUTPUT_INT(Table->AlgorithmId);
    OUTPUT_RAW("u;\n");
    OUTPUT_RAW("inline constexpr uint32_t hash_function_id = ");
    OUTPUT_INT(Table->HashFunctionId);
    OUTPUT_RAW("u;\n");
    OUTPUT_RAW("inline constexpr uint32_t mask_function_id = ");
    OUTPUT_INT(Table->MaskFunctionId);
    OUTPUT_RAW("u;\n\n");

    OUTPUT_RAW("inline constexpr uint32_t key_size_bytes = ");
    OUTPUT_INT(TableInfo->KeySizeInBytes);
    OUTPUT_RAW("u;\n");
    OUTPUT_RAW("inline constexpr uint32_t original_key_size_bytes = ");
    OUTPUT_INT(TableInfo->OriginalKeySizeInBytes);
    OUTPUT_RAW("u;\n");
    OUTPUT_RAW("inline constexpr size_t number_of_keys = ");
    DecimalString.Buffer = NumberOfKeysBuffer;
    DecimalString.MaximumLength = sizeof(NumberOfKeysBuffer);
    DecimalString.Length = 0;
    AppendLongLongIntegerToString(&DecimalString,
                                  TableInfo->NumberOfKeys.QuadPart,
                                  CountNumberOfLongLongDigitsInline(
                                      TableInfo->NumberOfKeys.QuadPart
                                  ),
                                  '\0');
    OUTPUT_STRING(&DecimalString);
    OUTPUT_RAW(";\n");
    OUTPUT_RAW("inline constexpr size_t number_of_table_elements = ");
    DecimalString.Buffer = NumberOfTableElementsBuffer;
    DecimalString.MaximumLength = sizeof(NumberOfTableElementsBuffer);
    DecimalString.Length = 0;
    AppendLongLongIntegerToString(&DecimalString,
                                  TableInfo->NumberOfTableElements.QuadPart,
                                  CountNumberOfLongLongDigitsInline(
                                      TableInfo->NumberOfTableElements.QuadPart
                                  ),
                                  '\0');
    OUTPUT_STRING(&DecimalString);
    OUTPUT_RAW(";\n\n");

    OUTPUT_RAW("inline constexpr uint32_t hash_mask = ");
    OUTPUT_HEX(TableInfo->HashMask);
    OUTPUT_RAW("u;\n");
    OUTPUT_RAW("inline constexpr uint32_t index_mask = ");
    OUTPUT_HEX(TableInfo->IndexMask);
    OUTPUT_RAW("u;\n");
    OUTPUT_RAW("inline constexpr uint64_t downsize_bitmap = ");
    OUTPUT_HEX64(Table->DownsizeBitmap);
    OUTPUT_RAW("ULL;\n\n");

    if (TableInfo->KeySizeInBytes <= 4) {
        OUTPUT_RAW("using key_type = uint32_t;\n");
    } else {
        OUTPUT_RAW("using key_type = uint64_t;\n");
    }

    if (TableInfo->OriginalKeySizeInBytes <= 4) {
        OUTPUT_RAW("using original_key_type = uint32_t;\n");
    } else {
        OUTPUT_RAW("using original_key_type = uint64_t;\n");
    }

    if (UsingAssigned16) {
        OUTPUT_RAW("using table_data_type = uint16_t;\n\n");
    } else {
        OUTPUT_RAW("using table_data_type = uint32_t;\n\n");
    }

    if ((Flags & PH_ONLINE_JIT_CUDA_SOURCE_FLAG_OMIT_TABLE_DATA) != 0) {
        OUTPUT_RAW("inline constexpr bool has_embedded_table_data = false;\n\n");
    } else {
        OUTPUT_RAW("inline constexpr bool has_embedded_table_data = true;\n\n");
    }

    OUTPUT_RAW("inline constexpr uint32_t seed1 = ");
    OUTPUT_HEX(Seed1);
    OUTPUT_RAW("u;\n");
    OUTPUT_RAW("inline constexpr uint32_t seed2 = ");
    OUTPUT_HEX(Seed2);
    OUTPUT_RAW("u;\n");
    OUTPUT_RAW("inline constexpr uint32_t seed3 = ");
    OUTPUT_HEX(Seed3);
    OUTPUT_RAW("u;\n");
    OUTPUT_RAW("inline constexpr uint32_t seed4 = ");
    OUTPUT_HEX(Seed4);
    OUTPUT_RAW("u;\n");
    OUTPUT_RAW("inline constexpr uint32_t seed5 = ");
    OUTPUT_HEX(Seed5);
    OUTPUT_RAW("u;\n");
    OUTPUT_RAW("inline constexpr uint32_t seed3_byte1 = ");
    OUTPUT_INT(Seed3Byte1);
    OUTPUT_RAW("u;\n");
    OUTPUT_RAW("inline constexpr uint32_t seed3_byte2 = ");
    OUTPUT_INT(Seed3Byte2);
    OUTPUT_RAW("u;\n");
    OUTPUT_RAW("inline constexpr uint32_t seed3_byte3 = ");
    OUTPUT_INT(Seed3Byte3);
    OUTPUT_RAW("u;\n");
    OUTPUT_RAW("inline constexpr uint32_t seed3_byte4 = ");
    OUTPUT_INT(Seed3Byte4);
    OUTPUT_RAW("u;\n\n");

    if ((Flags & PH_ONLINE_JIT_CUDA_SOURCE_FLAG_OMIT_TABLE_DATA) == 0) {
        OUTPUT_RAW("inline constexpr table_data_type table_data[number_of_table_elements] = {\n");
        if (UsingAssigned16) {
            Source16 = Table->Assigned16;
            for (Index = 0, Count = 0; Index < TableInfo->NumberOfTableElements.QuadPart; Index++) {
                if (Count == 0) {
                    PH_ONLINE_CUDA_INDENT();
                }
                OUTPUT_HEX(*Source16++);
                *Output++ = ',';
                if (++Count == 8) {
                    Count = 0;
                    *Output++ = '\n';
                } else {
                    *Output++ = ' ';
                }
            }
        } else {
            Source = Table->Assigned;
            for (Index = 0, Count = 0; Index < TableInfo->NumberOfTableElements.QuadPart; Index++) {
                if (Count == 0) {
                    PH_ONLINE_CUDA_INDENT();
                }
                OUTPUT_HEX(*Source++);
                *Output++ = ',';
                if (++Count == 4) {
                    Count = 0;
                    *Output++ = '\n';
                } else {
                    *Output++ = ' ';
                }
            }
        }
        if (*(Output - 1) == ' ') {
            *(Output - 1) = '\n';
        }
        OUTPUT_RAW("};\n\n");
    } else {
        OUTPUT_RAW("// table_data is supplied by the consumer at runtime.\n\n");
    }

    OUTPUT_RAW("__host__ __device__ __forceinline__ uint64_t extract_bits64("
               "uint64_t value, uint64_t bitmap) noexcept {\n");
    OUTPUT_RAW("    uint64_t result = 0;\n");
    OUTPUT_RAW("    uint64_t out_bit = 0;\n");
    OUTPUT_RAW("    while (bitmap) {\n");
    OUTPUT_RAW("        const uint64_t lsb = bitmap & (~bitmap + 1);\n");
    OUTPUT_RAW("        if (value & lsb) {\n");
    OUTPUT_RAW("            result |= (1ULL << out_bit);\n");
    OUTPUT_RAW("        }\n");
    OUTPUT_RAW("        bitmap ^= lsb;\n");
    OUTPUT_RAW("        ++out_bit;\n");
    OUTPUT_RAW("    }\n");
    OUTPUT_RAW("    return result;\n");
    OUTPUT_RAW("}\n\n");

    OUTPUT_RAW("__host__ __device__ __forceinline__ constexpr uint64_t mask_from_bits(uint32_t bits) noexcept {\n");
    OUTPUT_RAW("    return (bits >= 64u) ? ~0ULL : ((1ULL << bits) - 1ULL);\n");
    OUTPUT_RAW("}\n");
    OUTPUT_RAW("inline constexpr uint64_t key_mask = mask_from_bits(key_size_bytes * 8u);\n");
    OUTPUT_RAW("inline constexpr uint64_t original_key_mask = mask_from_bits(original_key_size_bytes * 8u);\n\n");

    OUTPUT_RAW("__host__ __device__ __forceinline__ key_type downsize_key("
               "uint64_t key) noexcept {\n");
    OUTPUT_RAW("    key &= original_key_mask;\n");
    OUTPUT_RAW("    if (downsize_bitmap) {\n");
    OUTPUT_RAW("        return static_cast<key_type>(extract_bits64(key, downsize_bitmap) & key_mask);\n");
    OUTPUT_RAW("    }\n");
    OUTPUT_RAW("    return static_cast<key_type>(key & key_mask);\n");
    OUTPUT_RAW("}\n\n");

    OUTPUT_RAW("__host__ __device__ __forceinline__ uint32_t rotr32("
               "uint32_t value, uint32_t shift) noexcept {\n");
    OUTPUT_RAW("    shift &= 31u;\n");
    OUTPUT_RAW("    if (shift == 0u) { return value; }\n");
    OUTPUT_RAW("    return (value >> shift) | (value << (32u - shift));\n");
    OUTPUT_RAW("}\n\n");

    OUTPUT_RAW("struct slot_pair_type {\n");
    OUTPUT_RAW("    uint32_t first;\n");
    OUTPUT_RAW("    uint32_t second;\n");
    OUTPUT_RAW("};\n\n");

    OUTPUT_RAW("__host__ __device__ __forceinline__ slot_pair_type slot_pair_from_key("
               "original_key_type key) noexcept {\n");
    OUTPUT_RAW("    const key_type downsized = downsize_key(static_cast<uint64_t>(key));\n");
    if (Table->HashFunctionId == PerfectHashHashMultiplyShiftRFunctionId) {
        OUTPUT_RAW("    key_type vertex1 = static_cast<key_type>(downsized * static_cast<key_type>(seed1));\n");
        OUTPUT_RAW("    vertex1 >>= seed3_byte1;\n");
        OUTPUT_RAW("    key_type vertex2 = static_cast<key_type>(downsized * static_cast<key_type>(seed2));\n");
        OUTPUT_RAW("    vertex2 >>= seed3_byte2;\n");
        OUTPUT_RAW("    return slot_pair_type{static_cast<uint32_t>(vertex1 & hash_mask), static_cast<uint32_t>(vertex2 & hash_mask)};\n");
    } else if (Table->HashFunctionId == PerfectHashHashMultiplyShiftLRFunctionId) {
        OUTPUT_RAW("    key_type vertex1 = static_cast<key_type>(downsized * static_cast<key_type>(seed1));\n");
        OUTPUT_RAW("    vertex1 = static_cast<key_type>(vertex1 << seed3_byte1);\n");
        OUTPUT_RAW("    key_type vertex2 = static_cast<key_type>(downsized * static_cast<key_type>(seed2));\n");
        OUTPUT_RAW("    vertex2 >>= seed3_byte2;\n");
        OUTPUT_RAW("    return slot_pair_type{static_cast<uint32_t>(vertex1 & hash_mask), static_cast<uint32_t>(vertex2 & hash_mask)};\n");
    } else if (Table->HashFunctionId == PerfectHashHashMultiplyShiftRMultiplyFunctionId) {
        OUTPUT_RAW("    key_type vertex1 = static_cast<key_type>(downsized * static_cast<key_type>(seed1));\n");
        OUTPUT_RAW("    vertex1 >>= seed3_byte1;\n");
        OUTPUT_RAW("    vertex1 = static_cast<key_type>(vertex1 * static_cast<key_type>(seed2));\n");
        OUTPUT_RAW("    key_type vertex2 = static_cast<key_type>(downsized * static_cast<key_type>(seed4));\n");
        OUTPUT_RAW("    vertex2 >>= seed3_byte2;\n");
        OUTPUT_RAW("    vertex2 = static_cast<key_type>(vertex2 * static_cast<key_type>(seed5));\n");
        OUTPUT_RAW("    return slot_pair_type{static_cast<uint32_t>(vertex1 & hash_mask), static_cast<uint32_t>(vertex2 & hash_mask)};\n");
    } else if (Table->HashFunctionId == PerfectHashHashMultiplyShiftR2FunctionId) {
        OUTPUT_RAW("    key_type vertex1 = static_cast<key_type>(downsized * static_cast<key_type>(seed1));\n");
        OUTPUT_RAW("    vertex1 >>= seed3_byte1;\n");
        OUTPUT_RAW("    vertex1 = static_cast<key_type>(vertex1 * static_cast<key_type>(seed2));\n");
        OUTPUT_RAW("    vertex1 >>= seed3_byte2;\n");
        OUTPUT_RAW("    key_type vertex2 = static_cast<key_type>(downsized * static_cast<key_type>(seed4));\n");
        OUTPUT_RAW("    vertex2 >>= seed3_byte3;\n");
        OUTPUT_RAW("    vertex2 = static_cast<key_type>(vertex2 * static_cast<key_type>(seed5));\n");
        OUTPUT_RAW("    vertex2 >>= seed3_byte4;\n");
        OUTPUT_RAW("    return slot_pair_type{static_cast<uint32_t>(vertex1 & hash_mask), static_cast<uint32_t>(vertex2 & hash_mask)};\n");
    } else if (Table->HashFunctionId == PerfectHashHashMultiplyShiftRXFunctionId) {
        OUTPUT_RAW("    key_type vertex1 = static_cast<key_type>(downsized * static_cast<key_type>(seed1));\n");
        OUTPUT_RAW("    vertex1 >>= seed3_byte1;\n");
        OUTPUT_RAW("    key_type vertex2 = static_cast<key_type>(downsized * static_cast<key_type>(seed2));\n");
        OUTPUT_RAW("    vertex2 >>= seed3_byte1;\n");
        OUTPUT_RAW("    return slot_pair_type{static_cast<uint32_t>(vertex1), static_cast<uint32_t>(vertex2)};\n");
    } else if (Table->HashFunctionId == PerfectHashHashMulshrolate1RXFunctionId) {
        OUTPUT_RAW("    const uint32_t downsized32 = static_cast<uint32_t>(downsized);\n");
        OUTPUT_RAW("    uint32_t vertex1 = downsized32 * seed1;\n");
        OUTPUT_RAW("    vertex1 = rotr32(vertex1, seed3_byte2);\n");
        OUTPUT_RAW("    vertex1 >>= seed3_byte1;\n");
        OUTPUT_RAW("    uint32_t vertex2 = downsized32 * seed2;\n");
        OUTPUT_RAW("    vertex2 >>= seed3_byte1;\n");
        OUTPUT_RAW("    return slot_pair_type{vertex1, vertex2};\n");
    } else if (Table->HashFunctionId == PerfectHashHashMulshrolate2RXFunctionId) {
        OUTPUT_RAW("    const uint32_t downsized32 = static_cast<uint32_t>(downsized);\n");
        OUTPUT_RAW("    uint32_t vertex1 = downsized32 * seed1;\n");
        OUTPUT_RAW("    vertex1 = rotr32(vertex1, seed3_byte2);\n");
        OUTPUT_RAW("    vertex1 >>= seed3_byte1;\n");
        OUTPUT_RAW("    uint32_t vertex2 = downsized32 * seed2;\n");
        OUTPUT_RAW("    vertex2 = rotr32(vertex2, seed3_byte3);\n");
        OUTPUT_RAW("    vertex2 >>= seed3_byte1;\n");
        OUTPUT_RAW("    return slot_pair_type{vertex1, vertex2};\n");
    } else if (Table->HashFunctionId == PerfectHashHashMulshrolate3RXFunctionId) {
        OUTPUT_RAW("    const uint32_t downsized32 = static_cast<uint32_t>(downsized);\n");
        OUTPUT_RAW("    uint32_t vertex1 = downsized32 * seed1;\n");
        OUTPUT_RAW("    vertex1 = rotr32(vertex1, seed3_byte2);\n");
        OUTPUT_RAW("    vertex1 = vertex1 * seed4;\n");
        OUTPUT_RAW("    vertex1 >>= seed3_byte1;\n");
        OUTPUT_RAW("    uint32_t vertex2 = downsized32 * seed2;\n");
        OUTPUT_RAW("    vertex2 = rotr32(vertex2, seed3_byte3);\n");
        OUTPUT_RAW("    vertex2 >>= seed3_byte1;\n");
        OUTPUT_RAW("    return slot_pair_type{vertex1, vertex2};\n");
    } else if (Table->HashFunctionId == PerfectHashHashMulshrolate4RXFunctionId) {
        OUTPUT_RAW("    const uint32_t downsized32 = static_cast<uint32_t>(downsized);\n");
        OUTPUT_RAW("    uint32_t vertex1 = downsized32 * seed1;\n");
        OUTPUT_RAW("    vertex1 = rotr32(vertex1, seed3_byte2);\n");
        OUTPUT_RAW("    vertex1 = vertex1 * seed4;\n");
        OUTPUT_RAW("    vertex1 >>= seed3_byte1;\n");
        OUTPUT_RAW("    uint32_t vertex2 = downsized32 * seed2;\n");
        OUTPUT_RAW("    vertex2 = rotr32(vertex2, seed3_byte3);\n");
        OUTPUT_RAW("    vertex2 = vertex2 * seed5;\n");
        OUTPUT_RAW("    vertex2 >>= seed3_byte1;\n");
        OUTPUT_RAW("    return slot_pair_type{vertex1, vertex2};\n");
    } else {
        return PH_E_NOT_IMPLEMENTED;
    }
    OUTPUT_RAW("}\n\n");

    OUTPUT_RAW("__host__ __device__ __forceinline__ uint32_t index_from_key("
               "original_key_type key, const table_data_type* table) noexcept {\n");
    OUTPUT_RAW("    const slot_pair_type slots = slot_pair_from_key(key);\n");
    OUTPUT_RAW("    const uint32_t value_low = static_cast<uint32_t>(table[slots.first]);\n");
    OUTPUT_RAW("    const uint32_t value_high = static_cast<uint32_t>(table[slots.second]);\n");
    OUTPUT_RAW("    return static_cast<uint32_t>((value_low + value_high) & index_mask);\n");
    OUTPUT_RAW("}\n\n");

    if ((Flags & PH_ONLINE_JIT_CUDA_SOURCE_FLAG_OMIT_KERNELS) == 0) {
        if ((Flags & PH_ONLINE_JIT_CUDA_SOURCE_FLAG_OMIT_TABLE_DATA) != 0) {
            OUTPUT_RAW("__global__ void PERFECTHASH_ONLINE_JIT_INDEX_KERNEL_NAME("
                       "const original_key_type* query_keys, "
                       "const table_data_type* table_data, "
                       "uint32_t* out, size_t count) {\n");
        } else {
            OUTPUT_RAW("__global__ void PERFECTHASH_ONLINE_JIT_INDEX_KERNEL_NAME("
                       "const original_key_type* query_keys, "
                       "uint32_t* out, size_t count) {\n");
        }
        OUTPUT_RAW("    const size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;\n");
        OUTPUT_RAW("    if (i < count) {\n");
        if ((Flags & PH_ONLINE_JIT_CUDA_SOURCE_FLAG_OMIT_TABLE_DATA) != 0) {
            OUTPUT_RAW("        out[i] = index_from_key(query_keys[i], table_data);\n");
        } else {
            OUTPUT_RAW("        out[i] = index_from_key(query_keys[i], "
                       "perfecthash::generated::"
                       "PERFECTHASH_ONLINE_JIT_NAMESPACE_NAME::table_data);\n");
        }
        OUTPUT_RAW("    }\n");
        OUTPUT_RAW("}\n\n");
    }

    OUTPUT_RAW("} // namespace perfecthash::generated::"
               "PERFECTHASH_ONLINE_JIT_NAMESPACE_NAME\n");

    *NumberOfBytesWrittenPointer = RtlPointerToOffset(Base, Output);
    return Result;
}

PH_ONLINE_JIT_API
int32_t
PhOnlineJitOpen(
    PH_ONLINE_JIT_CONTEXT **ContextPointer
    )
{
    HRESULT Result;
    PH_ONLINE_JIT_CONTEXT *Context = NULL;
    PICLASSFACTORY ClassFactory = NULL;
    PPERFECT_HASH_ONLINE Online = NULL;

    if (!ARGUMENT_PRESENT(ContextPointer)) {
        return E_POINTER;
    }

    *ContextPointer = NULL;

    Context = (PH_ONLINE_JIT_CONTEXT *)calloc(1, sizeof(*Context));
    if (!Context) {
        return E_OUTOFMEMORY;
    }

    Result = PerfectHashDllGetClassObject(&CLSID_PERFECT_HASH,
                                          &IID_PERFECT_HASH_ICLASSFACTORY,
                                          (PVOID *)&ClassFactory);
    if (FAILED(Result)) {
        goto Error;
    }

    Result = ClassFactory->Vtbl->CreateInstance(
        ClassFactory,
        NULL,
        &IID_PERFECT_HASH_ONLINE,
        (void **)&Online
    );
    if (FAILED(Result)) {
        goto Error;
    }

    Context->ClassFactory = ClassFactory;
    Context->Online = Online;

    *ContextPointer = Context;
    return S_OK;

Error:
    if (Online) {
        Online->Vtbl->Release(Online);
        Online = NULL;
    }

    if (ClassFactory) {
        ClassFactory->Vtbl->Release(ClassFactory);
        ClassFactory = NULL;
    }

    free(Context);
    Context = NULL;

    return Result;
}

PH_ONLINE_JIT_API
void
PhOnlineJitClose(
    PH_ONLINE_JIT_CONTEXT *Context
    )
{
    if (!Context) {
        return;
    }

    if (Context->Online) {
        Context->Online->Vtbl->Release(Context->Online);
        Context->Online = NULL;
    }

    if (Context->ClassFactory) {
        Context->ClassFactory->Vtbl->Release(Context->ClassFactory);
        Context->ClassFactory = NULL;
    }

    free(Context);
}

PH_ONLINE_JIT_API
int32_t
PhOnlineJitCreateTable32(
    PH_ONLINE_JIT_CONTEXT *Context,
    PH_ONLINE_JIT_HASH_FUNCTION HashFunction,
    const uint32_t *Keys,
    uint64_t NumberOfKeys,
    PH_ONLINE_JIT_TABLE **TablePointer
    )
{
    HRESULT Result;
    PERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags = {0};
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags = {0};
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId;
    PPERFECT_HASH_TABLE Table = NULL;
    PH_ONLINE_JIT_TABLE *Wrapper = NULL;

    if (!ARGUMENT_PRESENT(Context) ||
        !ARGUMENT_PRESENT(Context->Online) ||
        !ARGUMENT_PRESENT(Keys) ||
        NumberOfKeys == 0 ||
        !ARGUMENT_PRESENT(TablePointer)) {
        return E_INVALIDARG;
    }

    *TablePointer = NULL;

    Result = PhMapOnlineJitHashFunction(HashFunction, &HashFunctionId);
    if (FAILED(Result)) {
        return Result;
    }

    KeysLoadFlags.SortKeys = TRUE;
    KeysLoadFlags.KeysAreSorted = FALSE;

    TableCreateFlags.NoFileIo = TRUE;
    TableCreateFlags.Quiet = TRUE;
    TableCreateFlags.DoNotTryUseHash16Impl = TRUE;

    Result = Context->Online->Vtbl->CreateTableFromKeys(
        Context->Online,
        PerfectHashChm01AlgorithmId,
        HashFunctionId,
        PerfectHashAndMaskFunctionId,
        sizeof(ULONG),
        (ULONGLONG)NumberOfKeys,
        (PVOID)Keys,
        &KeysLoadFlags,
        &TableCreateFlags,
        NULL,
        &Table
    );
    if (FAILED(Result)) {
        return Result;
    }

    Wrapper = (PH_ONLINE_JIT_TABLE *)calloc(1, sizeof(*Wrapper));
    if (!Wrapper) {
        Table->Vtbl->Release(Table);
        return E_OUTOFMEMORY;
    }

    Wrapper->Table = Table;
    *TablePointer = Wrapper;

    return S_OK;
}

PH_ONLINE_JIT_API
int32_t
PhOnlineJitCreateTable64(
    PH_ONLINE_JIT_CONTEXT *Context,
    PH_ONLINE_JIT_HASH_FUNCTION HashFunction,
    const uint64_t *Keys,
    uint64_t NumberOfKeys,
    PH_ONLINE_JIT_TABLE **TablePointer
    )
{
    HRESULT Result;
    PERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags = {0};
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags = {0};
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId;
    PPERFECT_HASH_TABLE Table = NULL;
    PH_ONLINE_JIT_TABLE *Wrapper = NULL;

    if (!ARGUMENT_PRESENT(Context) ||
        !ARGUMENT_PRESENT(Context->Online) ||
        !ARGUMENT_PRESENT(Keys) ||
        NumberOfKeys == 0 ||
        !ARGUMENT_PRESENT(TablePointer)) {
        return E_INVALIDARG;
    }

    *TablePointer = NULL;

    Result = PhMapOnlineJitHashFunction(HashFunction, &HashFunctionId);
    if (FAILED(Result)) {
        return Result;
    }

    KeysLoadFlags.SortKeys = TRUE;
    KeysLoadFlags.KeysAreSorted = FALSE;

    TableCreateFlags.NoFileIo = TRUE;
    TableCreateFlags.Quiet = TRUE;
    TableCreateFlags.DoNotTryUseHash16Impl = TRUE;

    Result = Context->Online->Vtbl->CreateTableFromKeys(
        Context->Online,
        PerfectHashChm01AlgorithmId,
        HashFunctionId,
        PerfectHashAndMaskFunctionId,
        sizeof(ULONGLONG),
        (ULONGLONG)NumberOfKeys,
        (PVOID)Keys,
        &KeysLoadFlags,
        &TableCreateFlags,
        NULL,
        &Table
    );
    if (FAILED(Result)) {
        return Result;
    }
    if (!ARGUMENT_PRESENT(Table)) {
        return E_UNEXPECTED;
    }

    if (!ARGUMENT_PRESENT(Table->TableInfoOnDisk) ||
        Table->TableInfoOnDisk->OriginalKeySizeInBytes <= sizeof(uint32_t) ||
        Table->TableInfoOnDisk->KeySizeInBytes > sizeof(uint32_t)) {
        Table->Vtbl->Release(Table);
        return PH_E_NOT_IMPLEMENTED;
    }

    Wrapper = (PH_ONLINE_JIT_TABLE *)calloc(1, sizeof(*Wrapper));
    if (!Wrapper) {
        Table->Vtbl->Release(Table);
        return E_OUTOFMEMORY;
    }

    Wrapper->Table = Table;
    *TablePointer = Wrapper;

    return S_OK;
}

PH_ONLINE_JIT_API
int32_t
PhOnlineJitCompileTable(
    PH_ONLINE_JIT_CONTEXT *Context,
    PH_ONLINE_JIT_TABLE *Table,
    PH_ONLINE_JIT_BACKEND Backend,
    uint32_t VectorWidth,
    PH_ONLINE_JIT_MAX_ISA JitMaxIsa
    )
{
    return PhOnlineJitCompileTableEx(Context,
                                     Table,
                                     Backend,
                                     VectorWidth,
                                     JitMaxIsa,
                                     0,
                                     NULL,
                                     NULL);
}

PH_ONLINE_JIT_API
int32_t
PhOnlineJitCompileTableEx(
    PH_ONLINE_JIT_CONTEXT *Context,
    PH_ONLINE_JIT_TABLE *Table,
    PH_ONLINE_JIT_BACKEND Backend,
    uint32_t VectorWidth,
    PH_ONLINE_JIT_MAX_ISA JitMaxIsa,
    uint32_t Flags,
    PH_ONLINE_JIT_BACKEND *EffectiveBackend,
    uint32_t *EffectiveVectorWidth
    )
{
    HRESULT Result;
    HRESULT LastResult;
    ULONG CandidateWidths[8] = {0};
    ULONG CandidateCount = 0;
    ULONG Index;
    BOOLEAN StrictVectorWidth;

    if (!ARGUMENT_PRESENT(Context) ||
        !ARGUMENT_PRESENT(Context->Online) ||
        !ARGUMENT_PRESENT(Table) ||
        !ARGUMENT_PRESENT(Table->Table)) {
        return E_INVALIDARG;
    }

    //
    // Force lazy re-acquisition on subsequent vector index requests after
    // every compile path.
    //

    PhReleaseOnlineJitInterface(Table);

    if (!IsValidPerfectHashJitMaxIsaId((PERFECT_HASH_JIT_MAX_ISA_ID)JitMaxIsa)) {
        return E_INVALIDARG;
    }

    if (Backend != PhOnlineJitBackendAuto &&
        Backend != PhOnlineJitBackendRawDogJit &&
        Backend != PhOnlineJitBackendLlvmJit) {
        return E_INVALIDARG;
    }

    StrictVectorWidth = ((Flags & PH_ONLINE_JIT_COMPILE_FLAG_STRICT_VECTOR_WIDTH) != 0);

    if (ARGUMENT_PRESENT(EffectiveBackend)) {
        *EffectiveBackend = Backend;
    }
    if (ARGUMENT_PRESENT(EffectiveVectorWidth)) {
        *EffectiveVectorWidth = VectorWidth;
    }

    if (Backend == PhOnlineJitBackendLlvmJit) {
        Result = PhCompileOnlineJitBackend(Context,
                                           Table,
                                           Backend,
                                           VectorWidth,
                                           JitMaxIsa);
        if (SUCCEEDED(Result)) {
            if (ARGUMENT_PRESENT(EffectiveBackend)) {
                *EffectiveBackend = PhOnlineJitBackendLlvmJit;
            }
            if (ARGUMENT_PRESENT(EffectiveVectorWidth)) {
                *EffectiveVectorWidth = VectorWidth;
            }
        }
        return Result;
    }

    switch (VectorWidth) {
        case 0:
        case 1:
            CandidateWidths[CandidateCount++] = VectorWidth;
            break;
        case 2:
            CandidateWidths[CandidateCount++] = 2;
            if (!StrictVectorWidth) {
                CandidateWidths[CandidateCount++] = 1;
            }
            break;
        case 4:
            CandidateWidths[CandidateCount++] = 4;
            if (!StrictVectorWidth) {
                CandidateWidths[CandidateCount++] = 1;
            }
            break;
        case 8:
            CandidateWidths[CandidateCount++] = 8;
            if (!StrictVectorWidth) {
                CandidateWidths[CandidateCount++] = 4;
                CandidateWidths[CandidateCount++] = 1;
            }
            break;
        case 16:
            CandidateWidths[CandidateCount++] = 16;
            if (!StrictVectorWidth) {
                CandidateWidths[CandidateCount++] = 8;
                CandidateWidths[CandidateCount++] = 4;
                CandidateWidths[CandidateCount++] = 1;
            }
            break;
        default:
            return E_INVALIDARG;
    }

    LastResult = PH_E_NOT_IMPLEMENTED;

    for (Index = 0; Index < CandidateCount; Index++) {
        Result = PhCompileOnlineJitBackend(Context,
                                           Table,
                                           PhOnlineJitBackendRawDogJit,
                                           CandidateWidths[Index],
                                           JitMaxIsa);
        if (SUCCEEDED(Result)) {
            if (ARGUMENT_PRESENT(EffectiveBackend)) {
                *EffectiveBackend = PhOnlineJitBackendRawDogJit;
            }
            if (ARGUMENT_PRESENT(EffectiveVectorWidth)) {
                *EffectiveVectorWidth = CandidateWidths[Index];
            }
            return S_OK;
        }

        if (Result != PH_E_NOT_IMPLEMENTED &&
            Result != PH_E_INVARIANT_CHECK_FAILED) {
            if (Backend == PhOnlineJitBackendRawDogJit) {
                return Result;
            }
            LastResult = Result;
            break;
        }

        LastResult = Result;
    }

    if (Backend == PhOnlineJitBackendRawDogJit) {
        return LastResult;
    }

    for (Index = 0; Index < CandidateCount; Index++) {
        Result = PhCompileOnlineJitBackend(Context,
                                           Table,
                                           PhOnlineJitBackendLlvmJit,
                                           CandidateWidths[Index],
                                           JitMaxIsa);
        if (SUCCEEDED(Result)) {
            if (ARGUMENT_PRESENT(EffectiveBackend)) {
                *EffectiveBackend = PhOnlineJitBackendLlvmJit;
            }
            if (ARGUMENT_PRESENT(EffectiveVectorWidth)) {
                *EffectiveVectorWidth = CandidateWidths[Index];
            }
            return S_OK;
        }

        if (Result != PH_E_LLVM_BACKEND_NOT_FOUND &&
            Result != PH_E_NOT_IMPLEMENTED &&
            Result != PH_E_INVARIANT_CHECK_FAILED) {
            return Result;
        }

        LastResult = Result;
    }

    return LastResult;
}

PH_ONLINE_JIT_API
int32_t
PhOnlineJitGetCudaSource(
    PH_ONLINE_JIT_TABLE *Table,
    char **SourceText,
    size_t *SourceSize
    )
{
    return PhOnlineJitGetCudaSourceEx(Table, 0, SourceText, SourceSize);
}

PH_ONLINE_JIT_API
int32_t
PhOnlineJitGetCudaSourceEx(
    PH_ONLINE_JIT_TABLE *Table,
    uint32_t Flags,
    char **SourceText,
    size_t *SourceSize
    )
{
    HRESULT Result;
    ULONGLONG AllocationSize;
    PCHAR Buffer = NULL;
    ULONGLONG NumberOfBytesWritten = 0;
    STRING Name = {0};
    CHAR NameBuffer[] = "online_jit_table";

    if (!ARGUMENT_PRESENT(Table) ||
        !ARGUMENT_PRESENT(Table->Table) ||
        !ARGUMENT_PRESENT(SourceText)) {
        return E_INVALIDARG;
    }

    *SourceText = NULL;
    if (ARGUMENT_PRESENT(SourceSize)) {
        *SourceSize = 0;
    }

    Result = PhEstimateCudaSourceBytes(Table->Table, Flags, &AllocationSize);
    if (FAILED(Result)) {
        return Result;
    }

    Buffer = (PCHAR)calloc(1, (size_t)AllocationSize + 1);
    if (!Buffer) {
        return E_OUTOFMEMORY;
    }

    Name.Buffer = NameBuffer;
    Name.Length = sizeof(NameBuffer) - sizeof(CHAR);
    Name.MaximumLength = sizeof(NameBuffer);

    Result = PhEmitOnlineCudaSource(Table->Table,
                                    &Name,
                                    Flags,
                                    Buffer,
                                    &NumberOfBytesWritten);
    if (FAILED(Result)) {
        free(Buffer);
        return Result;
    }

    Buffer[NumberOfBytesWritten] = '\0';

    *SourceText = Buffer;
    if (ARGUMENT_PRESENT(SourceSize)) {
        *SourceSize = (size_t)NumberOfBytesWritten;
    }

    return S_OK;
}

PH_ONLINE_JIT_API
void
PhOnlineJitFreeCudaSource(
    char *SourceText
    )
{
    free(SourceText);
}

PH_ONLINE_JIT_API
int32_t
PhOnlineJitGetCudaTableData(
    PH_ONLINE_JIT_TABLE *Table,
    void **TableData,
    size_t *TableDataSize,
    uint32_t *TableDataElementSize,
    size_t *NumberOfTableElements
    )
{
    PTABLE_INFO_ON_DISK TableInfo;
    ULONGLONG NumberOfTableElements64;
    ULONG ElementSizeInBytes;
    size_t CopySize;
    PVOID Buffer;
    PVOID Source;

    if (!ARGUMENT_PRESENT(Table) ||
        !ARGUMENT_PRESENT(Table->Table) ||
        !ARGUMENT_PRESENT(TableData) ||
        !ARGUMENT_PRESENT(Table->Table->TableInfoOnDisk)) {
        return E_INVALIDARG;
    }

    *TableData = NULL;
    if (ARGUMENT_PRESENT(TableDataSize)) {
        *TableDataSize = 0;
    }
    if (ARGUMENT_PRESENT(TableDataElementSize)) {
        *TableDataElementSize = 0;
    }
    if (ARGUMENT_PRESENT(NumberOfTableElements)) {
        *NumberOfTableElements = 0;
    }

    TableInfo = Table->Table->TableInfoOnDisk;
    NumberOfTableElements64 = TableInfo->NumberOfTableElements.QuadPart;
    ElementSizeInBytes = TableInfo->AssignedElementSizeInBytes;
    if (ElementSizeInBytes == 0) {
        return E_INVALIDARG;
    }
    if (NumberOfTableElements64 >
        (((ULONGLONG)~((size_t)0)) / (ULONGLONG)ElementSizeInBytes)) {
        return E_OUTOFMEMORY;
    }
    CopySize = (size_t)(NumberOfTableElements64 * (ULONGLONG)ElementSizeInBytes);

    if (ElementSizeInBytes == sizeof(USHORT)) {
        Source = Table->Table->Assigned16;
    } else if (ElementSizeInBytes == sizeof(ULONG)) {
        Source = Table->Table->Assigned;
    } else {
        return PH_E_NOT_IMPLEMENTED;
    }

    if (Source == NULL) {
        return E_POINTER;
    }

    Buffer = malloc(CopySize);
    if (Buffer == NULL) {
        return E_OUTOFMEMORY;
    }

    CopyMemory(Buffer, Source, CopySize);

    *TableData = Buffer;
    if (ARGUMENT_PRESENT(TableDataSize)) {
        *TableDataSize = CopySize;
    }
    if (ARGUMENT_PRESENT(TableDataElementSize)) {
        *TableDataElementSize = ElementSizeInBytes;
    }
    if (ARGUMENT_PRESENT(NumberOfTableElements)) {
        *NumberOfTableElements = (size_t)NumberOfTableElements64;
    }

    return S_OK;
}

PH_ONLINE_JIT_API
void
PhOnlineJitFreeCudaTableData(
    void *TableData
    )
{
    free(TableData);
}

PH_ONLINE_JIT_API
int32_t
PhOnlineJitGetTableInfo(
    PH_ONLINE_JIT_TABLE *Table,
    PH_ONLINE_JIT_TABLE_INFO *TableInfo
    )
{
    HRESULT Result;
    PTABLE_INFO_ON_DISK Info;

    if (!ARGUMENT_PRESENT(Table) ||
        !ARGUMENT_PRESENT(Table->Table) ||
        !ARGUMENT_PRESENT(TableInfo) ||
        !ARGUMENT_PRESENT(Table->Table->TableInfoOnDisk)) {
        return E_INVALIDARG;
    }

    Info = Table->Table->TableInfoOnDisk;
    ZeroStructPointer(TableInfo);

    TableInfo->KeySizeInBytes = Info->KeySizeInBytes;
    TableInfo->OriginalKeySizeInBytes = Info->OriginalKeySizeInBytes;
    TableInfo->AssignedElementSizeInBytes = Info->AssignedElementSizeInBytes;
    {
        PH_ONLINE_JIT_HASH_FUNCTION PublicHashFunction;

        Result = PhMapPerfectHashHashFunctionToOnlineJit(
            Table->Table->HashFunctionId,
            &PublicHashFunction
        );
        if (FAILED(Result)) {
            return Result;
        }
        TableInfo->HashFunctionId = (uint32_t)PublicHashFunction;
    }
    TableInfo->MaskFunctionId = Table->Table->MaskFunctionId;
    TableInfo->HashMask = Info->HashMask;
    TableInfo->IndexMask = Info->IndexMask;
    TableInfo->Seed1 = Info->Seed1;
    TableInfo->Seed2 = Info->Seed2;
    TableInfo->Seed3 = Info->Seed3;
    TableInfo->Seed4 = Info->Seed4;
    TableInfo->Seed5 = Info->Seed5;
    TableInfo->NumberOfKeys = Info->NumberOfKeys.QuadPart;
    TableInfo->NumberOfTableElements = Info->NumberOfTableElements.QuadPart;
    TableInfo->DownsizeBitmap = Table->Table->DownsizeBitmap;

    return S_OK;
}

PH_ONLINE_JIT_API
int32_t
PhOnlineJitIndex32x2(
    PH_ONLINE_JIT_TABLE *Table,
    const uint32_t *Keys,
    uint32_t *Indexes
    )
{
    HRESULT Result;
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit;

    if (!ARGUMENT_PRESENT(Table) ||
        !ARGUMENT_PRESENT(Keys) ||
        !ARGUMENT_PRESENT(Indexes)) {
        return E_INVALIDARG;
    }

    Result = PhEnsureOnlineJitInterface(Table);
    if (FAILED(Result)) {
        return Result;
    }

    Jit = Table->JitInterface;
    return Jit->Vtbl->Index32x2(
        Jit,
        (ULONG)Keys[0],
        (ULONG)Keys[1],
        (PULONG)&Indexes[0],
        (PULONG)&Indexes[1]
    );
}

PH_ONLINE_JIT_API
int32_t
PhOnlineJitIndex32(
    PH_ONLINE_JIT_TABLE *Table,
    uint32_t Key,
    uint32_t *Index
    )
{
    PTABLE_INFO_ON_DISK TableInfo;

    if (!ARGUMENT_PRESENT(Table) ||
        !ARGUMENT_PRESENT(Table->Table) ||
        !ARGUMENT_PRESENT(Index)) {
        return E_INVALIDARG;
    }

    TableInfo = Table->Table->TableInfoOnDisk;
    if (ARGUMENT_PRESENT(TableInfo) &&
        TableInfo->OriginalKeySizeInBytes > sizeof(uint32_t)) {
        return PH_E_NOT_IMPLEMENTED;
    }

    return Table->Table->Vtbl->Index(Table->Table, (ULONG)Key, (PULONG)Index);
}

PH_ONLINE_JIT_API
int32_t
PhOnlineJitIndex64(
    PH_ONLINE_JIT_TABLE *Table,
    uint64_t Key,
    uint32_t *Index
    )
{
    PTABLE_INFO_ON_DISK TableInfo;
    ULONG DownsizedKey;

    if (!ARGUMENT_PRESENT(Table) ||
        !ARGUMENT_PRESENT(Index)) {
        return E_INVALIDARG;
    }

    if (!ARGUMENT_PRESENT(Table->Table) ||
        !ARGUMENT_PRESENT(Table->Table->TableInfoOnDisk)) {
        return E_INVALIDARG;
    }

    TableInfo = Table->Table->TableInfoOnDisk;
    if (TableInfo->OriginalKeySizeInBytes <= sizeof(uint32_t) ||
        TableInfo->KeySizeInBytes > sizeof(uint32_t)) {
        return PH_E_NOT_IMPLEMENTED;
    }
    DownsizedKey = (ULONG)ExtractBits64((ULONGLONG)Key, Table->Table->DownsizeBitmap);
    return Table->Table->Vtbl->Index(Table->Table, DownsizedKey, (PULONG)Index);
}

PH_ONLINE_JIT_API
int32_t
PhOnlineJitIndex32x4(
    PH_ONLINE_JIT_TABLE *Table,
    const uint32_t *Keys,
    uint32_t *Indexes
    )
{
    HRESULT Result;
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit;

    if (!ARGUMENT_PRESENT(Table) ||
        !ARGUMENT_PRESENT(Keys) ||
        !ARGUMENT_PRESENT(Indexes)) {
        return E_INVALIDARG;
    }

    Result = PhEnsureOnlineJitInterface(Table);
    if (FAILED(Result)) {
        return Result;
    }

    Jit = Table->JitInterface;
    return Jit->Vtbl->Index32x4(
        Jit,
        (ULONG)Keys[0],
        (ULONG)Keys[1],
        (ULONG)Keys[2],
        (ULONG)Keys[3],
        (PULONG)&Indexes[0],
        (PULONG)&Indexes[1],
        (PULONG)&Indexes[2],
        (PULONG)&Indexes[3]
    );
}

PH_ONLINE_JIT_API
int32_t
PhOnlineJitIndex32x8(
    PH_ONLINE_JIT_TABLE *Table,
    const uint32_t *Keys,
    uint32_t *Indexes
    )
{
    HRESULT Result;
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit;

    if (!ARGUMENT_PRESENT(Table) ||
        !ARGUMENT_PRESENT(Keys) ||
        !ARGUMENT_PRESENT(Indexes)) {
        return E_INVALIDARG;
    }

    Result = PhEnsureOnlineJitInterface(Table);
    if (FAILED(Result)) {
        return Result;
    }

    Jit = Table->JitInterface;
    return Jit->Vtbl->Index32x8(
        Jit,
        (ULONG)Keys[0],
        (ULONG)Keys[1],
        (ULONG)Keys[2],
        (ULONG)Keys[3],
        (ULONG)Keys[4],
        (ULONG)Keys[5],
        (ULONG)Keys[6],
        (ULONG)Keys[7],
        (PULONG)&Indexes[0],
        (PULONG)&Indexes[1],
        (PULONG)&Indexes[2],
        (PULONG)&Indexes[3],
        (PULONG)&Indexes[4],
        (PULONG)&Indexes[5],
        (PULONG)&Indexes[6],
        (PULONG)&Indexes[7]
    );
}

PH_ONLINE_JIT_API
int32_t
PhOnlineJitIndex32x16(
    PH_ONLINE_JIT_TABLE *Table,
    const uint32_t *Keys,
    uint32_t *Indexes
    )
{
    HRESULT Result;
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit;

    if (!ARGUMENT_PRESENT(Table) ||
        !ARGUMENT_PRESENT(Keys) ||
        !ARGUMENT_PRESENT(Indexes)) {
        return E_INVALIDARG;
    }

    Result = PhEnsureOnlineJitInterface(Table);
    if (FAILED(Result)) {
        return Result;
    }

    Jit = Table->JitInterface;
    return Jit->Vtbl->Index32x16(
        Jit,
        (ULONG)Keys[0],
        (ULONG)Keys[1],
        (ULONG)Keys[2],
        (ULONG)Keys[3],
        (ULONG)Keys[4],
        (ULONG)Keys[5],
        (ULONG)Keys[6],
        (ULONG)Keys[7],
        (ULONG)Keys[8],
        (ULONG)Keys[9],
        (ULONG)Keys[10],
        (ULONG)Keys[11],
        (ULONG)Keys[12],
        (ULONG)Keys[13],
        (ULONG)Keys[14],
        (ULONG)Keys[15],
        (PULONG)&Indexes[0],
        (PULONG)&Indexes[1],
        (PULONG)&Indexes[2],
        (PULONG)&Indexes[3],
        (PULONG)&Indexes[4],
        (PULONG)&Indexes[5],
        (PULONG)&Indexes[6],
        (PULONG)&Indexes[7],
        (PULONG)&Indexes[8],
        (PULONG)&Indexes[9],
        (PULONG)&Indexes[10],
        (PULONG)&Indexes[11],
        (PULONG)&Indexes[12],
        (PULONG)&Indexes[13],
        (PULONG)&Indexes[14],
        (PULONG)&Indexes[15]
    );
}

PH_ONLINE_JIT_API
void
PhOnlineJitReleaseTable(
    PH_ONLINE_JIT_TABLE *Table
    )
{
    if (!Table) {
        return;
    }

    PhReleaseOnlineJitInterface(Table);

    if (Table->Table) {
        Table->Table->Vtbl->Release(Table->Table);
        Table->Table = NULL;
    }

    free(Table);
}
