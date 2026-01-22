/*++

Copyright (c) 2025 Trent Nelson <trent@trent.me>

Module Name:

    ChmOnline01.c

Abstract:

    This module implements the online JIT compilation routines for CHM01
    perfect hash tables.

--*/

#include "stdafx.h"

PERFECT_HASH_TABLE_COMPILE_JIT PerfectHashTableCompileJit;
PERFECT_HASH_TABLE_JIT_RUNDOWN PerfectHashTableJitRundown;

typedef
ULONG
(PH_JIT_INDEX_FUNCTION)(
    _In_ ULONG Key
    );
typedef PH_JIT_INDEX_FUNCTION *PPH_JIT_INDEX_FUNCTION;

typedef
ULONG
(PH_JIT_INDEX64_FUNCTION)(
    _In_ ULONGLONG Key
    );
typedef PH_JIT_INDEX64_FUNCTION *PPH_JIT_INDEX64_FUNCTION;

typedef
VOID
(PH_JIT_INDEX2_FUNCTION)(
    _In_ ULONG Key1,
    _In_ ULONG Key2,
    _Out_ PULONG Index1,
    _Out_ PULONG Index2
    );
typedef PH_JIT_INDEX2_FUNCTION *PPH_JIT_INDEX2_FUNCTION;

typedef
VOID
(PH_JIT_INDEX4_FUNCTION)(
    _In_ ULONG Key1,
    _In_ ULONG Key2,
    _In_ ULONG Key3,
    _In_ ULONG Key4,
    _Out_ PULONG Index1,
    _Out_ PULONG Index2,
    _Out_ PULONG Index3,
    _Out_ PULONG Index4
    );
typedef PH_JIT_INDEX4_FUNCTION *PPH_JIT_INDEX4_FUNCTION;

typedef
VOID
(PH_JIT_INDEX8_FUNCTION)(
    _In_ ULONG Key1,
    _In_ ULONG Key2,
    _In_ ULONG Key3,
    _In_ ULONG Key4,
    _In_ ULONG Key5,
    _In_ ULONG Key6,
    _In_ ULONG Key7,
    _In_ ULONG Key8,
    _Out_ PULONG Index1,
    _Out_ PULONG Index2,
    _Out_ PULONG Index3,
    _Out_ PULONG Index4,
    _Out_ PULONG Index5,
    _Out_ PULONG Index6,
    _Out_ PULONG Index7,
    _Out_ PULONG Index8
    );
typedef PH_JIT_INDEX8_FUNCTION *PPH_JIT_INDEX8_FUNCTION;

typedef
VOID
(PH_JIT_INDEX2_64_FUNCTION)(
    _In_ ULONGLONG Key1,
    _In_ ULONGLONG Key2,
    _Out_ PULONG Index1,
    _Out_ PULONG Index2
    );
typedef PH_JIT_INDEX2_64_FUNCTION *PPH_JIT_INDEX2_64_FUNCTION;

typedef
VOID
(PH_JIT_INDEX4_64_FUNCTION)(
    _In_ ULONGLONG Key1,
    _In_ ULONGLONG Key2,
    _In_ ULONGLONG Key3,
    _In_ ULONGLONG Key4,
    _Out_ PULONG Index1,
    _Out_ PULONG Index2,
    _Out_ PULONG Index3,
    _Out_ PULONG Index4
    );
typedef PH_JIT_INDEX4_64_FUNCTION *PPH_JIT_INDEX4_64_FUNCTION;

typedef
VOID
(PH_JIT_INDEX8_64_FUNCTION)(
    _In_ ULONGLONG Key1,
    _In_ ULONGLONG Key2,
    _In_ ULONGLONG Key3,
    _In_ ULONGLONG Key4,
    _In_ ULONGLONG Key5,
    _In_ ULONGLONG Key6,
    _In_ ULONGLONG Key7,
    _In_ ULONGLONG Key8,
    _Out_ PULONG Index1,
    _Out_ PULONG Index2,
    _Out_ PULONG Index3,
    _Out_ PULONG Index4,
    _Out_ PULONG Index5,
    _Out_ PULONG Index6,
    _Out_ PULONG Index7,
    _Out_ PULONG Index8
    );
typedef PH_JIT_INDEX8_64_FUNCTION *PPH_JIT_INDEX8_64_FUNCTION;

PERFECT_HASH_TABLE_INDEX PerfectHashTableIndexJit;
PERFECT_HASH_TABLE_QUERY_INTERFACE PerfectHashTableQueryInterfaceJit;

PERFECT_HASH_TABLE_JIT_INTERFACE_QUERY_INTERFACE
    PerfectHashTableJitInterfaceQueryInterface;
PERFECT_HASH_TABLE_JIT_INTERFACE_ADD_REF
    PerfectHashTableJitInterfaceAddRef;
PERFECT_HASH_TABLE_JIT_INTERFACE_RELEASE
    PerfectHashTableJitInterfaceRelease;
PERFECT_HASH_TABLE_JIT_INTERFACE_CREATE_INSTANCE
    PerfectHashTableJitInterfaceCreateInstance;
PERFECT_HASH_TABLE_JIT_INTERFACE_LOCK_SERVER
    PerfectHashTableJitInterfaceLockServer;
PERFECT_HASH_TABLE_JIT_INDEX PerfectHashTableJitInterfaceIndex;
PERFECT_HASH_TABLE_JIT_INDEX64 PerfectHashTableJitInterfaceIndex64;
PERFECT_HASH_TABLE_JIT_INDEX2 PerfectHashTableJitInterfaceIndex2;
PERFECT_HASH_TABLE_JIT_INDEX4 PerfectHashTableJitInterfaceIndex4;
PERFECT_HASH_TABLE_JIT_INDEX2_64 PerfectHashTableJitInterfaceIndex2_64;
PERFECT_HASH_TABLE_JIT_INDEX4_64 PerfectHashTableJitInterfaceIndex4_64;
PERFECT_HASH_TABLE_JIT_INDEX8 PerfectHashTableJitInterfaceIndex8;
PERFECT_HASH_TABLE_JIT_INDEX8_64 PerfectHashTableJitInterfaceIndex8_64;

static const PERFECT_HASH_TABLE_JIT_INTERFACE_VTBL
    PerfectHashTableJitInterfaceVtbl = {
    &PerfectHashTableJitInterfaceQueryInterface,
    &PerfectHashTableJitInterfaceAddRef,
    &PerfectHashTableJitInterfaceRelease,
    &PerfectHashTableJitInterfaceCreateInstance,
    &PerfectHashTableJitInterfaceLockServer,
    &PerfectHashTableJitInterfaceIndex,
    &PerfectHashTableJitInterfaceIndex64,
    &PerfectHashTableJitInterfaceIndex2,
    &PerfectHashTableJitInterfaceIndex4,
    &PerfectHashTableJitInterfaceIndex2_64,
    &PerfectHashTableJitInterfaceIndex4_64,
    &PerfectHashTableJitInterfaceIndex8,
    &PerfectHashTableJitInterfaceIndex8_64,
};

#if defined(PH_HAS_LLVM)

#include <llvm-c/Analysis.h>
#include <llvm-c/Core.h>
#include <llvm-c/ExecutionEngine.h>
#include <llvm-c/Target.h>
#include <llvm-c/TargetMachine.h>

FORCEINLINE
BOOLEAN
IsSupportedJitHashFunctionId(
    _In_ PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId
    )
{
#pragma warning(push)
#pragma warning(disable: 4061)
    switch (HashFunctionId) {
        case PerfectHashHashMultiplyShiftRFunctionId:
        case PerfectHashHashMultiplyShiftRMultiplyFunctionId:
        case PerfectHashHashMultiplyShiftR2FunctionId:
        case PerfectHashHashMultiplyShiftRXFunctionId:
        case PerfectHashHashMultiplyShiftLRFunctionId:
        case PerfectHashHashMulshrolate1RXFunctionId:
        case PerfectHashHashMulshrolate2RXFunctionId:
        case PerfectHashHashMulshrolate3RXFunctionId:
        case PerfectHashHashMulshrolate4RXFunctionId:
            return TRUE;
        default:
            return FALSE;
    }
#pragma warning(pop)
}

FORCEINLINE
LLVMValueRef
BuildSplatVectorConstant(
    _In_ LLVMTypeRef VectorType,
    _In_ LLVMValueRef ScalarConstant,
    _In_ ULONG Lanes
    )
{
    LLVMValueRef Values[8];
    ULONG Index;

    for (Index = 0; Index < Lanes; Index++) {
        Values[Index] = ScalarConstant;
    }

    return LLVMConstVector(Values, Lanes);
}

FORCEINLINE
LLVMValueRef
BuildConstIntLike(
    _In_ LLVMValueRef Like,
    _In_ ULONG Value
    )
{
    LLVMTypeRef Type;

    Type = LLVMTypeOf(Like);

    if (LLVMGetTypeKind(Type) == LLVMVectorTypeKind) {
        LLVMTypeRef ElementType;
        LLVMValueRef Element;
        LLVMValueRef Values[8];
        ULONG Lanes;
        ULONG Index;

        ElementType = LLVMGetElementType(Type);
        Element = LLVMConstInt(ElementType, Value, FALSE);
        Lanes = LLVMGetVectorSize(Type);

        for (Index = 0; Index < Lanes; Index++) {
            Values[Index] = Element;
        }

        return LLVMConstVector(Values, Lanes);
    }

    return LLVMConstInt(Type, Value, FALSE);
}

FORCEINLINE
LLVMValueRef
BuildRotateRight32(
    _In_ LLVMBuilderRef Builder,
    _In_ LLVMValueRef Value,
    _In_ LLVMValueRef Shift
    )
{
    LLVMTypeRef Type;
    LLVMValueRef Mask;
    LLVMValueRef ShiftMasked;
    LLVMValueRef ShiftInverse;
    LLVMValueRef ShiftInverseMasked;
    LLVMValueRef Right;
    LLVMValueRef Left;

    Type = LLVMTypeOf(Value);
    Mask = BuildConstIntLike(Value, 31);
    ShiftMasked = LLVMBuildAnd(Builder, Shift, Mask, "shift");

    ShiftInverse = LLVMBuildSub(Builder,
                                BuildConstIntLike(Value, 32),
                                ShiftMasked,
                                "shift_inv");

    ShiftInverseMasked = LLVMBuildAnd(Builder,
                                      ShiftInverse,
                                      Mask,
                                      "shift_inv_masked");

    Right = LLVMBuildLShr(Builder, Value, ShiftMasked, "rotr.right");
    Left = LLVMBuildShl(Builder, Value, ShiftInverseMasked, "rotr.left");

    return LLVMBuildOr(Builder, Right, Left, "rotr");
}

FORCEINLINE
LLVMValueRef
BuildTableDataLoad(
    _In_ LLVMBuilderRef Builder,
    _In_ LLVMTypeRef ElementType,
    _In_ LLVMTypeRef IndexType,
    _In_ LLVMValueRef BasePointer,
    _In_ LLVMValueRef Index
    )
{
    LLVMValueRef Index64;
    LLVMValueRef Ptr;

    Index64 = LLVMBuildZExt(Builder, Index, IndexType, "idx64");
    Ptr = LLVMBuildInBoundsGEP2(Builder,
                                ElementType,
                                BasePointer,
                                &Index64,
                                1,
                                "table_ptr");

    return LLVMBuildLoad2(Builder, ElementType, Ptr, "table_value");
}

typedef struct _CHM01_JIT_CONTEXT {
    LLVMContextRef Context;
    LLVMModuleRef Module;
    LLVMBuilderRef Builder;
    LLVMTypeRef I1;
    LLVMTypeRef I16;
    LLVMTypeRef I32;
    LLVMTypeRef I64;
    LLVMTypeRef I32Ptr;
    LLVMTypeRef TableElementType;
    LLVMValueRef Seed1Const;
    LLVMValueRef Seed2Const;
    LLVMValueRef Seed3Byte1Const;
    LLVMValueRef Seed3Byte2Const;
    LLVMValueRef Seed3Byte3Const;
    LLVMValueRef Seed3Byte4Const;
    LLVMValueRef Seed4Const;
    LLVMValueRef Seed5Const;
    LLVMValueRef HashMaskConst;
    LLVMValueRef IndexMaskConst;
    LLVMValueRef TableDataConst;
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId;
    BOOLEAN UseAssigned16;
    BYTE Padding1[3];
} CHM01_JIT_CONTEXT;
typedef CHM01_JIT_CONTEXT *PCHM01_JIT_CONTEXT;

FORCEINLINE
LLVMValueRef
BuildTableDataLoadAndExtend(
    _In_ LLVMBuilderRef Builder,
    _In_ LLVMTypeRef ElementType,
    _In_ LLVMTypeRef IndexType,
    _In_ LLVMTypeRef ResultType,
    _In_ LLVMValueRef BasePointer,
    _In_ LLVMValueRef Index
    )
{
    LLVMValueRef Value;

    Value = BuildTableDataLoad(Builder,
                               ElementType,
                               IndexType,
                               BasePointer,
                               Index);

    if (ElementType != ResultType) {
        Value = LLVMBuildZExt(Builder, Value, ResultType, "table_value_ext");
    }

    return Value;
}

static
LLVMValueRef
BuildDownsizeKeyFunction(
    _In_ LLVMModuleRef Module,
    _In_ LLVMBuilderRef Builder,
    _In_ LLVMContextRef Context,
    _In_ ULONGLONG DownsizeBitmap,
    _In_ BOOLEAN Contiguous,
    _In_ BYTE TrailingZeros,
    _In_ ULONGLONG ShiftedMask
    )
{
    LLVMTypeRef I32;
    LLVMTypeRef I64;
    LLVMTypeRef FunctionType;
    LLVMValueRef Function;
    LLVMValueRef Key;
    LLVMBasicBlockRef Entry;

    I32 = LLVMInt32TypeInContext(Context);
    I64 = LLVMInt64TypeInContext(Context);

    FunctionType = LLVMFunctionType(I32, &I64, 1, FALSE);
    Function = LLVMAddFunction(Module,
                               "PerfectHashJitDownsizeKey",
                               FunctionType);

    Entry = LLVMAppendBasicBlockInContext(Context, Function, "entry");
    LLVMPositionBuilderAtEnd(Builder, Entry);

    Key = LLVMGetParam(Function, 0);

    if (Contiguous) {
        LLVMValueRef ShiftConst;
        LLVMValueRef MaskConst;
        LLVMValueRef Shifted;
        LLVMValueRef Masked;
        LLVMValueRef Result;

        ShiftConst = LLVMConstInt(I64, TrailingZeros, FALSE);
        MaskConst = LLVMConstInt(I64, ShiftedMask, FALSE);

        Shifted = LLVMBuildLShr(Builder, Key, ShiftConst, "key_shift");
        Masked = LLVMBuildAnd(Builder, Shifted, MaskConst, "key_mask");
        Result = LLVMBuildTrunc(Builder, Masked, I32, "key_downsized");

        LLVMBuildRet(Builder, Result);
        return Function;
    }

    //
    // Non-contiguous downsize bitmap; implement the equivalent of ExtractBits64
    // via a simple loop.
    //

    {
        LLVMValueRef One;
        LLVMValueRef Zero;
        LLVMValueRef ResultAlloca;
        LLVMValueRef BitPosAlloca;
        LLVMValueRef MaskAlloca;
        LLVMValueRef ValueAlloca;
        LLVMValueRef MaskConst;
        LLVMBasicBlockRef Loop;
        LLVMBasicBlockRef Body;
        LLVMBasicBlockRef SetBit;
        LLVMBasicBlockRef SkipSetBit;
        LLVMBasicBlockRef Done;

        One = LLVMConstInt(I64, 1, FALSE);
        Zero = LLVMConstInt(I64, 0, FALSE);
        MaskConst = LLVMConstInt(I64, DownsizeBitmap, FALSE);

        ResultAlloca = LLVMBuildAlloca(Builder, I64, "downsize.result");
        BitPosAlloca = LLVMBuildAlloca(Builder, I64, "downsize.bitpos");
        MaskAlloca = LLVMBuildAlloca(Builder, I64, "downsize.mask");
        ValueAlloca = LLVMBuildAlloca(Builder, I64, "downsize.value");

        LLVMBuildStore(Builder, Zero, ResultAlloca);
        LLVMBuildStore(Builder, Zero, BitPosAlloca);
        LLVMBuildStore(Builder, MaskConst, MaskAlloca);
        LLVMBuildStore(Builder, Key, ValueAlloca);

        Loop = LLVMAppendBasicBlockInContext(Context, Function, "downsize.loop");
        Body = LLVMAppendBasicBlockInContext(Context, Function, "downsize.body");
        SetBit = LLVMAppendBasicBlockInContext(Context,
                                               Function,
                                               "downsize.setbit");
        SkipSetBit = LLVMAppendBasicBlockInContext(
            Context,
            Function,
            "downsize.skipsetbit");
        Done = LLVMAppendBasicBlockInContext(Context, Function, "downsize.done");

        LLVMBuildBr(Builder, Loop);

        //
        // Loop condition: while (mask != 0).
        //

        LLVMPositionBuilderAtEnd(Builder, Loop);
        {
            LLVMValueRef MaskValue;
            LLVMValueRef MaskNotZero;

            MaskValue = LLVMBuildLoad2(Builder,
                                       I64,
                                       MaskAlloca,
                                       "mask_value");

            MaskNotZero = LLVMBuildICmp(Builder,
                                        LLVMIntNE,
                                        MaskValue,
                                        Zero,
                                        "mask_not_zero");

            LLVMBuildCondBr(Builder, MaskNotZero, Body, Done);
        }

        //
        // Loop body.
        //

        LLVMPositionBuilderAtEnd(Builder, Body);
        {
            LLVMValueRef MaskValue;
            LLVMValueRef Value;
            LLVMValueRef MaskLsb;
            LLVMValueRef MaskHasBit;

            MaskValue = LLVMBuildLoad2(Builder,
                                       I64,
                                       MaskAlloca,
                                       "mask_value");
            Value = LLVMBuildLoad2(Builder,
                                   I64,
                                   ValueAlloca,
                                   "value");

            MaskLsb = LLVMBuildAnd(Builder, MaskValue, One, "mask_lsb");
            MaskHasBit = LLVMBuildICmp(Builder,
                                       LLVMIntNE,
                                       MaskLsb,
                                       Zero,
                                       "mask_has_bit");

            LLVMBuildCondBr(Builder, MaskHasBit, SetBit, SkipSetBit);
        }

        //
        // If the mask has a bit set, emit the corresponding bit.
        //

        LLVMPositionBuilderAtEnd(Builder, SetBit);
        {
            LLVMValueRef ResultValue;
            LLVMValueRef Value;
            LLVMValueRef ValueBit;
            LLVMValueRef BitPos;
            LLVMValueRef ShiftedBit;
            LLVMValueRef NewResult;
            LLVMValueRef NewBitPos;

            ResultValue = LLVMBuildLoad2(Builder,
                                         I64,
                                         ResultAlloca,
                                         "result");
            Value = LLVMBuildLoad2(Builder,
                                   I64,
                                   ValueAlloca,
                                   "value");

            ValueBit = LLVMBuildAnd(Builder, Value, One, "value_bit");
            BitPos = LLVMBuildLoad2(Builder,
                                    I64,
                                    BitPosAlloca,
                                    "bitpos");

            ShiftedBit = LLVMBuildShl(Builder,
                                      ValueBit,
                                      BitPos,
                                      "value_bit_shift");

            NewResult = LLVMBuildOr(Builder,
                                    ResultValue,
                                    ShiftedBit,
                                    "result_new");

            NewBitPos = LLVMBuildAdd(Builder,
                                     BitPos,
                                     One,
                                     "bitpos_new");

            LLVMBuildStore(Builder, NewResult, ResultAlloca);
            LLVMBuildStore(Builder, NewBitPos, BitPosAlloca);

            LLVMBuildBr(Builder, SkipSetBit);
        }

        //
        // Update mask/value and continue looping.
        //

        LLVMPositionBuilderAtEnd(Builder, SkipSetBit);
        {
            LLVMValueRef MaskValue;
            LLVMValueRef Value;
            LLVMValueRef MaskShifted;
            LLVMValueRef ValueShifted;

            MaskValue = LLVMBuildLoad2(Builder,
                                       I64,
                                       MaskAlloca,
                                       "mask_value");
            Value = LLVMBuildLoad2(Builder,
                                   I64,
                                   ValueAlloca,
                                   "value");

            MaskShifted = LLVMBuildLShr(Builder,
                                        MaskValue,
                                        One,
                                        "mask_shifted");
            ValueShifted = LLVMBuildLShr(Builder,
                                         Value,
                                         One,
                                         "value_shifted");

            LLVMBuildStore(Builder, MaskShifted, MaskAlloca);
            LLVMBuildStore(Builder, ValueShifted, ValueAlloca);

            LLVMBuildBr(Builder, Loop);
        }

        //
        // Done.  Return the result.
        //

        LLVMPositionBuilderAtEnd(Builder, Done);
        {
            LLVMValueRef ResultValue;
            LLVMValueRef Result;

            ResultValue = LLVMBuildLoad2(Builder,
                                         I64,
                                         ResultAlloca,
                                         "result");

            Result = LLVMBuildTrunc(Builder,
                                    ResultValue,
                                    I32,
                                    "key_downsized");

            LLVMBuildRet(Builder, Result);
        }

        return Function;
    }
}

static
LLVMValueRef
BuildChm01IndexFunction(
    _In_ PCHM01_JIT_CONTEXT Ctx,
    _In_ PCSTR Name,
    _In_ LLVMTypeRef KeyType,
    _In_opt_ LLVMValueRef DownsizeFunction
    )
{
    LLVMTypeRef FunctionType;
    LLVMValueRef Function;
    LLVMValueRef Key;
    LLVMValueRef Key32;
    LLVMValueRef Index;
    LLVMValueRef Vertex1;
    LLVMValueRef Vertex2;
    LLVMValueRef MaskedLow;
    LLVMValueRef MaskedHigh;

    FunctionType = LLVMFunctionType(Ctx->I32, &KeyType, 1, FALSE);
    Function = LLVMAddFunction(Ctx->Module, Name, FunctionType);

    LLVMPositionBuilderAtEnd(
        Ctx->Builder,
        LLVMAppendBasicBlockInContext(Ctx->Context, Function, "entry")
    );

    Key = LLVMGetParam(Function, 0);

    if (KeyType == Ctx->I64) {
        if (DownsizeFunction) {
            LLVMTypeRef DownsizeType;
            DownsizeType = LLVMFunctionType(Ctx->I32, &Ctx->I64, 1, FALSE);
            Key32 = LLVMBuildCall2(Ctx->Builder,
                                   DownsizeType,
                                   DownsizeFunction,
                                   &Key,
                                   1,
                                   "key_downsized");
        } else {
            Key32 = LLVMBuildTrunc(Ctx->Builder, Key, Ctx->I32, "key32");
        }
    } else {
        Key32 = Key;
    }

#pragma warning(push)
#pragma warning(disable: 4061)
    switch (Ctx->HashFunctionId) {

        case PerfectHashHashMultiplyShiftRFunctionId:
            Vertex1 = LLVMBuildMul(Ctx->Builder,
                                   Key32,
                                   Ctx->Seed1Const,
                                   "v1.mul");
            Vertex1 = LLVMBuildLShr(Ctx->Builder,
                                    Vertex1,
                                    Ctx->Seed3Byte1Const,
                                    "v1.shr");

            Vertex2 = LLVMBuildMul(Ctx->Builder,
                                   Key32,
                                   Ctx->Seed2Const,
                                   "v2.mul");
            Vertex2 = LLVMBuildLShr(Ctx->Builder,
                                    Vertex2,
                                    Ctx->Seed3Byte2Const,
                                    "v2.shr");

            MaskedLow = LLVMBuildAnd(Ctx->Builder,
                                     Vertex1,
                                     Ctx->HashMaskConst,
                                     "masked_low");

            MaskedHigh = LLVMBuildAnd(Ctx->Builder,
                                      Vertex2,
                                      Ctx->HashMaskConst,
                                      "masked_high");

            Vertex1 = BuildTableDataLoadAndExtend(Ctx->Builder,
                                                  Ctx->TableElementType,
                                                  Ctx->I64,
                                                  Ctx->I32,
                                                  Ctx->TableDataConst,
                                                  MaskedLow);

            Vertex2 = BuildTableDataLoadAndExtend(Ctx->Builder,
                                                  Ctx->TableElementType,
                                                  Ctx->I64,
                                                  Ctx->I32,
                                                  Ctx->TableDataConst,
                                                  MaskedHigh);
            break;

        case PerfectHashHashMultiplyShiftRMultiplyFunctionId:
            Vertex1 = LLVMBuildMul(Ctx->Builder,
                                   Key32,
                                   Ctx->Seed1Const,
                                   "v1.mul1");
            Vertex1 = LLVMBuildLShr(Ctx->Builder,
                                    Vertex1,
                                    Ctx->Seed3Byte1Const,
                                    "v1.shr1");
            Vertex1 = LLVMBuildMul(Ctx->Builder,
                                   Vertex1,
                                   Ctx->Seed2Const,
                                   "v1.mul2");

            Vertex2 = LLVMBuildMul(Ctx->Builder,
                                   Key32,
                                   Ctx->Seed4Const,
                                   "v2.mul1");
            Vertex2 = LLVMBuildLShr(Ctx->Builder,
                                    Vertex2,
                                    Ctx->Seed3Byte2Const,
                                    "v2.shr1");
            Vertex2 = LLVMBuildMul(Ctx->Builder,
                                   Vertex2,
                                   Ctx->Seed5Const,
                                   "v2.mul2");

            MaskedLow = LLVMBuildAnd(Ctx->Builder,
                                     Vertex1,
                                     Ctx->HashMaskConst,
                                     "masked_low");

            MaskedHigh = LLVMBuildAnd(Ctx->Builder,
                                      Vertex2,
                                      Ctx->HashMaskConst,
                                      "masked_high");

            Vertex1 = BuildTableDataLoadAndExtend(Ctx->Builder,
                                                  Ctx->TableElementType,
                                                  Ctx->I64,
                                                  Ctx->I32,
                                                  Ctx->TableDataConst,
                                                  MaskedLow);

            Vertex2 = BuildTableDataLoadAndExtend(Ctx->Builder,
                                                  Ctx->TableElementType,
                                                  Ctx->I64,
                                                  Ctx->I32,
                                                  Ctx->TableDataConst,
                                                  MaskedHigh);
            break;

        case PerfectHashHashMultiplyShiftR2FunctionId:
            Vertex1 = LLVMBuildMul(Ctx->Builder,
                                   Key32,
                                   Ctx->Seed1Const,
                                   "v1.mul1");
            Vertex1 = LLVMBuildLShr(Ctx->Builder,
                                    Vertex1,
                                    Ctx->Seed3Byte1Const,
                                    "v1.shr1");
            Vertex1 = LLVMBuildMul(Ctx->Builder,
                                   Vertex1,
                                   Ctx->Seed2Const,
                                   "v1.mul2");
            Vertex1 = LLVMBuildLShr(Ctx->Builder,
                                    Vertex1,
                                    Ctx->Seed3Byte2Const,
                                    "v1.shr2");

            Vertex2 = LLVMBuildMul(Ctx->Builder,
                                   Key32,
                                   Ctx->Seed4Const,
                                   "v2.mul1");
            Vertex2 = LLVMBuildLShr(Ctx->Builder,
                                    Vertex2,
                                    Ctx->Seed3Byte3Const,
                                    "v2.shr1");
            Vertex2 = LLVMBuildMul(Ctx->Builder,
                                   Vertex2,
                                   Ctx->Seed5Const,
                                   "v2.mul2");
            Vertex2 = LLVMBuildLShr(Ctx->Builder,
                                    Vertex2,
                                    Ctx->Seed3Byte4Const,
                                    "v2.shr2");

            MaskedLow = LLVMBuildAnd(Ctx->Builder,
                                     Vertex1,
                                     Ctx->HashMaskConst,
                                     "masked_low");

            MaskedHigh = LLVMBuildAnd(Ctx->Builder,
                                      Vertex2,
                                      Ctx->HashMaskConst,
                                      "masked_high");

            Vertex1 = BuildTableDataLoadAndExtend(Ctx->Builder,
                                                  Ctx->TableElementType,
                                                  Ctx->I64,
                                                  Ctx->I32,
                                                  Ctx->TableDataConst,
                                                  MaskedLow);

            Vertex2 = BuildTableDataLoadAndExtend(Ctx->Builder,
                                                  Ctx->TableElementType,
                                                  Ctx->I64,
                                                  Ctx->I32,
                                                  Ctx->TableDataConst,
                                                  MaskedHigh);
            break;

        case PerfectHashHashMultiplyShiftRXFunctionId:
            Vertex1 = LLVMBuildMul(Ctx->Builder,
                                   Key32,
                                   Ctx->Seed1Const,
                                   "v1.mul");
            Vertex1 = LLVMBuildLShr(Ctx->Builder,
                                    Vertex1,
                                    Ctx->Seed3Byte1Const,
                                    "v1.shr");

            Vertex2 = LLVMBuildMul(Ctx->Builder,
                                   Key32,
                                   Ctx->Seed2Const,
                                   "v2.mul");
            Vertex2 = LLVMBuildLShr(Ctx->Builder,
                                    Vertex2,
                                    Ctx->Seed3Byte1Const,
                                    "v2.shr");

            Vertex1 = BuildTableDataLoadAndExtend(Ctx->Builder,
                                                  Ctx->TableElementType,
                                                  Ctx->I64,
                                                  Ctx->I32,
                                                  Ctx->TableDataConst,
                                                  Vertex1);

            Vertex2 = BuildTableDataLoadAndExtend(Ctx->Builder,
                                                  Ctx->TableElementType,
                                                  Ctx->I64,
                                                  Ctx->I32,
                                                  Ctx->TableDataConst,
                                                  Vertex2);
            break;

        case PerfectHashHashMultiplyShiftLRFunctionId:
            Vertex1 = LLVMBuildMul(Ctx->Builder,
                                   Key32,
                                   Ctx->Seed1Const,
                                   "v1.mul");
            Vertex1 = LLVMBuildShl(Ctx->Builder,
                                   Vertex1,
                                   Ctx->Seed3Byte1Const,
                                   "v1.shl");

            Vertex2 = LLVMBuildMul(Ctx->Builder,
                                   Key32,
                                   Ctx->Seed2Const,
                                   "v2.mul");
            Vertex2 = LLVMBuildLShr(Ctx->Builder,
                                    Vertex2,
                                    Ctx->Seed3Byte2Const,
                                    "v2.shr");

            MaskedLow = LLVMBuildAnd(Ctx->Builder,
                                     Vertex1,
                                     Ctx->HashMaskConst,
                                     "masked_low");

            MaskedHigh = LLVMBuildAnd(Ctx->Builder,
                                      Vertex2,
                                      Ctx->HashMaskConst,
                                      "masked_high");

            Vertex1 = BuildTableDataLoadAndExtend(Ctx->Builder,
                                                  Ctx->TableElementType,
                                                  Ctx->I64,
                                                  Ctx->I32,
                                                  Ctx->TableDataConst,
                                                  MaskedLow);

            Vertex2 = BuildTableDataLoadAndExtend(Ctx->Builder,
                                                  Ctx->TableElementType,
                                                  Ctx->I64,
                                                  Ctx->I32,
                                                  Ctx->TableDataConst,
                                                  MaskedHigh);
            break;

        case PerfectHashHashMulshrolate1RXFunctionId:
            Vertex1 = LLVMBuildMul(Ctx->Builder,
                                   Key32,
                                   Ctx->Seed1Const,
                                   "v1.mul");
            Vertex1 = BuildRotateRight32(Ctx->Builder,
                                         Vertex1,
                                         Ctx->Seed3Byte2Const);
            Vertex1 = LLVMBuildLShr(Ctx->Builder,
                                    Vertex1,
                                    Ctx->Seed3Byte1Const,
                                    "v1.shr");

            Vertex2 = LLVMBuildMul(Ctx->Builder,
                                   Key32,
                                   Ctx->Seed2Const,
                                   "v2.mul");
            Vertex2 = LLVMBuildLShr(Ctx->Builder,
                                    Vertex2,
                                    Ctx->Seed3Byte1Const,
                                    "v2.shr");

            Vertex1 = BuildTableDataLoadAndExtend(Ctx->Builder,
                                                  Ctx->TableElementType,
                                                  Ctx->I64,
                                                  Ctx->I32,
                                                  Ctx->TableDataConst,
                                                  Vertex1);

            Vertex2 = BuildTableDataLoadAndExtend(Ctx->Builder,
                                                  Ctx->TableElementType,
                                                  Ctx->I64,
                                                  Ctx->I32,
                                                  Ctx->TableDataConst,
                                                  Vertex2);
            break;

        case PerfectHashHashMulshrolate2RXFunctionId:
            Vertex1 = LLVMBuildMul(Ctx->Builder,
                                   Key32,
                                   Ctx->Seed1Const,
                                   "v1.mul");
            Vertex1 = BuildRotateRight32(Ctx->Builder,
                                         Vertex1,
                                         Ctx->Seed3Byte2Const);
            Vertex1 = LLVMBuildLShr(Ctx->Builder,
                                    Vertex1,
                                    Ctx->Seed3Byte1Const,
                                    "v1.shr");

            Vertex2 = LLVMBuildMul(Ctx->Builder,
                                   Key32,
                                   Ctx->Seed2Const,
                                   "v2.mul");
            Vertex2 = BuildRotateRight32(Ctx->Builder,
                                         Vertex2,
                                         Ctx->Seed3Byte3Const);
            Vertex2 = LLVMBuildLShr(Ctx->Builder,
                                    Vertex2,
                                    Ctx->Seed3Byte1Const,
                                    "v2.shr");

            Vertex1 = BuildTableDataLoadAndExtend(Ctx->Builder,
                                                  Ctx->TableElementType,
                                                  Ctx->I64,
                                                  Ctx->I32,
                                                  Ctx->TableDataConst,
                                                  Vertex1);

            Vertex2 = BuildTableDataLoadAndExtend(Ctx->Builder,
                                                  Ctx->TableElementType,
                                                  Ctx->I64,
                                                  Ctx->I32,
                                                  Ctx->TableDataConst,
                                                  Vertex2);
            break;

        case PerfectHashHashMulshrolate3RXFunctionId:
            Vertex1 = LLVMBuildMul(Ctx->Builder,
                                   Key32,
                                   Ctx->Seed1Const,
                                   "v1.mul1");
            Vertex1 = BuildRotateRight32(Ctx->Builder,
                                         Vertex1,
                                         Ctx->Seed3Byte2Const);
            Vertex1 = LLVMBuildMul(Ctx->Builder,
                                   Vertex1,
                                   Ctx->Seed4Const,
                                   "v1.mul2");
            Vertex1 = LLVMBuildLShr(Ctx->Builder,
                                    Vertex1,
                                    Ctx->Seed3Byte1Const,
                                    "v1.shr");

            Vertex2 = LLVMBuildMul(Ctx->Builder,
                                   Key32,
                                   Ctx->Seed2Const,
                                   "v2.mul");
            Vertex2 = BuildRotateRight32(Ctx->Builder,
                                         Vertex2,
                                         Ctx->Seed3Byte3Const);
            Vertex2 = LLVMBuildLShr(Ctx->Builder,
                                    Vertex2,
                                    Ctx->Seed3Byte1Const,
                                    "v2.shr");

            Vertex1 = BuildTableDataLoadAndExtend(Ctx->Builder,
                                                  Ctx->TableElementType,
                                                  Ctx->I64,
                                                  Ctx->I32,
                                                  Ctx->TableDataConst,
                                                  Vertex1);

            Vertex2 = BuildTableDataLoadAndExtend(Ctx->Builder,
                                                  Ctx->TableElementType,
                                                  Ctx->I64,
                                                  Ctx->I32,
                                                  Ctx->TableDataConst,
                                                  Vertex2);
            break;

        case PerfectHashHashMulshrolate4RXFunctionId:
            Vertex1 = LLVMBuildMul(Ctx->Builder,
                                   Key32,
                                   Ctx->Seed1Const,
                                   "v1.mul1");
            Vertex1 = BuildRotateRight32(Ctx->Builder,
                                         Vertex1,
                                         Ctx->Seed3Byte2Const);
            Vertex1 = LLVMBuildMul(Ctx->Builder,
                                   Vertex1,
                                   Ctx->Seed4Const,
                                   "v1.mul2");
            Vertex1 = LLVMBuildLShr(Ctx->Builder,
                                    Vertex1,
                                    Ctx->Seed3Byte1Const,
                                    "v1.shr");

            Vertex2 = LLVMBuildMul(Ctx->Builder,
                                   Key32,
                                   Ctx->Seed2Const,
                                   "v2.mul1");
            Vertex2 = BuildRotateRight32(Ctx->Builder,
                                         Vertex2,
                                         Ctx->Seed3Byte3Const);
            Vertex2 = LLVMBuildMul(Ctx->Builder,
                                   Vertex2,
                                   Ctx->Seed5Const,
                                   "v2.mul2");
            Vertex2 = LLVMBuildLShr(Ctx->Builder,
                                    Vertex2,
                                    Ctx->Seed3Byte1Const,
                                    "v2.shr");

            Vertex1 = BuildTableDataLoadAndExtend(Ctx->Builder,
                                                  Ctx->TableElementType,
                                                  Ctx->I64,
                                                  Ctx->I32,
                                                  Ctx->TableDataConst,
                                                  Vertex1);

            Vertex2 = BuildTableDataLoadAndExtend(Ctx->Builder,
                                                  Ctx->TableElementType,
                                                  Ctx->I64,
                                                  Ctx->I32,
                                                  Ctx->TableDataConst,
                                                  Vertex2);
            break;

        default:
            return NULL;
    }
#pragma warning(pop)

    Index = LLVMBuildAdd(Ctx->Builder, Vertex1, Vertex2, "index_add");
    Index = LLVMBuildAnd(Ctx->Builder,
                         Index,
                         Ctx->IndexMaskConst,
                         "index_masked");
    LLVMBuildRet(Ctx->Builder, Index);

    return Function;
}

static
LLVMValueRef
BuildChm01Index2Function(
    _In_ PCHM01_JIT_CONTEXT Ctx,
    _In_ PCSTR Name,
    _In_ LLVMValueRef IndexFunction,
    _In_ LLVMTypeRef IndexFunctionType,
    _In_ LLVMTypeRef KeyType
    )
{
    LLVMTypeRef FunctionType;
    LLVMTypeRef Params[4];
    LLVMValueRef Function;
    LLVMValueRef Key1;
    LLVMValueRef Key2;
    LLVMValueRef Index1Ptr;
    LLVMValueRef Index2Ptr;
    LLVMValueRef Index1;
    LLVMValueRef Index2;

    Params[0] = KeyType;
    Params[1] = KeyType;
    Params[2] = Ctx->I32Ptr;
    Params[3] = Ctx->I32Ptr;

    FunctionType = LLVMFunctionType(LLVMVoidTypeInContext(Ctx->Context),
                                    Params,
                                    ARRAYSIZE(Params),
                                    FALSE);

    Function = LLVMAddFunction(Ctx->Module, Name, FunctionType);

    LLVMPositionBuilderAtEnd(
        Ctx->Builder,
        LLVMAppendBasicBlockInContext(Ctx->Context, Function, "entry")
    );

    Key1 = LLVMGetParam(Function, 0);
    Key2 = LLVMGetParam(Function, 1);
    Index1Ptr = LLVMGetParam(Function, 2);
    Index2Ptr = LLVMGetParam(Function, 3);

    Index1 = LLVMBuildCall2(Ctx->Builder,
                            IndexFunctionType,
                            IndexFunction,
                            &Key1,
                            1,
                            "index1");

    Index2 = LLVMBuildCall2(Ctx->Builder,
                            IndexFunctionType,
                            IndexFunction,
                            &Key2,
                            1,
                            "index2");

    LLVMBuildStore(Ctx->Builder, Index1, Index1Ptr);
    LLVMBuildStore(Ctx->Builder, Index2, Index2Ptr);
    LLVMBuildRetVoid(Ctx->Builder);

    return Function;
}

static
LLVMValueRef
BuildChm01IndexVectorFunction(
    _In_ PCHM01_JIT_CONTEXT Ctx,
    _In_ PCSTR Name,
    _In_ LLVMTypeRef KeyType,
    _In_opt_ LLVMValueRef DownsizeFunction,
    _In_ ULONG Lanes
    )
{
    LLVMTypeRef FunctionType;
    LLVMTypeRef VecType;
    LLVMTypeRef Params[16];
    LLVMValueRef Function;
    LLVMValueRef KeyParams[8];
    LLVMValueRef IndexPtrs[8];
    LLVMValueRef KeyVec;
    LLVMValueRef Seed1Vec;
    LLVMValueRef Seed2Vec;
    LLVMValueRef Seed3Byte1Vec;
    LLVMValueRef Seed3Byte2Vec;
    LLVMValueRef Seed3Byte3Vec;
    LLVMValueRef Seed3Byte4Vec;
    LLVMValueRef Seed4Vec;
    LLVMValueRef Seed5Vec;
    LLVMValueRef HashMaskVec;
    LLVMValueRef Vertex1Vec;
    LLVMValueRef Vertex2Vec;
    LLVMValueRef IndexVec1;
    LLVMValueRef IndexVec2;
    LLVMValueRef LaneConst;
    LLVMValueRef Key;
    LLVMValueRef Key32;
    LLVMValueRef Index1;
    LLVMValueRef Index2;
    LLVMValueRef Vertex1;
    LLVMValueRef Vertex2;
    LLVMValueRef Index;
    BOOLEAN UseHashMask;
    ULONG IndexLane;

    if (Lanes != 2 && Lanes != 4 && Lanes != 8) {
        return NULL;
    }

    for (IndexLane = 0; IndexLane < Lanes; IndexLane++) {
        Params[IndexLane] = KeyType;
    }
    for (IndexLane = 0; IndexLane < Lanes; IndexLane++) {
        Params[Lanes + IndexLane] = Ctx->I32Ptr;
    }

    FunctionType = LLVMFunctionType(LLVMVoidTypeInContext(Ctx->Context),
                                    Params,
                                    Lanes * 2,
                                    FALSE);

    Function = LLVMAddFunction(Ctx->Module, Name, FunctionType);

    LLVMPositionBuilderAtEnd(
        Ctx->Builder,
        LLVMAppendBasicBlockInContext(Ctx->Context, Function, "entry")
    );

    for (IndexLane = 0; IndexLane < Lanes; IndexLane++) {
        KeyParams[IndexLane] = LLVMGetParam(Function, IndexLane);
        IndexPtrs[IndexLane] = LLVMGetParam(Function, Lanes + IndexLane);
    }

    VecType = LLVMVectorType(Ctx->I32, Lanes);
    KeyVec = LLVMGetUndef(VecType);

    for (IndexLane = 0; IndexLane < Lanes; IndexLane++) {
        Key = KeyParams[IndexLane];

        if (KeyType == Ctx->I64) {
            if (DownsizeFunction) {
                LLVMTypeRef DownsizeType;
                DownsizeType = LLVMFunctionType(Ctx->I32, &Ctx->I64, 1, FALSE);
                Key32 = LLVMBuildCall2(Ctx->Builder,
                                       DownsizeType,
                                       DownsizeFunction,
                                       &Key,
                                       1,
                                       "key_downsized");
            } else {
                Key32 = LLVMBuildTrunc(Ctx->Builder,
                                       Key,
                                       Ctx->I32,
                                       "key32");
            }
        } else {
            Key32 = Key;
        }

        LaneConst = LLVMConstInt(Ctx->I32, IndexLane, FALSE);
        KeyVec = LLVMBuildInsertElement(Ctx->Builder,
                                        KeyVec,
                                        Key32,
                                        LaneConst,
                                        "key_lane");
    }

    Seed1Vec = BuildSplatVectorConstant(VecType, Ctx->Seed1Const, Lanes);
    Seed2Vec = BuildSplatVectorConstant(VecType, Ctx->Seed2Const, Lanes);
    Seed3Byte1Vec = BuildSplatVectorConstant(VecType,
                                             Ctx->Seed3Byte1Const,
                                             Lanes);
    Seed3Byte2Vec = BuildSplatVectorConstant(VecType,
                                             Ctx->Seed3Byte2Const,
                                             Lanes);
    Seed3Byte3Vec = BuildSplatVectorConstant(VecType,
                                             Ctx->Seed3Byte3Const,
                                             Lanes);
    Seed3Byte4Vec = BuildSplatVectorConstant(VecType,
                                             Ctx->Seed3Byte4Const,
                                             Lanes);
    Seed4Vec = BuildSplatVectorConstant(VecType, Ctx->Seed4Const, Lanes);
    Seed5Vec = BuildSplatVectorConstant(VecType, Ctx->Seed5Const, Lanes);
    HashMaskVec = BuildSplatVectorConstant(VecType,
                                           Ctx->HashMaskConst,
                                           Lanes);

    UseHashMask = FALSE;

    switch (Ctx->HashFunctionId) {

        case PerfectHashHashMultiplyShiftRFunctionId:
            Vertex1Vec = LLVMBuildMul(Ctx->Builder,
                                      KeyVec,
                                      Seed1Vec,
                                      "v1.mul");
            Vertex1Vec = LLVMBuildLShr(Ctx->Builder,
                                       Vertex1Vec,
                                       Seed3Byte1Vec,
                                       "v1.shr");

            Vertex2Vec = LLVMBuildMul(Ctx->Builder,
                                      KeyVec,
                                      Seed2Vec,
                                      "v2.mul");
            Vertex2Vec = LLVMBuildLShr(Ctx->Builder,
                                       Vertex2Vec,
                                       Seed3Byte2Vec,
                                       "v2.shr");

            UseHashMask = TRUE;
            break;

        case PerfectHashHashMultiplyShiftRMultiplyFunctionId:
            Vertex1Vec = LLVMBuildMul(Ctx->Builder,
                                      KeyVec,
                                      Seed1Vec,
                                      "v1.mul1");
            Vertex1Vec = LLVMBuildLShr(Ctx->Builder,
                                       Vertex1Vec,
                                       Seed3Byte1Vec,
                                       "v1.shr1");
            Vertex1Vec = LLVMBuildMul(Ctx->Builder,
                                      Vertex1Vec,
                                      Seed2Vec,
                                      "v1.mul2");

            Vertex2Vec = LLVMBuildMul(Ctx->Builder,
                                      KeyVec,
                                      Seed4Vec,
                                      "v2.mul1");
            Vertex2Vec = LLVMBuildLShr(Ctx->Builder,
                                       Vertex2Vec,
                                       Seed3Byte2Vec,
                                       "v2.shr1");
            Vertex2Vec = LLVMBuildMul(Ctx->Builder,
                                      Vertex2Vec,
                                      Seed5Vec,
                                      "v2.mul2");

            UseHashMask = TRUE;
            break;

        case PerfectHashHashMultiplyShiftR2FunctionId:
            Vertex1Vec = LLVMBuildMul(Ctx->Builder,
                                      KeyVec,
                                      Seed1Vec,
                                      "v1.mul1");
            Vertex1Vec = LLVMBuildLShr(Ctx->Builder,
                                       Vertex1Vec,
                                       Seed3Byte1Vec,
                                       "v1.shr1");
            Vertex1Vec = LLVMBuildMul(Ctx->Builder,
                                      Vertex1Vec,
                                      Seed2Vec,
                                      "v1.mul2");
            Vertex1Vec = LLVMBuildLShr(Ctx->Builder,
                                       Vertex1Vec,
                                       Seed3Byte2Vec,
                                       "v1.shr2");

            Vertex2Vec = LLVMBuildMul(Ctx->Builder,
                                      KeyVec,
                                      Seed4Vec,
                                      "v2.mul1");
            Vertex2Vec = LLVMBuildLShr(Ctx->Builder,
                                       Vertex2Vec,
                                       Seed3Byte3Vec,
                                       "v2.shr1");
            Vertex2Vec = LLVMBuildMul(Ctx->Builder,
                                      Vertex2Vec,
                                      Seed5Vec,
                                      "v2.mul2");
            Vertex2Vec = LLVMBuildLShr(Ctx->Builder,
                                       Vertex2Vec,
                                       Seed3Byte4Vec,
                                       "v2.shr2");

            UseHashMask = TRUE;
            break;

        case PerfectHashHashMultiplyShiftRXFunctionId:
            Vertex1Vec = LLVMBuildMul(Ctx->Builder,
                                      KeyVec,
                                      Seed1Vec,
                                      "v1.mul");
            Vertex1Vec = LLVMBuildLShr(Ctx->Builder,
                                       Vertex1Vec,
                                       Seed3Byte1Vec,
                                       "v1.shr");

            Vertex2Vec = LLVMBuildMul(Ctx->Builder,
                                      KeyVec,
                                      Seed2Vec,
                                      "v2.mul");
            Vertex2Vec = LLVMBuildLShr(Ctx->Builder,
                                       Vertex2Vec,
                                       Seed3Byte1Vec,
                                       "v2.shr");
            break;

        case PerfectHashHashMultiplyShiftLRFunctionId:
            Vertex1Vec = LLVMBuildMul(Ctx->Builder,
                                      KeyVec,
                                      Seed1Vec,
                                      "v1.mul");
            Vertex1Vec = LLVMBuildShl(Ctx->Builder,
                                      Vertex1Vec,
                                      Seed3Byte1Vec,
                                      "v1.shl");

            Vertex2Vec = LLVMBuildMul(Ctx->Builder,
                                      KeyVec,
                                      Seed2Vec,
                                      "v2.mul");
            Vertex2Vec = LLVMBuildLShr(Ctx->Builder,
                                       Vertex2Vec,
                                       Seed3Byte2Vec,
                                       "v2.shr");

            UseHashMask = TRUE;
            break;

        case PerfectHashHashMulshrolate1RXFunctionId:
            Vertex1Vec = LLVMBuildMul(Ctx->Builder,
                                      KeyVec,
                                      Seed1Vec,
                                      "v1.mul");
            Vertex1Vec = BuildRotateRight32(Ctx->Builder,
                                            Vertex1Vec,
                                            Seed3Byte2Vec);
            Vertex1Vec = LLVMBuildLShr(Ctx->Builder,
                                       Vertex1Vec,
                                       Seed3Byte1Vec,
                                       "v1.shr");

            Vertex2Vec = LLVMBuildMul(Ctx->Builder,
                                      KeyVec,
                                      Seed2Vec,
                                      "v2.mul");
            Vertex2Vec = LLVMBuildLShr(Ctx->Builder,
                                       Vertex2Vec,
                                       Seed3Byte1Vec,
                                       "v2.shr");
            break;

        case PerfectHashHashMulshrolate2RXFunctionId:
            Vertex1Vec = LLVMBuildMul(Ctx->Builder,
                                      KeyVec,
                                      Seed1Vec,
                                      "v1.mul");
            Vertex1Vec = BuildRotateRight32(Ctx->Builder,
                                            Vertex1Vec,
                                            Seed3Byte2Vec);
            Vertex1Vec = LLVMBuildLShr(Ctx->Builder,
                                       Vertex1Vec,
                                       Seed3Byte1Vec,
                                       "v1.shr");

            Vertex2Vec = LLVMBuildMul(Ctx->Builder,
                                      KeyVec,
                                      Seed2Vec,
                                      "v2.mul");
            Vertex2Vec = BuildRotateRight32(Ctx->Builder,
                                            Vertex2Vec,
                                            Seed3Byte3Vec);
            Vertex2Vec = LLVMBuildLShr(Ctx->Builder,
                                       Vertex2Vec,
                                       Seed3Byte1Vec,
                                       "v2.shr");
            break;

        case PerfectHashHashMulshrolate3RXFunctionId:
            Vertex1Vec = LLVMBuildMul(Ctx->Builder,
                                      KeyVec,
                                      Seed1Vec,
                                      "v1.mul1");
            Vertex1Vec = BuildRotateRight32(Ctx->Builder,
                                            Vertex1Vec,
                                            Seed3Byte2Vec);
            Vertex1Vec = LLVMBuildMul(Ctx->Builder,
                                      Vertex1Vec,
                                      Seed4Vec,
                                      "v1.mul2");
            Vertex1Vec = LLVMBuildLShr(Ctx->Builder,
                                       Vertex1Vec,
                                       Seed3Byte1Vec,
                                       "v1.shr");

            Vertex2Vec = LLVMBuildMul(Ctx->Builder,
                                      KeyVec,
                                      Seed2Vec,
                                      "v2.mul");
            Vertex2Vec = BuildRotateRight32(Ctx->Builder,
                                            Vertex2Vec,
                                            Seed3Byte3Vec);
            Vertex2Vec = LLVMBuildLShr(Ctx->Builder,
                                       Vertex2Vec,
                                       Seed3Byte1Vec,
                                       "v2.shr");
            break;

        case PerfectHashHashMulshrolate4RXFunctionId:
            Vertex1Vec = LLVMBuildMul(Ctx->Builder,
                                      KeyVec,
                                      Seed1Vec,
                                      "v1.mul1");
            Vertex1Vec = BuildRotateRight32(Ctx->Builder,
                                            Vertex1Vec,
                                            Seed3Byte2Vec);
            Vertex1Vec = LLVMBuildMul(Ctx->Builder,
                                      Vertex1Vec,
                                      Seed4Vec,
                                      "v1.mul2");
            Vertex1Vec = LLVMBuildLShr(Ctx->Builder,
                                       Vertex1Vec,
                                       Seed3Byte1Vec,
                                       "v1.shr");

            Vertex2Vec = LLVMBuildMul(Ctx->Builder,
                                      KeyVec,
                                      Seed2Vec,
                                      "v2.mul1");
            Vertex2Vec = BuildRotateRight32(Ctx->Builder,
                                            Vertex2Vec,
                                            Seed3Byte3Vec);
            Vertex2Vec = LLVMBuildMul(Ctx->Builder,
                                      Vertex2Vec,
                                      Seed5Vec,
                                      "v2.mul2");
            Vertex2Vec = LLVMBuildLShr(Ctx->Builder,
                                       Vertex2Vec,
                                       Seed3Byte1Vec,
                                       "v2.shr");
            break;

        default:
            return NULL;
    }

    if (UseHashMask) {
        IndexVec1 = LLVMBuildAnd(Ctx->Builder,
                                 Vertex1Vec,
                                 HashMaskVec,
                                 "masked_low");
        IndexVec2 = LLVMBuildAnd(Ctx->Builder,
                                 Vertex2Vec,
                                 HashMaskVec,
                                 "masked_high");
    } else {
        IndexVec1 = Vertex1Vec;
        IndexVec2 = Vertex2Vec;
    }

    for (IndexLane = 0; IndexLane < Lanes; IndexLane++) {
        LaneConst = LLVMConstInt(Ctx->I32, IndexLane, FALSE);
        Index1 = LLVMBuildExtractElement(Ctx->Builder,
                                         IndexVec1,
                                         LaneConst,
                                         "index1");
        Index2 = LLVMBuildExtractElement(Ctx->Builder,
                                         IndexVec2,
                                         LaneConst,
                                         "index2");

        Vertex1 = BuildTableDataLoadAndExtend(Ctx->Builder,
                                              Ctx->TableElementType,
                                              Ctx->I64,
                                              Ctx->I32,
                                              Ctx->TableDataConst,
                                              Index1);

        Vertex2 = BuildTableDataLoadAndExtend(Ctx->Builder,
                                              Ctx->TableElementType,
                                              Ctx->I64,
                                              Ctx->I32,
                                              Ctx->TableDataConst,
                                              Index2);

        Index = LLVMBuildAdd(Ctx->Builder, Vertex1, Vertex2, "index_add");
        Index = LLVMBuildAnd(Ctx->Builder,
                             Index,
                             Ctx->IndexMaskConst,
                             "index_masked");
        LLVMBuildStore(Ctx->Builder, Index, IndexPtrs[IndexLane]);
    }

    LLVMBuildRetVoid(Ctx->Builder);

    return Function;
}

static
LLVMValueRef
BuildChm01Index4Function(
    _In_ PCHM01_JIT_CONTEXT Ctx,
    _In_ PCSTR Name,
    _In_ LLVMValueRef IndexFunction,
    _In_ LLVMTypeRef IndexFunctionType,
    _In_ LLVMTypeRef KeyType
    )
{
    LLVMTypeRef FunctionType;
    LLVMTypeRef Params[8];
    LLVMValueRef Function;
    LLVMValueRef Key1;
    LLVMValueRef Key2;
    LLVMValueRef Key3;
    LLVMValueRef Key4;
    LLVMValueRef Index1Ptr;
    LLVMValueRef Index2Ptr;
    LLVMValueRef Index3Ptr;
    LLVMValueRef Index4Ptr;
    LLVMValueRef Index1;
    LLVMValueRef Index2;
    LLVMValueRef Index3;
    LLVMValueRef Index4;

    Params[0] = KeyType;
    Params[1] = KeyType;
    Params[2] = KeyType;
    Params[3] = KeyType;
    Params[4] = Ctx->I32Ptr;
    Params[5] = Ctx->I32Ptr;
    Params[6] = Ctx->I32Ptr;
    Params[7] = Ctx->I32Ptr;

    FunctionType = LLVMFunctionType(LLVMVoidTypeInContext(Ctx->Context),
                                    Params,
                                    ARRAYSIZE(Params),
                                    FALSE);

    Function = LLVMAddFunction(Ctx->Module, Name, FunctionType);

    LLVMPositionBuilderAtEnd(
        Ctx->Builder,
        LLVMAppendBasicBlockInContext(Ctx->Context, Function, "entry")
    );

    Key1 = LLVMGetParam(Function, 0);
    Key2 = LLVMGetParam(Function, 1);
    Key3 = LLVMGetParam(Function, 2);
    Key4 = LLVMGetParam(Function, 3);
    Index1Ptr = LLVMGetParam(Function, 4);
    Index2Ptr = LLVMGetParam(Function, 5);
    Index3Ptr = LLVMGetParam(Function, 6);
    Index4Ptr = LLVMGetParam(Function, 7);

    Index1 = LLVMBuildCall2(Ctx->Builder,
                            IndexFunctionType,
                            IndexFunction,
                            &Key1,
                            1,
                            "index1");

    Index2 = LLVMBuildCall2(Ctx->Builder,
                            IndexFunctionType,
                            IndexFunction,
                            &Key2,
                            1,
                            "index2");

    Index3 = LLVMBuildCall2(Ctx->Builder,
                            IndexFunctionType,
                            IndexFunction,
                            &Key3,
                            1,
                            "index3");

    Index4 = LLVMBuildCall2(Ctx->Builder,
                            IndexFunctionType,
                            IndexFunction,
                            &Key4,
                            1,
                            "index4");

    LLVMBuildStore(Ctx->Builder, Index1, Index1Ptr);
    LLVMBuildStore(Ctx->Builder, Index2, Index2Ptr);
    LLVMBuildStore(Ctx->Builder, Index3, Index3Ptr);
    LLVMBuildStore(Ctx->Builder, Index4, Index4Ptr);
    LLVMBuildRetVoid(Ctx->Builder);

    return Function;
}

static
LLVMValueRef
BuildChm01Index8Function(
    _In_ PCHM01_JIT_CONTEXT Ctx,
    _In_ PCSTR Name,
    _In_ LLVMValueRef IndexFunction,
    _In_ LLVMTypeRef IndexFunctionType,
    _In_ LLVMTypeRef KeyType
    )
{
    LLVMTypeRef FunctionType;
    LLVMTypeRef Params[16];
    LLVMValueRef Function;
    LLVMValueRef KeyParams[8];
    LLVMValueRef IndexPtrs[8];
    LLVMValueRef IndexValue;
    ULONG Index;

    for (Index = 0; Index < 8; Index++) {
        Params[Index] = KeyType;
    }
    for (Index = 0; Index < 8; Index++) {
        Params[8 + Index] = Ctx->I32Ptr;
    }

    FunctionType = LLVMFunctionType(LLVMVoidTypeInContext(Ctx->Context),
                                    Params,
                                    ARRAYSIZE(Params),
                                    FALSE);

    Function = LLVMAddFunction(Ctx->Module, Name, FunctionType);

    LLVMPositionBuilderAtEnd(
        Ctx->Builder,
        LLVMAppendBasicBlockInContext(Ctx->Context, Function, "entry")
    );

    for (Index = 0; Index < 8; Index++) {
        KeyParams[Index] = LLVMGetParam(Function, Index);
        IndexPtrs[Index] = LLVMGetParam(Function, 8 + Index);
    }

    for (Index = 0; Index < 8; Index++) {
        IndexValue = LLVMBuildCall2(Ctx->Builder,
                                    IndexFunctionType,
                                    IndexFunction,
                                    &KeyParams[Index],
                                    1,
                                    "index");
        LLVMBuildStore(Ctx->Builder, IndexValue, IndexPtrs[Index]);
    }

    LLVMBuildRetVoid(Ctx->Builder);

    return Function;
}

_Must_inspect_result_
static
HRESULT
CompileChm01IndexJit(
    _In_ PPERFECT_HASH_TABLE Table,
    _Inout_ PPERFECT_HASH_TABLE_JIT Jit,
    _In_ PPERFECT_HASH_TABLE_COMPILE_FLAGS CompileFlags
    )
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG Seed3;
    ULONG Seed4;
    ULONG Seed5;
    ULONG HashMask;
    ULONG IndexMask;
    ULONG Seed3Byte1;
    ULONG Seed3Byte2;
    ULONG Seed3Byte3;
    ULONG Seed3Byte4;
    PVOID TableData;
    BOOLEAN UseAssigned16;
    BOOLEAN KeysDownsized;
    BOOLEAN CompileIndex64;
    BOOLEAN CompileIndex2;
    BOOLEAN CompileIndex4;
    BOOLEAN CompileIndex8;
    BOOLEAN CompileIndex2_64;
    BOOLEAN CompileIndex4_64;
    BOOLEAN CompileIndex8_64;
    BOOLEAN CompileVectorIndex2;
    BOOLEAN CompileVectorIndex4;
    BOOLEAN CompileVectorIndex8;
    BOOLEAN CompileVectorIndex2_64;
    BOOLEAN CompileVectorIndex4_64;
    BOOLEAN CompileVectorIndex8_64;
    HRESULT Result = S_OK;
    PTABLE_INFO_ON_DISK TableInfo;
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId;
    PPERFECT_HASH_KEYS Keys;
    PPERFECT_HASH_KEYS_BITMAP KeysBitmap;
    ULONGLONG DownsizeBitmap = 0;
    ULONGLONG DownsizeShiftedMask = 0;
    BYTE DownsizeTrailingZeros = 0;
    BOOLEAN DownsizeContiguous = FALSE;

    char *ErrorMessage = NULL;
    char *TargetTriple = NULL;
    char *DataLayout = NULL;

    LLVMTargetRef Target = NULL;
    LLVMContextRef Context = NULL;
    LLVMModuleRef Module = NULL;
    LLVMBuilderRef Builder = NULL;
    LLVMExecutionEngineRef Engine = NULL;
    LLVMTargetMachineRef TargetMachine = NULL;
    LLVMTargetDataRef TargetData = NULL;

    CHM01_JIT_CONTEXT JitContext;
    LLVMValueRef DownsizeFunction = NULL;
    LLVMValueRef IndexFunction = NULL;
    LLVMValueRef Index64Function = NULL;
    LLVMTypeRef IndexFunctionType = NULL;
    LLVMTypeRef Index64FunctionType = NULL;

    //
    // Initialize aliases.
    //

    TableInfo = Table->TableInfoOnDisk;
    HashFunctionId = Table->HashFunctionId;
    Keys = Table->Keys;
    UseAssigned16 = (Table->State.UsingAssigned16 != FALSE);
    KeysDownsized = (TableInfo->OriginalKeySizeInBytes >
                     TableInfo->KeySizeInBytes);

    if (Keys) {
        KeysDownsized = KeysWereDownsized(Keys);
    }

    TableData = UseAssigned16 ? (PVOID)Table->Assigned16 :
                                (PVOID)Table->TableData;

    Seed1 = TableInfo->Seed1;
    Seed2 = TableInfo->Seed2;
    Seed3 = TableInfo->Seed3;
    Seed4 = TableInfo->Seed4;
    Seed5 = TableInfo->Seed5;
    HashMask = TableInfo->HashMask;
    IndexMask = TableInfo->IndexMask;

    Seed3Byte1 = (Seed3 & 0xff);
    Seed3Byte2 = ((Seed3 >> 8) & 0xff);
    Seed3Byte3 = ((Seed3 >> 16) & 0xff);
    Seed3Byte4 = ((Seed3 >> 24) & 0xff);

    CompileVectorIndex2 = (CompileFlags->JitVectorIndex2 != FALSE);
    CompileVectorIndex4 = (CompileFlags->JitVectorIndex4 != FALSE);
    CompileVectorIndex8 = (CompileFlags->JitVectorIndex8 != FALSE);
    CompileIndex2 = (CompileFlags->JitIndex2 != FALSE) ||
                    (CompileVectorIndex2 != FALSE);
    CompileIndex4 = (CompileFlags->JitIndex4 != FALSE) ||
                    (CompileVectorIndex4 != FALSE);
    CompileIndex8 = (CompileFlags->JitIndex8 != FALSE) ||
                    (CompileVectorIndex8 != FALSE);

    if (CompileFlags->JitIndex64 && !KeysDownsized) {
        return PH_E_NOT_IMPLEMENTED;
    }

    CompileIndex64 = (KeysDownsized != FALSE);
    CompileIndex2_64 = (CompileIndex64 && CompileIndex2);
    CompileIndex4_64 = (CompileIndex64 && CompileIndex4);
    CompileIndex8_64 = (CompileIndex64 && CompileIndex8);
    CompileVectorIndex2_64 = (CompileIndex64 && CompileVectorIndex2);
    CompileVectorIndex4_64 = (CompileIndex64 && CompileVectorIndex4);
    CompileVectorIndex8_64 = (CompileIndex64 && CompileVectorIndex8);

    if (KeysDownsized) {
        if (Keys) {
            KeysBitmap = &Keys->Stats.KeysBitmap;
            DownsizeBitmap = Keys->DownsizeBitmap;
            DownsizeTrailingZeros = KeysBitmap->TrailingZeros;
            DownsizeShiftedMask = KeysBitmap->ShiftedMask;
            DownsizeContiguous = (KeysBitmap->Flags.Contiguous != FALSE);
        } else if (Table->State.DownsizeMetadataValid) {
            DownsizeBitmap = Table->DownsizeBitmap;
            DownsizeTrailingZeros = Table->DownsizeTrailingZeros;
            DownsizeShiftedMask = Table->DownsizeShiftedMask;
            DownsizeContiguous = (Table->DownsizeContiguous != FALSE);
        } else {
            return PH_E_NOT_IMPLEMENTED;
        }
    }

    if (LLVMInitializeNativeTarget() != 0 ||
        LLVMInitializeNativeAsmPrinter() != 0 ||
        LLVMInitializeNativeAsmParser() != 0) {
        return PH_E_TABLE_COMPILATION_FAILED;
    }

    LLVMLinkInMCJIT();

    Context = LLVMContextCreate();
    Module = LLVMModuleCreateWithNameInContext("PerfectHashChm01Jit", Context);
    Builder = LLVMCreateBuilderInContext(Context);

    TargetTriple = LLVMGetDefaultTargetTriple();
    if (LLVMGetTargetFromTriple(TargetTriple, &Target, &ErrorMessage) != 0) {
        Result = PH_E_TABLE_COMPILATION_FAILED;
        goto Error;
    }

    TargetMachine = LLVMCreateTargetMachine(Target,
                                            TargetTriple,
                                            "",
                                            "",
                                            LLVMCodeGenLevelAggressive,
                                            LLVMRelocDefault,
                                            LLVMCodeModelJITDefault);

    if (!TargetMachine) {
        Result = PH_E_TABLE_COMPILATION_FAILED;
        goto Error;
    }

    TargetData = LLVMCreateTargetDataLayout(TargetMachine);
    if (!TargetData) {
        Result = PH_E_TABLE_COMPILATION_FAILED;
        goto Error;
    }
    DataLayout = LLVMCopyStringRepOfTargetData(TargetData);

    LLVMSetTarget(Module, TargetTriple);
    LLVMSetDataLayout(Module, DataLayout);

    ZeroStructInline(JitContext);
    JitContext.Context = Context;
    JitContext.Module = Module;
    JitContext.Builder = Builder;
    JitContext.I1 = LLVMInt1TypeInContext(Context);
    JitContext.I16 = LLVMInt16TypeInContext(Context);
    JitContext.I32 = LLVMInt32TypeInContext(Context);
    JitContext.I64 = LLVMInt64TypeInContext(Context);
    JitContext.I32Ptr = LLVMPointerType(JitContext.I32, 0);
    JitContext.TableElementType = UseAssigned16 ?
                                  JitContext.I16 :
                                  JitContext.I32;

    JitContext.HashFunctionId = HashFunctionId;
    JitContext.UseAssigned16 = UseAssigned16;

    JitContext.Seed1Const = LLVMConstInt(JitContext.I32, Seed1, FALSE);
    JitContext.Seed2Const = LLVMConstInt(JitContext.I32, Seed2, FALSE);
    JitContext.Seed3Byte1Const = LLVMConstInt(JitContext.I32,
                                              Seed3Byte1,
                                              FALSE);
    JitContext.Seed3Byte2Const = LLVMConstInt(JitContext.I32,
                                              Seed3Byte2,
                                              FALSE);
    JitContext.Seed3Byte3Const = LLVMConstInt(JitContext.I32,
                                              Seed3Byte3,
                                              FALSE);
    JitContext.Seed3Byte4Const = LLVMConstInt(JitContext.I32,
                                              Seed3Byte4,
                                              FALSE);
    JitContext.Seed4Const = LLVMConstInt(JitContext.I32, Seed4, FALSE);
    JitContext.Seed5Const = LLVMConstInt(JitContext.I32, Seed5, FALSE);
    JitContext.HashMaskConst = LLVMConstInt(JitContext.I32,
                                            HashMask,
                                            FALSE);
    JitContext.IndexMaskConst = LLVMConstInt(JitContext.I32,
                                             IndexMask,
                                             FALSE);

    JitContext.TableDataConst = LLVMConstIntToPtr(
        LLVMConstInt(JitContext.I64,
                     (ULONGLONG)(ULONG_PTR)TableData,
                     FALSE),
        LLVMPointerType(JitContext.TableElementType, 0)
    );

    if (KeysDownsized) {
        DownsizeFunction = BuildDownsizeKeyFunction(Module,
                                                    Builder,
                                                    Context,
                                                    DownsizeBitmap,
                                                    DownsizeContiguous,
                                                    DownsizeTrailingZeros,
                                                    DownsizeShiftedMask);
        if (!DownsizeFunction) {
            Result = PH_E_TABLE_COMPILATION_FAILED;
            goto Error;
        }
    }

    IndexFunctionType = LLVMFunctionType(JitContext.I32,
                                         &JitContext.I32,
                                         1,
                                         FALSE);

    IndexFunction = BuildChm01IndexFunction(&JitContext,
                                            "PerfectHashJitIndex",
                                            JitContext.I32,
                                            NULL);

    if (!IndexFunction) {
        Result = PH_E_TABLE_COMPILATION_FAILED;
        goto Error;
    }

    if (CompileIndex64) {
        Index64FunctionType = LLVMFunctionType(JitContext.I32,
                                               &JitContext.I64,
                                               1,
                                               FALSE);

        Index64Function = BuildChm01IndexFunction(&JitContext,
                                                  "PerfectHashJitIndex64",
                                                  JitContext.I64,
                                                  DownsizeFunction);

        if (!Index64Function) {
            Result = PH_E_TABLE_COMPILATION_FAILED;
            goto Error;
        }
    }

    if (CompileIndex2) {
        if (CompileVectorIndex2) {
            if (!BuildChm01IndexVectorFunction(&JitContext,
                                               "PerfectHashJitIndex2Vector",
                                               JitContext.I32,
                                               NULL,
                                               2)) {
                CompileVectorIndex2 = FALSE;
            }
        }

        if (!CompileVectorIndex2) {
            if (!BuildChm01Index2Function(&JitContext,
                                          "PerfectHashJitIndex2",
                                          IndexFunction,
                                          IndexFunctionType,
                                          JitContext.I32)) {
                Result = PH_E_TABLE_COMPILATION_FAILED;
                goto Error;
            }
        }
    }

    if (CompileIndex4) {
        if (CompileVectorIndex4) {
            if (!BuildChm01IndexVectorFunction(&JitContext,
                                               "PerfectHashJitIndex4Vector",
                                               JitContext.I32,
                                               NULL,
                                               4)) {
                CompileVectorIndex4 = FALSE;
            }
        }

        if (!CompileVectorIndex4) {
            if (!BuildChm01Index4Function(&JitContext,
                                          "PerfectHashJitIndex4",
                                          IndexFunction,
                                          IndexFunctionType,
                                          JitContext.I32)) {
                Result = PH_E_TABLE_COMPILATION_FAILED;
                goto Error;
            }
        }
    }

    if (CompileIndex8) {
        if (CompileVectorIndex8) {
            if (!BuildChm01IndexVectorFunction(&JitContext,
                                               "PerfectHashJitIndex8Vector",
                                               JitContext.I32,
                                               NULL,
                                               8)) {
                CompileVectorIndex8 = FALSE;
            }
        }

        if (!CompileVectorIndex8) {
            if (!BuildChm01Index8Function(&JitContext,
                                          "PerfectHashJitIndex8",
                                          IndexFunction,
                                          IndexFunctionType,
                                          JitContext.I32)) {
                Result = PH_E_TABLE_COMPILATION_FAILED;
                goto Error;
            }
        }
    }

    if (CompileIndex2_64) {
        if (CompileVectorIndex2_64) {
            if (!BuildChm01IndexVectorFunction(&JitContext,
                                               "PerfectHashJitIndex2Vector_64",
                                               JitContext.I64,
                                               DownsizeFunction,
                                               2)) {
                CompileVectorIndex2_64 = FALSE;
            }
        }

        if (!CompileVectorIndex2_64) {
            if (!BuildChm01Index2Function(&JitContext,
                                          "PerfectHashJitIndex2_64",
                                          Index64Function,
                                          Index64FunctionType,
                                          JitContext.I64)) {
                Result = PH_E_TABLE_COMPILATION_FAILED;
                goto Error;
            }
        }
    }

    if (CompileIndex4_64) {
        if (CompileVectorIndex4_64) {
            if (!BuildChm01IndexVectorFunction(&JitContext,
                                               "PerfectHashJitIndex4Vector_64",
                                               JitContext.I64,
                                               DownsizeFunction,
                                               4)) {
                CompileVectorIndex4_64 = FALSE;
            }
        }

        if (!CompileVectorIndex4_64) {
            if (!BuildChm01Index4Function(&JitContext,
                                          "PerfectHashJitIndex4_64",
                                          Index64Function,
                                          Index64FunctionType,
                                          JitContext.I64)) {
                Result = PH_E_TABLE_COMPILATION_FAILED;
                goto Error;
            }
        }
    }

    if (CompileIndex8_64) {
        if (CompileVectorIndex8_64) {
            if (!BuildChm01IndexVectorFunction(&JitContext,
                                               "PerfectHashJitIndex8Vector_64",
                                               JitContext.I64,
                                               DownsizeFunction,
                                               8)) {
                CompileVectorIndex8_64 = FALSE;
            }
        }

        if (!CompileVectorIndex8_64) {
            if (!BuildChm01Index8Function(&JitContext,
                                          "PerfectHashJitIndex8_64",
                                          Index64Function,
                                          Index64FunctionType,
                                          JitContext.I64)) {
                Result = PH_E_TABLE_COMPILATION_FAILED;
                goto Error;
            }
        }
    }

    if (LLVMVerifyModule(Module,
                         LLVMReturnStatusAction,
                         &ErrorMessage) != 0) {
        Result = PH_E_TABLE_COMPILATION_FAILED;
        goto Error;
    }

    if (LLVMCreateJITCompilerForModule(&Engine,
                                       Module,
                                       3,
                                       &ErrorMessage) != 0) {
        Result = PH_E_TABLE_COMPILATION_FAILED;
        goto Error;
    }

    Jit->IndexFunction = (PVOID)(ULONG_PTR)
        LLVMGetFunctionAddress(Engine, "PerfectHashJitIndex");

    if (!Jit->IndexFunction) {
        Result = PH_E_TABLE_COMPILATION_FAILED;
        goto Error;
    }

    Jit->Flags.IndexCompiled = TRUE;

    if (CompileIndex64) {
        Jit->Index64Function = (PVOID)(ULONG_PTR)
            LLVMGetFunctionAddress(Engine, "PerfectHashJitIndex64");
        if (!Jit->Index64Function) {
            Result = PH_E_TABLE_COMPILATION_FAILED;
            goto Error;
        }
        Jit->Flags.Index64Compiled = TRUE;
    }

    if (CompileIndex2) {
        if (CompileVectorIndex2) {
            Jit->Index2Function = (PVOID)(ULONG_PTR)
                LLVMGetFunctionAddress(Engine, "PerfectHashJitIndex2Vector");
        } else {
            Jit->Index2Function = (PVOID)(ULONG_PTR)
                LLVMGetFunctionAddress(Engine, "PerfectHashJitIndex2");
        }
        if (!Jit->Index2Function) {
            Result = PH_E_TABLE_COMPILATION_FAILED;
            goto Error;
        }
        Jit->Flags.Index2Compiled = TRUE;
    }

    if (CompileIndex4) {
        if (CompileVectorIndex4) {
            Jit->Index4Function = (PVOID)(ULONG_PTR)
                LLVMGetFunctionAddress(Engine, "PerfectHashJitIndex4Vector");
        } else {
            Jit->Index4Function = (PVOID)(ULONG_PTR)
                LLVMGetFunctionAddress(Engine, "PerfectHashJitIndex4");
        }
        if (!Jit->Index4Function) {
            Result = PH_E_TABLE_COMPILATION_FAILED;
            goto Error;
        }
        Jit->Flags.Index4Compiled = TRUE;
    }

    if (CompileIndex8) {
        if (CompileVectorIndex8) {
            Jit->Index8Function = (PVOID)(ULONG_PTR)
                LLVMGetFunctionAddress(Engine, "PerfectHashJitIndex8Vector");
        } else {
            Jit->Index8Function = (PVOID)(ULONG_PTR)
                LLVMGetFunctionAddress(Engine, "PerfectHashJitIndex8");
        }
        if (!Jit->Index8Function) {
            Result = PH_E_TABLE_COMPILATION_FAILED;
            goto Error;
        }
        Jit->Flags.Index8Compiled = TRUE;
    }

    if (CompileIndex2_64) {
        if (CompileVectorIndex2_64) {
            Jit->Index2_64Function = (PVOID)(ULONG_PTR)
                LLVMGetFunctionAddress(Engine,
                                       "PerfectHashJitIndex2Vector_64");
        } else {
            Jit->Index2_64Function = (PVOID)(ULONG_PTR)
                LLVMGetFunctionAddress(Engine, "PerfectHashJitIndex2_64");
        }
        if (!Jit->Index2_64Function) {
            Result = PH_E_TABLE_COMPILATION_FAILED;
            goto Error;
        }
        Jit->Flags.Index2_64Compiled = TRUE;
    }

    if (CompileIndex4_64) {
        if (CompileVectorIndex4_64) {
            Jit->Index4_64Function = (PVOID)(ULONG_PTR)
                LLVMGetFunctionAddress(Engine,
                                       "PerfectHashJitIndex4Vector_64");
        } else {
            Jit->Index4_64Function = (PVOID)(ULONG_PTR)
                LLVMGetFunctionAddress(Engine, "PerfectHashJitIndex4_64");
        }
        if (!Jit->Index4_64Function) {
            Result = PH_E_TABLE_COMPILATION_FAILED;
            goto Error;
        }
        Jit->Flags.Index4_64Compiled = TRUE;
    }

    if (CompileIndex8_64) {
        if (CompileVectorIndex8_64) {
            Jit->Index8_64Function = (PVOID)(ULONG_PTR)
                LLVMGetFunctionAddress(Engine,
                                       "PerfectHashJitIndex8Vector_64");
        } else {
            Jit->Index8_64Function = (PVOID)(ULONG_PTR)
                LLVMGetFunctionAddress(Engine, "PerfectHashJitIndex8_64");
        }
        if (!Jit->Index8_64Function) {
            Result = PH_E_TABLE_COMPILATION_FAILED;
            goto Error;
        }
        Jit->Flags.Index8_64Compiled = TRUE;
    }

    Jit->ExecutionEngine = Engine;
    Jit->Context = Context;
    Engine = NULL;
    Context = NULL;
    Module = NULL;

    Result = S_OK;

Error:

    if (ErrorMessage) {
        LLVMDisposeMessage(ErrorMessage);
        ErrorMessage = NULL;
    }

    if (Builder) {
        LLVMDisposeBuilder(Builder);
        Builder = NULL;
    }

    if (Engine) {
        LLVMDisposeExecutionEngine(Engine);
        Engine = NULL;
    }

    if (Module) {
        LLVMDisposeModule(Module);
        Module = NULL;
    }

    if (Context) {
        LLVMContextDispose(Context);
        Context = NULL;
    }

    if (TargetData) {
        LLVMDisposeTargetData(TargetData);
        TargetData = NULL;
    }

    if (TargetMachine) {
        LLVMDisposeTargetMachine(TargetMachine);
        TargetMachine = NULL;
    }

    if (DataLayout) {
        LLVMDisposeMessage(DataLayout);
        DataLayout = NULL;
    }

    if (TargetTriple) {
        LLVMDisposeMessage(TargetTriple);
        TargetTriple = NULL;
    }

    return Result;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableCompileJit(
    PPERFECT_HASH_TABLE Table,
    PPERFECT_HASH_TABLE_COMPILE_FLAGS CompileFlagsPointer
    )
{
    HRESULT Result = S_OK;
    PALLOCATOR Allocator;
    PPERFECT_HASH_TABLE_JIT Jit;
    PERFECT_HASH_TABLE_COMPILE_FLAGS CompileFlags;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    if (!Table->Flags.Created) {
        return PH_E_TABLE_NOT_CREATED;
    }

    if (!ARGUMENT_PRESENT(Table->TableInfoOnDisk)) {
        return PH_E_INVARIANT_CHECK_FAILED;
    }

    if (Table->State.UsingAssigned16) {
        if (!ARGUMENT_PRESENT(Table->Assigned16)) {
            return PH_E_INVARIANT_CHECK_FAILED;
        }
    } else if (!ARGUMENT_PRESENT(Table->TableData)) {
        return PH_E_INVARIANT_CHECK_FAILED;
    }

    if (Table->AlgorithmId != PerfectHashChm01AlgorithmId) {
        return PH_E_NOT_IMPLEMENTED;
    }

    if (Table->TableInfoOnDisk->KeySizeInBytes != sizeof(ULONG)) {
        return PH_E_NOT_IMPLEMENTED;
    }

    if (Table->MaskFunctionId != PerfectHashAndMaskFunctionId) {
        return PH_E_NOT_IMPLEMENTED;
    }

    if (!IsSupportedJitHashFunctionId(Table->HashFunctionId)) {
        return PH_E_NOT_IMPLEMENTED;
    }

    //
    // Initialize local flags.
    //

    if (ARGUMENT_PRESENT(CompileFlagsPointer)) {
        Result = IsValidTableCompileFlags(CompileFlagsPointer);
        if (FAILED(Result)) {
            return PH_E_INVALID_TABLE_COMPILE_FLAGS;
        }
        CompileFlags.AsULong = CompileFlagsPointer->AsULong;
    } else {
        CompileFlags.AsULong = 0;
    }

    CompileFlags.Jit = TRUE;

    //
    // Reset any existing JIT state.
    //

    if (Table->Jit) {
        PerfectHashTableJitRundown(Table);
    }

    //
    // Allocate a new JIT state structure.
    //

    Allocator = Table->Allocator;
    Jit = (PPERFECT_HASH_TABLE_JIT)(
        Allocator->Vtbl->Calloc(
            Allocator,
            1,
            sizeof(*Jit)
        )
    );

    if (!Jit) {
        return E_OUTOFMEMORY;
    }

    Table->Jit = Jit;
    Jit->SizeOfStruct = sizeof(*Jit);
    Jit->AlgorithmId = Table->AlgorithmId;
    Jit->HashFunctionId = Table->HashFunctionId;
    Jit->MaskFunctionId = Table->MaskFunctionId;
    Jit->OriginalIndex = Table->Vtbl->Index;
    Jit->OriginalQueryInterface = Table->Vtbl->QueryInterface;
    Jit->Interface.Table = Table;
    Jit->Interface.Vtbl = &PerfectHashTableJitInterfaceVtbl;
    Table->Vtbl->QueryInterface = PerfectHashTableQueryInterfaceJit;

    Result = CompileChm01IndexJit(Table, Jit, &CompileFlags);
    if (FAILED(Result)) {
        PerfectHashTableJitRundown(Table);
        return Result;
    }

    Jit->Flags.Valid = TRUE;
    Table->Flags.JitEnabled = TRUE;
    Table->Vtbl->Index = PerfectHashTableIndexJit;

    return S_OK;
}

#else

_Use_decl_annotations_
HRESULT
PerfectHashTableCompileJit(
    PPERFECT_HASH_TABLE Table,
    PPERFECT_HASH_TABLE_COMPILE_FLAGS CompileFlagsPointer
    )
{
    UNREFERENCED_PARAMETER(Table);
    UNREFERENCED_PARAMETER(CompileFlagsPointer);

    return PH_E_NOT_IMPLEMENTED;
}

#endif

_Use_decl_annotations_
HRESULT
PerfectHashTableQueryInterfaceJit(
    PPERFECT_HASH_TABLE Table,
    REFIID InterfaceId,
    PVOID *Interface
    )
{
    BOOLEAN Match;
    PPERFECT_HASH_TABLE_JIT Jit;

    if (!ARGUMENT_PRESENT(Interface)) {
        return E_POINTER;
    }

    *Interface = NULL;

    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

#ifdef __cplusplus
    Match = InlineIsEqualGUID(InterfaceId,
                              IID_PERFECT_HASH_TABLE_JIT_INTERFACE);
#else
    Match = InlineIsEqualGUID(InterfaceId,
                              &IID_PERFECT_HASH_TABLE_JIT_INTERFACE);
#endif

    if (Match) {
        Jit = Table->Jit;
        if (!ARGUMENT_PRESENT(Jit) || !Jit->Flags.Valid) {
            return E_NOINTERFACE;
        }

        *Interface = &Jit->Interface;
        Table->Vtbl->AddRef(Table);
        return S_OK;
    }

    Jit = Table->Jit;
    if (ARGUMENT_PRESENT(Jit) &&
        ARGUMENT_PRESENT(Jit->OriginalQueryInterface)) {
        return Jit->OriginalQueryInterface(Table, InterfaceId, Interface);
    }

    return E_NOINTERFACE;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableJitInterfaceQueryInterface(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    REFIID InterfaceId,
    PVOID *Interface
    )
{
    if (!ARGUMENT_PRESENT(Jit) ||
        !ARGUMENT_PRESENT(Jit->Table) ||
        !ARGUMENT_PRESENT(Interface)) {
        return E_POINTER;
    }

    return Jit->Table->Vtbl->QueryInterface(Jit->Table,
                                            InterfaceId,
                                            Interface);
}

_Use_decl_annotations_
ULONG
PerfectHashTableJitInterfaceAddRef(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit
    )
{
    if (!ARGUMENT_PRESENT(Jit) || !ARGUMENT_PRESENT(Jit->Table)) {
        return 0;
    }

    return Jit->Table->Vtbl->AddRef(Jit->Table);
}

_Use_decl_annotations_
ULONG
PerfectHashTableJitInterfaceRelease(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit
    )
{
    if (!ARGUMENT_PRESENT(Jit) || !ARGUMENT_PRESENT(Jit->Table)) {
        return 0;
    }

    return Jit->Table->Vtbl->Release(Jit->Table);
}

_Use_decl_annotations_
HRESULT
PerfectHashTableJitInterfaceCreateInstance(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    PIUNKNOWN UnknownOuter,
    REFIID InterfaceId,
    PVOID *Interface
    )
{
    if (!ARGUMENT_PRESENT(Jit) || !ARGUMENT_PRESENT(Jit->Table)) {
        return E_POINTER;
    }

    return Jit->Table->Vtbl->CreateInstance(Jit->Table,
                                            UnknownOuter,
                                            InterfaceId,
                                            Interface);
}

_Use_decl_annotations_
HRESULT
PerfectHashTableJitInterfaceLockServer(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    BOOL Lock
    )
{
    if (!ARGUMENT_PRESENT(Jit) || !ARGUMENT_PRESENT(Jit->Table)) {
        return E_POINTER;
    }

    return Jit->Table->Vtbl->LockServer(Jit->Table, Lock);
}

_Use_decl_annotations_
HRESULT
PerfectHashTableJitInterfaceIndex(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    ULONG Key,
    PULONG Index
    )
{
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_JIT JitState;
    PPH_JIT_INDEX_FUNCTION IndexFunction;

    if (!ARGUMENT_PRESENT(Jit) || !ARGUMENT_PRESENT(Index)) {
        return E_POINTER;
    }

    Table = Jit->Table;
    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    JitState = Table->Jit;
    if (!ARGUMENT_PRESENT(JitState) ||
        !JitState->Flags.Valid ||
        !JitState->Flags.IndexCompiled) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction = (PPH_JIT_INDEX_FUNCTION)JitState->IndexFunction;
    if (!ARGUMENT_PRESENT(IndexFunction)) {
        return PH_E_NOT_IMPLEMENTED;
    }

    *Index = IndexFunction(Key);
    return S_OK;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableJitInterfaceIndex64(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    ULONGLONG Key,
    PULONG Index
    )
{
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_JIT JitState;
    PPH_JIT_INDEX64_FUNCTION IndexFunction;

    if (!ARGUMENT_PRESENT(Jit) || !ARGUMENT_PRESENT(Index)) {
        return E_POINTER;
    }

    Table = Jit->Table;
    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    JitState = Table->Jit;
    if (!ARGUMENT_PRESENT(JitState) ||
        !JitState->Flags.Valid ||
        !JitState->Flags.Index64Compiled) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction = (PPH_JIT_INDEX64_FUNCTION)JitState->Index64Function;
    if (!ARGUMENT_PRESENT(IndexFunction)) {
        return PH_E_NOT_IMPLEMENTED;
    }

    *Index = IndexFunction(Key);
    return S_OK;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableJitInterfaceIndex2(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    ULONG Key1,
    ULONG Key2,
    PULONG Index1,
    PULONG Index2
    )
{
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_JIT JitState;
    PPH_JIT_INDEX2_FUNCTION IndexFunction;

    if (!ARGUMENT_PRESENT(Jit) ||
        !ARGUMENT_PRESENT(Index1) ||
        !ARGUMENT_PRESENT(Index2)) {
        return E_POINTER;
    }

    Table = Jit->Table;
    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    JitState = Table->Jit;
    if (!ARGUMENT_PRESENT(JitState) ||
        !JitState->Flags.Valid ||
        !JitState->Flags.Index2Compiled) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction = (PPH_JIT_INDEX2_FUNCTION)JitState->Index2Function;
    if (!ARGUMENT_PRESENT(IndexFunction)) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction(Key1, Key2, Index1, Index2);
    return S_OK;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableJitInterfaceIndex4(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    ULONG Key1,
    ULONG Key2,
    ULONG Key3,
    ULONG Key4,
    PULONG Index1,
    PULONG Index2,
    PULONG Index3,
    PULONG Index4
    )
{
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_JIT JitState;
    PPH_JIT_INDEX4_FUNCTION IndexFunction;

    if (!ARGUMENT_PRESENT(Jit) ||
        !ARGUMENT_PRESENT(Index1) ||
        !ARGUMENT_PRESENT(Index2) ||
        !ARGUMENT_PRESENT(Index3) ||
        !ARGUMENT_PRESENT(Index4)) {
        return E_POINTER;
    }

    Table = Jit->Table;
    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    JitState = Table->Jit;
    if (!ARGUMENT_PRESENT(JitState) ||
        !JitState->Flags.Valid ||
        !JitState->Flags.Index4Compiled) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction = (PPH_JIT_INDEX4_FUNCTION)JitState->Index4Function;
    if (!ARGUMENT_PRESENT(IndexFunction)) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction(Key1,
                  Key2,
                  Key3,
                  Key4,
                  Index1,
                  Index2,
                  Index3,
                  Index4);
    return S_OK;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableJitInterfaceIndex8(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    ULONG Key1,
    ULONG Key2,
    ULONG Key3,
    ULONG Key4,
    ULONG Key5,
    ULONG Key6,
    ULONG Key7,
    ULONG Key8,
    PULONG Index1,
    PULONG Index2,
    PULONG Index3,
    PULONG Index4,
    PULONG Index5,
    PULONG Index6,
    PULONG Index7,
    PULONG Index8
    )
{
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_JIT JitState;
    PPH_JIT_INDEX8_FUNCTION IndexFunction;

    if (!ARGUMENT_PRESENT(Jit) ||
        !ARGUMENT_PRESENT(Index1) ||
        !ARGUMENT_PRESENT(Index2) ||
        !ARGUMENT_PRESENT(Index3) ||
        !ARGUMENT_PRESENT(Index4) ||
        !ARGUMENT_PRESENT(Index5) ||
        !ARGUMENT_PRESENT(Index6) ||
        !ARGUMENT_PRESENT(Index7) ||
        !ARGUMENT_PRESENT(Index8)) {
        return E_POINTER;
    }

    Table = Jit->Table;
    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    JitState = Table->Jit;
    if (!ARGUMENT_PRESENT(JitState) ||
        !JitState->Flags.Valid ||
        !JitState->Flags.Index8Compiled) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction = (PPH_JIT_INDEX8_FUNCTION)JitState->Index8Function;
    if (!ARGUMENT_PRESENT(IndexFunction)) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction(Key1,
                  Key2,
                  Key3,
                  Key4,
                  Key5,
                  Key6,
                  Key7,
                  Key8,
                  Index1,
                  Index2,
                  Index3,
                  Index4,
                  Index5,
                  Index6,
                  Index7,
                  Index8);
    return S_OK;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableJitInterfaceIndex2_64(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    ULONGLONG Key1,
    ULONGLONG Key2,
    PULONG Index1,
    PULONG Index2
    )
{
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_JIT JitState;
    PPH_JIT_INDEX2_64_FUNCTION IndexFunction;

    if (!ARGUMENT_PRESENT(Jit) ||
        !ARGUMENT_PRESENT(Index1) ||
        !ARGUMENT_PRESENT(Index2)) {
        return E_POINTER;
    }

    Table = Jit->Table;
    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    JitState = Table->Jit;
    if (!ARGUMENT_PRESENT(JitState) ||
        !JitState->Flags.Valid ||
        !JitState->Flags.Index2_64Compiled) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction = (PPH_JIT_INDEX2_64_FUNCTION)JitState->Index2_64Function;
    if (!ARGUMENT_PRESENT(IndexFunction)) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction(Key1, Key2, Index1, Index2);
    return S_OK;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableJitInterfaceIndex4_64(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    ULONGLONG Key1,
    ULONGLONG Key2,
    ULONGLONG Key3,
    ULONGLONG Key4,
    PULONG Index1,
    PULONG Index2,
    PULONG Index3,
    PULONG Index4
    )
{
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_JIT JitState;
    PPH_JIT_INDEX4_64_FUNCTION IndexFunction;

    if (!ARGUMENT_PRESENT(Jit) ||
        !ARGUMENT_PRESENT(Index1) ||
        !ARGUMENT_PRESENT(Index2) ||
        !ARGUMENT_PRESENT(Index3) ||
        !ARGUMENT_PRESENT(Index4)) {
        return E_POINTER;
    }

    Table = Jit->Table;
    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    JitState = Table->Jit;
    if (!ARGUMENT_PRESENT(JitState) ||
        !JitState->Flags.Valid ||
        !JitState->Flags.Index4_64Compiled) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction = (PPH_JIT_INDEX4_64_FUNCTION)JitState->Index4_64Function;
    if (!ARGUMENT_PRESENT(IndexFunction)) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction(Key1,
                  Key2,
                  Key3,
                  Key4,
                  Index1,
                  Index2,
                  Index3,
                  Index4);
    return S_OK;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableJitInterfaceIndex8_64(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    ULONGLONG Key1,
    ULONGLONG Key2,
    ULONGLONG Key3,
    ULONGLONG Key4,
    ULONGLONG Key5,
    ULONGLONG Key6,
    ULONGLONG Key7,
    ULONGLONG Key8,
    PULONG Index1,
    PULONG Index2,
    PULONG Index3,
    PULONG Index4,
    PULONG Index5,
    PULONG Index6,
    PULONG Index7,
    PULONG Index8
    )
{
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_JIT JitState;
    PPH_JIT_INDEX8_64_FUNCTION IndexFunction;

    if (!ARGUMENT_PRESENT(Jit) ||
        !ARGUMENT_PRESENT(Index1) ||
        !ARGUMENT_PRESENT(Index2) ||
        !ARGUMENT_PRESENT(Index3) ||
        !ARGUMENT_PRESENT(Index4) ||
        !ARGUMENT_PRESENT(Index5) ||
        !ARGUMENT_PRESENT(Index6) ||
        !ARGUMENT_PRESENT(Index7) ||
        !ARGUMENT_PRESENT(Index8)) {
        return E_POINTER;
    }

    Table = Jit->Table;
    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    JitState = Table->Jit;
    if (!ARGUMENT_PRESENT(JitState) ||
        !JitState->Flags.Valid ||
        !JitState->Flags.Index8_64Compiled) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction = (PPH_JIT_INDEX8_64_FUNCTION)JitState->Index8_64Function;
    if (!ARGUMENT_PRESENT(IndexFunction)) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction(Key1,
                  Key2,
                  Key3,
                  Key4,
                  Key5,
                  Key6,
                  Key7,
                  Key8,
                  Index1,
                  Index2,
                  Index3,
                  Index4,
                  Index5,
                  Index6,
                  Index7,
                  Index8);
    return S_OK;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableIndexJit(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    PULONG Index
    )
{
    PPERFECT_HASH_TABLE_JIT Jit;
    PPH_JIT_INDEX_FUNCTION IndexFunction;

    if (!ARGUMENT_PRESENT(Table) || !ARGUMENT_PRESENT(Index)) {
        return E_POINTER;
    }

    Jit = Table->Jit;
    if (!ARGUMENT_PRESENT(Jit) ||
        !Jit->Flags.Valid ||
        !Jit->Flags.IndexCompiled) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction = (PPH_JIT_INDEX_FUNCTION)Jit->IndexFunction;
    if (!ARGUMENT_PRESENT(IndexFunction)) {
        return PH_E_NOT_IMPLEMENTED;
    }

    *Index = IndexFunction(Key);
    return S_OK;
}

_Use_decl_annotations_
VOID
PerfectHashTableJitRundown(
    PPERFECT_HASH_TABLE Table
    )
{
    PALLOCATOR Allocator;
    PPERFECT_HASH_TABLE_JIT Jit;

    if (!ARGUMENT_PRESENT(Table)) {
        return;
    }

    Jit = Table->Jit;
    if (!ARGUMENT_PRESENT(Jit)) {
        return;
    }

    if (Jit->OriginalIndex) {
        Table->Vtbl->Index = Jit->OriginalIndex;
    }

    if (Jit->OriginalQueryInterface) {
        Table->Vtbl->QueryInterface = Jit->OriginalQueryInterface;
    }

    Table->Flags.JitEnabled = FALSE;

#if defined(PH_HAS_LLVM)
    if (Jit->ExecutionEngine) {
        LLVMDisposeExecutionEngine(
            (LLVMExecutionEngineRef)Jit->ExecutionEngine
        );
        Jit->ExecutionEngine = NULL;
    }

    if (Jit->Context) {
        LLVMContextDispose((LLVMContextRef)Jit->Context);
        Jit->Context = NULL;
    }
#endif

    Allocator = Table->Allocator;
    Allocator->Vtbl->FreePointer(Allocator, (PVOID *)&Table->Jit);
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
