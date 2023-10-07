/*++

Copyright (c) 2019-2020 Trent Nelson <trent@trent.me>

Module Name:

    Cu.c

Abstract:

    This is the implementation file for the NVIDIA CUDA component.  It
    implements the InitCu() function.
--*/

#include "stdafx.h"

extern LOAD_SYMBOLS LoadSymbols;

#ifdef PH_WINDOWS
#define PH_NVCUDA_DLL_NAME "nvcuda.dll"
#define PERFECT_HASH_CUDA_DLL_NAME "PerfectHashCuda.dll"
#else
#define PH_NVCUDA_DLL_NAME "libcuda.so"
#define PERFECT_HASH_CUDA_DLL_NAME "PerfectHashCuda.so"
#endif

//
// Forward decl.
//

HRESULT
CuLoadKernels(
    _In_ PCU Cu
    );

//
// COM scaffolding routines for initialization and rundown.
//

CU_INITIALIZE CuInitialize;

_Use_decl_annotations_
HRESULT
CuInitialize(
    PCU Cu
    )
/*++

Routine Description:

    Initializes a CU instance.

Arguments:

    Cu - Supplies a pointer to a CU structure for which initialization
        is to be performed.

Return Value:

    S_OK - Success.

    E_POINTER - Cu is NULL.

    E_UNEXPECTED - All other errors.

--*/
{
    HRESULT Result;
    HMODULE Module;
    BOOLEAN Success;
    CU_RESULT CuResult;
    ULONG NumberOfCuNames;
    ULONG NumberOfCuResolvedSymbols;
    ULONG NumberOfCuRandNames;
    ULONG NumberOfCuRandResolvedSymbols;
    ULONG NumberOfPerfectHashCudaNames;
    ULONG NumberOfPerfectHashCudaResolvedSymbols;
#ifdef PH_WINDOWS
    CHAR CuRandDll[] = "curand64_NM.dll";
    const BYTE CuRandDllVersionOffset = 9;
#else
    CHAR CuRandDll[] = "libcurand.so.10";
    const BYTE CuRandDllVersionOffset = 13;
#endif

#define EXPAND_AS_CU_NAME(Upper, Name) "cu" # Name,
#define EXPAND_AS_CU_V2_NAME(Upper, Name) "cu" # Name "_v2",

    CONST PCSZ CuNames[] = {
        CU_FUNCTION_TABLE_ENTRY(EXPAND_AS_CU_NAME)
        CU_FUNCTION_V2_TABLE_ENTRY(EXPAND_AS_CU_V2_NAME)
    };

#define EXPAND_AS_CURAND_NAME(Upper, Name) "curand"#Name,
    CONST PCSZ CuRandNames[] = {
        CURAND_FUNCTION_TABLE_ENTRY(EXPAND_AS_CURAND_NAME)
    };

#define EXPAND_AS_PERFECT_HASH_CUDA_NAME(Upper, Name) #Name,

    CONST PCSZ PerfectHashCudaNames[] = {
        PERFECT_HASH_CUDA_FUNCTION_TABLE_ENTRY(EXPAND_AS_PERFECT_HASH_CUDA_NAME)
    };

    //
    // Define appropriately-sized bitmaps we can passed to LoadSymbols().
    //

    ULONG CuBitmapBuffer[(ALIGN_UP(ARRAYSIZE(CuNames),
                         sizeof(ULONG) << 3) >> 5)+1] = { 0 };
    RTL_BITMAP CuFailedBitmap;

    ULONG CuRandBitmapBuffer[(ALIGN_UP(ARRAYSIZE(CuRandNames),
                             sizeof(ULONG) << 3) >> 5)+1] = { 0 };
    RTL_BITMAP CuRandFailedBitmap;

    ULONG PerfectHashCudaBitmapBuffer[(ALIGN_UP(ARRAYSIZE(PerfectHashCudaNames),
                                      sizeof(ULONG) << 3) >> 5)+1] = { 0 };
    RTL_BITMAP PerfectHashCudaFailedBitmap;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Cu)) {
        return E_POINTER;
    }

    Cu->SizeOfStruct = sizeof(*Cu);

    //
    // Continue initialization.  Wire up the FailedBitmap.
    //

    CuFailedBitmap.SizeOfBitMap = ARRAYSIZE(CuNames) + 1;
    CuFailedBitmap.Buffer = (PULONG)&CuBitmapBuffer;

    NumberOfCuNames = ARRAYSIZE(CuNames);

    Cu->NumberOfCuFunctions = sizeof(Cu->CuFunctions) / sizeof(ULONG_PTR);
    ASSERT(Cu->NumberOfCuFunctions == NumberOfCuNames);

    Module = LoadLibraryA(PH_NVCUDA_DLL_NAME);
    if (!IsValidHandle(Module)) {
        Result = PH_E_NVCUDA_DLL_LOAD_LIBRARY_FAILED;
        goto Error;
    }
    Cu->NvCudaModule = Module;

    Success = LoadSymbols(CuNames,
                          NumberOfCuNames,
                          (PULONG_PTR)&Cu->CuFunctions,
                          Cu->NumberOfCuFunctions,
                          Module,
                          &CuFailedBitmap,
                          &NumberOfCuResolvedSymbols);

    if (!Success) {
        Result = PH_E_NVCUDA_DLL_LOAD_SYMBOLS_FAILED;
        goto Error;
    }

    if (Cu->NumberOfCuFunctions != NumberOfCuResolvedSymbols) {
        Result = PH_E_NVCUDA_DLL_LOAD_SYMBOLS_FAILED_TO_LOAD_ALL_SYMBOLS;
        goto Error;
    }

    //
    // Initialize CUDA.
    //

    CuResult = Cu->Init(0);
    if (CU_FAILED(CuResult)) {
        CU_ERROR(CuInitialize_Init, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    //
    // Get the number of devices.
    //

    CuResult = Cu->DeviceGetCount(&Cu->NumberOfDevices);
    if (CU_FAILED(CuResult)) {
        CU_ERROR(CuInitialize_DeviceGetCount, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    //
    // Get the driver version.
    //

    CuResult = Cu->DriverGetVersion(&Cu->DriverVersionRaw);
    CU_CHECK(CuResult, DriverGetVersion);

    //
    // The returned driver version is constructed via `major * 1000 + minor *
    // 10`; extract the major/minor parts now.
    //

    Cu->DriverMajor = (USHORT)(Cu->DriverVersionRaw / (LONG)1000);
    Cu->DriverMinor = (USHORT)(
        (Cu->DriverVersionRaw - (((LONG)Cu->DriverMajor * (LONG)1000))) /
        (LONG)10
    );

    if (Cu->DriverMajor < 10) {
        Cu->DriverSuffix[0] = '0' + (CHAR)Cu->DriverMajor;
        Cu->DriverSuffix[1] = '0';
    } else {
        Cu->DriverSuffix[0] = '0' + (CHAR)(Cu->DriverMajor / (USHORT)10);
        Cu->DriverSuffix[1] = '0' + (CHAR)(Cu->DriverMajor % (USHORT)10);
    }

    //
    // Sanity check we've parsed the version properly.
    //

    ASSERT(
        (Cu->DriverSuffix[0] == '8' && Cu->DriverSuffix[1] == '0') ||
        (Cu->DriverSuffix[0] == '9' && Cu->DriverSuffix[1] == '0') ||
        (Cu->DriverSuffix[0] == '1' && Cu->DriverSuffix[1] == '0') ||
        (Cu->DriverSuffix[0] == '1' && Cu->DriverSuffix[1] == '1') ||
        (Cu->DriverSuffix[0] == '1' && Cu->DriverSuffix[1] == '2')
    );

#ifdef PH_WINDOWS
    //
    // Overlay the version into the curand dll name.
    //

    ASSERT(CuRandDll[CuRandDllVersionOffset] == 'N');
    ASSERT(CuRandDll[CuRandDllVersionOffset+1] == 'M');
#endif

    //
    // Temp hack: we only support curand64_10.dll for now.
    //

#if 0
    CuRandDll[CuRandDllVersionOffset]   = Cu->DriverSuffix[0];
    CuRandDll[CuRandDllVersionOffset+1] = Cu->DriverSuffix[1];
#else
    CuRandDll[CuRandDllVersionOffset]   = '1';
    CuRandDll[CuRandDllVersionOffset+1] = '0';
#endif

    Module = LoadLibraryA((PCSZ)CuRandDll);
    if (!IsValidHandle(Module)) {
        Result = PH_E_CURAND_DLL_LOAD_LIBRARY_FAILED;
        goto Error;
    }
    Cu->CuRandModule = Module;

    //
    // Wire up the CuRandFailedBitmap.
    //

    CuRandFailedBitmap.SizeOfBitMap = ARRAYSIZE(CuRandNames) + 1;
    CuRandFailedBitmap.Buffer = (PULONG)&CuRandBitmapBuffer;

    NumberOfCuRandNames = ARRAYSIZE(CuRandNames);

    Cu->NumberOfCuRandFunctions = (
        sizeof(Cu->CuRandFunctions) /
        sizeof(ULONG_PTR)
    );

    ASSERT(Cu->NumberOfCuRandFunctions == NumberOfCuRandNames);

    Success = LoadSymbols(CuRandNames,
                          NumberOfCuRandNames,
                          (PULONG_PTR)&Cu->CuRandFunctions,
                          Cu->NumberOfCuRandFunctions,
                          Module,
                          &CuRandFailedBitmap,
                          &NumberOfCuRandResolvedSymbols);

    if (!Success) {
        Result = PH_E_CURAND_DLL_LOAD_SYMBOLS_FAILED;
        goto Error;
    }

    if (Cu->NumberOfCuRandFunctions != NumberOfCuRandResolvedSymbols) {
        Result = PH_E_CURAND_DLL_LOAD_SYMBOLS_FAILED_TO_LOAD_ALL_SYMBOLS;
        goto Error;
    }

    //
    // Load PerfectHashCuda.dll's symbols.
    //

    PerfectHashCudaFailedBitmap.SizeOfBitMap = ARRAYSIZE(PerfectHashCudaNames) + 1;
    PerfectHashCudaFailedBitmap.Buffer = (PULONG)&PerfectHashCudaBitmapBuffer;

    NumberOfPerfectHashCudaNames = ARRAYSIZE(PerfectHashCudaNames);

    Cu->NumberOfPerfectHashCudaFunctions = (
        sizeof(Cu->PerfectHashCudaFunctions) /
        sizeof(ULONG_PTR)
    );
    ASSERT(Cu->NumberOfPerfectHashCudaFunctions == NumberOfPerfectHashCudaNames);

    Module = LoadLibraryA(PERFECT_HASH_CUDA_DLL_NAME);
    if (!IsValidHandle(Module)) {
        Result = PH_E_PERFECT_HASH_CUDA_DLL_LOAD_LIBRARY_FAILED;
        goto Error;
    }
    Cu->PerfectHashCudaModule = Module;

    Success = LoadSymbols(PerfectHashCudaNames,
                          NumberOfPerfectHashCudaNames,
                          (PULONG_PTR)&Cu->PerfectHashCudaFunctions,
                          Cu->NumberOfPerfectHashCudaFunctions,
                          Module,
                          &PerfectHashCudaFailedBitmap,
                          &NumberOfPerfectHashCudaResolvedSymbols);

    if (!Success) {
        Result = PH_E_PERFECT_HASH_CUDA_DLL_LOAD_SYMBOLS_FAILED;
        goto Error;
    }

    if (Cu->NumberOfPerfectHashCudaFunctions !=
        NumberOfPerfectHashCudaResolvedSymbols)
    {
        Result =
            PH_E_PERFECT_HASH_CUDA_DLL_LOAD_SYMBOLS_FAILED_TO_LOAD_ALL_SYMBOLS;
        goto Error;
    }

    //
    // We're done!  Indicate success and finish up.
    //

    Result = S_OK;
    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}

#if 0
HRESULT
CuLoadKernels(
    _In_ PCU Cu
    )
/*++

Routine Description:

    Loads kernels.

Arguments:

    Cu - Supplies a pointer to a CU structure for which initialization
        is to be performed.

Return Value:

    S_OK - Success.

    E_POINTER - Cu is NULL.

    E_UNEXPECTED - All other errors.

--*/
{
    HRESULT Result;
    CU_RESULT CuResult;
    PCU_MODULE Module;
    PCU_FUNCTION *Function;
    PCU_OCCUPANCY Occupancy;
    PCU_CONTEXT CuContext;
    PCSZ FunctionName;
    CU_DEVICE DeviceOrdinal;
    PPERFECT_HASH_CONTEXT Context;
    PPERFECT_HASH_TLS_CONTEXT TlsContext;

    PCSZ KernelFunctionNames[] = {
        (PCSZ)"PerfectHashCudaSeededHashAllMultiplyShiftR2",
    };

    CU_JIT_OPTION JitOptions[] = {
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_MAX_REGISTERS,
    };
    PVOID JitOptionValues[3];
    USHORT NumberOfJitOptions = ARRAYSIZE(JitOptions);

    //
    // Load the TLS context and device ordinal.
    //

    TlsContext = PerfectHashTlsEnsureContext();
    Context = TlsContext->Context;
    DeviceOrdinal = 0;
    //DeviceOrdinal = Context->CuDeviceOrdinal;

    //
    // Initialize the JIT options.
    //

    JitOptionValues[0] = (PVOID)sizeof(Cu->JitLogBuffer);
    JitOptionValues[1] = (PVOID)Cu->JitLogBuffer;
    //JitOptionValues[2] = (PVOID)Cu->JitMaxNumberOfRegisters;
    JitOptionValues[2] = (PVOID)64LL;

    //
    // Create a CUDA context.
    //

    CuResult = Cu->CtxCreate(&CuContext, CU_CTX_SCHED_AUTO, DeviceOrdinal);
    if (CU_FAILED(CuResult)) {
        CU_ERROR(CuCtxCreate, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    //
    // Load the module from the embedded PTX.
    //

    CuResult = Cu->ModuleLoadDataEx(&Module,
                                    (PCHAR)GraphPtxRawCStr,
                                    NumberOfJitOptions,
                                    JitOptions,
                                    JitOptionValues);
    if (CU_FAILED(CuResult)) {
        CU_ERROR(CuModuleLoadDataEx, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    //
    // Module loaded successfully, resolve the kernel.
    //

    Function = &Cu->Functions[0];
    FunctionName = KernelFunctionNames[0];
    CuResult = Cu->ModuleGetFunction(Function, Module, FunctionName);
    if (CU_FAILED(CuResult)) {
        CU_ERROR(CuModuleGetFunction, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    //
    // Get occupancy stats for each function.
    //

    Occupancy = &Cu->Occupancy[0];

    CuResult = Cu->OccupancyMaxPotentialBlockSizeWithFlags(
        &Occupancy->MinimumGridSize,
        &Occupancy->BlockSize,
        *Function,
        NULL,   // OccupancyBlockSizeToDynamicMemSize
        0,      // DynamicSharedMemorySize
        0,      // BlockSizeLimit
        0       // Flags
    );
    if (CU_FAILED(CuResult)) {
        CU_ERROR(OccupancyMaxPotentialBlockSizeWithFlags, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    CuResult = Cu->OccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        &Occupancy->NumBlocks,
        *Function,
        Occupancy->BlockSize,
        0, // DynamicSharedMemorySize
        0  // Flags
    );
    if (CU_FAILED(CuResult)) {
        CU_ERROR(OccupancyMaxActiveBlocksPerMultiprocessorWithFlags, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    //
    // We're done, finish up.
    //

    Result = S_OK;
    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}
#endif


CU_RUNDOWN CuRundown;

_Use_decl_annotations_
VOID
CuRundown(
    PCU Cu
    )
/*++

Routine Description:

    Release all resources associated with a CU instance.

Arguments:

    Cu - Supplies a pointer to a CU structure to rundown.

Return Value:

    None.

--*/
{
    //
    // Sanity check structure size.
    //

    ASSERT(Cu->SizeOfStruct == sizeof(*Cu));

    //
    // Release the PerfectHashCuda.dll module if applicable.
    //

    if (Cu->PerfectHashCudaModule != NULL) {
        FreeLibrary(Cu->PerfectHashCudaModule);
        Cu->PerfectHashCudaModule = NULL;
    }

    //
    // Release the curand module if applicable.
    //

    if (Cu->CuRandModule != NULL) {
        FreeLibrary(Cu->CuRandModule);
        Cu->CuRandModule = NULL;
    }

    //
    // Release the nvcuda.dll module if applicable.
    //

    if (Cu->NvCudaModule != NULL) {
        FreeLibrary(Cu->NvCudaModule);
        Cu->NvCudaModule = NULL;
    }

    //
    // Release applicable COM references.
    //

    RELEASE(Cu->Rtl);
    RELEASE(Cu->Allocator);

    return;
}

_Use_decl_annotations_
CU_RESULT
LoadCuDeviceAttributes(
    PCU Cu,
    PCU_DEVICE_ATTRIBUTES AttributesPointer,
    CU_DEVICE Device
    )
/*++

Routine Description:

    Loads CUDA device attributes for a given device.

Arguments:

    Cu - Supplies the CU instance.

    AttributesPointer - Supplies a pointer to a CU_DEVICE_ATTRIBUTES struct
        that will receive the attributes for the given device.

    Device - Supplies the CUDA device.

Return Value:

    CUDA_SUCCESS on success, otherwise a CUDA error code.

--*/
{
    PRTL Rtl;
    LONG Index;
    LONG NumberOfAttributes;
    PLONG Attribute;
    CU_RESULT Result;

    Rtl = Cu->Rtl;
    ZeroStructPointer(AttributesPointer);

    Attribute = (PLONG)AttributesPointer;
    NumberOfAttributes = sizeof(*AttributesPointer) / sizeof(ULONG);

    for (Index = 0; Index < NumberOfAttributes; Index++) {
        Result = Cu->DeviceGetAttribute(Attribute, Index+1, Device);
        if (CU_FAILED(Result)) {
            continue;
        }
        Attribute++;
    }

    return CUDA_SUCCESS;
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
