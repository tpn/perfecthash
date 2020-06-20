/*++

Copyright (c) 2017-2023 Trent Nelson <trent@trent.me>

Module Name:

    Cu.h

Abstract:

    This is the header file for the NVIDIA CUDA component.  It defines types
    and functions exposed by the CUDA API, such that it can be used dynamically
    at runtime if nvcuda.dll is present.

--*/

#pragma once

//
// We share this module with the PerfectHashCuda component, which doesn't use
// the precompiled header infrastructure, so, we need to include stdafx.h within
// this guard.
//

#ifdef _PERFECT_HASH_INTERNAL_BUILD
#include "stdafx.h"
#endif

//
// Define CUDA Device API Typedefs.
//

typedef LONG CU_DEVICE;
typedef ULONG_PTR CU_DEVICE_POINTER;
typedef CU_DEVICE *PCU_DEVICE;
typedef CU_DEVICE **PPCU_DEVICE;
typedef CU_DEVICE_POINTER *PCU_DEVICE_POINTER;
typedef CU_DEVICE_POINTER **PPCU_DEVICE_POINTER;

struct CU_CONTEXT;
typedef struct CU_CONTEXT *PCU_CONTEXT;
typedef struct CU_CONTEXT **PPCU_CONTEXT;

struct CU_MODULE;
typedef struct CU_MODULE *PCU_MODULE;
typedef struct CU_MODULE **PPCU_MODULE;

struct CU_EVENT;
typedef struct CU_EVENT *PCU_EVENT;
typedef struct CU_EVENT **PPCU_EVENT;

struct CU_STREAM;
typedef struct CU_STREAM *CU_STREAM;
typedef CU_STREAM *PCU_STREAM;

struct CU_FUNCTION;
typedef struct CU_FUNCTION *PCU_FUNCTION;
typedef struct CU_FUNCTION **PPCU_FUNCTION;

struct CU_LINK_STATE;
typedef struct CU_LINK_STATE *PCU_LINK_STATE;
typedef struct CU_LINK_STATE **PPCU_LINK_STATE;

#define CUDACBCALLTYPE __stdcall

typedef enum _Return_type_success_(return == 0) _CU_RESULT {

    //
    // The API call returned with no errors. In the case of query calls, this
    // can also mean that the operation being queried is complete (see
    // ::cuEventQuery() and ::cuStreamQuery()).
    //

    CUDA_SUCCESS                              = 0,

    //
    // This indicates that one or more of the parameters passed to the API call
    // is not within an acceptable range of values.
    //

    CUDA_ERROR_INVALID_VALUE                  = 1,

    //
    // The API call failed because it was unable to allocate enough memory to
    // perform the requested operation.
    //

    CUDA_ERROR_OUT_OF_MEMORY                  = 2,

    //
    // This indicates that the CUDA driver has not been initialized with
    // ::cuInit() or that initialization has failed.
    //

    CUDA_ERROR_NOT_INITIALIZED                = 3,

    //
    // This indicates that the CUDA driver is in the process of shutting down.
    //

    CUDA_ERROR_DEINITIALIZED                  = 4,

    //
    // This indicates profiling APIs are called while application is running
    // in visual profiler mode.
    //

    CUDA_ERROR_PROFILER_DISABLED           = 5,

    //
    // This indicates profiling has not been initialized for this context.
    // Call cuProfilerInitialize() to resolve this.
    //

    CUDA_ERROR_PROFILER_NOT_INITIALIZED       = 6,

    //
    // This indicates profiler has already been started and probably
    // cuProfilerStart() is incorrectly called.
    //

    CUDA_ERROR_PROFILER_ALREADY_STARTED       = 7,

    //
    // This indicates profiler has already been stopped and probably
    // cuProfilerStop() is incorrectly called.
    //

    CUDA_ERROR_PROFILER_ALREADY_STOPPED       = 8,

    //
    // This indicates that no CUDA-capable devices were detected by the
    // installed CUDA driver.
    //

    CUDA_ERROR_NO_DEVICE                      = 100,

    //
    // This indicates that the device ordinal supplied by the user does not
    // correspond to a valid CUDA device.
    //

    CUDA_ERROR_INVALID_DEVICE                 = 101,

    //
    // This indicates that the device kernel image is invalid. This can also
    // indicate an invalid CUDA module.
    //

    CUDA_ERROR_INVALID_IMAGE                  = 200,

    //
    // This most frequently indicates that there is no context bound to the
    // current thread. This can also be returned if the context passed to an
    // API call is not a valid handle (such as a context that has had
    // ::cuCtxDestroy() invoked on it). This can also be returned if a user
    // mixes different API versions (i.e. 3010 context with 3020 API calls).
    // See ::cuCtxGetApiVersion() for more details.
    //

    CUDA_ERROR_INVALID_CONTEXT                = 201,

    //
    // This indicated that the context being supplied as a parameter to the
    // API call was already the active context.
    // \deprecated
    // This error return is deprecated as of CUDA 3.2. It is no longer an
    // error to attempt to push the active context via ::cuCtxPushCurrent().
    //

    CUDA_ERROR_CONTEXT_ALREADY_CURRENT        = 202,

    //
    // This indicates that a map or register operation has failed.
    //

    CUDA_ERROR_MAP_FAILED                     = 205,

    //
    // This indicates that an unmap or unregister operation has failed.
    //

    CUDA_ERROR_UNMAP_FAILED                   = 206,

    //
    // This indicates that the specified array is currently mapped and thus
    // cannot be destroyed.
    //

    CUDA_ERROR_ARRAY_IS_MAPPED                = 207,

    //
    // This indicates that the resource is already mapped.
    //

    CUDA_ERROR_ALREADY_MAPPED                 = 208,

    //
    // This indicates that there is no kernel image available that is suitable
    // for the device. This can occur when a user specifies code generation
    // options for a particular CUDA source file that do not include the
    // corresponding device configuration.
    //

    CUDA_ERROR_NO_BINARY_FOR_GPU              = 209,

    //
    // This indicates that a resource has already been acquired.
    //

    CUDA_ERROR_ALREADY_ACQUIRED               = 210,

    //
    // This indicates that a resource is not mapped.
    //

    CUDA_ERROR_NOT_MAPPED                     = 211,

    //
    // This indicates that a mapped resource is not available for access as an
    // array.
    //

    CUDA_ERROR_NOT_MAPPED_AS_ARRAY            = 212,

    //
    // This indicates that a mapped resource is not available for access as a
    // pointer.
    //

    CUDA_ERROR_NOT_MAPPED_AS_POINTER          = 213,

    //
    // This indicates that an uncorrectable ECC error was detected during
    // execution.
    //

    CUDA_ERROR_ECC_UNCORRECTABLE              = 214,

    //
    // This indicates that the ::CUlimit passed to the API call is not
    // supported by the active device.
    //

    CUDA_ERROR_UNSUPPORTED_LIMIT              = 215,

    //
    // This indicates that the ::CUcontext passed to the API call can
    // only be bound to a single CPU thread at a time but is already
    // bound to a CPU thread.
    //

    CUDA_ERROR_CONTEXT_ALREADY_IN_USE         = 216,

    //
    // This indicates that the device kernel source is invalid.
    //

    CUDA_ERROR_INVALID_SOURCE                 = 300,

    //
    // This indicates that the file specified was not found.
    //

    CUDA_ERROR_FILE_NOT_FOUND                 = 301,

    //
    // This indicates that a link to a shared object failed to resolve.
    //

    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,

    //
    // This indicates that initialization of a shared object failed.
    //

    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED      = 303,

    //
    // This indicates that an OS call failed.
    //

    CUDA_ERROR_OPERATING_SYSTEM               = 304,


    //
    // This indicates that a resource handle passed to the API call was not
    // valid. Resource handles are opaque types like ::CUstream and ::CUevent.
    //

    CUDA_ERROR_INVALID_HANDLE                 = 400,


    //
    // This indicates that a named symbol was not found. Examples of symbols
    // are global/constant variable names, texture names, and surface names.
    //

    CUDA_ERROR_NOT_FOUND                      = 500,


    //
    // This indicates that asynchronous operations issued previously have not
    // completed yet. This result is not actually an error, but must be
    // indicated differently than ::CUDA_SUCCESS (which indicates completion).
    // Calls that may return this value include ::cuEventQuery() and
    // ::cuStreamQuery().
    //

    CUDA_ERROR_NOT_READY                      = 600,


    //
    // An exception occurred on the device while executing a kernel. Common
    // causes include dereferencing an invalid device pointer and accessing
    // out of bounds shared memory. The context cannot be used, so it must
    // be destroyed (and a new one should be created). All existing device
    // memory allocations from this context are invalid and must be
    // reconstructed if the program is to continue using CUDA.
    //

    CUDA_ERROR_LAUNCH_FAILED                  = 700,

    //
    // This indicates that a launch did not occur because it did not have
    // appropriate resources. This error usually indicates that the user has
    // attempted to pass too many arguments to the device kernel, or the
    // kernel launch specifies too many threads for the kernel's register
    // count. Passing arguments of the wrong size (i.e. a 64-bit pointer
    // when a 32-bit int is expected) is equivalent to passing too many
    // arguments and can also result in this error.
    //

    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES        = 701,

    //
    // This indicates that the device kernel took too long to execute. This can
    // only occur if timeouts are enabled - see the device attribute
    // ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. The
    // context cannot be used (and must be destroyed similar to
    // ::CUDA_ERROR_LAUNCH_FAILED). All existing device memory allocations from
    // this context are invalid and must be reconstructed if the program is to
    // continue using CUDA.
    //

    CUDA_ERROR_LAUNCH_TIMEOUT                 = 702,

    //
    // This error indicates a kernel launch that uses an incompatible texturing
    // mode.
    //

    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  = 703,

    //
    // This error indicates that a call to ::cuCtxEnablePeerAccess() is
    // trying to re-enable peer access to a context which has already
    // had peer access to it enabled.
    //

    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,

    //
    // This error indicates that a call to ::cuMemPeerRegister is trying to
    // register memory from a context which has not had peer access
    // enabled yet via ::cuCtxEnablePeerAccess(), or that
    // ::cuCtxDisablePeerAccess() is trying to disable peer access
    // which has not been enabled yet.
    //

    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED    = 705,

    //
    // This error indicates that a call to ::cuMemPeerRegister is trying to
    // register already-registered memory.
    //

    CUDA_ERROR_PEER_MEMORY_ALREADY_REGISTERED = 706,

    //
    // This error indicates that a call to ::cuMemPeerUnregister is trying to
    // unregister memory that has not been registered.
    //

    CUDA_ERROR_PEER_MEMORY_NOT_REGISTERED     = 707,

    //
    // This error indicates that ::cuCtxCreate was called with the flag
    // ::CU_CTX_PRIMARY on a device which already has initialized its
    // primary context.
    //

    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE         = 708,

    //
    // This error indicates that the context current to the calling thread
    // has been destroyed using ::cuCtxDestroy, or is a primary context which
    // has not yet been initialized.
    //

    CUDA_ERROR_CONTEXT_IS_DESTROYED           = 709,

    //
    // A device-side assert triggered during kernel execution. The context
    // cannot be used anymore, and must be destroyed. All existing device
    // memory allocations from this context are invalid and must be
    // reconstructed if the program is to continue using CUDA.
    //

    CUDA_ERROR_ASSERT                         = 710,

    //
    // This error indicates that the hardware resources required to enable
    // peer access have been exhausted for one or more of the devices
    // passed to ::cuCtxEnablePeerAccess().
    //

    CUDA_ERROR_TOO_MANY_PEERS                 = 711,

    //
    // This error indicates that the memory range passed to ::cuMemHostRegister()
    // has already been registered.
    //

    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,

    //
    // This error indicates that the pointer passed to ::cuMemHostUnregister()
    // does not correspond to any currently registered memory region.
    //

    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED     = 713,

    //
    // This indicates that an unknown internal error has occurred.
    //

    CUDA_ERROR_UNKNOWN                        = 999

} CU_RESULT;

#define CU_SUCCEEDED(Result) (Result == CUDA_SUCCESS)
#define CU_FAILED(Result) (Result != CUDA_SUCCESS)

typedef enum _CU_LIMIT {
    CU_LIMIT_STACK_SIZE                       = 0x00, /**< GPU thread stack size */
    CU_LIMIT_PRINTF_FIFO_SIZE                 = 0x01, /**< GPU printf FIFO size */
    CU_LIMIT_MALLOC_HEAP_SIZE                 = 0x02, /**< GPU malloc heap size */
    CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH           = 0x03, /**< GPU device runtime launch synchronize depth */
    CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 0x04, /**< GPU device runtime pending launch count */
    CU_LIMIT_MAX_L2_FETCH_GRANULARITY         = 0x05, /**< A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a hint */
    CU_LIMIT_MAX
} CU_LIMIT;

typedef enum _Enum_is_bitflag_ _CU_CTX_CREATE_FLAGS {
    CU_CTX_SCHED_AUTO           = 0x00,
    CU_CTX_SCHED_SPIN           = 0x01,
    CU_CTX_SCHED_YIELD          = 0x02,
    CU_CTX_SCHED_BLOCKING_SYNC  = 0x04,
    CU_CTX_BLOCKING_SYNC        = 0x04,
    CU_CTX_MAP_HOST             = 0x08,
    CU_CTX_LMEM_RESIZE_TO_MAX   = 0x10,
    CU_CTX_SCHED_MASK           = 0x07,
    CU_CTX_PRIMARY              = 0x20,
    CU_CTX_FLAGS_MASK           = 0x3f
} CU_CTX_CREATE_FLAGS;

typedef enum _Enum_is_bitflag_ _CU_EVENT_FLAGS {
    CU_EVENT_DEFAULT            = 0x0,
    CU_EVENT_BLOCKING_SYNC      = 0x1,
    CU_EVENT_DISABLE_TIMING     = 0x2,
    CU_EVENT_INTERPROCESS       = 0x4
} CU_EVENT_FLAGS;

typedef enum _Enum_is_bitflag_ _CU_STREAM_FLAGS {
    CU_STREAM_DEFAULT           =   0x0,
    CU_STREAM_NON_BLOCKING      =   0x1
} CU_STREAM_FLAGS;

typedef enum _Enum_is_bitflag_ _CU_MEM_ATTACH_FLAGS {
    CU_MEM_ATTACH_GLOBAL        = 0x1,
    CU_MEM_ATTACH_HOST          = 0x2,
    CU_MEM_ATTACH_SINGLE        = 0x4
} CU_MEM_ATTACH_FLAGS;

typedef enum _Enum_is_bitflag_ _CU_STREAM_WAIT_VALUE {
    CU_STREAM_WAIT_VALUE_GEQ   = 0x0,
    CU_STREAM_WAIT_VALUE_EQ    = 0x1,
    CU_STREAM_WAIT_VALUE_AND   = 0x2,
    CU_STREAM_WAIT_VALUE_FLUSH = 1<<30
} CU_STREAM_WAIT_VALUE;

typedef enum _Enum_is_bitflag_ _CU_STREAM_WRITE_VALUE {
    CU_STREAM_WRITE_VALUE_DEFAULT           = 0x0,
    CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = 0x1
} CU_STREAM_WRITE_VALUE;

typedef enum _CU_STREAM_BATCH_MEM_OP_TYPE {
    CU_STREAM_MEM_OP_WAIT_VALUE_32          = 1,
    CU_STREAM_MEM_OP_WRITE_VALUE_32         = 2,
    CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES    = 3
} CU_STREAM_BATCH_MEM_OP_TYPE;
typedef CU_STREAM_BATCH_MEM_OP_TYPE *PCU_STREAM_BATCH_MEM_OP_TYPE;

typedef enum _CU_JIT_INPUT_TYPE {
    CU_JIT_INPUT_CUBIN = 0,
    CU_JIT_INPUT_PTX,
    CU_JIT_INPUT_FATBINARY,
    CU_JIT_INPUT_OBJECT,
    CU_JIT_INPUT_LIBRARY,
    CU_JIT_NUM_INPUT_TYPES
} CU_JIT_INPUT_TYPE;
typedef CU_JIT_INPUT_TYPE *PCU_JIT_INPUT_TYPE;

//
// Disable padding warnings.
//

#pragma warning(push)
#pragma warning(disable: 4820)

typedef union _CU_STREAM_BATCH_MEM_OP_PARAMS {
    CU_STREAM_BATCH_MEM_OP_TYPE Operation;
    struct {
        CU_STREAM_BATCH_MEM_OP_TYPE Operation;
        PCU_DEVICE_POINTER Address;
        ULONG Value;
        ULONG Padding2;
        CU_STREAM_WAIT_VALUE Flags;
        ULONG Padding3;
        PCU_DEVICE_POINTER Alias;
    } Wait;
    struct {
        CU_STREAM_BATCH_MEM_OP_TYPE Operation;
        PCU_DEVICE_POINTER Address;
        ULONG Value;
        ULONG Padding1;
        CU_STREAM_WRITE_VALUE Flags;
        ULONG Padding2;
        PCU_DEVICE_POINTER Alias;
    } Write;
    struct {
        CU_STREAM_BATCH_MEM_OP_TYPE Operation;
        ULONG Flags;
    } FlushRemoteWrites;
    ULONG64 Padding[6];
} CU_STREAM_BATCH_MEM_OP_PARAMS;
typedef CU_STREAM_BATCH_MEM_OP_PARAMS *PCU_STREAM_BATCH_MEM_OP_PARAMS;

#pragma warning(pop)

typedef enum _CU_JIT_OPTION {
    CU_JIT_MAX_REGISTERS = 0,
    CU_JIT_THREADS_PER_BLOCK,
    CU_JIT_WALL_TIME,
    CU_JIT_INFO_LOG_BUFFER,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
    CU_JIT_ERROR_LOG_BUFFER,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
    CU_JIT_OPTIMIZATION_LEVEL,
    CU_JIT_TARGET_FROM_CUCONTEXT,
    CU_JIT_TARGET,
    CU_JIT_FALLBACK_STRATEGY,
    CU_JIT_GENERATE_DEBUG_INFO,
    CU_JIT_LOG_VERBOSE,
    CU_JIT_GENERATE_LINE_INFO,
    CU_JIT_CACHE_MODE,
    CU_JIT_NEW_SM3X_OPT,
    CU_JIT_FAST_COMPILE,
    CU_JIT_GLOBAL_SYMBOL_NAMES,
    CU_JIT_GLOBAL_SYMBOL_ADDRESSES,
    CU_JIT_GLOBAL_SYMBOL_COUNT,
    CU_JIT_NUM_OPTIONS
} CU_JIT_OPTION;
typedef CU_JIT_OPTION *PCU_JIT_OPTION;
typedef CU_JIT_OPTION **PPCU_JIT_OPTION;

typedef enum _CU_DEVICE_ATTRIBUTE {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 92,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 93,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 94,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX
} CU_DEVICE_ATTRIBUTE;
typedef CU_DEVICE_ATTRIBUTE *PCU_DEVICE_ATTRIBUTE;

#include "CuDeviceAttributes.h"

#define CU_MEMHOSTALLOC_PORTABLE        0x01
#define CU_MEMHOSTALLOC_DEVICEMAP       0x02
#define CU_MEMHOSTALLOC_WRITECOMBINED   0x04
#define CU_MEMHOSTREGISTER_PORTABLE     0x01
#define CU_MEMHOSTREGISTER_DEVICEMAP    0x02
#define CU_MEMHOSTREGISTER_IOMEMORY     0x04
#define CU_MEMPEERREGISTER_DEVICEMAP    0x02

typedef union _CU_MEM_HOST_ALLOC_FLAGS {
    struct _Struct_size_bytes_(sizeof(ULONG)) {
        ULONG Portable:1;
        ULONG DeviceMap:1;
        ULONG WriteCombined:1;
    };
    LONG AsLong;
    ULONG AsULong;
} CU_MEM_HOST_ALLOC_FLAGS;
C_ASSERT(sizeof(CU_MEM_HOST_ALLOC_FLAGS) == sizeof(ULONG));

typedef union _CU_MEM_HOST_REGISTER_FLAGS {
    struct _Struct_size_bytes_(sizeof(ULONG)) {
        ULONG Portable:1;
        ULONG DeviceMap:1;
        ULONG IoMemory:1;
    };
    LONG AsLong;
    ULONG AsULong;
} CU_MEM_HOST_REGISTER_FLAGS;
C_ASSERT(sizeof(CU_MEM_HOST_REGISTER_FLAGS) == sizeof(ULONG));

//
// Initialization.
//

typedef
_Must_inspect_result_
CU_RESULT
(CU_INIT)(
    _In_ ULONG Flags
    );
typedef CU_INIT *PCU_INIT;

//
// Errors.
//

typedef
_Must_inspect_result_
CU_RESULT
(CU_GET_ERROR_NAME)(
    _In_ CU_RESULT Error,
    _Outptr_result_z_ PCSZ *ErrorString
    );
typedef CU_GET_ERROR_NAME *PCU_GET_ERROR_NAME;

typedef
_Must_inspect_result_
CU_RESULT
(CU_GET_ERROR_STRING)(
    _In_ CU_RESULT Error,
    _Outptr_result_z_ PCSZ *ErrorString
    );
typedef CU_GET_ERROR_STRING *PCU_GET_ERROR_STRING;

//
// Driver Version.
//

typedef
_Must_inspect_result_
CU_RESULT
(CU_DRIVER_GET_VERSION)(
    _Out_ PLONG DriverVersion
    );
typedef CU_DRIVER_GET_VERSION *PCU_DRIVER_GET_VERSION;

//
// Device Management.
//

typedef
_Must_inspect_result_
CU_RESULT
(CU_DEVICE_GET)(
    _Out_ PCU_DEVICE Device,
    _In_ LONG Ordinal
    );
typedef CU_DEVICE_GET *PCU_DEVICE_GET;

typedef
_Must_inspect_result_
CU_RESULT
(CU_DEVICE_GET_COUNT)(
    _Out_ PLONG Count
    );
typedef CU_DEVICE_GET_COUNT *PCU_DEVICE_GET_COUNT;

typedef
_Must_inspect_result_
CU_RESULT
(CU_DEVICE_GET_NAME)(
    _Out_writes_z_(SizeOfNameBufferInBytes) PCHAR NameBuffer,
    _In_ LONG SizeOfNameBufferInBytes,
    _In_ CU_DEVICE Device
    );
typedef CU_DEVICE_GET_NAME *PCU_DEVICE_GET_NAME;

typedef
_Must_inspect_result_
CU_RESULT
(CU_DEVICE_TOTAL_MEMORY)(
    _Out_ PSIZE_T TotalMemoryInBytes,
    _In_ CU_DEVICE Device
    );
typedef CU_DEVICE_TOTAL_MEMORY *PCU_DEVICE_TOTAL_MEMORY;

typedef
_Must_inspect_result_
CU_RESULT
(CU_DEVICE_COMPUTE_CAPABILITY)(
    _Out_ PLONG Major,
    _Out_ PLONG Minor,
    _In_ CU_DEVICE Device
    );
typedef CU_DEVICE_COMPUTE_CAPABILITY *PCU_DEVICE_COMPUTE_CAPABILITY;

typedef
_Must_inspect_result_
CU_RESULT
(CU_DEVICE_GET_ATTRIBUTE)(
    _Out_ PLONG AttributeValuePointer,
    _In_ CU_DEVICE_ATTRIBUTE Attribute,
    _In_ CU_DEVICE Device
    );
typedef CU_DEVICE_GET_ATTRIBUTE *PCU_DEVICE_GET_ATTRIBUTE;

//
// Context Management.
//

typedef
_Must_inspect_result_
CU_RESULT
(CU_CTX_CREATE)(
    _Out_ PCU_CONTEXT *ContextPointer,
    _In_ CU_CTX_CREATE_FLAGS Flags,
    _In_ CU_DEVICE Device
    );
typedef CU_CTX_CREATE *PCU_CTX_CREATE;

typedef
_Must_inspect_result_
CU_RESULT
(CU_CTX_DESTROY)(
    _In_ _Post_invalid_ PCU_CONTEXT Context
    );
typedef CU_CTX_DESTROY *PCU_CTX_DESTROY;

typedef
_Must_inspect_result_
CU_RESULT
(CU_CTX_PUSH_CURRENT)(
    _In_ PCU_CONTEXT Context
    );
typedef CU_CTX_PUSH_CURRENT *PCU_CTX_PUSH_CURRENT;

typedef
_Must_inspect_result_
CU_RESULT
(CU_CTX_POP_CURRENT)(
    _Outptr_opt_result_maybenull_ PPCU_CONTEXT ContextPointer
    );
typedef CU_CTX_POP_CURRENT *PCU_CTX_POP_CURRENT;

typedef
_Must_inspect_result_
CU_RESULT
(CU_CTX_SET_CURRENT)(
    _In_ PCU_CONTEXT Context
    );
typedef CU_CTX_SET_CURRENT *PCU_CTX_SET_CURRENT;

typedef
_Must_inspect_result_
CU_RESULT
(CU_CTX_GET_CURRENT)(
    _Out_ PCU_CONTEXT *Context
    );
typedef CU_CTX_GET_CURRENT *PCU_CTX_GET_CURRENT;

typedef
_Must_inspect_result_
CU_RESULT
(CU_CTX_GET_DEVICE)(
    _Out_ PCU_DEVICE Device
    );
typedef CU_CTX_GET_DEVICE *PCU_CTX_GET_DEVICE;

typedef
_Must_inspect_result_
CU_RESULT
(CU_CTX_SYNCHRONIZE)(
    VOID
    );
typedef CU_CTX_SYNCHRONIZE *PCU_CTX_SYNCHRONIZE;

typedef
_Must_inspect_result_
CU_RESULT
(CU_CTX_GET_STREAM_PRIORITY_RANGE)(
    _Out_ PULONG LeastPriority,
    _Out_ PULONG GreatestPriority
    );
typedef CU_CTX_GET_STREAM_PRIORITY_RANGE *PCU_CTX_GET_STREAM_PRIORITY_RANGE;

//
// Module Management.
//

typedef
_Must_inspect_result_
CU_RESULT
(CU_MODULE_LOAD)(
    _Outptr_result_maybenull_ PPCU_MODULE ModulePointer,
    _In_ PCSZ Path
    );
typedef CU_MODULE_LOAD *PCU_MODULE_LOAD;

typedef
_Must_inspect_result_
CU_RESULT
(CU_MODULE_UNLOAD)(
    _In_ _Post_invalid_ PCU_MODULE Module
    );
typedef CU_MODULE_UNLOAD *PCU_MODULE_UNLOAD;

typedef
_Must_inspect_result_
CU_RESULT
(CU_MODULE_LOAD_DATA_EX)(
    _Out_ PPCU_MODULE ModulePointer,
    _In_z_ PCHAR Image,
    _In_ LONG NumberOfOptions,
    _In_reads_(NumberOfOptions) PCU_JIT_OPTION Options,
    _Out_writes_(NumberOfOptions) PVOID *OptionValuesPointer
    );
typedef CU_MODULE_LOAD_DATA_EX *PCU_MODULE_LOAD_DATA_EX;

typedef
_Must_inspect_result_
CU_RESULT
(CU_MODULE_GET_FUNCTION)(
    _Out_ PPCU_FUNCTION FunctionPointer,
    _In_ PCU_MODULE Module,
    _In_ PCSZ FunctioName
    );
typedef CU_MODULE_GET_FUNCTION *PCU_MODULE_GET_FUNCTION;

typedef
_Must_inspect_result_
CU_RESULT
(CU_MODULE_GET_GLOBAL)(
    _Out_ PPCU_DEVICE_POINTER DevicePtrPointer,
    _Out_ PSIZE_T SizeInBytes,
    _In_ PCU_MODULE Module,
    _In_ PCSZ Name
    );
typedef CU_MODULE_GET_GLOBAL *PCU_MODULE_GET_GLOBAL;

//
// Stream Management.
//

typedef
_Must_inspect_result_
CU_RESULT
(CU_STREAM_CREATE)(
    _Out_ PCU_STREAM Stream,
    _In_ CU_STREAM_FLAGS Flags
    );
typedef CU_STREAM_CREATE *PCU_STREAM_CREATE;

typedef
_Must_inspect_result_
CU_RESULT
(CU_STREAM_CREATE_WITH_PRIORITY)(
    _Out_ PCU_STREAM Stream,
    _In_ CU_STREAM_FLAGS Flags,
    _In_ ULONG Priority
    );
typedef CU_STREAM_CREATE_WITH_PRIORITY *PCU_STREAM_CREATE_WITH_PRIORITY;

typedef
_Must_inspect_result_
CU_RESULT
(CU_STREAM_DESTROY)(
    _In_ _Post_invalid_ CU_STREAM Stream
    );
typedef CU_STREAM_DESTROY *PCU_STREAM_DESTROY;

typedef
_Must_inspect_result_
CU_RESULT
(CU_STREAM_QUERY)(
    _In_ CU_STREAM Stream
    );
typedef CU_STREAM_QUERY *PCU_STREAM_QUERY;

typedef
_Must_inspect_result_
CU_RESULT
(CU_STREAM_SYNCHRONIZE)(
    _In_ CU_STREAM Stream
    );
typedef CU_STREAM_SYNCHRONIZE *PCU_STREAM_SYNCHRONIZE;

typedef
_Must_inspect_result_
CU_RESULT
(CALLBACK CU_STREAM_CALLBACK)(
    _In_opt_ CU_STREAM Stream,
    _In_ CU_RESULT Status,
    _In_opt_ PVOID UserData
    );
typedef CU_STREAM_CALLBACK *PCU_STREAM_CALLBACK;

typedef
_Must_inspect_result_
CU_RESULT
(CU_STREAM_ADD_CALLBACK)(
    _In_ CU_STREAM Stream,
    _In_ PCU_STREAM_CALLBACK Callback,
    _In_opt_ PVOID UserData,
    _In_ ULONG Flags
    );
typedef CU_STREAM_ADD_CALLBACK *PCU_STREAM_ADD_CALLBACK;

typedef
_Must_inspect_result_
CU_RESULT
(CU_STREAM_ATTACH_MEM_ASYNC)(
    _In_ CU_STREAM Stream,
    _In_ PCU_DEVICE_POINTER Address,
    _In_ _Pre_ _Field_range_(==, 0) SIZE_T Length,
    _In_ CU_MEM_ATTACH_FLAGS Flags
    );
typedef CU_STREAM_ATTACH_MEM_ASYNC *PCU_STREAM_ATTACH_MEM_ASYNC;

typedef
_Must_inspect_result_
CU_RESULT
(CU_STREAM_WAIT_EVENT)(
    _In_ PCU_STREAM Stream,
    _In_ PCU_EVENT Event,
    _In_ ULONG Unused
    );
typedef CU_STREAM_WAIT_EVENT *PCU_STREAM_WAIT_EVENT;

typedef
_Must_inspect_result_
CU_RESULT
(CU_STREAM_WAIT_VALUE_32)(
    _In_ CU_STREAM Stream,
    _In_ PCU_DEVICE_POINTER Address,
    _In_ ULONG Value,
    _In_ CU_STREAM_WAIT_VALUE Flags
    );
typedef CU_STREAM_WAIT_VALUE_32 *PCU_STREAM_WAIT_VALUE_32;

typedef
_Must_inspect_result_
CU_RESULT
(CU_STREAM_WRITE_VALUE_32)(
    _In_ CU_STREAM Stream,
    _In_ PCU_DEVICE_POINTER Address,
    _In_ ULONG Value,
    _In_ CU_STREAM_WRITE_VALUE Flags
    );
typedef CU_STREAM_WRITE_VALUE_32 *PCU_STREAM_WRITE_VALUE_32;

typedef
_Must_inspect_result_
CU_RESULT
(CU_STREAM_BATCH_MEM_OP)(
    _In_ CU_STREAM Stream,
    _In_ ULONG Count,
    _In_reads_(Count) PCU_STREAM_BATCH_MEM_OP_PARAMS Params,
    _In_ ULONG Flags
    );
typedef CU_STREAM_BATCH_MEM_OP *PCU_STREAM_BATCH_MEM_OP;

//
// Event Management.
//

typedef
_Must_inspect_result_
CU_RESULT
(CU_EVENT_CREATE)(
    _Outptr_result_maybenull_ PPCU_EVENT EventPointer,
    _In_ CU_EVENT_FLAGS Flags
    );
typedef CU_EVENT_CREATE *PCU_EVENT_CREATE;

typedef
_Must_inspect_result_
CU_RESULT
(CU_EVENT_DESTROY)(
    _In_ _Post_invalid_ PCU_EVENT Event
    );
typedef CU_EVENT_DESTROY *PCU_EVENT_DESTROY;

typedef
_Must_inspect_result_
CU_RESULT
(CU_EVENT_ELAPSED_TIME)(
    _Out_ PFLOAT Milliseconds,
    _In_ PCU_EVENT StartEvent,
    _In_ PCU_EVENT EndEvent
    );
typedef CU_EVENT_ELAPSED_TIME *PCU_EVENT_ELAPSED_TIME;

typedef
_Must_inspect_result_
CU_RESULT
(CU_EVENT_QUERY)(
    _In_ PCU_EVENT Event
    );
typedef CU_EVENT_QUERY *PCU_EVENT_QUERY;

typedef
_Must_inspect_result_
CU_RESULT
(CU_EVENT_RECORD)(
    _In_ PCU_EVENT Event,
    _In_ PCU_STREAM Stream
    );
typedef CU_EVENT_RECORD *PCU_EVENT_RECORD;

typedef
_Must_inspect_result_
CU_RESULT
(CU_EVENT_SYNCHRONIZE)(
    _In_ PCU_EVENT Event
    );
typedef CU_EVENT_SYNCHRONIZE *PCU_EVENT_SYNCHRONIZE;

//
// Memory Management.
//

typedef
_Must_inspect_result_
CU_RESULT
(CU_MEM_ALLOC)(
    _Out_ PCU_DEVICE_POINTER DevicePointer,
    _In_ SIZE_T SizeInBytes
    );
typedef CU_MEM_ALLOC *PCU_MEM_ALLOC;

typedef
_Must_inspect_result_
CU_RESULT
(CU_MEM_FREE)(
    _In_ CU_DEVICE_POINTER DevicePointer
    );
typedef CU_MEM_FREE *PCU_MEM_FREE;

typedef
_Must_inspect_result_
CU_RESULT
(CU_MEM_HOST_ALLOC)(
    _Outptr_result_nullonfailure_ PVOID *pHostPointer,
    _In_ SIZE_T SizeInBytes,
    _In_ CU_MEM_HOST_ALLOC_FLAGS Flags
    );
typedef CU_MEM_HOST_ALLOC *PCU_MEM_HOST_ALLOC;

typedef
_Must_inspect_result_
CU_RESULT
(CU_MEM_PREFETCH_ASYNC)(
    _In_ CU_DEVICE_POINTER DevicePointer,
    _In_ SIZE_T Count,
    _In_ CU_DEVICE DestDevice,
    _In_ PCU_STREAM Stream
    );
typedef CU_MEM_PREFETCH_ASYNC *PCU_MEM_PREFETCH_ASYNC;

typedef
_Must_inspect_result_
CU_RESULT
(CU_MEM_HOST_GET_DEVICE_POINTER)(
    _Out_ PCU_DEVICE_POINTER DevicePointer,
    _In_ PVOID HostPointer,
    _In_ LONG Unused
    );
typedef CU_MEM_HOST_GET_DEVICE_POINTER *PCU_MEM_HOST_GET_DEVICE_POINTER;

typedef
_Must_inspect_result_
CU_RESULT
(CU_MEM_HOST_REGISTER)(
    _In_ PVOID HostPointer,
    _In_ SIZE_T SizeInBytes,
    _In_ CU_MEM_HOST_REGISTER_FLAGS Flags
    );
typedef CU_MEM_HOST_REGISTER *PCU_MEM_HOST_REGISTER;

typedef
_Must_inspect_result_
CU_RESULT
(CU_MEM_HOST_UNREGISTER)(
    _In_ PVOID HostPointer
    );
typedef CU_MEM_HOST_UNREGISTER *PCU_MEM_HOST_UNREGISTER;

typedef
_Must_inspect_result_
CU_RESULT
(CU_MEM_FREE_HOST)(
    _In_ _Post_invalid_ PVOID HostPointer
    );
typedef CU_MEM_FREE_HOST *PCU_MEM_FREE_HOST;

//
// Memcpy Functions.
//

typedef
_Must_inspect_result_
CU_RESULT
(CU_MEMCPY_HOST_TO_DEVICE)(
    _In_ CU_DEVICE_POINTER DestDevicePointer,
    _In_reads_bytes_(ByteCount) PVOID SourceHostPointer,
    _In_ SIZE_T ByteCount
    );
typedef CU_MEMCPY_HOST_TO_DEVICE *PCU_MEMCPY_HOST_TO_DEVICE;

typedef
_Must_inspect_result_
CU_RESULT
(CU_MEMCPY_DEVICE_TO_HOST)(
    _Out_writes_(ByteCount) PVOID DestHostPointer,
    _In_ CU_DEVICE_POINTER SourceDevicePointer,
    _In_ SIZE_T ByteCount
    );
typedef CU_MEMCPY_DEVICE_TO_HOST *PCU_MEMCPY_DEVICE_TO_HOST;

typedef
_Must_inspect_result_
CU_RESULT
(CU_MEMCPY_HOST_TO_DEVICE_ASYNC)(
    _In_ CU_DEVICE_POINTER DestDevicePointer,
    _In_reads_bytes_(ByteCount) PVOID SourceHostPointer,
    _In_ SIZE_T ByteCount,
    _In_opt_ CU_STREAM Stream
    );
typedef CU_MEMCPY_HOST_TO_DEVICE_ASYNC *PCU_MEMCPY_HOST_TO_DEVICE_ASYNC;

typedef
_Must_inspect_result_
CU_RESULT
(CU_MEMCPY_DEVICE_TO_HOST_ASYNC)(
    _Out_writes_bytes_(ByteCount) PVOID DestHostPointer,
    _In_ CU_DEVICE_POINTER SourceDevicePointer,
    _In_ SIZE_T ByteCount,
    _In_opt_ CU_STREAM Stream
    );
typedef CU_MEMCPY_DEVICE_TO_HOST_ASYNC *PCU_MEMCPY_DEVICE_TO_HOST_ASYNC;

//
// Functions.
//

typedef
_Must_inspect_result_
CU_RESULT
(CU_LAUNCH_KERNEL)(
    _In_ PCU_FUNCTION Function,
    _In_ ULONG GridDimX,
    _In_ ULONG GridDimY,
    _In_ ULONG GridDimZ,
    _In_ ULONG BlockDimX,
    _In_ ULONG BlockDimY,
    _In_ ULONG BlockDimZ,
    _In_ ULONG SharedMemoryInBytes,
    _In_ CU_STREAM Stream,
    _In_opt_ PVOID *KernelParameters,
    _In_opt_ PVOID *Extra
    );
typedef CU_LAUNCH_KERNEL *PCU_LAUNCH_KERNEL;

//
// Occupancy
//

typedef enum _Enum_is_bitflag_ _CU_OCCUPANCY_FLAGS {
    CU_OCCUPANCY_DEFAULT                  = 0x0, /**< Default behavior */
    CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE = 0x1  /**< Assume global caching is enabled and cannot be automatically turned off */
} CU_OCCUPANCY_FLAGS;

typedef
_Must_inspect_result_
CU_RESULT
(CU_OCCUPANCY_MAX_ACTIVE_BLOCKS_PER_MULTIPROCESSOR)(
    _Out_ LONG *NumBlocks,
    _In_ PCU_FUNCTION Function,
    _In_ LONG BlockSize,
    _In_ SIZE_T DynamicSharedMemorySize
    );
typedef CU_OCCUPANCY_MAX_ACTIVE_BLOCKS_PER_MULTIPROCESSOR
      *PCU_OCCUPANCY_MAX_ACTIVE_BLOCKS_PER_MULTIPROCESSOR;

typedef
_Must_inspect_result_
CU_RESULT
(CU_OCCUPANCY_MAX_ACTIVE_BLOCKS_PER_MULTIPROCESSOR_WITH_FLAGS)(
    _Out_ LONG *NumBlocks,
    _In_ PCU_FUNCTION Function,
    _In_ LONG BlockSize,
    _In_ SIZE_T DynamicSharedMemorySize,
    _In_opt_ CU_OCCUPANCY_FLAGS OccupancyFlags
    );
typedef CU_OCCUPANCY_MAX_ACTIVE_BLOCKS_PER_MULTIPROCESSOR_WITH_FLAGS
      *PCU_OCCUPANCY_MAX_ACTIVE_BLOCKS_PER_MULTIPROCESSOR_WITH_FLAGS;

typedef
SIZE_T
(CUDACBCALLTYPE CU_OCCUPANCY_B2D_SIZE)(
    _In_ LONG Blocksize
    );
typedef CU_OCCUPANCY_B2D_SIZE *PCU_OCCUPANCY_B2D_SIZE;

typedef
_Must_inspect_result_
CU_RESULT
(CU_OCCUPANCY_MAX_POTENTIAL_BLOCK_SIZE)(
    _Out_ LONG *MinimumGridSize,
    _Out_ LONG *BlockSize,
    _In_ PCU_FUNCTION Function,
    _In_opt_ PCU_OCCUPANCY_B2D_SIZE OccupancyBlockSizeToDynamicMemSize,
    _In_opt_ SIZE_T DynamicSharedMemorySize,
    _In_opt_ LONG BlockSizeLimit
    );
typedef CU_OCCUPANCY_MAX_POTENTIAL_BLOCK_SIZE
      *PCU_OCCUPANCY_MAX_POTENTIAL_BLOCK_SIZE;

typedef
_Must_inspect_result_
CU_RESULT
(CU_OCCUPANCY_MAX_POTENTIAL_BLOCK_SIZE_WITH_FLAGS)(
    _Out_ LONG *MinimumGridSize,
    _Out_ LONG *BlockSize,
    _In_ PCU_FUNCTION Function,
    _In_opt_ PCU_OCCUPANCY_B2D_SIZE OccupancyBlockSizeToDynamicMemSize,
    _In_opt_ SIZE_T DynamicSharedMemorySize,
    _In_opt_ LONG BlockSizeLimit,
    _In_opt_ CU_OCCUPANCY_FLAGS OccupancyFlags
    );
typedef CU_OCCUPANCY_MAX_POTENTIAL_BLOCK_SIZE_WITH_FLAGS
      *PCU_OCCUPANCY_MAX_POTENTIAL_BLOCK_SIZE_WITH_FLAGS;

//
// Link Management
//

typedef
_Must_inspect_result_
CU_RESULT
(CU_LINK_CREATE)(
    _In_ ULONG NumberOfOptions,
    _In_reads_(NumberOfOptions) PCU_JIT_OPTION Options,
    _In_reads_(NumberOfOptions) PVOID *OptionValues,
    _Out_ PCU_LINK_STATE *LinkState
    );
typedef CU_LINK_CREATE *PCU_LINK_CREATE;

typedef
_Must_inspect_result_
CU_RESULT
(CU_LINK_ADD_DATA)(
    _In_ PCU_LINK_STATE LinkState,
    _In_ CU_JIT_INPUT_TYPE JitInputType,
    _In_reads_(SizeInBytes) PCHAR Data,
    _In_ SIZE_T SizeInBytes,
    _In_opt_ PCHAR Name,
    _In_ ULONG NumberOfOptions,
    _In_reads_(NumberOfOptions) PCU_JIT_OPTION Options,
    _In_reads_(NumberOfOptions) PVOID *OptionValues
    );
typedef CU_LINK_ADD_DATA *PCU_LINK_ADD_DATA;

typedef
_Must_inspect_result_
CU_RESULT
(CU_LINK_ADD_FILE)(
    _In_ PCU_LINK_STATE LinkState,
    _In_ CU_JIT_INPUT_TYPE JitInputType,
    _In_ PCSZ Path,
    _In_ ULONG NumberOfOptions,
    _In_reads_(NumberOfOptions) PCU_JIT_OPTION Options,
    _In_reads_(NumberOfOptions) PVOID *OptionValues
    );
typedef CU_LINK_ADD_FILE *PCU_LINK_ADD_FILE;

typedef
_Must_inspect_result_
CU_RESULT
(CU_LINK_COMPLETE)(
    _In_ PCU_LINK_STATE LinkState,
    _Out_writes_bytes_(*SizeInBytes) PVOID *Module,
    _Out_ SIZE_T *SizeInBytes
    );
typedef CU_LINK_COMPLETE *PCU_LINK_COMPLETE;

typedef
_Must_inspect_result_
CU_RESULT
(CU_LINK_DESTROY)(
    _In_ PCU_LINK_STATE LinkState
    );
typedef CU_LINK_DESTROY *PCU_LINK_DESTROY;

//
// Define the CU_FUNCTION_TABLE X-macro.  Each macro receives (Upper, Name) as
// as its parameters, where Upper represents the pointer type name (excluding
// the leading 'PCU'), and name is the capitalized name of the function, e.g.:
//
//      (INIT, Init)
//      (GET_ERROR_NAME, GetErrorName)
//

#define CU_FUNCTION_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY)          \
                                                                   \
    FIRST_ENTRY(                                                   \
        INIT,                                                      \
        Init                                                       \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        GET_ERROR_NAME,                                            \
        GetErrorName                                               \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        GET_ERROR_STRING,                                          \
        GetErrorString                                             \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        DRIVER_GET_VERSION,                                        \
        DriverGetVersion                                           \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        DEVICE_GET,                                                \
        DeviceGet                                                  \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        DEVICE_GET_COUNT,                                          \
        DeviceGetCount                                             \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        DEVICE_GET_NAME,                                           \
        DeviceGetName                                              \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        DEVICE_TOTAL_MEMORY,                                       \
        DeviceTotalMem                                             \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        DEVICE_COMPUTE_CAPABILITY,                                 \
        DeviceComputeCapability                                    \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        DEVICE_GET_ATTRIBUTE,                                      \
        DeviceGetAttribute                                         \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        CTX_SET_CURRENT,                                           \
        CtxSetCurrent                                              \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        CTX_GET_CURRENT,                                           \
        CtxGetCurrent                                              \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        CTX_GET_DEVICE,                                            \
        CtxGetDevice                                               \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        CTX_SYNCHRONIZE,                                           \
        CtxSynchronize                                             \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        CTX_GET_STREAM_PRIORITY_RANGE,                             \
        CtxGetStreamPriorityRange                                  \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        LINK_COMPLETE,                                             \
        LinkComplete                                               \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        LINK_DESTROY,                                              \
        LinkDestroy                                                \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        MODULE_LOAD,                                               \
        ModuleLoad                                                 \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        MODULE_UNLOAD,                                             \
        ModuleUnload                                               \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        MODULE_LOAD_DATA_EX,                                       \
        ModuleLoadDataEx                                           \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        MODULE_GET_FUNCTION,                                       \
        ModuleGetFunction                                          \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        STREAM_CREATE,                                             \
        StreamCreate                                               \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        STREAM_CREATE_WITH_PRIORITY,                               \
        StreamCreateWithPriority                                   \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        STREAM_QUERY,                                              \
        StreamQuery                                                \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        STREAM_SYNCHRONIZE,                                        \
        StreamSynchronize                                          \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        STREAM_ADD_CALLBACK,                                       \
        StreamAddCallback                                          \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        STREAM_ATTACH_MEM_ASYNC,                                   \
        StreamAttachMemAsync                                       \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        STREAM_WAIT_EVENT,                                         \
        StreamWaitEvent                                            \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        STREAM_WAIT_VALUE_32,                                      \
        StreamWaitValue32                                          \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        STREAM_WRITE_VALUE_32,                                     \
        StreamWriteValue32                                         \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        STREAM_BATCH_MEM_OP,                                       \
        StreamBatchMemOp                                           \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        EVENT_CREATE,                                              \
        EventCreate                                                \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        EVENT_ELAPSED_TIME,                                        \
        EventElapsedTime                                           \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        EVENT_QUERY,                                               \
        EventQuery                                                 \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        EVENT_RECORD,                                              \
        EventRecord                                                \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        EVENT_SYNCHRONIZE,                                         \
        EventSynchronize                                           \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        MEM_PREFETCH_ASYNC,                                        \
        MemPrefetchAsync                                           \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        MEM_HOST_UNREGISTER,                                       \
        MemHostUnregister                                          \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        MEM_HOST_ALLOC,                                            \
        MemHostAlloc                                               \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        MEM_FREE_HOST,                                             \
        MemFreeHost                                                \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        OCCUPANCY_MAX_ACTIVE_BLOCKS_PER_MULTIPROCESSOR,            \
        OccupancyMaxActiveBlocksPerMultiprocessor                  \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        OCCUPANCY_MAX_ACTIVE_BLOCKS_PER_MULTIPROCESSOR_WITH_FLAGS, \
        OccupancyMaxActiveBlocksPerMultiprocessorWithFlags         \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        OCCUPANCY_MAX_POTENTIAL_BLOCK_SIZE,                        \
        OccupancyMaxPotentialBlockSize                             \
    )                                                              \
                                                                   \
    ENTRY(                                                         \
        OCCUPANCY_MAX_POTENTIAL_BLOCK_SIZE_WITH_FLAGS,             \
        OccupancyMaxPotentialBlockSizeWithFlags                    \
    )                                                              \
                                                                   \
    LAST_ENTRY(                                                    \
        LAUNCH_KERNEL,                                             \
        LaunchKernel                                               \
    )

#define CU_FUNCTION_TABLE_ENTRY(ENTRY) \
    CU_FUNCTION_TABLE(ENTRY, ENTRY, ENTRY)

#define EXPAND_AS_CU_FUNCTION_STRUCT(Upper, Name) \
    PCU_##Upper Name;


//
// Define the CU_FUNCTION_V2_TABLE X-macro.  This is identical to the table
// above, except the _v2 versions of functions will be loaded.
//

#define CU_FUNCTION_V2_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
                                                             \
    ENTRY(                                                   \
        CTX_CREATE,                                          \
        CtxCreate                                            \
    )                                                        \
                                                             \
    ENTRY(                                                   \
        CTX_DESTROY,                                         \
        CtxDestroy                                           \
    )                                                        \
                                                             \
    ENTRY(                                                   \
        CTX_POP_CURRENT,                                     \
        CtxPopCurrent                                        \
    )                                                        \
                                                             \
    ENTRY(                                                   \
        CTX_PUSH_CURRENT,                                    \
        CtxPushCurrent                                       \
    )                                                        \
                                                             \
    ENTRY(                                                   \
        EVENT_DESTROY,                                       \
        EventDestroy                                         \
    )                                                        \
                                                             \
    ENTRY(                                                   \
        LINK_CREATE,                                         \
        LinkCreate                                           \
    )                                                        \
                                                             \
    ENTRY(                                                   \
        LINK_ADD_DATA,                                       \
        LinkAddData                                          \
    )                                                        \
                                                             \
    ENTRY(                                                   \
        LINK_ADD_FILE,                                       \
        LinkAddFile                                          \
    )                                                        \
                                                             \
    ENTRY(                                                   \
        MEM_HOST_GET_DEVICE_POINTER,                         \
        MemHostGetDevicePointer                              \
    )                                                        \
                                                             \
    ENTRY(                                                   \
        MEM_HOST_REGISTER,                                   \
        MemHostRegister                                      \
    )                                                        \
                                                             \
    ENTRY(                                                   \
        MEM_ALLOC,                                           \
        MemAlloc                                             \
    )                                                        \
                                                             \
    ENTRY(                                                   \
        MEMCPY_HOST_TO_DEVICE,                               \
        MemcpyHtoD                                           \
    )                                                        \
                                                             \
    ENTRY(                                                   \
        MEMCPY_DEVICE_TO_HOST,                               \
        MemcpyDtoH                                           \
    )                                                        \
                                                             \
    ENTRY(                                                   \
        MEMCPY_HOST_TO_DEVICE_ASYNC,                         \
        MemcpyHtoDAsync                                      \
    )                                                        \
                                                             \
    ENTRY(                                                   \
        MEMCPY_DEVICE_TO_HOST_ASYNC,                         \
        MemcpyDtoHAsync                                      \
    )                                                        \
                                                             \
    ENTRY(                                                   \
        MEM_FREE,                                            \
        MemFree                                              \
    )                                                        \
                                                             \
    ENTRY(                                                   \
        MODULE_GET_GLOBAL,                                   \
        ModuleGetGlobal                                      \
    )                                                        \
                                                             \
    ENTRY(                                                   \
        STREAM_DESTROY,                                      \
        StreamDestroy                                        \
    )

#define CU_FUNCTION_V2_TABLE_ENTRY(ENTRY) \
    CU_FUNCTION_V2_TABLE(ENTRY, ENTRY, ENTRY)

#define EXPAND_AS_CU_FUNCTION_V2_STRUCT(Upper, Name) \
    PCU_##Upper Name;

typedef struct _CU_FUNCTIONS {
    CU_FUNCTION_TABLE_ENTRY(EXPAND_AS_CU_FUNCTION_STRUCT)
    CU_FUNCTION_V2_TABLE_ENTRY(EXPAND_AS_CU_FUNCTION_STRUCT)
} CU_FUNCTIONS;
typedef CU_FUNCTIONS *PCU_FUNCTIONS;

//
// CuRand
//

typedef enum _Return_type_success_(return == 0) _CURAND_RESULT {
    CURAND_STATUS_SUCCESS = 0, ///< No errors
    CURAND_STATUS_VERSION_MISMATCH = 100, ///< Header file and linked library version do not match
    CURAND_STATUS_NOT_INITIALIZED = 101, ///< Generator not initialized
    CURAND_STATUS_ALLOCATION_FAILED = 102, ///< Memory allocation failed
    CURAND_STATUS_TYPE_ERROR = 103, ///< Generator is wrong type
    CURAND_STATUS_OUT_OF_RANGE = 104, ///< Argument out of range
    CURAND_STATUS_LENGTH_NOT_MULTIPLE = 105, ///< Length requested is not a multple of dimension
    CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106, ///< GPU does not have double precision required by MRG32k3a
    CURAND_STATUS_LAUNCH_FAILURE = 201, ///< Kernel launch failure
    CURAND_STATUS_PREEXISTING_FAILURE = 202, ///< Preexisting failure on library entry
    CURAND_STATUS_INITIALIZATION_FAILED = 203, ///< Initialization of CUDA failed
    CURAND_STATUS_ARCH_MISMATCH = 204, ///< Architecture mismatch, GPU does not support requested feature
    CURAND_STATUS_INTERNAL_ERROR = 999 ///< Internal library error
} CURAND_RESULT;

typedef enum _CURAND_RNG_TYPE {
    CURAND_RNG_TEST = 0,
    CURAND_RNG_PSEUDO_DEFAULT = 100, ///< Default pseudorandom generator
    CURAND_RNG_PSEUDO_XORWOW = 101, ///< XORWOW pseudorandom generator
    CURAND_RNG_PSEUDO_MRG32K3A = 121, ///< MRG32k3a pseudorandom generator
    CURAND_RNG_PSEUDO_MTGP32 = 141, ///< Mersenne Twister MTGP32 pseudorandom generator
    CURAND_RNG_PSEUDO_MT19937 = 142, ///< Mersenne Twister MT19937 pseudorandom generator
    CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161, ///< PHILOX-4x32-10 pseudorandom generator
    CURAND_RNG_QUASI_DEFAULT = 200, ///< Default quasirandom generator
    CURAND_RNG_QUASI_SOBOL32 = 201, ///< Sobol32 quasirandom generator
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202,  ///< Scrambled Sobol32 quasirandom generator
    CURAND_RNG_QUASI_SOBOL64 = 203, ///< Sobol64 quasirandom generator
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204  ///< Scrambled Sobol64 quasirandom generator
} CURAND_RNG_TYPE;

struct CURAND_GENERATOR;
typedef struct CURAND_GENERATOR *PCURAND_GENERATOR;

typedef
_Must_inspect_result_
CURAND_RESULT
(STDAPICALLTYPE CURAND_CREATE_GENERATOR)(
    _Out_ PCURAND_GENERATOR *Generator,
    _In_ CURAND_RNG_TYPE RngType
    );
typedef CURAND_CREATE_GENERATOR *PCURAND_CREATE_GENERATOR;

typedef
_Must_inspect_result_
CURAND_RESULT
(STDAPICALLTYPE CURAND_GENERATE)(
    _In_ PCURAND_GENERATOR Generator,
    _Out_writes_(NumberOfElements) PULONG Output,
    _In_ SIZE_T NumberOfElements
    );
typedef CURAND_GENERATE *PCURAND_GENERATE;

typedef
_Must_inspect_result_
CURAND_RESULT
(STDAPICALLTYPE CURAND_DESTROY_GENERATOR)(
    _In_ _Post_ptr_invalid_ PCURAND_GENERATOR Generator
    );
typedef CURAND_DESTROY_GENERATOR *PCURAND_DESTROY_GENERATOR;

#define CURAND_FUNCTION_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
                                                              \
    FIRST_ENTRY(                                              \
        CREATE_GENERATOR,                                     \
        CreateGenerator                                       \
    )                                                         \
                                                              \
    ENTRY(                                                    \
        GENERATE,                                             \
        Generate                                              \
    )                                                         \
                                                              \
    LAST_ENTRY(                                               \
        DESTROY_GENERATOR,                                    \
        DestroyGenerator                                      \
    )

#define CURAND_FUNCTION_TABLE_ENTRY(ENTRY) \
    CURAND_FUNCTION_TABLE(ENTRY, ENTRY, ENTRY)

#define EXPAND_AS_CURAND_FUNCTION_STRUCT(Upper, Name) \
    PCURAND_##Upper Name;

typedef struct _CURAND_FUNCTIONS {
    CURAND_FUNCTION_TABLE_ENTRY(EXPAND_AS_CURAND_FUNCTION_STRUCT)
} CURAND_FUNCTIONS;
typedef CURAND_FUNCTIONS *PCURAND_FUNCTIONS;

//
// Dimensions helper.
//

typedef struct _CU_DIM3 {
    ULONG X;
    ULONG Y;
    ULONG Z;
    ULONG Padding;
} CU_DIM3;
typedef CU_DIM3 *PCU_DIM3;

//
// Declare CU component and define vtbl methods.
//

DECLARE_COMPONENT(Cu, CU);

//
// Vtbl functions.
//

typedef
_Must_inspect_result_
_Success_(return >= 0)
CU_RESULT
(STDAPICALLTYPE LOAD_CU_DEVICE_ATTRIBUTES)(
    _In_ struct _CU *Cu,
    _Inout_ PCU_DEVICE_ATTRIBUTES AttributesPointer,
    _In_ CU_DEVICE Device
    );
typedef LOAD_CU_DEVICE_ATTRIBUTES *PLOAD_CU_DEVICE_ATTRIBUTES;

//
// CU vtbl.
//

typedef struct _CU_VTBL {
    DECLARE_COMPONENT_VTBL_HEADER(CU);
    PLOAD_CU_DEVICE_ATTRIBUTES LoadCuDeviceAttributes;
} CU_VTBL;
typedef CU_VTBL *PCU_VTBL;

//
// Private non-vtbl methods.
//

typedef
HRESULT
(NTAPI CU_INITIALIZE)(
    _In_ struct _CU *Cu
    );
typedef CU_INITIALIZE *PCU_INITIALIZE;

typedef
VOID
(NTAPI CU_RUNDOWN)(
    _In_ _Post_ptr_invalid_ struct _CU *Cu
    );
typedef CU_RUNDOWN *PCU_RUNDOWN;

//
// Occupancy stats per function.
//

typedef struct _CU_OCCUPANCY {

    //
    // Obtained via Cu->OccupancyMaxActiveBlocksPerMultiprocessorWithFlags.
    //

    LONG NumBlocks;

    //
    // Obtained via Cu->OccupancyMaxPotentialBlockSizeWithFlags.
    //

    LONG MinimumGridSize;
    LONG BlockSize;

    //
    // Pad out to an 8-byte boundary.
    //

    ULONG Padding;
} CU_OCCUPANCY;
typedef CU_OCCUPANCY *PCU_OCCUPANCY;

//
// Define the CU structure that encapsulates all CUDA Driver functionality.
//

DEFINE_UNUSED_STATE(CU);
DEFINE_UNUSED_FLAGS(CU);

#define PERFECT_HASH_CU_JIT_LOG_BUFFER_SIZE_IN_BYTES 1024
#define PERFECT_HASH_CU_DEFAULT_BLOCKS_PER_GRID 16
#define PERFECT_HASH_CU_DEFAULT_THREADS_PER_BLOCK 128
#define PERFECT_HASH_CU_DEFAULT_KERNEL_RUNTIME_TARGET_IN_MILLISECONDS 1500

typedef struct _Struct_size_bytes_(SizeOfStruct) _CU {
    COMMON_COMPONENT_HEADER(CU);

    //
    // Number of nvcuda function pointers.
    //

    ULONG NumberOfCuFunctions;

    //
    // Number of curand function pointers.
    //

    ULONG NumberOfCuRandFunctions;

    //
    // Driver Version.
    //

    LONG DriverVersionRaw;
    USHORT DriverMajor;
    USHORT DriverMinor;

    //
    // Character suffix of driver version that can be used to replace <NN> in
    // library names such as `curand64_<NN>.dll`.
    //

    CHAR DriverSuffix[2];

    //
    // Pad out to a 4-byte boundary.
    //

    CHAR Padding1[2];

    //
    // Number of CUDA devices present on the system.
    //

    LONG NumberOfDevices;

    //
    // nvcuda.dll module.
    //

    HMODULE NvCudaModule;

    //
    // curand64_[nn].dll module.
    //

    HMODULE CuRandModule;

    //
    // Function pointers.
    //

    union {

        //
        // Inline the Cu functions for convenience.
        //

        struct {
            CU_FUNCTION_TABLE_ENTRY(EXPAND_AS_CU_FUNCTION_STRUCT)
            CU_FUNCTION_V2_TABLE_ENTRY(EXPAND_AS_CU_FUNCTION_STRUCT)
        };

        CU_FUNCTIONS CuFunctions;
    };

    union {

        //
        // Inline the CuRand functions for convenience.
        //

        struct {
            CURAND_FUNCTION_TABLE_ENTRY(EXPAND_AS_CURAND_FUNCTION_STRUCT)
        };

        CURAND_FUNCTIONS CuRandFunctions;
    };

    //
    // The Cu vtbl interface.
    //

    CU_VTBL Interface;

} CU;
typedef CU *PCU;

//
// Decls
//

extern CU_INITIALIZE CuInitialize;
extern CU_RUNDOWN CuRundown;
extern LOAD_CU_DEVICE_ATTRIBUTES LoadCuDeviceAttributes;

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
