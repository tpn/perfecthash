/*++

Copyright (c) 2020 Trent Nelson <trent@trent.me>

Module Name:

    CuDeviceAttributes.h

Abstract:

    This header file contains the CU_DEVICE_ATTRIBUTES struct.  It is kept
    separate from Cu.h in order to allow the nvcc compiler to also include it.

--*/

#pragma once

#define CU_DEVICE_ATTRIBUTE_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    FIRST_ENTRY(MaxThreadsPerBlock)                               \
    ENTRY(MaxBlockDimX)                                           \
    ENTRY(MaxBlockDimY)                                           \
    ENTRY(MaxBlockDimZ)                                           \
    ENTRY(MaxGridDimX)                                            \
    ENTRY(MaxGridDimY)                                            \
    ENTRY(MaxGridDimZ)                                            \
    ENTRY(MaxSharedMemoryPerBlock)                                \
    ENTRY(TotalConstantMemory)                                    \
    ENTRY(WarpSize)                                               \
    ENTRY(MaxPitch)                                               \
    ENTRY(MaxRegistersPerBlock)                                   \
    ENTRY(ClockRate)                                              \
    ENTRY(TextureAlignment)                                       \
    ENTRY(GpuOverlap)                                             \
    ENTRY(MultiprocessorCount)                                    \
    ENTRY(KernelExecTimeout)                                      \
    ENTRY(Integrated)                                             \
    ENTRY(CanMapHostMemory)                                       \
    ENTRY(ComputeMode)                                            \
    ENTRY(MaximumTexture1DWidth)                                  \
    ENTRY(MaximumTexture2DWidth)                                  \
    ENTRY(MaximumTexture2DHeight)                                 \
    ENTRY(MaximumTexture3DWidth)                                  \
    ENTRY(MaximumTexture3DHeight)                                 \
    ENTRY(MaximumTexture3DDepth)                                  \
    ENTRY(MaximumTexture2DArrayWidth)                             \
    ENTRY(MaximumTexture2DArrayHeight)                            \
    ENTRY(MaximumTexture2DArrayNumslices)                         \
    ENTRY(SurfaceAlignment)                                       \
    ENTRY(ConcurrentKernels)                                      \
    ENTRY(EccEnabled)                                             \
    ENTRY(PciBusId)                                               \
    ENTRY(PciDeviceId)                                            \
    ENTRY(TccDriver)                                              \
    ENTRY(MemoryClockRate)                                        \
    ENTRY(GlobalMemoryBusWidth)                                   \
    ENTRY(L2CacheSize)                                            \
    ENTRY(MaxThreadsPerMultiprocessor)                            \
    ENTRY(AsyncEngineCount)                                       \
    ENTRY(UnifiedAddressing)                                      \
    ENTRY(MaximumTexture1DLayeredWidth)                           \
    ENTRY(MaximumTexture1DLayeredLayers)                          \
    ENTRY(CanTex2DGather)                                         \
    ENTRY(MaximumTexture2DGatherWidth)                            \
    ENTRY(MaximumTexture2DGatherHeight)                           \
    ENTRY(MaximumTexture3DWidthAlternate)                         \
    ENTRY(MaximumTexture3DHeightAlternate)                        \
    ENTRY(MaximumTexture3DDepthAlternate)                         \
    ENTRY(PciDomainId)                                            \
    ENTRY(TexturePitchAlignment)                                  \
    ENTRY(MaximumTexturecubemapWidth)                             \
    ENTRY(MaximumTexturecubemapLayeredWidth)                      \
    ENTRY(MaximumTexturecubemapLayeredLayers)                     \
    ENTRY(MaximumSurface1DWidth)                                  \
    ENTRY(MaximumSurface2DWidth)                                  \
    ENTRY(MaximumSurface2DHeight)                                 \
    ENTRY(MaximumSurface3DWidth)                                  \
    ENTRY(MaximumSurface3DHeight)                                 \
    ENTRY(MaximumSurface3DDepth)                                  \
    ENTRY(MaximumSurface1DLayeredWidth)                           \
    ENTRY(MaximumSurface1DLayeredLayers)                          \
    ENTRY(MaximumSurface2DLayeredWidth)                           \
    ENTRY(MaximumSurface2DLayeredHeight)                          \
    ENTRY(MaximumSurface2DLayeredLayers)                          \
    ENTRY(MaximumSurfacecubemapWidth)                             \
    ENTRY(MaximumSurfacecubemapLayeredWidth)                      \
    ENTRY(MaximumSurfacecubemapLayeredLayers)                     \
    ENTRY(MaximumTexture1DLinearWidth)                            \
    ENTRY(MaximumTexture2DLinearWidth)                            \
    ENTRY(MaximumTexture2DLinearHeight)                           \
    ENTRY(MaximumTexture2DLinearPitch)                            \
    ENTRY(MaximumTexture2DMipmappedWidth)                         \
    ENTRY(MaximumTexture2DMipmappedHeight)                        \
    ENTRY(ComputeCapabilityMajor)                                 \
    ENTRY(ComputeCapabilityMinor)                                 \
    ENTRY(MaximumTexture1DMipmappedWidth)                         \
    ENTRY(StreamPrioritiesSupported)                              \
    ENTRY(GlobalL1CacheSupported)                                 \
    ENTRY(LocalL1CacheSupported)                                  \
    ENTRY(MaxSharedMemoryPerMultiprocessor)                       \
    ENTRY(MaxRegistersPerMultiprocessor)                          \
    ENTRY(ManagedMemory)                                          \
    ENTRY(MultiGpuBoard)                                          \
    ENTRY(MultiGpuBoardGroupId)                                   \
    ENTRY(HostNativeAtomicSupported)                              \
    ENTRY(SingleToDoublePrecisionPerfRatio)                       \
    ENTRY(PageableMemoryAccess)                                   \
    ENTRY(ConcurrentManagedAccess)                                \
    ENTRY(ComputePreemptionSupported)                             \
    ENTRY(CanUseHostPointerForRegisteredMem)                      \
    ENTRY(CanUseStreamMemOps)                                     \
    ENTRY(CanUse64BitStreamMemOps)                                \
    ENTRY(CanUseStreamWaitValueNor)                               \
    ENTRY(CooperativeLaunch)                                      \
    ENTRY(CooperativeMultiDeviceLaunch)                           \
    ENTRY(MaxSharedMemoryPerBlockOptin)                           \
    ENTRY(CanFlushRemoteWrites)                                   \
    ENTRY(HostRegisterSupported)                                  \
    ENTRY(PageableMemoryAccessUsesHostPageTables)                 \
    ENTRY(DirectManagedMemAccessFromHost)                         \
    ENTRY(VirtualAddressManagementSupported)                      \
    ENTRY(HandleTypePosixFileDescriptorSupported)                 \
    ENTRY(HandleTypeWin32HandleSupported)                         \
    ENTRY(HandleTypeWin32KmtHandleSupported)                      \
    ENTRY(MaxBlocksPerMultiprocessor)                             \
    ENTRY(GenericCompressionSupported)                            \
    ENTRY(MaxPersistingL2CacheSize)                               \
    ENTRY(MaxAccessPolicyWindowSize)                              \
    ENTRY(GPUDirectRDMASupportedWithCUDAVMM)                      \
    ENTRY(ReservedSharedMemoryPerBlock)                           \
    ENTRY(SparseCUDAArraysSupported)                              \
    ENTRY(ReadOnlyHostRegisterSupported)                          \
    ENTRY(TimelineSemaphoreInteropSupported)                      \
    ENTRY(MemoryPoolsSupported)                                   \
    ENTRY(GPUDirectRDMASupported)                                 \
    ENTRY(GPUDirectRDMAFlushWritesOptions)                        \
    ENTRY(GPUDirectRDMANoWritesOrdering)                          \
    ENTRY(MemPoolSupportedHandleTypes)                            \
    ENTRY(ClusterLaunchSupported)                                 \
    ENTRY(DeferredMappingCUDAArraysSupported)                     \
    ENTRY(CanUse64BitStreamMemOpsV2)                              \
    ENTRY(CanUseStreamWaitValueNorV2)                             \
    LAST_ENTRY(DMABufSupported)

#define EXPAND_AS_CU_DEVICE_ATTRIBUTE_STRUCT(Name) ULONG Name;

#define CU_DEVICE_ATTRIBUTE_TABLE_ENTRY(ENTRY) \
    CU_DEVICE_ATTRIBUTE_TABLE(ENTRY, ENTRY, ENTRY)


typedef struct _CU_DEVICE_ATTRIBUTES {
    CU_DEVICE_ATTRIBUTE_TABLE_ENTRY(EXPAND_AS_CU_DEVICE_ATTRIBUTE_STRUCT)
} CU_DEVICE_ATTRIBUTES, *PCU_DEVICE_ATTRIBUTES;

typedef struct _CU_DEVICE_ATTRIBUTES_OLD {
    ULONG MaxThreadsPerBlock;
    ULONG MaxBlockDimX;
    ULONG MaxBlockDimY;
    ULONG MaxBlockDimZ;
    ULONG MaxGridDimX;
    ULONG MaxGridDimY;
    ULONG MaxGridDimZ;
    ULONG MaxSharedMemoryPerBlock;
    ULONG TotalConstantMemory;
    ULONG WarpSize;
    ULONG MaxPitch;
    ULONG MaxRegistersPerBlock;
    ULONG ClockRate;
    ULONG TextureAlignment;
    ULONG GpuOverlap;
    ULONG MultiprocessorCount;
    ULONG KernelExecTimeout;
    ULONG Integrated;
    ULONG CanMapHostMemory;
    ULONG ComputeMode;
    ULONG MaximumTexture1DWidth;
    ULONG MaximumTexture2DWidth;
    ULONG MaximumTexture2DHeight;
    ULONG MaximumTexture3DWidth;
    ULONG MaximumTexture3DHeight;
    ULONG MaximumTexture3DDepth;
    ULONG MaximumTexture2DArrayWidth;
    ULONG MaximumTexture2DArrayHeight;
    ULONG MaximumTexture2DArrayNumslices;
    ULONG SurfaceAlignment;
    ULONG ConcurrentKernels;
    ULONG EccEnabled;
    ULONG PciBusId;
    ULONG PciDeviceId;
    ULONG TccDriver;
    ULONG MemoryClockRate;
    ULONG GlobalMemoryBusWidth;
    ULONG L2CacheSize;
    ULONG MaxThreadsPerMultiprocessor;
    ULONG AsyncEngineCount;
    ULONG UnifiedAddressing;
    ULONG MaximumTexture1DLayeredWidth;
    ULONG MaximumTexture1DLayeredLayers;
    ULONG CanTex2DGather;
    ULONG MaximumTexture2DGatherWidth;
    ULONG MaximumTexture2DGatherHeight;
    ULONG MaximumTexture3DWidthAlternate;
    ULONG MaximumTexture3DHeightAlternate;
    ULONG MaximumTexture3DDepthAlternate;
    ULONG PciDomainId;
    ULONG TexturePitchAlignment;
    ULONG MaximumTexturecubemapWidth;
    ULONG MaximumTexturecubemapLayeredWidth;
    ULONG MaximumTexturecubemapLayeredLayers;
    ULONG MaximumSurface1DWidth;
    ULONG MaximumSurface2DWidth;
    ULONG MaximumSurface2DHeight;
    ULONG MaximumSurface3DWidth;
    ULONG MaximumSurface3DHeight;
    ULONG MaximumSurface3DDepth;
    ULONG MaximumSurface1DLayeredWidth;
    ULONG MaximumSurface1DLayeredLayers;
    ULONG MaximumSurface2DLayeredWidth;
    ULONG MaximumSurface2DLayeredHeight;
    ULONG MaximumSurface2DLayeredLayers;
    ULONG MaximumSurfacecubemapWidth;
    ULONG MaximumSurfacecubemapLayeredWidth;
    ULONG MaximumSurfacecubemapLayeredLayers;
    ULONG MaximumTexture1DLinearWidth;
    ULONG MaximumTexture2DLinearWidth;
    ULONG MaximumTexture2DLinearHeight;
    ULONG MaximumTexture2DLinearPitch;
    ULONG MaximumTexture2DMipmappedWidth;
    ULONG MaximumTexture2DMipmappedHeight;
    ULONG ComputeCapabilityMajor;
    ULONG ComputeCapabilityMinor;
    ULONG MaximumTexture1DMipmappedWidth;
    ULONG StreamPrioritiesSupported;
    ULONG GlobalL1CacheSupported;
    ULONG LocalL1CacheSupported;
    ULONG MaxSharedMemoryPerMultiprocessor;
    ULONG MaxRegistersPerMultiprocessor;
    ULONG ManagedMemory;
    ULONG MultiGpuBoard;
    ULONG MultiGpuBoardGroupId;
    ULONG HostNativeAtomicSupported;
    ULONG SingleToDoublePrecisionPerfRatio;
    ULONG PageableMemoryAccess;
    ULONG ConcurrentManagedAccess;
    ULONG ComputePreemptionSupported;
    ULONG CanUseHostPointerForRegisteredMem;
    ULONG CanUseStreamMemOps;
    ULONG CanUse64BitStreamMemOps;
    ULONG CanUseStreamWaitValueNor;
    ULONG CooperativeLaunch;
    ULONG CooperativeMultiDeviceLaunch;
    ULONG MaxSharedMemoryPerBlockOptin;
    ULONG CanFlushRemoteWrites;
    ULONG HostRegisterSupported;
    ULONG PageableMemoryAccessUsesHostPageTables;
    ULONG DirectManagedMemAccessFromHost;
    ULONG VirtualAddressManagementSupported;
    ULONG HandleTypePosixFileDescriptorSupported;
    ULONG HandleTypeWin32HandleSupported;
    ULONG HandleTypeWin32KmtHandleSupported;
    ULONG MaxBlocksPerMultiprocessor;
    ULONG GenericCompressionSupported;
    ULONG MaxPersistingL2CacheSize;
    ULONG MaxAccessPolicyWindowSize;
    ULONG GPUDirectRDMASupportedWithCUDAVMM;
    ULONG ReservedSharedMemoryPerBlock;
    ULONG SparseCUDAArraysSupported;
    ULONG ReadOnlyHostRegisterSupported;
    ULONG TimelineSemaphoreInteropSupported;
    ULONG MemoryPoolsSupported;
    ULONG GPUDirectRDMASupported;
    ULONG GPUDirectRDMAFlushWritesOptions;
    ULONG GPUDirectRDMANoWritesOrdering;
    ULONG MemPoolSupportedHandleTypes;
    ULONG ClusterLaunchSupported;
    ULONG DeferredMappingCUDAArraysSupported;
    ULONG CanUse64BitStreamMemOpsV2;
    ULONG CanUseStreamWaitValueNorV2;
    ULONG DMABufSupported;
} CU_DEVICE_ATTRIBUTES_OLD;
typedef CU_DEVICE_ATTRIBUTES *PCU_DEVICE_ATTRIBUTES;

C_ASSERT(sizeof(CU_DEVICE_ATTRIBUTES) == sizeof(CU_DEVICE_ATTRIBUTES_OLD));


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
