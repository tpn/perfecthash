/*++

Copyright (c) 2020 Trent Nelson <trent@trent.me>

Module Name:

    CuDeviceAttributes.h

Abstract:

    This header file contains the CU_DEVICE_ATTRIBUTES struct.  It is kept
    separate from Cu.h in order to allow the nvcc compiler to also include it.

--*/

#pragma once

typedef struct _CU_DEVICE_ATTRIBUTES {
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
} CU_DEVICE_ATTRIBUTES;
typedef CU_DEVICE_ATTRIBUTES *PCU_DEVICE_ATTRIBUTES;

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
