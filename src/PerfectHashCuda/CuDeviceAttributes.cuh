
GLOBAL
VOID
PrintCuDeviceAttributes(
    _In_ PCU_DEVICE_ATTRIBUTES Attributes
    )
{
    //
    // Print the entirety of the CU_DEVICE_ATTRIBUTES struct in the format
    // <name>:<value> for each field.
    //

#define EXPAND_AS_CU_DEVICE_ATTRIBUTE_PRINT(Name) \
    printf(#Name ": %u\n", Attributes->Name);

    CU_DEVICE_ATTRIBUTE_TABLE_ENTRY(EXPAND_AS_CU_DEVICE_ATTRIBUTE_PRINT)

#if 0
    printf("MaxBlockDimX: %lu\n", Attributes->MaxBlockDimX);
    printf("MaxBlockDimY: %lu\n", Attributes->MaxBlockDimY);
    printf("MaxBlockDimZ: %lu\n", Attributes->MaxBlockDimZ);
    printf("MaxGridDimX: %lu\n", Attributes->MaxGridDimX);
    printf("MaxGridDimY: %lu\n", Attributes->MaxGridDimY);
    printf("MaxGridDimZ: %lu\n", Attributes->MaxGridDimZ);
    printf("MaxSharedMemoryPerBlock: %lu\n", Attributes->MaxSharedMemoryPerBlock);
    printf("TotalConstantMemory: %lu\n", Attributes->TotalConstantMemory);
    printf("WarpSize: %lu\n", Attributes->WarpSize);
    printf("MaxPitch: %lu\n", Attributes->MaxPitch);
    printf("MaxRegistersPerBlock: %lu\n", Attributes->MaxRegistersPerBlock);
    printf("ClockRate: %lu\n", Attributes->ClockRate);
    printf("TextureAlignment: %lu\n", Attributes->TextureAlignment);
    printf("GpuOverlap: %lu\n", Attributes->GpuOverlap);
    printf("MultiprocessorCount: %lu\n", Attributes->MultiprocessorCount);
    printf("KernelExecTimeout: %lu\n", Attributes->KernelExecTimeout);
    printf("Integrated: %lu\n", Attributes->Integrated);
    printf("CanMapHostMemory: %lu\n", Attributes->CanMapHostMemory);
    printf("ComputeMode: %lu\n", Attributes->ComputeMode);
    printf("MaximumTexture1DWidth: %lu\n", Attributes->MaximumTexture1DWidth);
    printf("MaximumTexture2DWidth: %lu\n", Attributes->MaximumTexture2DWidth);
    printf("MaximumTexture2DHeight: %lu\n", Attributes->MaximumTexture2DHeight);
    printf("MaximumTexture3DWidth: %lu\n", Attributes->MaximumTexture3DWidth);
    printf("MaximumTexture3DHeight: %lu\n", Attributes->MaximumTexture3DHeight);
    printf("MaximumTexture3DDepth: %lu\n", Attributes->MaximumTexture3DDepth);
    printf("MaximumTexture2DArrayWidth: %lu\n", Attributes->MaximumTexture2DArrayWidth);
    printf("MaximumTexture2DArrayHeight: %lu\n", Attributes->MaximumTexture2DArrayHeight);
    printf("MaximumTexture2DArrayNumslices: %lu\n", Attributes->MaximumTexture2DArrayNumslices);
    printf("SurfaceAlignment: %lu\n", Attributes->SurfaceAlignment);
    printf("ConcurrentKernels: %lu\n", Attributes->ConcurrentKernels);
    printf("EccEnabled: %lu\n", Attributes->EccEnabled);
    printf("PciBusId: %lu\n", Attributes->PciBusId);
    printf("PciDeviceId: %lu\n", Attributes->PciDeviceId);
    printf("TccDriver: %lu\n", Attributes->TccDriver);
    printf("MemoryClockRate: %lu\n", Attributes->MemoryClockRate);
    printf("GlobalMemoryBusWidth: %lu\n", Attributes->GlobalMemoryBusWidth);
    printf("L2CacheSize: %lu\n", Attributes->L2CacheSize);
    printf("MaxThreadsPerMultiprocessor: %lu\n", Attributes->MaxThreadsPerMultiprocessor);
    printf("AsyncEngineCount: %lu\n", Attributes->AsyncEngineCount);
    printf("UnifiedAddressing: %lu\n", Attributes->UnifiedAddressing);
    printf("MaximumTexture1DLayeredWidth: %lu\n", Attributes->MaximumTexture1DLayeredWidth);
    printf("MaximumTexture1DLayeredLayers: %lu\n", Attributes->MaximumTexture1DLayeredLayers);
    printf("CanTex2DGather: %lu\n", Attributes->CanTex2DGather);
    printf("MaximumTexture2DGatherWidth: %lu\n", Attributes->MaximumTexture2DGatherWidth);
    printf("MaximumTexture2DGatherHeight: %lu\n", Attributes->MaximumTexture2DGatherHeight);
    printf("MaximumTexture3DWidthAlternate: %lu\n", Attributes->MaximumTexture3DWidthAlternate);
    printf("MaximumTexture3DHeightAlternate: %lu\n", Attributes->MaximumTexture3DHeightAlternate);
    printf("MaximumTexture3DDepthAlternate: %lu\n", Attributes->MaximumTexture3DDepthAlternate);
    printf("PciDomainId: %lu\n", Attributes->PciDomainId);
    printf("TexturePitchAlignment: %lu\n", Attributes->TexturePitchAlignment);
    printf("MaximumTexturecubemapWidth: %lu\n", Attributes->MaximumTexturecubemapWidth);
    printf("MaximumTexturecubemapLayeredWidth: %lu\n", Attributes->MaximumTexturecubemapLayeredWidth);
    printf("MaximumTexturecubemapLayeredLayers: %lu\n", Attributes->MaximumTexturecubemapLayeredLayers);
    printf("MaximumSurface1DWidth: %lu\n", Attributes->MaximumSurface1DWidth);
    printf("MaximumSurface2DWidth: %lu\n", Attributes->MaximumSurface2DWidth);
    printf("MaximumSurface2DHeight: %lu\n", Attributes->MaximumSurface2DHeight);
    printf("MaximumSurface3DWidth: %lu\n", Attributes->MaximumSurface3DWidth);
    printf("MaximumSurface3DHeight: %lu\n", Attributes->MaximumSurface3DHeight);
    printf("MaximumSurface3DDepth: %lu\n", Attributes->MaximumSurface3DDepth);
    printf("MaximumSurface1DLayeredWidth: %lu\n", Attributes->MaximumSurface1DLayeredWidth);

    printf("DMABufSupported: %lu\n", Attributes->DMABufSupported);
#endif
}

