set(target PerfectHashCuda)
#find_package(stdgpu REQUIRED)

################################################################################
# Source groups
################################################################################
set(Header_Files
    "Graph.cuh"
)
source_group("Header Files" FILES ${Header_Files})

set(Source_Files
    "Graph.cu"
    #"Graph3.cu"
    #"Graph4.cu"
    #"GraphThrust.cu"
)
source_group("Source Files" FILES ${Source_Files})

set(Resource_Files
    "PerfectHashCuda.def"
    "PerfectHashCuda.rc"
)
source_group("Resource Files" FILES ${Resource_Files})

set(ALL_FILES
    ${Header_Files}
    ${Source_Files}
)

if(WIN32)
    list(
        APPEND ALL_FILES
        ${Resource_Files}
    )
endif()

add_library(${target} SHARED ${ALL_FILES})
#set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_PTX_COMPILATION ON)

set_target_properties(${target}
   PROPERTIES
       CUDA_SEPARABLE_COMPILATION ON
       CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

################################################################################
# Target
################################################################################

perfecthash_apply_cuda_settings(${target})

################################################################################
# Output shared library name
################################################################################
set_target_properties(${target} PROPERTIES
    OUTPUT_NAME "PerfectHashCuda"
)
################################################################################
# Compile definitions
################################################################################
target_compile_definitions(
    ${target}
    PRIVATE
    "_PERFECT_HASH_CUDA_INTERNAL_BUILD"
    "_PERFECT_HASH_INTERNAL_BUILD"
)

################################################################################
# Compile and link options
################################################################################
if(CMAKE_CUDA_COMPILER_ID MATCHES NVIDIA)
    target_compile_options(
        ${target}
        PRIVATE
        --extended-lambda
        --source-in-ptx
        -Wno-deprecated-gpu-targets
        -v
        -rdc=true
        -cudart shared
        -maxrregcount=0
        --restrict
        --ptxas-options=-v
        --machine 64
        --forward-unknown-opts
    )
    target_compile_options(
        ${target}
        PRIVATE
        $<$<CONFIG:Debug>:-g;-G>
        $<$<NOT:$<CONFIG:Debug>>:-lineinfo>
    )

    target_compile_definitions(
        ${target}
        PRIVATE
        $<$<CONFIG:Debug>:_DEBUG>
        $<$<NOT:$<CONFIG:Debug>>:NDEBUG>
    )

    #if (IS_WINDOWS)
    #    target_link_options(
    #        ${PROJECT_NAME}
    #        PRIVATE
    #        -Xlinker "/FdPerfectHashCuda.pdb"
    #    )
    #endif()

endif()


################################################################################
# Dependencies
################################################################################
#add_dependencies(${target}
#    PerfectHash
#)

#target_link_libraries(${target} PerfectHash)
#target_link_libraries(${PROJECT_NAME} PUBLIC stdgpu::stdgpu)

if(CUDAToolkit_FOUND)
    target_link_libraries(${target} PRIVATE CUDA::cudart)
endif()

#  vim:set ts=8 sw=4 sts=4 tw=80 expandtab syntax=cmake                        :
