project(PerfectHashCuda)

set(PROJECT_NAME PerfectHashCuda)
set(IS_CUDA 1)

find_package(stdgpu REQUIRED)

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
)
source_group("Resource Files" FILES ${Resource_Files})

set(ALL_FILES
    ${Header_Files}
    ${Source_Files}
    ${Resource_Files}
)

add_library(${PROJECT_NAME} SHARED ${ALL_FILES})
#set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_PTX_COMPILATION ON)

set_target_properties(${PROJECT_NAME}
   PROPERTIES
       CUDA_SEPARABLE_COMPILATION ON
       CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

################################################################################
# Target
################################################################################

use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "${DEFAULT_CXX_PROPS}")
################################################################################
# Includes for CMake from *.props
################################################################################
use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "../PerfectHash.cmake")

set(ROOT_NAMESPACE PerfectHashCuda)

################################################################################
# Target name
################################################################################
set_target_properties(${PROJECT_NAME} PROPERTIES
    TARGET_NAME "PerfectHashCuda"
)
################################################################################
# Compile definitions
################################################################################
target_compile_definitions(
    ${PROJECT_NAME}
    PRIVATE
    "_PERFECT_HASH_CUDA_INTERNAL_BUILD"
    "_PERFECT_HASH_INTERNAL_BUILD"
    "PH_CUDA"
)

################################################################################
# Compile and link options
################################################################################
#if (CMAKE_CUDA_COMPILER_ID MATCHES NVIDIA)
    target_compile_options(
        ${PROJECT_NAME}
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
    target_compile_options_config(Debug "-g;-G")
    target_compile_options_not_config(Debug "-lineinfo")

    target_compile_definitions_config(Debug _DEBUG)
    target_compile_definitions_not_config(Debug NDEBUG)

#    if (IS_WINDOWS)
#        target_link_options(
#            ${PROJECT_NAME}
#            PRIVATE
#            -Xlinker "/FdPerfectHashCuda.pdb"
#        )
#
#    endif()

#endif()


################################################################################
# Dependencies
################################################################################
#add_dependencies(${PROJECT_NAME}
#    PerfectHash
#)

#target_link_libraries(${PROJECT_NAME} PerfectHash)
target_link_libraries(${PROJECT_NAME} PUBLIC stdgpu::stdgpu)

#  vim:set ts=8 sw=4 sts=4 tw=80 expandtab syntax=cmake                        :
