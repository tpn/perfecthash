project(PerfectHashCuda)

set(PROJECT_NAME PerfectHashCuda)
set(IS_CUDA 1)

################################################################################
# Source groups
################################################################################
set(Header_Files
    "Graph.cuh"
)
source_group("Header Files" FILES ${Header_Files})

set(Source_Files
    #"Graph.cu"
    "GraphThrust.cu"
)
source_group("Source Files" FILES ${Source_Files})

set(ALL_FILES
    ${Header_Files}
    ${Source_Files}
)

add_library(${PROJECT_NAME} SHARED ${ALL_FILES})

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
    "PH_CUDA"
)

################################################################################
# Compile and link options
################################################################################
if (CMAKE_CUDA_COMPILER_ID MATCHES NVIDIA)
    target_compile_options(
        ${PROJECT_NAME}
        PUBLIC
        --source-in-ptx
        -Wno-deprecated-gpu-targets
        -v
        -rdc=true
        -cudart shared
        #-ptx
        -maxrregcount=0
        --restrict
        --ptxas-options=-v
        --machine 64
        --keep
    )
endif()


################################################################################
# Dependencies
################################################################################
#add_dependencies(${PROJECT_NAME}
#    PerfectHash
#)

#target_link_libraries(${PROJECT_NAME} PerfectHash)

#  vim:set ts=8 sw=4 sts=4 tw=80 expandtab syntax=cmake                        :
