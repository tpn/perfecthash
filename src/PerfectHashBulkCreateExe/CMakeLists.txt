project(PerfectHashBulkCreateExe)

set(PROJECT_NAME PerfectHashBulkCreateExe)

################################################################################
# Source groups
################################################################################
set(Header_Files
    "stdafx.h"
    "targetver.h"
)
source_group("Header Files" FILES ${Header_Files})

set(Source_Files
    "PerfectHashBulkCreateExe.c"
    "stdafx.c"
)
source_group("Source Files" FILES ${Source_Files})

set(ALL_FILES
    ${Header_Files}
    ${Source_Files}
)

add_executable(PerfectHashBulkCreateExe ${ALL_FILES})

################################################################################
# Target
################################################################################

use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "${DEFAULT_CXX_PROPS}")
################################################################################
# Includes for CMake from *.props
################################################################################
use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "../PerfectHash.cmake")

set(ROOT_NAMESPACE PerfectHashBulkCreateExe)

################################################################################
# Target name
################################################################################
set_target_properties(${PROJECT_NAME} PROPERTIES
    TARGET_NAME "PerfectHashBulkCreate"
)
################################################################################
# Compile definitions
################################################################################

################################################################################
# Compile and link options
################################################################################
################################################################################
# Dependencies
################################################################################
add_dependencies(${PROJECT_NAME}
    PerfectHash
)

target_link_libraries(${PROJECT_NAME} PerfectHash)

#  vim:set ts=8 sw=4 sts=4 tw=80 expandtab syntax=cmake                        :