project(PerfectHashCreateExe)

set(PROJECT_NAME PerfectHashCreateExe)

################################################################################
# Source groups
################################################################################
set(Header_Files
    "stdafx.h"
    "targetver.h"
)
source_group("Header Files" FILES ${Header_Files})

set(Source_Files
    "PerfectHashCreateExe.c"
    #"stdafx.c"
)
source_group("Source Files" FILES ${Source_Files})

set(ALL_FILES
    ${Header_Files}
    ${Source_Files}
)

set(Resource_Files
    "PerfectHashCreateExe.rc"
)
source_group("Resource Files" FILES ${Resource_Files})

if (IS_WINDOWS)
    list(
        APPEND ALL_FILES
        ${Resource_Files}
    )
endif()

add_executable(PerfectHashCreateExe ${ALL_FILES})

################################################################################
# Target
################################################################################

use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "${DEFAULT_CXX_PROPS}")
################################################################################
# Includes for CMake from *.props
################################################################################
use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "../PerfectHash.cmake")

set(ROOT_NAMESPACE PerfectHashCreateExe)

################################################################################
# Output executable name
################################################################################
set_target_properties(${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME "PerfectHashCreate"
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

target_link_libraries(${PROJECT_NAME} PerfectHash ${CMAKE_DL_LIBS})

#  vim:set ts=8 sw=4 sts=4 tw=80 expandtab syntax=cmake                        :
