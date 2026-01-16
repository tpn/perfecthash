set(target PerfectHashCreateExe)

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

if(WIN32)
    list(
        APPEND ALL_FILES
        ${Resource_Files}
    )
endif()

add_executable(${target} ${ALL_FILES})

################################################################################
# Target
################################################################################

perfecthash_apply_common_settings(${target})

################################################################################
# Output executable name
################################################################################
set_target_properties(${target} PROPERTIES
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
add_dependencies(${target}
    PerfectHash
)

target_link_libraries(${target} PRIVATE PerfectHash ${CMAKE_DL_LIBS})

#  vim:set ts=8 sw=4 sts=4 tw=80 expandtab syntax=cmake                        :
