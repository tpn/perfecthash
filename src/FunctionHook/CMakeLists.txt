project(FunctionHook)

set(PROJECT_NAME FunctionHook)

################################################################################
# Source groups
################################################################################

set(Sources
    "dllmain.c"
    "FunctionHook.c"
    "_penter_x64.asm"
    "FunctionHook.def"
)
source_group("Source Files" FILES ${Sources})

set(ALL_FILES
    ${Sources}
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

set(ROOT_NAMESPACE FunctionHook)

################################################################################
# Compile definitions
################################################################################

################################################################################
# Compile and link options
################################################################################
