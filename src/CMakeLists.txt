cmake_minimum_required(VERSION 3.16.0 FATAL_ERROR)

set(CMAKE_SYSTEM_VERSION 10.0 CACHE STRING "" FORCE)

project(
    PerfectHash
    VERSION 0.62.0
    LANGUAGES C CXX CUDA
)

set(PROJECT_ROOT "${CMAKE_SOURCE_DIR}/..")
set(SRC_ROOT "${CMAKE_SOURCE_DIR}")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_EXPORT_LINK_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)

################################################################################
# Set target arch type if empty. Visual studio solution generator provides it.
################################################################################
if(NOT CMAKE_VS_PLATFORM_NAME)
    set(CMAKE_VS_PLATFORM_NAME "x64")
endif()
message("${CMAKE_VS_PLATFORM_NAME} architecture in use")

if(NOT ("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64"))
    message(FATAL_ERROR "${CMAKE_VS_PLATFORM_NAME} arch is not supported!")
endif()

################################################################################
# Global configuration types
################################################################################
set(CMAKE_CONFIGURATION_TYPES
    "Debug"
    "PGInstrument"
    "PGOptimize"
    "PGUpdate"
    "Release"
    CACHE STRING "" FORCE
)

###############################################################################
# CUDA
###############################################################################

# Define a CMake option to enable or disable CUDA manually
option(USE_CUDA "Enable CUDA support if available" ON)

# Find CUDA if the user did not explicitly disable it
if(USE_CUDA)
    find_package(CUDA)
endif()

# Set the HAS_CUDA variable based on whether CUDA was found
if(CUDA_FOUND)
    message(STATUS "CUDA found, enabling CUDA support")
    set(HAS_CUDA TRUE)
else()
    message(STATUS "CUDA not found, CUDA support is disabled")
    set(HAS_CUDA FALSE)
endif()

###############################################################################
# Global compiler options
###############################################################################
if (MSVC)
    # remove default flags provided with CMake for MSVC
    set(CMAKE_C_FLAGS "")
    set(CMAKE_C_FLAGS_DEBUG "")
    set(CMAKE_C_FLAGS_PGINSTRUMENT "")
    set(CMAKE_C_FLAGS_PGOPTIMIZE "")
    set(CMAKE_C_FLAGS_PGUPDATE "")
    set(CMAKE_C_FLAGS_RELEASE "")
endif()

################################################################################
# Global linker options
################################################################################
if(MSVC)
    # remove default flags provided with CMake for MSVC
    set(CMAKE_EXE_LINKER_FLAGS "")
    set(CMAKE_MODULE_LINKER_FLAGS "")
    set(CMAKE_SHARED_LINKER_FLAGS "")
    set(CMAKE_STATIC_LINKER_FLAGS "")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS_DEBUG "${CMAKE_MODULE_LINKER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS}")
    set(CMAKE_STATIC_LINKER_FLAGS_DEBUG "${CMAKE_STATIC_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS_PGINSTRUMENT "${CMAKE_EXE_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS_PGINSTRUMENT "${CMAKE_MODULE_LINKER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS_PGINSTRUMENT "${CMAKE_SHARED_LINKER_FLAGS}")
    set(CMAKE_STATIC_LINKER_FLAGS_PGINSTRUMENT "${CMAKE_STATIC_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS_PGOPTIMIZE "${CMAKE_EXE_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS_PGOPTIMIZE "${CMAKE_MODULE_LINKER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS_PGOPTIMIZE "${CMAKE_SHARED_LINKER_FLAGS}")
    set(CMAKE_STATIC_LINKER_FLAGS_PGOPTIMIZE "${CMAKE_STATIC_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS_PGUPDATE "${CMAKE_EXE_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS_PGUPDATE "${CMAKE_MODULE_LINKER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS_PGUPDATE "${CMAKE_SHARED_LINKER_FLAGS}")
    set(CMAKE_STATIC_LINKER_FLAGS_PGUPDATE "${CMAKE_STATIC_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS_RELEASE "${CMAKE_MODULE_LINKER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS}")
    set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "${CMAKE_STATIC_LINKER_FLAGS}")
endif()

################################################################################
# Nuget packages function stub.
################################################################################
function(use_package TARGET PACKAGE VERSION)
    message(WARNING "No implementation of use_package. Create yours. "
                    "Package \"${PACKAGE}\" with version \"${VERSION}\" "
                    "for target \"${TARGET}\" is ignored!")
endfunction()

################################################################################
# Common utils
################################################################################
include(cmake/Utils.cmake)

################################################################################
# Additional Global Settings(add specific info there)
################################################################################
include(cmake/GlobalSettingsInclude.cmake OPTIONAL)

################################################################################
# Use solution folders feature
################################################################################
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    # Nothing.
else()
    message(FATAL "Only 64-bit platforms are supported.")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(IS_WINDOWS 1)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(IS_LINUX 1)
    set(IS_UNIX 1)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(IS_MAC 1)
    set(IS_UNIX 1)
else()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()

################################################################################
# Sub-projects
################################################################################
if (HookPenter STREQUAL "1")
    add_subdirectory(FunctionHook)
endif()
if (WIN32)
    add_subdirectory(PerfectHashAsm)
endif()
add_subdirectory(PerfectHash)
add_subdirectory(PerfectHashCreateExe)
add_subdirectory(PerfectHashBulkCreateExe)

if (HAS_CUDA)
    add_subdirectory(PerfectHashCuda)
endif()

if (IS_WINDOWS)
    # N.B. PerfectHashResources is defined in PerfectHash's CMakeLists.txt.
    add_dependencies(PerfectHash PerfectHashResources)
    add_dependencies(PerfectHashCreateExe PerfectHashResources)
    add_dependencies(PerfectHashBulkCreateExe PerfectHashResources)
    if (HAS_CUDA)
        add_dependencies(PerfectHashCuda PerfectHashResources)
    endif()
endif()

if (HAS_CUDA)
    add_dependencies(PerfectHash PerfectHashCuda)
endif()

#  vim:set ts=8 sw=4 sts=4 tw=80 expandtab syntax=cmake                        :
