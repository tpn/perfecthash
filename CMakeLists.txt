cmake_minimum_required(VERSION 3.20)

if(WIN32 AND NOT DEFINED CMAKE_SYSTEM_VERSION)
    set(CMAKE_SYSTEM_VERSION 10.0 CACHE STRING "" FORCE)
endif()

set(PERFECTHASH_FALLBACK_VERSION "0.63.0")
set(
    PERFECTHASH_VERSION_OVERRIDE
    ""
    CACHE STRING
    "Override PerfectHash project version (for release automation)."
)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/PerfectHashVersion.cmake")
perfecthash_resolve_version(
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    FALLBACK_VERSION "${PERFECTHASH_FALLBACK_VERSION}"
    OVERRIDE "${PERFECTHASH_VERSION_OVERRIDE}"
    OUT_VERSION PERFECTHASH_PROJECT_VERSION
    OUT_SOURCE PERFECTHASH_PROJECT_VERSION_SOURCE
    OUT_SEMVER PERFECTHASH_PROJECT_SEMVER
    OUT_GIT_TAG PERFECTHASH_GIT_TAG
    OUT_GIT_SHA PERFECTHASH_GIT_SHA
    OUT_GIT_DIRTY PERFECTHASH_GIT_DIRTY
)

project(
    PerfectHash
    VERSION ${PERFECTHASH_PROJECT_VERSION}
    LANGUAGES C CXX
)

message(
    STATUS
    "PerfectHash version: ${PROJECT_VERSION} "
    "(source: ${PERFECTHASH_PROJECT_VERSION_SOURCE}, semver: ${PERFECTHASH_PROJECT_SEMVER})"
)

include(CTest)
include(CMakePackageConfigHelpers)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(
        CMAKE_INSTALL_PREFIX
        "${CMAKE_SOURCE_DIR}/install"
        CACHE PATH "Default install prefix." FORCE
    )
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(PerfectHashProject)
include(PerfectHashTargets)

add_subdirectory(src)

function(perfecthash_add_namespace_alias target_name)
    if(NOT TARGET "${target_name}")
        return()
    endif()
    if(TARGET "PerfectHash::${target_name}")
        return()
    endif()
    get_target_property(target_type "${target_name}" TYPE)
    if(target_type STREQUAL "EXECUTABLE")
        add_executable("PerfectHash::${target_name}" ALIAS "${target_name}")
    else()
        add_library("PerfectHash::${target_name}" ALIAS "${target_name}")
    endif()
endfunction()

set(PERFECTHASH_EXPORT_TARGETS)
if(PERFECTHASH_BUILD_PROFILE_NORMALIZED STREQUAL "online-rawdog")
    list(APPEND PERFECTHASH_EXPORT_TARGETS PerfectHashOnlineCore)
elseif(PERFECTHASH_BUILD_PROFILE_NORMALIZED STREQUAL "online-rawdog-llvm")
    list(APPEND PERFECTHASH_EXPORT_TARGETS PerfectHashOnlineCore)
    if(TARGET PerfectHashLLVM)
        list(APPEND PERFECTHASH_EXPORT_TARGETS PerfectHashLLVM)
    endif()
    if(TARGET PerfectHash)
        # PerfectHashLLVM currently links to PerfectHash.
        list(APPEND PERFECTHASH_EXPORT_TARGETS PerfectHash)
    endif()
else()
    list(
        APPEND PERFECTHASH_EXPORT_TARGETS
        PerfectHash
        PerfectHashOnline
        PerfectHashOnlineStatic
        PerfectHashOnlineCore
        PerfectHashOnlineCoreStatic
    )
    if(PERFECTHASH_BUILD_EXES)
        list(APPEND PERFECTHASH_EXPORT_TARGETS PerfectHashCreateExe PerfectHashBulkCreateExe)
    endif()
    if(TARGET PerfectHashLLVM)
        list(APPEND PERFECTHASH_EXPORT_TARGETS PerfectHashLLVM)
    endif()
endif()
if(TARGET PerfectHashAsm)
    list(APPEND PERFECTHASH_EXPORT_TARGETS PerfectHashAsm)
endif()
if(TARGET PerfectHashCuda)
    list(APPEND PERFECTHASH_EXPORT_TARGETS PerfectHashCuda)
endif()
if(TARGET FunctionHook)
    list(APPEND PERFECTHASH_EXPORT_TARGETS FunctionHook)
endif()
list(REMOVE_DUPLICATES PERFECTHASH_EXPORT_TARGETS)

foreach(perfecthash_export_target ${PERFECTHASH_EXPORT_TARGETS})
    perfecthash_add_namespace_alias("${perfecthash_export_target}")
endforeach()

if(PERFECTHASH_ENABLE_TESTS AND BUILD_TESTING)
    add_subdirectory(tests)
endif()

set(
    PERFECTHASH_INSTALL_CMAKEDIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/PerfectHash"
    CACHE PATH
    "Install path for PerfectHash CMake package files."
)

if(PERFECTHASH_ENABLE_INSTALL)
    install(DIRECTORY "${PERFECTHASH_ROOT_DIR}/include/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    if(PERFECTHASH_EXPORT_TARGETS)
        install(
            TARGETS ${PERFECTHASH_EXPORT_TARGETS}
            EXPORT PerfectHashTargets
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
    endif()

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PerfectHashConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashConfig.cmake"
        INSTALL_DESTINATION "${PERFECTHASH_INSTALL_CMAKEDIR}"
    )
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY SameMajorVersion
    )

    install(
        FILES
            "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashConfig.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashConfigVersion.cmake"
        DESTINATION "${PERFECTHASH_INSTALL_CMAKEDIR}"
    )
    install(
        EXPORT PerfectHashTargets
        NAMESPACE PerfectHash::
        FILE PerfectHashTargets.cmake
        DESTINATION "${PERFECTHASH_INSTALL_CMAKEDIR}"
    )

endif()
