if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
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
    include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/PerfectHashVersion.cmake")
    perfecthash_resolve_version(
        SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.."
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

    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
    include(PerfectHashProject)
    include(PerfectHashTargets)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(PERFECTHASH_ENABLE_PENTER)
    add_subdirectory(FunctionHook)
endif()
if(PERFECTHASH_ARCH_X86_64)
    add_subdirectory(PerfectHashAsm)
endif()
add_subdirectory(PerfectHash)
if(PERFECTHASH_BUILD_EXES)
    add_subdirectory(PerfectHashCreateExe)
    add_subdirectory(PerfectHashBulkCreateExe)
endif()

if(PERFECTHASH_HAS_CUDA)
    add_subdirectory(PerfectHashCuda)
endif()

if(WIN32)
    # N.B. PerfectHashResources is defined in PerfectHash's CMakeLists.txt.
    add_dependencies(PerfectHash PerfectHashResources)
    if(TARGET PerfectHashCreateExe)
        add_dependencies(PerfectHashCreateExe PerfectHashResources)
    endif()
    if(TARGET PerfectHashBulkCreateExe)
        add_dependencies(PerfectHashBulkCreateExe PerfectHashResources)
    endif()
    if(PERFECTHASH_HAS_CUDA)
        add_dependencies(PerfectHashCuda PerfectHashResources)
    endif()
endif()

if(PERFECTHASH_HAS_CUDA)
    add_dependencies(PerfectHash PerfectHashCuda)
endif()

#  vim:set ts=8 sw=4 sts=4 tw=80 expandtab syntax=cmake                        :
