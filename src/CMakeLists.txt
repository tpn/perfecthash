if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.20)

    if(WIN32 AND NOT DEFINED CMAKE_SYSTEM_VERSION)
        set(CMAKE_SYSTEM_VERSION 10.0 CACHE STRING "" FORCE)
    endif()

    project(
        PerfectHash
        VERSION 0.63.0
        LANGUAGES C CXX
    )

    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
    include(PerfectHashProject)
    include(PerfectHashTargets)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(PERFECTHASH_ENABLE_PENTER)
    add_subdirectory(FunctionHook)
endif()
add_subdirectory(PerfectHashAsm)
add_subdirectory(PerfectHash)
add_subdirectory(PerfectHashCreateExe)
add_subdirectory(PerfectHashBulkCreateExe)

if(PERFECTHASH_HAS_CUDA)
    add_subdirectory(PerfectHashCuda)
endif()

if(WIN32)
    # N.B. PerfectHashResources is defined in PerfectHash's CMakeLists.txt.
    add_dependencies(PerfectHash PerfectHashResources)
    add_dependencies(PerfectHashCreateExe PerfectHashResources)
    add_dependencies(PerfectHashBulkCreateExe PerfectHashResources)
    if(PERFECTHASH_HAS_CUDA)
        add_dependencies(PerfectHashCuda PerfectHashResources)
    endif()
endif()

if(PERFECTHASH_HAS_CUDA)
    add_dependencies(PerfectHash PerfectHashCuda)
endif()

#  vim:set ts=8 sw=4 sts=4 tw=80 expandtab syntax=cmake                        :
