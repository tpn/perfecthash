cmake_minimum_required(VERSION 3.20)

project(cpp-console-online-jit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(
    PH_ONLINE_JIT_PREFER_SHARED
    "Prefer shared PerfectHash runtime over static."
    ON
)
set(
    PH_ONLINE_JIT_USE_FETCHCONTENT AUTO CACHE STRING
    "Fetch PerfectHash via FetchContent (AUTO uses a local tree if PERFECTHASH_ROOT is set)."
)
set_property(CACHE PH_ONLINE_JIT_USE_FETCHCONTENT PROPERTY STRINGS AUTO ON OFF)

set(
    PERFECTHASH_GIT_REPOSITORY
    "https://github.com/tpn/perfecthash.git"
    CACHE STRING
    "PerfectHash git repository used with FetchContent."
)
set(
    PERFECTHASH_GIT_TAG
    "main"
    CACHE STRING
    "PerfectHash git ref used with FetchContent."
)
set(
    PERFECTHASH_BUILD_PROFILE
    "online-rawdog-and-llvm-jit"
    CACHE STRING
    "PerfectHash profile used with FetchContent."
)
set(
    PERFECTHASH_ROOT
    ""
    CACHE PATH
    "Path hint to a PerfectHash repo root, build tree, or install prefix."
)

set(PH_ONLINE_JIT_TARGET "")
set(PH_ONLINE_JIT_RUNTIME_TARGET "")
set(PH_ONLINE_JIT_LLVM_TARGET "")

if(PH_ONLINE_JIT_USE_FETCHCONTENT STREQUAL "AUTO")
    if(PERFECTHASH_ROOT)
        set(PH_ONLINE_JIT_USE_FETCHCONTENT OFF)
    else()
        set(PH_ONLINE_JIT_USE_FETCHCONTENT ON)
    endif()
endif()

if(PH_ONLINE_JIT_USE_FETCHCONTENT)
    include(FetchContent)

    set(PERFECTHASH_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(PERFECTHASH_ENABLE_INSTALL ON CACHE BOOL "" FORCE)
    set(PERFECTHASH_BUILD_PROFILE "${PERFECTHASH_BUILD_PROFILE}" CACHE STRING "" FORCE)

    FetchContent_Declare(
        PerfectHash
        GIT_REPOSITORY ${PERFECTHASH_GIT_REPOSITORY}
        GIT_TAG ${PERFECTHASH_GIT_TAG}
    )
    FetchContent_MakeAvailable(PerfectHash)

    if(TARGET PerfectHash::PerfectHashOnlineCore)
        set(PH_ONLINE_JIT_TARGET PerfectHash::PerfectHashOnlineCore)
        set(PH_ONLINE_JIT_RUNTIME_TARGET PerfectHash::PerfectHashOnlineCore)
    elseif(TARGET PerfectHash::PerfectHashOnline)
        set(PH_ONLINE_JIT_TARGET PerfectHash::PerfectHashOnline)
        set(PH_ONLINE_JIT_RUNTIME_TARGET PerfectHash::PerfectHashOnline)
    else()
        message(
            FATAL_ERROR
            "FetchContent build did not expose PerfectHash online targets. "
            "Verify PERFECTHASH_BUILD_PROFILE='${PERFECTHASH_BUILD_PROFILE}'."
        )
    endif()

    if(TARGET PerfectHash::PerfectHashLLVM)
        set(PH_ONLINE_JIT_LLVM_TARGET PerfectHash::PerfectHashLLVM)
    endif()
else()
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    find_package(PerfectHashOnlineJit REQUIRED)
    set(PH_ONLINE_JIT_TARGET PerfectHash::OnlineJit)
endif()

add_executable(cpp-console-online-jit
    src/main.cpp
)

target_link_libraries(cpp-console-online-jit
    PRIVATE
        ${PH_ONLINE_JIT_TARGET}
)

if(PH_ONLINE_JIT_LLVM_TARGET)
    target_compile_definitions(
        cpp-console-online-jit
        PRIVATE
            PH_ONLINE_JIT_LLVM_LIBRARY_PATH="$<TARGET_FILE:${PH_ONLINE_JIT_LLVM_TARGET}>"
    )
elseif(NOT PH_ONLINE_JIT_USE_FETCHCONTENT AND PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY)
    target_compile_definitions(
        cpp-console-online-jit
        PRIVATE
            PH_ONLINE_JIT_LLVM_LIBRARY_PATH="${PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY}"
    )
endif()

set(_ph_runtime_dirs)
if(PH_ONLINE_JIT_RUNTIME_TARGET)
    list(APPEND _ph_runtime_dirs "$<TARGET_FILE_DIR:${PH_ONLINE_JIT_RUNTIME_TARGET}>")
endif()
if(PH_ONLINE_JIT_LLVM_TARGET)
    list(APPEND _ph_runtime_dirs "$<TARGET_FILE_DIR:${PH_ONLINE_JIT_LLVM_TARGET}>")
endif()
if(NOT PH_ONLINE_JIT_USE_FETCHCONTENT AND PERFECTHASH_ONLINE_JIT_LIBRARY_DIR)
    list(APPEND _ph_runtime_dirs "${PERFECTHASH_ONLINE_JIT_LIBRARY_DIR}")
endif()
if(NOT PH_ONLINE_JIT_USE_FETCHCONTENT AND PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY_DIR)
    list(APPEND _ph_runtime_dirs "${PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY_DIR}")
endif()
list(REMOVE_DUPLICATES _ph_runtime_dirs)

if(_ph_runtime_dirs AND (APPLE OR UNIX))
    string(JOIN ";" _ph_build_rpath ${_ph_runtime_dirs})
    set_property(
        TARGET cpp-console-online-jit
        PROPERTY BUILD_RPATH "${_ph_build_rpath}"
    )
endif()

if(WIN32)
    if(PH_ONLINE_JIT_USE_FETCHCONTENT)
        set(_ph_runtime_files "$<TARGET_FILE:${PH_ONLINE_JIT_RUNTIME_TARGET}>")
        if(PH_ONLINE_JIT_LLVM_TARGET)
            list(APPEND _ph_runtime_files "$<TARGET_FILE:${PH_ONLINE_JIT_LLVM_TARGET}>"
                                          "$<TARGET_FILE:PerfectHash::PerfectHash>")
        endif()
    else()
        set(_ph_runtime_files "${PERFECTHASH_ONLINE_JIT_RUNTIME_FILES}")
    endif()

    # Staging is a build dependency so switching SDKs also updates an up-to-date executable.
    add_custom_target(cpp-console-online-jit-runtime
        COMMAND ${CMAKE_COMMAND} "-Dfiles=${_ph_runtime_files}"
                "-Ddestination=$<TARGET_FILE_DIR:cpp-console-online-jit>"
                "-Dstate_dir=${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/perfecthash-runtime"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/StagePerfectHashRuntime.cmake"
        VERBATIM)
    add_dependencies(cpp-console-online-jit cpp-console-online-jit-runtime)
endif()
