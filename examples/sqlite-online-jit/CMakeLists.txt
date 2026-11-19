cmake_minimum_required(VERSION 3.20)

project(sqlite-online-jit LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

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
    "Fetch PerfectHash via FetchContent (AUTO uses a local tree when an SDK root is set)."
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

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/PerfectHashOnlineJitRoots.cmake")
if(PH_ONLINE_JIT_USE_FETCHCONTENT STREQUAL "AUTO")
    if(_ph_sdk_roots)
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
    # This example prefers the full Online library and the installed header layout.
    set(PH_ONLINE_JIT_LIBRARY_NAMES PerfectHashOnline PerfectHashOnlineCore)
    set(PH_ONLINE_JIT_HEADER_NAMES PerfectHash/PerfectHashOnlineJit.h)
    set(PH_ONLINE_JIT_HEADER_SUFFIXES)
    find_package(PerfectHashOnlineJit REQUIRED)
    set(PH_ONLINE_JIT_TARGET PerfectHash::OnlineJit)
endif()

add_library(sqlite3-amalgamation STATIC
    sqlite/sqlite3.c
)

target_include_directories(sqlite3-amalgamation
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/sqlite"
)

target_compile_definitions(sqlite3-amalgamation
    PUBLIC
        SQLITE_THREADSAFE=1
        SQLITE_OMIT_LOAD_EXTENSION=1
)

if(MSVC)
    target_compile_definitions(sqlite3-amalgamation PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

add_executable(sqlite-online-jit
    src/main.cpp
    src/perfecthash_vtab.cpp
    src/perfecthash_vtab.h
)

target_include_directories(sqlite-online-jit
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/sqlite"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(sqlite-online-jit
    PRIVATE
        sqlite3-amalgamation
        ${PH_ONLINE_JIT_TARGET}
        ${CMAKE_DL_LIBS}
)

if(PH_ONLINE_JIT_LLVM_TARGET)
    target_compile_definitions(
        sqlite-online-jit
        PRIVATE
            PH_ONLINE_JIT_LLVM_LIBRARY_PATH="$<TARGET_FILE:${PH_ONLINE_JIT_LLVM_TARGET}>"
    )
elseif(NOT PH_ONLINE_JIT_USE_FETCHCONTENT AND PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY)
    target_compile_definitions(
        sqlite-online-jit
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
        TARGET sqlite-online-jit
        PROPERTY BUILD_RPATH "${_ph_build_rpath}"
    )
endif()

if(WIN32)
    if(PH_ONLINE_JIT_USE_FETCHCONTENT)
        set(_ph_runtime_files "$<TARGET_FILE:${PH_ONLINE_JIT_RUNTIME_TARGET}>")
        if(PH_ONLINE_JIT_LLVM_TARGET)
            list(APPEND _ph_runtime_files "$<TARGET_FILE:${PH_ONLINE_JIT_LLVM_TARGET}>")
        endif()
    else()
        set(_ph_runtime_files "${PERFECTHASH_ONLINE_JIT_RUNTIME_FILES}")
    endif()

    set(_ph_runtime_state "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/perfecthash-runtime")
    set(_ph_runtime_selection "${_ph_runtime_state}/$<CONFIG>.selection")
    set(_ph_stage_script "${CMAKE_CURRENT_SOURCE_DIR}/cmake/StagePerfectHashRuntime.cmake")
    # The stamp lives in the executable's default output directory, so removing
    # that directory also stages its runtime again.
    get_target_property(_ph_runtime_dir sqlite-online-jit RUNTIME_OUTPUT_DIRECTORY)
    if(NOT _ph_runtime_dir)
        set(_ph_runtime_dir "${CMAKE_CURRENT_BINARY_DIR}")
    endif()
    get_property(_ph_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(_ph_multi_config)
        string(APPEND _ph_runtime_dir "/$<CONFIG>")
    endif()
    set(_ph_runtime_stamp "${_ph_runtime_dir}/sqlite-online-jit.runtime.stamp")
    # Rewritten only when the selected files or their destination change.
    file(GENERATE OUTPUT "${_ph_runtime_selection}"
        CONTENT "$<TARGET_FILE_DIR:sqlite-online-jit>\n${_ph_runtime_files}\n")

    # Staging is a build dependency so switching SDKs also updates an up-to-date
    # executable. It reruns when the selection, a selected file or the script
    # changes. The quoted list keeps each configuration's files in one expression.
    add_custom_command(OUTPUT "${_ph_runtime_stamp}"
        COMMAND ${CMAKE_COMMAND} "-Dfiles=${_ph_runtime_files}"
                "-Ddestination=$<TARGET_FILE_DIR:sqlite-online-jit>"
                "-Dstate_dir=${_ph_runtime_state}"
                -P "${_ph_stage_script}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_ph_runtime_dir}"
        COMMAND ${CMAKE_COMMAND} -E touch "${_ph_runtime_stamp}"
        DEPENDS "${_ph_runtime_files}" "${_ph_runtime_selection}" "${_ph_stage_script}"
        VERBATIM)
    add_custom_target(sqlite-online-jit-runtime DEPENDS "${_ph_runtime_stamp}")
    add_dependencies(sqlite-online-jit sqlite-online-jit-runtime)
endif()
