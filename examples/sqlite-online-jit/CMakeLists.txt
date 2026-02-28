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
option(
    PH_ONLINE_JIT_USE_FETCHCONTENT
    "Fetch PerfectHash from GitHub via FetchContent."
    ON
)

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
)

if(PH_ONLINE_JIT_LLVM_TARGET)
    target_compile_definitions(
        sqlite-online-jit
        PRIVATE
            PH_ONLINE_JIT_LLVM_LIBRARY_PATH="$<TARGET_FILE:${PH_ONLINE_JIT_LLVM_TARGET}>"
    )
elseif(DEFINED PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY AND
       NOT PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY STREQUAL "")
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
if(DEFINED PERFECTHASH_ONLINE_JIT_LIBRARY_DIR AND
   NOT PERFECTHASH_ONLINE_JIT_LIBRARY_DIR STREQUAL "")
    list(APPEND _ph_runtime_dirs "${PERFECTHASH_ONLINE_JIT_LIBRARY_DIR}")
endif()
if(DEFINED PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY_DIR AND
   NOT PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY_DIR STREQUAL "")
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
    if(PH_ONLINE_JIT_RUNTIME_TARGET)
        add_custom_command(
            TARGET sqlite-online-jit
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:${PH_ONLINE_JIT_RUNTIME_TARGET}>"
                    "$<TARGET_FILE_DIR:sqlite-online-jit>"
            COMMENT "Copying FetchContent PerfectHash runtime DLL."
        )
    endif()
    if(PH_ONLINE_JIT_LLVM_TARGET)
        add_custom_command(
            TARGET sqlite-online-jit
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:${PH_ONLINE_JIT_LLVM_TARGET}>"
                    "$<TARGET_FILE_DIR:sqlite-online-jit>"
            COMMENT "Copying FetchContent PerfectHash LLVM runtime DLL."
        )
    endif()

    foreach(_dll_var
            PERFECTHASH_ONLINE_JIT_RUNTIME_DLL
            PERFECTHASH_ONLINE_JIT_LLVM_RUNTIME_DLL)
        if(DEFINED ${_dll_var} AND EXISTS "${${_dll_var}}")
            add_custom_command(
                TARGET sqlite-online-jit
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${${_dll_var}}"
                        "$<TARGET_FILE_DIR:sqlite-online-jit>"
                COMMENT "Copying ${_dll_var} next to sqlite-online-jit."
            )
        endif()
    endforeach()
endif()
