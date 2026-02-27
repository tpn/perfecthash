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
    PERFECTHASH_ROOT
    ""
    CACHE PATH
    "Path hint to a PerfectHash repo root, build tree, or install prefix."
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(PerfectHashOnlineJit REQUIRED)

add_executable(cpp-console-online-jit
    src/main.cpp
)

target_link_libraries(cpp-console-online-jit
    PRIVATE
        PerfectHash::OnlineJit
)

if(DEFINED PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY AND
   NOT PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY STREQUAL "")
    target_compile_definitions(
        cpp-console-online-jit
        PRIVATE
            PH_ONLINE_JIT_LLVM_LIBRARY_PATH="${PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY}"
    )
endif()

set(_ph_runtime_dirs)
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
        TARGET cpp-console-online-jit
        PROPERTY BUILD_RPATH "${_ph_build_rpath}"
    )
endif()

if(WIN32)
    foreach(_dll_var
            PERFECTHASH_ONLINE_JIT_RUNTIME_DLL
            PERFECTHASH_ONLINE_JIT_LLVM_RUNTIME_DLL)
        if(DEFINED ${_dll_var} AND EXISTS "${${_dll_var}}")
            add_custom_command(
                TARGET cpp-console-online-jit
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${${_dll_var}}"
                        "$<TARGET_FILE_DIR:cpp-console-online-jit>"
                COMMENT "Copying ${_dll_var} next to example executable."
            )
        endif()
    endforeach()
endif()
