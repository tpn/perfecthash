cmake_minimum_required(VERSION 3.20)

project(cpp-console-online-cuda-nvrtc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(
    PH_ONLINE_CUDA_NVRTC_PREFER_SHARED
    "Prefer shared PerfectHash runtime over static."
    ON
)
option(
    PH_ONLINE_CUDA_NVRTC_USE_FETCHCONTENT
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
    "online-rawdog-jit"
    CACHE STRING
    "PerfectHash profile used with FetchContent."
)
set(
    PERFECTHASH_ROOT
    ""
    CACHE PATH
    "Path hint to a PerfectHash repo root, build tree, or install prefix."
)

find_package(CUDAToolkit REQUIRED)
list(GET CUDAToolkit_INCLUDE_DIRS 0 PH_ONLINE_CUDA_NVRTC_CUDA_INCLUDE_DIR)

set(PH_ONLINE_CUDA_NVRTC_TARGET "")
set(PH_ONLINE_CUDA_NVRTC_RUNTIME_TARGET "")
set(PH_ONLINE_CUDA_NVRTC_LLVM_TARGET "")

if(PH_ONLINE_CUDA_NVRTC_USE_FETCHCONTENT)
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

    if(TARGET PerfectHash::PerfectHash)
        set(PH_ONLINE_CUDA_NVRTC_TARGET PerfectHash::PerfectHash)
        set(PH_ONLINE_CUDA_NVRTC_RUNTIME_TARGET PerfectHash::PerfectHash)
    elseif(TARGET PerfectHash::PerfectHashOnlineCore)
        set(PH_ONLINE_CUDA_NVRTC_TARGET PerfectHash::PerfectHashOnlineCore)
        set(PH_ONLINE_CUDA_NVRTC_RUNTIME_TARGET PerfectHash::PerfectHashOnlineCore)
    elseif(TARGET PerfectHash::PerfectHashOnline)
        set(PH_ONLINE_CUDA_NVRTC_TARGET PerfectHash::PerfectHashOnline)
        set(PH_ONLINE_CUDA_NVRTC_RUNTIME_TARGET PerfectHash::PerfectHashOnline)
    else()
        message(
            FATAL_ERROR
            "FetchContent build did not expose PerfectHash online targets. "
            "Verify PERFECTHASH_BUILD_PROFILE='${PERFECTHASH_BUILD_PROFILE}'."
        )
    endif()

    if(TARGET PerfectHash::PerfectHashLLVM)
        set(PH_ONLINE_CUDA_NVRTC_LLVM_TARGET PerfectHash::PerfectHashLLVM)
    endif()
else()
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cpp-console-online-jit/cmake")
    find_package(PerfectHashOnlineJit REQUIRED)
    set(PH_ONLINE_CUDA_NVRTC_FULL_LIBRARY "")
    if(EXISTS "${PERFECTHASH_ROOT}/build-verify-full/lib/libPerfectHash.so")
        set(PH_ONLINE_CUDA_NVRTC_FULL_LIBRARY
            "${PERFECTHASH_ROOT}/build-verify-full/lib/libPerfectHash.so")
    elseif(EXISTS "${PERFECTHASH_ROOT}/lib/libPerfectHash.so")
        set(PH_ONLINE_CUDA_NVRTC_FULL_LIBRARY
            "${PERFECTHASH_ROOT}/lib/libPerfectHash.so")
    else()
        find_library(
            PH_ONLINE_CUDA_NVRTC_FULL_LIBRARY
            NAMES PerfectHash
            HINTS ${PERFECTHASH_ROOT}
            PATH_SUFFIXES
                lib
                build/lib
                build-verify-full/lib
        )
    endif()
    if(PH_ONLINE_CUDA_NVRTC_FULL_LIBRARY)
        set(PH_ONLINE_CUDA_NVRTC_TARGET "${PH_ONLINE_CUDA_NVRTC_FULL_LIBRARY}")
    else()
        set(PH_ONLINE_CUDA_NVRTC_TARGET PerfectHash::OnlineJit)
    endif()
endif()

add_executable(cpp-console-online-cuda-nvrtc
    src/main.cpp
)

target_link_libraries(cpp-console-online-cuda-nvrtc
    PRIVATE
        ${PH_ONLINE_CUDA_NVRTC_TARGET}
        CUDA::cudart
        CUDA::nvrtc
        CUDA::nvJitLink
        CUDA::cuda_driver
)

if(DEFINED PERFECTHASH_ONLINE_JIT_INCLUDE_DIR AND
   NOT PERFECTHASH_ONLINE_JIT_INCLUDE_DIR STREQUAL "")
    get_filename_component(_ph_include_name "${PERFECTHASH_ONLINE_JIT_INCLUDE_DIR}" NAME)
    target_include_directories(
        cpp-console-online-cuda-nvrtc
        PRIVATE
            "${PERFECTHASH_ONLINE_JIT_INCLUDE_DIR}"
    )
    if(_ph_include_name STREQUAL "PerfectHash")
        get_filename_component(_ph_include_parent "${PERFECTHASH_ONLINE_JIT_INCLUDE_DIR}" DIRECTORY)
        target_include_directories(
            cpp-console-online-cuda-nvrtc
            PRIVATE
                "${_ph_include_parent}"
        )
    endif()
endif()

target_compile_definitions(
    cpp-console-online-cuda-nvrtc
    PRIVATE
        PH_CUDA_INCLUDE_DIR="${PH_ONLINE_CUDA_NVRTC_CUDA_INCLUDE_DIR}"
)

if(PH_ONLINE_CUDA_NVRTC_LLVM_TARGET)
    target_compile_definitions(
        cpp-console-online-cuda-nvrtc
        PRIVATE
            PH_ONLINE_JIT_LLVM_LIBRARY_PATH="$<TARGET_FILE:${PH_ONLINE_CUDA_NVRTC_LLVM_TARGET}>"
    )
elseif(DEFINED PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY AND
       NOT PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY STREQUAL "")
    target_compile_definitions(
        cpp-console-online-cuda-nvrtc
        PRIVATE
            PH_ONLINE_JIT_LLVM_LIBRARY_PATH="${PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY}"
    )
elseif(EXISTS "${PERFECTHASH_ROOT}/build-verify-full/lib/libPerfectHashLLVM.so")
    target_compile_definitions(
        cpp-console-online-cuda-nvrtc
        PRIVATE
            PH_ONLINE_JIT_LLVM_LIBRARY_PATH="${PERFECTHASH_ROOT}/build-verify-full/lib/libPerfectHashLLVM.so"
    )
elseif(EXISTS "${PERFECTHASH_ROOT}/lib/libPerfectHashLLVM.so")
    target_compile_definitions(
        cpp-console-online-cuda-nvrtc
        PRIVATE
            PH_ONLINE_JIT_LLVM_LIBRARY_PATH="${PERFECTHASH_ROOT}/lib/libPerfectHashLLVM.so"
    )
endif()

set(_ph_runtime_dirs)
if(PH_ONLINE_CUDA_NVRTC_RUNTIME_TARGET)
    list(APPEND _ph_runtime_dirs "$<TARGET_FILE_DIR:${PH_ONLINE_CUDA_NVRTC_RUNTIME_TARGET}>")
endif()
if(PH_ONLINE_CUDA_NVRTC_LLVM_TARGET)
    list(APPEND _ph_runtime_dirs "$<TARGET_FILE_DIR:${PH_ONLINE_CUDA_NVRTC_LLVM_TARGET}>")
endif()
if(DEFINED PH_ONLINE_CUDA_NVRTC_FULL_LIBRARY AND
   NOT PH_ONLINE_CUDA_NVRTC_FULL_LIBRARY STREQUAL "")
    get_filename_component(_ph_full_lib_dir "${PH_ONLINE_CUDA_NVRTC_FULL_LIBRARY}" DIRECTORY)
    list(APPEND _ph_runtime_dirs "${_ph_full_lib_dir}")
endif()
list(APPEND _ph_runtime_dirs
    "$<TARGET_FILE_DIR:CUDA::cudart>"
    "$<TARGET_FILE_DIR:CUDA::nvrtc>"
    "$<TARGET_FILE_DIR:CUDA::nvJitLink>"
)
list(REMOVE_DUPLICATES _ph_runtime_dirs)

if(_ph_runtime_dirs AND (APPLE OR UNIX))
    string(JOIN ";" _ph_build_rpath ${_ph_runtime_dirs})
    set_property(
        TARGET cpp-console-online-cuda-nvrtc
        PROPERTY BUILD_RPATH "${_ph_build_rpath}"
    )
endif()
