cmake_minimum_required(VERSION 3.20)

project(cpp-console-cuco-static-multiset-bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(
    CUCO_ROOT
    ""
    CACHE PATH
    "Path to a local cuCollections checkout."
)

find_package(CUDAToolkit REQUIRED)

find_path(
    CUCO_INCLUDE_DIR
    NAMES cuco/static_multiset.cuh
    HINTS ${CUCO_ROOT}
    PATH_SUFFIXES include
    REQUIRED
)

add_executable(cpp-console-cuco-static-multiset-bench
    src/main.cu
)

target_include_directories(
    cpp-console-cuco-static-multiset-bench
    PRIVATE
        "${CUCO_INCLUDE_DIR}"
)

target_link_libraries(
    cpp-console-cuco-static-multiset-bench
    PRIVATE
        CUDA::cudart
)

target_compile_options(
    cpp-console-cuco-static-multiset-bench
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
)

if(APPLE OR UNIX)
    set_property(
        TARGET cpp-console-cuco-static-multiset-bench
        PROPERTY BUILD_RPATH "$<TARGET_FILE_DIR:CUDA::cudart>"
    )
endif()
