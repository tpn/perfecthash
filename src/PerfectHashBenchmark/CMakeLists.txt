cmake_minimum_required(VERSION 3.10)

project(PerfectHashBenchmark)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

enable_testing()

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

find_package(benchmark REQUIRED)
include_directories(${BENCHMARK_INCLUDE_DIRS})

# Add Google Benchmark
# include(FetchContent)
# FetchContent_Declare(
# benchmark
# GIT_REPOSITORY https://github.com/google/benchmark.git
# GIT_TAG v1.5.0
# )
# FetchContent_MakeAvailable(benchmark)

# Add the executable
add_executable(PerfectHashBenchmark main.cpp)

target_link_libraries(PerfectHashBenchmark PRIVATE PerfectHash)
target_link_libraries(PerfectHashBenchmark PUBLIC benchmark::benchmark)
