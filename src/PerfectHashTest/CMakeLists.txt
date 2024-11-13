cmake_minimum_required(VERSION 3.10)

# Set the project name
project(PerfectHashTest)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

enable_testing()

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Add the executable
add_executable(PerfectHashTest main.cpp)

# Link libraries
target_link_libraries(PerfectHashTest PerfectHash)