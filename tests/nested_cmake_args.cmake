set(nested_cmake_compiler_args "")
if(DEFINED TEST_CMAKE_TOOLCHAIN_FILE AND NOT TEST_CMAKE_TOOLCHAIN_FILE STREQUAL "")
  list(APPEND nested_cmake_compiler_args
       "-DCMAKE_TOOLCHAIN_FILE=${TEST_CMAKE_TOOLCHAIN_FILE}")
endif()
if(DEFINED TEST_CMAKE_C_COMPILER AND NOT TEST_CMAKE_C_COMPILER STREQUAL "")
  list(APPEND nested_cmake_compiler_args
       "-DCMAKE_C_COMPILER=${TEST_CMAKE_C_COMPILER}")
endif()
if(DEFINED TEST_CMAKE_CXX_COMPILER AND NOT TEST_CMAKE_CXX_COMPILER STREQUAL "")
  list(APPEND nested_cmake_compiler_args
       "-DCMAKE_CXX_COMPILER=${TEST_CMAKE_CXX_COMPILER}")
endif()

#
# Keep the inherited generator make program separate from the Unix Makefiles
# fallback make program.  Some tests run one nested configure with the parent
# generator and a different nested configure with Unix Makefiles; these argument
# lists must not be concatenated into one CMake invocation.
#
set(nested_cmake_unix_make_args "")
if(DEFINED TEST_CMAKE_UNIX_MAKE_PROGRAM AND
   NOT TEST_CMAKE_UNIX_MAKE_PROGRAM STREQUAL "")
  list(APPEND nested_cmake_unix_make_args
       "-DCMAKE_MAKE_PROGRAM=${TEST_CMAKE_UNIX_MAKE_PROGRAM}")
endif()

set(nested_cmake_generator_args "")
if(DEFINED TEST_CMAKE_GENERATOR AND NOT TEST_CMAKE_GENERATOR STREQUAL "")
  list(APPEND nested_cmake_generator_args -G "${TEST_CMAKE_GENERATOR}")
endif()
if(DEFINED TEST_CMAKE_GENERATOR_PLATFORM AND
   NOT TEST_CMAKE_GENERATOR_PLATFORM STREQUAL "")
  list(APPEND nested_cmake_generator_args
       -A "${TEST_CMAKE_GENERATOR_PLATFORM}")
endif()
if(DEFINED TEST_CMAKE_GENERATOR_TOOLSET AND
   NOT TEST_CMAKE_GENERATOR_TOOLSET STREQUAL "")
  list(APPEND nested_cmake_generator_args
       -T "${TEST_CMAKE_GENERATOR_TOOLSET}")
endif()
if(DEFINED TEST_CMAKE_MAKE_PROGRAM AND NOT TEST_CMAKE_MAKE_PROGRAM STREQUAL "")
  list(APPEND nested_cmake_generator_args
       "-DCMAKE_MAKE_PROGRAM=${TEST_CMAKE_MAKE_PROGRAM}")
endif()
