include_guard(GLOBAL)

get_filename_component(PERFECTHASH_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}" DIRECTORY)
set(PERFECTHASH_SRC_DIR "${PERFECTHASH_ROOT_DIR}/src")

include(GNUInstallDirs)

option(PERFECTHASH_USE_CUDA "Enable CUDA support if available" OFF)
option(PERFECTHASH_ENABLE_PENTER "Enable FunctionHook support" OFF)
option(PERFECTHASH_ENABLE_NATIVE_ARCH "Enable -march=native on supported compilers" ON)
option(PERFECTHASH_ENABLE_INSTALL "Enable install rules" ON)
option(PERFECTHASH_ENABLE_TESTS "Enable tests" ON)
option(PERFECTHASH_ENABLE_LLVM "Enable LLVM support if available" ON)
option(PERFECTHASH_STATIC_LLVM "Link LLVM statically when available" ON)

if(DEFINED USE_CUDA AND NOT DEFINED PERFECTHASH_USE_CUDA)
    set(PERFECTHASH_USE_CUDA "${USE_CUDA}" CACHE BOOL "" FORCE)
endif()
set(USE_CUDA "${PERFECTHASH_USE_CUDA}" CACHE BOOL "" FORCE)

if(DEFINED HookPenter AND NOT DEFINED PERFECTHASH_ENABLE_PENTER)
    set(PERFECTHASH_ENABLE_PENTER "${HookPenter}" CACHE BOOL "" FORCE)
endif()
set(HookPenter "${PERFECTHASH_ENABLE_PENTER}" CACHE BOOL "" FORCE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_CONFIGURATION_TYPES)
    set(PERFECTHASH_CONFIGURATION_TYPES
        Debug
        Release
        RelWithDebInfo
        MinSizeRel
        PGInstrument
        PGOptimize
        PGUpdate
    )
    set(CMAKE_CONFIGURATION_TYPES "${PERFECTHASH_CONFIGURATION_TYPES}" CACHE STRING "" FORCE)
    foreach(config ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER "${config}" config_upper)
        foreach(var CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS)
            if(NOT DEFINED ${var}_${config_upper})
                set(${var}_${config_upper} "" CACHE STRING "" FORCE)
            endif()
        endforeach()
    endforeach()
elseif(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

    foreach(var CMAKE_C_FLAGS_DEBUG CMAKE_CXX_FLAGS_DEBUG)
        if(DEFINED ${var})
            foreach(opt "/RTC1" "/RTCc" "/RTCs" "/RTCu" "/RTC")
                string(REPLACE "${opt}" "" ${var} "${${var}}")
            endforeach()
            string(REGEX REPLACE "[ ]+" " " ${var} "${${var}}")
            set(${var} "${${var}}" CACHE STRING "" FORCE)
        endif()
    endforeach()
endif()

set(PERFECTHASH_IS_WINDOWS FALSE)
set(PERFECTHASH_IS_LINUX FALSE)
set(PERFECTHASH_IS_MAC FALSE)
set(PERFECTHASH_IS_UNIX FALSE)
set(PERFECTHASH_IS_EMSCRIPTEN FALSE)

if(WIN32)
    set(PERFECTHASH_IS_WINDOWS TRUE)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(PERFECTHASH_IS_LINUX TRUE)
    set(PERFECTHASH_IS_UNIX TRUE)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(PERFECTHASH_IS_MAC TRUE)
    set(PERFECTHASH_IS_UNIX TRUE)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    set(PERFECTHASH_IS_EMSCRIPTEN TRUE)
endif()

set(PERFECTHASH_ARCH_X86_64 FALSE)
set(PERFECTHASH_ARCH_X86 FALSE)
set(PERFECTHASH_ARCH_ARM64 FALSE)
set(PERFECTHASH_ARCH_ARM FALSE)
if(CMAKE_SYSTEM_PROCESSOR)
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" PERFECTHASH_SYSTEM_PROCESSOR)
    if(PERFECTHASH_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
        set(PERFECTHASH_ARCH_X86_64 TRUE)
    elseif(PERFECTHASH_SYSTEM_PROCESSOR MATCHES "^(i[3-6]86|x86)$")
        set(PERFECTHASH_ARCH_X86 TRUE)
    elseif(PERFECTHASH_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        set(PERFECTHASH_ARCH_ARM64 TRUE)
    elseif(PERFECTHASH_SYSTEM_PROCESSOR MATCHES "^arm")
        set(PERFECTHASH_ARCH_ARM TRUE)
    endif()
endif()

if(PERFECTHASH_IS_LINUX)
    # Ensure installed binaries can find ../lib without LD_LIBRARY_PATH.
    set(CMAKE_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
elseif(PERFECTHASH_IS_MAC)
    set(CMAKE_INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
endif()

if(PERFECTHASH_IS_LINUX OR PERFECTHASH_IS_MAC)
    execute_process(
        COMMAND getconf PAGE_SIZE
        OUTPUT_VARIABLE PERFECTHASH_PAGE_SIZE
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(PERFECTHASH_PAGE_SIZE MATCHES "^[0-9]+$")
        set(_ph_page_shift 0)
        set(_ph_page_size "${PERFECTHASH_PAGE_SIZE}")
        while(_ph_page_size GREATER 1)
            math(EXPR _ph_page_size "${_ph_page_size} / 2")
            math(EXPR _ph_page_shift "${_ph_page_shift} + 1")
        endwhile()
        set(PERFECTHASH_PAGE_SHIFT "${_ph_page_shift}" CACHE INTERNAL "" FORCE)
    endif()
endif()

if(NOT PERFECTHASH_IS_EMSCRIPTEN AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "Only 64-bit platforms are supported.")
endif()

set(PERFECTHASH_EMSCRIPTEN_COMPILE_OPTIONS "")
set(PERFECTHASH_EMSCRIPTEN_LINK_OPTIONS "")
if(PERFECTHASH_IS_EMSCRIPTEN)
    set(CMAKE_EXECUTABLE_SUFFIX ".html")
    set(PERFECTHASH_EMSCRIPTEN_COMPILE_OPTIONS
        -s ALLOW_MEMORY_GROWTH=1
        -s USE_PTHREADS=1
        -s WASM_BIGINT=1
    )
    set(PERFECTHASH_EMSCRIPTEN_LINK_OPTIONS
        ${PERFECTHASH_EMSCRIPTEN_COMPILE_OPTIONS}
        -s ASSERTIONS=1
    )
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")

if(CMAKE_CONFIGURATION_TYPES)
    foreach(config ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER "${config}" config_upper)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config_upper}
            "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${config}")
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config_upper}
            "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${config}")
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config_upper}
            "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${config}")
    endforeach()
endif()

set(PERFECTHASH_HAS_CUDA FALSE)
if(PERFECTHASH_USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        find_package(CUDAToolkit)
        if(CUDAToolkit_FOUND)
            set(PERFECTHASH_HAS_CUDA TRUE)
        else()
            message(WARNING "CUDA compiler found, but CUDAToolkit is missing. CUDA support is disabled.")
        endif()
    else()
        message(WARNING "CUDA requested, but no CUDA compiler was found. CUDA support is disabled.")
    endif()
endif()

set(PERFECTHASH_HAS_LLVM FALSE)
if(PERFECTHASH_ENABLE_LLVM)
    if(PERFECTHASH_STATIC_LLVM)
        set(LLVM_LINK_LLVM_DYLIB OFF CACHE BOOL "" FORCE)
    else()
        set(LLVM_LINK_LLVM_DYLIB ON CACHE BOOL "" FORCE)
    endif()
    find_package(LLVM CONFIG)
    if(LLVM_FOUND)
        if(LLVM_PACKAGE_VERSION VERSION_LESS 15)
            message(WARNING "LLVM ${LLVM_PACKAGE_VERSION} found, but 15+ is required; online JIT support is disabled.")
        else()
            message(STATUS "LLVM found: ${LLVM_PACKAGE_VERSION}")
            set(PERFECTHASH_HAS_LLVM TRUE)
        endif()
    else()
        message(WARNING "LLVM not found; online JIT support is disabled.")
    endif()
endif()

find_package(Threads REQUIRED)
