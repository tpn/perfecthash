include(FindPackageHandleStandardArgs)

set(_ph_root_hints)
if(PERFECTHASH_ROOT)
    list(APPEND _ph_root_hints "${PERFECTHASH_ROOT}")
endif()
if(DEFINED ENV{PERFECTHASH_ROOT} AND NOT "$ENV{PERFECTHASH_ROOT}" STREQUAL "")
    list(APPEND _ph_root_hints "$ENV{PERFECTHASH_ROOT}")
endif()
if(PerfectHashOnlineJit_ROOT)
    list(APPEND _ph_root_hints "${PerfectHashOnlineJit_ROOT}")
endif()

list(REMOVE_DUPLICATES _ph_root_hints)

set(_ph_search_hints ${_ph_root_hints})
foreach(_ph_hint IN LISTS _ph_root_hints)
    if(IS_DIRECTORY "${_ph_hint}")
        file(GLOB _ph_build_hints LIST_DIRECTORIES TRUE "${_ph_hint}/build*")
        foreach(_ph_build_hint IN LISTS _ph_build_hints)
            if(IS_DIRECTORY "${_ph_build_hint}")
                list(APPEND _ph_search_hints "${_ph_build_hint}")
            endif()
        endforeach()
    endif()
endforeach()

list(REMOVE_DUPLICATES _ph_search_hints)

find_path(
    PERFECTHASH_ONLINE_JIT_INCLUDE_DIR
    NAMES PerfectHashOnlineJit.h
    HINTS ${_ph_search_hints}
    PATH_SUFFIXES
        include
)

set(_ph_lib_suffixes)
if(WIN32)
    if(PH_ONLINE_JIT_PREFER_SHARED)
        list(APPEND _ph_lib_suffixes
            lib
            build/lib
            lib/Release
            build/lib/Release
            lib/RelWithDebInfo
            build/lib/RelWithDebInfo
            lib/static
            build/lib/static
            lib/Debug
            build/lib/Debug
        )
    else()
        list(APPEND _ph_lib_suffixes
            lib/static
            build/lib/static
            lib
            build/lib
            lib/Release
            build/lib/Release
            lib/RelWithDebInfo
            build/lib/RelWithDebInfo
            lib/Debug
            build/lib/Debug
        )
    endif()
else()
    list(APPEND _ph_lib_suffixes
        lib
        build/lib
        lib/Release
        build/lib/Release
        lib/RelWithDebInfo
        build/lib/RelWithDebInfo
        lib/Debug
        build/lib/Debug
        lib/static
        build/lib/static
        lib/static/Release
        build/lib/static/Release
        lib/static/RelWithDebInfo
        build/lib/static/RelWithDebInfo
        lib/static/Debug
        build/lib/static/Debug
    )
endif()

set(_ph_saved_suffixes "${CMAKE_FIND_LIBRARY_SUFFIXES}")
if(NOT WIN32)
    if(PH_ONLINE_JIT_PREFER_SHARED)
        if(APPLE)
            set(CMAKE_FIND_LIBRARY_SUFFIXES .dylib .so .a)
        else()
            set(CMAKE_FIND_LIBRARY_SUFFIXES .so .a)
        endif()
    else()
        if(APPLE)
            set(CMAKE_FIND_LIBRARY_SUFFIXES .a .dylib .so)
        else()
            set(CMAKE_FIND_LIBRARY_SUFFIXES .a .so)
        endif()
    endif()
endif()

find_library(
    PERFECTHASH_ONLINE_JIT_LIBRARY
    NAMES
        PerfectHashOnline
    HINTS ${_ph_search_hints}
    PATH_SUFFIXES ${_ph_lib_suffixes}
)

find_library(
    PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY
    NAMES
        PerfectHashLLVM
    HINTS ${_ph_search_hints}
    PATH_SUFFIXES ${_ph_lib_suffixes}
)

set(CMAKE_FIND_LIBRARY_SUFFIXES "${_ph_saved_suffixes}")

if(PERFECTHASH_ONLINE_JIT_LIBRARY)
    get_filename_component(
        PERFECTHASH_ONLINE_JIT_LIBRARY_DIR
        "${PERFECTHASH_ONLINE_JIT_LIBRARY}"
        DIRECTORY
    )
endif()

if(PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY)
    get_filename_component(
        PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY_DIR
        "${PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY}"
        DIRECTORY
    )
endif()

if(WIN32)
    find_file(
        PERFECTHASH_ONLINE_JIT_RUNTIME_DLL
        NAMES
            PerfectHashOnline.dll
        HINTS
            ${_ph_search_hints}
            "${PERFECTHASH_ONLINE_JIT_LIBRARY_DIR}"
        PATH_SUFFIXES
            bin
            build/bin
            bin/Release
            build/bin/Release
            bin/RelWithDebInfo
            build/bin/RelWithDebInfo
            bin/Debug
            build/bin/Debug
    )

    find_file(
        PERFECTHASH_ONLINE_JIT_LLVM_RUNTIME_DLL
        NAMES
            PerfectHashLLVM.dll
        HINTS
            ${_ph_search_hints}
            "${PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY_DIR}"
        PATH_SUFFIXES
            bin
            build/bin
            bin/Release
            build/bin/Release
            bin/RelWithDebInfo
            build/bin/RelWithDebInfo
            bin/Debug
            build/bin/Debug
    )
endif()

find_package_handle_standard_args(
    PerfectHashOnlineJit
    REQUIRED_VARS
        PERFECTHASH_ONLINE_JIT_LIBRARY
        PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY
        PERFECTHASH_ONLINE_JIT_INCLUDE_DIR
)

if(PerfectHashOnlineJit_FOUND AND NOT TARGET PerfectHash::OnlineJit)
    add_library(PerfectHash::OnlineJit UNKNOWN IMPORTED)
    set_target_properties(
        PerfectHash::OnlineJit
        PROPERTIES
            IMPORTED_LOCATION "${PERFECTHASH_ONLINE_JIT_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${PERFECTHASH_ONLINE_JIT_INCLUDE_DIR}"
    )

    set(_ph_assume_static FALSE)
    if(NOT WIN32)
        if(PERFECTHASH_ONLINE_JIT_LIBRARY MATCHES "\\.a$")
            set(_ph_assume_static TRUE)
        endif()
    endif()

    if(WIN32)
        if(PERFECTHASH_ONLINE_JIT_LIBRARY MATCHES "([/\\\\]|^)static([/\\\\]|$)")
            set(_ph_assume_static TRUE)
        elseif(NOT PERFECTHASH_ONLINE_JIT_RUNTIME_DLL)
            set(_ph_assume_static TRUE)
        endif()

        if(_ph_assume_static)
            set_property(
                TARGET PerfectHash::OnlineJit
                APPEND
                PROPERTY INTERFACE_COMPILE_DEFINITIONS PERFECT_HASH_ONLINE_JIT_STATIC
            )
        endif()
    endif()

    if(_ph_assume_static)
        find_library(
            PERFECTHASH_ASM_LIBRARY
            NAMES PerfectHashAsm
            HINTS ${_ph_search_hints}
            PATH_SUFFIXES ${_ph_lib_suffixes}
        )

        if(PERFECTHASH_ASM_LIBRARY)
            set_property(
                TARGET PerfectHash::OnlineJit
                APPEND
                PROPERTY INTERFACE_LINK_LIBRARIES "${PERFECTHASH_ASM_LIBRARY}"
            )
        endif()

        if(WIN32)
            set_property(
                TARGET PerfectHash::OnlineJit
                APPEND
                PROPERTY INTERFACE_LINK_LIBRARIES rpcrt4
            )
        elseif(UNIX)
            find_package(Threads REQUIRED)
            set_property(
                TARGET PerfectHash::OnlineJit
                APPEND
                PROPERTY INTERFACE_LINK_LIBRARIES Threads::Threads
            )
            if(NOT APPLE)
                set_property(
                    TARGET PerfectHash::OnlineJit
                    APPEND
                    PROPERTY INTERFACE_LINK_LIBRARIES rt
                )
            endif()
        endif()
    endif()
endif()

mark_as_advanced(
    PERFECTHASH_ONLINE_JIT_INCLUDE_DIR
    PERFECTHASH_ONLINE_JIT_LIBRARY
    PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY
    PERFECTHASH_ASM_LIBRARY
    PERFECTHASH_ONLINE_JIT_LIBRARY_DIR
    PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY_DIR
    PERFECTHASH_ONLINE_JIT_RUNTIME_DLL
    PERFECTHASH_ONLINE_JIT_LLVM_RUNTIME_DLL
)
