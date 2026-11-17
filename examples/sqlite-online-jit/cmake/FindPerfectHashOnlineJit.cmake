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
    NAMES PerfectHash/PerfectHashOnlineJit.h
    HINTS ${_ph_search_hints}
    PATH_SUFFIXES
        include
)

set(_ph_lib_suffixes)
if(WIN32)
    set(_ph_configs ${CMAKE_BUILD_TYPE} Release RelWithDebInfo Debug MinSizeRel ${CMAKE_CONFIGURATION_TYPES})
    list(REMOVE_DUPLICATES _ph_configs)
    set(_ph_shared_suffixes lib build/lib)
    set(_ph_static_suffixes lib/static build/lib/static)
    foreach(_ph_config IN LISTS _ph_configs)
        list(APPEND _ph_shared_suffixes "lib/${_ph_config}" "build/lib/${_ph_config}")
        list(APPEND _ph_static_suffixes "lib/${_ph_config}/static" "build/lib/${_ph_config}/static")
    endforeach()
    if(PH_ONLINE_JIT_PREFER_SHARED)
        set(_ph_lib_suffixes ${_ph_shared_suffixes} ${_ph_static_suffixes})
    else()
        set(_ph_lib_suffixes ${_ph_static_suffixes} ${_ph_shared_suffixes})
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
        PerfectHashOnlineCore
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

set(_ph_assume_static FALSE)
unset(_ph_required_runtime)
unset(_ph_runtime_error)
if(WIN32)
    # The Windows archives live under lib/static or lib/<config>/static.
    string(TOLOWER "${PERFECTHASH_ONLINE_JIT_LIBRARY_DIR}" _ph_archive_dir)
    if(_ph_archive_dir MATCHES "[/\\\\]static$")
        set(_ph_assume_static TRUE)
        set(PERFECTHASH_ONLINE_JIT_RUNTIME_DLL "")
    elseif(PERFECTHASH_ONLINE_JIT_LIBRARY)
        get_filename_component(_ph_import_name "${PERFECTHASH_ONLINE_JIT_LIBRARY}" NAME_WE)
        set(_ph_dll_name "${_ph_import_name}.dll")
        # Re-evaluate the DLL when the import library, architecture or config changes.
        unset(PERFECTHASH_ONLINE_JIT_RUNTIME_DLL CACHE)
        unset(PERFECTHASH_ONLINE_JIT_RUNTIME_DLL)
        get_filename_component(_ph_lib_config "${PERFECTHASH_ONLINE_JIT_LIBRARY_DIR}" NAME)
        if(_ph_lib_config IN_LIST _ph_configs)
            set(_ph_bin_dir "${PERFECTHASH_ONLINE_JIT_LIBRARY_DIR}/../../bin/${_ph_lib_config}")
        else()
            set(_ph_bin_dir "${PERFECTHASH_ONLINE_JIT_LIBRARY_DIR}/../bin")
        endif()
        find_file(PERFECTHASH_ONLINE_JIT_RUNTIME_DLL
            NAMES "${_ph_dll_name}"
            PATHS "${PERFECTHASH_ONLINE_JIT_LIBRARY_DIR}" "${_ph_bin_dir}"
            NO_DEFAULT_PATH)
        set(_ph_required_runtime PERFECTHASH_ONLINE_JIT_RUNTIME_DLL)
        if(NOT PERFECTHASH_ONLINE_JIT_RUNTIME_DLL)
            set(_ph_runtime_error REASON_FAILURE_MESSAGE
                "The import library ${PERFECTHASH_ONLINE_JIT_LIBRARY} requires ${_ph_dll_name}.")
        endif()
    endif()

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
elseif(PERFECTHASH_ONLINE_JIT_LIBRARY MATCHES "\\.a$")
    set(_ph_assume_static TRUE)
endif()

find_package_handle_standard_args(
    PerfectHashOnlineJit
    REQUIRED_VARS
        PERFECTHASH_ONLINE_JIT_LIBRARY
        PERFECTHASH_ONLINE_JIT_INCLUDE_DIR
        ${_ph_required_runtime}
    ${_ph_runtime_error}
)

if(PerfectHashOnlineJit_FOUND AND NOT TARGET PerfectHash::OnlineJit)
    add_library(PerfectHash::OnlineJit UNKNOWN IMPORTED)
    set_target_properties(
        PerfectHash::OnlineJit
        PROPERTIES
            IMPORTED_LOCATION "${PERFECTHASH_ONLINE_JIT_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${PERFECTHASH_ONLINE_JIT_INCLUDE_DIR}"
    )

    if(WIN32 AND _ph_assume_static)
        set_property(
            TARGET PerfectHash::OnlineJit
            APPEND
            PROPERTY INTERFACE_COMPILE_DEFINITIONS PERFECT_HASH_ONLINE_JIT_STATIC
        )
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
