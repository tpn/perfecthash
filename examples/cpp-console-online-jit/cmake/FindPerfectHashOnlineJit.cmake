include(FindPackageHandleStandardArgs)

# On Windows the public cache variables are overrides. Keep discoveries local to
# this configure so a new SDK, configuration, or linkage preference takes effect.
function(_ph_find finder variable)
    if(NOT WIN32)
        cmake_language(CALL ${finder} ${variable} ${ARGN})
        return()
    endif()
    # Migrate automatic cache entries from earlier versions of this finder.
    get_property(old_help CACHE "${variable}" PROPERTY HELPSTRING)
    if(old_help MATCHES "^Path to a (file|library)[.]$")
        unset(${variable} CACHE)
    endif()
    if(EXISTS "${${variable}}")
        set(${variable}_IS_OVERRIDE TRUE PARENT_SCOPE)
        return()
    endif()
    cmake_language(CALL ${finder} _ph_discovered_path ${ARGN})
    set(${variable} "${_ph_discovered_path}" PARENT_SCOPE)
    unset(_ph_discovered_path CACHE)
endfunction()

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

_ph_find(find_path
    PERFECTHASH_ONLINE_JIT_INCLUDE_DIR
    NAMES
        PerfectHash/PerfectHashOnlineJit.h
        PerfectHashOnlineJit.h
    HINTS ${_ph_search_hints}
    PATH_SUFFIXES
        include
        .
)

set(_ph_lib_suffixes)
if(WIN32)
    set(_ph_configs ${CMAKE_BUILD_TYPE} Release RelWithDebInfo Debug MinSizeRel ${CMAKE_CONFIGURATION_TYPES})
    list(REMOVE_DUPLICATES _ph_configs)
    set(_ph_shared_suffixes lib build/lib)
    set(_ph_static_suffixes lib/static build/lib/static)
    foreach(_ph_config IN LISTS _ph_configs)
        list(APPEND _ph_shared_suffixes "lib/${_ph_config}" "build/lib/${_ph_config}")
        list(APPEND _ph_static_suffixes
            "lib/${_ph_config}/static" "build/lib/${_ph_config}/static"
            "lib/static/${_ph_config}" "build/lib/static/${_ph_config}")
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

_ph_find(find_library
    PERFECTHASH_ONLINE_JIT_LIBRARY
    NAMES
        PerfectHashOnlineCore
        PerfectHashOnline
    HINTS ${_ph_search_hints}
    PATH_SUFFIXES ${_ph_lib_suffixes}
)

if(NOT WIN32)
    find_library(PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY NAMES PerfectHashLLVM
        HINTS ${_ph_search_hints} PATH_SUFFIXES ${_ph_lib_suffixes})
endif()

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
    # Static archives and DLL import libraries share the same .lib suffix.
    get_filename_component(_ph_leaf "${PERFECTHASH_ONLINE_JIT_LIBRARY_DIR}" NAME)
    get_filename_component(_ph_parent "${PERFECTHASH_ONLINE_JIT_LIBRARY_DIR}" DIRECTORY)
    string(TOLOWER "${_ph_leaf}" _ph_leaf_lower)
    string(TOLOWER "${_ph_parent}" _ph_parent_lower)
    list(TRANSFORM _ph_configs TOLOWER OUTPUT_VARIABLE _ph_configs_lower)
    set(_ph_shared_dir "${PERFECTHASH_ONLINE_JIT_LIBRARY_DIR}")
    if(_ph_leaf_lower STREQUAL "static")
        set(_ph_assume_static TRUE)
        set(_ph_shared_dir "${_ph_parent}")
    elseif(_ph_parent_lower MATCHES "[/\\\\]static$" AND
           (_ph_leaf_lower IN_LIST _ph_configs_lower OR _ph_parent_lower MATCHES "[/\\\\]lib[/\\\\]static$"))
        set(_ph_assume_static TRUE)
        get_filename_component(_ph_lib_base "${_ph_parent}" DIRECTORY)
        set(_ph_shared_dir "${_ph_lib_base}/${_ph_leaf}")
    endif()

    function(_ph_find_runtime output import_library)
        get_filename_component(lib_dir "${import_library}" DIRECTORY)
        get_filename_component(config "${lib_dir}" NAME)
        get_filename_component(parent "${lib_dir}" DIRECTORY)
        get_filename_component(parent_name "${parent}" NAME)
        string(TOLOWER "${config}" config_lower)
        string(TOLOWER "${parent_name}" parent_lower)
        if(config_lower IN_LIST _ph_configs_lower OR parent_lower STREQUAL "lib")
            set(bin_dir "${parent}/../bin/${config}")
        else()
            set(bin_dir "${parent}/bin")
        endif()
        get_filename_component(stem "${import_library}" NAME_WE)
        get_filename_component(override_name "${${output}}" NAME)
        string(TOLOWER "${stem}.dll" expected)
        string(TOLOWER "${override_name}" actual)
        get_filename_component(override_dir "${${output}}" DIRECTORY)
        get_filename_component(override_config "${override_dir}" NAME)
        string(TOLOWER "${override_config}" override_config_lower)
        if(NOT actual STREQUAL expected OR
           (config_lower IN_LIST _ph_configs_lower AND override_config_lower IN_LIST _ph_configs_lower
            AND NOT config_lower STREQUAL override_config_lower))
            set(${output} "")
        endif()
        _ph_find(find_file ${output} NAMES "${stem}.dll" PATHS "${lib_dir}" "${bin_dir}" NO_DEFAULT_PATH)
        set(${output} "${${output}}" PARENT_SCOPE)
        set(${output}_IS_OVERRIDE "${${output}_IS_OVERRIDE}" PARENT_SCOPE)
    endfunction()

    if(_ph_assume_static)
        set(PERFECTHASH_ONLINE_JIT_RUNTIME_DLL "")
    elseif(PERFECTHASH_ONLINE_JIT_LIBRARY)
        _ph_find_runtime(PERFECTHASH_ONLINE_JIT_RUNTIME_DLL "${PERFECTHASH_ONLINE_JIT_LIBRARY}")
        set(_ph_required_runtime PERFECTHASH_ONLINE_JIT_RUNTIME_DLL)
        if(NOT PERFECTHASH_ONLINE_JIT_RUNTIME_DLL)
            get_filename_component(_ph_import_name "${PERFECTHASH_ONLINE_JIT_LIBRARY}" NAME_WE)
            set(_ph_runtime_error REASON_FAILURE_MESSAGE
                "The import library ${PERFECTHASH_ONLINE_JIT_LIBRARY} requires ${_ph_import_name}.dll.")
        endif()
    endif()
    if(PERFECTHASH_ONLINE_JIT_LIBRARY)
        _ph_find(find_library PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY
            NAMES PerfectHashLLVM PATHS "${_ph_shared_dir}" NO_DEFAULT_PATH)
    endif()
    if(PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY)
        get_filename_component(PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY_DIR "${PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY}" DIRECTORY)
        _ph_find_runtime(PERFECTHASH_ONLINE_JIT_LLVM_RUNTIME_DLL "${PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY}")
    elseif(PERFECTHASH_ONLINE_JIT_LIBRARY)
        # A runtime-only plugin can still be loaded without its import library.
        _ph_find_runtime(PERFECTHASH_ONLINE_JIT_LLVM_RUNTIME_DLL "${_ph_shared_dir}/PerfectHashLLVM.lib")
    endif()
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
        if(WIN32)
            _ph_find(find_library PERFECTHASH_ASM_LIBRARY NAMES PerfectHashAsm
                PATHS "${_ph_shared_dir}" "${PERFECTHASH_ONLINE_JIT_LIBRARY_DIR}" NO_DEFAULT_PATH)
        else()
            find_library(PERFECTHASH_ASM_LIBRARY NAMES PerfectHashAsm
                HINTS ${_ph_search_hints} PATH_SUFFIXES ${_ph_lib_suffixes})
        endif()

        if(WIN32 AND PERFECTHASH_ASM_LIBRARY)
            add_library(PerfectHash::OnlineJitAsm UNKNOWN IMPORTED)
            set_target_properties(PerfectHash::OnlineJitAsm PROPERTIES IMPORTED_LOCATION "${PERFECTHASH_ASM_LIBRARY}")
            target_link_libraries(PerfectHash::OnlineJit INTERFACE PerfectHash::OnlineJitAsm)
        elseif(PERFECTHASH_ASM_LIBRARY)
            set_property(
                TARGET PerfectHash::OnlineJit
                APPEND
                PROPERTY INTERFACE_LINK_LIBRARIES "${PERFECTHASH_ASM_LIBRARY}"
            )
        endif()

        if(WIN32)
            if(CMAKE_CONFIGURATION_TYPES AND NOT PERFECTHASH_ONLINE_JIT_LIBRARY_IS_OVERRIDE)
                # Imported locations must follow the consuming configuration when
                # an SDK provides more than one variant of the same archive.
                get_filename_component(_ph_shared_parent "${_ph_shared_dir}" DIRECTORY)
                get_filename_component(_ph_shared_parent_leaf "${_ph_shared_parent}" NAME)
                get_filename_component(_ph_shared_leaf "${_ph_shared_dir}" NAME)
                string(TOLOWER "${_ph_shared_parent_leaf}" _ph_shared_parent_lower)
                string(TOLOWER "${_ph_shared_leaf}" _ph_shared_lower)
                if(_ph_shared_lower IN_LIST _ph_configs_lower OR _ph_shared_parent_lower STREQUAL "lib")
                    set(_ph_lib_base "${_ph_shared_parent}")
                else()
                    set(_ph_lib_base "${_ph_shared_dir}")
                endif()
                get_filename_component(_ph_archive "${PERFECTHASH_ONLINE_JIT_LIBRARY}" NAME)
                foreach(_ph_config IN LISTS CMAKE_CONFIGURATION_TYPES)
                    string(TOUPPER "${_ph_config}" _ph_upper)
                    set(_ph_variant_paths "${_ph_lib_base}/${_ph_config}/static" "${_ph_lib_base}/static/${_ph_config}")
                    unset(_ph_variant)
                    _ph_find(find_file _ph_variant NAMES "${_ph_archive}" PATHS ${_ph_variant_paths} NO_DEFAULT_PATH)
                    set(_ph_plugin "${PERFECTHASH_ONLINE_JIT_LLVM_RUNTIME_DLL}")
                    if(_ph_variant)
                        set_property(TARGET PerfectHash::OnlineJit PROPERTY IMPORTED_LOCATION_${_ph_upper} "${_ph_variant}")
                        set(_ph_variant_dir "${_ph_lib_base}/${_ph_config}")
                        if(TARGET PerfectHash::OnlineJitAsm AND NOT PERFECTHASH_ASM_LIBRARY_IS_OVERRIDE)
                            unset(_ph_asm)
                            _ph_find(find_library _ph_asm NAMES PerfectHashAsm PATHS "${_ph_variant_dir}" ${_ph_variant_paths} NO_DEFAULT_PATH)
                            if(_ph_asm)
                                set_property(TARGET PerfectHash::OnlineJitAsm PROPERTY IMPORTED_LOCATION_${_ph_upper} "${_ph_asm}")
                            endif()
                        endif()
                        if(NOT PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY_IS_OVERRIDE AND
                           NOT PERFECTHASH_ONLINE_JIT_LLVM_RUNTIME_DLL_IS_OVERRIDE)
                            unset(_ph_plugin)
                            _ph_find_runtime(_ph_plugin "${_ph_variant_dir}/PerfectHashLLVM.lib")
                        endif()
                    endif()
                    get_filename_component(_ph_plugin_dir "${_ph_plugin}" DIRECTORY)
                    if(_ph_plugin AND EXISTS "${_ph_plugin_dir}/PerfectHash.dll")
                        set(_ph_copy copy_if_different "${_ph_plugin}" "${_ph_plugin_dir}/PerfectHash.dll" .)
                    else()
                        set(_ph_copy true)
                    endif()
                    list(APPEND PERFECTHASH_ONLINE_JIT_STATIC_DEPLOY_ARGS "$<$<CONFIG:${_ph_config}>:${_ph_copy}>")
                endforeach()
            endif()
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
