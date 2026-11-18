include(FindPackageHandleStandardArgs)

# On Windows the public cache variables are overrides. Keep discoveries local to
# this configure so a new SDK, configuration, or linkage preference takes effect.
function(_ph_find finder variable)
    if(NOT WIN32)
        cmake_language(CALL ${finder} ${variable} ${ARGN})
        return()
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

if(WIN32 AND PERFECTHASH_ONLINE_JIT_LIBRARY AND
   (NOT EXISTS "${PERFECTHASH_ONLINE_JIT_LIBRARY}" OR IS_DIRECTORY "${PERFECTHASH_ONLINE_JIT_LIBRARY}"))
    set(_ph_invalid_library "${PERFECTHASH_ONLINE_JIT_LIBRARY}")
    set(PERFECTHASH_ONLINE_JIT_LIBRARY "PERFECTHASH_ONLINE_JIT_LIBRARY-NOTFOUND")
else()
    _ph_find(find_library
        PERFECTHASH_ONLINE_JIT_LIBRARY
        NAMES
            PerfectHashOnlineCore
            PerfectHashOnline
        HINTS ${_ph_search_hints}
        PATH_SUFFIXES ${_ph_lib_suffixes}
    )
endif()

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
if(_ph_invalid_library)
    set(_ph_runtime_error REASON_FAILURE_MESSAGE "The requested library ${_ph_invalid_library} is not a file.")
endif()
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

    function(_ph_find_runtime output import_library companion)
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
        set(directories "${lib_dir}" "${bin_dir}" ${ARGN})
        if(EXISTS "${${output}}" AND NOT IS_DIRECTORY "${${output}}" AND actual STREQUAL expected AND
           NOT (config_lower IN_LIST _ph_configs_lower AND override_config_lower IN_LIST _ph_configs_lower
                AND NOT config_lower STREQUAL override_config_lower))
            set(directories "${override_dir}")
        endif()
        set(found "")
        set(missing "")
        foreach(directory IN LISTS directories)
            set(candidate "${directory}/${stem}.dll")
            if(NOT EXISTS "${candidate}" OR IS_DIRECTORY "${candidate}")
                continue()
            endif()
            if(companion AND (NOT EXISTS "${directory}/${companion}" OR IS_DIRECTORY "${directory}/${companion}"))
                if(NOT missing)
                    get_filename_component(missing "${directory}/${companion}" ABSOLUTE)
                endif()
                continue()
            endif()
            get_filename_component(found "${candidate}" ABSOLUTE)
            set(missing "")
            break()
        endforeach()
        set(${output} "${found}" PARENT_SCOPE)
        set(${output}_MISSING_COMPANION "${missing}" PARENT_SCOPE)
    endfunction()

    # One bundle contains the link dependencies and runtime files for the same library.
    function(_ph_windows_bundle output library shared_dir)
        set(files)
        if(_ph_assume_static)
            get_filename_component(archive_dir "${library}" DIRECTORY)
            _ph_find(find_library PERFECTHASH_ASM_LIBRARY NAMES PerfectHashAsm
                PATHS "${shared_dir}" "${archive_dir}" "${_ph_lib_base}" NO_DEFAULT_PATH)
            set(${output}_asm "${PERFECTHASH_ASM_LIBRARY}" PARENT_SCOPE)
        else()
            _ph_find_runtime(PERFECTHASH_ONLINE_JIT_RUNTIME_DLL "${library}" "")
            if(PERFECTHASH_ONLINE_JIT_RUNTIME_DLL)
                list(APPEND files "${PERFECTHASH_ONLINE_JIT_RUNTIME_DLL}")
            endif()
            set(${output}_core "${PERFECTHASH_ONLINE_JIT_RUNTIME_DLL}" PARENT_SCOPE)
        endif()

        get_filename_component(bundle_config "${shared_dir}" NAME)
        get_filename_component(override_dir "${PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY}" DIRECTORY)
        get_filename_component(override_config "${override_dir}" NAME)
        string(TOLOWER "${bundle_config}" bundle_config)
        string(TOLOWER "${override_config}" override_config)
        if(bundle_config IN_LIST _ph_configs_lower AND override_config IN_LIST _ph_configs_lower
           AND NOT bundle_config STREQUAL override_config)
            set(PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY "")
        endif()
        _ph_find(find_library PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY NAMES PerfectHashLLVM
            PATHS "${shared_dir}" "${_ph_lib_base}" NO_DEFAULT_PATH)
        if(PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY_IS_OVERRIDE)
            set(plugin_library "${PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY}")
        else()
            # Prefer this configuration's runtime, even if it has no import library.
            set(plugin_library "${shared_dir}/PerfectHashLLVM.lib")
        endif()
        get_filename_component(flat_bin "${_ph_lib_base}/../bin" ABSOLUTE)
        _ph_find_runtime(PERFECTHASH_ONLINE_JIT_LLVM_RUNTIME_DLL "${plugin_library}" PerfectHash.dll
            "${_ph_lib_base}" "${flat_bin}")
        if(PERFECTHASH_ONLINE_JIT_LLVM_RUNTIME_DLL)
            get_filename_component(plugin_dir "${PERFECTHASH_ONLINE_JIT_LLVM_RUNTIME_DLL}" DIRECTORY)
            list(APPEND files "${PERFECTHASH_ONLINE_JIT_LLVM_RUNTIME_DLL}" "${plugin_dir}/PerfectHash.dll")
        elseif(PERFECTHASH_ONLINE_JIT_LLVM_RUNTIME_DLL_MISSING_COMPANION)
            set(${output}_missing "${PERFECTHASH_ONLINE_JIT_LLVM_RUNTIME_DLL_MISSING_COMPANION}" PARENT_SCOPE)
        endif()
        set(${output}_library "${library}" PARENT_SCOPE)
        set(${output}_llvm "${PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY}" PARENT_SCOPE)
        set(${output}_plugin "${PERFECTHASH_ONLINE_JIT_LLVM_RUNTIME_DLL}" PARENT_SCOPE)
        set(${output}_files "${files}" PARENT_SCOPE)
    endfunction()

    if(PERFECTHASH_ONLINE_JIT_LIBRARY)
        get_filename_component(_ph_shared_leaf "${_ph_shared_dir}" NAME)
        get_filename_component(_ph_shared_parent "${_ph_shared_dir}" DIRECTORY)
        get_filename_component(_ph_shared_parent_leaf "${_ph_shared_parent}" NAME)
        string(TOLOWER "${_ph_shared_leaf}" _ph_shared_lower)
        string(TOLOWER "${_ph_shared_parent_leaf}" _ph_shared_parent_lower)
        if(_ph_shared_lower IN_LIST _ph_configs_lower OR _ph_shared_parent_lower STREQUAL "lib")
            set(_ph_lib_base "${_ph_shared_parent}")
        else()
            set(_ph_lib_base "${_ph_shared_dir}")
        endif()
        set(_ph_all_runtimes TRUE)
        set(_ph_required_runtime _ph_all_runtimes)
        _ph_windows_bundle(_ph_default "${PERFECTHASH_ONLINE_JIT_LIBRARY}" "${_ph_shared_dir}")
        set(_ph_fallback _ph_default)
        get_filename_component(_ph_basename "${PERFECTHASH_ONLINE_JIT_LIBRARY}" NAME)
        set(_ph_candidates)
        if(CMAKE_CONFIGURATION_TYPES)
            set(_ph_candidates ${_ph_configs})
        endif()
        foreach(_ph_config IN LISTS _ph_candidates)
            set(_ph_bundle _ph_default)
            if(NOT PERFECTHASH_ONLINE_JIT_LIBRARY_IS_OVERRIDE)
                set(_ph_config_dir "${_ph_lib_base}/${_ph_config}")
                if(_ph_assume_static)
                    set(_ph_paths "${_ph_config_dir}/static" "${_ph_lib_base}/static/${_ph_config}")
                else()
                    set(_ph_paths "${_ph_config_dir}")
                endif()
                unset(_ph_variant)
                _ph_find(find_file _ph_variant NAMES "${_ph_basename}" PATHS ${_ph_paths} NO_DEFAULT_PATH)
                if(_ph_variant AND NOT _ph_variant STREQUAL PERFECTHASH_ONLINE_JIT_LIBRARY)
                    set(_ph_bundle "_ph_${_ph_config}")
                    _ph_windows_bundle(${_ph_bundle} "${_ph_variant}" "${_ph_config_dir}")
                endif()
            endif()
            if(NOT ${_ph_fallback}_core AND ${_ph_bundle}_core)
                set(_ph_fallback "${_ph_bundle}")
            endif()
            if(NOT _ph_default_asm AND ${_ph_bundle}_asm)
                set(_ph_default_asm "${${_ph_bundle}_asm}")
            endif()
            set(_ph_bundle_${_ph_config} "${_ph_bundle}")
        endforeach()
        if(NOT _ph_assume_static AND NOT ${_ph_fallback}_core)
            get_filename_component(stem "${PERFECTHASH_ONLINE_JIT_LIBRARY}" NAME_WE)
            set(_ph_all_runtimes FALSE)
            set(_ph_runtime_error REASON_FAILURE_MESSAGE
                "The import library ${PERFECTHASH_ONLINE_JIT_LIBRARY} requires ${stem}.dll.")
        endif()
        # Retain complete link/runtime pairs, even when one installed configuration
        # is incomplete. An explicitly selected library is the only candidate.
        set(_ph_missing_dependencies ${${_ph_fallback}_missing})
        foreach(_ph_config IN LISTS CMAKE_CONFIGURATION_TYPES)
            set(_ph_bundle "${_ph_bundle_${_ph_config}}")
            if(NOT _ph_assume_static AND NOT ${_ph_bundle}_core)
                set(_ph_bundle "${_ph_fallback}")
                set(_ph_bundle_${_ph_config} "${_ph_bundle}")
            endif()
            if(_ph_assume_static AND NOT ${_ph_bundle}_asm)
                set(${_ph_bundle}_asm "${_ph_default_asm}")
            endif()
            list(APPEND _ph_missing_dependencies ${${_ph_bundle}_missing})
            list(APPEND PERFECTHASH_ONLINE_JIT_RUNTIME_FILES "$<$<CONFIG:${_ph_config}>:${${_ph_bundle}_files}>")
        endforeach()
        if(NOT CMAKE_CONFIGURATION_TYPES)
            set(PERFECTHASH_ONLINE_JIT_RUNTIME_FILES "${_ph_default_files}")
        endif()
        list(REMOVE_DUPLICATES _ph_missing_dependencies)
        foreach(_ph_missing IN LISTS _ph_missing_dependencies)
            message(WARNING "Skipping the optional LLVM plugin: ${_ph_missing} is missing.")
        endforeach()
        set(PERFECTHASH_ONLINE_JIT_LIBRARY "${${_ph_fallback}_library}")
        get_filename_component(PERFECTHASH_ONLINE_JIT_LIBRARY_DIR "${PERFECTHASH_ONLINE_JIT_LIBRARY}" DIRECTORY)
        set(PERFECTHASH_ONLINE_JIT_RUNTIME_DLL "${${_ph_fallback}_core}")
        set(PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY "${${_ph_fallback}_llvm}")
        set(PERFECTHASH_ONLINE_JIT_LLVM_RUNTIME_DLL "${${_ph_fallback}_plugin}")
        get_filename_component(PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY_DIR "${PERFECTHASH_ONLINE_JIT_LLVM_LIBRARY}" DIRECTORY)
        set(PERFECTHASH_ASM_LIBRARY "${_ph_default_asm}")
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

    if(WIN32)
        foreach(_ph_config IN LISTS CMAKE_CONFIGURATION_TYPES)
            set(_ph_bundle "${_ph_bundle_${_ph_config}}")
            string(TOUPPER "${_ph_config}" _ph_upper)
            set_property(TARGET PerfectHash::OnlineJit PROPERTY
                IMPORTED_LOCATION_${_ph_upper} "${${_ph_bundle}_library}")
            if(${_ph_bundle}_asm)
                target_link_libraries(PerfectHash::OnlineJit INTERFACE "$<$<CONFIG:${_ph_config}>:${${_ph_bundle}_asm}>")
            endif()
        endforeach()
        if(NOT CMAKE_CONFIGURATION_TYPES AND PERFECTHASH_ASM_LIBRARY)
            target_link_libraries(PerfectHash::OnlineJit INTERFACE "${PERFECTHASH_ASM_LIBRARY}")
        endif()
        if(_ph_assume_static)
            target_compile_definitions(PerfectHash::OnlineJit INTERFACE PERFECT_HASH_ONLINE_JIT_STATIC)
            target_link_libraries(PerfectHash::OnlineJit INTERFACE rpcrt4)
        endif()
    elseif(_ph_assume_static)
        find_library(PERFECTHASH_ASM_LIBRARY NAMES PerfectHashAsm
            HINTS ${_ph_search_hints} PATH_SUFFIXES ${_ph_lib_suffixes})
        if(PERFECTHASH_ASM_LIBRARY)
            target_link_libraries(PerfectHash::OnlineJit INTERFACE "${PERFECTHASH_ASM_LIBRARY}")
        endif()
        if(UNIX)
            find_package(Threads REQUIRED)
            target_link_libraries(PerfectHash::OnlineJit INTERFACE Threads::Threads ${CMAKE_DL_LIBS})
            if(NOT APPLE)
                target_link_libraries(PerfectHash::OnlineJit INTERFACE rt)
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
