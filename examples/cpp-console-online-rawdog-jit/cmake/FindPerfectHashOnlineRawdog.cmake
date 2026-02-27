include(FindPackageHandleStandardArgs)

set(_ph_root_hints)
if(PERFECTHASH_ROOT)
    list(APPEND _ph_root_hints "${PERFECTHASH_ROOT}")
endif()
if(DEFINED ENV{PERFECTHASH_ROOT} AND NOT "$ENV{PERFECTHASH_ROOT}" STREQUAL "")
    list(APPEND _ph_root_hints "$ENV{PERFECTHASH_ROOT}")
endif()
if(PerfectHashOnlineRawdog_ROOT)
    list(APPEND _ph_root_hints "${PerfectHashOnlineRawdog_ROOT}")
endif()

list(REMOVE_DUPLICATES _ph_root_hints)

find_path(
    PERFECTHASH_ONLINE_RAWDOG_INCLUDE_DIR
    NAMES PerfectHashOnlineRawdog.h
    HINTS ${_ph_root_hints}
    PATH_SUFFIXES
        include
)

set(_ph_lib_suffixes)
if(WIN32)
    if(PH_ONLINE_RAWDOG_PREFER_SHARED)
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
        lib/static
        build/lib/static
    )
endif()

set(_ph_saved_suffixes "${CMAKE_FIND_LIBRARY_SUFFIXES}")
if(NOT WIN32)
    if(PH_ONLINE_RAWDOG_PREFER_SHARED)
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
    PERFECTHASH_ONLINE_RAWDOG_LIBRARY
    NAMES
        PerfectHashOnlineCore
        PerfectHashOnline
    HINTS ${_ph_root_hints}
    PATH_SUFFIXES ${_ph_lib_suffixes}
)

set(CMAKE_FIND_LIBRARY_SUFFIXES "${_ph_saved_suffixes}")

if(PERFECTHASH_ONLINE_RAWDOG_LIBRARY)
    get_filename_component(
        PERFECTHASH_ONLINE_RAWDOG_LIBRARY_DIR
        "${PERFECTHASH_ONLINE_RAWDOG_LIBRARY}"
        DIRECTORY
    )
endif()

if(WIN32)
    find_file(
        PERFECTHASH_ONLINE_RAWDOG_RUNTIME_DLL
        NAMES
            PerfectHashOnlineCore.dll
            PerfectHashOnline.dll
        HINTS
            ${_ph_root_hints}
            "${PERFECTHASH_ONLINE_RAWDOG_LIBRARY_DIR}"
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
    PerfectHashOnlineRawdog
    REQUIRED_VARS
        PERFECTHASH_ONLINE_RAWDOG_LIBRARY
        PERFECTHASH_ONLINE_RAWDOG_INCLUDE_DIR
)

if(PerfectHashOnlineRawdog_FOUND AND NOT TARGET PerfectHash::OnlineRawdog)
    add_library(PerfectHash::OnlineRawdog UNKNOWN IMPORTED)
    set_target_properties(
        PerfectHash::OnlineRawdog
        PROPERTIES
            IMPORTED_LOCATION "${PERFECTHASH_ONLINE_RAWDOG_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${PERFECTHASH_ONLINE_RAWDOG_INCLUDE_DIR}"
    )

    set(_ph_assume_static FALSE)
    if(NOT WIN32)
        if(PERFECTHASH_ONLINE_RAWDOG_LIBRARY MATCHES "\\.a$")
            set(_ph_assume_static TRUE)
        endif()
    endif()

    if(WIN32)
        if(PERFECTHASH_ONLINE_RAWDOG_LIBRARY MATCHES "([/\\\\]|^)static([/\\\\]|$)")
            set(_ph_assume_static TRUE)
        elseif(NOT PERFECTHASH_ONLINE_RAWDOG_RUNTIME_DLL)
            set(_ph_assume_static TRUE)
        endif()

        if(_ph_assume_static)
            set_property(
                TARGET PerfectHash::OnlineRawdog
                APPEND
                PROPERTY INTERFACE_COMPILE_DEFINITIONS PERFECT_HASH_ONLINE_RAWDOG_STATIC
            )
        endif()
    endif()

    if(_ph_assume_static)
        find_library(
            PERFECTHASH_ASM_LIBRARY
            NAMES PerfectHashAsm
            HINTS ${_ph_root_hints}
            PATH_SUFFIXES ${_ph_lib_suffixes}
        )

        if(PERFECTHASH_ASM_LIBRARY)
            set_property(
                TARGET PerfectHash::OnlineRawdog
                APPEND
                PROPERTY INTERFACE_LINK_LIBRARIES "${PERFECTHASH_ASM_LIBRARY}"
            )
        endif()

        if(WIN32)
            set_property(
                TARGET PerfectHash::OnlineRawdog
                APPEND
                PROPERTY INTERFACE_LINK_LIBRARIES rpcrt4
            )
        elseif(UNIX)
            find_package(Threads REQUIRED)
            set_property(
                TARGET PerfectHash::OnlineRawdog
                APPEND
                PROPERTY INTERFACE_LINK_LIBRARIES Threads::Threads
            )
            if(NOT APPLE)
                set_property(
                    TARGET PerfectHash::OnlineRawdog
                    APPEND
                    PROPERTY INTERFACE_LINK_LIBRARIES rt
                )
            endif()
        endif()
    endif()
endif()

mark_as_advanced(
    PERFECTHASH_ONLINE_RAWDOG_INCLUDE_DIR
    PERFECTHASH_ONLINE_RAWDOG_LIBRARY
    PERFECTHASH_ASM_LIBRARY
    PERFECTHASH_ONLINE_RAWDOG_RUNTIME_DLL
    PERFECTHASH_ONLINE_RAWDOG_LIBRARY_DIR
)
