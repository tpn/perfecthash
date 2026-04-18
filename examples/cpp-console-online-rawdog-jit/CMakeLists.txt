cmake_minimum_required(VERSION 3.20)

project(cpp-console-online-rawdog-jit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(
    PH_ONLINE_RAWDOG_PREFER_SHARED
    "Prefer shared PerfectHash runtime over static."
    ON
)
option(
    PH_ONLINE_RAWDOG_USE_FETCHCONTENT
    "Fetch PerfectHash from GitHub via FetchContent."
    ON
)

set(
    PERFECTHASH_GIT_REPOSITORY
    "https://github.com/tpn/perfecthash.git"
    CACHE STRING
    "PerfectHash git repository used with FetchContent."
)
set(
    PERFECTHASH_GIT_TAG
    "main"
    CACHE STRING
    "PerfectHash git ref used with FetchContent."
)
set(
    PERFECTHASH_BUILD_PROFILE
    "online-rawdog-jit"
    CACHE STRING
    "PerfectHash profile used with FetchContent."
)
set(
    PERFECTHASH_ROOT
    ""
    CACHE PATH
    "Path hint to a PerfectHash repo root, build tree, or install prefix."
)

set(PH_ONLINE_RAWDOG_TARGET "")
set(PH_ONLINE_RAWDOG_RUNTIME_TARGET "")

if(PERFECTHASH_ROOT AND NOT PERFECTHASH_ROOT STREQUAL "")
    set(PH_ONLINE_RAWDOG_USE_FETCHCONTENT OFF CACHE BOOL "" FORCE)
endif()

if(PH_ONLINE_RAWDOG_USE_FETCHCONTENT)
    include(FetchContent)

    set(PERFECTHASH_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(PERFECTHASH_ENABLE_INSTALL ON CACHE BOOL "" FORCE)
    set(PERFECTHASH_BUILD_PROFILE "${PERFECTHASH_BUILD_PROFILE}" CACHE STRING "" FORCE)

    FetchContent_Declare(
        PerfectHash
        GIT_REPOSITORY ${PERFECTHASH_GIT_REPOSITORY}
        GIT_TAG ${PERFECTHASH_GIT_TAG}
    )
    FetchContent_MakeAvailable(PerfectHash)

    if(TARGET PerfectHash::PerfectHashOnlineCore)
        set(PH_ONLINE_RAWDOG_TARGET PerfectHash::PerfectHashOnlineCore)
        set(PH_ONLINE_RAWDOG_RUNTIME_TARGET PerfectHash::PerfectHashOnlineCore)
    elseif(TARGET PerfectHash::PerfectHashOnline)
        set(PH_ONLINE_RAWDOG_TARGET PerfectHash::PerfectHashOnline)
        set(PH_ONLINE_RAWDOG_RUNTIME_TARGET PerfectHash::PerfectHashOnline)
    else()
        message(
            FATAL_ERROR
            "FetchContent build did not expose PerfectHash online targets. "
            "Verify PERFECTHASH_BUILD_PROFILE='${PERFECTHASH_BUILD_PROFILE}'."
        )
    endif()
else()
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    find_package(PerfectHashOnlineRawdog REQUIRED)
    set(PH_ONLINE_RAWDOG_TARGET PerfectHash::OnlineRawdog)
endif()

add_executable(cpp-console-online-rawdog-jit
    src/main.cpp
)

target_link_libraries(cpp-console-online-rawdog-jit
    PRIVATE
        ${PH_ONLINE_RAWDOG_TARGET}
)

if(PH_ONLINE_RAWDOG_RUNTIME_TARGET)
    if(APPLE OR (UNIX AND NOT WIN32))
        set_property(
            TARGET cpp-console-online-rawdog-jit
            PROPERTY BUILD_RPATH "$<TARGET_FILE_DIR:${PH_ONLINE_RAWDOG_RUNTIME_TARGET}>"
        )
    endif()
endif()

if(DEFINED PERFECTHASH_ONLINE_RAWDOG_LIBRARY_DIR AND
   NOT PERFECTHASH_ONLINE_RAWDOG_LIBRARY_DIR STREQUAL "")
    if(APPLE)
        set_property(
            TARGET cpp-console-online-rawdog-jit
            PROPERTY BUILD_RPATH "${PERFECTHASH_ONLINE_RAWDOG_LIBRARY_DIR}"
        )
    elseif(UNIX AND NOT WIN32)
        set_property(
            TARGET cpp-console-online-rawdog-jit
            PROPERTY BUILD_RPATH "${PERFECTHASH_ONLINE_RAWDOG_LIBRARY_DIR}"
        )
    endif()
endif()

if(WIN32)
    if(PH_ONLINE_RAWDOG_RUNTIME_TARGET)
        add_custom_command(
            TARGET cpp-console-online-rawdog-jit
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:${PH_ONLINE_RAWDOG_RUNTIME_TARGET}>"
                    "$<TARGET_FILE_DIR:cpp-console-online-rawdog-jit>"
            COMMENT "Copying FetchContent PerfectHash runtime DLL."
        )
    endif()

    if(DEFINED PERFECTHASH_ONLINE_RAWDOG_RUNTIME_DLL AND
       EXISTS "${PERFECTHASH_ONLINE_RAWDOG_RUNTIME_DLL}")
        add_custom_command(
            TARGET cpp-console-online-rawdog-jit
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${PERFECTHASH_ONLINE_RAWDOG_RUNTIME_DLL}"
                    "$<TARGET_FILE_DIR:cpp-console-online-rawdog-jit>"
            COMMENT "Copying PerfectHash runtime DLL next to example executable."
        )
    endif()
endif()
