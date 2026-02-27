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

set(
    PERFECTHASH_ROOT
    ""
    CACHE PATH
    "Path hint to a PerfectHash repo root, build tree, or install prefix."
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(PerfectHashOnlineRawdog REQUIRED)

add_executable(cpp-console-online-rawdog-jit
    src/main.cpp
)

target_link_libraries(cpp-console-online-rawdog-jit
    PRIVATE
        PerfectHash::OnlineRawdog
)

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

if(WIN32 AND DEFINED PERFECTHASH_ONLINE_RAWDOG_RUNTIME_DLL AND
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

