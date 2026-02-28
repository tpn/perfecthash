cmake_minimum_required(VERSION 3.20)

project(PerfectHashConsumer VERSION 0.1.0 LANGUAGES C)

option(
    PERFECTHASH_CONSUMER_USE_FETCHCONTENT
    "Build PerfectHash from GitHub via FetchContent."
    ON
)
set(PERFECTHASH_GIT_TAG "main" CACHE STRING "PerfectHash git ref used with FetchContent.")
set(
    PERFECTHASH_BUILD_PROFILE
    "online-rawdog"
    CACHE STRING
    "PerfectHash profile: online-rawdog, online-rawdog-llvm, full."
)
set(
    PERFECTHASH_CONSUMER_TARGET
    "PerfectHash::PerfectHashOnlineCore"
    CACHE STRING
    "PerfectHash target linked by this example."
)

if(PERFECTHASH_CONSUMER_USE_FETCHCONTENT)
    include(FetchContent)

    set(PERFECTHASH_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(PERFECTHASH_ENABLE_INSTALL ON CACHE BOOL "" FORCE)
    set(PERFECTHASH_BUILD_PROFILE "${PERFECTHASH_BUILD_PROFILE}" CACHE STRING "" FORCE)

    FetchContent_Declare(
        PerfectHash
        GIT_REPOSITORY https://github.com/tpn/perfecthash.git
        GIT_TAG ${PERFECTHASH_GIT_TAG}
    )
    FetchContent_MakeAvailable(PerfectHash)
else()
    find_package(PerfectHash CONFIG REQUIRED)
endif()

if(NOT TARGET ${PERFECTHASH_CONSUMER_TARGET})
    message(
        FATAL_ERROR
        "Requested target '${PERFECTHASH_CONSUMER_TARGET}' is not available. "
        "Adjust PERFECTHASH_BUILD_PROFILE or PERFECTHASH_CONSUMER_TARGET."
    )
endif()

add_executable(perfecthash-consumer src/main.c)
target_link_libraries(perfecthash-consumer PRIVATE ${PERFECTHASH_CONSUMER_TARGET})
