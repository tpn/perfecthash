include_guard(GLOBAL)

function(perfecthash_normalize_version INPUT_VERSION OUT_VERSION)
    string(STRIP "${INPUT_VERSION}" _ph_input)
    string(REGEX REPLACE "^v" "" _ph_input "${_ph_input}")
    if(_ph_input MATCHES "^[0-9]+(\\.[0-9]+)?(\\.[0-9]+)?(\\.[0-9]+)?$")
        set(${OUT_VERSION} "${_ph_input}" PARENT_SCOPE)
    else()
        set(${OUT_VERSION} "" PARENT_SCOPE)
    endif()
endfunction()

function(perfecthash_resolve_version)
    set(options)
    set(one_value_args
        SOURCE_DIR
        FALLBACK_VERSION
        OVERRIDE
        OUT_VERSION
        OUT_SOURCE
        OUT_SEMVER
        OUT_GIT_TAG
        OUT_GIT_SHA
        OUT_GIT_DIRTY
    )
    cmake_parse_arguments(PH "${options}" "${one_value_args}" "" ${ARGN})

    if(NOT PH_SOURCE_DIR)
        set(PH_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
    if(NOT PH_FALLBACK_VERSION)
        set(PH_FALLBACK_VERSION "0.0.0")
    endif()

    set(_ph_version "")
    set(_ph_source "")
    set(_ph_semver "")
    set(_ph_git_tag "")
    set(_ph_git_sha "")
    set(_ph_git_dirty OFF)

    if(PH_OVERRIDE)
        perfecthash_normalize_version("${PH_OVERRIDE}" _ph_override_version)
        if(NOT _ph_override_version)
            message(FATAL_ERROR "Invalid PERFECTHASH_VERSION_OVERRIDE: '${PH_OVERRIDE}'.")
        endif()
        set(_ph_version "${_ph_override_version}")
        set(_ph_source "override")
    endif()

    if(NOT _ph_version)
        find_package(Git QUIET)
        if(GIT_FOUND)
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" -C "${PH_SOURCE_DIR}" rev-parse --short=12 HEAD
                RESULT_VARIABLE _ph_git_sha_result
                OUTPUT_VARIABLE _ph_git_sha_output
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(_ph_git_sha_result EQUAL 0 AND _ph_git_sha_output)
                set(_ph_git_sha "${_ph_git_sha_output}")
            endif()

            execute_process(
                COMMAND "${GIT_EXECUTABLE}" -C "${PH_SOURCE_DIR}" diff-index --quiet HEAD --
                RESULT_VARIABLE _ph_git_dirty_result
                OUTPUT_QUIET
                ERROR_QUIET
            )
            if(_ph_git_dirty_result EQUAL 1)
                set(_ph_git_dirty ON)
            endif()

            execute_process(
                COMMAND "${GIT_EXECUTABLE}" -C "${PH_SOURCE_DIR}"
                        describe --tags --exact-match --match "v[0-9]*"
                RESULT_VARIABLE _ph_exact_tag_result
                OUTPUT_VARIABLE _ph_exact_tag
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(_ph_exact_tag_result EQUAL 0 AND _ph_exact_tag)
                perfecthash_normalize_version("${_ph_exact_tag}" _ph_exact_version)
                if(_ph_exact_version)
                    set(_ph_version "${_ph_exact_version}")
                    set(_ph_source "git-exact-tag")
                    set(_ph_git_tag "${_ph_exact_tag}")
                endif()
            endif()

            if(NOT _ph_version)
                execute_process(
                    COMMAND "${GIT_EXECUTABLE}" -C "${PH_SOURCE_DIR}"
                            describe --tags --abbrev=0 --match "v[0-9]*"
                    RESULT_VARIABLE _ph_latest_tag_result
                    OUTPUT_VARIABLE _ph_latest_tag
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET
                )
                if(_ph_latest_tag_result EQUAL 0 AND _ph_latest_tag)
                    perfecthash_normalize_version("${_ph_latest_tag}" _ph_latest_version)
                    if(_ph_latest_version)
                        set(_ph_version "${_ph_latest_version}")
                        set(_ph_source "git-latest-tag")
                        set(_ph_git_tag "${_ph_latest_tag}")
                    endif()
                endif()
            endif()
        endif()
    endif()

    if(NOT _ph_version)
        perfecthash_normalize_version("${PH_FALLBACK_VERSION}" _ph_fallback_version)
        if(NOT _ph_fallback_version)
            message(FATAL_ERROR "Invalid PERFECTHASH_FALLBACK_VERSION: '${PH_FALLBACK_VERSION}'.")
        endif()
        set(_ph_version "${_ph_fallback_version}")
        set(_ph_source "fallback")
    endif()

    set(_ph_semver "${_ph_version}")
    if(NOT _ph_source STREQUAL "git-exact-tag" AND _ph_git_sha)
        set(_ph_semver "${_ph_version}-dev+g${_ph_git_sha}")
    endif()
    if(_ph_git_dirty)
        set(_ph_semver "${_ph_semver}.dirty")
    endif()

    if(PH_OUT_VERSION)
        set(${PH_OUT_VERSION} "${_ph_version}" PARENT_SCOPE)
    endif()
    if(PH_OUT_SOURCE)
        set(${PH_OUT_SOURCE} "${_ph_source}" PARENT_SCOPE)
    endif()
    if(PH_OUT_SEMVER)
        set(${PH_OUT_SEMVER} "${_ph_semver}" PARENT_SCOPE)
    endif()
    if(PH_OUT_GIT_TAG)
        set(${PH_OUT_GIT_TAG} "${_ph_git_tag}" PARENT_SCOPE)
    endif()
    if(PH_OUT_GIT_SHA)
        set(${PH_OUT_GIT_SHA} "${_ph_git_sha}" PARENT_SCOPE)
    endif()
    if(PH_OUT_GIT_DIRTY)
        set(${PH_OUT_GIT_DIRTY} "${_ph_git_dirty}" PARENT_SCOPE)
    endif()
endfunction()
