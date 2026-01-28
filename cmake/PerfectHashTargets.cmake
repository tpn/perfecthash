include_guard(GLOBAL)

function(perfecthash_apply_common_definitions target)
    target_include_directories(
        ${target}
        PUBLIC
        "${PERFECTHASH_ROOT_DIR}/include"
        "${PERFECTHASH_SRC_DIR}"
    )

    target_compile_definitions(${target} PUBLIC PERFECT_HASH_CMAKE)
    target_compile_definitions(${target} PUBLIC "PERFECT_HASH_BUILD_CONFIG=\"$<CONFIG>\"")
    target_compile_definitions(
        ${target}
        PUBLIC
        "$<$<CONFIG:Debug>:_DEBUG>"
        "$<$<NOT:$<CONFIG:Debug>>:NDEBUG>"
        "PH_COMPILER_$<UPPER_CASE:$<C_COMPILER_ID>>"
    )

    if(PERFECTHASH_USE_CUDA)
        target_compile_definitions(${target} PUBLIC PH_USE_CUDA)
    endif()
    if(PERFECTHASH_HAS_LLVM)
        target_compile_definitions(${target} PUBLIC PH_HAS_LLVM)
    endif()

    if(PERFECTHASH_IS_WINDOWS)
        target_compile_definitions(${target} PUBLIC PH_WINDOWS _WINDOWS _UNICODE UNICODE)
    else()
        target_compile_definitions(${target} PUBLIC PH_COMPAT)
        if(PERFECTHASH_IS_LINUX)
            target_compile_definitions(${target} PUBLIC PH_LINUX)
        elseif(PERFECTHASH_IS_MAC)
            target_compile_definitions(${target} PUBLIC PH_MAC)
        endif()
        if(DEFINED PERFECTHASH_PAGE_SHIFT)
            target_compile_definitions(${target} PUBLIC "PH_PAGE_SHIFT=${PERFECTHASH_PAGE_SHIFT}")
        endif()
    endif()
endfunction()

function(perfecthash_apply_common_settings target)
    perfecthash_apply_common_definitions(${target})

    if(MSVC)
        set(ph_asm_output_dir "${CMAKE_CURRENT_BINARY_DIR}/asm/${target}")
        file(MAKE_DIRECTORY "${ph_asm_output_dir}")
        target_compile_options(
            ${target}
            PRIVATE
            /Gz         # __stdcall
            /TC         # Compile as C Code
            /FAcs       # Assembler Output: Assembly, Machine Code & Source
            /Fa${ph_asm_output_dir}/
            /WX         # Warnings as errors
            /Wall       # All warnings
            /FC         # Use full paths
            /GS-        # No security checks
        )
        target_compile_options(
            ${target}
            PRIVATE
            $<$<CONFIG:Debug>:/ZI;/Od;/Ob0>
            $<$<NOT:$<CONFIG:Debug>>:/Zi>
        )
        target_link_options(
            ${target}
            PRIVATE
            /NODEFAULTLIB           # Ignore default libraries
            /DEBUG                  # Generate debug information
            /NXCOMPAT               # Data execution prevention
            chkstk.obj              # Link with chkstk.obj
            bufferoverflowU.lib     # Link with bufferoverflowU.lib
        )
        get_target_property(ph_skip_runtime_libs ${target} PERFECTHASH_SKIP_RUNTIME_LIBS)
        if(NOT ph_skip_runtime_libs)
            target_link_libraries(
                ${target}
                PRIVATE
                $<$<CONFIG:Debug>:ucrtd;vcruntimed;msvcrtd>
                $<$<NOT:$<CONFIG:Debug>>:ucrt;vcruntime>
            )
        endif()
        target_link_options(
            ${target}
            PRIVATE
            $<$<NOT:$<CONFIG:Debug>>:/LTCG;/PROFILE;/OPT:REF;/OPT:ICF;/RELEASE;/MANIFESTUAC>
        )
    else()
        target_compile_options(
            ${target}
            PRIVATE
            -Wno-deprecated-declarations
            -Wno-multichar
            -fno-omit-frame-pointer
        )
        target_compile_options(
            ${target}
            PRIVATE
            $<$<COMPILE_LANGUAGE:C>:-Wno-incompatible-pointer-types>
        )
        if(PERFECTHASH_ENABLE_NATIVE_ARCH)
            target_compile_options(${target} PRIVATE -march=native)
        endif()
        if(CMAKE_C_COMPILER_ID MATCHES Clang)
            target_compile_options(
                ${target}
                PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-Wno-enum-conversion>
            )
        endif()
        if(PERFECTHASH_IS_EMSCRIPTEN)
            target_compile_options(${target} PRIVATE ${PERFECTHASH_EMSCRIPTEN_COMPILE_OPTIONS})
            target_link_options(${target} PRIVATE ${PERFECTHASH_EMSCRIPTEN_LINK_OPTIONS})
        endif()
        target_link_libraries(${target} PRIVATE Threads::Threads)
        if(PERFECTHASH_IS_LINUX)
            target_link_libraries(${target} PRIVATE rt)
        endif()
    endif()
endfunction()

function(perfecthash_apply_cuda_settings target)
    perfecthash_apply_common_definitions(${target})
    target_compile_definitions(${target} PUBLIC PH_CUDA)
endfunction()
