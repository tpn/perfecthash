
if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(IS_WINDOWS 1)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(IS_LINUX 1)
    set(IS_UNIX 1)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(IS_MAC 1)
    set(IS_UNIX 1)
else()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()

if (IS_WINDOWS)
    enable_language(ASM_MASM)
    target_compile_definitions(${PROJECT_NAME} PUBLIC PH_WINDOWS)
else()
    target_compile_definitions(${PROJECT_NAME} PUBLIC PH_COMPAT)
endif()


target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    ${PROJECT_ROOT}/include
    ${SRC_ROOT}
)

target_compile_definitions(${PROJECT_NAME} PUBLIC "PERFECT_HASH_CMAKE")

if (IS_WINDOWS)

    target_compile_definitions(
        ${PROJECT_NAME}
        PUBLIC
        "PERFECT_HASH_BUILD_CONFIG=\"$<CONFIG>\""
    )


    #target_precompile_headers(${PROJECT_NAME} PUBLIC "stdafx.h")

    # Compile options.
    target_compile_options(
        ${PROJECT_NAME}
        PUBLIC
        /Gz         # __stdcall
        /TC         # Compile as C Code
        /FAcs       # Assembler Output: Assembly, Machine Code & Source
        /WX         # Warnings as errors
        /Wall       # All warnings
        /FC         # Use full paths
        /GS-        # No security checks
        /FR         # Generate browse info
    )

    target_compile_options_config(Debug /ZI)
    target_compile_options_not_config(Debug /Zi)

    target_compile_definitions(${PROJECT_NAME} PUBLIC _WINDOWS _UNICODE UNICODE)

    target_compile_definitions_config(Debug _DEBUG)
    target_compile_definitions_not_config(Debug NDEBUG)

    # Link options.
    target_link_options(
        ${PROJECT_NAME}
        PUBLIC
        /NODEFAULTLIB           # Ignore default libraries
        /DEBUG                  # Generate debug information
        /NXCOMPAT               # Data execution prevention
        chkstk.obj              # Link with chkstk.obj
        bufferoverflowU.lib     # Link with bufferoverflowU.lib
    )

    target_link_options_not_config(
        Debug
        /LTCG           # Link-time code generation
        /PROFILE        # Allow profiling
        /OPT:REF        # References
        /OPT:ICF        # Enable COMDAT folding
        /RELEASE        # Set checksum
        /MANIFESTUAC    # Enable UAC
    )


else()

    target_compile_definitions(
        ${PROJECT_NAME}
        PUBLIC
        "PERFECT_HASH_BUILD_CONFIG=\"${CMAKE_SYSTEM_NAME}\""
        "PH_COMPAT"
        "PH_COMPILER_$<UPPER_CASE:${CMAKE_C_COMPILER_ID}>"
    )

    target_compile_options(
        ${PROJECT_NAME}
        PUBLIC
        -march=native
        -Wno-incompatible-pointer-types
        -Wno-multichar # For Rtlc: CpuInfo.Ebx = (LONG)'uneG'
    )

    if (CMAKE_C_COMPILER_ID MATCHES Clang)

        target_compile_options(
            ${PROJECT_NAME}
            PUBLIC
            -Wno-enum-conversion # IsValidVCProjectFileId((FILE_ID)Id)
        )
    endif()


endif()
