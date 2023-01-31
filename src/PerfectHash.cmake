target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    ${PROJECT_ROOT}/include
    ${SRC_ROOT}
)

target_compile_definitions(${PROJECT_NAME} PUBLIC "PERFECT_HASH_CMAKE")


target_compile_definitions(
    ${PROJECT_NAME}
    PUBLIC
    "PERFECT_HASH_BUILD_CONFIG=\"$<CONFIG>\""
)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(IS_WINDOWS 1)
else()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()

if (IS_WINDOWS)

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


endif()
