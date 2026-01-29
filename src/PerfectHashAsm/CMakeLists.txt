set(target PerfectHashAsm)

include(CheckLanguage)

set(Assembly_Source_Files)
set(ASM_LANGUAGE "")

# CMake's ASM_NASM detection doesn't always search PATH; seed it explicitly.
if(NOT CMAKE_ASM_NASM_COMPILER)
    find_program(PERFECTHASH_NASM_EXECUTABLE nasm)
    if(PERFECTHASH_NASM_EXECUTABLE)
        set(CMAKE_ASM_NASM_COMPILER "${PERFECTHASH_NASM_EXECUTABLE}"
            CACHE FILEPATH "NASM compiler" FORCE)
    endif()
endif()

check_language(ASM_NASM)
if(CMAKE_ASM_NASM_COMPILER)
    enable_language(ASM_NASM)
    set(ASM_LANGUAGE ASM_NASM)
    list(
        APPEND Assembly_Source_Files
        "../PerfectHash/memset_x64.nasm"
        "../PerfectHash/RtlCopyPages_x64.nasm"
        "../PerfectHash/RtlFillPages_x64.nasm"
        "../PerfectHash/PerfectHashTableFastIndexEx_x64_01.nasm"
    )
elseif (WIN32)
    enable_language(ASM_MASM)
    set(ASM_LANGUAGE ASM_MASM)
    list(
        APPEND Assembly_Source_Files
        "../PerfectHash/memset_x64.asm"
        "../PerfectHash/RtlCopyPages_x64.asm"
        "../PerfectHash/RtlFillPages_x64.asm"
        "../PerfectHash/PerfectHashTableFastIndexEx_x64_01.asm"
    )
else()
    message(FATAL_ERROR "NASM is required to build PerfectHashAsm on non-Windows platforms.")
endif()

################################################################################
# Source groups
################################################################################

set_source_files_properties(
    ${Assembly_Source_Files}
    PROPERTIES
    LANGUAGE ${ASM_LANGUAGE}
)

source_group("Source Files" FILES ${Assembly_Source_Files})

set(ALL_FILES ${Assembly_Source_Files})

add_library(${target} STATIC ${ALL_FILES})
if (ASM_LANGUAGE STREQUAL "ASM_MASM")
    set(CMAKE_ASM_MASM_CREATE_STATIC_LIBRARY
        "<CMAKE_AR> /NOLOGO /OUT:<TARGET> <LINK_FLAGS> <OBJECTS>")
elseif (WIN32 AND ASM_LANGUAGE STREQUAL "ASM_NASM")
    set(CMAKE_ASM_NASM_CREATE_STATIC_LIBRARY
        "<CMAKE_AR> /NOLOGO /OUT:<TARGET> <LINK_FLAGS> <OBJECTS>")
endif()

################################################################################
# Target
################################################################################
#use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "${DEFAULT_ASM_MASM_PROPS}")

#set(ROOT_NAMESPACE PerfectHash)

################################################################################
# Compile definitions
################################################################################
target_compile_definitions(
    ${target}
    PRIVATE
    "_PERFECT_HASH_INTERNAL_BUILD"
)

################################################################################
# Compile and link options
################################################################################
if (ASM_LANGUAGE STREQUAL "ASM_MASM")
    target_compile_options(
        ${target}
        PRIVATE
        /Zd         # Add line number debug info
        /Zf         # Make all symbols public
        /WX         # Warnings as errors
        /nologo
        /I${PERFECTHASH_ROOT_DIR}/include
    )

    # ^ This appears to have stopped working, so hack it in here.
    set(CMAKE_ASM_MASM_FLAGS
        "${CMAKE_ASM_MASM_FLAGS} /Zd /Zf /WX /nologo /I${PERFECTHASH_ROOT_DIR}/include")
else()
    target_compile_options(
        ${target}
        PRIVATE
        -I${PERFECTHASH_ROOT_DIR}/include
    )
    set(CMAKE_ASM_NASM_FLAGS
        "${CMAKE_ASM_NASM_FLAGS} -I${PERFECTHASH_ROOT_DIR}/include")
endif()

################################################################################
# Dependencies
################################################################################

#  vim:set ts=8 sw=4 sts=4 tw=80 expandtab syntax=cmake                        :
