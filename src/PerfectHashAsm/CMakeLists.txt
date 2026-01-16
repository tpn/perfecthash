set(target PerfectHashAsm)

if (WIN32)
    enable_language(ASM_MASM)
endif()

################################################################################
# Source groups
################################################################################

set(Assembly_Source_Files
    "../PerfectHash/memset_x64.asm"
    "../PerfectHash/RtlCopyPages_x64.asm"
    "../PerfectHash/RtlFillPages_x64.asm"
    "../PerfectHash/PerfectHashTableFastIndexEx_x64_01.asm")

set_source_files_properties(
    ${Assembly_Source_Files}
    PROPERTIES
    LANGUAGE ASM_MASM
)

source_group("Source Files" FILES ${Assembly_Source_Files})

set(ALL_FILES ${Assembly_Source_Files})

add_library(${target} STATIC ${ALL_FILES})
set(CMAKE_ASM_MASM_CREATE_STATIC_LIBRARY
    "<CMAKE_AR> /NOLOGO /OUT:<TARGET> <LINK_FLAGS> <OBJECTS>")

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

################################################################################
# Dependencies
################################################################################

#  vim:set ts=8 sw=4 sts=4 tw=80 expandtab syntax=cmake                        :
