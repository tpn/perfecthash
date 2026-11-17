cmake_minimum_required(VERSION 3.20)
# The static Core archive includes a C++ thread pool on Linux.
project(InstalledOnlineJit LANGUAGES C CXX)

find_package(PerfectHash CONFIG REQUIRED)
enable_testing()

foreach(core IN ITEMS PerfectHashOnlineCore PerfectHashOnlineCoreStatic)
    add_executable(${core}-client main.c)
    target_link_libraries(${core}-client PRIVATE PerfectHash::${core} ${CMAKE_DL_LIBS})
    set(plugin "$<TARGET_FILE:PerfectHash::PerfectHashLLVM>")
    if(WIN32 AND core STREQUAL "PerfectHashOnlineCoreStatic")
        # Static Core loads its plugin beside the consumer executable.
        set_target_properties(${core}-client PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/static")
        add_custom_command(TARGET ${core}-client POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${plugin}" "$<TARGET_FILE:PerfectHash::PerfectHash>"
                "$<TARGET_FILE_DIR:${core}-client>" VERBATIM)
        set(plugin "$<TARGET_FILE_DIR:${core}-client>/$<TARGET_FILE_NAME:PerfectHash::PerfectHashLLVM>")
    endif()
    add_test(NAME ${core}-client COMMAND ${core}-client "$<SHELL_PATH:${plugin}>")
endforeach()
