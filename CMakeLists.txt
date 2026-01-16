cmake_minimum_required(VERSION 3.20)

if(WIN32 AND NOT DEFINED CMAKE_SYSTEM_VERSION)
    set(CMAKE_SYSTEM_VERSION 10.0 CACHE STRING "" FORCE)
endif()

project(
    PerfectHash
    VERSION 0.63.0
    LANGUAGES C CXX
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(PerfectHashProject)
include(PerfectHashTargets)

add_subdirectory(src)

if(PERFECTHASH_ENABLE_INSTALL)
    install(DIRECTORY "${PERFECTHASH_ROOT_DIR}/include/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    install(
        TARGETS
            PerfectHash
            PerfectHashCreateExe
            PerfectHashBulkCreateExe
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    if(PERFECTHASH_HAS_CUDA)
        install(
            TARGETS PerfectHashCuda
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
    endif()

    if(PERFECTHASH_ENABLE_PENTER)
        install(
            TARGETS FunctionHook
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
    endif()
endif()
