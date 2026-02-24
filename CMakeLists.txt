cmake_minimum_required(VERSION 3.20)

if(WIN32 AND NOT DEFINED CMAKE_SYSTEM_VERSION)
    set(CMAKE_SYSTEM_VERSION 10.0 CACHE STRING "" FORCE)
endif()

project(
    PerfectHash
    VERSION 0.63.0
    LANGUAGES C CXX
)

include(CTest)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(
        CMAKE_INSTALL_PREFIX
        "${CMAKE_SOURCE_DIR}/install"
        CACHE PATH "Default install prefix." FORCE
    )
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(PerfectHashProject)
include(PerfectHashTargets)

add_subdirectory(src)

if(PERFECTHASH_ENABLE_TESTS AND BUILD_TESTING)
    add_subdirectory(tests)
endif()

if(PERFECTHASH_ENABLE_INSTALL)
    install(DIRECTORY "${PERFECTHASH_ROOT_DIR}/include/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    install(
        TARGETS
            PerfectHash
            PerfectHashOnline
            PerfectHashOnlineStatic
            PerfectHashOnlineCore
            PerfectHashOnlineCoreStatic
            PerfectHashCreateExe
            PerfectHashBulkCreateExe
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    if(TARGET PerfectHashLLVM)
        install(
            TARGETS PerfectHashLLVM
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
    endif()

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
