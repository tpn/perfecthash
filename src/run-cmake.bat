rem echo off

rem debug
if exist build.debug (
    rmdir /q /s build.debug
)
if exist build.debug.ninja (
    rmdir /q /s build.debug.ninja
)
cmake -B build.debug -G "Visual Studio 17 2022"
cmake -B build.debug.ninja -G "Ninja Multi-Config"

rem release
if exist build.release (
    rmdir /q /s build.release
)
if exist build.release.ninja (
    rmdir /q /s build.release.ninja
)
cmake -B build.release -G "Visual Studio 17 2022"
cmake -B build.release.ninja -G "Ninja Multi-Config"
