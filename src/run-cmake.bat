rem echo off

set script_dir=%~dp0
for %%I in ("%script_dir%..") do set root_dir=%%~fI

rem debug
if exist "%root_dir%\\build.debug" (
    rmdir /q /s "%root_dir%\\build.debug"
)
if exist "%root_dir%\\build.debug.ninja" (
    rmdir /q /s "%root_dir%\\build.debug.ninja"
)
cmake -S "%root_dir%" -B "%root_dir%\\build.debug" -G "Visual Studio 17 2022"
cmake -S "%root_dir%" -B "%root_dir%\\build.debug.ninja" -G "Ninja Multi-Config"

rem release
if exist "%root_dir%\\build.release" (
    rmdir /q /s "%root_dir%\\build.release"
)
if exist "%root_dir%\\build.release.ninja" (
    rmdir /q /s "%root_dir%\\build.release.ninja"
)
cmake -S "%root_dir%" -B "%root_dir%\\build.release" -G "Visual Studio 17 2022"
cmake -S "%root_dir%" -B "%root_dir%\\build.release.ninja" -G "Ninja Multi-Config"
