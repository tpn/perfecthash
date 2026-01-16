rem echo off

set script_dir=%~dp0
for %%I in ("%script_dir%..") do set root_dir=%%~fI

rem debug
if exist "%root_dir%\\build.cuda.debug" (
    rmdir /q /s "%root_dir%\\build.cuda.debug"
)
if exist "%root_dir%\\build.cuda.debug.ninja" (
    rmdir /q /s "%root_dir%\\build.cuda.debug.ninja"
)
cmake -S "%root_dir%" -B "%root_dir%\\build.cuda.debug" -G "Visual Studio 17 2022" -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake -S "%root_dir%" -B "%root_dir%\\build.cuda.debug.ninja" -G "Ninja Multi-Config" -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89

rem release
if exist "%root_dir%\\build.cuda.release" (
    rmdir /q /s "%root_dir%\\build.cuda.release"
)
if exist "%root_dir%\\build.cuda.release.ninja" (
    rmdir /q /s "%root_dir%\\build.cuda.release.ninja"
)
cmake -S "%root_dir%" -B "%root_dir%\\build.cuda.release" -G "Visual Studio 17 2022" -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake -S "%root_dir%" -B "%root_dir%\\build.cuda.release.ninja" -G "Ninja Multi-Config" -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
