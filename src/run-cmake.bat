rem echo off
setlocal EnableDelayedExpansion

rem cmake -B build -G"Visual Studio 17 2022"  -DCMAKE_CUDA_ARCHITECTURES="70" -DCMAKE_CUDA_SEPARABLE_COMPILATION=ON
rem cmake -B build -G "Visual Studio 17 2022" -DCMAKE_CUDA_ARCHITECTURES=75 -Dstdgpu_DEBUG_DIR=..\..\stdgpu\debug\lib\cmake\stdgpu -Dstdgpu_RELEASE_DIR=..\..\stdgpu\bin\lib\cmake\stdgpu
rem cmake -B build -G "Visual Studio 17 2022" -DCMAKE_CUDA_ARCHITECTURES=75 -Dstdgpu_DIR=..\..\stdgpu\debug\lib\cmake\stdgpu -Dstdgpu_DEBUG_DIR=..\..\stdgpu\debug\lib\cmake\stdgpu -Dstdgpu_RELEASE_DIR=..\..\stdgpu\bin\lib\cmake\stdgpu

set _cuda_arch=75
set _vs_gen="Visual Studio 17 2022"
set _ninja_gen="Ninja Multi-Config"

set _relative_path=..\..\stdgpu\debug\lib\cmake\stdgpu
for %%i in ("%_relative_path%") do set "_absolute_path=%%~fi"
echo abs path: %_absolute_path%
if not exist build.debug (
    mkdir build.debug
)
cmake -B build.debug -G %_vs_gen% -DCMAKE_CUDA_ARCHITECTURES=%_cuda_arch% -Dstdgpu_DIR=%_absolute_path%

set _relative_path=..\..\stdgpu\release\lib\cmake\stdgpu
for %%i in ("%_relative_path%") do set "_absolute_path=%%~fi"
echo abs path: %_absolute_path%
if not exist build.release (
    mkdir build.release
)
cmake -B build.release -G %_vs_gen% -DCMAKE_CUDA_ARCHITECTURES=%_cuda_arch% -Dstdgpu_DIR=%_absolute_path%

