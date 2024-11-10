rem echo off

rem debug
if exist build.cuda.debug (
    rmdir /q /s build.cuda.debug
)
if exist build.cuda.debug.ninja (
    rmdir /q /s build.cuda.debug.ninja
)
cmake -B build.cuda.debug -G "Visual Studio 17 2022" -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake -B build.cuda.debug.ninja -G "Ninja Multi-Config" -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89

rem release
if exist build.cuda.release (
    rmdir /q /s build.cuda.release
)
if exist build.cuda.release.ninja (
    rmdir /q /s build.cuda.release.ninja
)
cmake -B build.cuda.release -G "Visual Studio 17 2022" -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake -B build.cuda.release.ninja -G "Ninja Multi-Config" -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
