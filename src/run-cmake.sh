#/bin/sh

_ARCH=70

mkdir -p build.debug
cmake -B build.debug -G "Ninja Multi-Config" -DCMAKE_CUDA_ARCHITECTURES=$_ARCH -Dstdgpu_DIR=$(realpath ../../stdgpu/debug/lib/cmake/stdgpu/)

mkdir -p build.release
cmake -B build.release -G "Ninja Multi-Config" -DCMAKE_CUDA_ARCHITECTURES=$_ARCH -Dstdgpu_DIR=$(realpath ../../stdgpu/release/lib/cmake/stdgpu/)
