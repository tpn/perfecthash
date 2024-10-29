#/bin/sh

_ARCH=${CUDA_ARCH:-70}

if [ -d build.debug ]; then
    rm -rf build.debug
fi

mkdir -p build.debug
cmake -B build.debug -G "Ninja Multi-Config" -DCMAKE_CUDA_ARCHITECTURES=$_ARCH -Dstdgpu_DIR=$(realpath ../../stdgpu/debug/lib/cmake/stdgpu/)

if [ -d build.release ]; then
    rm -rf build.release
fi

mkdir -p build.release
cmake -B build.release -G "Ninja Multi-Config" -DCMAKE_CUDA_ARCHITECTURES=$_ARCH -Dstdgpu_DIR=$(realpath ../../stdgpu/release/lib/cmake/stdgpu/)
