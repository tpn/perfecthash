#!/bin/sh

_ARCH=${CUDA_ARCH:-70}

set -e

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root_dir=$(cd "$script_dir/.." && pwd)

if [ -d "$root_dir/build.debug" ]; then
    rm -rf "$root_dir/build.debug"
fi

mkdir -p "$root_dir/build.debug"
cmake -S "$root_dir" -B "$root_dir/build.debug" -G "Ninja Multi-Config" \
    -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$_ARCH"

if [ -d "$root_dir/build.release" ]; then
    rm -rf "$root_dir/build.release"
fi

mkdir -p "$root_dir/build.release"
cmake -S "$root_dir" -B "$root_dir/build.release" -G "Ninja Multi-Config" \
    -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$_ARCH"
