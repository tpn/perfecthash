#/bin/sh

if [ -d build.debug ]; then
    rm -rf build.debug
fi

mkdir -p build.debug
cmake -B build.debug -G "Ninja Multi-Config"

if [ -d build.release ]; then
    rm -rf build.release
fi

mkdir -p build.release
cmake -B build.release -G "Ninja Multi-Config"
