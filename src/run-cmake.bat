rem cmake -B build -G"Visual Studio 17 2022"  -DCMAKE_CUDA_ARCHITECTURES="70" -DCMAKE_CUDA_SEPARABLE_COMPILATION=ON
rem cmake -B build -G "Visual Studio 17 2022" -DCMAKE_CUDA_ARCHITECTURES=75 -Dstdgpu_DEBUG_DIR=..\..\stdgpu\debug\lib\cmake\stdgpu -Dstdgpu_RELEASE_DIR=..\..\stdgpu\bin\lib\cmake\stdgpu
cmake -B build -G "Visual Studio 17 2022" -DCMAKE_CUDA_ARCHITECTURES=75 -Dstdgpu_DIR=..\..\stdgpu\debug\lib\cmake\stdgpu -Dstdgpu_DEBUG_DIR=..\..\stdgpu\debug\lib\cmake\stdgpu -Dstdgpu_RELEASE_DIR=..\..\stdgpu\bin\lib\cmake\stdgpu
