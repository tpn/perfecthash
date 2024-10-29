rem @echo off

nvcc -ptx ..\PerfectHashCuda\Graph.cu -o ..\CompiledPerfectHashTable\GraphDebug.ptx -I../../include -D_PERFECT_HASH_INTERNAL_BUILD -D_PERFECT_HASH_CUDA_INTERNAL_BUILD -DPH_WINDOWS -DPH_CUDA -Wno-deprecated-gpu-targets --device-debug --restrict -rdc=true --source-in-ptx --keep -maxrregcount=0 --ptxas-options=-v --machine 64 -ptx -cudart shared -gencode=arch=compute_75,code=\"sm_75,compute_75\"

rem nvcc -ptx ..\PerfectHashCuda\%%s.cu -o ..\CompiledPerfectHashTable\%%sRelease.ptx -Wno-deprecated-gpu-targets -I../../include -D_PERFECT_HASH_CUDA_INTERNAL_BUILD -DPH_WINDOWS -DPH_CUDA --optimize 3 --generate-line-info --restrict -rdc=true --source-in-ptx --keep -maxrregcount=0 --ptxas-options=-v --machine 64 -ptx -cudart shared -gencode=arch=compute_75,code=\"sm_75,compute_75\"

@rem vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                   :
