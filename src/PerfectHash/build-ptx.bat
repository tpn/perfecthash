rem @echo off

@set sources=Graph

@for %%s in (%sources%) do (
    nvcc -ptx ..\PerfectHashCuda\%%s.cu                 ^
        -o ..\CompiledPerfectHashTable\%%sDebug.ptx     ^
        -I../../include                                 ^
        -D_PERFECT_HASH_CUDA_INTERNAL_BUILD             ^
        -DPH_WINDOWS                                    ^
        -DPH_CUDA                                       ^
        -Wno-deprecated-gpu-targets                     ^
        --device-debug                                  ^
        --restrict                                      ^
        -rdc=true                                       ^
        --source-in-ptx                                 ^
        --keep                                          ^
        -maxrregcount=0                                 ^
        --ptxas-options=-v                              ^
        --machine 64                                    ^
        -ptx -cudart shared                             ^
        -gencode=arch=compute_60,code=\"sm_60,compute_60\"

    rem Currently disabled due to .ptx lines being too long for msvc.
    rem cmd /c ph update-raw-c-string-file -i ..\CompiledPerfectHashTable\%%sDebug.ptx

    nvcc -ptx ..\PerfectHashCuda\%%s.cu                 ^
        -o ..\CompiledPerfectHashTable\%%sRelease.ptx   ^
        -Wno-deprecated-gpu-targets                     ^
        -I../../include                                 ^
        -D_PERFECT_HASH_CUDA_INTERNAL_BUILD             ^
        -DPH_WINDOWS                                    ^
        -DPH_CUDA                                       ^
        --optimize 3                                    ^
        --generate-line-info                            ^
        --restrict                                      ^
        -rdc=true                                       ^
        --source-in-ptx                                 ^
        --keep                                          ^
        -maxrregcount=0                                 ^
        --ptxas-options=-v                              ^
        --machine 64                                    ^
        -ptx -cudart shared                             ^
        -gencode=arch=compute_60,code=\"sm_60,compute_60\"

    rem cmd /c ph update-raw-c-string-file -i ..\CompiledPerfectHashTable\%%sRelease.ptx
)


@rem vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                   :
