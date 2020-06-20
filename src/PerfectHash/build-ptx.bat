rem @echo off

@set sources=Graph

@for %%s in (%sources%) do (
    nvcc -ptx ..\PerfectHashCuda\%%s.cu                 ^
        -o ..\CompiledPerfectHashTable\%%sDebug.ptx     ^
        -I../../include                                 ^
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
        -gencode=arch=compute_35,code=\"sm_35,compute_35\"


    rem Currently disabled due to .ptx lines being too long for msvc.
    rem cmd /c ph update-raw-c-string-file -i ..\CompiledPerfectHashTable\%%sDebug.ptx

    nvcc -ptx ..\PerfectHashCuda\%%s.cu                 ^
        -o ..\CompiledPerfectHashTable\%%sRelease.ptx   ^
        -Wno-deprecated-gpu-targets                     ^
        -I../../include                                 ^
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
        -gencode=arch=compute_35,code=\"sm_35,compute_35\"

    rem cmd /c ph update-raw-c-string-file -i ..\CompiledPerfectHashTable\%%sRelease.ptx
)


@rem vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                   :
