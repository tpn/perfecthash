rem @echo off

@set sources=Graph

@for %%s in (%sources%) do (
    nvcc -ptx ..\PerfectHashCuda\%%s.cu                 ^
        -o ..\CompiledPerfectHashTable\%%sDebug.ptx     ^
        -Wno-deprecated-gpu-targets                     ^
        --cudart=none                                   ^
        --device-debug                                  ^
        --restrict

    cmd /c ph update-raw-c-string-file -i ..\CompiledPerfectHashTable\%%sDebug.ptx

    nvcc -ptx ..\PerfectHashCuda\%%s.cu                 ^
        -o ..\CompiledPerfectHashTable\%%sRelease.ptx   ^
        -Wno-deprecated-gpu-targets                     ^
        --optimize 3                                    ^
        --generate-line-info                            ^
        --restrict

    cmd /c ph update-raw-c-string-file -i ..\CompiledPerfectHashTable\%%sRelease.ptx
)


@rem vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                   :
