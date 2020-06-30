rem @echo off

@set sources=Graph

@for %%s in (%sources%) do (
    nvcc -ptx ..\PerfectHashCuda\%%s.cu -o ..\x64\Debug\%%s.ptx     ^
        -Wno-deprecated-gpu-targets                                 ^
        --cudart=none                                               ^
        --device-debug                                              ^
        --restrict

    nvcc -ptx ..\PerfectHashCuda\%%s.cu -o ..\x64\Release\%%s.ptx   ^
        -Wno-deprecated-gpu-targets                                 ^
        --optimize 3                                                ^
        --generate-line-info                                        ^
        --restrict
)

@rem vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                   :
