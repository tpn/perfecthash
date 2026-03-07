@echo off

for /f "usebackq" %%i in (`dir /b ..\..\include\PerfectHash\CompiledPerfectHash*.*`) do (
    cmd /c ph update-raw-c-string-file -i ..\..\include\PerfectHash\%%i
)

cmd /c ph update-raw-c-string-file -i ..\..\include\PerfectHash\no_sal2.h

for /f "usebackq" %%i in (`dir /b ..\CompiledPerfectHashTable\*.*`) do (
    cmd /c ph update-raw-c-string-file -i ..\CompiledPerfectHashTable\%%i
)

