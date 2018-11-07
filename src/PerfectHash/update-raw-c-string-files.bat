@echo off

for /f "usebackq" %%i in (`dir /b *Usage.txt`) do (
    cmd /c ph update-raw-c-string-file -i %%i
)

for /f "usebackq" %%i in (`dir /b ..\..\include\CompiledPerfectHash*.*`) do (
    cmd /c ph update-raw-c-string-file -i ..\..\include\%%i
)

for /f "usebackq" %%i in (`dir /b ..\CompiledPerfectHashTable\*.*`) do (
    cmd /c ph update-raw-c-string-file -i ..\CompiledPerfectHashTable\%%i
)

