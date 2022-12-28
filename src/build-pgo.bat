@echo off
setlocal enableextensions

echo on
del /q .\x64\*.pgd
del /q .\x64\PGInstrument\*.pgc

msbuild /nologo /m /t:Build /p:Configuration=PGInstrument;Platform=x64 PerfectHash.sln
..\bin\timemem.exe .\x64\PGInstrument\PerfectHashCreate.exe c:\src\perfecthash-keys\sys32\HologramWorld-31016.keys c:\temp\ph.pgi Chm01 MultiplyShiftR And 12 --FindBestGraph --BestCoverageType=HighestScore --BestCoverageAttempts=4 --HashAllKeysFirst --TryUseAvx2HashFunction --InitialNumberOfTableResizes=1
..\bin\timemem.exe .\x64\PGInstrument\PerfectHashBulkCreate.exe c:\src\perfecthash-keys\hard c:\temp\ph.pgi Chm01 RotateMultiplyXorRotate2 And 3 --AttemptsBeforeTableResize=10000 --MaxNumberOfTableResizes=0 --FindBestGraph --BestCoverageAttempts=2 --BestCoverageType=HighestNumberOfEmptyCacheLines --MainWorkThreadpoolPriority=Low

@echo off

rem If input is PerfectHash!1.pgc, then:
rem     i = PerfectHash
rem     j = 1
rem     k = pgc
rem
rem We want to generate:
rem     pgomgr /merge x64\PGInstrument\PerfectHash!1.pgc x64\PerfectHash.pgd

echo on

@for /f "usebackq tokens=1,2,3 delims=!." %%i in (`dir /b x64\PGInstrument\*.pgc`) do (
    pgomgr /merge "x64\PGInstrument\%%i!%%j.%%k" x64\%%i.pgd
)

msbuild /nologo /m /t:Build /p:Configuration=PGOptimize;Platform=x64 PerfectHash.sln
