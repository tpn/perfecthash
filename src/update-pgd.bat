@echo off

setlocal enableextensions

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
