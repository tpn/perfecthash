# Copyright (c) 2026 Trent Nelson <trent@trent.me>

param(
    [string]$BuildDir = "build-win",
    [string]$Config = "Debug",
    [string]$OutputDir = "build-win\\minidump-test",
    [switch]$ForceFallback
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$bulkExe = Join-Path $BuildDir ("bin\\{0}\\PerfectHashBulkCreate.exe" -f $Config)
$serverExe = Join-Path $BuildDir ("bin\\{0}\\PerfectHashServer.exe" -f $Config)

if (-not (Test-Path $bulkExe)) {
    throw "Bulk-create exe not found: $bulkExe"
}

if (-not (Test-Path $serverExe)) {
    throw "Server exe not found: $serverExe"
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$dumpDir = (Resolve-Path $OutputDir).Path

$bulkDump = Join-Path $dumpDir "PerfectHashBulkCreateCrash.dmp"
$bulkLog = Join-Path $dumpDir "PerfectHashBulkCreateCrash.log"
$serverDump = Join-Path $dumpDir "PerfectHashServerCrash.dmp"
$serverLog = Join-Path $dumpDir "PerfectHashServerCrash.log"

Remove-Item $bulkDump, $bulkLog, $serverDump, $serverLog -ErrorAction SilentlyContinue

$envSnapshot = @{
    PH_LOG_BULK_CREATE_CRASH = $env:PH_LOG_BULK_CREATE_CRASH
    PH_BULK_CREATE_CRASH_TEST = $env:PH_BULK_CREATE_CRASH_TEST
    PH_BULK_CREATE_MINIDUMP_FORCE_FALLBACK = $env:PH_BULK_CREATE_MINIDUMP_FORCE_FALLBACK
    PH_BULK_CREATE_CRASH_DIR = $env:PH_BULK_CREATE_CRASH_DIR
    PH_LOG_SERVER_CRASH = $env:PH_LOG_SERVER_CRASH
    PH_SERVER_CRASH_TEST = $env:PH_SERVER_CRASH_TEST
    PH_SERVER_MINIDUMP_FORCE_FALLBACK = $env:PH_SERVER_MINIDUMP_FORCE_FALLBACK
    PH_SERVER_CRASH_DIR = $env:PH_SERVER_CRASH_DIR
}

try {
    $env:PH_BULK_CREATE_CRASH_DIR = $dumpDir
    $env:PH_LOG_BULK_CREATE_CRASH = "1"
    $env:PH_BULK_CREATE_CRASH_TEST = "AV"
    if ($ForceFallback) {
        $env:PH_BULK_CREATE_MINIDUMP_FORCE_FALLBACK = "1"
    } else {
        Remove-Item Env:PH_BULK_CREATE_MINIDUMP_FORCE_FALLBACK `
            -ErrorAction SilentlyContinue
    }

    $bulkProcess = Start-Process -FilePath $bulkExe `
                                 -PassThru `
                                 -NoNewWindow `
                                 -Wait

    if ($bulkProcess.ExitCode -eq 0) {
        throw ("Bulk-create crash test exited with code {0}" -f `
            $bulkProcess.ExitCode)
    }

    if (-not (Test-Path $bulkDump)) {
        throw "Bulk-create minidump not found."
    }

    if ((Get-Item $bulkDump).Length -eq 0) {
        throw "Bulk-create minidump is empty."
    }

    if (-not (Test-Path $bulkLog)) {
        throw "Bulk-create crash log not found."
    }

    if ((Get-Item $bulkLog).Length -eq 0) {
        throw "Bulk-create crash log is empty."
    }

    $env:PH_SERVER_CRASH_DIR = $dumpDir
    $env:PH_LOG_SERVER_CRASH = "1"
    $env:PH_SERVER_CRASH_TEST = "AV"
    if ($ForceFallback) {
        $env:PH_SERVER_MINIDUMP_FORCE_FALLBACK = "1"
    } else {
        Remove-Item Env:PH_SERVER_MINIDUMP_FORCE_FALLBACK `
            -ErrorAction SilentlyContinue
    }

    $serverProcess = Start-Process -FilePath $serverExe `
                                   -PassThru `
                                   -NoNewWindow `
                                   -Wait

    if ($serverProcess.ExitCode -eq 0) {
        throw ("Server crash test exited with code {0}" -f `
            $serverProcess.ExitCode)
    }

    if (-not (Test-Path $serverDump)) {
        throw "Server minidump not found."
    }

    if ((Get-Item $serverDump).Length -eq 0) {
        throw "Server minidump is empty."
    }

    if (-not (Test-Path $serverLog)) {
        throw "Server crash log not found."
    }

    if ((Get-Item $serverLog).Length -eq 0) {
        throw "Server crash log is empty."
    }
} finally {
    foreach ($name in $envSnapshot.Keys) {
        $value = $envSnapshot[$name]
        if ($null -eq $value) {
            Remove-Item ("Env:{0}" -f $name) -ErrorAction SilentlyContinue
        } else {
            Set-Item ("Env:{0}" -f $name) -Value $value
        }
    }
}

Write-Host "Minidump test succeeded."
