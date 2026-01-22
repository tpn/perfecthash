param(
    [string]$BuildDir = "build-win",
    [string]$Config = "Debug",
    [string]$Endpoint = "\\.\\pipe\\PerfectHashServer-Smoke",
    [int]$TimeoutSeconds = 10,
    [bool]$WaitForServer = $true,
    [int]$ConnectTimeoutMs = 10000
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$serverExe = Join-Path $BuildDir ("bin\\{0}\\PerfectHashServer.exe" -f $Config)
$clientExe = Join-Path $BuildDir ("bin\\{0}\\PerfectHashClient.exe" -f $Config)
$workDir = Join-Path $BuildDir "iocp-smoke"
$keysFile = Join-Path $workDir "smoke_keys_16.keys"
$outputDir = Join-Path $workDir "output"

if (-not (Test-Path $serverExe)) {
    throw "Server exe not found: $serverExe"
}

if (-not (Test-Path $clientExe)) {
    throw "Client exe not found: $clientExe"
}

New-Item -ItemType Directory -Path $workDir -Force | Out-Null
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

$keys = 1..128
$stream = [System.IO.File]::Open($keysFile,
                                 [System.IO.FileMode]::Create,
                                 [System.IO.FileAccess]::Write,
                                 [System.IO.FileShare]::None)
$writer = New-Object System.IO.BinaryWriter($stream)
foreach ($key in $keys) {
    $writer.Write([UInt32]$key)
}
$writer.Flush()
$writer.Dispose()
$stream.Dispose()

$keysPath = (Resolve-Path $keysFile).Path
$outputPath = (Resolve-Path $outputDir).Path

$createCommand = "PerfectHashCreate.exe $keysPath $outputPath " +
                 "Chm01 Mulshrolate1RX And 0 --DisableCsvOutputFile"
$tableArg = "--TableCreate=$createCommand"

$serverArgs = @("--Endpoint=$Endpoint")
$clientArgs = @("--Endpoint=$Endpoint", "--Shutdown")
if ($WaitForServer) {
    $clientArgs += "--WaitForServer"
    if ($ConnectTimeoutMs -gt 0) {
        $clientArgs += ("--ConnectTimeout={0}" -f $ConnectTimeoutMs)
    }
}

$server = Start-Process -FilePath $serverExe `
                        -ArgumentList $serverArgs `
                        -PassThru `
                        -NoNewWindow

Start-Sleep -Milliseconds 500

$clientCreateArgs = @("--Endpoint=$Endpoint", "`"$tableArg`"")
if ($WaitForServer) {
    $clientCreateArgs += "--WaitForServer"
    if ($ConnectTimeoutMs -gt 0) {
        $clientCreateArgs += ("--ConnectTimeout={0}" -f $ConnectTimeoutMs)
    }
}

$clientCreate = Start-Process -FilePath $clientExe `
                              -ArgumentList $clientCreateArgs `
                              -PassThru `
                              -NoNewWindow `
                              -Wait

if ($clientCreate.ExitCode -ne 0) {
    throw ("Client create exited with code {0}" -f $clientCreate.ExitCode)
}

$clientShutdown = Start-Process -FilePath $clientExe `
                        -ArgumentList $clientArgs `
                        -PassThru `
                        -NoNewWindow `
                        -Wait

if ($clientShutdown.ExitCode -ne 0) {
    throw ("Client shutdown exited with code {0}" -f $clientShutdown.ExitCode)
}

if (-not $server.WaitForExit($TimeoutSeconds * 1000)) {
    $server.Kill()
    throw ("Server did not exit within {0}s" -f $TimeoutSeconds)
}

$server.WaitForExit()
$server.Refresh()

if ($null -eq $server.ExitCode) {
    $serverProcess = Get-Process -Id $server.Id -ErrorAction SilentlyContinue
    if ($null -ne $serverProcess) {
        throw "Server exit code unavailable after wait."
    }
    $serverExitCode = 0
} else {
    $serverExitCode = $server.ExitCode
}

if ($serverExitCode -ne 0) {
    throw ("Server exited with code {0}" -f $serverExitCode)
}

Write-Host "IOCP smoke test succeeded."
