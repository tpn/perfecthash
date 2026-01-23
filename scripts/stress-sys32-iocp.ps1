param(
    [string]$BuildDir = "build-win",
    [string]$Config = "Debug",
    [string]$Endpoint = "\\.\\pipe\\PerfectHashServer-StressSys32",
    [string]$KeysDir = "..\\perfecthash-keys\\sys32",
    [string]$OutputDir = "build-win\\iocp-stress-sys32",
    [string]$Algorithm = "Chm01",
    [string]$HashFunction = "Mulshrolate4RX",
    [string]$MaskFunction = "And",
    [int]$MaximumConcurrency = 0,
    [int]$IocpConcurrency = 0,
    [int]$MaxThreads = 0,
    [string[]]$ExtraArgs = @(),
    [int]$TimeoutSeconds = 300,
    [bool]$WaitForServer = $true,
    [int]$ConnectTimeoutMs = 10000,
    [bool]$NoFileIo = $true,
    [bool]$VerboseServer = $false
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Remove-Item Env:PH_LOG_BULK_CREATE_COUNTS -ErrorAction SilentlyContinue
Remove-Item Env:PH_LOG_CHM01_ASYNC_JOB -ErrorAction SilentlyContinue

$serverExe = Join-Path $BuildDir ("bin\\{0}\\PerfectHashServer.exe" -f $Config)
$clientExe = Join-Path $BuildDir ("bin\\{0}\\PerfectHashClient.exe" -f $Config)

if (-not (Test-Path $serverExe)) {
    throw "Server exe not found: $serverExe"
}

if (-not (Test-Path $clientExe)) {
    throw "Client exe not found: $clientExe"
}

if (-not (Test-Path $KeysDir)) {
    throw "Keys dir not found: $KeysDir"
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

$keysPath = (Resolve-Path $KeysDir).Path
$outputPath = (Resolve-Path $OutputDir).Path

if ($MaximumConcurrency -le 0) {
    $MaximumConcurrency = [Environment]::ProcessorCount
}

$bulkCommand = ('"{0}" "{1}" {2} {3} {4} {5}' -f $keysPath,
                                                    $outputPath,
                                                    $Algorithm,
                                                    $HashFunction,
                                                    $MaskFunction,
                                                    $MaximumConcurrency)

if ($ExtraArgs -and $ExtraArgs.Count -gt 0) {
    $bulkCommand += " " + ($ExtraArgs -join " ")
}

$bulkArg = "--BulkCreateDirectory=$bulkCommand"

$serverArgs = @("--Endpoint=$Endpoint")
if ($IocpConcurrency -gt 0) {
    $serverArgs += ("--IocpConcurrency={0}" -f $IocpConcurrency)
}
if ($MaxThreads -gt 0) {
    $serverArgs += ("--MaxThreads={0}" -f $MaxThreads)
}
if ($NoFileIo) {
    $serverArgs += "--NoFileIo"
}
if ($VerboseServer) {
    $serverArgs += "--Verbose"
}

$server = Start-Process -FilePath $serverExe `
                        -ArgumentList $serverArgs `
                        -PassThru `
                        -NoNewWindow

$clientArgs = @("--Endpoint=$Endpoint")
if ($WaitForServer) {
    $clientArgs += "--WaitForServer"
    if ($ConnectTimeoutMs -gt 0) {
        $clientArgs += ("--ConnectTimeout={0}" -f $ConnectTimeoutMs)
    }
}

$clientCreateArgs = @($clientArgs + "`"$bulkArg`"")
$clientCreate = Start-Process -FilePath $clientExe `
                              -ArgumentList $clientCreateArgs `
                              -PassThru `
                              -NoNewWindow `
                              -Wait

$bulkSuccess = 0x2004000F
if ($clientCreate.ExitCode -ne 0 -and $clientCreate.ExitCode -ne $bulkSuccess) {
    throw ("Client bulk-create exited with code {0}" -f $clientCreate.ExitCode)
}

$clientShutdownArgs = @($clientArgs + "--Shutdown")
$clientShutdown = Start-Process -FilePath $clientExe `
                                -ArgumentList $clientShutdownArgs `
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

Write-Host "IOCP stress test complete."
