param(
    [string]$BuildDir = "build-win",
    [string]$Config = "Debug",
    [string]$KeysDir = "..\\perfecthash-keys\\sys32",
    [string]$OutputDir = "build-win\\stress-sys32",
    [string]$Algorithm = "Chm01",
    [string]$HashFunction = "Mulshrolate4RX",
    [string]$MaskFunction = "And",
    [int]$MaximumConcurrency = 0,
    [string[]]$ExtraArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$exe = Join-Path $BuildDir ("bin\\{0}\\PerfectHashBulkCreate.exe" -f $Config)

if (-not (Test-Path $exe)) {
    throw "Bulk create exe not found: $exe"
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

$args = @($keysPath,
          $outputPath,
          $Algorithm,
          $HashFunction,
          $MaskFunction,
          $MaximumConcurrency)

if ($ExtraArgs -and $ExtraArgs.Count -gt 0) {
    $args += $ExtraArgs
}

& $exe @args
if ($LASTEXITCODE -ne 0) {
    throw ("Bulk create exited with code {0}" -f $LASTEXITCODE)
}

Write-Host "Stress test complete."
