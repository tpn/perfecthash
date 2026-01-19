Param(
    [string]$ReleaseVersion = "",
    [string]$Configuration = $env:CONFIG,
    [string]$Generator = $env:CMAKE_GENERATOR,
    [string]$BuildDir = $env:BUILD_DIR,
    [string]$InstallDir = $env:INSTALL_DIR,
    [string]$StageDir = $env:STAGE_DIR,
    [string]$DistDir = $env:DIST_DIR,
    [switch]$SkipTests,
    [switch]$SkipInstall,
    [switch]$SkipPackage,
    [switch]$Clean,
    [switch]$DryRun,
    [switch]$Help
)

if ($Help) {
    Write-Host @"
Usage: pwsh -File ci/release-windows.ps1 [options]

Options:
  -ReleaseVersion <ver>  Release version (defaults to env/git/CMakeLists)
  -Configuration <cfg>   Build config (default: Release)
  -Generator <gen>       CMake generator (default: Ninja Multi-Config)
  -BuildDir <dir>        Build directory
  -InstallDir <dir>      Install directory
  -StageDir <dir>        Staging directory
  -DistDir <dir>         Artifact output directory
  -SkipTests             Skip ctest
  -SkipInstall           Skip cmake --install
  -SkipPackage           Skip packaging
  -Clean                 Remove the default output base dir before building
  -DryRun                Print commands without executing them
"@
    exit 0
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Resolve-Path (Join-Path $scriptDir "..")

function Invoke-Run {
    param([string[]]$Command)
    if ($DryRun) {
        Write-Host ("+ " + ($Command -join " "))
        return
    }
    & $Command[0] $Command[1..($Command.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $($Command -join ' ')"
    }
}

function Get-DefaultGenerator {
    $help = & cmake --help 2>$null
    if ($help -match "Ninja Multi-Config") {
        return "Ninja Multi-Config"
    }
    if ($help -match "  Ninja") {
        return "Ninja"
    }
    return "Visual Studio 17 2022"
}

function Get-ReleaseVersion {
    param([string]$RootDir)
    if ($ReleaseVersion) { return $ReleaseVersion.Trim().TrimStart("v") }
    if ($env:RELEASE_VERSION) { return $env:RELEASE_VERSION.Trim().TrimStart("v") }
    $line = Get-Content (Join-Path $RootDir "CMakeLists.txt") |
        Select-String -Pattern "^\s*VERSION\s+[0-9]" |
        Select-Object -First 1
    if ($line) {
        $parts = $line.Line -split "\s+"
        $idx = [Array]::IndexOf($parts, "VERSION")
        if ($idx -ge 0 -and ($idx + 1) -lt $parts.Length) {
            return $parts[$idx + 1]
        }
    }
    if (Get-Command git -ErrorAction SilentlyContinue) {
        $tag = & git -C $RootDir describe --tags --abbrev=0 2>$null
        if ($LASTEXITCODE -eq 0 -and $tag) {
            return $tag.Trim().TrimStart("v")
        }
    }
    throw "Unable to determine release version (set RELEASE_VERSION)."
}

function Get-PlatformLabel {
    $arch = $env:PROCESSOR_ARCHITECTURE
    switch -Regex ($arch) {
        "AMD64" { $arch = "x86_64" }
        "ARM64" { $arch = "arm64" }
        default { $arch = $arch.ToLower() }
    }
    return "windows-$arch"
}

function Write-Sha256 {
    param([string]$File)
    if ($DryRun) {
        Write-Host "dry-run: would write checksum for $File"
        return
    }
    $hash = Get-FileHash -Algorithm SHA256 -Path $File
    "$($hash.Hash.ToLower())  $([IO.Path]::GetFileName($File))" |
        Out-File -FilePath ("$File.sha256") -Encoding ASCII
}

if (-not $Configuration) { $Configuration = "Release" }
if (-not $Generator) { $Generator = Get-DefaultGenerator }

$version = Get-ReleaseVersion -RootDir $rootDir
$platform = Get-PlatformLabel
$baseDir = Join-Path $rootDir "out/release/$version/$platform"

if (-not $BuildDir) { $BuildDir = Join-Path $baseDir "build" }
if (-not $InstallDir) { $InstallDir = Join-Path $baseDir "install" }
if (-not $StageDir) { $StageDir = Join-Path $baseDir "stage\perfecthash-$version-$platform" }
if (-not $DistDir) { $DistDir = Join-Path $baseDir "dist" }

if ($Clean) {
    if (Test-Path $baseDir) {
        if (-not $DryRun) {
            Remove-Item -Recurse -Force $baseDir
        } else {
            Write-Host "+ Remove-Item -Recurse -Force $baseDir"
        }
    }
}

Write-Host "release version: $version"
Write-Host "platform: $platform"
Write-Host "build dir: $BuildDir"
Write-Host "install dir: $InstallDir"
Write-Host "dist dir: $DistDir"

Invoke-Run @("cmake", "-S", $rootDir, "-B", $BuildDir,
    "-G", $Generator,
    "-DCMAKE_INSTALL_PREFIX=$InstallDir",
    "-DCMAKE_BUILD_TYPE=$Configuration",
    "-DPERFECTHASH_ENABLE_TESTS=ON",
    "-DBUILD_TESTING=ON")

Invoke-Run @("cmake", "--build", $BuildDir, "--config", $Configuration, "--parallel")

if (-not $SkipTests) {
    $keys = Join-Path $rootDir "keys\HologramWorld-31016.keys"
    if (-not (Test-Path $keys)) {
        throw "missing keys\HologramWorld-31016.keys for tests"
    }
    Invoke-Run @("ctest", "--test-dir", $BuildDir, "--output-on-failure", "-C", $Configuration, "--timeout", "300")
}

if (-not $SkipInstall) {
    Invoke-Run @("cmake", "--install", $BuildDir, "--config", $Configuration)
}

if (-not $SkipPackage) {
    $stageRoot = Split-Path -Parent $StageDir
    if (Test-Path $stageRoot) {
        if (-not $DryRun) {
            Remove-Item -Recurse -Force $stageRoot
        } else {
            Write-Host "+ Remove-Item -Recurse -Force $stageRoot"
        }
    }
    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path $StageDir | Out-Null
    } else {
        Write-Host "+ New-Item -ItemType Directory -Force -Path $StageDir"
    }

    if (-not $DryRun) {
        Copy-Item -Recurse -Force (Join-Path $InstallDir "*") $StageDir
        foreach ($doc in @("README.md", "LICENSE", "USAGE.txt")) {
            $src = Join-Path $rootDir $doc
            if (Test-Path $src) {
                Copy-Item -Force $src $StageDir
            }
        }
        New-Item -ItemType Directory -Force -Path $DistDir | Out-Null
    } else {
        Write-Host "+ Copy-Item -Recurse -Force $InstallDir\\* $StageDir"
        Write-Host "+ Copy-Item -Force README.md LICENSE USAGE.txt -> $StageDir"
        Write-Host "+ New-Item -ItemType Directory -Force -Path $DistDir"
    }

    $packageName = "perfecthash-$version-$platform"
    $zipPath = Join-Path $DistDir "$packageName.zip"
    if (-not $DryRun) {
        if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
        Compress-Archive -Path (Join-Path $stageRoot $packageName) -DestinationPath $zipPath
        Write-Sha256 -File $zipPath
    } else {
        Write-Host "+ Compress-Archive -Path $stageRoot\\$packageName -DestinationPath $zipPath"
        Write-Host "dry-run: would write checksum for $zipPath"
    }
}
