Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Test-Command {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    return [bool](Get-Command -Name $Name -ErrorAction SilentlyContinue)
}

function Resolve-MambaEnvFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$EnvFile
    )

    if ($env:WITH_RUST -eq "1") {
        return $EnvFile
    }

    $content = Get-Content -Path $EnvFile
    $filtered = $content | Where-Object { $_ -notmatch '^\s*-\s*rust(\s|$)' }
    if ($filtered.Count -eq $content.Count) {
        return $EnvFile
    }

    $tempFile = Join-Path $env:TEMP ("perfecthash-win-env-{0}.yaml" -f ([guid]::NewGuid().ToString("N")))
    Set-Content -Path $tempFile -Value $filtered -Encoding ASCII
    return $tempFile
}

function Get-MambaExe {
    if ($env:MAMBA_EXE -and (Test-Path -Path $env:MAMBA_EXE)) {
        return $env:MAMBA_EXE
    }

    if (Test-Command "mamba") {
        return (Get-Command -Name "mamba").Source
    }

    $candidates = @(
        "$env:LOCALAPPDATA\\mambaforge\\Scripts\\mamba.exe",
        "$env:LOCALAPPDATA\\Mambaforge\\Scripts\\mamba.exe",
        "$env:LOCALAPPDATA\\miniforge3\\Scripts\\mamba.exe",
        "$env:USERPROFILE\\mambaforge\\Scripts\\mamba.exe",
        "$env:LOCALAPPDATA\\miniconda3\\Scripts\\mamba.exe",
        "$env:USERPROFILE\\miniconda3\\Scripts\\mamba.exe",
        "$env:LOCALAPPDATA\\miniconda3\\Library\\bin\\mamba.exe",
        "$env:USERPROFILE\\miniconda3\\Library\\bin\\mamba.exe",
        "$env:LOCALAPPDATA\\mambaforge\\Library\\bin\\mamba.exe",
        "$env:LOCALAPPDATA\\miniforge3\\Library\\bin\\mamba.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -Path $candidate) {
            return $candidate
        }
    }

    return $null
}

function Invoke-DownloadWithRetry {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Uri,
        [Parameter(Mandatory = $true)]
        [string]$OutFile,
        [int]$Retries = 3
    )

    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

    for ($attempt = 1; $attempt -le $Retries; $attempt++) {
        try {
            $ProgressPreference = "SilentlyContinue"
            if (Test-Command "curl.exe") {
                & curl.exe -L --retry 3 --retry-delay 2 -o $OutFile $Uri
                if ($LASTEXITCODE -eq 0) {
                    return
                }
            }
            Invoke-WebRequest -Uri $Uri -OutFile $OutFile -Headers @{ "User-Agent" = "Mozilla/5.0" }
            return
        } catch {
            if (Test-Command "Start-BitsTransfer") {
                try {
                    Start-BitsTransfer -Source $Uri -Destination $OutFile
                    return
                } catch {
                    # Fall through to retry/backoff.
                }
            }
            if ($attempt -eq $Retries) {
                throw
            }
            Start-Sleep -Seconds (2 * $attempt)
        }
    }
}

function Install-Mambaforge {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InstallDir
    )

    if ($InstallDir -match "\\s") {
        Write-Host "MAMBAFORGE_DIR must not contain spaces: $InstallDir"
        exit 1
    }

    $installerName = "Mambaforge-Windows-x86_64.exe"
    $installerFile = "Mambaforge-Windows-x86_64-$([guid]::NewGuid().ToString('N')).exe"
    $installerPath = Join-Path $env:TEMP $installerFile
    $url = "https://github.com/conda-forge/miniforge/releases/latest/download/$installerName"

    Write-Host "Downloading Mambaforge..."
    Invoke-DownloadWithRetry -Uri $url -OutFile $installerPath
    if ((Get-Item -Path $installerPath).Length -lt 10000000) {
        Remove-Item -Force -Path $installerPath
        throw "Mambaforge download failed or was blocked."
    }

    Write-Host "Installing Mambaforge to $InstallDir..."
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    Start-Process -FilePath $installerPath -ArgumentList "/S /D=$InstallDir" -Wait -NoNewWindow
    if (Test-Path -Path $installerPath) {
        Remove-Item -Force -Path $installerPath
    }
}

function Install-Miniconda {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InstallDir
    )

    if ($InstallDir -match "\\s") {
        Write-Host "MINICONDA_DIR must not contain spaces: $InstallDir"
        exit 1
    }

    $condaExe = Join-Path $InstallDir "Scripts\\conda.exe"
    if (-not (Test-Path -Path $condaExe)) {

    $installerName = "Miniconda3-latest-Windows-x86_64.exe"
    $installerFile = "Miniconda3-latest-Windows-x86_64-$([guid]::NewGuid().ToString('N')).exe"
    $installerPath = Join-Path $env:TEMP $installerFile
    $url = "https://repo.anaconda.com/miniconda/$installerName"

    Write-Host "Downloading Miniconda..."
    Invoke-DownloadWithRetry -Uri $url -OutFile $installerPath
    if ((Get-Item -Path $installerPath).Length -lt 10000000) {
        Remove-Item -Force -Path $installerPath
        throw "Miniconda download failed or was blocked."
    }

    Write-Host "Installing Miniconda to $InstallDir..."
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    Start-Process -FilePath $installerPath -ArgumentList "/S /D=$InstallDir" -Wait -NoNewWindow
    if (Test-Path -Path $installerPath) {
        Remove-Item -Force -Path $installerPath
    }

    if (-not (Test-Path -Path $condaExe)) {
        throw "conda.exe not found after Miniconda install."
    }
    }

    Write-Host "Installing mamba into Miniconda base..."
    & $condaExe tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    & $condaExe tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    & $condaExe tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
    & $condaExe install -y -n base --override-channels -c conda-forge mamba
}

function Ensure-Mamba {
    $mambaExe = Get-MambaExe
    if ($mambaExe) {
        Write-Host "mamba already available."
        return $mambaExe
    }

    $installDir = if ($env:MAMBAFORGE_DIR) { $env:MAMBAFORGE_DIR } else { "$env:LOCALAPPDATA\\mambaforge" }
    try {
        Install-Mambaforge -InstallDir $installDir
    } catch {
        Write-Host "Mambaforge install failed, falling back to Miniconda."
        $miniDir = if ($env:MINICONDA_DIR) { $env:MINICONDA_DIR } else { "$env:LOCALAPPDATA\\miniconda3" }
        Install-Miniconda -InstallDir $miniDir
    }

    $mambaExe = Get-MambaExe
    if (-not $mambaExe) {
        Write-Host "mamba not found after install. Re-run with MAMBAFORGE_DIR set to the correct path."
        exit 1
    }

    $rootPrefix = Get-MambaRootPrefix -MambaExe $mambaExe
    $env:PATH = "$rootPrefix\\Scripts;$rootPrefix\\Library\\bin;$rootPrefix;$env:PATH"
    $env:MAMBA_ROOT_PREFIX = $rootPrefix
    $env:MAMBA_EXE = $mambaExe

    return $mambaExe
}

function Get-MambaRootPrefix {
    param(
        [Parameter(Mandatory = $true)]
        [string]$MambaExe
    )

    $info = & $MambaExe info --json | ConvertFrom-Json
    if ($info.PSObject.Properties.Name -contains "root_prefix") {
        return $info.root_prefix
    }
    if ($info.PSObject.Properties.Name -contains "base environment") {
        return $info."base environment"
    }
    if ($info.PSObject.Properties.Name -contains "env location") {
        return $info."env location"
    }
    throw "Unable to determine mamba root prefix."
}

function Ensure-MambaEnv {
    param(
        [Parameter(Mandatory = $true)]
        [string]$MambaExe,
        [Parameter(Mandatory = $true)]
        [string]$EnvName,
        [Parameter(Mandatory = $true)]
        [string]$EnvFile
    )

    if (-not (Test-Path -Path $EnvFile)) {
        Write-Host "Env file not found: $EnvFile"
        exit 1
    }

    $rootPrefix = Get-MambaRootPrefix -MambaExe $MambaExe
    $envPath = Join-Path $rootPrefix "envs\\$EnvName"

    if (Test-Path -Path $envPath) {
        Write-Host "Updating mamba env $EnvName..."
        & $MambaExe env update -n $EnvName -f $EnvFile --prune -y --no-rc --no-env | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "mamba env update failed."
        }
    } else {
        Write-Host "Creating mamba env $EnvName..."
        & $MambaExe env create -n $EnvName -f $EnvFile -y --no-rc --no-env | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "mamba env create failed."
        }
    }

    return $envPath
}

$mambaExe = Ensure-Mamba
$rootPrefix = Get-MambaRootPrefix -MambaExe $mambaExe
$envName = if ($env:PH_MAMBA_ENV) { $env:PH_MAMBA_ENV } else { "perfecthash-win" }
$envFile = Join-Path $PSScriptRoot "windows-mamba-env.yaml"
$resolvedEnvFile = Resolve-MambaEnvFile -EnvFile $envFile
$envPath = Ensure-MambaEnv -MambaExe $mambaExe -EnvName $envName -EnvFile $resolvedEnvFile
if ($resolvedEnvFile -ne $envFile -and (Test-Path -Path $resolvedEnvFile)) {
    Remove-Item -Force -Path $resolvedEnvFile
}

$llvmDir = Join-Path $envPath "Library\\lib\\cmake\\llvm"
if (-not (Test-Path -Path (Join-Path $llvmDir "LLVMConfig.cmake"))) {
    Write-Host "LLVMConfig.cmake not found in env. Ensure llvmdev is installed."
    exit 1
}

$env:LLVM_DIR = $llvmDir
Write-Host "LLVM_DIR set to $llvmDir"

Write-Host "To use this environment:"
Write-Host "  $mambaExe run -n $envName <cmd>"
Write-Host ("  Or run: {0}\\Scripts\\activate then mamba activate {1}" -f $rootPrefix, $envName)
Write-Host "Done. Open a new terminal before building if PATH changes are needed."
