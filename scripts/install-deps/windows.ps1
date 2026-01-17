Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Test-Command {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    return [bool](Get-Command -Name $Name -ErrorAction SilentlyContinue)
}

function Ensure-Winget {
    if (-not (Test-Command "winget")) {
        Write-Host "winget not found. Install App Installer from the Microsoft Store, then re-run this script."
        exit 1
    }
}

function Install-Package {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Id
    )

    Ensure-Winget
    winget install -e --id $Id --source winget
}

if (-not (Test-Command "cmake")) {
    Write-Host "Installing CMake..."
    Install-Package -Id "Kitware.CMake"
} else {
    Write-Host "CMake already available."
}

if (-not (Test-Command "nasm")) {
    Write-Host "Installing NASM..."
    Install-Package -Id "NASM.NASM"
} else {
    Write-Host "NASM already available."
}

Write-Host "Done. If new tools were installed, open a new terminal before building."
