$ErrorActionPreference = "Stop"

# Resolve project root as the folder containing this script
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$venvDir = Join-Path $projectRoot ".venv"
$pythonExe = Join-Path $venvDir "Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Host "Creating virtual environment..."
    py -m venv .venv
}

$pythonExe = Join-Path $venvDir "Scripts\python.exe"

Write-Host "Upgrading pip..."
& $pythonExe -m pip install --upgrade pip

Write-Host "Installing dependencies from requirements.txt..."
& $pythonExe -m pip install -r requirements.txt

Write-Host "Running optimization script..."
& $pythonExe grid_search_grad_descent_TVBOptim.py
