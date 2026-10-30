<#
Fresh, reproducible setup of the thesis environment (Phase -1, step 5).

Usage (from the folder that contains this script and requirements-thesis.txt):
    powershell -ExecutionPolicy Bypass -File .\setup_thesis_env.ps1
    powershell -ExecutionPolicy Bypass -File .\setup_thesis_env.ps1 -Branch test   # before the branch cleanup

Requirements: Git, Python 3.9 (Windows "py" launcher), NVIDIA driver with CUDA 11.3 support
(the cu113 torch build also runs on CPU if no GPU is available).
#>
param(
    [string]$Branch = "thesis",
    [string]$Target = "DRL-thesis",
    [string]$Repo   = "https://github.com/araratcetinkaya-a11y/DRL.git"
)

$ErrorActionPreference = "Stop"
$Requirements = Join-Path $PSScriptRoot "requirements-thesis.txt"

function Run {
    # Runs a native command and stops the script if it fails
    param([string]$Exe, [string[]]$Arguments)
    Write-Host ">> $Exe $($Arguments -join ' ')" -ForegroundColor Cyan
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit code $LASTEXITCODE): $Exe $($Arguments -join ' ')" }
}

# 0. Pre-checks
if (-not (Test-Path $Requirements)) { throw "requirements-thesis.txt not found next to this script." }
if (Test-Path $Target) { throw "Folder '$Target' already exists. Choose another -Target or remove it." }
Run "py" @("-3.9", "--version")

# 1. Fresh clone, including the Stable-Baselines3 fork submodule
Run "git" @("clone", "--recurse-submodules", "-b", $Branch, $Repo, $Target)
Set-Location $Target
Run "git" @("submodule", "status")

# 2. New virtual environment with Python 3.9
Run "py" @("-3.9", "-m", "venv", ".venv")
$Py = Join-Path (Get-Location) ".venv\Scripts\python.exe"

# 3. Old pip/setuptools/wheel: gym 0.19 has package metadata that newer versions reject
Run $Py @("-m", "pip", "install", "pip<24.1", "setuptools==65.5.0", "wheel==0.38.4")

# 4. airsim's setup needs numpy and msgpack-rpc-python to be installed before it
Run $Py @("-m", "pip", "install", "numpy==1.26.4", "msgpack-rpc-python==0.4.1", "tornado==4.5.3")

# 5. All pinned packages from the training machine
Run $Py @("-m", "pip", "install", "-r", $Requirements)

# 6. Editable local packages. --no-deps keeps the pinned versions from step 5 unchanged.
Run $Py @("-m", "pip", "install", "--no-deps", "-e", ".\stable-baselines3")
Run $Py @("-m", "pip", "install", "--no-deps", "-e", ".\gym_env")

# 7. Sanity check: versions and where SB3 is loaded from
Run $Py @("-c", "import sys, torch, gym, airsim, stable_baselines3 as sb3; print('python', sys.version.split()[0]); print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available()); print('gym', gym.__version__); print('sb3', sb3.__version__, sb3.__file__)")

# 8. Record the exact environment (ASCII, so pip can read it back later)
& $Py -m pip freeze | Out-File -Encoding ascii "requirements-thesis-lock.txt"
Write-Host "Done. Environment recorded in $Target\requirements-thesis-lock.txt" -ForegroundColor Green
Write-Host "Activate with: $Target\.venv\Scripts\Activate.ps1"
