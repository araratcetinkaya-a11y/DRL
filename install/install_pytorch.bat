@echo off

REM Uninstall packages that may break rpc
pip uninstall ipykernel jupyter jupyter-client
IF NOT %ERRORLEVEL% EQU 0 (
    echo Failed to uninstall required packages using pip, they do not exist.
)

REM Install required libraries using pip
pip install -r requirements.txt
IF NOT %ERRORLEVEL% EQU 0 (
    echo Failed to install requirements using pip, cannot continue without them.
    exit /b 1
)

REM Install PyTorch using conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
IF NOT %ERRORLEVEL% EQU 0 (
    echo Failed to install PyTorch using conda, aborting.
    exit /b 1
)

REM Install stable-baselines
pip install stable-baselines3[extra]
IF NOT %ERRORLEVEL% EQU 0 (
    echo Failed to install stable-baselines using pip, aborting.
    exit /b 1
)

REM Install stable-baselines3-contrib
pip install git+https://github.com/DLR-RM/stable-baselines3@feat/gymnasium-support
pip install git+https://github.com/Stable-Baselines-Team/stable-baselines3-contrib@feat/gymnasium-support
IF NOT %ERRORLEVEL% EQU 0 (
    echo Failed to install stable-baselines3-contrib using pip, aborting.
    exit /b 1
)

echo Installation complete.
