$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

$python = ".\.venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
& $python -m pip install --force-reinstall --no-deps torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
& $python -m pip install scikit-learn pandas pillow matplotlib seaborn tqdm
& $python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not found')"

Read-Host "Setup finished. Press Enter to close"
