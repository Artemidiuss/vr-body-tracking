# One-time setup of FBT on Windows: a Python 3.12 environment in .venv with ONNX Runtime on the
# NVIDIA GPU (CUDA and cuDNN come as pip packages, no CUDA install needed). Run windows\setup.cmd.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

# uv: Python and packages (winget install --id astral-sh.uv -e)
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    $uv = @("$env:LOCALAPPDATA\Microsoft\WinGet\Links\uv.exe") + @(Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter uv.exe -ErrorAction SilentlyContinue | ForEach-Object FullName) |
        Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
}
if (-not $uv) {
    Write-Host "Немає uv. Встанови його командою:  winget install --id astral-sh.uv -e   і запусти setup ще раз."
    exit 1
}

# A .venv copied from a Mac does not run on Windows: keep it aside, do not delete it
if ((Test-Path .venv) -and -not (Test-Path .venv\Scripts\python.exe)) {
    Write-Host "Папка .venv не з Windows (мабуть, скопійована з Mac): перейменовую на .venv-mac"
    Rename-Item .venv .venv-mac
}
if (-not (Test-Path .venv\Scripts\python.exe)) { & $uv venv --python 3.12 .venv; if ($LASTEXITCODE) { exit 1 } }

$env:UV_LINK_MODE = "copy"
& $uv pip install --python .venv\Scripts\python.exe -e ".[dev]" -r windows\requirements-cuda.txt --override windows\overrides.txt
if ($LASTEXITCODE) { exit 1 }

Write-Host "`nПеревіряю відеокарту..."
& .venv\Scripts\python.exe -c "from fbt.pose import make_backend; b = make_backend('rtmpose-m:gpu'); print('Готово, модель на відеокарті:', b.name)"
if ($LASTEXITCODE) { Write-Host "Відеокарта не запрацювала: онови драйвер NVIDIA і запусти setup ще раз."; exit 1 }
& powershell -NoProfile -ExecutionPolicy Bypass -File "$PSScriptRoot\shortcut.ps1"
Write-Host "Запускай FBT ярликом на робочому столі (або windows\FBT.cmd)"
