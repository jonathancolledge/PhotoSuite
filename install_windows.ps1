# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
# Photo Suite - Windows installer for Photo Library, RAW Demosaic and PANOSTITCH
# (run install_windows.bat, or: powershell -ExecutionPolicy Bypass -File install_windows.ps1)
#
# Creates ONE Python environment (.venv, in this folder) for all three apps, uses the NVIDIA GPU when there is
# one (checking that ONNX Runtime really runs on it), and adds desktop shortcuts. Safe to run again: it updates
# the environment in place. Photos, Photo Library's catalog, settings and models are never touched.
#
#   -Cpu          CPU only (no CUDA packages; everything runs, the AI and denoisers slowly)
#   -Models       also download the models (Photo Library's AI, RAW Demosaic's, PANOSTITCH's)
#   -NoShortcut   don't create the desktop shortcuts
#   -Yes          answer yes to questions (installing Python with winget; BM3D's and the face models'
#                 non-commercial licences)
#   -Uninstall    remove the environment and the shortcuts
param([switch]$Cpu, [switch]$Models, [switch]$NoShortcut, [switch]$Yes, [switch]$Uninstall)

# 'Continue': Windows PowerShell 5.1 treats any stderr line from a program (e.g. a pip notice) as an error
# under 'Stop' - failures are caught by checking exit codes instead
$ErrorActionPreference = 'Continue'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $Here '.venv'
$VPy = Join-Path $Venv 'Scripts\python.exe'
$VPyw = Join-Path $Venv 'Scripts\pythonw.exe'
$Desktop = [Environment]::GetFolderPath('Desktop')
$Apps = @(
    @{ Name = 'Photo Library'; Script = 'PhotoLibrary\desktop\photolibrary_desktop.py'; Desc = 'Photo catalog' },
    @{ Name = 'RAW Demosaic'; Script = 'RAWDemosaic\desktop\rawdemosaic_desktop.py'; Desc = 'RAW processing' },
    @{ Name = 'PANOSTITCH'; Script = 'panoramamaker\desktop\panostitch_desktop.py'; Desc = 'Panorama stitcher' })
$TorchIndex = 'https://download.pytorch.org/whl/cu130'
$FallbackOrt = '1.26.0'     # last ONNX Runtime built for CUDA 12: works with older drivers (525+)

function Say($t)  { Write-Host ''; Write-Host "==> $t" -ForegroundColor Yellow }
function Ok($t)   { Write-Host "    OK  $t" -ForegroundColor Green }
function Warn($t) { Write-Host "    !   $t" -ForegroundColor DarkYellow }
function Fail($t) { Write-Host ''; Write-Host "ERROR: $t" -ForegroundColor Red; exit 1 }
function Ask($q) {
    if ($Yes) { return $true }
    $a = Read-Host "    $q [Y/n]"
    return ($a -eq '' -or $a -match '^[Yy]')
}

if ($Uninstall) {
    if (Test-Path $Venv) { Remove-Item -Recurse -Force $Venv }
    foreach ($a in $Apps) { $l = Join-Path $Desktop "$($a.Name).lnk"; if (Test-Path $l) { Remove-Item -Force $l } }
    Write-Host 'Removed the Python environment and the shortcuts. Photos, the catalog, settings and models are untouched.'
    exit 0
}

# ---------------------------------------------------------------- Python
Say 'Checking Python'
function Find-Python {
    # real Pythons only: the "python" in WindowsApps is a stub that opens the Microsoft Store
    $cands = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in '-3.13', '-3.12', '-3.11', '-3') {
            try {
                $exe = (& py $v -c "import sys; print(sys.executable)" 2>$null)
                if ($LASTEXITCODE -eq 0 -and $exe) { $cands += $exe.Trim() }
            } catch {}
        }
    }
    $p = Get-Command python -ErrorAction SilentlyContinue
    if ($p -and $p.Source -notmatch 'WindowsApps') { $cands += $p.Source }
    foreach ($exe in ($cands | Select-Object -Unique)) {
        try {
            $ok = & $exe -c "import sys, struct; print(int(sys.version_info >= (3, 11) and struct.calcsize('P') == 8))" 2>$null
            if ($ok -and $ok.Trim() -eq '1') { return $exe }
        } catch {}
    }
    return $null
}
$Py = Find-Python
if (-not $Py) {
    Warn 'No 64-bit Python 3.11 or newer was found.'
    if ((Get-Command winget -ErrorAction SilentlyContinue) -and (Ask 'Install Python 3.13 now (with winget, from python.org)?')) {
        winget install --id Python.Python.3.13 -e --scope user --accept-package-agreements --accept-source-agreements
        $env:Path = [Environment]::GetEnvironmentVariable('Path', 'User') + ';' + [Environment]::GetEnvironmentVariable('Path', 'Machine')
        $Py = Find-Python
    }
    if (-not $Py) {
        Fail ('Please install Python 3.13 (64-bit) from https://www.python.org/downloads/ - tick "Add python.exe to PATH" ' +
              '- then run this installer again.')
    }
}
Ok ("Python " + (& $Py -c "import sys; print(sys.version.split()[0])") + "  ($Py)")

# long paths: the GPU packages have deep folders (up to ~145 characters inside .venv)
$longOk = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -ErrorAction SilentlyContinue).LongPathsEnabled -eq 1
if (-not $longOk -and $Venv.Length -gt 100) {
    Fail ("This folder's path is too long for Windows' 260-character limit once the GPU packages are inside:`n    $Here`n" +
          "    Move the folder somewhere shorter (e.g. C:\PhotoSuite) and run the installer again.")
}

# ---------------------------------------------------------------- GPU
Say 'Looking for an NVIDIA GPU'
$Gpu = $false
if (-not $Cpu -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    $rows = & nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $rows) {
        $Gpu = $true
        foreach ($r in $rows) { Ok "GPU $r" }
        $drv = [int](($rows | Select-Object -First 1).Split(',')[2].Trim().Split('.')[0])
        if ($drv -lt 580) { Warn "driver $drv is older than 580: PyTorch's CUDA 13 build needs 580+ - update it from nvidia.com, or install with -Cpu" }
    }
}
if (-not $Gpu) { if ($Cpu) { Ok 'CPU only (-Cpu)' } else { Warn 'no NVIDIA GPU found - installing the CPU version (models will be slow)' } }

# ---------------------------------------------------------------- environment + packages
Say "Creating the Python environment in $Venv"
$healthy = $false
if (Test-Path $VPy) { try { & $VPy -c "import sys" 2>$null; $healthy = ($LASTEXITCODE -eq 0) } catch {} }
if ((Test-Path $Venv) -and -not $healthy) {
    Warn 'an environment copied from another PC was found - it can''t work here, recreating it'
    Remove-Item -Recurse -Force $Venv
}
if (-not (Test-Path $VPy)) { & $Py -m venv $Venv; if ($LASTEXITCODE) { Fail 'could not create the environment' }; Ok 'created' } else { Ok 'exists - updating it' }

function Pip { & $VPy -m pip @args --disable-pip-version-check; if ($LASTEXITCODE) { Fail "pip $($args -join ' ') failed (internet connection?)" } }
Pip install --quiet --upgrade pip setuptools wheel
Say 'Installing PyTorch (the first time this downloads ~3 GB with the GPU libraries - be patient)'
if ($Gpu) { Pip install --quiet --upgrade torch torchvision --index-url $TorchIndex }
else { Pip install --quiet --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu }
Ok (& $VPy -c "import torch; print('PyTorch', torch.__version__, '- CUDA' if torch.cuda.is_available() else '- CPU')")
Say 'Installing the other packages'
Pip install --quiet --upgrade -r (Join-Path $Here 'requirements.txt')
$env:SAM2_BUILD_CUDA = '0'       # SAM2's optional CUDA extension needs a compiler; it runs fine without it
Pip install --quiet --upgrade --no-build-isolation sam2
Remove-Item Env:\SAM2_BUILD_CUDA
Ok 'rawpy, Numba, OpenCV, transformers, RawForge, SAM2, PyMaxflow and the rest'
Write-Host '    BM3D (RAW Demosaic''s BM3D denoise option) is free for non-commercial use only - it is not open source.'
if (Ask 'Install BM3D for non-commercial use?') { Pip install --quiet --upgrade bm3d; Ok 'BM3D' }
else { Warn 'BM3D not installed: RAW Demosaic''s BM3D option will not work (everything else does)' }

# Does the GPU build of ONNX Runtime really load and run here? Checked in a separate process so a crash can't stop us.
$probe = @'
import os, sys, glob
import torch
import onnxruntime as o
try: o.preload_dlls()
except Exception: pass
models = sorted(glob.glob(os.path.join(sys.argv[1], 'models', 'panoramamaker', 'raco_aliked_lightglue_pipeline_k*.onnx')))
if models:
    import numpy as np
    s = o.InferenceSession(models[0], providers=['CUDAExecutionProvider'])
    assert s.get_providers()[0] == 'CUDAExecutionProvider', 'CUDA not used'
    s.run(None, {'images': np.random.rand(2, 3, 256, 384).astype(np.float32)})
else:
    assert 'CUDAExecutionProvider' in o.get_available_providers()
'@
$probeFile = Join-Path $env:TEMP 'photosuite_probe.py'
Set-Content -Path $probeFile -Value $probe -Encoding ASCII
function Test-Gpu {
    & $VPy $probeFile $Here *> $null
    return ($LASTEXITCODE -eq 0)
}
function Get-OrtVersion { (& $VPy -c "import onnxruntime as o; print(o.__version__)").Trim() }
if ($Gpu) {
    & $VPy -m pip uninstall --quiet -y onnxruntime 2>$null | Out-Null       # the CPU build would hide the GPU one
    # with its own CUDA and cuDNN packages: PANOSTITCH runs ONNX Runtime without importing torch (whose DLLs it could borrow)
    Pip install --quiet --upgrade "onnxruntime-gpu[cuda,cudnn]"
    if (Test-Gpu) {
        Ok ("ONNX Runtime " + (Get-OrtVersion) + " (GPU)")
    } else {
        Warn ("ONNX Runtime " + (Get-OrtVersion) + " can't use the GPU here - trying $FallbackOrt (CUDA 12)")
        & $VPy -m pip uninstall --quiet -y onnxruntime-gpu 2>$null | Out-Null
        Pip install --quiet "onnxruntime-gpu[cuda,cudnn]==$FallbackOrt"
        if (Test-Gpu) { Ok "ONNX Runtime $FallbackOrt (GPU, CUDA 12) works here" }
        else {
            Warn ('no GPU build of ONNX Runtime works here - using the CPU version. If the GPU should work: update the NVIDIA driver, ' +
                  'install the Visual C++ runtime (https://aka.ms/vs/17/release/vc_redist.x64.exe), then run this again.')
            & $VPy -m pip uninstall --quiet -y onnxruntime-gpu 2>$null | Out-Null
            Pip install --quiet --upgrade onnxruntime
        }
    }
} else {
    & $VPy -m pip uninstall --quiet -y onnxruntime-gpu 2>$null | Out-Null
    Pip install --quiet --upgrade onnxruntime
    Ok ("ONNX Runtime " + (Get-OrtVersion) + " (CPU)")
}
Remove-Item $probeFile -ErrorAction SilentlyContinue

# ---------------------------------------------------------------- models
if ($Models) {
    Say "Downloading Photo Library's AI models (~6 GB)"
    $plArgs = @((Join-Path $Here 'PhotoLibrary\desktop\photolibrary_setup_models.py'))
    if ($Yes) { $plArgs += '--yes' }      # -Yes also accepts the face models' non-commercial licence
    & $VPy @plArgs
    if ($LASTEXITCODE) { Warn 'the download failed - try again from Photo Library: Settings & AI -> Download missing models' }
    Say "Downloading PANOSTITCH's ALIKED + LightGlue models (~450 MB)"
    & $VPy (Join-Path $Here 'panoramamaker\desktop\panostitch_setup_models.py')
    if ($LASTEXITCODE) { Warn 'the download failed - PANOSTITCH matches with SIFT until the models are there' }
    Say "Fetching RAW Demosaic's models"
    & $VPy (Join-Path $Here 'RAWDemosaic\desktop\setup_models.py')
    if ($LASTEXITCODE) { Warn 'RAW Demosaic''s model download failed - run it again later' }
}

# ---------------------------------------------------------------- final check
Say 'Checking the installation'
& $VPy (Join-Path $Here 'check_install.py')
if ($LASTEXITCODE) { Fail 'the final check failed - see the message above' }

# ---------------------------------------------------------------- shortcuts
if (-not $NoShortcut) {
    Say 'Adding desktop shortcuts'
    $ws = New-Object -ComObject WScript.Shell
    foreach ($a in $Apps) {
        $lnk = $ws.CreateShortcut((Join-Path $Desktop "$($a.Name).lnk"))
        $lnk.TargetPath = $VPyw
        $lnk.Arguments = '"' + (Join-Path $Here $a.Script) + '"'
        $lnk.WorkingDirectory = Split-Path -Parent (Join-Path $Here $a.Script)
        $lnk.Description = $a.Desc
        $lnk.Save()
        Ok "$($a.Name).lnk"
    }
}

Say 'Done'
Write-Host '    Start the apps from the desktop shortcuts, or with PhotoLibrary.bat, RAWDemosaic.bat and PANOSTITCH.bat in this folder.'
Write-Host '    To remove the environment again: install_windows.bat -Uninstall'
