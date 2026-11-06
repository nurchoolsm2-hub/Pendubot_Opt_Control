$ErrorActionPreference = 'Stop'
$appRoot = $PSScriptRoot
$appPython = Join-Path $appRoot '.venv\Scripts\pythonw.exe'
$appUrl = 'http://127.0.0.1:8765/'
function Open-Pendubot {
    try { Start-Process $appUrl }
    catch { Write-Host "The simulation is ready. Open $appUrl in your browser." }
}
if (-not (Test-Path -LiteralPath (Join-Path $appRoot '.venv\pendubot-ready.txt'))) {
    & (Join-Path $appRoot 'Install Pendubot.ps1')
}
$existingApp = $null
try { $existingApp = Invoke-RestMethod -Uri ($appUrl + 'api/health') -TimeoutSec 2 } catch { }
if ($existingApp -and $existingApp.app -eq 'pendubot-control-lab') {
    Open-Pendubot
    exit 0
}
$runtimeDir = Join-Path $appRoot '.runtime'
New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
$appProcess = Start-Process -FilePath $appPython -ArgumentList @('"' + (Join-Path $appRoot 'interface.py') + '"') -WorkingDirectory $appRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtimeDir 'app.log') -RedirectStandardError (Join-Path $runtimeDir 'error.log')
Set-Content -LiteralPath (Join-Path $runtimeDir 'server.pid') -Value $appProcess.Id
for ($attempt = 0; $attempt -lt 80; $attempt++) {
    Start-Sleep -Milliseconds 500
    try {
        $health = Invoke-RestMethod -Uri ($appUrl + 'api/health') -TimeoutSec 1
        if ($health.app -eq 'pendubot-control-lab') {
            Open-Pendubot
            exit 0
        }
    } catch { }
}
throw 'The interface could not start. See .runtime\error.log in the simulation folder.'
