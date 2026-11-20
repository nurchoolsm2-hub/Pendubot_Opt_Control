@echo off
setlocal
title Install Pendubot
pushd "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "%~dp0launcher.py" --install
) else (
    where py.exe >nul 2>&1
    if errorlevel 1 (
        python "%~dp0launcher.py" --install
    ) else (
        py -3 "%~dp0launcher.py" --install
    )
)
set "pendubot_status=%errorlevel%"
popd
pause
exit /b %pendubot_status%
