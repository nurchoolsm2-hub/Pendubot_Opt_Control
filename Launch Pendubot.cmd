@echo off
setlocal
title Pendubot Control Lab
pushd "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "%~dp0launcher.py" %*
) else (
    where py.exe >nul 2>&1
    if errorlevel 1 (
        python "%~dp0launcher.py" %*
    ) else (
        py -3 "%~dp0launcher.py" %*
    )
)
set "pendubot_status=%errorlevel%"
popd
if not "%pendubot_status%"=="0" pause
exit /b %pendubot_status%
