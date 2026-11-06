@echo off
title Pendubot Control Lab
powershell.exe -NoProfile -File "%~dp0Launch Pendubot.ps1"
if errorlevel 1 pause
