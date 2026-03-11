@echo off
setlocal

REM Run setup + install + execution with one command
powershell -ExecutionPolicy Bypass -File "%~dp0setup_and_run.ps1"

if errorlevel 1 (
    echo.
    echo Script failed. Check the output above.
    exit /b 1
)

echo.
echo Done.
