@echo off
setlocal
cd /d "%~dp0"
echo ================================================
echo       PHISHGUARD - EMAIL DETECTION SYSTEM
echo ================================================
where py >nul 2>nul
if %errorlevel%==0 (
    py -3.14 --version >nul 2>nul
    if %errorlevel%==0 (
        py -3.14 -m venv .venv
    ) else (
        py -m venv .venv
    )
) else (
    python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
    echo Could not create the Python environment. Install Python 3.11-3.14 and try again.
    pause
    exit /b 1
)
echo.
echo Installing required packages. This may take a few minutes...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto install_error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto install_error
echo.
echo Starting PhishGuard...
".venv\Scripts\python.exe" app\app.py
if errorlevel 1 goto run_error
goto end
:install_error
echo.
echo Package installation failed. Check your internet connection and Python version.
pause
exit /b 1
:run_error
echo.
echo The application stopped because of an error. Read the message above.
pause
exit /b 1
:end
pause
