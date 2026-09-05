@echo off
REM 🔐 Privacy-First Federated Learning Platform - Quick Start

echo 🚀 Privacy-First Federated Learning Platform
echo =============================================
echo.

where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ❌ Python not found. Please install Python 3.8+
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYVER=%%i
echo ✓ Python found: %PYVER%
echo.

echo 📦 Installing dependencies...
pip install -q -r requirements.txt

if %ERRORLEVEL% neq 0 (
    echo ❌ Failed to install dependencies
    exit /b 1
)

echo ✓ Dependencies installed
echo.

echo 🎯 Starting system...
echo    - Backend: Federated Learning Server
echo    - API: http://localhost:8000
echo    - Dashboard: http://localhost:8501
echo.

python main.py %*
