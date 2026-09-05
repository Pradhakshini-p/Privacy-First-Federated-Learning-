@echo off
echo 🔍 Smart Federated Learning Platform Launcher
echo =====================================
echo.
echo 📡 Checking available ports...
echo.

REM Try different ports and find one that works
echo 🚀 Attempting port 8501...
python -m streamlit run advanced_federated_platform.py --server.port 8501 --server.address localhost
if %ERRORLEVEL% EQU 0 (
    echo ✅ SUCCESS! Platform running on port 8501
    echo 🌐 Open: http://localhost:8501
    goto success
)

echo.
echo ❌ Port 8501 failed, trying port 8502...
python -m streamlit run advanced_federated_platform.py --server.port 8502 --server.address localhost
if %ERRORLEVEL% EQU 0 (
    echo ✅ SUCCESS! Platform running on port 8502
    echo 🌐 Open: http://localhost:8502
    goto success
)

echo.
echo ❌ Port 8502 failed, trying port 8503...
python -m streamlit run advanced_federated_platform.py --server.port 8503 --server.address localhost
if %ERRORLEVEL% EQU 0 (
    echo ✅ SUCCESS! Platform running on port 8503
    echo 🌐 Open: http://localhost:8503
    goto success
)

echo.
echo ❌ Port 8503 failed, trying port 8504...
python -m streamlit run advanced_federated_platform.py --server.port 8504 --server.address localhost
if %ERRORLEVEL% EQU 0 (
    echo ✅ SUCCESS! Platform running on port 8504
    echo 🌐 Open: http://localhost:8504
    goto success
)

echo.
echo ❌ Port 8504 failed, trying port 8505...
python -m streamlit run advanced_federated_platform.py --server.port 8505 --server.address localhost
if %ERRORLEVEL% EQU 0 (
    echo ✅ SUCCESS! Platform running on port 8505
    echo 🌐 Open: http://localhost:8505
    goto success
)

echo.
echo 🔥 All common ports failed!
echo 💡 Manual troubleshooting steps:
echo    1. Check Windows Firewall
echo    2. Check Antivirus software
echo    3. Try: python -m streamlit run advanced_federated_platform.py --server.address 0.0.0.0
echo    4. Use different port: --server.port 8080
echo.

:success
echo.
echo 🎉 Platform should be running!
echo 🌐 Check your browser for the URL shown above
echo.
pause
