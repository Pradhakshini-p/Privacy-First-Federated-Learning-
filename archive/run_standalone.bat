@echo off
title Federated Learning Platform - Standalone Version
echo ==========================================
echo.
echo 🔍 Finding available port...
echo.

REM Find available port and start server
python federated_standalone.py

echo.
echo ==========================================
echo 🎉 Platform should be running!
echo 🌐 Check the sidebar for the correct URL
echo.
pause
