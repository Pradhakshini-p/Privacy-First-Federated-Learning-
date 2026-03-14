@echo off
echo Starting Real Federated Learning Platform...
echo.
echo Please wait while the application starts...
echo.
python -m streamlit run app.py --server.port 8501 --server.address localhost
pause
