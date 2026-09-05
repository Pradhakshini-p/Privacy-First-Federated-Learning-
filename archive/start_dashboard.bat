@echo off
echo 🚀 Starting The Inclusion-Privacy Balance Dashboard...
echo =====================================
echo.

echo 📡 Starting Streamlit server on port 8506...
python -m streamlit run advanced_federated_platform.py --server.port 8506 --server.address localhost

echo.
echo ✅ Dashboard should be running!
echo 🌐 Open: http://localhost:8506
echo 📄 Or open: result_website.html
echo.
pause
