Write-Host "🚀 Starting Federated Learning Platform..." -ForegroundColor Green
Write-Host "📁 Current directory: $(Get-Location)" -ForegroundColor Yellow
Write-Host "📄 Files in directory:" -ForegroundColor Cyan

# List files
Get-ChildItem | ForEach-Object {
    Write-Host "   📄 $($_.Name)" -ForegroundColor White
}

Write-Host ""
Write-Host "🔗 Starting Streamlit app..." -ForegroundColor Green
Write-Host "🌐 Open: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""

# Run the app
try {
    streamlit run simple_app.py
} catch {
    Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "💡 Try running manually: streamlit run simple_app.py" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
