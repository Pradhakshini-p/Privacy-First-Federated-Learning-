#!/usr/bin/env python3
"""
Quick Launch Script for Enhanced Dashboard with Pro Tweaks
Run this for impressive demos and presentations!
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the enhanced dashboard"""
    print("🚀 Launching Enhanced Federated Learning Dashboard...")
    print("📋 Features:")
    print("   🎨 Privacy Color Coding (Green→Yellow→Red)")
    print("   📈 Accuracy Sparkline Trends")
    print("   👥 Enhanced Client Status Indicators")
    print("   🎬 Demo Mode for Presentations")
    print()
    
    # Get the dashboard file path
    dashboard_path = Path(__file__).parent / "src" / "enhanced_dashboard_v4.py"
    
    if not dashboard_path.exists():
        print(f"❌ Error: Dashboard not found at {dashboard_path}")
        sys.exit(1)
    
    # Launch Streamlit
    try:
        print(f"🌐 Starting dashboard at: http://localhost:8502")
        print("💡 Tip: Enable 'Demo Mode' in the sidebar for impressive presentations!")
        print()
        
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port", "8502",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ]
        
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching dashboard: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")

if __name__ == "__main__":
    main()
