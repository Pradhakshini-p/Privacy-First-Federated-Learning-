#!/usr/bin/env python3
"""
🔐 Privacy-First Federated Learning Platform
Quick Start Script - Production Deployment
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def main():
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║     🔐 Privacy-First Federated Learning Platform                  ║
    ║     Enterprise Federated Learning with Differential Privacy       ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check dependencies
    print("✓ Checking dependencies...")
    try:
        import fastapi
        import uvicorn
        import torch
        import flwr
        print("  ✓ All dependencies installed")
    except ImportError as e:
        print(f"  ✗ Missing dependency: {e}")
        print("  Install with: pip install -r requirements.txt")
        sys.exit(1)
    
    # Set environment
    os.environ["LOG_LEVEL"] = "INFO"
    os.environ["API_PORT"] = "8000"
    os.environ["ENVIRONMENT"] = "production"
    
    print("\n📋 Configuration:")
    print("  • API Port: 8000")
    print("  • Environment: Production")
    print("  • Frontend: Built-in Dashboard")
    print("  • Backend: FastAPI with async support")
    
    print("\n🚀 Starting application...")
    time.sleep(1)
    
    # Start the application
    cmd = [sys.executable, "app.py"]
    
    print(f"\n💻 Running: {' '.join(cmd)}")
    print("-" * 70)
    
    try:
        # Open browser after a delay
        time.sleep(3)
        print("\n🌐 Opening dashboard in browser...")
        webbrowser.open("http://localhost:8000")
        
        # Run the app
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\n✓ Application stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
