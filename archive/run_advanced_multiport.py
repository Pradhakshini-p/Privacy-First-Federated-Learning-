import streamlit as st
import socket
import time
import subprocess
import sys

def check_port(port):
    """Check if port is available"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('localhost', port))
            return False
    except:
        return True

def find_available_port(start_port=8501, max_port=8600):
    """Find an available port"""
    for port in range(start_port, max_port):
        if check_port(port):
            return port
    return None

def main():
    """Main function to find port and run app"""
    print("🔍 Checking available ports...")
    
    # Find available port
    available_port = find_available_port()
    
    if available_port:
        print(f"✅ Found available port: {available_port}")
        print(f"🚀 Starting federated learning platform on port {available_port}...")
        print(f"🌐 Open: http://localhost:{available_port}")
        print("⏳ Starting application...")
        
        # Run the actual app
        try:
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", 
                "advanced_federated_platform.py",
                f"--server.port={available_port}",
                "--server.address=localhost"
            ], check=True)
        except Exception as e:
            print(f"❌ Error starting app: {e}")
    else:
        print("❌ No available ports found")
        print("💡 Try manually: streamlit run advanced_federated_platform.py --server.port 8501")

if __name__ == "__main__":
    main()
