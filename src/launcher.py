"""
Comprehensive Federated Learning Launcher
Coordinates server, clients, and API for complete system deployment
"""

import argparse
import asyncio
import json
import logging
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import threading
import requests
import psutil
import os

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config import *
from federated_backend import create_backend
from api_server import app
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_DIR / "launcher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FederatedLearningLauncher:
    """Comprehensive launcher for federated learning system"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the launcher"""
        self.config = config or {}
        self.processes: Dict[str, subprocess.Popen] = {}
        self.backend = None
        self.api_server = None
        self.running = False
        
        # System configuration
        self.num_clients = self.config.get("num_clients", NUM_CLIENTS)
        self.rounds = self.config.get("rounds", ROUNDS)
        self.start_api = self.config.get("start_api", True)
        self.start_dashboard = self.config.get("start_dashboard", True)
        
        # Ports
        self.server_port = self.config.get("server_port", SERVER_PORT)
        self.api_port = self.config.get("api_port", 8000)
        self.dashboard_port = self.config.get("dashboard_port", 8502)
        
        logger.info("Federated Learning Launcher initialized")
        logger.info(f"Clients: {self.num_clients}, Rounds: {self.rounds}")
        logger.info(f"API: {self.start_api}, Dashboard: {self.start_dashboard}")
    
    def check_dependencies(self) -> bool:
        """Check if all dependencies are available"""
        try:
            import torch
            import flwr
            import fastapi
            import uvicorn
            import streamlit
            
            logger.info("✓ All dependencies available")
            return True
            
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            return False
    
    def setup_environment(self) -> bool:
        """Setup the environment for federated learning"""
        try:
            # Create necessary directories
            LOG_DIR.mkdir(exist_ok=True)
            DATA_DIR.mkdir(exist_ok=True)
            
            # Check data availability
            data_path = DATA_CONFIG.get("data_path")
            if data_path and not Path(data_path).exists():
                logger.warning(f"Data file not found: {data_path}")
                logger.info("Please ensure data is available before starting training")
            
            logger.info("✓ Environment setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up environment: {e}")
            return False
    
    def start_backend_server(self) -> bool:
        """Start the federated learning backend server"""
        try:
            logger.info("Starting backend server...")
            
            # Create backend instance
            self.backend = create_backend(self.config)
            
            # Start backend in background thread
            def run_backend():
                try:
                    self.backend.start_training(self.rounds)
                except Exception as e:
                    logger.error(f"Backend error: {e}")
            
            backend_thread = threading.Thread(target=run_backend)
            backend_thread.daemon = True
            backend_thread.start()
            
            # Wait a bit for server to start
            time.sleep(3)
            
            logger.info("✓ Backend server started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting backend server: {e}")
            return False
    
    def start_api_server(self) -> bool:
        """Start the API server"""
        try:
            logger.info("Starting API server...")
            
            # Start API server in background process
            api_cmd = [
                sys.executable, "-m", "uvicorn", 
                "api_server:app",
                "--host", "0.0.0.0",
                "--port", str(self.api_port),
                "--log-level", "info"
            ]
            
            api_process = subprocess.Popen(
                api_cmd,
                cwd=Path(__file__).parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.processes["api"] = api_process
            
            # Wait for API server to start
            time.sleep(5)
            
            # Check if API server is responding
            try:
                response = requests.get(f"http://localhost:{self.api_port}/health", timeout=5)
                if response.status_code == 200:
                    logger.info("✓ API server started successfully")
                    return True
                else:
                    logger.error(f"API server returned status {response.status_code}")
                    return False
            except requests.exceptions.RequestException as e:
                logger.error(f"API server not responding: {e}")
                return False
            
        except Exception as e:
            logger.error(f"Error starting API server: {e}")
            return False
    
    def start_clients(self) -> bool:
        """Start federated learning clients"""
        try:
            logger.info(f"Starting {self.num_clients} clients...")
            
            for i in range(1, self.num_clients + 1):
                client_id = f"client_{i}"
                silo_id = f"silo_{i}"
                
                # Client command
                client_cmd = [
                    sys.executable, "enhanced_client.py",
                    client_id,
                    "--silo", silo_id,
                    "--server", f"localhost:{self.server_port}"
                ]
                
                # Add privacy flag based on config
                if CLIENT_CONFIGS.get(client_id, {}).get("privacy_enabled", True):
                    client_cmd.append("--privacy")
                else:
                    client_cmd.append("--no-privacy")
                
                # Start client process
                client_process = subprocess.Popen(
                    client_cmd,
                    cwd=Path(__file__).parent,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                self.processes[client_id] = client_process
                
                # Small delay between client starts
                time.sleep(2)
                
                logger.info(f"✓ Client {client_id} started")
            
            logger.info(f"✓ All {self.num_clients} clients started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting clients: {e}")
            return False
    
    def start_dashboard(self) -> bool:
        """Start the Streamlit dashboard"""
        try:
            logger.info("Starting dashboard...")
            
            # Dashboard command
            dashboard_cmd = [
                sys.executable, "-m", "streamlit", "run",
                "enhanced_dashboard_v4.py",
                "--server.port", str(self.dashboard_port),
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false"
            ]
            
            dashboard_process = subprocess.Popen(
                dashboard_cmd,
                cwd=Path(__file__).parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.processes["dashboard"] = dashboard_process
            
            # Wait for dashboard to start
            time.sleep(5)
            
            logger.info("✓ Dashboard started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")
            return False
    
    def monitor_processes(self):
        """Monitor all running processes"""
        while self.running:
            try:
                for name, process in self.processes.items():
                    if process.poll() is not None:
                        logger.warning(f"Process {name} has stopped with return code {process.returncode}")
                        
                        # Restart critical processes
                        if name == "api" and self.start_api:
                            logger.info("Restarting API server...")
                            self.start_api_server()
                        elif name == "dashboard" and self.start_dashboard:
                            logger.info("Restarting dashboard...")
                            self.start_dashboard()
                        elif name.startswith("client_"):
                            logger.info(f"Restarting client {name}...")
                            # Restart client logic here
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring processes: {e}")
                time.sleep(5)
    
    def start_system(self) -> bool:
        """Start the complete federated learning system"""
        try:
            logger.info("🚀 Starting Federated Learning System")
            logger.info("=" * 50)
            
            # Check dependencies
            if not self.check_dependencies():
                return False
            
            # Setup environment
            if not self.setup_environment():
                return False
            
            self.running = True
            
            # Start components in order
            if not self.start_backend_server():
                return False
            
            if self.start_api and not self.start_api_server():
                return False
            
            if not self.start_clients():
                return False
            
            if self.start_dashboard and not self.start_dashboard():
                return False
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=self.monitor_processes)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # Print system status
            self.print_system_status()
            
            logger.info("🎉 Federated Learning System started successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error starting system: {e}")
            return False
    
    def print_system_status(self):
        """Print current system status"""
        print("\n" + "=" * 60)
        print("🔐 PRIVACY-FIRST FEDERATED LEARNING SYSTEM")
        print("=" * 60)
        print(f"📊 System Status: RUNNING")
        print(f"🤖 Clients: {self.num_clients}")
        print(f"🔄 Training Rounds: {self.rounds}")
        print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n🌐 Access URLs:")
        print(f"   📈 Dashboard: http://localhost:{self.dashboard_port}")
        print(f"   🔌 API Server: http://localhost:{self.api_port}")
        print(f"   📚 API Docs: http://localhost:{self.api_port}/docs")
        print(f"   🔍 Health Check: http://localhost:{self.api_port}/health")
        print("\n💡 Commands:")
        print("   • Press Ctrl+C to stop the system")
        print("   • Check logs in logs/ directory")
        print("   • Monitor training in the dashboard")
        print("=" * 60)
    
    def stop_system(self):
        """Stop all running processes"""
        logger.info("🛑 Stopping Federated Learning System...")
        
        self.running = False
        
        # Stop all processes
        for name, process in self.processes.items():
            try:
                logger.info(f"Stopping {name}...")
                process.terminate()
                
                # Wait for process to stop
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {name}...")
                    process.kill()
                    process.wait()
                
                logger.info(f"✓ {name} stopped")
                
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
        
        # Clear processes
        self.processes.clear()
        
        logger.info("🎯 System stopped successfully")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop_system()
        sys.exit(0)

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description="Privacy-First Federated Learning Launcher")
    
    # System configuration
    parser.add_argument("--clients", type=int, default=NUM_CLIENTS, help="Number of clients")
    parser.add_argument("--rounds", type=int, default=ROUNDS, help="Number of training rounds")
    parser.add_argument("--no-api", action="store_true", help="Don't start API server")
    parser.add_argument("--no-dashboard", action="store_true", help="Don't start dashboard")
    
    # Port configuration
    parser.add_argument("--server-port", type=int, default=SERVER_PORT, help="FL server port")
    parser.add_argument("--api-port", type=int, default=8000, help="API server port")
    parser.add_argument("--dashboard-port", type=int, default=8502, help="Dashboard port")
    
    # Privacy configuration
    parser.add_argument("--privacy", action="store_true", default=True, help="Enable privacy")
    parser.add_argument("--no-privacy", action="store_true", help="Disable privacy")
    parser.add_argument("--epsilon", type=float, default=EPSILON, help="Privacy epsilon")
    parser.add_argument("--noise-multiplier", type=float, default=NOISE_MULTIPLIER, help="Noise multiplier")
    
    # Mode selection
    parser.add_argument("--mode", choices=["server", "client", "api", "dashboard", "all"], 
                       default="all", help="What to start")
    
    args = parser.parse_args()
    
    # Create launcher configuration
    config = {
        "num_clients": args.clients,
        "rounds": args.rounds,
        "start_api": not args.no_api,
        "start_dashboard": not args.no_dashboard,
        "server_port": args.server_port,
        "api_port": args.api_port,
        "dashboard_port": args.dashboard_port,
        "privacy_enabled": args.privacy and not args.no_privacy,
        "epsilon": args.epsilon,
        "noise_multiplier": args.noise_multiplier
    }
    
    # Create and start launcher
    launcher = FederatedLearningLauncher(config)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, launcher.signal_handler)
    signal.signal(signal.SIGTERM, launcher.signal_handler)
    
    try:
        if args.mode == "all":
            # Start complete system
            if launcher.start_system():
                # Keep running
                while launcher.running:
                    time.sleep(1)
            else:
                logger.error("Failed to start system")
                sys.exit(1)
        
        elif args.mode == "server":
            # Start only backend server
            launcher.start_backend_server()
            while launcher.running:
                time.sleep(1)
        
        elif args.mode == "api":
            # Start only API server
            launcher.start_api_server()
            while launcher.running:
                time.sleep(1)
        
        elif args.mode == "dashboard":
            # Start only dashboard
            launcher.start_dashboard()
            while launcher.running:
                time.sleep(1)
        
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        launcher.stop_system()

if __name__ == "__main__":
    main()
