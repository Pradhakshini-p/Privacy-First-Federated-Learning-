#!/usr/bin/env python3
"""
🔐 Privacy-First Federated Learning Platform
Main Entry Point - Integrated Backend & Frontend System

This script coordinates the complete federated learning system:
- Backend FL Server with privacy controls
- API Server for real-time monitoring
- Streamlit Dashboard for visualization
- WebSocket for live data streaming
"""

import argparse
import asyncio
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemManager:
    """Manages complete federated learning system"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.processes = {}
        self.running = False
        
    def check_dependencies(self) -> bool:
        """Verify all required packages are installed"""
        required = ['torch', 'flwr', 'fastapi', 'uvicorn', 'streamlit']
        missing = []
        
        for package in required:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        
        if missing:
            logger.error(f"Missing packages: {', '.join(missing)}")
            logger.info(f"Install with: pip install -r requirements.txt")
            return False
        
        return True
    
    def start_api_server(self, port: int = 8000) -> bool:
        """Start FastAPI server with real backend connection"""
        try:
            logger.info(f"🚀 Starting API Server on port {port}...")
            
            cmd = [
                sys.executable, "-m", "uvicorn",
                "src.api_server:app",
                "--host", "0.0.0.0",
                "--port", str(port),
                "--reload"
            ]
            
            process = subprocess.Popen(
                cmd,
                cwd=Path(__file__).parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            self.processes['api'] = process
            time.sleep(2)
            
            logger.info(f"✅ API Server running at http://localhost:{port}")
            logger.info(f"📖 Docs available at http://localhost:{port}/docs")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start API Server: {e}")
            return False
    
    def start_dashboard(self, port: int = 8501) -> bool:
        """Start Streamlit dashboard"""
        try:
            logger.info(f"🖥️  Starting Dashboard on port {port}...")
            
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                "src/enhanced_dashboard_v4.py",
                "--server.port", str(port),
                "--server.address", "localhost"
            ]
            
            process = subprocess.Popen(
                cmd,
                cwd=Path(__file__).parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            self.processes['dashboard'] = process
            time.sleep(3)
            
            logger.info(f"✅ Dashboard running at http://localhost:{port}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start Dashboard: {e}")
            return False
    
    def start_backend(self) -> bool:
        """Initialize and start the backend training"""
        try:
            logger.info("🤖 Initializing Federated Learning Backend...")
            
            # Import here to avoid issues if dependencies missing
            from src.federated_backend import create_backend
            
            config = {
                "num_clients": self.config.get("clients", 3),
                "rounds": self.config.get("rounds", 5),
                "model_type": self.config.get("model", "mlp"),
            }
            
            backend = create_backend(config)
            
            # Start training in background thread
            def run_training():
                try:
                    logger.info(f"📊 Starting training with {config['num_clients']} clients for {config['rounds']} rounds...")
                    backend.start_training(config['rounds'])
                except Exception as e:
                    logger.error(f"Backend training error: {e}")
            
            training_thread = threading.Thread(target=run_training, daemon=True)
            training_thread.start()
            
            logger.info("✅ Backend initialized and training started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start Backend: {e}")
            logger.info("⚠️  Backend may require additional setup - check src/federated_backend.py")
            return False
    
    def start_all(self, args) -> bool:
        """Start all system components"""
        logger.info("=" * 60)
        logger.info("🔐 Privacy-First Federated Learning Platform")
        logger.info("=" * 60)
        
        if not self.check_dependencies():
            return False
        
        # Start components in order
        components_started = []
        
        if not args.skip_backend:
            if self.start_backend():
                components_started.append("Backend")
            time.sleep(1)
        
        if self.start_api_server(args.api_port):
            components_started.append("API Server")
        
        if not args.skip_dashboard:
            if self.start_dashboard(args.dashboard_port):
                components_started.append("Dashboard")
        
        logger.info("=" * 60)
        logger.info(f"✅ Started: {', '.join(components_started)}")
        logger.info("=" * 60)
        logger.info("")
        logger.info("🌐 Access URLs:")
        logger.info(f"   Dashboard:     http://localhost:{args.dashboard_port}")
        logger.info(f"   API Docs:      http://localhost:{args.api_port}/docs")
        logger.info(f"   API Health:    http://localhost:{args.api_port}/health")
        logger.info("")
        logger.info("Press Ctrl+C to stop all services...")
        logger.info("")
        
        return True
    
    def cleanup(self):
        """Clean up running processes"""
        logger.info("Shutting down services...")
        for name, process in self.processes.items():
            try:
                process.terminate()
                logger.info(f"  ✓ Stopped {name}")
            except:
                pass
        
        # Wait for processes to terminate
        for process in self.processes.values():
            try:
                process.wait(timeout=5)
            except:
                process.kill()


def main():
    parser = argparse.ArgumentParser(
        description="🔐 Privacy-First Federated Learning Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Start all components
  python main.py --clients 5        # With 5 federated clients
  python main.py --skip-dashboard   # Start backend & API only
  python main.py --api-port 9000    # Run API on custom port
        """
    )
    
    parser.add_argument(
        '--clients',
        type=int,
        default=3,
        help='Number of federated clients (default: 3)'
    )
    parser.add_argument(
        '--rounds',
        type=int,
        default=5,
        help='Number of training rounds (default: 5)'
    )
    parser.add_argument(
        '--model',
        default='mlp',
        help='Model type: mlp, cnn (default: mlp)'
    )
    parser.add_argument(
        '--api-port',
        type=int,
        default=8000,
        help='API server port (default: 8000)'
    )
    parser.add_argument(
        '--dashboard-port',
        type=int,
        default=8501,
        help='Dashboard port (default: 8501)'
    )
    parser.add_argument(
        '--skip-backend',
        action='store_true',
        help='Skip federated learning backend'
    )
    parser.add_argument(
        '--skip-dashboard',
        action='store_true',
        help='Skip dashboard (API only mode)'
    )
    
    args = parser.parse_args()
    
    manager = SystemManager({
        'clients': args.clients,
        'rounds': args.rounds,
        'model': args.model,
    })
    
    try:
        if manager.start_all(args):
            # Keep running
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n")
        manager.cleanup()
        logger.info("👋 Shutdown complete")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        manager.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
