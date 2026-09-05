#!/usr/bin/env python3
"""
Main Launcher Script for Privacy-First Federated Learning Platform
Provides a single entry point for all operations
"""

import argparse
import subprocess
import sys
import os
import socket
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['torch', 'flwr', 'sklearn', 'pandas', 'numpy']
    missing = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.info("Install with: pip install -r requirements.txt")
        return False

    return True


def find_available_port(start=8080, end=8099):
    """Find an available TCP port on localhost."""
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No available port in range {start}-{end - 1}")


def get_server_address(port=None):
    """Resolve Flower server address, picking a free port if needed."""
    if port:
        return f"127.0.0.1:{port}"
    return f"127.0.0.1:{find_available_port()}"


def start_server(args):
    """Start the federated learning server"""
    server_address = get_server_address(getattr(args, "port", None))
    logger.info(f"🚀 Starting Federated Learning Server at {server_address}...")
    cmd = [
        sys.executable, "server.py",
        "--model", args.model,
        "--rounds", str(args.rounds),
        "--min-clients", str(args.min_clients),
        "--address", server_address,
    ]
    subprocess.run(cmd)


def start_client(args):
    """Start a federated learning client"""
    logger.info(f"🚀 Starting Client {args.client_id}...")
    cmd = [
        sys.executable, "client.py", str(args.client_id),
        "--hospital", args.hospital,
        "--model", args.model,
        "--server", args.server
    ]
    if args.no_privacy:
        cmd.append("--no-privacy")
    subprocess.run(cmd)


def start_centralized_training(args):
    """Start centralized training for comparison"""
    logger.info("🚀 Starting Centralized Training...")
    cmd = [sys.executable, "train_centralized.py", "--model", args.model, "--epochs", str(args.epochs), "--lr", str(args.lr)]
    subprocess.run(cmd)


def start_evaluation(args):
    """Run evaluation and comparison"""
    logger.info("🚀 Running Evaluation...")
    cmd = [sys.executable, "evaluate.py", "--model", args.model]
    subprocess.run(cmd)


def start_api(args):
    """Start the inference API"""
    logger.info("🚀 Starting Inference API...")
    cmd = [sys.executable, "api.py", "--host", args.host, "--port", str(args.api_port)]
    if args.debug:
        cmd.append("--debug")
    subprocess.run(cmd)


def run_demo(args):
    """Run complete demo with server and clients"""
    logger.info("=" * 60)
    logger.info("🎯 Starting Complete Federated Learning Demo")
    logger.info("=" * 60)

    if not check_dependencies():
        return

    # Start server in background
    server_address = get_server_address(getattr(args, "port", None))
    logger.info(f"🌸 Starting server at {server_address}...")
    server_cmd = [
        sys.executable, "server.py",
        "--model", args.model,
        "--rounds", str(args.rounds),
        "--min-clients", str(args.min_clients),
        "--address", server_address,
    ]
    env = os.environ.copy()
    env["FL_SERVER_ADDRESS"] = server_address
    server_process = subprocess.Popen(server_cmd, env=env)

    import time
    time.sleep(3)

    # Start clients
    logger.info("🏥 Starting hospital clients...")
    client_processes = []
    for client_id in range(1, args.min_clients + 1):
        hospital_id = f"hospital_{client_id}"
        client_cmd = [
            sys.executable, "client.py", str(client_id),
            "--hospital", hospital_id,
            "--model", args.model,
            "--server", server_address
        ]
        if args.no_privacy:
            client_cmd.append("--no-privacy")
        client_process = subprocess.Popen(client_cmd)
        client_processes.append(client_process)
        time.sleep(1)

    logger.info("=" * 60)
    logger.info("✅ Demo started! Training in progress...")
    logger.info("=" * 60)

    exit_code = server_process.wait()

    for client_process in client_processes:
        if client_process.poll() is None:
            client_process.terminate()
            try:
                client_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                client_process.kill()

    if exit_code == 0:
        logger.info("✅ Federated training completed successfully!")
        if args.run_evaluate:
            start_evaluation(args)
    else:
        logger.error(f"❌ Server exited with code {exit_code}")


def run_full_pipeline(args):
    """Run complete HR demo pipeline: centralized -> federated -> evaluate"""
    logger.info("=" * 60)
    logger.info("🎯 Running Full Pipeline for HR Review")
    logger.info("=" * 60)

    if not check_dependencies():
        return

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    logger.info("Step 1/3: Centralized baseline training...")
    start_centralized_training(args)

    logger.info("Step 2/3: Federated learning with 3 hospitals...")
    args.run_evaluate = False
    run_demo(args)

    logger.info("Step 3/3: Evaluation and comparison...")
    start_evaluation(args)

    logger.info("=" * 60)
    logger.info("✅ Full pipeline completed!")
    logger.info("   Models:  models/centralized_model.pth, models/global_model.pth")
    logger.info("   Results: results/comparison_plots.png, results/comparison_report.json")
    logger.info("   API:     python launch.py --mode api")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Privacy-First Federated Learning Platform - Main Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch.py --mode server                    # Start FL server
  python launch.py --mode client --id 1             # Start client 1
  python launch.py --mode centralized               # Train centralized model
  python launch.py --mode evaluate                  # Run evaluation
  python launch.py --mode api                       # Start inference API
  python launch.py --mode demo                      # Run complete demo
  python launch.py --mode full                      # Run full HR pipeline
        """
    )

    parser.add_argument(
        '--run-evaluate',
        action='store_true',
        help='Run evaluation automatically after demo completes'
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['server', 'client', 'centralized', 'evaluate', 'api', 'demo', 'full'],
        help='Mode to run'
    )

    # Server options
    parser.add_argument('--rounds', type=int, default=5, help='Number of FL rounds')
    parser.add_argument('--min-clients', type=int, default=3, help='Minimum clients required')

    # Client options
    parser.add_argument('--id', type=int, dest='client_id', help='Client ID')
    parser.add_argument('--hospital', type=str, help='Hospital ID (e.g., hospital_1)')
    parser.add_argument('--server', type=str, default='localhost:8080', help='Server address')
    parser.add_argument('--no-privacy', action='store_true', help='Disable differential privacy')

    # Centralized training options
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    # API options
    parser.add_argument('--host', type=str, default='0.0.0.0', help='API host')
    parser.add_argument('--port', type=int, default=None, help='Flower server port (auto if unset)')
    parser.add_argument('--api-port', type=int, default=5000, help='API port')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    # Model options
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn'], help='Model type')

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Execute based on mode
    if args.mode == 'server':
        start_server(args)
    elif args.mode == 'client':
        if args.client_id is None:
            logger.error("Client ID required for client mode. Use --id <client_id>")
            sys.exit(1)
        start_client(args)
    elif args.mode == 'centralized':
        start_centralized_training(args)
    elif args.mode == 'evaluate':
        start_evaluation(args)
    elif args.mode == 'api':
        start_api(args)
    elif args.mode == 'demo':
        run_demo(args)
    elif args.mode == 'full':
        args.run_evaluate = True
        run_full_pipeline(args)


if __name__ == "__main__":
    main()
