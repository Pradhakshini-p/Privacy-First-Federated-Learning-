#!/usr/bin/env python3
"""
Setup Script for Privacy-First Federated Learning Pipeline v2
Automates environment setup and validation
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FederatedLearningSetup:
    """Setup automation for the federated learning pipeline"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.src_dir = self.project_root / "src"
        self.data_dir = self.project_root / "data"
        self.logs_dir = self.project_root / "logs"
        
    def check_python_version(self):
        """Check Python version compatibility"""
        logger.info("Checking Python version...")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            logger.error(f"Python 3.8+ required, found {version.major}.{version.minor}")
            return False
        
        logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro} compatible")
        return True
    
    def create_directories(self):
        """Create necessary directories"""
        logger.info("Creating directories...")
        
        directories = [
            self.data_dir,
            self.logs_dir,
            self.src_dir / "__pycache__"
        ]
        
        for directory in directories:
            try:
                directory.mkdir(exist_ok=True)
                logger.info(f"✓ Created directory: {directory}")
            except Exception as e:
                logger.error(f"Failed to create {directory}: {e}")
                return False
        
        return True
    
    def check_dependencies(self):
        """Check and install required dependencies"""
        logger.info("Checking dependencies...")
        
        requirements_file = self.project_root / "requirements.txt"
        
        if not requirements_file.exists():
            logger.warning("requirements.txt not found, creating basic requirements...")
            basic_requirements = [
                "torch>=1.9.0",
                "torchvision>=0.10.0",
                "numpy>=1.19.0",
                "pandas>=1.3.0",
                "scikit-learn>=1.0.0",
                "streamlit>=1.0.0",
                "plotly>=5.0.0",
                "opacus>=1.0.0",
                "tqdm>=4.60.0"
            ]
            
            with open(requirements_file, 'w') as f:
                f.write('\n'.join(basic_requirements))
        
        try:
            # Install requirements
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning(f"Some dependencies failed to install: {result.stderr}")
                logger.info("Attempting to install core dependencies...")
                
                core_deps = ["torch", "numpy", "pandas", "scikit-learn", "streamlit", "plotly"]
                for dep in core_deps:
                    try:
                        subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                                     capture_output=True, check=True)
                        logger.info(f"✓ Installed {dep}")
                    except subprocess.CalledProcessError:
                        logger.error(f"Failed to install {dep}")
            else:
                logger.info("✓ All dependencies installed successfully")
            
        except Exception as e:
            logger.error(f"Error installing dependencies: {e}")
            return False
        
        return True
    
    def validate_configuration(self):
        """Validate the configuration"""
        logger.info("Validating configuration...")
        
        try:
            # Import and run config validation
            sys.path.append(str(self.src_dir))
            from config import validate_config, get_data_info
            
            # Run validation
            if validate_config():
                logger.info("✓ Configuration validation passed")
            else:
                logger.error("Configuration validation failed")
                return False
            
            # Display data info
            logger.info("Data configuration:")
            get_data_info()
            
        except ImportError as e:
            logger.error(f"Could not import config: {e}")
            return False
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
        
        return True
    
    def create_sample_data(self):
        """Create sample data for testing"""
        logger.info("Creating sample data...")
        
        try:
            import pandas as pd
            import numpy as np
            
            # Generate sample healthcare data
            np.random.seed(42)
            n_samples = 1000
            
            data = {
                'age': np.random.randint(18, 80, n_samples),
                'bmi': np.random.uniform(18.5, 35.0, n_samples),
                'blood_pressure': np.random.randint(90, 180, n_samples),
                'cholesterol': np.random.randint(150, 300, n_samples),
                'glucose': np.random.randint(70, 200, n_samples),
                'smoking': np.random.randint(0, 2, n_samples),
                'exercise': np.random.randint(0, 5, n_samples),
                'disease_risk': np.random.randint(0, 2, n_samples)  # Target variable
            }
            
            df = pd.DataFrame(data)
            sample_file = self.data_dir / "healthcare_disease_prediction.csv"
            df.to_csv(sample_file, index=False)
            
            logger.info(f"✓ Created sample data: {sample_file}")
            logger.info(f"  Shape: {df.shape}")
            logger.info(f"  Features: {list(df.columns)}")
            
        except Exception as e:
            logger.error(f"Error creating sample data: {e}")
            return False
        
        return True
    
    def test_imports(self):
        """Test critical imports"""
        logger.info("Testing imports...")
        
        test_modules = [
            ("torch", "PyTorch"),
            ("numpy", "NumPy"),
            ("pandas", "Pandas"),
            ("sklearn", "Scikit-learn"),
            ("streamlit", "Streamlit"),
            ("plotly", "Plotly"),
        ]
        
        failed_imports = []
        
        for module_name, display_name in test_modules:
            try:
                __import__(module_name)
                logger.info(f"✓ {display_name} imported successfully")
            except ImportError:
                logger.error(f"✗ {display_name} not available")
                failed_imports.append(display_name)
        
        # Test privacy-specific imports
        try:
            import opacus
            logger.info("✓ Opacus (privacy) imported successfully")
        except ImportError:
            logger.warning("⚠ Opacus not available - privacy features will be limited")
            failed_imports.append("Opacus")
        
        if failed_imports:
            logger.warning(f"Some imports failed: {failed_imports}")
            logger.info("Run: pip install " + " ".join(failed_imports))
        
        return len(failed_imports) == 0
    
    def create_run_scripts(self):
        """Create convenient run scripts"""
        logger.info("Creating run scripts...")
        
        # Server script
        server_script = self.project_root / "run_server.bat"
        server_content = '''@echo off
echo Starting Federated Learning Server...
python -m src.perfect_federated_platform_v2 --mode server --rounds 5
pause
'''
        
        # Client script
        client_script = self.project_root / "run_client.bat"
        client_content = '''@echo off
set /p client_id="Enter client ID (1-3): "
python -m src.perfect_federated_platform_v2 --mode client --client-id %client_id%
pause
'''
        
        # Dashboard script
        dashboard_script = self.project_root / "run_dashboard.bat"
        dashboard_content = '''@echo off
echo Starting Federated Learning Dashboard...
streamlit run src/enhanced_dashboard_v3.py --server.port 8501
pause
'''
        
        # Docker script
        docker_script = self.project_root / "run_docker.bat"
        docker_content = '''@echo off
echo Starting Federated Learning with Docker...
docker-compose up --build
pause
'''
        
        scripts = [
            (server_script, server_content),
            (client_script, client_content),
            (dashboard_script, dashboard_content),
            (docker_script, docker_content)
        ]
        
        for script_path, content in scripts:
            try:
                with open(script_path, 'w') as f:
                    f.write(content)
                logger.info(f"✓ Created script: {script_path.name}")
            except Exception as e:
                logger.error(f"Failed to create {script_path}: {e}")
        
        return True
    
    def check_docker(self):
        """Check Docker availability"""
        logger.info("Checking Docker...")
        
        try:
            # Check Docker
            result = subprocess.run(["docker", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✓ Docker: {result.stdout.strip()}")
            else:
                logger.warning("Docker not found - Docker features unavailable")
                return False
            
            # Check Docker Compose
            result = subprocess.run(["docker-compose", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✓ Docker Compose: {result.stdout.strip()}")
            else:
                logger.warning("Docker Compose not found - Docker features unavailable")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Docker check failed: {e}")
            return False
    
    def run_setup(self):
        """Run complete setup process"""
        logger.info("🚀 Starting Federated Learning Pipeline Setup")
        logger.info("=" * 50)
        
        setup_steps = [
            ("Python Version", self.check_python_version),
            ("Create Directories", self.create_directories),
            ("Check Dependencies", self.check_dependencies),
            ("Test Imports", self.test_imports),
            ("Validate Configuration", self.validate_configuration),
            ("Create Sample Data", self.create_sample_data),
            ("Create Run Scripts", self.create_run_scripts),
            ("Check Docker", self.check_docker),
        ]
        
        failed_steps = []
        
        for step_name, step_func in setup_steps:
            logger.info(f"\n📋 {step_name}")
            logger.info("-" * 30)
            
            try:
                if not step_func():
                    failed_steps.append(step_name)
                    logger.error(f"✗ {step_name} failed")
                else:
                    logger.info(f"✓ {step_name} completed")
            except Exception as e:
                logger.error(f"✗ {step_name} error: {e}")
                failed_steps.append(step_name)
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("📊 SETUP SUMMARY")
        logger.info("=" * 50)
        
        if not failed_steps:
            logger.info("🎉 All setup steps completed successfully!")
            logger.info("\n🚀 You can now run the system:")
            logger.info("1. Manual: Run the .bat scripts or use Python commands")
            logger.info("2. Docker: docker-compose up --build")
            logger.info("3. Dashboard: http://localhost:8501")
        else:
            logger.error(f"❌ {len(failed_steps)} steps failed:")
            for step in failed_steps:
                logger.error(f"  - {step}")
            logger.info("\n🔧 Please fix the failed steps before running the system")
        
        return len(failed_steps) == 0

def main():
    """Main setup function"""
    setup = FederatedLearningSetup()
    success = setup.run_setup()
    
    if success:
        logger.info("\n✨ Setup complete! Your federated learning pipeline is ready.")
        logger.info("📖 Read README_v2.md for detailed usage instructions.")
    else:
        logger.error("\n💥 Setup failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
