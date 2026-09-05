# Privacy-First Federated Learning Pipeline v2

A production-ready federated learning platform with real-time privacy monitoring, differential privacy, and comprehensive dashboard visualization.

## 🚀 Key Features

### ✅ Centralized Configuration
- **Single source of truth** for all paths and hyperparameters
- Consistent configuration across server, clients, and dashboard
- Easy customization of privacy parameters and model settings

### 📊 Real-time Logging & Monitoring
- **CSV-based logging bridge** for seamless data sharing
- Real-time dashboard updates with auto-refresh
- Comprehensive metrics tracking (accuracy, loss, privacy budget)

### 🐳 Production Docker Support
- **Proper service orchestration** with health checks
- Sequential startup to prevent connection failures
- Resource limits and restart policies
- Network isolation and volume management

### 🔒 Advanced Privacy Engine
- **Differential privacy** with Opacus integration
- **Real-time epsilon tracking** per client
- Privacy budget monitoring and exhaustion warnings
- Configurable noise multipliers per client

## 📁 Updated Project Structure

```
Privacy-First Federated Learning Pipeline/
├── src/
│   ├── config.py                    # 🆕 Centralized configuration
│   ├── logging_bridge.py            # 🆕 CSV-based logging system
│   ├── privacy_engine.py            # 🆕 Differential privacy engine
│   ├── enhanced_dashboard_v3.py     # 🆕 Real-time dashboard
│   ├── perfect_federated_platform_v2.py  # 🆕 Privacy-first FL platform
│   ├── model.py                     # Model definitions
│   ├── data.py                      # Data loading utilities
│   └── [other existing modules...]
├── data/                            # Dataset directory
├── logs/                            # 🆕 CSV logs directory
├── docker-compose.yml               # 🔄 Updated orchestration
├── Dockerfile                       # Container definition
└── README_v2.md                     # This file
```

## 🛠️ Quick Start

### Prerequisites
- Python 3.8+
- Docker & Docker Compose
- Git

### 1. Clone and Setup
```bash
git clone <repository-url>
cd "Privacy-First Federated Learning Pipeline"
```

### 2. Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Additional requirements for privacy features
pip install opacus>=1.0.0
```

### 3. Prepare Data
```bash
# Place your dataset in the data/ directory
# Update src/config.py to point to your data file
python -m src.config  # Verify configuration
```

### 4. Run the System

#### Option A: Docker (Recommended)
```bash
# Build and start all services
docker-compose up --build

# Services will start in order:
# 1. Server (port 8080)
# 2. Clients (3 instances)
# 3. Dashboard (port 8501)
```

#### Option B: Manual (Development)
```bash
# Terminal 1: Start Dashboard
streamlit run src/enhanced_dashboard_v3.py --server.port 8501

# Terminal 2: Start FL Server
python -m src.perfect_federated_platform_v2 --mode server --rounds 5

# Terminal 3-5: Start Clients (one per terminal)
python -m src.perfect_federated_platform_v2 --mode client --client-id 1
python -m src.perfect_federated_platform_v2 --mode client --client-id 2
python -m src.perfect_federated_platform_v2 --mode client --client-id 3
```

## 📊 Dashboard Access

Once running, access the dashboard at:
- **Local**: http://localhost:8501
- **Network**: http://YOUR_IP:8501

### Dashboard Features
- **Training Metrics**: Real-time accuracy and loss tracking
- **Privacy Metrics**: Epsilon spending and budget monitoring
- **Client Performance**: Individual client analytics
- **Auto-refresh**: Configurable update intervals

## ⚙️ Configuration

### Main Configuration (`src/config.py`)

```python
# Federated Learning Parameters
NUM_CLIENTS = 3
ROUNDS = 5
LEARNING_RATE = 0.01
BATCH_SIZE = 32

# Privacy Parameters
EPSILON = 1.0
DELTA = 1e-5
NOISE_MULTIPLIER = 1.0
MAX_GRAD_NORM = 1.0

# Client-Specific Privacy
CLIENT_CONFIGS = {
    "client_1": {"privacy_enabled": True, "noise_multiplier": 1.0},
    "client_2": {"privacy_enabled": True, "noise_multiplier": 1.5},
    "client_3": {"privacy_enabled": True, "noise_multiplier": 2.0},
}
```

### Data Configuration
```python
DATA_CONFIG = {
    "data_path": "data/your_dataset.csv",
    "target_column": "target_variable"
}
```

## 🔒 Privacy Features

### Differential Privacy
- **Opacus Integration**: Industry-standard DP library
- **Per-Client Privacy**: Independent privacy budgets
- **Real-time Tracking**: Live epsilon monitoring
- **Budget Management**: Automatic exhaustion detection

### Privacy Monitoring
```python
# During training, you'll see:
# Round 1 Privacy Update for client_1:
#   Current ε: 0.1234
#   ε spent this round: 0.1234
#   Privacy budget used: 12.34%
#   Data size: 1000
```

### Privacy Budget Warnings
- Automatic alerts when budget > 80%
- Training stops when budget exhausted
- Configurable per-client thresholds

## 🐳 Docker Configuration

### Service Orchestration
- **Health Checks**: Ensures services are ready
- **Sequential Startup**: Server starts before clients
- **Resource Limits**: Memory constraints per service
- **Volume Mounts**: Shared logs and data

### Network Configuration
- **Isolated Network**: `fl-network` (172.20.0.0/16)
- **Service Discovery**: Clients find server via hostname
- **Port Mapping**: Dashboard (8501), Server (8080)

## 📈 Logging & Monitoring

### CSV Log Files
All metrics are stored in CSV format in the `logs/` directory:

- `training_metrics.csv`: Global training progress
- `privacy_metrics.csv`: Privacy budget tracking
- `client_metrics.csv`: Individual client performance

### Real-time Updates
- Dashboard reads CSV files every 5 seconds (configurable)
- No database required - file-based persistence
- Thread-safe logging with locks

## 🧪 Testing & Validation

### Configuration Validation
```bash
python -m src.config
```
Checks:
- Directory structure
- Data file existence
- Hyperparameter validity

### Privacy Validation
```python
# Check privacy budget status
privacy_status = privacy_engine.get_privacy_status()
print(f"Budget used: {privacy_status['privacy_budget_used']:.2%}")
```

## 🔧 Troubleshooting

### Common Issues

#### 1. "File Not Found" Errors
- **Solution**: Ensure terminal is in project root
- **Use**: `python -m src.module_name` instead of `python src/module_name.py`

#### 2. Docker Connection Errors
- **Solution**: Check docker-compose health checks
- **Verify**: Server starts before clients (30s delay)

#### 3. Privacy Budget Exhausted
- **Solution**: Increase EPSILON in config
- **Or**: Reduce NOISE_MULTIPLIER for less privacy

#### 4. Dashboard Not Updating
- **Solution**: Check logs/ directory permissions
- **Verify**: CSV files are being written

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Advanced Usage

### Custom Privacy Parameters
```python
# In src/config.py
CLIENT_CONFIGS = {
    "high_privacy": {
        "privacy_enabled": True,
        "noise_multiplier": 3.0,
        "max_grad_norm": 0.5
    },
    "low_privacy": {
        "privacy_enabled": True,
        "noise_multiplier": 0.5,
        "max_grad_norm": 2.0
    }
}
```

### Custom Model Architecture
```python
# In src/model.py
def create_custom_model(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, output_dim)
    )
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Opacus**: Differential privacy library by Meta
- **Flower**: Federated learning framework
- **Streamlit**: Dashboard framework
- **PyTorch**: Deep learning framework

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in the `logs/` directory
3. Create an issue with detailed error messages

---

**🎉 Congratulations!** You now have a production-ready federated learning platform with comprehensive privacy monitoring and real-time visualization.
