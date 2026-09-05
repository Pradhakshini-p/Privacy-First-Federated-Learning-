# 🔐 Privacy-First Federated Learning Platform

## 📋 Quick Start

### Option 1: Automated Setup (Recommended)

**Windows:**
```bash
start.bat
```

**Linux/Mac:**
```bash
bash start.sh
```

### Option 2: Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start the complete system
python main.py

# Or start with custom parameters
python main.py --clients 5 --rounds 10 --api-port 9000
```

### Option 3: Docker

```bash
docker-compose up --build
# Access at http://localhost:8501
```

---

## 🎯 Quick Navigation

- **🖥️ Dashboard**: http://localhost:8501
- **📡 API Docs**: http://localhost:8000/docs  
- **🏥 API Health**: http://localhost:8000/health

---

## 🌟 Key Features

### 🔐 Privacy-First Design
- **Differential Privacy**: Built-in ε-δ privacy guarantees
- **Secure Aggregation**: Cryptographic protection of model updates
- **Privacy Budget Tracking**: Real-time monitoring and control

### 📊 Advanced Dashboard
- **Real-time Monitoring**: Live training metrics and client status
- **Interactive Controls**: Dynamic privacy parameter tuning
- **Visualization**: Professional charts and network topology
- **5-Tab Interface**: Training • Privacy • Security • Debugging • Controls

### 🚀 Enterprise Ready
- **Scalable Architecture**: Support for N clients
- **REST API**: Full programmatic control
- **WebSocket Support**: Real-time data streaming
- **Comprehensive Logging**: CSV-based metrics export

### 🤖 Federated Learning
- **Multi-Client Training**: Coordinate training across distributed clients
- **Configurable Rounds**: Control training iterations
- **Privacy Per Client**: Individual privacy settings per participant
- **Automatic Aggregation**: Weighted model parameter averaging

---

## 📖 Example Scenarios

### Scenario 1: Quick Demo (Default)
```bash
python main.py
# 3 clients, 5 rounds, default privacy settings
```

### Scenario 2: Enterprise Scale
```bash
python main.py --clients 10 --rounds 20
# 10 federated clients, 20 training rounds
```

### Scenario 3: Development/Testing
```bash
python main.py --skip-backend
# API only - useful for frontend testing
```

### Scenario 4: Custom Configuration
```bash
python main.py \
  --clients 5 \
  --rounds 15 \
  --api-port 9000 \
  --dashboard-port 8502
```

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────┐
│         Dashboard (Streamlit)                      │
│  • Real-time metrics visualization               │
│  • Interactive privacy controls                   │
│  • Client topology and performance                │
└────────────────┬─────────────────────────────────┘
                 │ WebSocket
                 ▼
┌────────────────────────────────────────────────────┐
│     API Server (FastAPI)                          │
│  • REST API endpoints                             │
│  • WebSocket connections                          │
│  • Real-time data broadcasting                    │
└────────────────┬─────────────────────────────────┘
                 │ Python
                 ▼
┌────────────────────────────────────────────────────┐
│     Backend (Federated Learning)                  │
│  • Global model training                          │
│  • Client coordination                            │
│  • Privacy budget management                      │
│  • Metrics collection                             │
└────────────────────────────────────────────────────┘
      │              │              │
      ▼              ▼              ▼
   Client 1      Client 2      Client N
  (Privacy)     (Privacy)     (Privacy)
```

---

## 🔧 Configuration

### Default Parameters
```python
CLIENTS = 3              # Number of federated clients
ROUNDS = 5              # Training rounds
EPSILON = 1.0           # Privacy budget
DELTA = 1e-5            # Privacy failure probability
LEARNING_RATE = 0.01    # Model learning rate
BATCH_SIZE = 32         # Training batch size
LOCAL_EPOCHS = 5        # Local training epochs
```

### Configure via Command Line
```bash
python main.py --clients 5 --rounds 10 --model cnn
```

### Configure via Environment Variables
```bash
export FL_CLIENTS=5
export FL_ROUNDS=10
export FL_EPSILON=0.5
python main.py
```

---

## 📊 Monitoring & Metrics

### Real-time Metrics
- **Global Accuracy**: Model accuracy on aggregated test set
- **Training Loss**: Loss during local training
- **Client Status**: Active, inactive, completed
- **Privacy Spent**: Cumulative privacy budget usage
- **Computation Time**: Per-round training duration

### Metrics Export
Metrics are automatically saved to:
```
logs/
├── client_metrics.csv      # Per-client metrics
├── training_metrics.csv    # Global training metrics
├── privacy_metrics.csv     # Privacy budget tracking
└── backend.log             # System logs
```

---

## 🔒 Security Features

### Privacy Controls
- ✅ Differential privacy with configurable ε and δ
- ✅ Gradient clipping per client
- ✅ Noise injection per-client
- ✅ Privacy budget exhaustion detection
- ✅ Secure aggregation support

### Security Levels
```
BASIC       - No privacy (testing only)
STANDARD    - Differential privacy enabled
MILITARY    - Enhanced privacy + secure aggregation
```

---

## 📡 API Endpoints

### Status
- `GET /api/status` - Get current system status
- `GET /api/privacy/status` - Get privacy metrics
- `GET /health` - Health check

### Training Control
- `POST /api/training/start` - Start federated training
- `POST /api/training/stop` - Stop training

### Metrics
- `GET /api/metrics/global` - Global training metrics
- `GET /api/metrics/clients` - All client metrics
- `GET /api/metrics/clients/{id}` - Specific client metrics

### Configuration
- `POST /api/privacy/config` - Update privacy settings
- `POST /api/privacy/client/{id}/config` - Client privacy settings

### Real-time
- `WS /ws` - WebSocket for real-time updates

**Full API Documentation**: http://localhost:8000/docs

---

## 🐛 Troubleshooting

### Dashboard shows "No data yet"
This is normal when training just started. Wait 5-10 seconds for data collection.

### API returns 500 errors
Check logs: `cat logs/backend.log`

### Port already in use
```bash
python main.py --api-port 9000 --dashboard-port 8502
```

### Missing dependencies
```bash
pip install -r requirements.txt --upgrade
```

---

## 📚 Documentation

- [Backend Architecture](docs/BACKEND.md) - Detail backend system
- [Frontend Integration](docs/FRONTEND.md) - Dashboard and UI
- [API Reference](docs/API.md) - Complete API documentation
- [Privacy Implementation](docs/PRIVACY.md) - Differential privacy details
- [Development Guide](docs/DEVELOPMENT.md) - Contributing and extending

---

## 🎓 Research & Papers

This implementation is based on:
- **Federated Learning**: "Communication-Efficient Learning of Deep Networks from Decentralized Data" (McMahan et al., 2017)
- **Differential Privacy**: "Deep Learning with Differential Privacy" (Abadi et al., 2016)
- **Opacus Library**: PyTorch's differential privacy library

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## 📧 Support

- 📖 Documentation: Check docs/ folder
- 🐛 Issues: GitHub Issues
- 💬 Questions: GitHub Discussions

---

## 🌟 Highlights for Recruiters

### Technical Excellence
- ✅ Production-ready federated learning platform
- ✅ PyTorch + Flower framework integration
- ✅ Differential privacy (Opacus) implementation
- ✅ Real-time WebSocket data streaming
- ✅ REST API with FastAPI

### Best Practices
- ✅ Clean, modular architecture
- ✅ Comprehensive error handling
- ✅ Extensive logging and monitoring
- ✅ Docker containerization
- ✅ Automated deployment ready

### Career Impact
- ✅ Enterprise-level ML systems
- ✅ Privacy-preserving AI implementation
- ✅ Distributed systems coordination
- ✅ Real-time data visualization
- ✅ Production deployment experience

---

**Ready to get started?** Run `python main.py` now! 🚀
