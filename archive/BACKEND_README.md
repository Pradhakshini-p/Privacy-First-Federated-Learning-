# 🔐 Federated Learning Backend System

A comprehensive backend implementation for privacy-first federated learning with real-time monitoring, secure aggregation, and differential privacy.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client 1      │    │   Client 2      │    │   Client N      │
│  (Enhanced)     │    │  (Enhanced)     │    │  (Enhanced)     │
│                 │    │                 │    │                 │
│ 🤖 Local Model  │    │ 🤖 Local Model  │    │ 🤖 Local Model  │
│ 🔒 Privacy      │    │ 🔒 Privacy      │    │ 🔒 Privacy      │
│ 📊 Training     │    │ 📊 Training     │    │ 📊 Training     │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │   🔐 Encrypted        │                      │
          │   Updates            │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │   Backend Server         │
                    │  🤖 Global Model        │
                    │  📊 Aggregation         │
                    │  🔒 Secure Agg          │
                    │  📈 Monitoring         │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │   API Server            │
                    │  🌐 REST Endpoints      │
                    │  📡 WebSocket          │
                    │  📊 Real-time Updates  │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │   Dashboard              │
                    │  📈 Visualizations      │
                    │  🎛️ Controls           │
                    │  📊 Monitoring         │
                    └───────────────────────────┘
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_backend.txt
```

### 2. Start Complete System

```bash
cd src
python launcher.py --clients 3 --rounds 5
```

### 3. Access Components

- **Dashboard**: http://localhost:8502
- **API Server**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📁 Core Components

### 1. **Federated Backend** (`federated_backend.py`)

**Main orchestration system** that manages:
- Global model training and aggregation
- Privacy budget tracking
- Client coordination
- Metrics collection

**Key Features:**
- Real-time training monitoring
- Privacy budget management
- Client performance tracking
- Automatic failover handling

### 2. **Enhanced Client** (`enhanced_client.py`)

**Privacy-aware client implementation** with:
- Differential privacy integration (Opacus)
- Secure model updates
- Local training with privacy protection
- Real-time metrics reporting

**Key Features:**
- Configurable privacy parameters
- Automatic privacy budget tracking
- Encrypted parameter transmission
- Client-side monitoring

### 3. **Secure Aggregation** (`secure_aggregation.py`)

**Cryptographic protection** for model updates:
- Parameter encryption/decryption
- Secure multi-party aggregation
- Privacy-preserving noise injection
- Signature verification

**Security Levels:**
- **Basic**: Standard encryption
- **Standard**: Enhanced security with PBKDF2
- **Military**: Maximum security with SHA512

### 4. **API Server** (`api_server.py`)

**RESTful API** for system control and monitoring:
- Real-time WebSocket updates
- Training control endpoints
- Privacy configuration
- Metrics and status APIs

**Endpoints:**
- `GET /api/status` - System status
- `POST /api/training/start` - Start training
- `GET /api/privacy/status` - Privacy status
- `WebSocket /ws` - Real-time updates

### 5. **Privacy Engine** (`privacy_engine.py`)

**Differential privacy implementation**:
- ε-differential privacy tracking
- Real-time budget monitoring
- Configurable noise injection
- Privacy recommendations

## 🎛️ Configuration

### System Configuration (`config.py`)

```python
# Federated Learning
NUM_CLIENTS = 3
ROUNDS = 5
LEARNING_RATE = 0.01

# Privacy Parameters
EPSILON = 1.0
DELTA = 1e-5
NOISE_MULTIPLIER = 1.0
MAX_GRAD_NORM = 1.0

# Client Configurations
CLIENT_CONFIGS = {
    "client_1": {
        "privacy_enabled": True,
        "noise_multiplier": 1.0,
        "local_epochs": 5
    },
    # ... more clients
}
```

### Privacy Configuration

```python
# Differential Privacy
privacy_config = {
    "epsilon": 1.0,        # Privacy budget
    "delta": 1e-5,         # Failure probability
    "noise_multiplier": 1.0, # Noise scale
    "max_grad_norm": 1.0   # Gradient clipping
}
```

### Security Configuration

```python
# Secure Aggregation
security_config = {
    "security_level": "standard",  # basic, standard, military
    "enable_encryption": True,
    "enable_signatures": True
}
```

## 🚀 Launcher Options

### Complete System

```bash
# Start all components
python launcher.py --clients 3 --rounds 5

# With custom ports
python launcher.py --api-port 8080 --dashboard-port 8503

# Privacy configuration
python launcher.py --epsilon 0.5 --noise-multiplier 1.5
```

### Individual Components

```bash
# Start only backend server
python launcher.py --mode server

# Start only API server
python launcher.py --mode api --api-port 8080

# Start only dashboard
python launcher.py --mode dashboard --dashboard-port 8503
```

### Privacy Options

```bash
# Enable privacy
python launcher.py --privacy --epsilon 1.0

# Disable privacy
python launcher.py --no-privacy

# Custom privacy parameters
python launcher.py --epsilon 0.1 --noise-multiplier 2.0
```

## 📊 API Usage

### Start Training

```bash
curl -X POST "http://localhost:8000/api/training/start" \
     -H "Content-Type: application/json" \
     -d '{"num_rounds": 5, "learning_rate": 0.01}'
```

### Get System Status

```bash
curl "http://localhost:8000/api/status"
```

### Update Privacy Configuration

```bash
curl -X POST "http://localhost:8000/api/privacy/config" \
     -H "Content-Type: application/json" \
     -d '{"epsilon": 1.0, "noise_multiplier": 1.0}'
```

### WebSocket Real-time Updates

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Update:', data);
};
```

## 🔒 Privacy Features

### Differential Privacy

- **ε-Differential Privacy**: Mathematically proven privacy guarantees
- **Real-time Tracking**: Live privacy budget monitoring
- **Configurable Parameters**: Adjustable ε, δ, noise multiplier
- **Budget Enforcement**: Automatic training stop when budget exhausted

### Secure Aggregation

- **Parameter Encryption**: AES-256-GCM encryption
- **Secure Multi-Party Computation**: Cryptographic aggregation
- **Signature Verification**: Client authentication
- **Multiple Security Levels**: Basic to military-grade

### Privacy Monitoring

- **Budget Tracking**: Per-client and global privacy spending
- **Real-time Alerts**: Privacy budget warnings
- **Recommendations**: Privacy optimization suggestions
- **Audit Trail**: Complete privacy usage logs

## 📈 Monitoring and Logging

### Metrics Collection

- **Training Metrics**: Accuracy, loss, convergence
- **Privacy Metrics**: ε spending, budget usage
- **Client Metrics**: Performance, contribution scores
- **System Metrics**: Resource usage, latency

### Logging

- **Structured Logs**: JSON format for easy parsing
- **Multiple Levels**: DEBUG, INFO, WARNING, ERROR
- **File Rotation**: Automatic log management
- **Real-time Streaming**: Live log viewing

### Dashboard Integration

- **Real-time Updates**: 5-second refresh intervals
- **Interactive Controls**: Dynamic parameter adjustment
- **Visual Analytics**: Charts, graphs, gauges
- **Alert System**: Privacy and performance warnings

## 🛠️ Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run privacy tests
pytest tests/test_privacy.py

# Run API tests
pytest tests/test_api.py
```

### Code Structure

```
src/
├── federated_backend.py      # Main backend orchestration
├── enhanced_client.py        # Privacy-aware client
├── secure_aggregation.py     # Cryptographic aggregation
├── api_server.py            # REST API and WebSocket
├── privacy_engine.py        # Differential privacy
├── launcher.py              # System launcher
├── config.py               # Configuration management
├── model.py                # Model definitions
└── data.py                 # Data loading utilities
```

### Adding New Features

1. **New Privacy Mechanisms**: Extend `privacy_engine.py`
2. **New Aggregation Methods**: Modify `secure_aggregation.py`
3. **New API Endpoints**: Add to `api_server.py`
4. **New Client Features**: Extend `enhanced_client.py`

## 🔧 Troubleshooting

### Common Issues

1. **Port Conflicts**: Change ports with `--api-port`, `--dashboard-port`
2. **Missing Dependencies**: Install with `pip install -r requirements_backend.txt`
3. **Privacy Budget Exhausted**: Increase ε or reduce training rounds
4. **Client Connection Issues**: Check server address and firewall

### Debug Mode

```bash
# Enable debug logging
python launcher.py --log-level DEBUG

# Start individual components for debugging
python -m federated_backend
python -m api_server
python enhanced_client.py client_1 --silo silo_1
```

### Health Checks

```bash
# API health check
curl "http://localhost:8000/health"

# System status
curl "http://localhost:8000/api/system/info"
```

## 📚 Advanced Usage

### Custom Privacy Mechanisms

```python
from privacy_engine import FederatedPrivacyEngine

# Create custom privacy engine
privacy_engine = FederatedPrivacyEngine(
    model=model,
    target_epsilon=0.5,
    noise_multiplier=2.0,
    client_id="custom_client"
)
```

### Custom Aggregation

```python
from secure_aggregation import SecureAggregator

# Create military-grade aggregator
aggregator = SecureAggregator(security_level="military")

# Encrypt and aggregate updates
secure_update = aggregator.encrypt_parameters(params, "client_1", 1)
result = aggregator.secure_aggregate([secure_update], 1)
```

### Real-time Monitoring

```python
import requests
import json

# Subscribe to real-time updates
response = requests.get("http://localhost:8000/api/status")
status = response.json()

print(f"Training Active: {status['training_active']}")
print(f"Current Round: {status['current_round']}")
print(f"Global Accuracy: {status['global_accuracy']:.4f}")
```

## 🏆 Production Deployment

### Docker Deployment

```bash
# Build and run with Docker
docker-compose up -d

# Scale clients
docker-compose up -d --scale client=5
```

### Environment Variables

```bash
export FL_SERVER_HOST=0.0.0.0
export FL_SERVER_PORT=8080
export API_SERVER_PORT=8000
export DASHBOARD_PORT=8502
export LOG_LEVEL=INFO
```

### Monitoring Setup

- **Prometheus**: Metrics collection
- **Grafana**: Visualization
- **ELK Stack**: Log aggregation
- **Alertmanager**: Alert management

---

**🔐 Privacy-First Federated Learning: Where Privacy Meets Performance** 🚀

Built with ❤️ for secure, privacy-preserving machine learning
