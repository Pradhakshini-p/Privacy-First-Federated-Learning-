# 🔐 Privacy-First Federated Learning Platform

**Enterprise-Grade Distributed Machine Learning with Differential Privacy**

A production-ready federated learning platform that enables organizations to train machine learning models across decentralized data sources while maintaining strict privacy guarantees.

---

## ✨ Key Features

### 🎯 Core Capabilities
- **Federated Learning**: Distribute model training across multiple clients without centralizing data
- **Differential Privacy**: Built-in privacy guarantees using PyTorch Opacus
- **Real-time Monitoring**: Live dashboard with metrics and privacy budget tracking
- **Enterprise-Ready**: Production-grade code with proper error handling and logging
- **REST API**: Full-featured API for integration with external systems

### 🛡️ Privacy & Security
- **Differential Privacy**: ε-δ privacy guarantees
- **Secure Aggregation**: Encrypted model aggregation
- **Privacy Budget Tracking**: Real-time monitoring of privacy consumption
- **Gradient Clipping**: Protection against membership inference attacks

### 📊 Monitoring & Analytics
- **Real-time Dashboard**: Live training progress visualization
- **Comprehensive Metrics**: Accuracy, loss, client participation, privacy usage
- **Performance Analytics**: Track global model convergence
- **Historical Data**: Complete training history and metrics

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone or Download the Repository**
   ```bash
   cd Privacy-First\ Federated\ Learning\ Pipeline
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Application**
   ```bash
   python start.py
   ```
   Or directly:
   ```bash
   python app.py
   ```

4. **Access the Dashboard**
   - Open your browser to: **http://localhost:8000**
   - The dashboard will auto-open when you run `start.py`

---

## 📚 API Documentation

The platform provides a comprehensive REST API for programmatic access:

### Health & Status
- **GET** `/api/health` - Health check
- **GET** `/api/status` - Current system status

### Metrics
- **GET** `/api/metrics/global` - Global training metrics
- **GET** `/api/metrics/clients` - Client participation metrics
- **GET** `/api/metrics/privacy-budget` - Privacy budget metrics

### Privacy
- **GET** `/api/privacy/status` - Privacy budget information
- **POST** `/api/privacy/config` - Configure privacy parameters

### Training Control
- **POST** `/api/training/start` - Start training
- **POST** `/api/training/stop` - Stop training
- **POST** `/api/training/reset` - Reset training state

### Interactive API Docs
- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│           Privacy-First Federated Learning              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │  FastAPI Backend + Built-in Dashboard (HTML)    │  │
│  │  • REST API Endpoints                           │  │
│  │  • Real-time Status Updates                     │  │
│  │  • CORS Enabled for Easy Integration            │  │
│  └─────────────────────────────────────────────────┘  │
│                        │                                │
│  ┌────────────────────┴────────────────────┐           │
│  │                                         │           │
│  ▼                                         ▼           │
│ ┌──────────────────┐             ┌──────────────────┐  │
│ │   FL Framework   │             │  Privacy Engine  │  │
│ │  (Flower)        │             │  (Opacus)        │  │
│ │                  │             │                  │  │
│ │  • Aggregation   │             │  • Differential  │  │
│ │  • Model Sync    │             │    Privacy       │  │
│ │  • Client Mgmt   │             │  • Gradient      │  │
│ │                  │             │    Clipping      │  │
│ └──────────────────┘             └──────────────────┘  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Data Layer                                      │  │
│  │  • Sample datasets (CSV, numpy)                  │  │
│  │  • Configurable data loading                     │  │
│  │  • Privacy compliance                            │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 💾 Project Structure

```
Privacy-First Federated Learning Pipeline/
│
├── app.py                          # Main application (Backend + Frontend)
├── start.py                        # Quick start script
│
├── backend_api.py                  # Alternative API server
├── requirements.txt                # Python dependencies
├── config.json                     # Configuration file
│
├── data/                           # Sample datasets
│   ├── iris_data.csv
│   ├── diabetes.csv
│   ├── cancer_data.csv
│   └── ...
│
├── src/                            # Source code (optional)
│
├── docs/                           # Documentation
│   ├── DEPLOYMENT.md
│   ├── API.md
│   └── ...
│
├── examples/                       # Example implementations
│   ├── simple_demo.py
│   └── healthcare_demo.py
│
└── README.md                       # This file
```

---

## 🎓 Usage Examples

### Start Training via Browser
1. Open http://localhost:8000
2. Click "▶ Start Training"
3. Watch real-time metrics and charts update
4. Monitor privacy budget consumption
5. Stop or reset as needed

### Start Training via API
```bash
curl -X POST http://localhost:8000/api/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "num_rounds": 10,
    "num_clients": 5,
    "learning_rate": 0.01,
    "batch_size": 32
  }'
```

### Check Status
```bash
curl http://localhost:8000/api/status
```

### Get Metrics
```bash
curl http://localhost:8000/api/metrics/global
curl http://localhost:8000/api/metrics/privacy-budget
```

---

## ⚙️ Configuration

Edit `config.json` to customize:

```json
{
  "federated_learning": {
    "num_rounds": 10,
    "num_clients": 5,
    "learning_rate": 0.01,
    "batch_size": 32
  },
  "privacy": {
    "epsilon": 8.0,
    "delta": 1e-5,
    "max_grad_norm": 1.0
  },
  "server": {
    "port": 8000,
    "host": "0.0.0.0"
  }
}
```

---

## 🔬 Technical Stack

### Backend
- **FastAPI** - Modern, fast web framework
- **Uvicorn** - ASGI server for async support
- **Pydantic** - Data validation and settings management

### Machine Learning
- **PyTorch** - Deep learning framework
- **Flower** - Federated learning framework
- **Opacus** - Differential privacy library
- **scikit-learn** - ML utilities

### Privacy & Security
- **Cryptography** - Secure aggregation
- **PyCryptodome** - Cryptographic primitives

### Frontend
- **HTML5/CSS3/JavaScript** - Built-in dashboard
- **Chart.js** - Interactive visualizations
- **Tailwind CSS** - Responsive styling

---

## 📈 Performance Metrics

### Accuracy Tracking
- Real-time global model accuracy
- Per-round accuracy progression
- Client-wise accuracy metrics

### Privacy Monitoring
- ε-δ privacy budget consumption
- Privacy per round
- Privacy budget remaining

### System Performance
- Client participation rate
- Round completion time
- Model convergence speed

---

## 🔒 Privacy Guarantees

The platform implements:

1. **Differential Privacy (DP)**
   - Client-level DP with ε=8.0, δ=1e-5
   - Per-sample gradient clipping
   - Gaussian noise addition

2. **Secure Aggregation**
   - End-to-end encrypted model updates
   - No intermediate access to gradients

3. **Privacy Budget Management**
   - Real-time budget tracking
   - Accumulated privacy loss per round

---

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t federated-learning .
```

### Run Container
```bash
docker run -p 8000:8000 federated-learning
```

### Docker Compose
```bash
docker-compose up
```

---

## 📝 API Response Examples

### Status
```json
{
  "training_active": true,
  "current_round": 5,
  "total_rounds": 10,
  "global_accuracy": 0.8542,
  "active_clients": 5,
  "timestamp": "2024-04-18T10:30:45Z"
}
```

### Metrics
```json
{
  "metrics": [
    {
      "round": 1,
      "accuracy": 0.7200,
      "loss": 0.5432,
      "clients": 5
    },
    {
      "round": 2,
      "accuracy": 0.7850,
      "loss": 0.4721,
      "clients": 5
    }
  ]
}
```

---

## 🚨 Troubleshooting

### Port Already in Use
```bash
# Change port in environment or config
export API_PORT=8001
python app.py
```

### Missing Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### CORS Issues
- The platform has CORS enabled by default
- All origins are allowed (`allow_origins=["*"]`)

### Dashboard Not Loading
1. Check if API server is running
2. Try hard refresh (Ctrl+Shift+R)
3. Check browser console for errors
4. Verify port 8000 is accessible

---

## 📚 Documentation

- [API Documentation](docs/API.md) - Detailed API reference
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment
- [Architecture Guide](docs/PROJECT_STRUCTURE.md) - System design

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Advanced privacy mechanisms
- [ ] More FL strategies
- [ ] Enhanced visualization
- [ ] Performance optimization
- [ ] Additional datasets

---

## 📄 License

This project is provided as-is for research and educational purposes.

---

## 🎯 Production Readiness Checklist

✅ **Code Quality**
- Type hints throughout
- Comprehensive logging
- Error handling
- Clean architecture

✅ **Performance**
- Async support
- Efficient data structures
- Optimized algorithms

✅ **Monitoring**
- Health checks
- Metrics collection
- Status endpoints

✅ **Documentation**
- API docs (Swagger/ReDoc)
- README with examples
- Inline code comments

✅ **Deployment**
- Docker support
- Environment configuration
- Production defaults

---

## 🎓 For Recruiters

This project demonstrates:

1. **Full-Stack Development** - Backend (FastAPI), Frontend (HTML/JS), APIs
2. **Enterprise Architecture** - Scalable, maintainable, production-ready code
3. **ML/AI Integration** - PyTorch, Federated Learning, Differential Privacy
4. **Cloud-Native** - Docker, async operations, CORS, scalable design
5. **Best Practices** - Type hints, logging, error handling, documentation
6. **Real-time Systems** - Live updates, WebSocket ready, async I/O
7. **Data Engineering** - Privacy preservation, data handling, metrics
8. **DevOps** - Configuration management, environment setup, deployment

---

## 📞 Support

For issues or questions:
1. Check the documentation
2. Review API docs at http://localhost:8000/api/docs
3. Examine example code in `examples/`
4. Check logs for error messages

---

**Built with ❤️ - Privacy-First Federated Learning Platform**

*Making distributed ML accessible, secure, and privacy-preserving*
