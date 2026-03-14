# Privacy-First Federated Learning Pipeline

A comprehensive, production-ready federated learning system for privacy-preserving fraud detection using Flower framework and PyTorch.

## 🌟 Features

- **🌸 Federated Learning**: Multiple clients train locally without sharing raw data
- **🛡️ Differential Privacy**: Formal privacy guarantees using Opacus
- **📊 Real-time Dashboard**: Live monitoring with Streamlit
- **🐳 Containerized**: Docker support for easy deployment
- **☁️ Cloud Ready**: AWS deployment configuration
- **🤖 Multiple Models**: MLP and CNN architectures
- **📈 Comprehensive Metrics**: Privacy-accuracy trade-off analysis

## 🏗️ Project Structure

```
Privacy-First Federated Learning Pipeline/
├── server.py              # Flower server with FedAvg strategy
├── client.py              # Standard Flower client
├── privacy_client.py      # Privacy-enhanced client with Opacus
├── model.py               # PyTorch models (MLP/CNN)
├── data.py                # Data loading and silo creation
├── dashboard.py           # Streamlit dashboard for visualization
├── demo.py                # Comprehensive demo script
├── Dockerfile             # Container configuration
├── docker-compose.yml     # Standard deployment
├── docker-compose.privacy.yml  # Privacy-focused deployment
├── aws-deployment-guide.md    # AWS deployment instructions
├── requirements.txt       # Python dependencies
├── logs/                  # Training logs and metrics
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <repository-url>
cd "Privacy-First Federated Learning Pipeline"

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Basic Demo
```bash
# Quick demo with all features
python demo.py --demo basic

# Or run components manually
python server.py --model mlp --rounds 5
# In separate terminals:
python client.py 1 --silo silo_1
python client.py 2 --silo silo_2
python client.py 3 --silo silo_3
```

### 3. Run Privacy-Enhanced Demo
```bash
# Demo with differential privacy
python demo.py --demo privacy

# Or run manually with different privacy settings
python server.py --model mlp --rounds 5
python privacy_client.py 1 --silo silo_1 --noise-multiplier 0.5
python privacy_client.py 2 --silo silo_2 --noise-multiplier 1.0
python privacy_client.py 3 --silo silo_3 --noise-multiplier 2.0
```

### 4. View Dashboard
```bash
# Start dashboard (after some training)
streamlit run dashboard.py
# Visit: http://localhost:8501
```

## 🐳 Docker Deployment

### Standard Deployment
```bash
# Build and run all containers
docker-compose up --build

# Or run specific services
docker-compose up server
docker-compose up client1 client2 client3
docker-compose up dashboard
```

### Privacy-Enhanced Deployment
```bash
# Run with all clients using differential privacy
docker-compose -f docker-compose.privacy.yml up --build
```

## ☁️ AWS Cloud Deployment

### Quick Deployment
```bash
# Follow the AWS deployment guide
cat aws-deployment-guide.md

# Summary:
# 1. Launch EC2 instance
# 2. Configure security groups (ports 8080, 8501)
# 3. Install Docker
# 4. Deploy application
# 5. Connect clients from anywhere
```

## 📊 Dashboard Features

The Streamlit dashboard provides:

- **🎯 Real-time Metrics**: Global accuracy and loss tracking
- **🤖 Client Performance**: Individual client comparison
- **🛡️ Privacy Analysis**: Privacy-accuracy trade-off visualization
- **📋 Training History**: Detailed logs and timestamps
- **🔄 Auto-refresh**: Live monitoring

## 🔬 Privacy Features

### Differential Privacy with Opacus

- **Gaussian Noise**: Added to gradients for privacy
- **Gradient Clipping**: Limits sensitivity of individual samples
- **Privacy Budget (ε)**: Tracks cumulative privacy loss
- **Configurable Parameters**: 
  - `--noise-multiplier`: Privacy level (higher = more privacy)
  - `--max-grad-norm`: Gradient clipping threshold
  - `--target-delta`: Failure probability

### Privacy-Accuracy Trade-off

```
Higher Noise Multiplier → More Privacy → Lower Accuracy
Lower Max Grad Norm     → More Privacy → Lower Accuracy
```

## 🧠 Model Architectures

### Multi-Layer Perceptron (MLP)
- **Input**: 30 features (credit card transaction data)
- **Hidden Layers**: [128, 64, 32] with BatchNorm and Dropout
- **Output**: 2 classes (fraud/not fraud)

### 1D Convolutional Neural Network (CNN)
- **Input**: Sequential treatment of features
- **Conv Layers**: [64, 128, 256] filters
- **Global Pooling**: Adaptive average pooling
- **Classifier**: Fully connected layers

## 📈 Usage Examples

### Basic Federated Learning
```bash
# Start server
python server.py --model mlp --rounds 10 --min-clients 3

# Start clients
python client.py 1 --silo silo_1 --model mlp
python client.py 2 --silo silo_2 --model mlp  
python client.py 3 --silo silo_3 --model mlp
```

### Privacy-Enhanced Learning
```bash
# Start server
python server.py --model mlp --rounds 10

# Start private clients with different privacy levels
python privacy_client.py 1 --noise-multiplier 0.5 --max-grad-norm 1.0
python privacy_client.py 2 --noise-multiplier 1.0 --max-grad-norm 0.8
python privacy_client.py 3 --noise-multiplier 2.0 --max-grad-norm 0.5
```

### Remote Client Connection
```bash
# Connect to remote server
python client.py 1 --server AWS_EC2_PUBLIC_IP:8080 --silo silo_1
```

## 📊 Monitoring and Logs

### Training Logs
- **Location**: `logs/training_log.json`
- **Content**: Global accuracy, loss, timestamps
- **Format**: JSON for easy parsing

### Client Metrics
- **Location**: `logs/client_metrics.json`
- **Content**: Individual client performance, privacy metrics
- **Usage**: Dashboard visualization

### Real-time Monitoring
```bash
# View live logs
tail -f logs/training_log.json

# Monitor with dashboard
streamlit run dashboard.py
```

## 🔧 Configuration

### Server Parameters
```bash
python server.py --model mlp --rounds 10 --min-clients 3
```

### Client Parameters
```bash
python client.py 1 --silo silo_1 --model mlp --server localhost:8080
```

### Privacy Parameters
```bash
python privacy_client.py 1 \
  --noise-multiplier 1.0 \
  --max-grad-norm 1.0 \
  --target-delta 1e-5
```

## 🎯 Performance Benchmarks

### Typical Results (Synthetic Data)
- **Non-Private Accuracy**: ~85-90%
- **Private Accuracy** (ε=1.0): ~80-85%
- **Training Time**: 2-5 minutes per round
- **Memory Usage**: ~500MB per client

### Privacy Budget Consumption
- **Low Privacy** (noise=0.5): ε ≈ 2.0 per round
- **Medium Privacy** (noise=1.0): ε ≈ 1.0 per round  
- **High Privacy** (noise=2.0): ε ≈ 0.5 per round

## 🛠️ Development

### Adding New Models
```python
# In model.py
class NewModel(nn.Module):
    def __init__(self, input_dim=30):
        # Your model architecture
        
# Update create_model() function
def create_model(model_type="new_model", input_dim=30):
    if model_type == "new_model":
        return NewModel(input_dim)
```

### Custom Privacy Mechanisms
```python
# In privacy_client.py
class CustomPrivacyClient(PrivateFraudDetectionClient):
    def _apply_privacy(self, gradients):
        # Your custom privacy mechanism
        return private_gradients
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test: `python demo.py --demo basic`
4. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Flower Framework**: Federated learning infrastructure
- **Opacus**: Differential privacy for PyTorch
- **Streamlit**: Dashboard visualization
- **PyTorch**: Deep learning framework

## 📞 Support

- **Issues**: GitHub Issues
- **Documentation**: Check README and inline comments
- **Demo**: Run `python demo.py` for examples

---

🌸 **Built with Flower Framework for production-ready federated learning** 🌸
