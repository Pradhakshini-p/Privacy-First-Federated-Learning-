# 3-Client Privacy-First Federated Learning System
# Advanced Placement Level Implementation

## 🎯 Project Overview

This is a comprehensive **3-Client Privacy-First Federated Learning System** that demonstrates the fundamental trade-offs between privacy protection and model performance in a real-world scenario with Non-IID data distribution.

## 🏗️ System Architecture

### **3-Client Server-Client Model**
- **Server**: Central aggregator managing global model (CNN for CIFAR-10)
- **Client A**: Mobile device/hospital with Trucks-focused data
- **Client B**: Mobile device/hospital with Birds-focused data  
- **Client C**: Mobile device/hospital with Mixed class data

### **Privacy Mechanisms**
- **Differential Privacy**: Opacus library with Gaussian noise injection
- **Privacy Budget (ε)**: Real-time adjustable privacy protection
- **Secure Aggregation**: Individual weight obfuscation before transmission

## 📊 Non-IID Data Setup (Phase 1)

### **CIFAR-10 Partitioning Strategy**
```
Client A (Trucks-focused): 70% Trucks + 30% Other classes
Client B (Birds-focused): 70% Birds + 30% Other classes  
Client C (Mixed): Remaining data from all classes
```

### **Data Distribution Visualization**
```
Client A: 🚚🚚🚚🚚🚚🚚🚚 + 🎯🐦🐶🐱
Client B: 🐦🐦🐦🐦🐦🐦🐦 + 🚚🎯🐶🐱
Client C: 🎯🎯🎯 + 🚚🚚 + 🐦🐦 + 🐶🐶 + 🐱🐱
```

## 🔐 Differential Privacy Implementation (Phase 2)

### **Privacy Budget Configuration**
```python
privacy_config = PrivacyConfig(
    epsilon=3.0,        # Privacy budget (adjustable)
    delta=1e-5,         # Failure probability
    noise_multiplier=1.0,  # Gaussian noise scale
    max_grad_norm=1.0      # Gradient clipping
)
```

### **Privacy vs Accuracy Trade-off**
| ε (Privacy Budget) | Privacy Level | Expected Accuracy | Use Case |
|-------------------|--------------|------------------|----------|
| 0.1 - 1.0 | 🔒 Very High | 0.55 - 0.65 | Medical data |
| 1.0 - 3.0 | 🛡️ High | 0.65 - 0.75 | Personal data |
| 3.0 - 5.0 | ⚖️ Medium | 0.75 - 0.85 | Business data |
| 5.0 - 10.0 | ⚠️ Low | 0.85 - 0.90 | Public data |

### **Gaussian Noise Mechanism**
```python
# Before sending weights to server
noise = np.random.normal(0, sigma, weights.shape)
obfuscated_weights = weights + noise
# Server receives: obfuscated_weights (cannot see original)
```

## 🤖 Federated Loop Implementation (Phase 3)

### **Server Script (server.py)**
```python
class FederatedServer:
    def __init__(self, privacy_config):
        self.global_model = SimpleCNN()
        self.strategy = FedAvg()
    
    def aggregate_weights(self, client_weights):
        # Federated Averaging
        return weighted_average(client_weights)
```

### **Client Script (client.py)**
```python
class PrivacyPreservingClient:
    def train_locally(self, global_weights):
        # 1. Load global weights
        self.set_parameters(global_weights)
        
        # 2. Train on private data
        for epoch in range(3):
            self.train_step()
        
        # 3. Add differential privacy noise
        noisy_weights = self.add_dp_noise(self.get_parameters())
        
        # 4. Send only weights (never data!)
        return noisy_weights
```

## 📱 Dashboard Visualization (Phase 4)

### **Interactive Components**

#### **🌐 Network Map**
- Visual representation of 3 clients connected to central server
- Real-time status indicators (Training/Idle)
- Client data distribution labels

#### **🎛️ Epsilon Slider**
- Real-time privacy budget adjustment (0.1 - 10.0)
- Immediate impact on expected accuracy
- Privacy level indicators (Very High → Low)

#### **📈 Accuracy Graph**
- Live-updating line chart for all 3 clients
- Global model accuracy progression
- Round-by-round training progress

#### **📝 Privacy Log**
- Real-time obfuscation logging
- Example: `"Client A weights obfuscated with σ=0.15 noise. Uploading..."`
- Server aggregation confirmations

## 🚀 Quick Start

### **1. Install Dependencies**
```bash
pip install torch torchvision flwr opacus streamlit plotly
```

### **2. Run Core System**
```bash
python three_client_fl.py
```

### **3. Launch Dashboard**
```bash
streamlit run privacy_dashboard.py
```

### **4. Access Dashboard**
**🌐 http://localhost:8523**

## 🎯 Dashboard Features

### **Real-time Controls**
- **Epsilon Slider**: Adjust privacy budget in real-time
- **Training Controls**: Start/Pause/Reset federated training
- **Auto-refresh**: Live updates during training

### **Visualizations**
- **Network Topology**: 3-client communication map
- **Accuracy Progress**: Multi-client training curves
- **Privacy Trade-off**: ε vs Accuracy relationship
- **Privacy Log**: Real-time obfuscation messages

### **Status Monitoring**
- **Client Status**: Individual client training states
- **Server Status**: Aggregation and model updates
- **Privacy Metrics**: Current ε and noise levels

## 📊 Expected Results

### **Training Progress**
```
Round 1: Client A: 0.62, Client B: 0.52, Client C: 0.72, Global: 0.62
Round 5: Client A: 0.68, Client B: 0.58, Client C: 0.78, Global: 0.68
Round 10: Client A: 0.74, Client B: 0.64, Client C: 0.82, Global: 0.73
```

### **Privacy Impact**
```
ε = 1.0: High privacy, 10-15% accuracy reduction
ε = 3.0: Medium privacy, 5-10% accuracy reduction  
ε = 5.0: Low privacy, 2-5% accuracy reduction
ε = 10.0: Minimal privacy, <2% accuracy reduction
```

## 🔧 Technical Implementation

### **Model Architecture**
```python
class SimpleCNN(nn.Module):
    def __init__(self):
        # 3 Conv blocks + 2 FC layers
        # Optimized for CIFAR-10 (32x32 images)
        # ~1.2M parameters
```

### **Federated Learning Settings**
```python
config = {
    'num_clients': 3,
    'num_rounds': 10,
    'local_epochs': 3,
    'batch_size': 32,
    'learning_rate': 0.001
}
```

### **Privacy Settings**
```python
privacy_config = {
    'epsilon': 3.0,        # Adjustable via dashboard
    'delta': 1e-5,
    'noise_multiplier': 1.0,
    'max_grad_norm': 1.0
}
```

## 🎓 Educational Value

### **Advanced Placement Concepts**
- **Non-IID Data**: Real-world data heterogeneity
- **Differential Privacy**: Mathematical privacy guarantees
- **Federated Averaging**: Weight aggregation without data sharing
- **Privacy-Accuracy Trade-offs**: Empirical analysis

### **Industry Relevance**
- **Healthcare**: Hospital collaboration without patient data sharing
- **Mobile Devices**: On-device learning without data collection
- **Finance**: Fraud detection without transaction data sharing

## 🔮 Extensions

### **Advanced Privacy**
- **Secure Multi-Party Computation**: Cryptographic aggregation
- **Homomorphic Encryption**: Compute on encrypted gradients
- **Zero-Knowledge Proofs**: Verifiable privacy guarantees

### **Enhanced FL**
- **Async Federated Learning**: Handle client dropouts
- **Personalized FL**: Client-specific model customization
- **Cross-Silo FL**: Multi-organization federated learning

## 🏆 Project Highlights

✅ **3-Client Architecture**: Realistic federated scenario  
✅ **Non-IID Data**: Trucks/Birds/Mixed distribution  
✅ **Differential Privacy**: Adjustable ε with real-time impact  
✅ **Interactive Dashboard**: Live visualization and control  
✅ **Privacy Logging**: Real-time obfuscation tracking  
✅ **AP Level**: University-grade implementation  
✅ **Production Ready**: Scalable and deployable  

---

## 📁 File Structure

```
three_client_fl.py          # Core FL system
privacy_dashboard.py        # Interactive dashboard
requirements_privacy_fl.txt # Dependencies
PRIVACY_FL_README.md       # Documentation
```

---

**🔒 3-Client Privacy-First Federated Learning: Making Privacy Visible!** 🚀

Access the interactive dashboard: **http://localhost:8523** 🌐
