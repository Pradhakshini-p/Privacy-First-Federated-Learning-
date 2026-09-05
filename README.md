# Privacy-First Federated Learning Pipeline

[![Live Demo](https://img.shields.io/badge/Live_Demo-GitHub_Pages-2ea44f?style=for-the-badge&logo=github)](https://pradhakshini-p.github.io/Privacy-First-Federated-Learning-/)
[![CI](https://github.com/Pradhakshini-p/Privacy-First-Federated-Learning-/actions/workflows/ci.yml/badge.svg)](https://github.com/Pradhakshini-p/Privacy-First-Federated-Learning-/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Live portfolio:** https://pradhakshini-p.github.io/Privacy-First-Federated-Learning-/

Production-ready **federated learning** system for diabetes prediction with **differential privacy**, multi-hospital training, centralized baseline comparison, and a REST inference API.

Built for technical interviews, portfolio reviews, and HR demonstrations.

## Highlights

| Feature | Description |
|---------|-------------|
| **Healthcare use case** | Pima Indians Diabetes dataset (768 patients, 8 clinical features) |
| **Federated learning** | Flower framework with FedAvg across 3 hospital silos |
| **Differential privacy** | Opacus DP-SGD with gradient clipping and ε-budget tracking |
| **Baseline comparison** | Centralized vs federated evaluation with plots and metrics |
| **Inference API** | Flask REST API for single and batch predictions |
| **One-command demo** | Full pipeline: train → federate → evaluate |

## Architecture

```
 Hospital A          Hospital B          Hospital C
 (Client 1)          (Client 2)          (Client 3)
 Local data          Local data          Local data
 DP training         DP training         DP training
      \                  |                  /
       \                 |                 /
        -------- Model updates only --------
                          |
                   FL Server (FedAvg)
                          |
              Evaluation + Inference API
```

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/Pradhakshini-p/Privacy-First-Federated-Learning-.git
cd Privacy-First-Federated-Learning-
pip install -r requirements.txt
cp .env.example .env   # optional configuration
```

### Run Full Pipeline (Recommended for Demo)

```bash
python launch.py --mode full
```

This runs:
1. Centralized baseline training
2. Federated learning (3 hospitals, 5 rounds)
3. Evaluation with comparison plots

### Individual Commands

```bash
# Federated learning demo (server + 3 clients)
python launch.py --mode demo

# Centralized baseline
python launch.py --mode centralized --epochs 20

# Evaluation (requires trained models)
python launch.py --mode evaluate

# Inference API
python launch.py --mode api --port 5000

# CLI prediction
python predict.py
```

### API Example

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":{"Pregnancies":6,"Glucose":148,"BloodPressure":72,"SkinThickness":35,"Insulin":0,"BMI":33.6,"DiabetesPedigreeFunction":0.627,"Age":50}}'
```

## Project Structure

```
privacy-first-federated-learning/
├── launch.py              # Main entry point (all modes)
├── server.py              # Flower FL server
├── client.py              # Flower client with differential privacy
├── train_centralized.py   # Centralized baseline trainer
├── evaluate.py            # Model comparison and plots
├── api.py                 # Flask inference API
├── predict.py             # CLI prediction tool
├── src/
│   ├── model.py           # PyTorch MLP/CNN models
│   ├── data.py            # Data loading and hospital silos
│   └── privacy.py         # Opacus differential privacy
├── data/
│   └── diabetes.csv       # Pima Indians Diabetes dataset
├── tests/                 # Smoke tests
├── docs/                  # Deployment and integration guides
├── requirements.txt       # Runtime dependencies
└── requirements-dev.txt   # Development and CI dependencies
```

See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for details.

## Expected Results

| Metric | Centralized | Federated (with DP) |
|--------|-------------|---------------------|
| Accuracy | ~70–73% | ~65–72% |
| Training time | ~1–2 min | ~5–10 min |
| Privacy | N/A | ε ≈ 3.0 per client |

## Docker

```bash
# Local
docker compose up server client1 client2 client3

# Pull pre-built image from GitHub Container Registry
docker pull ghcr.io/pradhakshini-p/privacy-first-federated-learning-:latest
docker run -p 5000:5000 ghcr.io/pradhakshini-p/privacy-first-federated-learning-:latest python api.py
```

### Cloud Deploy (Render)

1. Fork or connect this repo on [Render](https://render.com)
2. Use the included `render.yaml` blueprint, or create a **Web Service**:
   - **Build:** `pip install -r requirements.txt && python train_centralized.py --epochs 10`
   - **Start:** `python api.py --host 0.0.0.0 --port $PORT`
3. Health check path: `/health`

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for production deployment.

## Development

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
python train_centralized.py --epochs 2   # quick smoke test
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Interview Talking Points

1. **Distributed systems** — Multi-client Flower server with FedAvg aggregation; raw data never leaves hospitals.
2. **Privacy engineering** — Opacus DP-SGD with gradient clipping, noise injection, and ε-budget tracking.
3. **ML engineering** — PyTorch MLP, train/val/test splits, metrics (accuracy, F1, ROC-AUC).
4. **Full stack** — CLI launcher, REST API, automated evaluation pipeline.

See [DEMO_GUIDE.md](DEMO_GUIDE.md) for a 5-minute interview script.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- [Flower](https://flower.dev/) — Federated learning framework
- [Opacus](https://opacus.ai/) — Differential privacy for PyTorch
- [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

## Links

- **Live portfolio site:** https://pradhakshini-p.github.io/Privacy-First-Federated-Learning-/
- **Repository**: [github.com/Pradhakshini-p/Privacy-First-Federated-Learning-](https://github.com/Pradhakshini-p/Privacy-First-Federated-Learning-)
- **CI / Deploy**: [GitHub Actions](https://github.com/Pradhakshini-p/Privacy-First-Federated-Learning-/actions)
- **Docker image**: `ghcr.io/pradhakshini-p/privacy-first-federated-learning-:latest`
- **Issues**: [GitHub Issues](https://github.com/Pradhakshini-p/Privacy-First-Federated-Learning-/issues)
