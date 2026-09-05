# Project Structure

## Overview

```
privacy-first-federated-learning/
├── launch.py                 # Main CLI entry point
├── server.py                 # Flower federated server
├── client.py                 # Flower client with DP
├── train_centralized.py      # Centralized baseline
├── evaluate.py               # Model comparison
├── api.py                    # Flask inference API
├── predict.py                # CLI predictions
├── requirements.txt          # Runtime dependencies
├── requirements-dev.txt      # Dev/test dependencies
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .github/workflows/ci.yml
├── src/
│   ├── __init__.py
│   ├── model.py              # MLP and CNN models
│   ├── data.py               # Dataset loading and silos
│   └── privacy.py            # Opacus differential privacy
├── data/
│   └── diabetes.csv          # Pima Indians Diabetes dataset
├── tests/
│   └── test_smoke.py         # Unit/smoke tests
├── docs/
│   ├── DEPLOYMENT.md
│   ├── INTEGRATION.md
│   └── PROJECT_STRUCTURE.md
├── results/                  # Generated plots and reports (gitignored)
├── models/                   # Saved model weights (gitignored)
├── logs/                     # Training logs (gitignored)
└── archive/                  # Legacy experiments (reference only)
```

## Entry Points

| File | Purpose |
|------|---------|
| `launch.py --mode full` | Complete HR demo pipeline |
| `launch.py --mode demo` | Federated training only |
| `launch.py --mode api` | Start inference server |
| `predict.py` | Single prediction from CLI |

## Data Flow

1. `src/data.py` loads and preprocesses diabetes.csv
2. Data split into 3 hospital silos + global test set
3. `client.py` trains locally; sends model weights only
4. `server.py` aggregates with FedAvg
5. `evaluate.py` compares centralized vs federated models
6. `api.py` serves predictions from saved model

## Configuration

Copy `.env.example` to `.env` to override defaults:

- `FL_SERVER_ADDRESS` — Flower server bind address
- `FL_ROUNDS` — Number of federated rounds
- `PRIVACY_EPSILON` — DP budget per client
