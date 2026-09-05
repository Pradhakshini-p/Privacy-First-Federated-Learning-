# Changelog

All notable changes to this project are documented here.

## [1.0.0] - 2026-09-05

### Added
- Unified `launch.py` entry point with `full`, `demo`, `server`, `client`, `centralized`, `evaluate`, and `api` modes
- Flower federated learning server and client with FedAvg
- Differential privacy via Opacus (DP-SGD, gradient clipping, ε tracking)
- Centralized baseline training and comparison evaluation
- Flask inference API and CLI `predict.py`
- GitHub Actions CI workflow
- Docker Compose for multi-service deployment
- Smoke tests in `tests/`

### Fixed
- Pima Indians Diabetes dataset preprocessing (zero-as-missing imputation)
- Flower 1.8 metrics aggregation compatibility
- Windows gRPC server binding (`127.0.0.1:8080`)
- Opacus privacy engine per-round initialization
- Global model saving after federated training

### Changed
- Simplified MLP architecture (removed BatchNorm for DP compatibility)
- Pinned `numpy<2.0` for Flower compatibility
- Professional README and project documentation for GitHub deployment
