# Contributing

Thank you for your interest in contributing to the Privacy-First Federated Learning Pipeline.

## Development Setup

```bash
git clone https://github.com/Pradhakshini-p/Privacy-First-Federated-Learning-.git
cd privacy-first-federated-learning
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
cp .env.example .env
```

## Running Tests

```bash
pytest tests/ -v
python launch.py --mode centralized --epochs 2
```

## Code Style

- Follow existing naming and module layout
- Keep changes focused and minimal
- Run `black` and `flake8` before submitting

## Pull Request Process

1. Fork the repository and create a feature branch
2. Add tests for new functionality
3. Ensure CI passes (`pytest`, smoke training)
4. Update documentation if behavior changes
5. Open a PR with a clear description and test plan

## Reporting Issues

Include:
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs from `logs/`

## Security

Do not commit secrets, API tokens, or `.env` files. Report security issues privately to the maintainer.
