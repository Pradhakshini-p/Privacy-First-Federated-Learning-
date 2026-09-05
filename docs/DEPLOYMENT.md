# Deployment Guide

## Local Development

```bash
pip install -r requirements.txt
python launch.py --mode full
```

## Docker Compose

Start the full federated stack:

```bash
docker compose up --build server client1 client2 client3
```

Start inference API (after models are trained):

```bash
docker compose up api
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FL_SERVER_ADDRESS` | `127.0.0.1:8080` | Flower server address |
| `FL_ROUNDS` | `5` | Federated learning rounds |
| `FL_MIN_CLIENTS` | `3` | Minimum participating clients |
| `API_PORT` | `5000` | Inference API port |
| `PRIVACY_EPSILON` | `3.0` | DP privacy budget |

## Production Notes

- Use `127.0.0.1:8080` on Windows; `0.0.0.0:8080` in Docker/Linux
- Train models before starting the API (`models/global_model.pth`)
- Pin dependencies: `numpy<2.0` required for Flower 1.8
- Do not commit `.env` or model files with sensitive data

## CI/CD

GitHub Actions runs on push/PR:

- **CI** (`.github/workflows/ci.yml`): smoke tests + training validation
- **Deploy** (`.github/workflows/deploy.yml`): Docker image → GitHub Container Registry + release artifacts

See `.github/workflows/` for details.
