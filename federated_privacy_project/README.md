# Privacy-Preserving Federated Learning System

A complete Python project that simulates a privacy-preserving federated learning pipeline with multiple clients, a central server, local training, differential privacy, secure aggregation simulation, and visualization.

## Project Structure

```
federated_privacy_project/
├── client.py          # Simulated federated learning client implementation
├── server.py          # Central server and optional Flask API
├── model.py           # PyTorch model definitions (MLP and CNN)
├── utils.py           # Dataset loading, non-IID splitting, DP noise, aggregation, plotting
├── main.py            # Full simulation entrypoint
├── requirements.txt   # Dependencies for this project
└── README.md          # This documentation file
```

## What This Project Does

- Simulates **5 federated clients** training locally on non-IID data
- Uses **FedAvg** to aggregate client model weights on the central server
- Adds **differential privacy noise** to client updates before sharing
- Simulates **secure aggregation** by masking client updates
- Uses a **digit classification dataset** (scikit-learn digits)
- Tracks global accuracy and loss across communication rounds
- Plots results in `plots/global_metrics.png`

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the simulation:
   ```bash
   python main.py
   ```

3. Optional CLI flags:
   ```bash
   python main.py --rounds 8 --clients 5 --epochs 2 --model mlp --dp-epsilon 2.0
   ```

4. Start the optional Flask API:
   ```bash
   python main.py --enable-api
   ```

   The API will run on `http://127.0.0.1:5000` by default.

## Architecture Diagram

```
Client 1          Client 2          Client 3         Client 4          Client 5
   │                 │                │                │                 │
   │ Local training  │ Local training │ Local training │ Local training  │ Local training
   │ (MLP/CNN)       │ (MLP/CNN)      │ (MLP/CNN)      │ (MLP/CNN)       │ (MLP/CNN)
   │                 │                │                │                 │
   ▼                 ▼                ▼                ▼                 ▼
Masked updates ---> Secure Aggregation Simulation ---> Global server
                    (DP noise + masks)                   │
                                                         │
                                          Aggregate model weights (FedAvg)
                                                         │
                                                         ▼
                                                Updated global model
                                                         │
                                                         ▼
                                            New weights sent to each client
```

## Privacy Protection Explained

### How user privacy is protected

- **Local training only**: each client trains on its own data and never shares raw records.
- **Differential privacy simulation**: clients add Gaussian noise to their model weights before sending them to the server.
- **Secure aggregation simulation**: clients mask their weight updates and the server removes the masks only after aggregation, reducing the risk of inferring individual updates.

### Why this is better than centralized ML

Centralized ML collects raw data in one place, which can expose:
- sensitive personal information
- data breaches
- regulatory non-compliance

Federated learning keeps raw data local and only shares **model updates**, significantly reducing privacy risk.

## Files Overview

### `model.py`
Contains two model definitions:
- `SimpleMLP`: feed-forward network for 64-dimensional digit features
- `SimpleCNN`: convolutional network for image-shaped digits

### `utils.py`
Includes utilities for:
- loading the digits dataset
- splitting data into **non-IID client silos**
- adding **differential privacy noise**
- simulating **secure aggregation**
- plotting accuracy and loss

### `client.py`
Defines `FederatedClient`:
- loads local training and validation data
- performs local model training
- applies DP noise and a secure mask
- returns data ready for aggregation

### `server.py`
Defines `FederatedServer`:
- aggregates client updates with FedAvg
- evaluates the global model on a held-out test set
- optionally starts a Flask API for status and payload testing

### `main.py`
Runs the full simulation:
- constructs clients and server
- executes federated rounds
- saves metrics plots
- prints performance summaries

## Advanced Usage

### Change the model to CNN

```bash
python main.py --model cnn
```

### Increase privacy noise

```bash
python main.py --dp-epsilon 1.0
```

Lower epsilon means more noise and stronger privacy.

### More rounds and local epochs

```bash
python main.py --rounds 10 --epochs 2
```

## Project Suitability

This project is ideal for:
- resumes
- final year projects
- research demos
- technical interviews

## Notes

- The current implementation uses a built-in digits dataset to avoid external downloads.
- Differential privacy is simulated for educational purposes.
- Secure aggregation is implemented as a basic proof-of-concept.

## Next Improvements

- Add real CSV dataset support
- Add an actual secure aggregation protocol
- Enable real networked clients instead of in-process simulation
- Add a web dashboard for live visualization
- Deploy as Docker container
