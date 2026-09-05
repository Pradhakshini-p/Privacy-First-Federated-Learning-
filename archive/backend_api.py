from fastapi import FastAPI
import random
import time

app = FastAPI()

# Dummy training state
training_active = False
current_round = 0

@app.get("/health")
def health():
    return {"status": "running"}

@app.get("/api/status")
def status():
    global current_round
    if training_active:
        current_round += 1
    
    return {
        "current_round": current_round,
        "total_rounds": 10,
        "global_accuracy": round(random.uniform(0.7, 0.9), 4),
        "active_clients": 5,
        "training_active": training_active
    }

@app.get("/api/privacy/status")
def privacy():
    return {
        "total_clients": 5,
        "active_clients": 5,
        "global_budget_used": random.uniform(0.2, 0.8)
    }

@app.get("/api/metrics/global")
def metrics():
    data = []
    for i in range(1, current_round + 1):
        data.append({
            "round_num": i,
            "global_accuracy": random.uniform(0.7, 0.9),
            "global_loss": random.uniform(0.3, 0.6)
        })
    return data

@app.post("/api/training/start")
def start():
    global training_active, current_round
    training_active = True
    current_round = 0
    return {"message": "Training started"}

@app.post("/api/training/stop")
def stop():
    global training_active
    training_active = False
    return {"message": "Training stopped"}

@app.post("/api/privacy/config")
def config(data: dict):
    return {"message": "Privacy updated", "config": data}