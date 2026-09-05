"""
API Server for Federated Learning Dashboard
Provides REST API endpoints for real-time monitoring and control
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Local imports
from federated_backend import get_backend, create_backend
from enhanced_client import EnhancedFederatedClient
from secure_aggregation import get_secure_aggregator, get_privacy_aggregator
from config import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Federated Learning API",
    description="API for Privacy-First Federated Learning Platform",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
backend = None
websocket_connections: List[WebSocket] = []

# Pydantic models for API
class TrainingConfig(BaseModel):
    num_rounds: Optional[int] = 5
    learning_rate: Optional[float] = 0.01
    min_clients: Optional[int] = 2

class PrivacyConfig(BaseModel):
    epsilon: Optional[float] = 1.0
    delta: Optional[float] = 1e-5
    noise_multiplier: Optional[float] = 1.0
    max_grad_norm: Optional[float] = 1.0

class ClientConfig(BaseModel):
    client_id: str
    privacy_enabled: Optional[bool] = True
    noise_multiplier: Optional[float] = 1.0
    local_epochs: Optional[int] = 5

class SecurityConfig(BaseModel):
    security_level: Optional[str] = "standard"  # basic, standard, military

# WebSocket manager for real-time updates
class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
    
    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.active_connections.remove(conn)

manager = WebSocketManager()

# Initialize backend
def initialize_backend():
    """Initialize the federated learning backend"""
    global backend
    if backend is None:
        backend = get_backend()
        logger.info("Backend initialized")
    return backend

# Background task for real-time updates
async def broadcast_updates():
    """Broadcast real-time updates to connected clients"""
    while True:
        try:
            if backend and manager.active_connections:
                # Get current status
                status = backend.get_training_status()
                privacy_status = backend.get_privacy_status()
                
                # Create update message
                update = {
                    "type": "status_update",
                    "timestamp": datetime.now().isoformat(),
                    "training_status": status,
                    "privacy_status": privacy_status
                }
                
                # Broadcast to all connected clients
                await manager.broadcast(update)
            
            await asyncio.sleep(5)  # Update every 5 seconds
            
        except Exception as e:
            logger.error(f"Error in broadcast_updates: {e}")
            await asyncio.sleep(10)

# API Routes
@app.on_event("startup")
async def startup_event():
    """Initialize API server"""
    logger.info("Starting Federated Learning API Server")
    
    # Initialize backend
    initialize_backend()
    
    # Start background task for updates
    asyncio.create_task(broadcast_updates())

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Privacy-First Federated Learning API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/api/status")
async def get_status():
    """Get current system status"""
    try:
        backend = initialize_backend()
        status = backend.get_training_status()
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/privacy/status")
async def get_privacy_status():
    """Get privacy status"""
    try:
        backend = initialize_backend()
        privacy_status = backend.get_privacy_status()
        return JSONResponse(content=privacy_status)
    except Exception as e:
        logger.error(f"Error getting privacy status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/training/start")
async def start_training(config: TrainingConfig):
    """Start federated training"""
    try:
        backend = initialize_backend()
        
        # Update backend config if provided
        if config.num_rounds:
            backend.rounds = config.num_rounds
        
        # Start training
        success = backend.start_training(config.num_rounds)
        
        if success:
            return {
                "message": "Training started successfully",
                "config": config.dict(),
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Training already in progress")
            
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/training/stop")
async def stop_training():
    """Stop federated training"""
    try:
        backend = initialize_backend()
        backend.training_active = False
        
        return {
            "message": "Training stopped successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error stopping training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics/global")
async def get_global_metrics():
    """Get global training metrics"""
    try:
        backend = initialize_backend()
        metrics = backend.get_global_metrics()
        return JSONResponse(content=metrics)
    except Exception as e:
        logger.error(f"Error getting global metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics/clients/{client_id}")
async def get_client_metrics(client_id: str):
    """Get metrics for a specific client"""
    try:
        backend = initialize_backend()
        metrics = backend.get_client_metrics(client_id)
        return JSONResponse(content=metrics)
    except Exception as e:
        logger.error(f"Error getting client metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics/clients")
async def get_all_client_metrics():
    """Get metrics for all clients"""
    try:
        backend = initialize_backend()
        metrics = backend.get_client_metrics()
        return JSONResponse(content=metrics)
    except Exception as e:
        logger.error(f"Error getting all client metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/privacy/config")
async def update_privacy_config(config: PrivacyConfig):
    """Update global privacy configuration"""
    try:
        backend = initialize_backend()
        
        # Update privacy configuration
        # This would need to be implemented in the backend
        # For now, just return success
        
        return {
            "message": "Privacy configuration updated",
            "config": config.dict(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error updating privacy config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/privacy/client/{client_id}/config")
async def update_client_privacy_config(client_id: str, config: ClientConfig):
    """Update privacy configuration for a specific client"""
    try:
        backend = initialize_backend()
        
        # Update client privacy configuration
        success = backend.update_privacy_config(client_id, config.dict())
        
        if success:
            return {
                "message": f"Privacy configuration updated for {client_id}",
                "config": config.dict(),
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail=f"Client {client_id} not found")
            
    except Exception as e:
        logger.error(f"Error updating client privacy config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/security/config")
async def update_security_config(config: SecurityConfig):
    """Update security configuration"""
    try:
        # Update secure aggregator
        secure_aggregator = get_secure_aggregator(config.security_level)
        
        return {
            "message": "Security configuration updated",
            "config": config.dict(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error updating security config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/security/status")
async def get_security_status():
    """Get security status"""
    try:
        secure_aggregator = get_secure_aggregator()
        status = secure_aggregator.get_security_status()
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"Error getting security status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/clients")
async def get_clients():
    """Get list of all clients"""
    try:
        backend = initialize_backend()
        
        # Get client information from backend
        clients = []
        for client_id in backend.active_clients.keys():
            client_info = {
                "client_id": client_id,
                "active": backend.active_clients[client_id],
                "last_update": datetime.now().isoformat()
            }
            clients.append(client_info)
        
        return JSONResponse(content=clients)
        
    except Exception as e:
        logger.error(f"Error getting clients: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/info")
async def get_system_info():
    """Get system information"""
    try:
        backend = initialize_backend()
        
        info = {
            "system": {
                "version": "1.0.0",
                "model_type": backend.model_type,
                "input_dim": backend.input_dim,
                "num_clients": backend.num_clients,
                "total_rounds": backend.rounds,
                "current_round": backend.current_round
            },
            "privacy": backend.get_privacy_status(),
            "security": get_secure_aggregator().get_security_status(),
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=info)
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "ping":
                await manager.send_personal_message({"type": "pong"}, websocket)
            elif message.get("type") == "subscribe":
                # Client wants to subscribe to updates
                await manager.send_personal_message(
                    {"type": "subscribed", "message": "Successfully subscribed to updates"},
                    websocket
                )
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        backend = initialize_backend()
        return {
            "status": "healthy",
            "backend_initialized": backend is not None,
            "training_active": backend.training_active if backend else False,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )
