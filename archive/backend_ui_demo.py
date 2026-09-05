#!/usr/bin/env python3
"""
Backend-UI Integration Demo
Shows how the backend system works with the dashboard UI
"""

import streamlit as st
import requests
import json
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Backend-UI Integration Demo",
    page_icon="🔗",
    layout="wide"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

def get_api_status():
    """Get backend API status"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def get_training_status():
    """Get training status from backend"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/status", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def get_privacy_status():
    """Get privacy status from backend"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/privacy/status", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def get_global_metrics():
    """Get global training metrics"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/metrics/global", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def start_training(num_rounds=5, learning_rate=0.01):
    """Start training via backend API"""
    try:
        config = {"num_rounds": num_rounds, "learning_rate": learning_rate}
        response = requests.post(f"{API_BASE_URL}/api/training/start", 
                             json=config, timeout=10)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def stop_training():
    """Stop training via backend API"""
    try:
        response = requests.post(f"{API_BASE_URL}/api/training/stop", timeout=10)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def update_privacy_config(epsilon=1.0, noise_multiplier=1.0):
    """Update privacy configuration"""
    try:
        config = {"epsilon": epsilon, "noise_multiplier": noise_multiplier}
        response = requests.post(f"{API_BASE_URL}/api/privacy/config", 
                             json=config, timeout=10)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def main():
    """Main dashboard application"""
    st.title("🔗 Backend-UI Integration Demo")
    st.markdown("---")
    
    # Sidebar for system status
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Check API connection
        api_status = get_api_status()
        if api_status:
            st.success("✅ Backend API Connected")
            st.json(api_status)
        else:
            st.error("❌ Backend API Not Connected")
            st.warning("Please start the backend first:")
            st.code("python launcher.py --mode api")
            return
        
        st.markdown("---")
        
        # Training controls
        st.header("🚀 Training Controls")
        
        if st.button("🎯 Start Training", type="primary"):
            with st.spinner("Starting training..."):
                result = start_training()
                if result:
                    st.success("Training started!")
                    st.json(result)
                else:
                    st.error("Failed to start training")
        
        if st.button("⏹️ Stop Training"):
            with st.spinner("Stopping training..."):
                result = stop_training()
                if result:
                    st.success("Training stopped!")
                    st.json(result)
                else:
                    st.error("Failed to stop training")
        
        st.markdown("---")
        
        # Privacy controls
        st.header("🔒 Privacy Controls")
        
        epsilon = st.slider("Epsilon (ε)", 0.1, 10.0, 1.0, 0.1)
        noise_multiplier = st.slider("Noise Multiplier", 0.1, 5.0, 1.0, 0.1)
        
        if st.button("🔄 Update Privacy Config"):
            with st.spinner("Updating privacy..."):
                result = update_privacy_config(epsilon, noise_multiplier)
                if result:
                    st.success("Privacy config updated!")
                    st.json(result)
                else:
                    st.error("Failed to update privacy config")
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📊 Training Status")
        
        # Get training status
        training_status = get_training_status()
        if training_status:
            # Display key metrics
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.metric(
                    "Current Round",
                    training_status.get("current_round", 0),
                    help="Current training round"
                )
            
            with metrics_col2:
                st.metric(
                    "Global Accuracy",
                    f"{training_status.get('global_accuracy', 0):.4f}",
                    help="Current global model accuracy"
                )
            
            with metrics_col3:
                st.metric(
                    "Active Clients",
                    training_status.get("active_clients", 0),
                    help="Number of active clients"
                )
            
            # Training progress
            if training_status.get("training_active", False):
                st.success("🟢 Training Active")
                progress = training_status.get("current_round", 0) / training_status.get("total_rounds", 1)
                st.progress(progress)
            else:
                st.info("⏸️ Training Inactive")
            
            # Show full status
            with st.expander("📋 Full Status Details"):
                st.json(training_status)
        else:
            st.warning("No training status available")
    
    with col2:
        st.header("🔒 Privacy Status")
        
        # Get privacy status
        privacy_status = get_privacy_status()
        if privacy_status:
            # Privacy metrics
            privacy_col1, privacy_col2 = st.columns(2)
            
            with privacy_col1:
                st.metric(
                    "Total Clients",
                    privacy_status.get("total_clients", 0),
                    help="Total number of clients"
                )
            
            with privacy_col2:
                st.metric(
                    "Active Clients",
                    privacy_status.get("active_clients", 0),
                    help="Currently active clients"
                )
            
            # Privacy budget
            budget_used = privacy_status.get("global_budget_used", 0)
            st.metric(
                "Privacy Budget Used",
                f"{budget_used:.2%}",
                help="Percentage of privacy budget used"
            )
            
            # Privacy budget gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = budget_used * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Privacy Budget (%)"},
                delta = {'reference': 80},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show full privacy status
            with st.expander("🔐 Privacy Details"):
                st.json(privacy_status)
        else:
            st.warning("No privacy status available")
    
    # Global metrics section
    st.header("📈 Global Training Metrics")
    
    global_metrics = get_global_metrics()
    if global_metrics:
        # Convert to DataFrame for visualization
        if global_metrics:
            df = pd.DataFrame(global_metrics)
            
            # Accuracy over time
            if not df.empty:
                fig_accuracy = px.line(
                    df, 
                    x='round_num', 
                    y='global_accuracy',
                    title='Global Accuracy Over Rounds',
                    markers=True
                )
                fig_accuracy.update_layout(
                    xaxis_title="Training Round",
                    yaxis_title="Accuracy",
                    yaxis=dict(range=[0, 1])
                )
                st.plotly_chart(fig_accuracy, use_container_width=True)
                
                # Loss over time
                fig_loss = px.line(
                    df, 
                    x='round_num', 
                    y='global_loss',
                    title='Global Loss Over Rounds',
                    markers=True,
                    color_discrete_sequence=['red']
                )
                fig_loss.update_layout(
                    xaxis_title="Training Round",
                    yaxis_title="Loss"
                )
                st.plotly_chart(fig_loss, use_container_width=True)
            else:
                st.info("No training metrics available yet")
    else:
        st.info("No global metrics available")
    
    # System architecture diagram
    st.header("🏗️ Backend-UI Architecture")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### **Backend Components**
        - 🤖 **Federated Backend**
        - 🔒 **Enhanced Clients**
        - 🔐 **Secure Aggregation**
        - 🌐 **API Server**
        """)
    
    with col2:
        st.markdown("""
        ### **Communication Flow**
        1. **UI → API**: REST calls
        2. **API → Backend**: Internal calls
        3. **Backend → Clients**: Flower protocol
        4. **Clients → Backend**: Model updates
        5. **Backend → UI**: Real-time updates
        """)
    
    with col3:
        st.markdown("""
        ### **UI Integration**
        - 📊 **Real-time Status**
        - 🎛️ **Training Controls**
        - 🔒 **Privacy Settings**
        - 📈 **Live Metrics**
        """)
    
    # API endpoint documentation
    st.header("📚 API Endpoints")
    
    endpoints_data = [
        {"Endpoint": "GET /health", "Description": "Health check", "Usage": "System status"},
        {"Endpoint": "GET /api/status", "Description": "Training status", "Usage": "Current training info"},
        {"Endpoint": "POST /api/training/start", "Description": "Start training", "Usage": "Begin federated learning"},
        {"Endpoint": "POST /api/training/stop", "Description": "Stop training", "Usage": "Halt training"},
        {"Endpoint": "GET /api/privacy/status", "Description": "Privacy status", "Usage": "Privacy budget info"},
        {"Endpoint": "POST /api/privacy/config", "Description": "Update privacy", "Usage": "Change privacy settings"},
        {"Endpoint": "GET /api/metrics/global", "Description": "Global metrics", "Usage": "Training metrics"},
        {"Endpoint": "WebSocket /ws", "Description": "Real-time updates", "Usage": "Live data streaming"}
    ]
    
    df_endpoints = pd.DataFrame(endpoints_data)
    st.dataframe(df_endpoints, use_container_width=True)
    
    # Auto-refresh
    st.markdown("---")
    st.info("🔄 Dashboard auto-refreshes every 5 seconds for real-time updates")

if __name__ == "__main__":
    main()
