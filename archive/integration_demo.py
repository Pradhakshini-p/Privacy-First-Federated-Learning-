#!/usr/bin/env python3
"""
Backend-UI Integration Demonstration
Shows how the backend system works with the UI through API calls
"""

import streamlit as st
import time
import json
import random
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Backend-UI Integration Demo",
    page_icon="🔗",
    layout="wide"
)

# Simulated backend data (for demo purposes)
def simulate_backend_status():
    """Simulate backend API responses"""
    return {
        "training_active": random.choice([True, False]),
        "current_round": random.randint(0, 10),
        "total_rounds": 10,
        "global_accuracy": random.uniform(0.7, 0.95),
        "global_loss": random.uniform(0.1, 0.5),
        "active_clients": random.randint(2, 5),
        "total_clients": 5,
        "privacy_budget_used": random.uniform(0.0, 1.0),
        "timestamp": datetime.now().isoformat()
    }

def simulate_privacy_status():
    """Simulate privacy API responses"""
    return {
        "total_clients": 5,
        "active_clients": random.randint(2, 5),
        "total_epsilon_spent": random.uniform(0.5, 2.5),
        "total_epsilon_budget": 3.0,
        "global_budget_used": random.uniform(0.2, 0.9),
        "exhausted_clients": random.randint(0, 1),
        "security_level": "standard"
    }

def simulate_training_metrics():
    """Simulate training metrics API response"""
    rounds = list(range(1, 11))
    accuracy = [0.65 + (i * 0.03) + random.uniform(-0.02, 0.02) for i in rounds]
    loss = [0.8 - (i * 0.06) + random.uniform(-0.05, 0.05) for i in rounds]
    
    return [
        {
            "round_num": r,
            "global_accuracy": a,
            "global_loss": l,
            "timestamp": datetime.now().isoformat()
        }
        for r, a, l in zip(rounds, accuracy, loss)
    ]

def main():
    """Main demonstration application"""
    st.title("🔗 Backend-UI Integration Demo")
    st.markdown("---")
    
    # Introduction
    st.markdown("""
    ## 🎯 How Backend Works with UI
    
    This demo shows the **integration between the federated learning backend and the dashboard UI**:
    
    ### 📡 Communication Flow:
    1. **UI → Backend**: User actions (start/stop training, privacy settings)
    2. **Backend → UI**: Real-time updates (training progress, privacy status)
    3. **Continuous Sync**: Auto-refresh every 5 seconds
    
    ### 🔌 API Integration:
    - **REST API**: HTTP requests for commands and data
    - **WebSocket**: Real-time streaming updates
    - **JSON Format**: Structured data exchange
    """)
    
    # Simulate API connection status
    with st.sidebar:
        st.header("🔧 Simulated Backend")
        
        # Simulate connection status
        connection_status = st.selectbox(
            "Backend Connection",
            ["Connected", "Disconnected", "Error"],
            index=0
        )
        
        if connection_status == "Connected":
            st.success("✅ Backend API Connected")
            st.info("Simulating real API responses")
        elif connection_status == "Disconnected":
            st.error("❌ Backend API Not Connected")
            st.warning("Please start the backend first")
            return
        else:
            st.error("🔥 Backend API Error")
            st.warning("Check backend logs")
            return
        
        st.markdown("---")
        
        # Simulate training controls
        st.header("🚀 Training Controls")
        
        if st.button("🎯 Start Training", type="primary"):
            st.success("✅ Training started!")
            st.balloons()
        
        if st.button("⏹️ Stop Training"):
            st.warning("⏹️ Training stopped")
        
        st.markdown("---")
        
        # Simulate privacy controls
        st.header("🔒 Privacy Controls")
        
        epsilon = st.slider("Epsilon (ε)", 0.1, 10.0, 1.0, 0.1)
        noise_multiplier = st.slider("Noise Multiplier", 0.1, 5.0, 1.0, 0.1)
        
        if st.button("🔄 Update Privacy Config"):
            st.success(f"Privacy updated: ε={epsilon}, noise={noise_multiplier}")
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📊 Training Status")
        
        # Get simulated backend status
        training_status = simulate_backend_status()
        
        # Display key metrics
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric(
                "Current Round",
                training_status["current_round"],
                delta=1 if random.random() > 0.5 else 0,
                help="Current training round"
            )
        
        with metrics_col2:
            st.metric(
                "Global Accuracy",
                f"{training_status['global_accuracy']:.4f}",
                delta=f"{random.uniform(-0.01, 0.01):.4f}",
                help="Current global model accuracy"
            )
        
        with metrics_col3:
            st.metric(
                "Active Clients",
                training_status["active_clients"],
                help="Number of active clients"
            )
        
        # Training progress
        if training_status["training_active"]:
            st.success("🟢 Training Active")
            progress = training_status["current_round"] / training_status["total_rounds"]
            st.progress(progress)
            st.caption(f"Round {training_status['current_round']} of {training_status['total_rounds']}")
        else:
            st.info("⏸️ Training Inactive")
        
        # Show simulated API response
        with st.expander("📋 Simulated API Response"):
            st.json(training_status)
    
    with col2:
        st.header("🔒 Privacy Status")
        
        # Get simulated privacy status
        privacy_status = simulate_privacy_status()
        
        # Privacy metrics
        privacy_col1, privacy_col2 = st.columns(2)
        
        with privacy_col1:
            st.metric(
                "Total Clients",
                privacy_status["total_clients"],
                help="Total number of clients"
            )
        
        with privacy_col2:
            st.metric(
                "Active Clients",
                privacy_status["active_clients"],
                help="Currently active clients"
            )
        
        # Privacy budget gauge
        budget_used = privacy_status["global_budget_used"]
        st.metric(
            "Privacy Budget Used",
            f"{budget_used:.2%}",
            delta=f"{random.uniform(-0.05, 0.05):.2%}",
            help="Percentage of privacy budget used"
        )
        
        # Privacy budget gauge visualization
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = budget_used * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Privacy Budget (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
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
        
        # Show privacy details
        with st.expander("🔐 Privacy Details"):
            st.json(privacy_status)
    
    # Global metrics section
    st.header("📈 Global Training Metrics")
    
    # Get simulated training metrics
    training_metrics = simulate_training_metrics()
    
    if training_metrics:
        # Convert to DataFrame for visualization
        df = pd.DataFrame(training_metrics)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy over time
            fig_accuracy = px.line(
                df, 
                x='round_num', 
                y='global_accuracy',
                title='📊 Global Accuracy Over Rounds',
                markers=True,
                line_shape='spline'
            )
            fig_accuracy.update_layout(
                xaxis_title="Training Round",
                yaxis_title="Accuracy",
                yaxis=dict(range=[0.6, 1.0])
            )
            st.plotly_chart(fig_accuracy, use_container_width=True)
        
        with col2:
            # Loss over time
            fig_loss = px.line(
                df, 
                x='round_num', 
                y='global_loss',
                title='📉 Global Loss Over Rounds',
                markers=True,
                line_shape='spline',
                color_discrete_sequence=['red']
            )
            fig_loss.update_layout(
                xaxis_title="Training Round",
                yaxis_title="Loss"
            )
            st.plotly_chart(fig_loss, use_container_width=True)
    
    # API Integration Details
    st.header("🔌 API Integration Details")
    
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
    
    # API Endpoints Table
    st.header("📚 API Endpoints Used by UI")
    
    endpoints_data = [
        {"Endpoint": "GET /health", "Description": "Health check", "UI Usage": "System status"},
        {"Endpoint": "GET /api/status", "Description": "Training status", "UI Usage": "Current training info"},
        {"Endpoint": "POST /api/training/start", "Description": "Start training", "UI Usage": "Begin federated learning"},
        {"Endpoint": "POST /api/training/stop", "Description": "Stop training", "UI Usage": "Halt training"},
        {"Endpoint": "GET /api/privacy/status", "Description": "Privacy status", "UI Usage": "Privacy budget info"},
        {"Endpoint": "POST /api/privacy/config", "Description": "Update privacy", "UI Usage": "Change privacy settings"},
        {"Endpoint": "GET /api/metrics/global", "Description": "Global metrics", "UI Usage": "Training metrics"},
        {"Endpoint": "WebSocket /ws", "Description": "Real-time updates", "UI Usage": "Live data streaming"}
    ]
    
    df_endpoints = pd.DataFrame(endpoints_data)
    st.dataframe(df_endpoints, use_container_width=True)
    
    # Real-time Update Simulation
    st.header("🔄 Real-time Update Simulation")
    
    st.markdown("""
    ### **How Real-time Updates Work:**
    
    1. **Backend generates data** (training progress, privacy metrics)
    2. **API Server broadcasts** via WebSocket every 5 seconds
    3. **UI receives updates** and refreshes displays automatically
    4. **Charts update** without page refresh
    5. **Status indicators** change in real-time
    """)
    
    # Simulate real-time update
    if st.button("🔄 Simulate Real-time Update"):
        with st.spinner("Updating from backend..."):
            time.sleep(1)
            st.success("✅ Updated with latest backend data!")
            st.rerun()
    
    # Auto-refresh indicator
    st.markdown("---")
    st.info("🔄 **Auto-refresh**: Dashboard would auto-refresh every 5 seconds for real-time updates")
    
    # Instructions
    st.header("🚀 How to Run Real Integration")
    
    st.markdown("""
    ### **Step-by-Step Integration:**
    
    1. **Start Backend API Server:**
    ```bash
    cd src
    python launcher.py --mode api
    ```
    
    2. **Start Federated Learning:**
    ```bash
    python launcher.py --clients 3 --rounds 5
    ```
    
    3. **Start Dashboard:**
    ```bash
    streamlit run backend_ui_demo.py --server.port 8503
    ```
    
    4. **Access Integration Demo:**
    ```
    http://localhost:8503
    ```
    
    ### **What You'll See:**
    - ✅ Real backend API responses
    - 📊 Live training metrics
    - 🔒 Privacy budget tracking
    - 🎛️ Functional controls
    - 🔄 Real-time updates
    """)

if __name__ == "__main__":
    main()
