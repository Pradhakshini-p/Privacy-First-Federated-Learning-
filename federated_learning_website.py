#!/usr/bin/env python3
"""
Federated Learning Website - Professional Design
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime
import random

def create_federated_learning_website():
    """Create a professional federated learning website"""
    
    # Page configuration
    st.set_page_config(
        page_title="Federated Learning Platform",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Professional CSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        color: #1f2937;
        margin-bottom: 2rem;
        padding: 1rem 0;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .sidebar-section {
        background: #f9fafb;
        padding: 1rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
    }
    
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #d1d5db;
    }
    
    .metric-box {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .metric-box:hover {
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transform: translateY(-1px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #3b82f6;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        font-weight: 500;
    }
    
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .tab-content {
        padding: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f9fafb;
        border-radius: 0.75rem;
        padding: 0.5rem;
        border: 1px solid #e5e7eb;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        color: #374151;
        border: 1px solid #e5e7eb;
        margin: 0 0.25rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
        border-color: #3b82f6;
    }
    
    .stSelectbox > div > div {
        background-color: white;
        border: 1px solid #d1d5db;
        border-radius: 0.5rem;
    }
    
    .stSlider > div > div {
        background-color: #3b82f6;
    }
    
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 0.5rem;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        width: 100%;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: #2563eb;
    }
    
    .data-table {
        background: white;
        border-radius: 0.75rem;
        overflow: hidden;
        border: 1px solid #e5e7eb;
    }
    
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    
    .status-running { background-color: #10b981; }
    .status-idle { background-color: #f59e0b; }
    .status-completed { background-color: #3b82f6; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">Federated Learning Platform</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = "Customer Churn Prediction"
        st.session_state.current_accuracy = 0.786
        st.session_state.current_round = 5
        st.session_state.current_clients = 10
        st.session_state.current_status = "idle"
        st.session_state.training_started = False
    
    # Sidebar
    with st.sidebar:
        # Dataset Selection
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">📊 Dataset Selection</div>', unsafe_allow_html=True)
        
        datasets = {
            "Customer Churn Prediction": {"accuracy": 0.786, "round": 5, "samples": 1000},
            "Medical Diagnosis (Diabetes)": {"accuracy": 0.823, "round": 6, "samples": 768},
            "House Price Prediction": {"accuracy": 0.791, "round": 4, "samples": 800},
            "Student Performance": {"accuracy": 0.847, "round": 7, "samples": 600},
            "Iris Flower Classification": {"accuracy": 0.923, "round": 8, "samples": 150}
        }
        
        selected_dataset = st.selectbox(
            "Select Dataset",
            list(datasets.keys()),
            index=list(datasets.keys()).index(st.session_state.current_dataset)
        )
        
        # Show dataset info
        dataset_info = datasets[selected_dataset]
        st.markdown(f"""
        **Dataset Info:**
        - Samples: {dataset_info['samples']:,}
        - Features: Auto-detected
        - Target: Classification
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Training Configuration
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">⚙️ Training Configuration</div>', unsafe_allow_html=True)
        
        n_clients = st.slider("Number of Clients", 2, 20, st.session_state.current_clients)
        n_rounds = st.slider("Training Rounds", 1, 20, 10)
        learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Privacy Settings
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">🔒 Privacy Settings</div>', unsafe_allow_html=True)
        
        enable_dp = st.checkbox("Enable Differential Privacy", value=True)
        if enable_dp:
            epsilon = st.slider("Privacy Budget (ε)", 0.1, 5.0, 1.0, 0.1)
        
        enable_encryption = st.checkbox("Enable Encryption", value=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Control Panel
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">🎮 Control Panel</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button("▶ Start", type="primary")
        with col2:
            stop_button = st.button("⏹ Stop")
        
        if start_button:
            st.session_state.current_dataset = selected_dataset
            st.session_state.current_accuracy = dataset_info["accuracy"]
            st.session_state.current_round = dataset_info["round"]
            st.session_state.current_clients = n_clients
            st.session_state.current_status = "running"
            st.session_state.training_started = True
            st.rerun()
        
        if stop_button:
            st.session_state.current_status = "idle"
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # System Status
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">📊 System Status</div>', unsafe_allow_html=True)
        
        status_color = "status-running" if st.session_state.current_status == "running" else "status-idle"
        st.markdown(f"""
        <span class="status-indicator {status_color}"></span>
        **Status:** {st.session_state.current_status.title()}
        
        **Active Clients:** {st.session_state.current_clients}
        
        **Current Round:** {st.session_state.current_round}/{n_rounds}
        
        **Accuracy:** {st.session_state.current_accuracy:.1%}
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard", "📈 Analytics", "🏢 Clients", "🔒 Privacy"
    ])
    
    with tab1:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{st.session_state.current_status.title()}</div>
                <div class="metric-label">System Status</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{st.session_state.current_round}/{n_rounds}</div>
                <div class="metric-label">Training Progress</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{st.session_state.current_clients}</div>
                <div class="metric-label">Active Clients</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{st.session_state.current_accuracy:.1%}</div>
                <div class="metric-label">Model Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Progress bar
        if st.session_state.training_started:
            progress = st.session_state.current_round / n_rounds
            st.progress(progress, text=f"Training Progress: {progress:.1%}")
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # Generate accuracy data
            rounds = list(range(1, n_rounds + 1))
            accuracy_data = []
            for i, round_num in enumerate(rounds):
                if round_num <= st.session_state.current_round:
                    progress = round_num / n_rounds
                    accuracy = 0.5 + (st.session_state.current_accuracy - 0.5) * progress + random.uniform(-0.02, 0.02)
                else:
                    accuracy = None
                accuracy_data.append(accuracy)
            
            # Filter out None values for completed rounds
            completed_rounds = [r for r, a in zip(rounds, accuracy_data) if a is not None]
            completed_accuracy = [a for a in accuracy_data if a is not None]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=completed_rounds,
                y=completed_accuracy,
                mode='lines+markers',
                name='Accuracy',
                line=dict(color='#3b82f6', width=3),
                marker=dict(size=8, color='#3b82f6')
            ))
            
            fig.update_layout(
                title="Training Accuracy",
                xaxis_title="Round",
                yaxis_title="Accuracy",
                template="plotly_white",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # Loss data
            loss_data = []
            for i, round_num in enumerate(rounds):
                if round_num <= st.session_state.current_round:
                    progress = round_num / n_rounds
                    loss = 1.0 - (progress * 0.7) + random.uniform(-0.05, 0.05)
                else:
                    loss = None
                loss_data.append(loss)
            
            completed_loss = [l for l in loss_data if l is not None]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=completed_rounds,
                y=completed_loss,
                mode='lines+markers',
                name='Loss',
                line=dict(color='#ef4444', width=3),
                marker=dict(size=8, color='#ef4444'),
                fill='tonexty',
                fillcolor='rgba(239, 68, 68, 0.1)'
            ))
            
            fig.update_layout(
                title="Training Loss",
                xaxis_title="Round",
                yaxis_title="Loss",
                template="plotly_white",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        
        # Performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # Client performance distribution
            client_names = [f'Client {i}' for i in range(1, st.session_state.current_clients + 1)]
            client_accuracy = []
            
            for i in range(st.session_state.current_clients):
                base = st.session_state.current_accuracy
                client_acc = base + random.uniform(-0.1, 0.1)
                client_accuracy.append(max(0, min(1, client_acc)))
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=client_names,
                y=client_accuracy,
                marker=dict(
                    color=client_accuracy,
                    colorscale='viridis',
                    showscale=True,
                    colorbar=dict(title="Accuracy")
                )
            ))
            
            fig.update_layout(
                title="Client Performance Distribution",
                xaxis_title="Clients",
                yaxis_title="Accuracy",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # Data distribution
            client_samples = [random.randint(500, 2000) for _ in range(st.session_state.current_clients)]
            
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=client_names,
                values=client_samples,
                hole=0.4,
                marker=dict(colors=px.colors.qualitative.Set3)
            ))
            
            fig.update_layout(
                title="Data Distribution",
                template="plotly_white",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        
        # Client details table
        client_data = []
        for i in range(1, st.session_state.current_clients + 1):
            accuracy = st.session_state.current_accuracy + random.uniform(-0.1, 0.1)
            samples = random.randint(500, 2000)
            status = "Active" if random.random() > 0.1 else "Idle"
            
            client_data.append({
                "Client ID": f"Client {i}",
                "Status": status,
                "Accuracy": f"{max(0, min(1, accuracy)):.3f}",
                "Samples": samples,
                "Last Update": datetime.now().strftime("%H:%M:%S")
            })
        
        df_clients = pd.DataFrame(client_data)
        st.markdown('<div class="data-table">', unsafe_allow_html=True)
        st.dataframe(df_clients, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Client metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("🟢 Active Clients", sum(1 for c in client_data if c["Status"] == "Active"))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("📊 Total Samples", f"{sum(c['Samples'] for c in client_data):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            avg_accuracy = sum(float(c["Accuracy"]) for c in client_data) / len(client_data)
            st.metric("🎯 Avg Accuracy", f"{avg_accuracy:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        
        # Privacy metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # Privacy budget gauge
            privacy_used = st.session_state.current_round / n_rounds if enable_dp else 0
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=privacy_used,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Privacy Budget Used"},
                gauge={'axis': {'range': [None, 1]},
                       'bar': {'color': "#3b82f6"},
                       'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                                {'range': [0.5, 0.8], 'color': "#fbbf24"},
                                {'range': [0.8, 1], 'color': "#ef4444"}]}
            ))
            
            fig.update_layout(
                height=400,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # Security metrics
            security_metrics = {
                "Encryption": 95 if enable_encryption else 0,
                "Differential Privacy": 90 if enable_dp else 0,
                "Data Integrity": 98,
                "Access Control": 92
            }
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(security_metrics.keys()),
                y=list(security_metrics.values()),
                marker=dict(color=['#3b82f6', '#10b981', '#f59e0b', '#ef4444'])
            ))
            
            fig.update_layout(
                title="Security Metrics",
                yaxis_title="Score (%)",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Privacy settings summary
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 🔒 Privacy Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Differential Privacy**
            - **Enabled**: {enable_dp}
            - **Privacy Budget (ε)**: {epsilon if enable_dp else 'N/A'}
            - **Budget Used**: {privacy_used:.1%}
            """)
        
        with col2:
            st.markdown(f"""
            **Encryption**
            - **Enabled**: {enable_encryption}
            - **Protocol**: AES-256
            - **Key Exchange**: Diffie-Hellman
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main function"""
    create_federated_learning_website()

if __name__ == "__main__":
    main()
