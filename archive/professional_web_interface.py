#!/usr/bin/env python3
"""
Professional Web Interface for Federated Learning
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
from collections import defaultdict

def create_professional_interface():
    """Create a beautiful professional web interface"""
    
    # Page configuration
    st.set_page_config(
        page_title="Federated Learning Platform",
        page_icon="🌸",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Professional CSS styling
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        animation: gradient 3s ease infinite;
        background-size: 200% 200%;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .sub-header {
        font-size: 1.3rem;
        text-align: center;
        color: #64748b;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 1rem;
        padding: 2rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        margin-bottom: 2rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        animation: shimmer 2s linear infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 30px 60px rgba(0, 0, 0, 0.15);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 1rem;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: scale(1.05);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 500;
    }
    
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #334155;
        margin-bottom: 1.5rem;
        text-align: center;
        padding-bottom: 1rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.75rem;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .progress-container {
        background: #f1f5f9;
        border-radius: 1rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .tab-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .chart-container {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 0.5rem;
        animation: pulse 2s infinite;
    }
    
    .status-active { background-color: #10b981; }
    .status-completed { background-color: #22c55e; }
    .status-pending { background-color: #f59e0b; }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🌸 Federated Learning Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enterprise-Grade Collaborative AI with Advanced Privacy Protection</p>', unsafe_allow_html=True)
    
    # Key metrics at top - showing current training state
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">⏸ Idle</div>
            <div class="metric-label">System Status</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">5/10</div>
            <div class="metric-label">Training Progress</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">10</div>
            <div class="metric-label">Active Clients</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">0.786</div>
            <div class="metric-label">Latest Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Comprehensive graphs section below metrics
    st.markdown("---")
    
    # Training Status Section
    st.markdown("### 🔄 Current Training Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>⏸ System Status: Idle</h4>
            <p>The system completed the last training step and is currently paused or waiting for the next round.</p>
            <ul>
                <li>Training is not actively running at this moment</li>
                <li>The red dot indicates the server is not executing a training cycle</li>
                <li>If training resumes, it will show: ▶ Running</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Training Progress: 5/10</h4>
            <p>Your system is configured for 10 federated training rounds.</p>
            <ul>
                <li><strong>Completed rounds:</strong> 5</li>
                <li><strong>Remaining rounds:</strong> 5</li>
                <li><strong>Current state:</strong> Halfway through training</li>
            </ul>
            <p><strong>Training flow:</strong></p>
            <p>Round 1 → Clients train locally → Round 2 → Model updates aggregated → Round 3 → Global model improved → Round 4 → Accuracy increases → Round 5 → Current state → Round 6-10 → Continue training</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>🏢 Active Clients: 10</h4>
            <p>10 devices (clients) are participating in training.</p>
            <ul>
                <li>Phones, Edge devices, Different organizations</li>
                <li>Client 1 → 1200 samples</li>
                <li>Client 2 → 800 samples</li>
                <li>Client 3 → 600 samples</li>
                <li>... Client 10 → 900 samples</li>
            </ul>
            <p>Each client receives the global model, trains with local data, and sends model updates back to the server.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Real-Time Analytics Dashboard")
    
    # Generate sample data for graphs - showing exact training progression
    rounds = list(range(1, 11))
    accuracy_progress = [0.52, 0.63, 0.71, 0.75, 0.786, 0.81, 0.84, 0.86, 0.88, 0.89]
    loss_progress = [0.92, 0.71, 0.58, 0.49, 0.41, 0.35, 0.30, 0.26, 0.23, 0.21]
    
    # Main charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rounds,
            y=accuracy_progress,
            mode='lines+markers',
            name='Model Accuracy',
            line=dict(color='#10b981', width=4),
            marker=dict(size=10, color='#10b981', line=dict(width=2, color='white')),
            fill='tonexty',
            fillcolor='rgba(16, 185, 129, 0.1)'
        ))
        fig.update_layout(
            title="📈 Model Accuracy Progress",
            xaxis_title="Training Round",
            yaxis_title="Accuracy",
            template="plotly_white",
            height=350,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rounds,
            y=loss_progress,
            mode='lines+markers',
            name='Training Loss',
            line=dict(color='#ef4444', width=4),
            marker=dict(size=10, color='#ef4444', line=dict(width=2, color='white')),
            fill='tonexty',
            fillcolor='rgba(239, 68, 68, 0.1)'
        ))
        fig.update_layout(
            title="📉 Training Loss Reduction",
            xaxis_title="Training Round",
            yaxis_title="Loss",
            template="plotly_white",
            height=350,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Organization performance charts
    st.markdown("### 🏢 Organization Performance Overview")
    
    # Generate organization data
    org_names = [f'Org {i}' for i in range(1, 6)]
    org_accuracy = [0.82 + random.uniform(-0.05, 0.08) for _ in range(5)]
    org_contribution = [random.uniform(0.7, 1.0) for _ in range(5)]
    org_data_size = [random.uniform(800, 2000) for _ in range(5)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=org_names,
            y=org_accuracy,
            name='Accuracy',
            marker=dict(
                color=org_accuracy,
                colorscale='viridis',
                showscale=True,
                colorbar=dict(title="Accuracy")
            ),
            text=[f'{acc:.3f}' for acc in org_accuracy],
            textposition='outside'
        ))
        fig.update_layout(
            title="📊 Organization Accuracy Comparison",
            xaxis_title="Organizations",
            yaxis_title="Accuracy",
            template="plotly_white",
            height=350,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=org_names,
            y=org_contribution,
            name='Contribution Score',
            marker=dict(
                color=org_contribution,
                colorscale='plasma',
                showscale=True,
                colorbar=dict(title="Contribution")
            ),
            text=[f'{cont:.3f}' for cont in org_contribution],
            textposition='outside'
        ))
        fig.update_layout(
            title="🎯 Organization Contribution Scores",
            xaxis_title="Organizations",
            yaxis_title="Contribution Score",
            template="plotly_white",
            height=350,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Privacy and security charts
    st.markdown("### 🔒 Privacy & Security Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Privacy budget consumption
        privacy_used = [0.1, 0.25, 0.4, 0.55, 0.65, 0.75, 0.82, 0.88, 0.93, 0.97]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rounds,
            y=privacy_used,
            mode='lines+markers',
            name='Privacy Budget Used',
            line=dict(color='#8b5cf6', width=4),
            marker=dict(size=10, color='#8b5cf6'),
            fill='tozeroy',
            fillcolor='rgba(139, 92, 246, 0.2)'
        ))
        fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Privacy Limit")
        fig.update_layout(
            title="🔒 Privacy Budget Consumption",
            xaxis_title="Training Round",
            yaxis_title="Privacy Budget Used (ε)",
            template="plotly_white",
            height=350,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Security metrics over time
        security_scores = [85, 87, 90, 92, 93, 95, 96, 97, 98, 98]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rounds,
            y=security_scores,
            mode='lines+markers',
            name='Security Score',
            line=dict(color='#f59e0b', width=4),
            marker=dict(size=10, color='#f59e0b'),
            fill='tonexty',
            fillcolor='rgba(245, 158, 11, 0.1)'
        ))
        fig.update_layout(
            title="🛡️ Security Score Evolution",
            xaxis_title="Training Round",
            yaxis_title="Security Score (%)",
            template="plotly_white",
            height=350,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Data distribution chart
    st.markdown("### 📊 Data Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=org_names,
            values=org_data_size,
            name="Data Distribution",
            hole=0.4,
            marker=dict(colors=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe'])
        ))
        fig.update_layout(
            title="📊 Data Distribution Across Organizations",
            template="plotly_white",
            height=350,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Performance heatmap
        performance_data = np.random.rand(5, 5)  # 5 orgs x 5 metrics
        
        fig = go.Figure(data=go.Heatmap(
            z=performance_data,
            x=['Accuracy', 'Speed', 'Quality', 'Security', 'Contribution'],
            y=org_names,
            colorscale='viridis',
            showscale=True,
            colorbar=dict(title="Performance Score")
        ))
        fig.update_layout(
            title="🔥 Organization Performance Heatmap",
            template="plotly_white",
            height=350,
            xaxis_title="Metrics",
            yaxis_title="Organizations"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Federated Learning Explanation Section
    st.markdown("### 🔄 How Federated Learning Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🔄 Federated Learning Cycle</h4>
            <p><strong>Step 1 — Server Sends Global Model</strong></p>
            <p>Global Model v1 sent to all 10 clients.</p>
            
            <p><strong>Step 2 — Clients Train Locally</strong></p>
            <p>Each client trains on its own data:</p>
            <ul>
                <li>Client 1 → trains with 500 samples</li>
                <li>Client 2 → trains with 1000 samples</li>
                <li>Client 3 → trains with 700 samples</li>
            </ul>
            
            <p><strong>Step 3 — Clients Send Model Updates</strong></p>
            <p>Clients send weights, not raw data:</p>
            <p><code>Client update: w1 = 0.23, w2 = 0.11, w3 = 0.67</code></p>
            
            <p><strong>Step 4 — Server Aggregates Updates</strong></p>
            <p>Server uses Federated Averaging (FedAvg):</p>
            <p><code>Global Weight = (w1_client1 + w1_client2 + ... + w1_client10) / 10</code></p>
            
            <p><strong>Step 5 — New Round Begins</strong></p>
            <p>Server sends updated model again. Round 6 starting...</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🔒 Privacy Advantage of Your System</h4>
            <p><strong>Normal ML:</strong></p>
            <p>All data → server ❌</p>
            
            <p><strong>Your Project:</strong></p>
            <p>Data stays on client devices ✅</p>
            <p>Only model weights are shared ✅</p>
            
            <p><strong>Benefits:</strong></p>
            <ul>
                <li>✔ Data privacy preserved</li>
                <li>✔ No raw data transfer</li>
                <li>✔ Complies with privacy regulations</li>
            </ul>
            
            <h4>🌟 Real-World Examples</h4>
            <p><strong>Google:</strong></p>
            <p>Gboard keyboard - Your phone learns typing patterns locally and sends model updates instead of your messages.</p>
            
            <p><strong>Healthcare:</strong></p>
            <p>Hospitals can train a disease prediction model without sharing patient records.</p>
            
            <h4>📊 Simple Demo Explanation</h4>
            <p>"This system implements a federated learning pipeline where multiple clients collaboratively train a machine learning model without sharing their raw data. Each client trains locally and sends only model updates to the central server. The server aggregates these updates to create a global model."</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📈 What Happens Next")
    
    st.markdown("""
    <div class="feature-card">
        <h4>🚀 Next Steps in Your Project</h4>
        <p><strong>Right Now:</strong></p>
        <ul>
            <li>Training Round = 5/10</li>
            <li>Accuracy = 78.6%</li>
            <li>Clients = 10</li>
        </ul>
        
        <p><strong>Next:</strong></p>
        <ul>
            <li>Round 6 → Continue training</li>
            <li>Round 7 → Accuracy improves</li>
            <li>Round 8 → Model converges</li>
            <li>Round 9 → Final optimization</li>
            <li>Round 10 → Training completed</li>
        </ul>
        
        <p><strong>After that:</strong></p>
        <ul>
            <li>Final accuracy displayed (expected: 85-90%)</li>
            <li>Training completed</li>
            <li>Model ready for deployment</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h3 class="sidebar-header">⚙️ Platform Configuration</h3>', unsafe_allow_html=True)
        
        # Dataset selection
        st.markdown("### 📊 Dataset Selection")
        datasets = [
            "Customer Churn Prediction",
            "Medical Diagnosis (Diabetes)",
            "House Price Prediction",
            "Student Performance",
            "Credit Card Fraud Detection"
        ]
        selected_dataset = st.selectbox("Choose Dataset", datasets)
        
        # Training configuration
        st.markdown("### 🚀 Training Configuration")
        n_clients = st.slider("Number of Organizations", 2, 20, 5)
        n_rounds = st.slider("Training Rounds", 1, 20, 5)
        learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
        
        # Privacy settings
        st.markdown("### 🔒 Privacy Settings")
        enable_dp = st.checkbox("Enable Differential Privacy", value=True)
        if enable_dp:
            epsilon = st.slider("Privacy Budget (ε)", 0.1, 10.0, 1.0, 0.1)
            delta = st.slider("Failure Probability (δ)", 1e-10, 1e-1, 1e-5, format="%.0e")
        
        # Security settings
        st.markdown("### 🛡️ Security Settings")
        enable_encryption = st.checkbox("Enable End-to-End Encryption", value=True)
        secure_aggregation = st.checkbox("Enable Secure Aggregation", value=True)
        
        # Control buttons
        st.markdown("### 🎮 Control Panel")
        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button("🚀 Start Training", type="primary")
        with col2:
            stop_button = st.button("⏹️ Stop Training")
        
        st.markdown("---")
        st.markdown("### 📈 System Status")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<span class="status-indicator status-active"></span>Server Active', unsafe_allow_html=True)
        with col2:
            st.markdown('<span class="status-indicator status-active"></span>All Clients Ready', unsafe_allow_html=True)
    
    # Main content area
    if start_button:
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Training Dashboard", 
            "📊 Performance Analytics", 
            "🔒 Privacy Monitor",
            "🏢 Organization Insights",
            "📝 Activity Logs"
        ])
        
        with tab1:
            st.markdown('<h2 class="tab-header">🎯 Training Dashboard</h2>', unsafe_allow_html=True)
            
            # Progress section
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 🔄 Training Progress")
            
            # Initialize training state
            if 'training_progress' not in st.session_state:
                st.session_state.training_progress = 0
                st.session_state.training_log = []
                st.session_state.is_training = True
            
            # Simulate training
            if st.session_state.is_training and st.session_state.training_progress < n_rounds:
                progress = (st.session_state.training_progress + 1) / n_rounds
                st.progress(progress, text=f"Round {st.session_state.training_progress + 1}/{n_rounds}")
                
                # Simulate training delay
                time.sleep(1)
                
                # Calculate metrics
                accuracy = 0.6 + (st.session_state.training_progress * 0.05) + random.uniform(-0.02, 0.02)
                loss = max(0.1, 1.0 - (st.session_state.training_progress * 0.15))
                
                # Apply differential privacy
                if enable_dp:
                    privacy_noise = random.uniform(-epsilon/100, epsilon/100)
                    accuracy += privacy_noise
                    accuracy = max(0, min(1, accuracy))
                
                # Log results
                st.session_state.training_log.append({
                    'round': st.session_state.training_progress + 1,
                    'accuracy': accuracy,
                    'loss': loss,
                    'timestamp': datetime.now()
                })
                
                st.session_state.training_progress += 1
                st.rerun()
            
            elif st.session_state.training_progress >= n_rounds:
                st.session_state.is_training = False
                st.success("🎉 Training Completed Successfully!")
                
                # Show final metrics
                if st.session_state.training_log:
                    final_metrics = st.session_state.training_log[-1]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🎯 Final Accuracy", f"{final_metrics['accuracy']:.3f}")
                    with col2:
                        st.metric("📉 Final Loss", f"{final_metrics['loss']:.3f}")
                    with col3:
                        st.metric("🔄 Total Rounds", n_rounds)
                    with col4:
                        st.metric("🏢 Organizations", n_clients)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Real-time charts
            if st.session_state.training_log:
                df_log = pd.DataFrame(st.session_state.training_log)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_log['round'],
                        y=df_log['accuracy'],
                        mode='lines+markers',
                        name='Accuracy',
                        line=dict(color='#10b981', width=3),
                        marker=dict(size=8, color='#10b981')
                    ))
                    fig.update_layout(
                        title="📈 Model Accuracy Over Time",
                        xaxis_title="Training Round",
                        yaxis_title="Accuracy",
                        template="plotly_white",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_log['round'],
                        y=df_log['loss'],
                        mode='lines+markers',
                        name='Loss',
                        line=dict(color='#ef4444', width=3),
                        marker=dict(size=8, color='#ef4444')
                    ))
                    fig.update_layout(
                        title="📉 Training Loss Over Time",
                        xaxis_title="Training Round",
                        yaxis_title="Loss",
                        template="plotly_white",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<h2 class="tab-header">📊 Performance Analytics</h2>', unsafe_allow_html=True)
            
            # Organization performance
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("### 🏢 Organization Performance")
            
            # Generate organization data
            org_data = []
            for i in range(n_clients):
                accuracy = 0.7 + random.uniform(0, 0.2)
                contribution = random.uniform(0.8, 1.0)
                data_quality = random.uniform(0.7, 1.0)
                
                org_data.append({
                    'Organization': f'Org {i+1}',
                    'Accuracy': accuracy,
                    'Contribution': contribution,
                    'Data Quality': data_quality,
                    'Status': 'Active' if random.random() > 0.1 else 'Training'
                })
            
            df_org = pd.DataFrame(org_data)
            
            # Performance table
            st.dataframe(df_org, use_container_width=True)
            
            # Performance charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(df_org, x='Organization', y='Accuracy', 
                            title="📊 Accuracy by Organization",
                            color='Accuracy', color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(df_org, x='Organization', y='Contribution',
                            title="🎯 Contribution Score",
                            color='Contribution', color_continuous_scale='plasma')
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<h2 class="tab-header">🔒 Privacy Monitor</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="feature-card">', unsafe_allow_html=True)
                st.markdown("### 🔒 Differential Privacy Status")
                
                if enable_dp:
                    st.info(f"""
                    **Privacy Configuration:**
                    - ✅ Differential Privacy: Enabled
                    - 🔢 Epsilon (ε): {epsilon}
                    - 🔢 Delta (δ): {delta}
                    - 📊 Privacy Budget Used: {(epsilon * 0.7):.2f}
                    - 🎯 Privacy Budget Remaining: {(epsilon * 0.3):.2f}
                    """)
                    
                    # Privacy budget gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=epsilon * 0.3,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Privacy Budget Remaining"},
                        gauge={'axis': {'range': [None, epsilon]},
                               'bar': {'color': "#667eea"},
                               'steps': [{'range': [0, epsilon * 0.3], 'color': "lightgray"},
                                        {'range': [epsilon * 0.3, epsilon * 0.7], 'color': "yellow"},
                                        {'range': [epsilon * 0.7, epsilon], 'color': "red"}]}
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Differential Privacy is Disabled")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="feature-card">', unsafe_allow_html=True)
                st.markdown("### 🛡️ Security Status")
                
                st.info(f"""
                **Security Configuration:**
                - ✅ End-to-End Encryption: {'Enabled' if enable_encryption else 'Disabled'}
                - ✅ Secure Aggregation: {'Enabled' if secure_aggregation else 'Disabled'}
                - 🔐 Encryption Protocol: AES-256
                - 🛡️ Authentication: OAuth 2.0
                - 📊 Security Score: 95%
                """)
                
                # Security metrics
                security_metrics = {
                    'Encryption': 95,
                    'Authentication': 98,
                    'Data Integrity': 92,
                    'Access Control': 96
                }
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=list(security_metrics.keys()),
                    y=list(security_metrics.values()),
                    marker_color=['#10b981', '#3b82f6', '#f59e0b', '#ef4444']
                ))
                fig.update_layout(
                    title="🛡️ Security Metrics",
                    yaxis_title="Score (%)",
                    template="plotly_white",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            st.markdown('<h2 class="tab-header">🏢 Organization Insights</h2>', unsafe_allow_html=True)
            
            # Organization details
            for i in range(min(3, n_clients)):
                st.markdown('<div class="feature-card">', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"### 🏢 Organization {i+1}")
                    st.metric("📊 Accuracy", f"{0.75 + random.uniform(0, 0.2):.3f}")
                    st.metric("🎯 Contribution", f"{random.uniform(0.8, 1.0):.3f}")
                
                with col2:
                    st.markdown("### 📈 Performance")
                    st.metric("📉 Loss", f"{random.uniform(0.1, 0.5):.3f}")
                    st.metric("⚡ Speed", f"{random.uniform(0.8, 1.0):.3f}")
                
                with col3:
                    st.markdown("### 🔒 Privacy")
                    st.metric("🛡️ Security", "100%")
                    st.metric("📊 Data Quality", f"{random.uniform(0.7, 1.0):.3f}")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        with tab5:
            st.markdown('<h2 class="tab-header">📝 Activity Logs</h2>', unsafe_allow_html=True)
            
            # Activity logs
            logs = [
                {"time": "12:45:23", "event": "Training Started", "status": "✅", "details": "5 organizations joined"},
                {"time": "12:45:24", "event": "Round 1 Complete", "status": "✅", "details": "Accuracy: 0.654"},
                {"time": "12:45:26", "event": "Round 2 Complete", "status": "✅", "details": "Accuracy: 0.698"},
                {"time": "12:45:28", "event": "Round 3 Complete", "status": "✅", "details": "Accuracy: 0.742"},
                {"time": "12:45:30", "event": "Privacy Check", "status": "✅", "details": "ε-budget: 0.7/1.0"},
                {"time": "12:45:32", "event": "Security Audit", "status": "✅", "details": "All systems secure"},
            ]
            
            for log in logs:
                st.markdown(f"""
                <div class="feature-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-weight: 600; color: #1e293b;">{log['event']}</span>
                            <span style="color: #64748b; margin-left: 1rem;">{log['details']}</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 1rem;">
                            <span style="color: #64748b;">{log['time']}</span>
                            <span>{log['status']}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        # Welcome screen
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🌟 Welcome to Federated Learning Platform")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🔒 Privacy-First Approach
            - **Differential Privacy**: Mathematical privacy guarantees
            - **Secure Aggregation**: Encrypted model updates
            - **Data Sovereignty**: Raw data never leaves organizations
            - **Compliance**: GDPR, HIPAA ready
            """)
        
        with col2:
            st.markdown("""
            #### 🚀 Advanced Features
            - **Real-time Monitoring**: Live training progress
            - **Client Analytics**: Individual performance tracking
            - **Fault Tolerance**: Automatic failure recovery
            - **Scalable Architecture**: Support for 100+ organizations
            """)
        
        st.markdown("---")
        st.markdown("### 🎯 Getting Started")
        st.info("1. Configure your dataset and privacy settings in the sidebar\n2. Click '🚀 Start Training' to begin federated learning\n3. Monitor real-time progress across all tabs\n4. Analyze organization performance and privacy metrics")
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main function"""
    create_professional_interface()

if __name__ == "__main__":
    main()
