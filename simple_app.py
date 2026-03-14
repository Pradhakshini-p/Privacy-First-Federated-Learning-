import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime
import random

# Simple page config
st.set_page_config(
    page_title="Federated Learning Platform",
    page_icon="🔗",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
    min-height: 100vh;
}
.metric-card {
    background: rgba(30, 41, 59, 0.8);
    backdrop-filter: blur(10px);
    border-radius: 1rem;
    padding: 1.5rem;
    border: 1px solid rgba(96, 165, 250, 0.3);
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
    margin: 0.5rem;
    text-align: center;
}
.chart-container {
    background: rgba(30, 41, 59, 0.8);
    backdrop-filter: blur(10px);
    border-radius: 1rem;
    padding: 1.5rem;
    border: 1px solid rgba(96, 165, 250, 0.3);
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
    margin: 0.5rem;
    height: 400px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
<h1 style='font-size: 3rem; font-weight: 900; background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #2563eb 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
🔗 Federated Learning Platform
</h1>
<p style='color: #94a3b8; font-size: 1.1rem; margin-top: 0.5rem;'>
Distributed Machine Learning Dashboard
</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    dataset = st.selectbox("Dataset", ["Synthetic", "Iris", "Wine", "Breast Cancer"])
    model_type = st.selectbox("Model", ["Logistic Regression", "Neural Network"])
    n_clients = st.slider("Number of Clients", 2, 20, 10)
    n_rounds = st.slider("Training Rounds", 1, 50, 10)
    learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01)
    privacy_noise = st.slider("Privacy Noise", 0.0, 1.0, 0.1)
    
    st.markdown("### 🚀 Training Control")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Training", type="primary"):
            st.session_state.training = True
            st.session_state.current_round = 0
            st.session_state.max_rounds = n_rounds
            st.session_state.accuracy_history = []
            st.session_state.loss_history = []
            st.session_state.client_data = []
    
    with col2:
        if st.button("Stop Training"):
            st.session_state.training = False

# Initialize session state
if 'training' not in st.session_state:
    st.session_state.training = False
if 'current_round' not in st.session_state:
    st.session_state.current_round = 0
if 'max_rounds' not in st.session_state:
    st.session_state.max_rounds = 10
if 'accuracy_history' not in st.session_state:
    st.session_state.accuracy_history = []
if 'loss_history' not in st.session_state:
    st.session_state.loss_history = []
if 'client_data' not in st.session_state:
    st.session_state.client_data = []

# Simulate training
if st.session_state.training and st.session_state.current_round < st.session_state.max_rounds:
    # Simulate one round
    st.session_state.current_round += 1
    
    # Simulate client accuracies
    round_clients = []
    for i in range(n_clients):
        # Simulate improving accuracy with noise
        base_acc = 0.5 + (st.session_state.current_round / st.session_state.max_rounds) * 0.4
        client_acc = base_acc + np.random.normal(0, 0.05)
        client_acc = max(0.1, min(1.0, client_acc))
        
        round_clients.append({
            'client_id': f'Client_{i+1}',
            'accuracy': client_acc,
            'round': st.session_state.current_round
        })
    
    # Calculate global accuracy (average of clients)
    global_acc = np.mean([c['accuracy'] for c in round_clients])
    global_loss = 1.0 - global_acc + np.random.normal(0, 0.02)
    
    st.session_state.accuracy_history.append(global_acc)
    st.session_state.loss_history.append(global_loss)
    st.session_state.client_data.extend(round_clients)
    
    # Auto-refresh
    time.sleep(1)
    st.rerun()

# Status Metrics
st.markdown("### 📊 System Status")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
    <h3 style='color: #3b82f6; font-size: 2rem; margin: 0;'>{st.session_state.current_round}</h3>
    <p style='color: #94a3b8; margin: 0;'>Current Round</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    status = "🟢 Training" if st.session_state.training else "🔴 Idle"
    st.markdown(f"""
    <div class='metric-card'>
    <h3 style='color: #10b981; font-size: 2rem; margin: 0;'>{status}</h3>
    <p style='color: #94a3b8; margin: 0;'>System Status</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    accuracy = st.session_state.accuracy_history[-1] if st.session_state.accuracy_history else 0.0
    st.markdown(f"""
    <div class='metric-card'>
    <h3 style='color: #60a5fa; font-size: 2rem; margin: 0;'>{accuracy:.3f}</h3>
    <p style='color: #94a3b8; margin: 0;'>Global Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    loss = st.session_state.loss_history[-1] if st.session_state.loss_history else 0.0
    st.markdown(f"""
    <div class='metric-card'>
    <h3 style='color: #f59e0b; font-size: 2rem; margin: 0;'>{loss:.3f}</h3>
    <p style='color: #94a3b8; margin: 0;'>Global Loss</p>
    </div>
    """, unsafe_allow_html=True)

# Progress Bar
if st.session_state.max_rounds > 0:
    progress = st.session_state.current_round / st.session_state.max_rounds
    st.progress(progress, text=f"Training Progress: {st.session_state.current_round}/{st.session_state.max_rounds} rounds")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Analytics", "👥 Clients"])

with tab1:
    st.markdown("### 📊 Training Progress")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        if st.session_state.accuracy_history:
            # Create simple line chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(st.session_state.accuracy_history) + 1)),
                y=st.session_state.accuracy_history,
                mode='lines+markers',
                name='Global Accuracy',
                line=dict(color='#3b82f6', width=3),
                marker=dict(color='#3b82f6', size=8)
            ))
            fig.update_layout(
                title="📈 Global Accuracy Over Rounds",
                xaxis_title="Round",
                yaxis_title="Accuracy",
                template="plotly_dark",
                height=350,
                showlegend=True,
                paper_bgcolor='rgba(30, 41, 59, 0.8)',
                plot_bgcolor='rgba(30, 41, 59, 0.8)'
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        if st.session_state.loss_history:
            # Create simple line chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(st.session_state.loss_history) + 1)),
                y=st.session_state.loss_history,
                mode='lines+markers',
                name='Global Loss',
                line=dict(color='#ef4444', width=3),
                marker=dict(color='#ef4444', size=8)
            ))
            fig.update_layout(
                title="📉 Global Loss Over Rounds",
                xaxis_title="Round",
                yaxis_title="Loss",
                template="plotly_dark",
                height=350,
                showlegend=True,
                paper_bgcolor='rgba(30, 41, 59, 0.8)',
                plot_bgcolor='rgba(30, 41, 59, 0.8)'
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("### 📈 Analytics")
    
    # Client participation heatmap
    if st.session_state.client_data:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Create participation matrix
        participation_data = []
        for round_num in range(1, min(st.session_state.current_round + 1, 11)):
            for client_id in range(1, min(n_clients + 1, 11)):
                participated = any(
                    d['round'] == round_num and d['client_id'] == f'Client_{client_id}'
                    for d in st.session_state.client_data
                )
                participation_data.append({
                    'Round': round_num,
                    'Client': f'Client_{client_id}',
                    'Participated': 1 if participated else 0
                })
        
        if participation_data:
            df_participation = pd.DataFrame(participation_data)
            fig = px.density_heatmap(
                df_participation,
                x='Round',
                y='Client',
                z='Participated',
                title="🔥 Client Participation Heatmap",
                template="plotly_dark",
                height=350,
                color_continuous_scale='Blues'
            )
            fig.update_layout(
                paper_bgcolor='rgba(30, 41, 59, 0.8)',
                plot_bgcolor='rgba(30, 41, 59, 0.8)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown("### 👥 Client Performance")
    
    if st.session_state.client_data:
        # Create client performance table
        client_stats = {}
        for data in st.session_state.client_data:
            client_id = data['client_id']
            if client_id not in client_stats:
                client_stats[client_id] = []
            client_stats[client_id].append(data['accuracy'])
        
        # Calculate statistics
        performance_data = []
        for client_id, accuracies in client_stats.items():
            performance_data.append({
                'Client': client_id,
                'Rounds': len(accuracies),
                'Avg Accuracy': np.mean(accuracies),
                'Best Accuracy': np.max(accuracies),
                'Latest Accuracy': accuracies[-1] if accuracies else 0.0
            })
        
        df_performance = pd.DataFrame(performance_data)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.dataframe(df_performance.round(3), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #94a3b8; padding: 2rem;'>
<p>🔗 Federated Learning Platform - Real-time Distributed Training Simulation</p>
<p>Built with Streamlit, Plotly, and NumPy</p>
</div>
""", unsafe_allow_html=True)
