#!/usr/bin/env python3
"""
Professional Dashboard - Clean, Data-Focused Interface
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

def create_professional_dashboard():
    """Create a professional, data-focused dashboard"""
    
    # Professional page config
    st.set_page_config(
        page_title="Federated Learning Analytics",
        page_icon="📊",
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
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        border: 1px solid #e5e7eb;
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e40af;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .chart-container {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        border: 1px solid #e5e7eb;
    }
    
    .sidebar-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .stSelectbox > div > div {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
    }
    
    .stSlider > div > div {
        background-color: #3b82f6;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e3a8a 100%);
        transform: translateY(-1px);
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f2937;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">📊 Federated Learning Analytics</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = "Customer Churn Prediction"
        st.session_state.current_accuracy = 0.786
        st.session_state.current_round = 5
        st.session_state.current_clients = 10
        st.session_state.current_status = "⏸ Idle"
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h3 class="sidebar-header">Configuration</h3>', unsafe_allow_html=True)
        
        # Dataset selection
        datasets = {
            "Customer Churn Prediction": {"accuracy": 0.786, "round": 5, "color": "#3b82f6"},
            "Medical Diagnosis (Diabetes)": {"accuracy": 0.823, "round": 6, "color": "#10b981"},
            "House Price Prediction": {"accuracy": 0.791, "round": 4, "color": "#f59e0b"},
            "Student Performance": {"accuracy": 0.847, "round": 7, "color": "#8b5cf6"},
            "Iris Flower Classification": {"accuracy": 0.923, "round": 8, "color": "#ec4899"},
            "Wine Type Classification": {"accuracy": 0.891, "round": 6, "color": "#ef4444"},
            "Breast Cancer Detection": {"accuracy": 0.934, "round": 7, "color": "#06b6d4"},
            "Sales Revenue Prediction": {"accuracy": 0.812, "round": 5, "color": "#84cc16"},
            "Student Grade Prediction": {"accuracy": 0.768, "round": 4, "color": "#f97316"}
        }
        
        selected_dataset = st.selectbox(
            "Dataset",
            list(datasets.keys()),
            index=list(datasets.keys()).index(st.session_state.current_dataset)
        )
        
        # Configuration
        n_clients = st.slider("Clients", 2, 20, st.session_state.current_clients)
        n_rounds = st.slider("Rounds", 1, 20, 10)
        privacy_enabled = st.checkbox("Privacy Protection", value=True)
        
        if privacy_enabled:
            epsilon = st.slider("Privacy Budget", 0.1, 5.0, 1.0, 0.1)
        
        # Apply button
        if st.button("Apply Configuration", type="primary"):
            dataset_info = datasets[selected_dataset]
            st.session_state.current_dataset = selected_dataset
            st.session_state.current_accuracy = dataset_info["accuracy"]
            st.session_state.current_round = dataset_info["round"]
            st.session_state.current_clients = n_clients
            st.session_state.current_status = "▶ Running"
            st.session_state.dataset_color = dataset_info["color"]
            st.rerun()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.current_status}</div>
            <div class="metric-label">Status</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.current_round}/{n_rounds}</div>
            <div class="metric-label">Progress</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.current_clients}</div>
            <div class="metric-label">Clients</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.current_accuracy:.1%}</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance Overview
    st.markdown('<h2 class="section-header">📈 Performance Overview</h2>', unsafe_allow_html=True)
    
    # Generate data based on current dataset
    max_rounds = n_rounds
    current_round = st.session_state.current_round
    base_accuracy = 0.5
    target_accuracy = st.session_state.current_accuracy
    
    # Create realistic progression
    rounds = list(range(1, max_rounds + 1))
    accuracy_data = []
    loss_data = []
    
    for i, round_num in enumerate(rounds):
        if round_num <= current_round:
            # Completed rounds - realistic progression
            progress = round_num / max_rounds
            accuracy = base_accuracy + (target_accuracy - base_accuracy) * progress + random.uniform(-0.02, 0.02)
            loss = 1.0 - (progress * 0.8) + random.uniform(-0.05, 0.05)
        else:
            # Future rounds - projected
            progress = current_round / max_rounds
            accuracy = base_accuracy + (target_accuracy - base_accuracy) * progress + ((round_num - current_round) * 0.01)
            loss = 1.0 - (progress * 0.8) - ((round_num - current_round) * 0.02)
        
        accuracy_data.append(max(0, min(1, accuracy)))
        loss_data.append(max(0, min(1, loss)))
    
    # Charts row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = go.Figure()
        
        # Add completed rounds
        completed_rounds = rounds[:current_round]
        completed_accuracy = accuracy_data[:current_round]
        
        fig.add_trace(go.Scatter(
            x=completed_rounds,
            y=completed_accuracy,
            mode='lines+markers',
            name='Completed',
            line=dict(color=st.session_state.get('dataset_color', '#3b82f6'), width=3),
            marker=dict(size=8, color=st.session_state.get('dataset_color', '#3b82f6'))
        ))
        
        # Add projected rounds
        if current_round < max_rounds:
            projected_rounds = rounds[current_round:]
            projected_accuracy = accuracy_data[current_round:]
            
            fig.add_trace(go.Scatter(
                x=projected_rounds,
                y=projected_accuracy,
                mode='lines+markers',
                name='Projected',
                line=dict(color='#9ca3af', width=2, dash='dash'),
                marker=dict(size=6, color='#9ca3af')
            ))
        
        fig.update_layout(
            title="Model Accuracy",
            xaxis_title="Round",
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
        
        # Loss chart
        fig.add_trace(go.Scatter(
            x=rounds,
            y=loss_data,
            mode='lines+markers',
            name='Training Loss',
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
            height=350,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Client Performance
    st.markdown('<h2 class="section-header">🏢 Client Performance</h2>', unsafe_allow_html=True)
    
    # Generate client data
    client_names = [f'Client {i}' for i in range(1, st.session_state.current_clients + 1)]
    client_accuracy = []
    client_samples = []
    
    for i in range(st.session_state.current_clients):
        # Accuracy varies by client
        base = st.session_state.current_accuracy
        client_acc = base + random.uniform(-0.1, 0.1)
        client_accuracy.append(max(0, min(1, client_acc)))
        client_samples.append(random.randint(500, 2000))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=client_names,
            y=client_accuracy,
            name='Accuracy',
            marker=dict(
                color=client_accuracy,
                colorscale='viridis',
                showscale=True,
                colorbar=dict(title="Score")
            ),
            text=[f'{acc:.3f}' for acc in client_accuracy],
            textposition='outside'
        ))
        fig.update_layout(
            title="Client Accuracy Distribution",
            xaxis_title="Clients",
            yaxis_title="Accuracy",
            template="plotly_white",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=client_names,
            y=client_samples,
            name='Samples',
            marker=dict(
                color=client_samples,
                colorscale='blues',
                showscale=True,
                colorbar=dict(title="Count")
            ),
            text=[f'{samples}' for samples in client_samples],
            textposition='outside'
        ))
        fig.update_layout(
            title="Data Samples per Client",
            xaxis_title="Clients",
            yaxis_title="Sample Count",
            template="plotly_white",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # System Metrics
    st.markdown('<h2 class="section-header">📊 System Metrics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Privacy budget gauge
        privacy_used = current_round / max_rounds if privacy_enabled else 0
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
        fig.update_layout(height=300, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Data distribution
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
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Performance heatmap
        performance_matrix = np.random.rand(st.session_state.current_clients, 3)
        
        fig = go.Figure(data=go.Heatmap(
            z=performance_matrix,
            x=['Accuracy', 'Speed', 'Quality'],
            y=client_names,
            colorscale='viridis',
            showscale=True,
            colorbar=dict(title="Score")
        ))
        fig.update_layout(
            title="Performance Matrix",
            template="plotly_white",
            height=300,
            xaxis_title="Metrics",
            yaxis_title="Clients"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main function"""
    create_professional_dashboard()

if __name__ == "__main__":
    main()
