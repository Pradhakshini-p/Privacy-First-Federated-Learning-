#!/usr/bin/env python3
"""
Minimal Federated Learning Dashboard
Clean UI with minimal text, maximum visuals
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time
import random
from datetime import datetime

# Page config
st.set_page_config(
    page_title="FL Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for minimal design
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stMetric {
        background: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .plot-container {
        background: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Simulate data
def get_data():
    return {
        'round': random.randint(1, 10),
        'accuracy': random.uniform(0.7, 0.95),
        'loss': random.uniform(0.1, 0.5),
        'clients': random.randint(2, 5),
        'privacy': random.uniform(0.0, 1.0),
        'training': random.choice([True, False])
    }

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("🔐 Federated Learning")
    st.markdown("---")

# Main metrics
data = get_data()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Round", data['round'])
with col2:
    st.metric("Accuracy", f"{data['accuracy']:.3f}")
with col3:
    st.metric("Loss", f"{data['loss']:.3f}")
with col4:
    st.metric("Clients", data['clients'])

# Status indicators
col1, col2 = st.columns(2)

with col1:
    # Training status
    if data['training']:
        st.success("🟢 Training")
    else:
        st.warning("⏸️ Stopped")
    
    # Progress bar
    progress = data['round'] / 10
    st.progress(progress)

with col2:
    # Privacy gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=data['privacy'] * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 80], 'color': "yellow"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'value': 90
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

# Charts
col1, col2 = st.columns(2)

with col1:
    # Accuracy chart
    rounds = list(range(1, data['round'] + 1))
    accuracy = [0.7 + (i * 0.025) + random.uniform(-0.02, 0.02) for i in rounds]
    
    fig = px.line(
        x=rounds, 
        y=accuracy,
        markers=True,
        line_shape='spline'
    )
    fig.update_layout(
        title="Accuracy",
        xaxis_title="Round",
        yaxis_title="Value",
        height=300,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Loss chart
    loss = [0.5 - (i * 0.04) + random.uniform(-0.02, 0.02) for i in rounds]
    
    fig = px.line(
        x=rounds, 
        y=loss,
        markers=True,
        line_shape='spline',
        color_discrete_sequence=['red']
    )
    fig.update_layout(
        title="Loss",
        xaxis_title="Round",
        yaxis_title="Value",
        height=300,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

# Controls
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("▶️ Start", type="primary"):
        st.rerun()

with col2:
    if st.button("⏹️ Stop"):
        st.rerun()

with col3:
    if st.button("🔄 Refresh"):
        st.rerun()

# Auto-refresh
time.sleep(2)
st.rerun()
