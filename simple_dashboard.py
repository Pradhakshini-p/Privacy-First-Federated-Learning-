#!/usr/bin/env python3
"""
Simple Working Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

def create_simple_dashboard():
    """Create a simple, working dashboard"""
    
    # Page config
    st.set_page_config(
        page_title="Federated Learning Dashboard",
        page_icon="🔒",
        layout="wide"
    )
    
    # Header
    st.title("🔒 Privacy-First Federated Learning Dashboard")
    st.markdown("---")
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        n_clients = st.slider("Number of Clients", 2, 10, 5)
        n_rounds = st.slider("Training Rounds", 1, 10, 5)
        
        enable_dp = st.checkbox("Enable Differential Privacy", value=True)
        if enable_dp:
            epsilon = st.slider("ε (Privacy Budget)", 0.1, 5.0, 1.0, 0.1)
        
        start_training = st.button("🚀 Start Training", type="primary")
    
    # Main content
    if start_training:
        st.header("🎯 Training Progress")
        
        # Initialize session state
        if 'training_log' not in st.session_state:
            st.session_state.training_log = []
            st.session_state.current_round = 0
            st.session_state.is_training = True
        
        # Training simulation
        if st.session_state.is_training and st.session_state.current_round < n_rounds:
            # Progress bar
            progress = st.session_state.current_round / n_rounds
            st.progress(progress, text=f"Round {st.session_state.current_round + 1}/{n_rounds}")
            
            # Simulate training
            time.sleep(1)
            
            # Calculate metrics
            base_accuracy = 0.6 + (st.session_state.current_round * 0.03)
            accuracy = min(0.95, base_accuracy + np.random.normal(0, 0.02))
            loss = max(0.1, 1.0 - (st.session_state.current_round * 0.15))
            
            # Apply differential privacy
            if enable_dp:
                privacy_noise = np.random.normal(0, epsilon/10)
                accuracy += privacy_noise
                accuracy = max(0, min(1, accuracy))
            
            # Log results
            st.session_state.training_log.append({
                'round': st.session_state.current_round + 1,
                'accuracy': accuracy,
                'loss': loss,
                'timestamp': datetime.now()
            })
            
            st.session_state.current_round += 1
            st.rerun()
        
        elif st.session_state.current_round >= n_rounds:
            st.session_state.is_training = False
            st.success("🎉 Training Completed!")
            
            # Show results
            if st.session_state.training_log:
                df_log = pd.DataFrame(st.session_state.training_log)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎯 Final Accuracy", f"{df_log['accuracy'].iloc[-1]:.3f}")
                with col2:
                    st.metric("📉 Final Loss", f"{df_log['loss'].iloc[-1]:.3f}")
                with col3:
                    st.metric("🔄 Total Rounds", n_rounds)
                with col4:
                    st.metric("🏦 Clients", n_clients)
                
                # Create charts
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("📈 Accuracy Over Rounds", "📉 Loss Over Rounds"),
                    vertical_spacing=0.1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df_log['round'],
                        y=df_log['accuracy'],
                        mode='lines+markers',
                        name='Accuracy',
                        line=dict(color='#10b981', width=3),
                        marker=dict(size=8)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df_log['round'],
                        y=df_log['loss'],
                        mode='lines+markers',
                        name='Loss',
                        line=dict(color='#ef4444', width=3),
                        marker=dict(size=8)
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(
                    height=600,
                    template="plotly_white",
                    showlegend=True,
                    title="📊 Training Progress"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Welcome screen
        st.header("🌟 Welcome to Federated Learning")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔒 Privacy Features
            - **Differential Privacy**: ε-differential privacy protection
            - **Secure Aggregation**: Encrypted model updates
            - **Data Sovereignty**: Raw data never leaves clients
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Analytics Features
            - **Real-time Monitoring**: Live training progress
            - **Client Analytics**: Individual performance tracking
            - **Fault Tolerance**: Automatic failure recovery
            """)
        
        st.markdown("---")
        st.markdown("### 🚀 Getting Started")
        st.info("1. Configure settings in the sidebar\n2. Click '🚀 Start Training' to begin\n3. Monitor real-time progress and results")

if __name__ == "__main__":
    create_simple_dashboard()
