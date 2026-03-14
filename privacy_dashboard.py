#!/usr/bin/env python3
"""
Privacy-First Federated Learning Dashboard
Real-time 3-Client Visualization with Privacy vs Accuracy Trade-off

Components:
- Network Map: 3 nodes communicating with central hub
- Epsilon Slider: Real-time privacy adjustment
- Accuracy Graph: Live-updating training progress
- Privacy Log: Real-time obfuscation logging
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import threading
import queue
import random

# Set page configuration
st.set_page_config(
    page_title="Privacy-First Federated Learning",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
        color: white;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .card {
        background-color: #1e293b;
        border: 2px solid #3b82f6;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        color: white;
    }
    
    .card h3 {
        color: #3b82f6;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .card p {
        color: white;
        font-size: 1rem;
        margin: 0.5rem 0;
    }
    
    .section-header {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        padding: 1.2rem 1.8rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        font-weight: 600;
        font-size: 1.3rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .privacy-log {
        background-color: #1e293b;
        border: 1px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        color: #10b981;
        height: 200px;
        overflow-y: auto;
    }
    
    .client-status {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: white;
        font-weight: bold;
    }
    
    .server-status {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: white;
        font-weight: bold;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.4);
    }
    
    .chart-container {
        background-color: #1e293b;
        border: 1px solid #3b82f6;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .network-map {
        background-color: #1e293b;
        border: 2px solid #3b82f6;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        height: 400px;
        display: flex;
        justify-content: center;
        align-items: center;
        position: relative;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class TrainingRound:
    """Training round data"""
    round: int
    client_a_accuracy: float
    client_b_accuracy: float
    client_c_accuracy: float
    global_accuracy: float
    epsilon: float
    noise_sigma: float
    timestamp: str

class PrivacyFLDashboard:
    """3-Client Privacy-First Federated Learning Dashboard"""
    
    def __init__(self):
        self.training_history = []
        self.is_training = False
        self.current_round = 0
        self.total_rounds = 10
        self.current_epsilon = 3.0
        
        # Initialize session state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables"""
        defaults = {
            'training_active': False,
            'current_epsilon': 3.0,
            'noise_multiplier': 1.0,
            'max_grad_norm': 1.0,
            'training_completed': False,
            'privacy_log': [],
            'client_status': {'A': 'Idle', 'B': 'Idle', 'C': 'Idle'},
            'server_status': 'Waiting',
            'training_rounds': []
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">🔒 Privacy-First Federated Learning</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Quick status overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.session_state.training_active:
                st.markdown('<div class="client-status">🤖 Training Active</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="client-status">⏸️ Training Paused</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="server-status">🔒 ε = {st.session_state.current_epsilon}</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="client-status">👥 3 Clients</div>', unsafe_allow_html=True)
        
        with col4:
            if st.session_state.training_completed:
                st.markdown('<div class="client-status">✅ Completed</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="client-status">⏳ In Progress</div>', unsafe_allow_html=True)
    
    def render_network_map(self):
        """Render network topology visualization"""
        st.markdown('<div class="section-header">🌐 Network Topology</div>', unsafe_allow_html=True)
        
        # Create network visualization
        fig = go.Figure()
        
        # Server (center)
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers+text',
            marker=dict(size=30, color='#3b82f6', symbol='diamond'),
            text=['Server'],
            textposition='middle center',
            name='Server',
            hovertemplate='Server<br>Global Model Aggregator'
        ))
        
        # Clients (in triangle)
        client_positions = [
            (2, 2),   # Client A (top-right)
            (-2, 2),  # Client B (top-left)
            (0, -2)   # Client C (bottom)
        ]
        
        client_names = ['Client A<br>(Trucks)', 'Client B<br>(Birds)', 'Client C<br>(Mixed)']
        client_colors = ['#10b981', '#f59e0b', '#ef4444']
        
        for i, (x, y) in enumerate(client_positions):
            status = st.session_state.client_status[['A', 'B', 'C'][i]]
            color = client_colors[i] if status == 'Training' else '#6b7280'
            
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(size=25, color=color, symbol='circle'),
                text=[client_names[i]],
                textposition='middle center',
                name=f'Client {chr(65+i)}',
                hovertemplate=f'{client_names[i]}<br>Status: {status}<br>Data: {["Trucks-focused", "Birds-focused", "Mixed"][i]}'
            ))
        
        # Connections
        for x, y in client_positions:
            fig.add_trace(go.Scatter(
                x=[0, x], y=[0, y],
                mode='lines',
                line=dict(color='#3b82f6', width=2, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title='3-Client Federated Learning Network',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template='plotly_dark',
            font=dict(color='white'),
            paper_bgcolor='#1e293b',
            plot_bgcolor='#1e293b',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Client status details
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('''
            <div class="card">
                <h3>🚚 Client A</h3>
                <p><strong>Data:</strong> Trucks-focused</p>
                <p><strong>Status:</strong> {}</p>
                <p><strong>Privacy:</strong> ε = {:.1f}</p>
            </div>
            '''.format(st.session_state.client_status['A'], st.session_state.current_epsilon), unsafe_allow_html=True)
        
        with col2:
            st.markdown('''
            <div class="card">
                <h3>🐦 Client B</h3>
                <p><strong>Data:</strong> Birds-focused</p>
                <p><strong>Status:</strong> {}</p>
                <p><strong>Privacy:</strong> ε = {:.1f}</p>
            </div>
            '''.format(st.session_state.client_status['B'], st.session_state.current_epsilon), unsafe_allow_html=True)
        
        with col3:
            st.markdown('''
            <div class="card">
                <h3>🎯 Client C</h3>
                <p><strong>Data:</strong> Mixed classes</p>
                <p><strong>Status:</strong> {}</p>
                <p><strong>Privacy:</strong> ε = {:.1f}</p>
            </div>
            '''.format(st.session_state.client_status['C'], st.session_state.current_epsilon), unsafe_allow_html=True)
    
    def render_privacy_controls(self):
        """Render privacy configuration controls"""
        st.markdown('<div class="section-header">🔒 Privacy Configuration</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🎛️ Epsilon Slider")
            
            # Epsilon slider with real-time updates
            epsilon = st.slider(
                "Privacy Budget (ε)",
                min_value=0.1,
                max_value=10.0,
                value=st.session_state.current_epsilon,
                step=0.1,
                help="Lower ε = Higher Privacy, Lower Accuracy"
            )
            
            # Update session state
            if epsilon != st.session_state.current_epsilon:
                st.session_state.current_epsilon = epsilon
                self._add_privacy_log(f"🔒 Privacy budget updated: ε = {epsilon:.1f}")
            
            # Privacy level indicator
            if epsilon < 1.0:
                privacy_level = "🔒 Very High Privacy"
                color = "#10b981"
            elif epsilon < 3.0:
                privacy_level = "🛡️ High Privacy"
                color = "#3b82f6"
            elif epsilon < 5.0:
                privacy_level = "⚖️ Medium Privacy"
                color = "#f59e0b"
            else:
                privacy_level = "⚠️ Low Privacy"
                color = "#ef4444"
            
            st.markdown(f'<div style="color: {color}; font-weight: bold; font-size: 1.2rem;">{privacy_level}</div>', unsafe_allow_html=True)
            
            # Expected accuracy based on epsilon
            expected_accuracy = min(0.9, 0.5 + (epsilon / 10.0) * 0.4)
            st.markdown(f'**Expected Accuracy:** {expected_accuracy:.2f}')
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📊 Privacy vs Accuracy Trade-off")
            
            # Generate trade-off curve
            epsilon_values = np.linspace(0.1, 10.0, 50)
            accuracies = []
            
            for eps in epsilon_values:
                # Simulate privacy-accuracy trade-off
                base_acc = 0.5
                acc_increase = min(0.4, (eps / 10.0) * 0.4)
                accuracy = base_acc + acc_increase + np.random.normal(0, 0.02)
                accuracies.append(accuracy)
            
            # Create trade-off plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=epsilon_values,
                y=accuracies,
                mode='lines',
                name='Privacy-Accuracy Trade-off',
                line=dict(color='#8b5cf6', width=3)
            ))
            
            # Add current epsilon point
            current_accuracy = min(0.9, 0.5 + (epsilon / 10.0) * 0.4)
            fig.add_trace(go.Scatter(
                x=[epsilon],
                y=[current_accuracy],
                mode='markers',
                name='Current Setting',
                marker=dict(size=15, color='#ef4444')
            ))
            
            fig.update_layout(
                title='Privacy Budget vs Model Accuracy',
                xaxis_title='Privacy Budget (ε)',
                yaxis_title='Model Accuracy',
                template='plotly_dark',
                font=dict(color='white'),
                paper_bgcolor='#1e293b',
                plot_bgcolor='#1e293b',
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_training_progress(self):
        """Render training progress visualization"""
        st.markdown('<div class="section-header">📈 Training Progress</div>', unsafe_allow_html=True)
        
        if not st.session_state.training_rounds:
            st.markdown('''
            <div class="card">
                <h3>🎯 No Training Data Available</h3>
                <p>Start training to see real-time accuracy progress across rounds.</p>
            </div>
            ''', unsafe_allow_html=True)
            return
        
        # Extract data from training rounds
        rounds = [r.round for r in st.session_state.training_rounds]
        client_a_acc = [r.client_a_accuracy for r in st.session_state.training_rounds]
        client_b_acc = [r.client_b_accuracy for r in st.session_state.training_rounds]
        client_c_acc = [r.client_c_accuracy for r in st.session_state.training_rounds]
        global_acc = [r.global_accuracy for r in st.session_state.training_rounds]
        
        # Create accuracy progress chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=rounds,
            y=client_a_acc,
            mode='lines+markers',
            name='Client A (Trucks)',
            line=dict(color='#10b981', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=rounds,
            y=client_b_acc,
            mode='lines+markers',
            name='Client B (Birds)',
            line=dict(color='#f59e0b', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=rounds,
            y=client_c_acc,
            mode='lines+markers',
            name='Client C (Mixed)',
            line=dict(color='#ef4444', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=rounds,
            y=global_acc,
            mode='lines+markers',
            name='Global Model',
            line=dict(color='#3b82f6', width=4, dash='dash'),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title='📈 Accuracy Progress Over Training Rounds',
            xaxis_title='Training Round',
            yaxis_title='Accuracy',
            yaxis=dict(range=[0, 1]),
            template='plotly_dark',
            font=dict(color='white'),
            paper_bgcolor='#1e293b',
            plot_bgcolor='#1e293b',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Latest metrics
        if st.session_state.training_rounds:
            latest = st.session_state.training_rounds[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🚚 Client A", f"{latest.client_a_accuracy:.3f}")
            
            with col2:
                st.metric("🐦 Client B", f"{latest.client_b_accuracy:.3f}")
            
            with col3:
                st.metric("🎯 Client C", f"{latest.client_c_accuracy:.3f}")
            
            with col4:
                st.metric("🌐 Global", f"{latest.global_accuracy:.3f}")
    
    def render_privacy_log(self):
        """Render real-time privacy log"""
        st.markdown('<div class="section-header">📝 Privacy Log</div>', unsafe_allow_html=True)
        
        # Privacy log display
        log_container = st.container()
        
        with log_container:
            st.markdown('<div class="privacy-log">', unsafe_allow_html=True)
            
            # Display log entries
            if st.session_state.privacy_log:
                for entry in st.session_state.privacy_log[-10:]:  # Show last 10 entries
                    st.markdown(f'<div style="color: #10b981; margin: 0.2rem 0;">{entry}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="color: #6b7280;">Waiting for training activity...</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def _add_privacy_log(self, message: str):
        """Add entry to privacy log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        st.session_state.privacy_log.append(log_entry)
        
        # Keep only last 50 entries
        if len(st.session_state.privacy_log) > 50:
            st.session_state.privacy_log = st.session_state.privacy_log[-50:]
    
    def render_training_controls(self):
        """Render training control panel"""
        st.markdown('<div class="section-header">🚀 Training Control</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("▶️ Start Training", type="primary", disabled=st.session_state.training_active):
                st.session_state.training_active = True
                st.session_state.training_completed = False
                self._add_privacy_log("🚀 Training started with 3 clients")
                self._simulate_training()
        
        with col2:
            if st.button("⏸️ Pause Training", disabled=not st.session_state.training_active):
                st.session_state.training_active = False
                self._add_privacy_log("⏸️ Training paused by user")
        
        with col3:
            if st.button("🔄 Reset Training"):
                st.session_state.training_active = False
                st.session_state.training_completed = False
                st.session_state.training_rounds = []
                st.session_state.privacy_log = []
                st.session_state.client_status = {'A': 'Idle', 'B': 'Idle', 'C': 'Idle'}
                self._add_privacy_log("🔄 Training reset")
                st.rerun()
    
    def _simulate_training(self):
        """Simulate federated learning training with privacy"""
        num_rounds = 10
        epsilon = st.session_state.current_epsilon
        
        for round_num in range(num_rounds):
            if not st.session_state.training_active:
                break
            
            # Update client statuses
            st.session_state.client_status = {'A': 'Training', 'B': 'Training', 'C': 'Training'}
            st.session_state.server_status = 'Aggregating'
            
            # Simulate client training with privacy
            noise_sigma = epsilon * 0.1  # Noise based on epsilon
            
            # Client-specific accuracies (based on their data distribution)
            base_acc_a = 0.6 + (round_num / num_rounds) * 0.2  # Trucks-focused
            base_acc_b = 0.5 + (round_num / num_rounds) * 0.25  # Birds-focused
            base_acc_c = 0.7 + (round_num / num_rounds) * 0.15  # Mixed (best baseline)
            
            # Add privacy impact
            privacy_impact = max(0.1, 1.0 - (epsilon / 10.0))
            
            client_a_acc = base_acc_a - privacy_impact * 0.1 + np.random.normal(0, 0.02)
            client_b_acc = base_acc_b - privacy_impact * 0.1 + np.random.normal(0, 0.02)
            client_c_acc = base_acc_c - privacy_impact * 0.1 + np.random.normal(0, 0.02)
            
            # Global accuracy (aggregated)
            global_acc = (client_a_acc + client_b_acc + client_c_acc) / 3 + np.random.normal(0, 0.01)
            
            # Create training round
            training_round = TrainingRound(
                round=round_num + 1,
                client_a_accuracy=np.clip(client_a_acc, 0, 1),
                client_b_accuracy=np.clip(client_b_acc, 0, 1),
                client_c_accuracy=np.clip(client_c_acc, 0, 1),
                global_accuracy=np.clip(global_acc, 0, 1),
                epsilon=epsilon,
                noise_sigma=noise_sigma,
                timestamp=datetime.now().strftime("%H:%M:%S")
            )
            
            st.session_state.training_rounds.append(training_round)
            
            # Log privacy obfuscation
            self._add_privacy_log(f"Client A weights obfuscated with σ={noise_sigma:.4f} noise. Uploading...")
            self._add_privacy_log(f"Client B weights obfuscated with σ={noise_sigma:.4f} noise. Uploading...")
            self._add_privacy_log(f"Client C weights obfuscated with σ={noise_sigma:.4f} noise. Uploading...")
            self._add_privacy_log(f"🔐 Server aggregated weights from 3 clients")
            self._add_privacy_log(f"📈 Round {round_num + 1}: Global accuracy = {global_acc:.3f}")
            
            # Update client statuses
            st.session_state.client_status = {'A': 'Idle', 'B': 'Idle', 'C': 'Idle'}
            st.session_state.server_status = 'Waiting'
            
            # Update session state
            st.session_state.current_round = round_num + 1
            
            # Simulate delay
            time.sleep(1)
        
        st.session_state.training_active = False
        st.session_state.training_completed = True
        self._add_privacy_log("✅ Training completed successfully!")
    
    def run(self):
        """Run the dashboard"""
        # Header
        self.render_header()
        
        # Main content layout
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_network_map()
            self.render_privacy_controls()
        
        with col2:
            self.render_training_progress()
            self.render_privacy_log()
        
        # Training controls
        self.render_training_controls()
        
        # Auto-refresh
        if st.sidebar.checkbox("🔄 Auto Refresh", value=False):
            time.sleep(2)
            st.rerun()

def main():
    """Main function"""
    dashboard = PrivacyFLDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
