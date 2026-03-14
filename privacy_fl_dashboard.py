#!/usr/bin/env python3
"""
Privacy-First Federated Learning Dashboard
Advanced Visualization and Monitoring Interface

This dashboard provides comprehensive monitoring of the Privacy-First FL system
including real-time training progress, privacy budget tracking, and performance analysis.
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
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import threading
import queue

# Set page configuration
st.set_page_config(
    page_title="Privacy-First Federated Learning Dashboard",
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
    
    .privacy-metric {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: white;
        font-weight: bold;
    }
    
    .warning-metric {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: white;
        font-weight: bold;
    }
    
    .error-metric {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
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
</style>
""", unsafe_allow_html=True)

@dataclass
class TrainingMetrics:
    """Training metrics for visualization"""
    round: int
    accuracy: float
    loss: float
    privacy_spent: float
    privacy_budget: float
    num_clients: int
    communication_cost: float
    training_time: float
    timestamp: str

@dataclass
class PrivacyMetrics:
    """Privacy-specific metrics"""
    epsilon: float
    delta: float
    noise_multiplier: float
    max_grad_norm: float
    secure_aggregation: bool
    privacy_spent: float
    privacy_remaining: float

class PrivacyFLDashboard:
    """Privacy-First Federated Learning Dashboard"""
    
    def __init__(self):
        self.training_history = []
        self.privacy_metrics = []
        self.client_metrics = []
        self.is_training = False
        self.current_round = 0
        self.total_rounds = 20
        
        # Initialize session state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables"""
        defaults = {
            'training_active': False,
            'current_epsilon': 3.0,
            'current_delta': 1e-5,
            'noise_multiplier': 1.0,
            'max_grad_norm': 1.0,
            'secure_aggregation': True,
            'num_clients': 10,
            'num_rounds': 20,
            'local_epochs': 5,
            'batch_size': 32,
            'learning_rate': 0.001,
            'dataset_loaded': False,
            'training_completed': False,
            'privacy_budget_analysis': None,
            'fl_results': None,
            'centralized_results': None
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">🔒 Privacy-First Federated Learning Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Quick status overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.session_state.training_active:
                st.markdown('<div class="privacy-metric">🤖 Training Active</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-metric">⏸️ Training Paused</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="privacy-metric">🔒 ε = {st.session_state.current_epsilon}</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'<div class="privacy-metric">👥 {st.session_state.num_clients} Clients</div>', unsafe_allow_html=True)
        
        with col4:
            if st.session_state.training_completed:
                st.markdown('<div class="privacy-metric">✅ Completed</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-metric">⏳ In Progress</div>', unsafe_allow_html=True)
    
    def render_privacy_configuration(self):
        """Render privacy configuration panel"""
        st.markdown('<div class="section-header">🔒 Privacy Configuration</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🔐 Differential Privacy Settings")
            
            epsilon = st.slider("Privacy Budget (ε)", 0.1, 10.0, st.session_state.current_epsilon, 0.1)
            delta = st.slider("Failure Probability (δ)", 1e-6, 1e-3, st.session_state.current_delta, format="%.2e")
            noise_multiplier = st.slider("Noise Multiplier", 0.1, 5.0, st.session_state.noise_multiplier, 0.1)
            max_grad_norm = st.slider("Max Gradient Norm", 0.1, 10.0, st.session_state.max_grad_norm, 0.1)
            
            st.session_state.current_epsilon = epsilon
            st.session_state.current_delta = delta
            st.session_state.noise_multiplier = noise_multiplier
            st.session_state.max_grad_norm = max_grad_norm
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🛡️ Security Settings")
            
            secure_aggregation = st.checkbox("Enable Secure Aggregation", st.session_state.secure_aggregation)
            st.session_state.secure_aggregation = secure_aggregation
            
            if secure_aggregation:
                st.success("🔒 Secure Aggregation enabled - server cannot see individual updates")
            else:
                st.warning("⚠️ Secure Aggregation disabled - individual updates visible to server")
            
            # Privacy budget visualization
            remaining_budget = epsilon - (epsilon * 0.1)  # Simulate spent budget
            budget_percentage = (remaining_budget / epsilon) * 100
            
            st.markdown("### 📊 Privacy Budget Status")
            st.progress(budget_percentage / 100)
            st.markdown(f"**Remaining**: {remaining_budget:.2f} / {epsilon:.2f} ({budget_percentage:.1f}%)")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_training_configuration(self):
        """Render training configuration panel"""
        st.markdown('<div class="section-header">🤖 Training Configuration</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("⚙️ Federated Learning Settings")
            
            num_clients = st.slider("Number of Clients", 3, 50, st.session_state.num_clients)
            num_rounds = st.slider("Number of Rounds", 5, 100, st.session_state.num_rounds)
            local_epochs = st.slider("Local Epochs", 1, 20, st.session_state.local_epochs)
            batch_size = st.slider("Batch Size", 16, 128, st.session_state.batch_size)
            
            st.session_state.num_clients = num_clients
            st.session_state.num_rounds = num_rounds
            st.session_state.local_epochs = local_epochs
            st.session_state.batch_size = batch_size
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📊 Non-IID Data Configuration")
            
            partition_type = st.selectbox(
                "Data Partition Type",
                ["dirichlet", "pathological"],
                format_func=lambda x: {
                    "dirichlet": "📊 Dirichlet Distribution",
                    "pathological": "🎯 Pathological Partition"
                }[x]
            )
            
            alpha = st.slider("Dirichlet Alpha (Non-IID Level)", 0.1, 10.0, 0.5, 0.1)
            
            if partition_type == "dirichlet":
                st.info(f"📊 Alpha = {alpha} (lower = more Non-IID)")
            else:
                st.info("🎯 Each client gets only specific classes")
            
            # Data distribution visualization
            self._simulate_data_distribution(num_clients, partition_type, alpha)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def _simulate_data_distribution(self, num_clients: int, partition_type: str, alpha: float):
        """Simulate and visualize data distribution"""
        # Create simulation data
        np.random.seed(42)
        
        if partition_type == "dirichlet":
            # Dirichlet distribution simulation
            proportions = np.random.dirichlet([alpha] * num_clients, size=10)
            
            fig = go.Figure(data=go.Heatmap(
                z=proportions,
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Proportion")
            ))
            
            fig.update_layout(
                title="📊 Non-IID Data Distribution (Dirichlet)",
                xaxis_title="Client ID",
                yaxis_title="Class",
                template='plotly_dark',
                font=dict(color='white'),
                paper_bgcolor='#1e293b',
                plot_bgcolor='#1e293b'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Pathological distribution visualization
            classes_per_client = 2
            client_classes = {}
            
            for i in range(num_clients):
                client_classes[i] = np.random.choice(10, classes_per_client, replace=False)
            
            # Create visualization
            fig = go.Figure()
            
            for client_id, classes in client_classes.items():
                fig.add_trace(go.Bar(
                    x=[f"Class {c}" for c in classes],
                    y=[1] * len(classes),
                    name=f"Client {client_id}",
                    opacity=0.7
                ))
            
            fig.update_layout(
                title="🎯 Pathological Data Distribution",
                xaxis_title="Class",
                yaxis_title="Presence",
                template='plotly_dark',
                font=dict(color='white'),
                paper_bgcolor='#1e293b',
                plot_bgcolor='#1e293b',
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_training_control(self):
        """Render training control panel"""
        st.markdown('<div class="section-header">🚀 Training Control</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("▶️ Start Training", type="primary", disabled=st.session_state.training_active):
                st.session_state.training_active = True
                st.session_state.training_completed = False
                self._simulate_training()
        
        with col2:
            if st.button("⏸️ Pause Training", disabled=not st.session_state.training_active):
                st.session_state.training_active = False
        
        with col3:
            if st.button("🔄 Reset Training"):
                st.session_state.training_active = False
                st.session_state.training_completed = False
                self.training_history = []
                st.rerun()
    
    def _simulate_training(self):
        """Simulate federated learning training progress"""
        # Generate simulated training data
        num_rounds = st.session_state.num_rounds
        num_clients = st.session_state.num_clients
        epsilon = st.session_state.current_epsilon
        
        # Simulate training progress
        for round_num in range(num_rounds):
            if not st.session_state.training_active:
                break
            
            # Simulate metrics
            base_accuracy = 0.6 + (round_num / num_rounds) * 0.3
            accuracy = base_accuracy + np.random.normal(0, 0.02)
            loss = 1.0 - accuracy + np.random.normal(0, 0.01)
            
            # Privacy spending
            privacy_spent_per_round = epsilon / num_rounds
            total_privacy_spent = (round_num + 1) * privacy_spent_per_round
            
            # Communication cost
            comm_cost = num_clients * 1000 * (round_num + 1)  # Simulated
            
            # Training time
            training_time = np.random.uniform(30, 60)  # seconds
            
            # Create metrics
            metrics = TrainingMetrics(
                round=round_num + 1,
                accuracy=accuracy,
                loss=loss,
                privacy_spent=total_privacy_spent,
                privacy_budget=epsilon,
                num_clients=num_clients,
                communication_cost=comm_cost,
                training_time=training_time,
                timestamp=datetime.now().strftime("%H:%M:%S")
            )
            
            self.training_history.append(metrics)
            
            # Update session state
            st.session_state.current_round = round_num + 1
            
            # Simulate delay
            time.sleep(0.5)
        
        st.session_state.training_active = False
        st.session_state.training_completed = True
    
    def render_training_progress(self):
        """Render training progress visualization"""
        st.markdown('<div class="section-header">📊 Training Progress</div>', unsafe_allow_html=True)
        
        if not self.training_history:
            st.markdown('''
            <div class="card">
                <h3>🎯 No Training Data Available</h3>
                <p>Start training to see real-time progress visualization.</p>
            </div>
            ''', unsafe_allow_html=True)
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('📈 Accuracy Progress', '📉 Loss Progress', '🔒 Privacy Budget', '👥 Client Participation'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        rounds = [m.round for m in self.training_history]
        accuracies = [m.accuracy for m in self.training_history]
        losses = [m.loss for m in self.training_history]
        privacy_spent = [m.privacy_spent for m in self.training_history]
        num_clients = [m.num_clients for m in self.training_history]
        
        # Accuracy
        fig.add_trace(
            go.Scatter(x=rounds, y=accuracies, name='Accuracy', line=dict(color='#10b981', width=3)),
            row=1, col=1
        )
        
        # Loss
        fig.add_trace(
            go.Scatter(x=rounds, y=losses, name='Loss', line=dict(color='#ef4444', width=3)),
            row=1, col=2
        )
        
        # Privacy Budget
        fig.add_trace(
            go.Scatter(x=rounds, y=privacy_spent, name='Privacy Spent', line=dict(color='#3b82f6', width=3)),
            row=2, col=1
        )
        
        # Client Participation
        fig.add_trace(
            go.Bar(x=rounds, y=num_clients, name='Clients', marker_color='#8b5cf6'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            template='plotly_dark',
            font=dict(color='white'),
            paper_bgcolor='#1e293b',
            plot_bgcolor='#1e293b',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Latest metrics
        if self.training_history:
            latest = self.training_history[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📈 Final Accuracy", f"{latest.accuracy:.4f}")
            
            with col2:
                st.metric("📉 Final Loss", f"{latest.loss:.4f}")
            
            with col3:
                st.metric("🔒 Privacy Spent", f"{latest.privacy_spent:.2f}")
            
            with col4:
                st.metric("👥 Active Clients", latest.num_clients)
    
    def render_privacy_analysis(self):
        """Render privacy budget analysis"""
        st.markdown('<div class="section-header">🔒 Privacy Budget Analysis</div>', unsafe_allow_html=True)
        
        # Privacy budget vs accuracy trade-off
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📊 Privacy Budget vs Accuracy")
            
            # Simulate privacy-accuracy trade-off
            epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            accuracies = []
            
            for eps in epsilon_values:
                # Simulate accuracy based on epsilon (higher epsilon = higher accuracy)
                base_acc = 0.6
                acc_increase = min(0.3, np.log(eps + 1) / np.log(11) * 0.3)
                accuracy = base_acc + acc_increase + np.random.normal(0, 0.02)
                accuracies.append(accuracy)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=epsilon_values,
                y=accuracies,
                mode='lines+markers',
                name='Accuracy',
                line=dict(color='#10b981', width=4),
                marker=dict(size=10, color='#10b981')
            ))
            
            fig.update_layout(
                title='Privacy Budget (ε) vs Model Accuracy',
                xaxis_title='Privacy Budget (ε)',
                yaxis_title='Model Accuracy',
                xaxis_type='log',
                template='plotly_dark',
                font=dict(color='white'),
                paper_bgcolor='#1e293b',
                plot_bgcolor='#1e293b'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📈 Privacy Spending Analysis")
            
            # Privacy spending over time
            if self.training_history:
                rounds = [m.round for m in self.training_history]
                privacy_spent = [m.privacy_spent for m in self.training_history]
                privacy_budget = [m.privacy_budget for m in self.training_history]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=rounds,
                    y=privacy_spent,
                    mode='lines+markers',
                    name='Privacy Spent',
                    line=dict(color='#ef4444', width=4),
                    marker=dict(size=8, color='#ef4444')
                ))
                
                fig.add_trace(go.Scatter(
                    x=rounds,
                    y=privacy_budget,
                    mode='lines',
                    name='Privacy Budget',
                    line=dict(color='#3b82f6', width=2, dash='dash'),
                    fill='tonexty'
                ))
                
                fig.update_layout(
                    title='Privacy Spending Over Time',
                    xaxis_title='Training Round',
                    yaxis_title='Privacy Budget (ε)',
                    template='plotly_dark',
                    font=dict(color='white'),
                    paper_bgcolor='#1e293b',
                    plot_bgcolor='#1e293b'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Start training to see privacy spending analysis")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_client_analysis(self):
        """Render client analysis panel"""
        st.markdown('<div class="section-header">👥 Client Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📊 Client Performance Distribution")
            
            # Simulate client performance
            num_clients = st.session_state.num_clients
            client_accuracies = np.random.normal(0.75, 0.1, num_clients)
            client_accuracies = np.clip(client_accuracies, 0.3, 0.95)
            
            fig = go.Figure(data=[
                go.Histogram(
                    x=client_accuracies,
                    nbinsx=20,
                    marker_color='#8b5cf6',
                    opacity=0.7
                )
            ])
            
            fig.update_layout(
                title='Client Accuracy Distribution',
                xaxis_title='Accuracy',
                yaxis_title='Number of Clients',
                template='plotly_dark',
                font=dict(color='white'),
                paper_bgcolor='#1e293b',
                plot_bgcolor='#1e293b'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🔒 Client Privacy Settings")
            
            # Simulate client privacy settings
            privacy_enabled = np.random.choice([True, False], num_clients, p=[0.7, 0.3])
            noise_multipliers = np.random.uniform(0.5, 2.0, num_clients)
            
            # Create client data
            client_data = []
            for i in range(num_clients):
                client_data.append({
                    'Client ID': f'Client_{i}',
                    'Privacy Enabled': privacy_enabled[i],
                    'Noise Multiplier': noise_multipliers[i],
                    'Accuracy': client_accuracies[i]
                })
            
            df = pd.DataFrame(client_data)
            
            # Privacy enabled distribution
            privacy_counts = df['Privacy Enabled'].value_counts()
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=['Privacy Enabled', 'No Privacy'],
                    values=[privacy_counts.get(True, 0), privacy_counts.get(False, 0)],
                    marker_colors=['#10b981', '#ef4444']
                )
            ])
            
            fig.update_layout(
                title='Client Privacy Adoption',
                template='plotly_dark',
                font=dict(color='white'),
                paper_bgcolor='#1e293b',
                plot_bgcolor='#1e293b'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show client table
            st.markdown("### 📋 Client Details")
            st.dataframe(df, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def run(self):
        """Run the dashboard"""
        # Header
        self.render_header()
        
        # Sidebar navigation
        with st.sidebar:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); 
                        padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                <h2 style="color: white; margin: 0;">🧭 Navigation</h2>
            </div>
            """, unsafe_allow_html=True)
            
            page = st.selectbox(
                "Choose Section",
                ["🔒 Privacy Configuration", "🤖 Training Control", "📊 Training Progress", 
                 "🔒 Privacy Analysis", "👥 Client Analysis"],
                key="navigation"
            )
        
        # Main content
        if page == "🔒 Privacy Configuration":
            self.render_privacy_configuration()
        elif page == "🤖 Training Control":
            self.render_training_configuration()
            self.render_training_control()
        elif page == "📊 Training Progress":
            self.render_training_progress()
        elif page == "🔒 Privacy Analysis":
            self.render_privacy_analysis()
        elif page == "👥 Client Analysis":
            self.render_client_analysis()
        
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
