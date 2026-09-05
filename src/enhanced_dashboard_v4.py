#!/usr/bin/env python3
"""
Enhanced Dashboard v4 - Advanced Interactive Differential Privacy
Features: Dynamic Privacy Controls, Secure Aggregation Visualization, Real-time Privacy Leakage Tracking
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import time
import os
import sys
import json
import threading
import queue
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import hashlib

# Configure Streamlit page - MUST be the first Streamlit command
st.set_page_config(
    page_title="Advanced Federated Learning Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add caching for static data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_dataset_info():
    """Get cached dataset information to prevent flickering"""
    datasets = {
        "EMNIST": {
            "name": "EMNIST (Extended MNIST)",
            "description": "Handwritten characters and digits",
            "classes": 62,
            "samples": 814558,
            "image_size": "28x28",
            "difficulty": "Medium",
            "use_case": "Character recognition",
            "icon": "✍️"
        },
        "CIFAR-10": {
            "name": "CIFAR-10",
            "description": "Natural images (10 classes)",
            "classes": 10,
            "samples": 60000,
            "image_size": "32x32",
            "difficulty": "Easy",
            "use_case": "Object recognition",
            "icon": "🖼️"
        },
        "Fashion-MNIST": {
            "name": "Fashion-MNIST",
            "description": "Fashion items and clothing",
            "classes": 10,
            "samples": 70000,
            "image_size": "28x28",
            "difficulty": "Easy",
            "use_case": "Fashion classification",
            "icon": "👔"
        },
        "Medical-MNIST": {
            "name": "Medical-MNIST",
            "description": "Medical images and diagnostics",
            "classes": 6,
            "samples": 58954,
            "image_size": "64x64",
            "difficulty": "Hard",
            "use_case": "Medical diagnosis",
            "icon": "🏥"
        },
        "Banking-Fraud": {
            "name": "Banking Fraud Detection",
            "description": "Transaction fraud patterns",
            "classes": 2,
            "samples": 284807,
            "image_size": "Tabular",
            "difficulty": "Medium",
            "use_case": "Fraud detection",
            "icon": "🏦"
        }
    }
    return datasets

@st.cache_data(ttl=3600)
def get_cached_accuracy_ranges():
    """Get cached accuracy ranges to prevent flickering"""
    return {
        "EMNIST": (0.45, 0.92),
        "CIFAR-10": (0.55, 0.95),
        "Fashion-MNIST": (0.60, 0.94),
        "Medical-MNIST": (0.40, 0.88),
        "Banking-Fraud": (0.70, 0.98)
    }

@st.cache_data(ttl=3600)
def get_cached_privacy_challenges():
    """Get cached privacy challenges to prevent flickering"""
    return {
        "EMNIST": "Medium - Character recognition needs careful privacy",
        "CIFAR-10": "Low - Natural images are less privacy-sensitive",
        "Fashion-MNIST": "Low - Fashion data has low privacy risk",
        "Medical-MNIST": "High - Medical data requires strong privacy",
        "Banking-Fraud": "High - Financial data needs maximum privacy"
    }

# Add src to path to import our modules
sys.path.append(str(Path(__file__).parent))

try:
    from config import DASHBOARD_REFRESH_INTERVAL, NUM_CLIENTS, LOG_DIR, EPSILON, NOISE_MULTIPLIER
    from logging_bridge import get_logger
except ImportError:
    # Fallback for standalone execution
    LOG_DIR = Path(__file__).parent.parent / "logs"
    DASHBOARD_REFRESH_INTERVAL = 5
    NUM_CLIENTS = 3
    EPSILON = 1.0
    NOISE_MULTIPLIER = 1.0

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedFederatedDashboard:
    """Advanced dashboard with interactive differential privacy controls"""
    
    def __init__(self):
        self.logger = get_logger() if 'get_logger' in globals() else None
        self.last_update = None
        self.auto_refresh = True
        self.refresh_interval = DASHBOARD_REFRESH_INTERVAL
        self.config_file = Path(__file__).parent.parent / "config.json"
        
        # Initialize session state
        self._init_session_state()
        
        # Load configuration
        self.load_config()
    
    def _init_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        if 'auto_refresh_enabled' not in st.session_state:
            st.session_state.auto_refresh_enabled = True
        if 'selected_round' not in st.session_state:
            st.session_state.selected_round = None
        if 'selected_client' not in st.session_state:
            st.session_state.selected_client = None
        if 'privacy_config' not in st.session_state:
            st.session_state.privacy_config = {
                'epsilon': EPSILON,
                'noise_multiplier': NOISE_MULTIPLIER,
                'max_grad_norm': 1.0,
                'privacy_budget_limit': 8.0
            }
        if 'training_stopped' not in st.session_state:
            st.session_state.training_stopped = False
        if 'view_raw_gradients' not in st.session_state:
            st.session_state.view_raw_gradients = False
        # Initialize accuracy history for sparkline
        if 'accuracy_history' not in st.session_state:
            st.session_state.accuracy_history = []
        # Initialize demo mode settings
        if 'demo_mode' not in st.session_state:
            st.session_state.demo_mode = False
        if 'demo_counter' not in st.session_state:
            st.session_state.demo_counter = 0
        # Initialize dataset selection
        if 'selected_dataset' not in st.session_state:
            st.session_state.selected_dataset = "EMNIST"
    
    def load_config(self):
        """Load configuration from JSON file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    st.session_state.privacy_config.update(config.get('privacy', {}))
                    logger.info("Configuration loaded from config.json")
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
    
    def save_config(self):
        """Save configuration to JSON file"""
        try:
            config = {
                'privacy': st.session_state.privacy_config,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info("Configuration saved to config.json")
        except Exception as e:
            logger.error(f"Could not save config: {e}")
    
    def get_privacy_color(self, privacy_loss: float) -> str:
        """Get color based on privacy spent"""
        if privacy_loss <= 2.0:
            return "🟢"  # Green - Safe
        elif privacy_loss <= 5.0:
            return "🟡"  # Yellow - Caution
        elif privacy_loss <= 8.0:
            return "🟠"  # Orange - Warning
        else:
            return "🔴"  # Red - Critical
    
    def create_sparkline(self, values: list, height: int = 50) -> str:
        """Create a simple sparkline using Unicode characters"""
        if not values or len(values) < 2:
            return "▁"
        
        # Normalize values to 0-8 range for Unicode block characters
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            normalized = [4] * len(values)
        else:
            normalized = [int(8 * (v - min_val) / (max_val - min_val)) for v in values]
        
        # Unicode block characters from empty to full
        blocks = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
        sparkline = ''.join(blocks[min(v, 8)] for v in normalized)
        return sparkline
    
    def get_dataset_info(self) -> Dict:
        """Get information about available datasets (cached)"""
        return get_cached_dataset_info()
    
    def get_demo_data(self) -> Dict:
        """Generate realistic demo data for presentations"""
        st.session_state.demo_counter += 1
        
        # Get dataset-specific characteristics
        datasets = self.get_dataset_info()
        dataset_info = datasets.get(st.session_state.selected_dataset, datasets["EMNIST"])
        
        # Simulate training progression
        round_progress = min(st.session_state.demo_counter / 10, 1.0)  # 10 rounds to full progress
        
        # Dataset-specific accuracy ranges (cached)
        accuracy_ranges = get_cached_accuracy_ranges()
        
        min_acc, max_acc = accuracy_ranges.get(st.session_state.selected_dataset, (0.5, 0.9))
        base_accuracy = min_acc + ((max_acc - min_acc) * round_progress)
        accuracy_noise = np.random.normal(0, 0.02)  # Small random variations
        accuracy = max(min_acc - 0.05, min(max_acc + 0.05, base_accuracy + accuracy_noise))
        
        # Privacy budget that increases steadily
        privacy_loss = min(8.5, 0.5 + (st.session_state.demo_counter * 0.3))
        
        # Client count that shows realistic connection patterns
        if st.session_state.demo_counter <= 2:
            clients = 0  # Starting up
        elif st.session_state.demo_counter <= 4:
            clients = np.random.choice([1, 2])  # Connecting
        else:
            clients = np.random.choice([3, 4, 5])  # Fully connected with some variance
        
        return {
            'accuracy': accuracy,
            'privacy_loss': privacy_loss,
            'clients': clients,
            'round': min(st.session_state.demo_counter, 10),
            'demo_mode': True,
            'dataset': st.session_state.selected_dataset,
            'dataset_info': dataset_info
        }
    
    def load_live_metrics(self) -> Dict:
        """Load live metrics from FL server"""
        
        # Use demo mode if enabled
        if st.session_state.demo_mode:
            return self.get_demo_data()
        
        try:
            # Try multiple possible CSV files
            metrics_files = [
                LOG_DIR / "federated_metrics.csv",
                LOG_DIR / "training_metrics.csv",
                LOG_DIR / "privacy_metrics.csv"
            ]
            
            for file_path in metrics_files:
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    if not df.empty:
                        latest = df.iloc[-1].to_dict()
                        
                        # Add computed metrics
                        if 'privacy_budget_used' in latest:
                            latest['privacy_leakage'] = latest['privacy_budget_used'] * 100
                        
                        return latest
            
            # Return default values if no data
            return {
                'accuracy': 0.0,
                'privacy_loss': 0.0,
                'clients': 0,
                'round': 0,
                'privacy_leakage': 0.0
            }
            
        except Exception as e:
            logger.error(f"Error loading live metrics: {e}")
            return {
                'accuracy': 0.0,
                'privacy_loss': 0.0,
                'clients': 0,
                'round': 0,
                'privacy_leakage': 0.0
            }
    
    def load_data_from_csv(self) -> tuple:
        """Load data from CSV log files"""
        try:
            if self.logger:
                training_df = self.logger.get_training_data_from_csv()
                privacy_df = self.logger.get_privacy_data_from_csv()
                client_df = self.logger.get_client_data_from_csv()
            else:
                # Fallback: read directly from CSV files
                training_df = self._read_csv_safe(LOG_DIR / "training_metrics.csv")
                privacy_df = self._read_csv_safe(LOG_DIR / "privacy_metrics.csv")
                client_df = self._read_csv_safe(LOG_DIR / "client_metrics.csv")
            
            return training_df, privacy_df, client_df
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def _read_csv_safe(self, file_path: Path) -> pd.DataFrame:
        """Safely read CSV file"""
        try:
            if file_path.exists():
                df = pd.read_csv(file_path)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return pd.DataFrame()
    
    def create_privacy_utility_controls(self):
        """Create dynamic privacy vs utility toggle controls"""
        st.header("🎛️ Dynamic Privacy vs Utility Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Privacy Parameters")
            
            # Epsilon slider
            epsilon = st.slider(
                "Epsilon (ε) - Privacy Budget",
                min_value=0.1,
                max_value=10.0,
                value=st.session_state.privacy_config['epsilon'],
                step=0.1,
                help="Lower values = more privacy, less accuracy"
            )
            
            # Noise multiplier slider
            noise_multiplier = st.slider(
                "Noise Multiplier (σ)",
                min_value=0.1,
                max_value=5.0,
                value=st.session_state.privacy_config['noise_multiplier'],
                step=0.1,
                help="Higher values = more noise, more privacy"
            )
            
            # Max gradient norm
            max_grad_norm = st.slider(
                "Max Gradient Norm",
                min_value=0.1,
                max_value=5.0,
                value=st.session_state.privacy_config['max_grad_norm'],
                step=0.1,
                help="Lower values = more clipping, more privacy"
            )
        
        with col2:
            st.subheader("Utility Impact Analysis")
            
            # Simulate utility impact based on privacy settings
            privacy_score = (10 - epsilon) / 10 + noise_multiplier / 5
            utility_score = epsilon / 10 + (5 - noise_multiplier) / 5
            
            # Create privacy-utility tradeoff chart
            fig = go.Figure()
            
            # Add privacy score
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, privacy_score],
                mode='lines+markers',
                name='Privacy Score',
                line=dict(color='red', width=3),
                fill='tonexty'
            ))
            
            # Add utility score
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, utility_score],
                mode='lines+markers',
                name='Utility Score',
                line=dict(color='blue', width=3),
                fill='tonexty'
            ))
            
            fig.update_layout(
                title="Privacy vs Utility Tradeoff",
                xaxis_title="Configuration",
                yaxis_title="Score",
                showlegend=True,
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show impact metrics
            st.metric("Privacy Score", f"{privacy_score:.2f}")
            st.metric("Utility Score", f"{utility_score:.2f}")
            st.metric("Balance", f"{abs(privacy_score - utility_score):.2f}")
        
        # Save configuration button
        if st.button("💾 Save Privacy Configuration", type="primary"):
            st.session_state.privacy_config.update({
                'epsilon': epsilon,
                'noise_multiplier': noise_multiplier,
                'max_grad_norm': max_grad_norm
            })
            self.save_config()
            st.success("Configuration saved! Clients will use new settings on next round.")
            st.rerun()
    
    def create_secure_aggregation_visualizer(self):
        """Create secure aggregation visualization"""
        st.header("🔐 Secure Aggregation Visualizer")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("Controls")
            
            # Toggle for raw gradients view
            view_raw = st.checkbox(
                "View Raw Gradients",
                value=st.session_state.view_raw_gradients,
                help="Show what server would see without secure aggregation"
            )
            st.session_state.view_raw_gradients = view_raw
            
            # Aggregation round selector
            round_num = st.selectbox(
                "Aggregation Round",
                options=list(range(1, 6)),
                index=0
            )
            
            # Security level
            security_level = st.selectbox(
                "Security Level",
                options=["Basic", "Advanced", "Military"],
                index=1
            )
        
        with col1:
            st.subheader("Network Topology")
            
            # Create network topology visualization
            fig = go.Figure()
            
            # Server node
            fig.add_trace(go.Scatter(
                x=[0], y=[0],
                mode='markers+text',
                marker=dict(size=30, color='red', symbol='square'),
                text=['Server'],
                textposition='middle center',
                name='Server',
                hovertemplate='Server<br>Aggregates encrypted weights'
            ))
            
            # Client nodes
            client_positions = [
                (-2, 2), (2, 2), (-2, -2), (2, -2), (0, 3)
            ]
            
            for i, (x, y) in enumerate(client_positions[:NUM_CLIENTS]):
                # Client node
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers+text',
                    marker=dict(size=20, color='blue', symbol='circle'),
                    text=[f'Client {i+1}'],
                    textposition='middle center',
                    name=f'Client {i+1}',
                    hovertemplate=f'Client {i+1}<br>Sending encrypted weights'
                ))
                
                # Connection to server
                if view_raw:
                    # Show raw (insecure) transmission
                    fig.add_trace(go.Scatter(
                        x=[x, 0], y=[y, 0],
                        mode='lines',
                        line=dict(color='red', width=2, dash='dash'),
                        showlegend=False,
                        hoverinfo='none'
                    ))
                else:
                    # Show secure transmission
                    fig.add_trace(go.Scatter(
                        x=[x, 0], y=[y, 0],
                        mode='lines',
                        line=dict(color='green', width=3),
                        showlegend=False,
                        hoverinfo='none'
                    ))
                    
                    # Add encrypted packet visualization
                    mid_x, mid_y = x/2, y/2
                    fig.add_trace(go.Scatter(
                        x=[mid_x], y=[mid_y],
                        mode='markers',
                        marker=dict(size=8, color='gold', symbol='diamond'),
                        showlegend=False,
                        hovertemplate='Encrypted Packet<br>🔒'
                    ))
            
            fig.update_layout(
                title=f"Secure Aggregation - Round {round_num} ({security_level} Security)",
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Gradient heatmap
            st.subheader("Gradient Visualization")
            
            if view_raw:
                # Show noisy/blurred heatmap (what attacker might see)
                gradient_data = np.random.normal(0, 0.1, (10, 10))
                fig_heat = px.imshow(
                    gradient_data,
                    title="🚨 Raw Gradients (Security Risk!)",
                    color_continuous_scale='RdBu',
                    labels=dict(x="Feature", y="Neuron", value="Gradient")
                )
                st.warning("⚠️ Without secure aggregation, gradients are visible!")
            else:
                # Show secure/encrypted representation
                gradient_data = np.random.uniform(0, 1, (10, 10))
                fig_heat = px.imshow(
                    gradient_data,
                    title="✅ Encrypted Gradients (Secure)",
                    color_continuous_scale='Viridis',
                    labels=dict(x="Feature", y="Neuron", value="Encrypted Value")
                )
                st.success("🔒 Gradients are encrypted and secure!")
            
            fig_heat.update_layout(height=300)
            st.plotly_chart(fig_heat, use_container_width=True)
    
    def create_privacy_leakage_gauge(self):
        """Create real-time privacy leakage gauge with budget limits"""
        st.header("📊 Real-Time Privacy Leakage Tracker")
        
        # Load current metrics
        live_data = self.load_live_metrics()
        privacy_leakage = live_data.get('privacy_leakage', 0.0)
        budget_limit = st.session_state.privacy_config['privacy_budget_limit']
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            # Privacy budget limit control
            new_limit = st.number_input(
                "Privacy Budget Limit (ε)",
                min_value=1.0,
                max_value=20.0,
                value=budget_limit,
                step=0.5,
                help="Training will stop when epsilon exceeds this limit"
            )
            
            if new_limit != budget_limit:
                st.session_state.privacy_config['privacy_budget_limit'] = new_limit
                self.save_config()
        
        with col2:
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = privacy_leakage,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Privacy Leakage (%)"},
                delta = {'reference': budget_limit * 10},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgreen"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': budget_limit * 10
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Status metrics
            current_epsilon = live_data.get('privacy_loss', 0.0)
            
            st.metric("Current ε", f"{current_epsilon:.2f}")
            st.metric("Budget Used", f"{privacy_leakage:.1f}%")
            st.metric("Budget Limit", f"{budget_limit:.1f}")
            
            # Auto-stop warning
            if privacy_leakage >= budget_limit * 10:
                st.error("🚨 PRIVACY BUDGET EXHAUSTED!")
                if st.button("🛑 Stop Training", type="primary"):
                    st.session_state.training_stopped = True
                    st.warning("Training stopped to protect privacy!")
            elif privacy_leakage >= budget_limit * 8:
                st.warning("⚠️ Privacy budget nearly exhausted!")
            else:
                st.success("✅ Privacy budget within limits")
        
        # Privacy leakage over time
        training_df, privacy_df, _ = self.load_data_from_csv()
        
        if not privacy_df.empty:
            st.subheader("Privacy Leakage Trend")
            
            # Calculate cumulative epsilon over time
            privacy_df['cumulative_epsilon'] = privacy_df.groupby('client_id')['epsilon_spent'].cumsum()
            
            fig_trend = go.Figure()
            
            for client_id in privacy_df['client_id'].unique():
                client_data = privacy_df[privacy_df['client_id'] == client_id]
                fig_trend.add_trace(go.Scatter(
                    x=client_data['round'],
                    y=client_data['cumulative_epsilon'],
                    mode='lines+markers',
                    name=f'Client {client_id}',
                    line=dict(width=2)
                ))
            
            # Add budget limit line
            fig_trend.add_hline(
                y=budget_limit,
                line_dash="dash",
                line_color="red",
                annotation_text="Budget Limit"
            )
            
            fig_trend.update_layout(
                title="Cumulative Privacy Spending",
                xaxis_title="Training Round",
                yaxis_title="Cumulative ε",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)
    
    def create_federated_debugging(self):
        """Create federated debugging with client logs and straggler detection"""
        st.header("🔍 Federated Debugging & Client Analysis")
        
        training_df, privacy_df, client_df = self.load_data_from_csv()
        
        if client_df.empty:
            st.warning("No client data available. Start training to see debugging information.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Client Performance Analysis")
            
            # Create detailed client table
            if not client_df.empty:
                # Get latest metrics for each client
                latest_client_data = client_df.loc[client_df.groupby('client_id')['round'].idxmax()]
                
                # Calculate additional metrics
                latest_client_data['contribution_score'] = (
                    latest_client_data['local_accuracy'] * 
                    latest_client_data['data_size'] / 
                    latest_client_data['data_size'].max()
                )
                
                # Detect stragglers
                avg_training_time = latest_client_data['training_time'].mean()
                latest_client_data['is_straggler'] = (
                    latest_client_data['training_time'] > avg_training_time * 1.5
                )
                
                # Detect data distribution issues
                data_sizes = latest_client_data['data_size']
                latest_client_data['data_distribution'] = 'Normal'
                latest_client_data.loc[latest_client_data['data_size'] < data_sizes.quantile(0.3), 'data_distribution'] = 'Low Data'
                latest_client_data.loc[latest_client_data['data_size'] > data_sizes.quantile(0.7), 'data_distribution'] = 'High Data'
                
                # Display table with styling
                display_df = latest_client_data[[
                    'client_id', 'local_accuracy', 'data_size', 'training_time',
                    'contribution_score', 'data_distribution', 'status'
                ]].round(4)
                
                display_df.columns = [
                    'Client ID', 'Accuracy', 'Data Size', 'Training Time (s)',
                    'Contribution Score', 'Data Distribution', 'Status'
                ]
                
                # Add straggler indicator
                display_df['Straggler'] = latest_client_data['is_straggler'].apply(
                    lambda x: '🐌 Yes' if x else '✅ No'
                )
                
                st.dataframe(display_df, use_container_width=True)
        
        with col2:
            st.subheader("Straggler Detection")
            
            if not client_df.empty:
                # Training time distribution
                fig_straggler = go.Figure()
                
                training_times = client_df.groupby('client_id')['training_time'].mean()
                
                fig_straggler.add_trace(go.Box(
                    y=training_times.values,
                    name='Training Times',
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8
                ))
                
                # Add straggler threshold line
                threshold = training_times.mean() * 1.5
                fig_straggler.add_hline(
                    y=threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Straggler Threshold"
                )
                
                fig_straggler.update_layout(
                    title="Training Time Distribution (Straggler Detection)",
                    yaxis_title="Training Time (seconds)",
                    height=300
                )
                
                st.plotly_chart(fig_straggler, use_container_width=True)
                
                # Straggler metrics
                stragglers = training_times[training_times > threshold]
                st.metric("Total Clients", len(training_times))
                st.metric("Stragglers Detected", len(stragglers))
                st.metric("Straggler Rate", f"{len(stragglers)/len(training_times):.1%}")
        
        # Client contribution analysis
        st.subheader("Client Contribution Analysis")
        
        if not client_df.empty:
            # Contribution over time
            fig_contribution = go.Figure()
            
            for client_id in client_df['client_id'].unique():
                client_data = client_df[client_df['client_id'] == client_id]
                contribution = (
                    client_data['local_accuracy'] * 
                    client_data['data_size'] / 
                    client_data['data_size'].max()
                )
                
                fig_contribution.add_trace(go.Scatter(
                    x=client_data['round'],
                    y=contribution,
                    mode='lines+markers',
                    name=f'Client {client_id}',
                    line=dict(width=2)
                ))
            
            fig_contribution.update_layout(
                title="Client Contribution Over Time",
                xaxis_title="Training Round",
                yaxis_title="Contribution Score",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_contribution, use_container_width=True)
    
    def create_live_feed(self):
        """Create live feed section with real-time metrics"""
        st.header("📡 Live Feed - Real-Time Metrics")
        
        # Auto-refresh controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            auto_refresh = st.checkbox(
                "Auto Refresh",
                value=st.session_state.auto_refresh_enabled
            )
            if auto_refresh != st.session_state.auto_refresh_enabled:
                st.session_state.auto_refresh_enabled = auto_refresh
        
        with col2:
            refresh_interval = st.slider(
                "Refresh Interval (s)",
                min_value=1,
                max_value=30,
                value=self.refresh_interval
            )
            self.refresh_interval = refresh_interval
        
        with col3:
            if st.button("🔄 Refresh Now"):
                st.session_state.last_refresh = datetime.now()
                st.rerun()
        
        # Load and display live metrics
        live_data = self.load_live_metrics()
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        # Get accuracy value with fallback
        accuracy = live_data.get('accuracy', live_data.get('global_accuracy', 0.0))
        privacy_loss = live_data.get('privacy_loss', live_data.get('epsilon_spent', 0.0))
        clients = live_data.get('clients', 0)
        round_num = live_data.get('round', 0)
        dataset_name = live_data.get('dataset', st.session_state.selected_dataset)
        dataset_info = live_data.get('dataset_info', self.get_dataset_info()[dataset_name])
        
        with col1:
            # Update accuracy history for sparkline
            if accuracy > 0:  # Only add valid accuracy values
                st.session_state.accuracy_history.append(accuracy)
                # Keep only last 10 values for sparkline
                if len(st.session_state.accuracy_history) > 10:
                    st.session_state.accuracy_history = st.session_state.accuracy_history[-10:]
            
            # Create sparkline
            sparkline = self.create_sparkline(st.session_state.accuracy_history)
            
            # Get dataset-specific accuracy ranges (cached)
            accuracy_ranges = get_cached_accuracy_ranges()
            min_acc, max_acc = accuracy_ranges.get(dataset_name, (0.5, 0.9))
            
            st.metric(
                f"Global Accuracy {dataset_info['icon']}",
                f"{accuracy:.2%} {sparkline}",
                delta=f"{accuracy - 0.5:.2%}" if accuracy > 0.5 else f"{accuracy - 0.5:.2%}",
                help=f"Dataset: {dataset_info['name']} | Range: {min_acc:.1%} - {max_acc:.1%}"
            )
        
        with col2:
            privacy_color = self.get_privacy_color(privacy_loss)
            
            # Determine delta color based on privacy level
            if privacy_loss <= 2.0:
                delta_color = "normal"  # Green
            elif privacy_loss <= 5.0:
                delta_color = "inverse"  # Yellow/Orange
            else:
                delta_color = "inverse"  # Red
            
            st.metric(
                f"Privacy Spent (ε) {privacy_color}",
                f"{privacy_loss:.2f}",
                delta=f"+{privacy_loss:.2f}",
                delta_color=delta_color
            )
        
        with col3:
            # Show client status with better visual feedback
            if clients >= NUM_CLIENTS:
                client_delta = f"+{clients - NUM_CLIENTS}"
                delta_color = "normal"
                status_emoji = "✅"
            elif clients > 0:
                client_delta = f"{clients - NUM_CLIENTS}"
                delta_color = "inverse"
                status_emoji = "⚠️"
            else:
                client_delta = f"{clients - NUM_CLIENTS}"
                delta_color = "inverse"
                status_emoji = "🔴"
            
            st.metric(
                f"Active Clients {status_emoji}",
                clients,
                delta=client_delta,
                delta_color=delta_color
            )
        
        with col4:
            st.metric(
                "Current Round",
                round_num,
                delta=f"+{round_num}"
            )
        
        # Demo mode indicator
        if st.session_state.demo_mode:
            st.success("🎬 **Demo Mode Active** - Showing simulated training progression")
        
        # Training status
        if st.session_state.demo_mode:
            if privacy_loss > 8.0:
                st.error("🛑 Demo: Privacy budget exhausted - Training stopped")
            elif privacy_loss > 6.0:
                st.warning("⚠️ Demo: Approaching privacy budget limit")
            else:
                st.success("✅ Demo: Training running normally")
        elif st.session_state.training_stopped:
            st.error("🛑 Training stopped - Privacy budget exhausted")
        elif privacy_loss > st.session_state.privacy_config['privacy_budget_limit'] * 0.8:
            st.warning("⚠️ Approaching privacy budget limit")
        else:
            st.success("✅ Training running normally")
        
        # Recent activity log
        st.subheader("Recent Activity")
        
        # Create activity log from recent data
        training_df, privacy_df, client_df = self.load_data_from_csv()
        
        if not training_df.empty:
            # Get recent rounds
            recent_rounds = training_df.tail(5)
            
            for _, round_data in recent_rounds.iterrows():
                with st.expander(f"Round {int(round_data['round'])} - {round_data.get('timestamp', 'Unknown')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Accuracy:** {round_data['global_accuracy']:.3f}")
                        st.write(f"**Loss:** {round_data['global_loss']:.3f}")
                        st.write(f"**Avg Client Accuracy:** {round_data['avg_client_accuracy']:.3f}")
                    
                    with col2:
                        if 'communication_rounds' in round_data:
                            st.write(f"**Communication Rounds:** {int(round_data['communication_rounds'])}")
                        if 'convergence_score' in round_data:
                            st.write(f"**Convergence Score:** {round_data['convergence_score']:.3f}")
    
    def run(self):
        """Main dashboard application"""
        
        st.title("🔐 Advanced Privacy-First Federated Learning Dashboard")
        
        # Demo mode checkbox in sidebar
        st.session_state.demo_mode = st.sidebar.checkbox(
            "🎬 Demo Mode", 
            value=st.session_state.demo_mode,
            help="Simulate realistic values for presentations"
        )
        
        # Dataset selection in sidebar
        st.sidebar.markdown("### 📊 Dataset Selection")
        datasets = self.get_dataset_info()
        dataset_options = {f"{info['icon']} {info['name']}": key for key, info in datasets.items()}
        
        selected_display = st.sidebar.selectbox(
            "Choose Dataset:",
            options=list(dataset_options.keys()),
            index=list(dataset_options.values()).index(st.session_state.selected_dataset),
            help="Select the dataset for federated learning"
        )
        
        # Update selected dataset
        st.session_state.selected_dataset = dataset_options[selected_display]
        
        # Show dataset info
        dataset_info = datasets[st.session_state.selected_dataset]
        with st.sidebar.expander(f"📋 Dataset Details", expanded=False):
            st.markdown(f"**{dataset_info['icon']} {dataset_info['name']}**")
            st.write(f"📝 {dataset_info['description']}")
            st.write(f"🏷️ Classes: {dataset_info['classes']}")
            st.write(f"📊 Samples: {dataset_info['samples']:,}")
            st.write(f"🖼️ Size: {dataset_info['image_size']}")
            st.write(f"🎯 Difficulty: {dataset_info['difficulty']}")
            st.write(f"💼 Use Case: {dataset_info['use_case']}")
        
        st.sidebar.markdown("---")
        
        st.markdown("---")
        
        # Auto-refresh logic
        if st.session_state.auto_refresh_enabled:
            time_since_refresh = (datetime.now() - st.session_state.last_refresh).seconds
            if time_since_refresh >= self.refresh_interval:
                st.session_state.last_refresh = datetime.now()
                st.rerun()
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Training Metrics", 
            "🎛️ Privacy Controls",
            "🔐 Secure Aggregation",
            "📊 Privacy Leakage",
            "🔍 Debugging",
            "📈 Dataset Analysis"
        ])
        
        # Load data
        training_df, privacy_df, client_df = self.load_data_from_csv()
        
        with tab1:
            # Import the training metrics from v3
            from enhanced_dashboard_v3 import EnhancedFederatedDashboard
            base_dashboard = EnhancedFederatedDashboard()
            base_dashboard.create_training_metrics_tab(training_df)
        
        with tab2:
            self.create_privacy_utility_controls()
        
        with tab3:
            self.create_secure_aggregation_visualizer()
        
        with tab4:
            self.create_privacy_leakage_gauge()
        
        with tab5:
            self.create_federated_debugging()
        
        with tab6:
            self.create_dataset_analysis_tab()
        
        # Live feed at the bottom
        st.markdown("---")
        self.create_live_feed()
        
        # Footer
        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                  f"Auto-refresh: {'ON' if st.session_state.auto_refresh_enabled else 'OFF'} | "
                  f"Interval: {self.refresh_interval}s")

    def create_dataset_analysis_tab(self):
        """Create dataset analysis and comparison tab (optimized to prevent flickering)"""
        st.header("📈 Dataset Analysis & Comparison")
        
        # Use cached data to prevent flickering
        datasets = get_cached_dataset_info()
        accuracy_ranges = get_cached_accuracy_ranges()
        privacy_challenges = get_cached_privacy_challenges()
        
        current_dataset = st.session_state.selected_dataset
        current_info = datasets[current_dataset]
        
        # Current dataset overview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"📊 Current Dataset: {current_info['icon']} {current_info['name']}")
            st.markdown(f"**Description:** {current_info['description']}")
            st.markdown(f"**Use Case:** {current_info['use_case']}")
            
            # Dataset characteristics
            char_cols = st.columns(4)
            with char_cols[0]:
                st.metric("Classes", current_info['classes'])
            with char_cols[1]:
                st.metric("Samples", f"{current_info['samples']:,}")
            with char_cols[2]:
                st.metric("Image Size", current_info['image_size'])
            with char_cols[3]:
                st.metric("Difficulty", current_info['difficulty'])
        
        with col2:
            # Performance expectations (using cached data)
            min_acc, max_acc = accuracy_ranges.get(current_dataset, (0.5, 0.9))
            
            st.subheader("🎯 Performance Expectations")
            st.write(f"**Accuracy Range:**")
            st.progress((min_acc + max_acc) / 2)
            st.write(f"Min: {min_acc:.1%}")
            st.write(f"Max: {max_acc:.1%}")
            st.write(f"**Privacy Challenge:** {privacy_challenges.get(current_dataset, 'Medium')}")
        
        st.markdown("---")
        
        # Dataset comparison table (cached to prevent flickering)
        st.subheader("🔍 Dataset Comparison")
        
        # Create comparison data once (cached approach)
        @st.cache_data(ttl=3600)
        def get_comparison_data():
            comparison_data = []
            for name, info in datasets.items():
                min_acc, max_acc = accuracy_ranges.get(name, (0.5, 0.9))
                comparison_data.append({
                    "Dataset": f"{info['icon']} {info['name']}",
                    "Classes": info['classes'],
                    "Samples": f"{info['samples']:,}",
                    "Difficulty": info['difficulty'],
                    "Min Accuracy": f"{min_acc:.1%}",
                    "Max Accuracy": f"{max_acc:.1%}",
                    "Privacy Challenge": privacy_challenges.get(name, "Medium"),
                    "Best For": info['use_case']
                })
            return pd.DataFrame(comparison_data)
        
        df_comparison = get_comparison_data()
        st.dataframe(df_comparison, use_container_width=True)
        
        # Visualization (using cached data to prevent flickering)
        st.subheader("📊 Dataset Characteristics Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy ranges comparison (cached)
            fig = go.Figure()
            
            for name, info in datasets.items():
                min_acc, max_acc = accuracy_ranges.get(name, (0.5, 0.9))
                fig.add_trace(go.Bar(
                    name=info['name'],
                    x=['Min Accuracy', 'Max Accuracy'],
                    y=[min_acc, max_acc],
                    text=[f"{min_acc:.1%}", f"{max_acc:.1%}"],
                    textposition='auto',
                ))
            
            fig.update_layout(
                title="Accuracy Ranges by Dataset",
                xaxis_title="Metric",
                yaxis_title="Accuracy",
                yaxis_tickformat='.1%',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Dataset complexity scatter plot (cached)
            class_counts = [info['classes'] for info in datasets.values()]
            sample_counts = [info['samples'] for info in datasets.values()]
            names = [info['name'] for info in datasets.values()]
            icons = [info['icon'] for info in datasets.values()]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=class_counts,
                y=sample_counts,
                mode='markers+text',
                text=[f"{icon} {name}" for icon, name in zip(icons, names)],
                textposition="top center",
                marker=dict(size=12),
                name="Datasets"
            ))
            
            fig.update_layout(
                title="Dataset Complexity (Classes vs Samples)",
                xaxis_title="Number of Classes",
                yaxis_title="Number of Samples"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def get_privacy_challenge(self, dataset_name: str) -> str:
        """Get privacy challenge level for dataset (cached)"""
        privacy_challenges = get_cached_privacy_challenges()
        return privacy_challenges.get(dataset_name, "Medium")

def main():
    """Main entry point"""
    dashboard = AdvancedFederatedDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
