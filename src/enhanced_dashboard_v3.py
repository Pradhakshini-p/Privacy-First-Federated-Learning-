#!/usr/bin/env python3
"""
Enhanced Dashboard v3 - CSV-based Real-time Monitoring
Reads from CSV logs created by the logging bridge for real-time federated learning monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import sys
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import threading
from pathlib import Path

# Add src to path to import our modules
sys.path.append(str(Path(__file__).parent))

try:
    from config import DASHBOARD_REFRESH_INTERVAL, NUM_CLIENTS, LOG_DIR
    from logging_bridge import get_logger
except ImportError:
    # Fallback for standalone execution
    LOG_DIR = Path(__file__).parent.parent / "logs"
    DASHBOARD_REFRESH_INTERVAL = 5
    NUM_CLIENTS = 3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedFederatedDashboard:
    """Enhanced dashboard with CSV-based real-time monitoring"""
    
    def __init__(self):
        self.logger = get_logger() if 'get_logger' in globals() else None
        self.last_update = None
        self.auto_refresh = True
        self.refresh_interval = DASHBOARD_REFRESH_INTERVAL
        
        # Initialize session state
        self._init_session_state()
    
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
                # Convert timestamp column to datetime if it exists
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return pd.DataFrame()
    
    def create_training_metrics_tab(self, training_df: pd.DataFrame):
        """Create training metrics visualization tab"""
        st.header("📊 Training Metrics")
        
        if training_df.empty:
            st.warning("No training data available. Start the federated learning platform to see metrics.")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        if not training_df.empty:
            latest = training_df.iloc[-1]
            col1.metric("Current Round", int(latest.get('round', 0)))
            col2.metric("Global Accuracy", f"{latest.get('global_accuracy', 0):.3f}")
            col3.metric("Global Loss", f"{latest.get('global_loss', 0):.3f}")
            col4.metric("Avg Client Accuracy", f"{latest.get('avg_client_accuracy', 0):.3f}")
        
        # Training progress charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy over rounds
            fig_accuracy = go.Figure()
            fig_accuracy.add_trace(go.Scatter(
                x=training_df['round'],
                y=training_df['global_accuracy'],
                mode='lines+markers',
                name='Global Accuracy',
                line=dict(color='#2E86AB', width=3)
            ))
            fig_accuracy.add_trace(go.Scatter(
                x=training_df['round'],
                y=training_df['avg_client_accuracy'],
                mode='lines+markers',
                name='Avg Client Accuracy',
                line=dict(color='#A23B72', width=2, dash='dash')
            ))
            fig_accuracy.update_layout(
                title="Accuracy Progress",
                xaxis_title="Round",
                yaxis_title="Accuracy",
                hovermode='x unified'
            )
            st.plotly_chart(fig_accuracy, use_container_width=True)
        
        with col2:
            # Loss over rounds
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=training_df['round'],
                y=training_df['global_loss'],
                mode='lines+markers',
                name='Global Loss',
                line=dict(color='#F18F01', width=3)
            ))
            fig_loss.update_layout(
                title="Loss Progress",
                xaxis_title="Round",
                yaxis_title="Loss",
                hovermode='x unified'
            )
            st.plotly_chart(fig_loss, use_container_width=True)
        
        # Convergence and communication metrics
        if 'convergence_score' in training_df.columns or 'communication_rounds' in training_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'convergence_score' in training_df.columns:
                    fig_conv = go.Figure()
                    fig_conv.add_trace(go.Scatter(
                        x=training_df['round'],
                        y=training_df['convergence_score'],
                        mode='lines+markers',
                        name='Convergence Score',
                        line=dict(color='#C73E1D', width=3)
                    ))
                    fig_conv.update_layout(
                        title="Convergence Score",
                        xaxis_title="Round",
                        yaxis_title="Score"
                    )
                    st.plotly_chart(fig_conv, use_container_width=True)
            
            with col2:
                if 'communication_rounds' in training_df.columns:
                    fig_comm = go.Figure()
                    fig_comm.add_trace(go.Scatter(
                        x=training_df['round'],
                        y=training_df['communication_rounds'],
                        mode='lines+markers',
                        name='Communication Rounds',
                        line=dict(color='#4CAF50', width=3)
                    ))
                    fig_comm.update_layout(
                        title="Communication Rounds",
                        xaxis_title="Training Round",
                        yaxis_title="Rounds"
                    )
                    st.plotly_chart(fig_comm, use_container_width=True)
    
    def create_privacy_metrics_tab(self, privacy_df: pd.DataFrame):
        """Create privacy metrics visualization tab"""
        st.header("🔒 Privacy Metrics")
        
        if privacy_df.empty:
            st.warning("No privacy data available. Start training with privacy-enabled clients to see metrics.")
            return
        
        # Privacy budget overview
        col1, col2, col3 = st.columns(3)
        
        latest_privacy = privacy_df.iloc[-1]
        col1.metric("Latest Epsilon Spent", f"{latest_privacy.get('epsilon_spent', 0):.3f}")
        col2.metric("Privacy Budget Used", f"{latest_privacy.get('privacy_budget_used', 0):.1%}")
        col3.metric("Noise Multiplier", f"{latest_privacy.get('noise_multiplier', 0):.2f}")
        
        # Privacy spending over time
        col1, col2 = st.columns(2)
        
        with col1:
            # Epsilon spending by round
            fig_epsilon = go.Figure()
            
            # Group by round and get max epsilon for each round
            round_epsilon = privacy_df.groupby('round')['epsilon_spent'].max().reset_index()
            
            fig_epsilon.add_trace(go.Scatter(
                x=round_epsilon['round'],
                y=round_epsilon['epsilon_spent'],
                mode='lines+markers',
                name='Epsilon Spent',
                line=dict(color='#FF6B6B', width=3)
            ))
            fig_epsilon.update_layout(
                title="Privacy Budget Consumption",
                xaxis_title="Round",
                yaxis_title="Epsilon (ε)",
                hovermode='x unified'
            )
            st.plotly_chart(fig_epsilon, use_container_width=True)
        
        with col2:
            # Client-wise privacy spending
            client_privacy = privacy_df.groupby('client_id')['epsilon_spent'].sum().reset_index()
            
            fig_client_privacy = go.Figure()
            fig_client_privacy.add_trace(go.Bar(
                x=client_privacy['client_id'],
                y=client_privacy['epsilon_spent'],
                name='Total Epsilon Spent',
                marker_color='#4ECDC4'
            ))
            fig_client_privacy.update_layout(
                title="Privacy Spending by Client",
                xaxis_title="Client ID",
                yaxis_title="Total Epsilon Spent"
            )
            st.plotly_chart(fig_client_privacy, use_container_width=True)
        
        # Privacy budget utilization table
        st.subheader("Privacy Budget Details")
        
        # Create summary table
        privacy_summary = privacy_df.groupby('client_id').agg({
            'epsilon_spent': ['sum', 'max', 'count'],
            'privacy_budget_used': 'max',
            'noise_multiplier': 'first'
        }).round(4)
        
        privacy_summary.columns = ['Total Epsilon', 'Max Epsilon', 'Rounds', 'Budget Used', 'Noise Mult']
        privacy_summary = privacy_summary.reset_index()
        
        st.dataframe(privacy_summary, use_container_width=True)
    
    def create_client_metrics_tab(self, client_df: pd.DataFrame):
        """Create client-specific metrics visualization tab"""
        st.header("👥 Client Performance")
        
        if client_df.empty:
            st.warning("No client data available. Start the federated learning platform to see client metrics.")
            return
        
        # Client status overview
        active_clients = client_df[client_df['status'] == 'active']['client_id'].nunique()
        total_clients = client_df['client_id'].nunique()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Active Clients", active_clients)
        col2.metric("Total Clients", total_clients)
        col3.metric("Avg Data Size", f"{client_df['data_size'].mean():.0f}")
        col4.metric("Avg Training Time", f"{client_df['training_time'].mean():.2f}s")
        
        # Client performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Client accuracy over rounds
            fig_client_acc = go.Figure()
            
            for client_id in client_df['client_id'].unique():
                client_data = client_df[client_df['client_id'] == client_id]
                fig_client_acc.add_trace(go.Scatter(
                    x=client_data['round'],
                    y=client_data['local_accuracy'],
                    mode='lines+markers',
                    name=f'Client {client_id}',
                    line=dict(width=2)
                ))
            
            fig_client_acc.update_layout(
                title="Client Accuracy Over Rounds",
                xaxis_title="Round",
                yaxis_title="Local Accuracy",
                hovermode='x unified'
            )
            st.plotly_chart(fig_client_acc, use_container_width=True)
        
        with col2:
            # Data distribution
            data_dist = client_df.groupby('client_id')['data_size'].mean().reset_index()
            
            fig_data_dist = go.Figure()
            fig_data_dist.add_trace(go.Pie(
                labels=data_dist['client_id'],
                values=data_dist['data_size'],
                name="Data Distribution"
            ))
            fig_data_dist.update_layout(title="Data Distribution Across Clients")
            st.plotly_chart(fig_data_dist, use_container_width=True)
        
        # Detailed client metrics table
        st.subheader("Detailed Client Metrics")
        
        # Get latest metrics for each client
        latest_client_metrics = client_df.loc[client_df.groupby('client_id')['round'].idxmax()]
        
        # Select columns to display
        display_cols = ['client_id', 'round', 'local_accuracy', 'local_loss', 'data_size', 
                       'training_time', 'communication_cost', 'status']
        
        if all(col in latest_client_metrics.columns for col in display_cols):
            display_df = latest_client_metrics[display_cols].round(4)
            display_df.columns = ['Client', 'Round', 'Accuracy', 'Loss', 'Data Size', 
                                 'Training Time (s)', 'Comm Cost', 'Status']
            st.dataframe(display_df, use_container_width=True)
    
    def create_control_panel(self):
        """Create control panel for dashboard settings"""
        st.header("⚙️ Dashboard Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Auto-refresh toggle
            auto_refresh = st.checkbox("Auto Refresh", value=st.session_state.auto_refresh_enabled)
            if auto_refresh != st.session_state.auto_refresh_enabled:
                st.session_state.auto_refresh_enabled = auto_refresh
        
        with col2:
            # Refresh interval
            refresh_interval = st.slider(
                "Refresh Interval (seconds)",
                min_value=1,
                max_value=30,
                value=self.refresh_interval
            )
            self.refresh_interval = refresh_interval
        
        with col3:
            # Manual refresh button
            if st.button("🔄 Refresh Now"):
                st.session_state.last_refresh = datetime.now()
                st.rerun()
        
        # Data information
        training_df, privacy_df, client_df = self.load_data_from_csv()
        
        st.subheader("📈 Data Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Training Records", len(training_df))
            if not training_df.empty:
                latest_round = training_df['round'].max()
                st.metric("Latest Round", int(latest_round))
        
        with col2:
            st.metric("Privacy Records", len(privacy_df))
            if not privacy_df.empty:
                unique_clients = privacy_df['client_id'].nunique()
                st.metric("Clients with Privacy", unique_clients)
        
        with col3:
            st.metric("Client Records", len(client_df))
            if not client_df.empty:
                active_clients = client_df[client_df['status'] == 'active']['client_id'].nunique()
                st.metric("Active Clients", active_clients)
    
    def run(self):
        """Main dashboard application"""
        st.set_page_config(
            page_title="Federated Learning Dashboard",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🔐 Privacy-First Federated Learning Dashboard")
        st.markdown("---")
        
        # Auto-refresh logic
        if st.session_state.auto_refresh_enabled:
            time_since_refresh = (datetime.now() - st.session_state.last_refresh).seconds
            if time_since_refresh >= self.refresh_interval:
                st.session_state.last_refresh = datetime.now()
                st.rerun()
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Training Metrics", 
            "🔒 Privacy Metrics", 
            "👥 Client Performance",
            "⚙️ Controls"
        ])
        
        # Load data
        training_df, privacy_df, client_df = self.load_data_from_csv()
        
        with tab1:
            self.create_training_metrics_tab(training_df)
        
        with tab2:
            self.create_privacy_metrics_tab(privacy_df)
        
        with tab3:
            self.create_client_metrics_tab(client_df)
        
        with tab4:
            self.create_control_panel()
        
        # Footer with last update time
        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                  f"Auto-refresh: {'ON' if st.session_state.auto_refresh_enabled else 'OFF'} | "
                  f"Interval: {self.refresh_interval}s")

def main():
    """Main entry point"""
    dashboard = EnhancedFederatedDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
