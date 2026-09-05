#!/usr/bin/env python3
"""
Enhanced Dashboard with Client Analytics and Real-time Monitoring
Includes client contribution analytics, fault tolerance, and real-time logs
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
import os
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedFederatedDashboard:
    def __init__(self):
        self.log_queue = queue.Queue()
        self.client_contributions = {}
        self.fault_tolerance_stats = {
            "total_clients": 0,
            "active_clients": 0,
            "dropped_clients": 0,
            "recovered_clients": 0
        }
        self.attack_detection = {
            "malicious_detected": 0,
            "anomaly_score": 0.0,
            "security_level": "HIGH"
        }
        
        # Initialize session state
        if 'logs' not in st.session_state:
            st.session_state.logs = []
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
        if 'refresh_interval' not in st.session_state:
            st.session_state.refresh_interval = 5
    
    def load_training_data(self):
        """Load training data from logs."""
        try:
            # Load training log
            if os.path.exists("logs/training_log.json"):
                with open("logs/training_log.json", "r") as f:
                    training_data = json.load(f)
            else:
                training_data = []
            
            # Load client metrics
            if os.path.exists("logs/client_metrics.json"):
                with open("logs/client_metrics.json", "r") as f:
                    client_data = json.load(f)
            else:
                client_data = []
            
            # Load secure aggregation log
            if os.path.exists("logs/secure_aggregation_log.json"):
                with open("logs/secure_aggregation_log.json", "r") as f:
                    security_data = json.load(f)
            else:
                security_data = []
            
            # Load fault tolerance log
            if os.path.exists("logs/fault_tolerance_log.json"):
                with open("logs/fault_tolerance_log.json", "r") as f:
                    fault_data = json.load(f)
            else:
                fault_data = {}
            
            return training_data, client_data, security_data, fault_data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return [], [], [], {}
    
    def calculate_client_contributions(self, client_data):
        """Calculate client contribution scores."""
        contributions = {}
        
        for round_data in client_data:
            metrics = round_data.get("metrics", {})
            client_id = metrics.get("client_id", "unknown")
            num_samples = round_data.get("num_samples", 0)
            accuracy = metrics.get("val_accuracy", 0)
            
            if client_id not in contributions:
                contributions[client_id] = {
                    "total_samples": 0,
                    "avg_accuracy": 0,
                    "rounds_participated": 0,
                    "contribution_score": 0,
                    "privacy_enabled": False,
                    "avg_epsilon": 0.0
                }
            
            contributions[client_id]["total_samples"] += num_samples
            contributions[client_id]["avg_accuracy"] += accuracy
            contributions[client_id]["rounds_participated"] += 1
            
            # Track privacy metrics
            if metrics.get("privacy_enabled", False):
                contributions[client_id]["privacy_enabled"] = True
                contributions[client_id]["avg_epsilon"] += metrics.get("epsilon", 0)
        
        # Calculate contribution scores
        for client_id, data in contributions.items():
            if data["rounds_participated"] > 0:
                data["avg_accuracy"] /= data["rounds_participated"]
                if data["privacy_enabled"]:
                    data["avg_epsilon"] /= data["rounds_participated"]
                
                # Enhanced contribution score calculation
                privacy_bonus = 1.2 if data["privacy_enabled"] else 1.0
                data["contribution_score"] = (
                    data["total_samples"] * 0.2 + 
                    data["avg_accuracy"] * data["rounds_participated"] * 0.5 +
                    privacy_bonus * 0.3
                )
        
        return contributions
    
    def detect_anomalies(self, client_data):
        """Detect potential malicious clients or anomalies."""
        anomalies = []
        
        # Simple anomaly detection based on unusual patterns
        for round_data in client_data:
            metrics = round_data.get("metrics", {})
            client_id = metrics.get("client_id", "unknown")
            
            # Check for unusually low accuracy
            accuracy = metrics.get("val_accuracy", 0)
            if accuracy < 0.3:  # Threshold for suspiciously low accuracy
                anomalies.append({
                    "client_id": client_id,
                    "type": "LOW_ACCURACY",
                    "value": accuracy,
                    "timestamp": round_data.get("timestamp", datetime.now().isoformat())
                })
            
            # Check for privacy budget exhaustion
            epsilon = metrics.get("epsilon", 0)
            if epsilon > 10:  # High epsilon might indicate privacy issues
                anomalies.append({
                    "client_id": client_id,
                    "type": "PRIVACY_BUDGET_HIGH",
                    "value": epsilon,
                    "timestamp": round_data.get("timestamp", datetime.now().isoformat())
                })
        
        return anomalies
    
    def add_log_entry(self, level, message, source="System"):
        """Add a log entry to the real-time log."""
        log_entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "level": level,
            "source": source,
            "message": message
        }
        
        st.session_state.logs.append(log_entry)
        
        # Keep only last 100 logs
        if len(st.session_state.logs) > 100:
            st.session_state.logs = st.session_state.logs[-100:]
    
    def render_header(self):
        """Render dashboard header with system status."""
        st.set_page_config(
            page_title="Enhanced Federated Learning Dashboard",
            page_icon="🔐",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🔐 Enhanced Federated Learning Dashboard")
        st.markdown("---")
        
        # System Status
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_color = "🟢" if self.fault_tolerance_stats["active_clients"] > 0 else "🔴"
            st.metric("System Status", f"{status_color} Running")
        
        with col2:
            st.metric("Active Clients", self.fault_tolerance_stats["active_clients"])
        
        with col3:
            security_level = self.attack_detection["security_level"]
            st.metric("Security Level", security_level)
        
        with col4:
            anomalies = len(self.detect_anomalies(self.load_training_data()[1]))
            st.metric("Anomalies Detected", anomalies)
    
    def render_client_analytics(self, client_data):
        """Render client contribution analytics."""
        st.subheader("📊 Client Contribution Analytics")
        
        contributions = self.calculate_client_contributions(client_data)
        
        if not contributions:
            st.info("No client data available yet. Start training to see analytics.")
            return
        
        # Create DataFrame for visualization
        df_contributions = pd.DataFrame.from_dict(contributions, orient='index')
        df_contributions = df_contributions.reset_index().rename(columns={'index': 'client_id'})
        
        # Contribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Contribution Score Chart
            fig = px.bar(
                df_contributions.sort_values('contribution_score', ascending=True),
                x='contribution_score',
                y='client_id',
                title='Client Contribution Scores',
                orientation='h',
                color='contribution_score',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sample Distribution
            fig = px.pie(
                df_contributions,
                values='total_samples',
                names='client_id',
                title='Training Sample Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Privacy vs Performance Analysis
        st.subheader("🛡️ Privacy vs Performance Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Privacy-enabled clients
            privacy_counts = df_contributions['privacy_enabled'].value_counts()
            fig = px.pie(
                values=privacy_counts.values,
                names=['Privacy Enabled', 'No Privacy'],
                title='Privacy Protection Adoption'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Privacy budget usage
            privacy_clients = df_contributions[df_contributions['privacy_enabled'] == True]
            if not privacy_clients.empty:
                fig = px.bar(
                    privacy_clients,
                    x='client_id',
                    y='avg_epsilon',
                    title='Average Privacy Budget (ε) Usage',
                    color='avg_epsilon',
                    color_continuous_scale='reds'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No privacy-enabled clients in this round")
        
        # Detailed client table
        st.subheader("📋 Client Performance Details")
        df_display = df_contributions[[
            'client_id', 'total_samples', 'avg_accuracy', 'rounds_participated', 
            'contribution_score', 'privacy_enabled', 'avg_epsilon'
        ]]
        df_display = df_display.sort_values('contribution_score', ascending=False)
        
        # Format for display
        df_display['privacy_enabled'] = df_display['privacy_enabled'].map({True: '✅ Yes', False: '❌ No'})
        df_display['avg_epsilon'] = df_display['avg_epsilon'].round(3)
        df_display['contribution_score'] = df_display['contribution_score'].round(2)
        
        st.dataframe(df_display, use_container_width=True)
    
    def render_fault_tolerance(self, security_data, fault_data):
        """Render fault tolerance statistics."""
        st.subheader("🛡️ Fault Tolerance & Reliability")
        
        # Update stats from fault data
        if fault_data and 'client_stats' in fault_data:
            client_stats = fault_data['client_stats']
            self.fault_tolerance_stats.update({
                "total_clients": client_stats.get("registered_clients", 0),
                "active_clients": client_stats.get("active_clients", 0),
                "dropped_clients": client_stats.get("inactive_clients", 0),
                "recovered_clients": client_stats.get("total_successes", 0)
            })
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            success_rate = 0
            if self.fault_tolerance_stats["total_clients"] > 0:
                success_rate = (self.fault_tolerance_stats["active_clients"] / 
                              self.fault_tolerance_stats["total_clients"]) * 100
            
            st.metric(
                "Success Rate",
                f"{success_rate:.1f}%",
                delta=f"+{success_rate - 90:.1f}%" if success_rate > 90 else f"{success_rate - 90:.1f}%"
            )
        
        with col2:
            st.metric("Dropped Clients", self.fault_tolerance_stats["dropped_clients"])
        
        with col3:
            st.metric("Recovered Clients", self.fault_tolerance_stats["recovered_clients"])
        
        # Fault tolerance timeline
        if security_data:
            df_security = pd.DataFrame(security_data)
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Client Participation Over Time", "Success Rate Trend"),
                vertical_spacing=0.1
            )
            
            # Client participation
            fig.add_trace(
                go.Scatter(
                    x=df_security['round'],
                    y=df_security['successful_clients'],
                    name='Successful Clients',
                    line=dict(color='green')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df_security['round'],
                    y=df_security['failed_clients'],
                    name='Failed Clients',
                    line=dict(color='red')
                ),
                row=1, col=1
            )
            
            # Success rate
            if 'total_clients' in df_security.columns and df_security['total_clients'].sum() > 0:
                success_rates = (df_security['successful_clients'] / df_security['total_clients']) * 100
                fig.add_trace(
                    go.Scatter(
                        x=df_security['round'],
                        y=success_rates,
                        name='Success Rate (%)',
                        line=dict(color='blue')
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        # Client health details
        if fault_data and 'client_health' in fault_data:
            st.subheader("🏥 Client Health Details")
            client_health = fault_data['client_health']
            
            health_data = []
            for client_id, health in client_health.items():
                if health['last_seen']:
                    last_seen = datetime.fromisoformat(health['last_seen'].replace('Z', '+00:00')) if isinstance(health['last_seen'], str) else health['last_seen']
                    time_since = datetime.now() - last_seen.replace(tzinfo=None) if hasattr(last_seen, 'tzinfo') else datetime.now() - last_seen
                    status = "🟢 Active" if time_since.total_seconds() < 60 else "🔴 Inactive"
                else:
                    status = "❓ Unknown"
                    time_since = timedelta(0)
                
                health_data.append({
                    "Client ID": client_id,
                    "Status": status,
                    "Successes": health['successes'],
                    "Failures": health['failures'],
                    "Last Seen": f"{time_since.total_seconds():.0f}s ago" if time_since.total_seconds() < 3600 else "Unknown"
                })
            
            df_health = pd.DataFrame(health_data)
            st.dataframe(df_health, use_container_width=True)
    
    def render_realtime_logs(self):
        """Render real-time logging panel."""
        st.subheader("📝 Real-time System Logs")
        
        # Log controls
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🔄 Refresh Logs"):
                self.add_log_entry("INFO", "Manual refresh triggered", "Dashboard")
        
        with col2:
            if st.button("🗑️ Clear Logs"):
                st.session_state.logs = []
                self.add_log_entry("INFO", "Logs cleared", "Dashboard")
        
        with col3:
            auto_refresh = st.checkbox("🔄 Auto Refresh", value=st.session_state.auto_refresh)
            st.session_state.auto_refresh = auto_refresh
        
        # Display logs
        if st.session_state.logs:
            # Create log DataFrame
            df_logs = pd.DataFrame(st.session_state.logs)
            
            # Color coding by level
            def color_log_level(val):
                if val == "ERROR":
                    return "background-color: #ffebee"
                elif val == "WARNING":
                    return "background-color: #fff3e0"
                elif val == "INFO":
                    return "background-color: #e8f5e8"
                else:
                    return ""
            
            styled_df = df_logs.style.applymap(color_log_level, subset=['level'])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("No logs available. System events will appear here.")
        
        # Auto refresh
        if st.session_state.auto_refresh:
            time.sleep(st.session_state.refresh_interval)
            st.rerun()
    
    def render_attack_detection(self, client_data):
        """Render attack detection and security monitoring."""
        st.subheader("🚨 Attack Detection & Security")
        
        anomalies = self.detect_anomalies(client_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Security metrics
            st.metric("Malicious Clients Detected", len(anomalies))
            security_score = max(0, 100 - len(anomalies) * 10)
            st.metric("Security Score", security_score)
            
            # Anomaly types
            if anomalies:
                anomaly_types = {}
                for anomaly in anomalies:
                    anomaly_type = anomaly["type"]
                    anomaly_types[anomaly_type] = anomaly_types.get(anomaly_type, 0) + 1
                
                fig = px.pie(
                    values=list(anomaly_types.values()),
                    names=list(anomaly_types.keys()),
                    title="Anomaly Types"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Recent anomalies
            if anomalies:
                st.write("**Recent Anomalies:**")
                for anomaly in anomalies[-5:]:  # Show last 5
                    st.warning(f"⚠️ Client {anomaly['client_id']}: {anomaly['type']} = {anomaly['value']:.3f}")
            else:
                st.success("✅ No anomalies detected")
        
        # Security recommendations
        st.subheader("🔒 Security Recommendations")
        
        if len(anomalies) > 0:
            st.error("⚠️ Security issues detected. Consider the following actions:")
            st.markdown("""
            - Review client training procedures
            - Implement stricter validation
            - Consider removing suspicious clients
            - Increase privacy budget monitoring
            """)
        else:
            st.success("✅ System security looks good!")
            st.markdown("""
            - Continue monitoring client behavior
            - Regular security audits recommended
            - Keep privacy budgets in check
            """)
    
    def render_training_progress(self, training_data):
        """Render enhanced training progress charts."""
        st.subheader("📈 Training Progress & Performance")
        
        if not training_data:
            st.info("No training data available yet.")
            return
        
        df_training = pd.DataFrame(training_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy chart with confidence intervals
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_training['round'],
                y=df_training['accuracy'],
                mode='lines+markers',
                name='Accuracy',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ))
            
            # Add trend line
            z = np.polyfit(df_training['round'], df_training['accuracy'], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=df_training['round'],
                y=p(df_training['round']),
                mode='lines',
                name='Trend',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title='Model Accuracy Progress',
                xaxis_title='Training Round',
                yaxis_title='Accuracy',
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Loss chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_training['round'],
                y=df_training['loss'],
                mode='lines+markers',
                name='Loss',
                line=dict(color='orange', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title='Model Loss Progress',
                xaxis_title='Training Round',
                yaxis_title='Loss'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Run the enhanced dashboard."""
        # Load data
        training_data, client_data, security_data, fault_data = self.load_training_data()
        
        # Add initial log
        if not st.session_state.logs:
            self.add_log_entry("INFO", "Enhanced Dashboard initialized", "System")
        
        # Render components
        self.render_header()
        
        # Tab navigation
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Training Progress", 
            "👥 Client Analytics", 
            "🛡️ Fault Tolerance",
            "🚨 Security Monitor",
            "📝 System Logs"
        ])
        
        with tab1:
            self.render_training_progress(training_data)
        
        with tab2:
            self.render_client_analytics(client_data)
        
        with tab3:
            self.render_fault_tolerance(security_data, fault_data)
        
        with tab4:
            self.render_attack_detection(client_data)
        
        with tab5:
            self.render_realtime_logs()
        
        # Auto-refresh logic
        if st.session_state.auto_refresh:
            time.sleep(2)
            st.rerun()

def main():
    """Main function to run the enhanced dashboard."""
    dashboard = EnhancedFederatedDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
