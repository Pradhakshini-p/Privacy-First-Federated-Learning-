#!/usr/bin/env python3
"""
Fixed Enhanced Federated Learning Dashboard
Fixed dashboard graph display issues after data loading
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import threading
from sklearn.datasets import make_classification, make_regression, load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import uuid
import random
import base64
from data_manager import FederatedDataManager

@dataclass
class ClientInfo:
    """Client information and training data"""
    client_id: str
    num_samples: int
    local_accuracy: float
    local_loss: float
    contribution_score: float
    privacy_enabled: bool
    noise_multiplier: float
    training_time: float
    last_round: int
    dataset_distribution: str

@dataclass
class RoundHistory:
    """Training round history"""
    round_num: int
    participating_clients: List[str]
    global_accuracy: float
    global_loss: float
    avg_client_accuracy: float
    total_samples: int
    training_time: float
    timestamp: str
    dataset_name: str

class FederatedClient:
    """Individual federated learning client"""
    
    def __init__(self, client_id: str, X: np.ndarray, y: np.ndarray, 
                 privacy_enabled: bool = False, noise_multiplier: float = 0.1):
        self.client_id = client_id
        self.X = X
        self.y = y
        self.n_samples = len(X)
        self.n_features = X.shape[1]
        self.privacy_enabled = privacy_enabled
        self.noise_multiplier = noise_multiplier
        
        # Initialize local model
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.local_weights = None
        
        # Training history
        self.training_history = []
        
    def local_train(self, global_weights: Optional[np.ndarray] = None, epochs: int = 5) -> Tuple[np.ndarray, float, float]:
        """Train model locally on client data"""
        start_time = time.time()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Get model weights (coefficients and intercept)
        weights = np.concatenate([self.model.coef_.flatten(), [self.model.intercept_[0]]])
        
        # Add differential privacy noise if enabled
        if self.privacy_enabled:
            noise = np.random.normal(0, self.noise_multiplier, weights.shape)
            weights = weights + noise
        
        # Calculate local metrics
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        val_accuracy = accuracy_score(y_val, val_pred)
        
        # Calculate loss (simplified)
        val_loss = mean_squared_error(y_val, val_pred)
        
        training_time = time.time() - start_time
        
        # Store training history
        self.training_history.append({
            'round': len(self.training_history),
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'loss': val_loss,
            'training_time': training_time
        })
        
        self.local_weights = weights
        
        return weights, val_accuracy, val_loss
    
    def evaluate_global_model(self, global_weights: np.ndarray) -> float:
        """Evaluate global model on client's local data"""
        # Create temporary model with global weights
        temp_model = LogisticRegression(max_iter=1000, random_state=42)
        temp_model.fit(self.X[:10], self.y[:10])  # Minimal fitting to initialize
        
        # Set global weights
        temp_model.coef_ = global_weights[:-1].reshape(1, -1)
        temp_model.intercept_ = np.array([global_weights[-1]])
        
        # Evaluate
        predictions = temp_model.predict(self.X)
        accuracy = accuracy_score(self.y, predictions)
        
        return accuracy

class EnhancedFederatedServer:
    """Enhanced federated learning server with data management"""
    
    def __init__(self, n_features: int, dataset_name: str = "synthetic"):
        self.n_features = n_features
        self.dataset_name = dataset_name
        self.global_weights = np.random.randn(n_features + 1) * 0.1
        self.round_history = []
        self.clients = {}
        self.current_round = 0
        self.data_manager = FederatedDataManager()
        
    def register_client(self, client: FederatedClient):
        """Register a new client"""
        self.clients[client.client_id] = client
        
    def federated_averaging(self, client_weights: Dict[str, np.ndarray], 
                           client_samples: Dict[str, int]) -> np.ndarray:
        """Perform FedAvg aggregation"""
        total_samples = sum(client_samples.values())
        
        # Weighted average of client weights
        aggregated_weights = np.zeros_like(self.global_weights)
        
        for client_id, weights in client_weights.items():
            weight_factor = client_samples[client_id] / total_samples
            aggregated_weights += weight_factor * weights
        
        return aggregated_weights
    
    def run_training_round(self, participating_clients: List[str], 
                          min_clients: int = 3) -> RoundHistory:
        """Run one round of federated training"""
        start_time = time.time()
        
        # Check minimum participation
        if len(participating_clients) < min_clients:
            st.warning(f"Insufficient clients: {len(participating_clients)} < {min_clients}")
            return None
        
        # Collect client updates
        client_weights = {}
        client_samples = {}
        client_accuracies = {}
        
        for client_id in participating_clients:
            if client_id in self.clients:
                client = self.clients[client_id]
                
                # Local training
                weights, accuracy, loss = client.local_train(self.global_weights)
                
                client_weights[client_id] = weights
                client_samples[client_id] = client.n_samples
                client_accuracies[client_id] = accuracy
        
        # Federated averaging
        self.global_weights = self.federated_averaging(client_weights, client_samples)
        
        # Evaluate global model on all clients
        global_accuracies = []
        for client_id, client in self.clients.items():
            acc = client.evaluate_global_model(self.global_weights)
            global_accuracies.append(acc)
        
        # Calculate metrics
        avg_global_accuracy = np.mean(global_accuracies)
        avg_client_accuracy = np.mean(list(client_accuracies.values()))
        total_samples = sum(client_samples.values())
        
        # Calculate global loss (simplified)
        global_loss = 1.0 - avg_global_accuracy
        
        training_time = time.time() - start_time
        
        # Create round history
        round_info = RoundHistory(
            round_num=self.current_round,
            participating_clients=participating_clients,
            global_accuracy=avg_global_accuracy,
            global_loss=global_loss,
            avg_client_accuracy=avg_client_accuracy,
            total_samples=total_samples,
            training_time=training_time,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            dataset_name=self.dataset_name
        )
        
        self.round_history.append(round_info)
        self.current_round += 1
        
        return round_info
    
    def get_client_info(self) -> List[ClientInfo]:
        """Get information about all clients"""
        client_info = []
        
        for client_id, client in self.clients.items():
            # Calculate contribution score
            contribution = self._calculate_contribution(client_id)
            
            info = ClientInfo(
                client_id=client_id,
                num_samples=client.n_samples,
                local_accuracy=client.training_history[-1]['val_accuracy'] if client.training_history else 0.0,
                local_loss=client.training_history[-1]['loss'] if client.training_history else 0.0,
                contribution_score=contribution,
                privacy_enabled=client.privacy_enabled,
                noise_multiplier=client.noise_multiplier,
                training_time=sum(h['training_time'] for h in client.training_history),
                last_round=len(client.training_history),
                dataset_distribution=self.dataset_name
            )
            client_info.append(info)
        
        return client_info
    
    def _calculate_contribution(self, client_id: str) -> float:
        """Calculate client contribution score"""
        if client_id not in self.clients:
            return 0.0
        
        client = self.clients[client_id]
        
        # Simple contribution based on data size and recent performance
        data_contribution = client.n_samples / 1000  # Normalize by 1000 samples
        
        performance_contribution = 0.0
        if client.training_history:
            recent_accuracy = client.training_history[-1]['val_accuracy']
            performance_contribution = recent_accuracy
        
        # Privacy bonus
        privacy_bonus = 1.2 if client.privacy_enabled else 1.0
        
        contribution = (data_contribution * 0.4 + performance_contribution * 0.6) * privacy_bonus
        
        return contribution

class FixedFederatedDashboard:
    """Fixed Streamlit dashboard with proper graph display"""
    
    def __init__(self):
        self.server = None
        self.training_active = False
        self.auto_refresh = False
        self.data_manager = FederatedDataManager()
        
        # Initialize session state
        if 'server' not in st.session_state:
            st.session_state.server = None
        if 'training_log' not in st.session_state:
            st.session_state.training_log = []
        if 'system_status' not in st.session_state:
            st.session_state.system_status = "Initialized"
        if 'current_dataset' not in st.session_state:
            st.session_state.current_dataset = None
        if 'client_data' not in st.session_state:
            st.session_state.client_data = {}
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
    
    def add_log(self, level: str, message: str):
        """Add training log entry"""
        log_entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "level": level,
            "message": message
        }
        st.session_state.training_log.append(log_entry)
        
        # Keep only last 100 logs
        if len(st.session_state.training_log) > 100:
            st.session_state.training_log = st.session_state.training_log[-100:]
    
    def render_data_management(self):
        """Render data management section"""
        st.header("📁 Data Management")
        
        # Status indicator
        if st.session_state.data_loaded:
            st.success(f"✅ Dataset loaded: {st.session_state.current_dataset}")
        else:
            st.info("ℹ️ No dataset loaded. Please load a dataset to continue.")
        
        # Create tabs for different data sources
        tab1, tab2, tab3 = st.tabs(["📊 Built-in Datasets", "📁 CSV Upload", "📋 Sample Data"])
        
        with tab1:
            st.subheader("📊 Built-in Datasets")
            
            # List available built-in datasets
            datasets = self.data_manager.list_available_datasets()
            
            selected_dataset = st.selectbox(
                "Select Dataset",
                options=list(datasets.keys()),
                format_func=lambda x: f"{datasets[x]['name']} - {datasets[x]['description']}"
            )
            
            if st.button("📥 Load Built-in Dataset"):
                try:
                    X, y = self.data_manager.load_builtin_dataset(selected_dataset)
                    st.session_state.current_dataset = selected_dataset
                    st.session_state.data_loaded = True
                    
                    # Display dataset info
                    info = self.data_manager.get_dataset_info(selected_dataset)
                    st.success(f"✅ Loaded {datasets[selected_dataset]['name']}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Samples", info['n_samples'])
                    with col2:
                        st.metric("Features", info['n_features'])
                    with col3:
                        st.metric("Classes", info.get('n_classes', 'N/A'))
                    
                    # Show data preview
                    st.subheader("📋 Data Preview")
                    df_preview = pd.DataFrame(X[:5])
                    st.dataframe(df_preview)
                    
                    self.add_log("INFO", f"Loaded built-in dataset: {selected_dataset}")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error loading dataset: {e}")
        
        with tab2:
            st.subheader("📁 Upload CSV Dataset")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload a CSV file with features and a target column"
            )
            
            if uploaded_file is not None:
                try:
                    # Read CSV
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ File uploaded: {uploaded_file.name}")
                    
                    # Show data preview
                    st.subheader("📋 Data Preview")
                    st.dataframe(df.head())
                    
                    # Column selection
                    target_column = st.selectbox(
                        "Select Target Column",
                        options=df.columns.tolist(),
                        help="The column you want to predict"
                    )
                    
                    # Categorical columns
                    categorical_cols = st.multiselect(
                        "Select Categorical Columns (Optional)",
                        options=[col for col in df.columns if df[col].dtype == 'object'],
                        help="Select columns that should be treated as categorical"
                    )
                    
                    if st.button("📥 Process CSV Dataset"):
                        try:
                            # Save uploaded file temporarily
                            temp_path = f"temp_{uploaded_file.name}"
                            df.to_csv(temp_path, index=False)
                            
                            # Load dataset
                            X, y = self.data_manager.load_csv_dataset(temp_path, target_column, categorical_cols)
                            
                            # Store dataset info
                            dataset_name = f"csv_{uploaded_file.name}"
                            st.session_state.current_dataset = dataset_name
                            st.session_state.data_loaded = True
                            
                            # Display dataset info
                            info = self.data_manager.get_dataset_info(dataset_name)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Samples", info['n_samples'])
                            with col2:
                                st.metric("Features", info['n_features'])
                            with col3:
                                st.metric("Type", info['type'])
                            
                            st.success(f"✅ Dataset processed successfully!")
                            self.add_log("INFO", f"Loaded CSV dataset: {uploaded_file.name}")
                            
                            # Clean up temp file
                            os.remove(temp_path)
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error processing dataset: {e}")
                
                except Exception as e:
                    st.error(f"❌ Error reading file: {e}")
        
        with tab3:
            st.subheader("📋 Sample Datasets")
            
            st.write("Use these sample datasets for testing:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🎓 Student Performance"):
                    try:
                        X, y = self.data_manager.load_csv_dataset(
                            "sample_data/student_performance.csv", 
                            "final_grade"
                        )
                        st.session_state.current_dataset = "csv_student_performance.csv"
                        st.session_state.data_loaded = True
                        st.success("✅ Student Performance dataset loaded!")
                        self.add_log("INFO", "Loaded Student Performance dataset")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            
            with col2:
                if st.button("💳 Credit Card Fraud"):
                    try:
                        X, y = self.data_manager.load_csv_dataset(
                            "sample_data/credit_card_fraud.csv", 
                            "is_fraud"
                        )
                        st.session_state.current_dataset = "csv_credit_card_fraud.csv"
                        st.session_state.data_loaded = True
                        st.success("✅ Credit Card Fraud dataset loaded!")
                        self.add_log("INFO", "Loaded Credit Card Fraud dataset")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            
            with col3:
                if st.button("👥 Customer Churn"):
                    try:
                        X, y = self.data_manager.load_csv_dataset(
                            "sample_data/customer_churn.csv", 
                            "churned"
                        )
                        st.session_state.current_dataset = "csv_customer_churn.csv"
                        st.session_state.data_loaded = True
                        st.success("✅ Customer Churn dataset loaded!")
                        self.add_log("INFO", "Loaded Customer Churn dataset")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            
            # Display sample dataset info
            if st.session_state.current_dataset and st.session_state.current_dataset.startswith("csv_"):
                info = self.data_manager.get_dataset_info(st.session_state.current_dataset)
                st.subheader("📊 Current Dataset Info")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Samples", info['n_samples'])
                with col2:
                    st.metric("Features", info['n_features'])
                with col3:
                    st.metric("Type", info['type'])
    
    def initialize_system_with_data(self, n_clients: int, distribution: str, 
                                 privacy_ratio: float):
        """Initialize system with loaded data"""
        if not st.session_state.data_loaded:
            st.error("❌ Please load a dataset first!")
            return None
        
        try:
            # Get loaded data
            if st.session_state.current_dataset in self.data_manager.loaded_data:
                X, y = self.data_manager.loaded_data[st.session_state.current_dataset]
            else:
                st.error("❌ Dataset not loaded properly!")
                return None
            
            # Create server
            server = EnhancedFederatedServer(X.shape[1], st.session_state.current_dataset)
            
            # Distribute data to clients
            client_data = self.data_manager.distribute_data_to_clients(
                st.session_state.current_dataset, n_clients, distribution
            )
            
            # Create clients
            for client_id, (client_X, client_y) in client_data.items():
                privacy_enabled = random.random() < privacy_ratio
                noise_multiplier = random.uniform(0.1, 0.5) if privacy_enabled else 0.0
                
                client = FederatedClient(
                    client_id=client_id,
                    X=client_X,
                    y=client_y,
                    privacy_enabled=privacy_enabled,
                    noise_multiplier=noise_multiplier
                )
                
                server.register_client(client)
            
            st.session_state.server = server
            st.session_state.client_data = client_data
            st.session_state.system_status = "Ready"
            st.session_state.system_initialized = True
            
            self.add_log("INFO", f"System initialized with {n_clients} clients using {distribution} distribution")
            
            # Display client distribution info
            st.subheader("👥 Client Data Distribution")
            distribution_data = []
            for client_id, (client_X, client_y) in client_data.items():
                if len(np.unique(client_y)) <= 10:  # Classification
                    class_dist = np.bincount(client_y)
                    distribution_data.append({
                        'Client': client_id,
                        'Samples': len(client_X),
                        'Class Distribution': str(class_dist)
                    })
                else:  # Regression
                    distribution_data.append({
                        'Client': client_id,
                        'Samples': len(client_X),
                        'Target Range': f"[{client_y.min():.2f}, {client_y.max():.2f}]"
                    })
            
            df_distribution = pd.DataFrame(distribution_data)
            st.dataframe(df_distribution, use_container_width=True)
            
            return server
            
        except Exception as e:
            st.error(f"❌ Error initializing system: {e}")
            return None
    
    def run_training_rounds(self, server: EnhancedFederatedServer, n_rounds: int, 
                           participation_rate: float = 0.8):
        """Run multiple training rounds"""
        self.training_active = True
        st.session_state.system_status = "Training"
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for round_num in range(n_rounds):
            if not self.training_active:
                break
            
            # Select participating clients
            all_clients = list(server.clients.keys())
            num_participating = max(3, int(len(all_clients) * participation_rate))
            participating_clients = random.sample(all_clients, num_participating)
            
            # Run training round
            round_info = server.run_training_round(participating_clients)
            
            if round_info:
                self.add_log("INFO", 
                    f"Round {round_info.round_num}: {len(participating_clients)} clients, "
                    f"Accuracy: {round_info.global_accuracy:.4f}, Loss: {round_info.global_loss:.4f}"
                )
            
            # Update progress
            progress = (round_num + 1) / n_rounds
            progress_bar.progress(progress)
            status_text.text(f"Round {round_num + 1}/{n_rounds} - Accuracy: {round_info.global_accuracy:.4f}")
            
            # Small delay for visualization
            time.sleep(0.5)
        
        self.training_active = False
        st.session_state.system_status = "Completed"
        self.add_log("INFO", f"Training completed after {n_rounds} rounds")
        
        progress_bar.progress(1.0)
        status_text.text("Training completed!")
    
    def render_dashboard(self, server: EnhancedFederatedServer):
        """Render main dashboard with proper graph display"""
        st.header("🎯 System Overview")
        
        # Check if server exists and has training history
        if not server:
            st.warning("⚠️ No server initialized. Please load data and initialize the system first.")
            return
        
        if not server.round_history:
            st.info("ℹ️ No training rounds completed yet. Start training to see dashboard metrics.")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            latest_round = server.round_history[-1]
            st.metric("Global Accuracy", f"{latest_round.global_accuracy:.4f}")
        
        with col2:
            st.metric("Global Loss", f"{latest_round.global_loss:.4f}")
        
        with col3:
            st.metric("Total Rounds", len(server.round_history))
        
        with col4:
            participating_clients = len(latest_round.participating_clients)
            st.metric("Active Clients", participating_clients)
        
        # Training progress charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy over rounds
            rounds = [r.round_num for r in server.round_history]
            accuracies = [r.global_accuracy for r in server.round_history]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rounds, y=accuracies,
                mode='lines+markers',
                name='Global Accuracy',
                line=dict(color='green', width=3)
            ))
            
            fig.update_layout(
                title='📈 Global Accuracy Progress',
                xaxis_title='Round',
                yaxis_title='Accuracy',
                yaxis=dict(range=[0, 1]),
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Loss over rounds
            losses = [r.global_loss for r in server.round_history]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rounds, y=losses,
                mode='lines+markers',
                name='Global Loss',
                line=dict(color='red', width=3)
            ))
            
            fig.update_layout(
                title='📉 Global Loss Progress',
                xaxis_title='Round',
                yaxis_title='Loss',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional analytics
        st.subheader("📊 Training Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Client participation
            participation_data = []
            for round_info in server.round_history:
                participation_data.append({
                    'Round': round_info.round_num,
                    'Participating Clients': len(round_info.participating_clients),
                    'Total Samples': round_info.total_samples
                })
            
            df_participation = pd.DataFrame(participation_data)
            
            fig = px.bar(
                df_participation,
                x='Round',
                y='Participating Clients',
                title='👥 Client Participation per Round',
                color='Participating Clients'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Training time
            training_times = [r.training_time for r in server.round_history]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rounds, y=training_times,
                mode='lines+markers',
                name='Training Time',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title='⏱️ Training Time per Round',
                xaxis_title='Round',
                yaxis_title='Time (seconds)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_settings(self):
        """Render settings section with data management"""
        st.header("⚙️ System Configuration")
        
        # Data Management Section
        st.subheader("📁 Data Configuration")
        self.render_data_management()
        
        st.divider()
        
        # Training Configuration
        st.subheader("🔧 Training Configuration")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load a dataset first before configuring the system.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_clients = st.slider("Number of Clients", 3, 20, 10)
            distribution = st.selectbox(
                "Data Distribution",
                options=["iid", "non_iid", "quantity_skew"],
                format_func=lambda x: {
                    "iid": "IID (Independent)",
                    "non_iid": "Non-IID (Skewed Classes)",
                    "quantity_skew": "Quantity Skewed"
                }[x]
            )
            privacy_ratio = st.slider("Privacy Ratio", 0.0, 1.0, 0.6)
        
        with col2:
            if st.button("🚀 Initialize System", type="primary"):
                server = self.initialize_system_with_data(n_clients, distribution, privacy_ratio)
                if server:
                    st.success(f"✅ System initialized with {n_clients} clients!")
                    st.rerun()
        
        st.divider()
        
        # Training Parameters
        st.subheader("🎯 Training Parameters")
        
        if not st.session_state.system_initialized:
            st.warning("⚠️ Please initialize the system first.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_rounds = st.slider("Training Rounds", 1, 50, 10)
            participation_rate = st.slider("Participation Rate", 0.5, 1.0, 0.8)
        
        with col2:
            if st.session_state.server:
                if st.button("▶️ Start Training", type="primary"):
                    self.run_training_rounds(
                        st.session_state.server, n_rounds, participation_rate
                    )
                    st.success("✅ Training completed!")
                    st.rerun()
                
                if st.button("⏹️ Stop Training"):
                    self.training_active = False
                    st.session_state.system_status = "Stopped"
                    self.add_log("WARNING", "Training stopped by user")
            else:
                st.info("Please initialize the system first")
        
        # System Status
        st.subheader("📊 System Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_color = {
                "Initialized": "🟡",
                "Ready": "🟢", 
                "Training": "🔵",
                "Completed": "✅",
                "Stopped": "🔴"
            }
            
            current_status = st.session_state.system_status
            st.metric("Status", f"{status_color.get(current_status, '⚪')} {current_status}")
        
        with col2:
            if st.session_state.server:
                total_clients = len(st.session_state.server.clients)
                st.metric("Registered Clients", total_clients)
            else:
                st.metric("Registered Clients", 0)
        
        with col3:
            if st.session_state.server and st.session_state.server.round_history:
                completed_rounds = len(st.session_state.server.round_history)
                st.metric("Completed Rounds", completed_rounds)
            else:
                st.metric("Completed Rounds", 0)
        
        # Current Dataset Info
        if st.session_state.current_dataset:
            st.subheader("📊 Current Dataset")
            info = self.data_manager.get_dataset_info(st.session_state.current_dataset)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Dataset", st.session_state.current_dataset)
            with col2:
                st.metric("Samples", info.get('n_samples', 'N/A'))
            with col3:
                st.metric("Features", info.get('n_features', 'N/A'))
    
    def run(self):
        """Main dashboard application"""
        st.set_page_config(
            page_title="Fixed Federated Learning Dashboard",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🤖 Fixed Federated Learning Dashboard")
        st.markdown("---")
        
        # Sidebar navigation
        st.sidebar.title("🧭 Navigation")
        page = st.sidebar.selectbox(
            "Select Page",
            ["📁 Data Management", "🎯 Dashboard", "⚙️ Settings"]
        )
        
        # Main content
        server = st.session_state.server
        
        if page == "📁 Data Management":
            self.render_data_management()
        elif page == "🎯 Dashboard":
            self.render_dashboard(server)
        elif page == "⚙️ Settings":
            self.render_settings()
        
        # Auto-refresh option
        if st.sidebar.checkbox("🔄 Auto Refresh", value=False):
            time.sleep(2)
            st.rerun()

def main():
    """Main application entry point"""
    dashboard = FixedFederatedDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
