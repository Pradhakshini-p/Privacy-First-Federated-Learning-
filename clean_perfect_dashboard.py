#!/usr/bin/env python3
"""
Clean Perfect Federated Learning Dashboard
Perfect UI with no visibility issues
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
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import uuid
import random
import base64
import seaborn as sns
import matplotlib.pyplot as plt
from data_manager import FederatedDataManager

# Set matplotlib to avoid display issues
import matplotlib
matplotlib.use('Agg')

# Set page configuration - MUST be called first
st.set_page_config(
    page_title="Clean Perfect Federated Learning Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean CSS with perfect visibility
st.markdown("""
<style>
    /* Main styling */
    .main {
        background-color: #0f172a;
        color: white;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Cards */
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
    
    .card strong {
        color: #60a5fa;
        font-weight: bold;
    }
    
    /* Section headers */
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
    
    /* Status indicators */
    .status-success {
        border-left: 4px solid #10b981;
        background-color: #064e3b;
    }
    
    .status-warning {
        border-left: 4px solid #f59e0b;
        background-color: #451a03;
    }
    
    .status-error {
        border-left: 4px solid #ef4444;
        background-color: #7f1d1d;
    }
    
    /* Buttons */
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
    
    /* Select boxes and inputs */
    .stSelectbox > div > div, .stSlider > div > div {
        background-color: #1e293b;
        color: white;
        border: 1px solid #3b82f6;
    }
    
    /* Dataframes */
    .stDataFrame {
        background-color: #1e293b;
        color: white;
        border: 1px solid #3b82f6;
    }
    
    /* Metrics */
    .stMetric {
        background-color: #1e293b;
        color: white;
        border: 1px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* Charts */
    .chart-container {
        background-color: #1e293b;
        border: 1px solid #3b82f6;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    /* Text elements */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: white;
    }
    
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: white;
    }
    
    /* Success/error messages */
    .stSuccess {
        background-color: #064e3b;
        color: white;
        border: 1px solid #10b981;
    }
    
    .stError {
        background-color: #7f1d1d;
        color: white;
        border: 1px solid #ef4444;
    }
    
    .stWarning {
        background-color: #451a03;
        color: white;
        border: 1px solid #f59e0b;
    }
    
    .stInfo {
        background-color: #1e3a8a;
        color: white;
        border: 1px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

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
    model_type: str
    communication_cost: float

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
    communication_cost: float
    model_convergence: float

class CleanFederatedClient:
    """Clean federated learning client"""
    
    def __init__(self, client_id: str, X: np.ndarray, y: np.ndarray, 
                 privacy_enabled: bool = False, noise_multiplier: float = 0.1,
                 model_type: str = "logistic_regression"):
        self.client_id = client_id
        self.X = X
        self.y = y
        self.n_samples = len(X)
        self.n_features = X.shape[1]
        self.privacy_enabled = privacy_enabled
        self.noise_multiplier = noise_multiplier
        self.model_type = model_type
        
        # Initialize model based on type
        if model_type == "logistic_regression":
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        
        self.local_weights = None
        self.training_history = []
        self.communication_cost = 0.0
        
    def local_train(self, global_weights: Optional[np.ndarray] = None, epochs: int = 5) -> Tuple[np.ndarray, float, float]:
        """Train model locally on client data"""
        start_time = time.time()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Get model weights
        if hasattr(self.model, 'coef_'):
            weights = np.concatenate([self.model.coef_.flatten(), [self.model.intercept_[0]]])
        else:
            # For tree-based models, use feature importances as proxy
            if hasattr(self.model, 'feature_importances_'):
                weights = np.concatenate([self.model.feature_importances_, [0.5]])
            else:
                weights = np.random.randn(self.n_features + 1) * 0.1
        
        # Add differential privacy noise if enabled
        if self.privacy_enabled:
            noise = np.random.normal(0, self.noise_multiplier, weights.shape)
            weights = weights + noise
        
        # Calculate local metrics
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        val_accuracy = accuracy_score(y_val, val_pred)
        
        # Calculate loss
        val_loss = mean_squared_error(y_val, val_pred)
        
        training_time = time.time() - start_time
        
        # Calculate communication cost
        model_size = len(weights) * 4  # 4 bytes per float
        self.communication_cost += model_size
        
        # Store training history
        self.training_history.append({
            'round': len(self.training_history),
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'loss': val_loss,
            'training_time': training_time,
            'communication_cost': self.communication_cost
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

class CleanFederatedServer:
    """Clean federated learning server"""
    
    def __init__(self, n_features: int, dataset_name: str = "synthetic"):
        self.n_features = n_features
        self.dataset_name = dataset_name
        self.global_weights = np.random.randn(n_features + 1) * 0.1
        self.round_history = []
        self.clients = {}
        self.current_round = 0
        self.data_manager = FederatedDataManager()
        
    def register_client(self, client: CleanFederatedClient):
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
        
        # Calculate global loss
        global_loss = 1.0 - avg_global_accuracy
        
        # Calculate model convergence
        convergence = 0.0
        if len(self.round_history) > 0:
            prev_accuracy = self.round_history[-1].global_accuracy
            convergence = avg_global_accuracy - prev_accuracy
        
        # Calculate communication cost
        total_communication = sum(client.communication_cost for client in self.clients.values())
        
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
            dataset_name=self.dataset_name,
            communication_cost=total_communication,
            model_convergence=convergence
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
                dataset_distribution=self.dataset_name,
                model_type=client.model_type,
                communication_cost=client.communication_cost
            )
            client_info.append(info)
        
        return client_info
    
    def _calculate_contribution(self, client_id: str) -> float:
        """Calculate client contribution score"""
        if client_id not in self.clients:
            return 0.0
        
        client = self.clients[client_id]
        
        # Data contribution
        data_contribution = client.n_samples / 1000
        
        # Performance contribution
        performance_contribution = 0.0
        if client.training_history:
            recent_accuracy = client.training_history[-1]['val_accuracy']
            performance_contribution = recent_accuracy
        
        # Privacy bonus
        privacy_bonus = 1.2 if client.privacy_enabled else 1.0
        
        # Model type bonus
        model_bonus = 1.1 if client.model_type == "random_forest" else 1.0
        
        contribution = (data_contribution * 0.3 + performance_contribution * 0.5) * privacy_bonus * model_bonus
        
        return contribution

class CleanPerfectDashboard:
    """Clean perfect federated learning dashboard"""
    
    def __init__(self):
        self.server = None
        self.training_active = False
        self.data_manager = FederatedDataManager()
        
        # Initialize session state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize all session state variables"""
        defaults = {
            'server': None,
            'training_log': [],
            'system_status': "Initialized",
            'current_dataset': None,
            'client_data': {},
            'data_loaded': False,
            'system_initialized': False,
            'training_completed': False,
            'dataset_info': {},
            'client_distribution_info': {},
            'training_progress': 0,
            'current_round': 0,
            'total_rounds': 0,
            'loaded_X': None,
            'loaded_y': None,
            'data_loading_error': None
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
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
    
    def _validate_data_loaded(self):
        """Validate that data is properly loaded"""
        if not st.session_state.data_loaded:
            return False, "No dataset loaded"
        
        if not st.session_state.current_dataset:
            return False, "No dataset name specified"
        
        if st.session_state.loaded_X is None or st.session_state.loaded_y is None:
            return False, "Data arrays are None"
        
        if len(st.session_state.loaded_X) == 0 or len(st.session_state.loaded_y) == 0:
            return False, "Empty data arrays"
        
        return True, "Data properly loaded"
    
    def render_clean_header(self):
        """Render clean header"""
        st.markdown('<h1 class="main-header">🤖 Clean Perfect Federated Learning</h1>', unsafe_allow_html=True)
        st.markdown("---")
    
    def render_clean_status_overview(self):
        """Render clean status overview"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_class = "status-success" if st.session_state.data_loaded else "status-error"
            st.markdown(f'''
            <div class="card {status_class}">
                <h3>📊 Data Status</h3>
                <p><strong>{"✅ Loaded" if st.session_state.data_loaded else "❌ Not Loaded"}</strong></p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            system_class = "status-success" if st.session_state.system_initialized else "status-warning"
            st.markdown(f'''
            <div class="card {system_class}">
                <h3>🤖 System</h3>
                <p><strong>{"✅ Ready" if st.session_state.system_initialized else "⚠️ Not Ready"}</strong></p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            training_class = "status-success" if st.session_state.training_completed else "status-warning"
            st.markdown(f'''
            <div class="card {training_class}">
                <h3>🎯 Training</h3>
                <p><strong>{"✅ Completed" if st.session_state.training_completed else "⚠️ Not Started"}</strong></p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            status_colors = {
                "Initialized": "🟡",
                "Ready": "🟢", 
                "Training": "🔵",
                "Completed": "✅",
                "Stopped": "🔴"
            }
            current_status = st.session_state.system_status
            st.markdown(f'''
            <div class="card">
                <h3>📈 Status</h3>
                <p><strong>{status_colors.get(current_status, '⚪')} {current_status}</strong></p>
            </div>
            ''', unsafe_allow_html=True)
    
    def render_clean_data_management(self):
        """Render clean data management section"""
        st.markdown('<div class="section-header">📁 Data Management</div>', unsafe_allow_html=True)
        
        # Data loading tabs
        tab1, tab2, tab3 = st.tabs(["📊 Built-in Datasets", "📁 Upload CSV", "📋 Sample Datasets"])
        
        with tab1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📊 Built-in Datasets")
            
            datasets = self.data_manager.list_available_datasets()
            
            selected_dataset = st.selectbox(
                "Select Dataset",
                options=list(datasets.keys()),
                format_func=lambda x: f"🎯 {datasets[x]['name']} - {datasets[x]['description']}"
            )
            
            if st.button("📥 Load Dataset", type="primary", key="load_builtin"):
                try:
                    st.session_state.data_loading_error = None
                    
                    X, y = self.data_manager.load_builtin_dataset(selected_dataset)
                    
                    st.session_state.loaded_X = X
                    st.session_state.loaded_y = y
                    st.session_state.current_dataset = selected_dataset
                    st.session_state.data_loaded = True
                    st.session_state.dataset_info = self.data_manager.get_dataset_info(selected_dataset)
                    
                    info = st.session_state.dataset_info
                    st.success(f"✅ Successfully loaded {datasets[selected_dataset]['name']}")
                    
                    # Display dataset info
                    st.markdown(f"### 📋 Dataset Information")
                    st.markdown(f"- **Samples**: {info['n_samples']}")
                    st.markdown(f"- **Features**: {info['n_features']}")
                    st.markdown(f"- **Type**: {info.get('type', 'N/A')}")
                    
                    # Data preview
                    st.markdown("### 📊 Data Preview")
                    df_preview = pd.DataFrame(X[:5])
                    st.dataframe(df_preview, use_container_width=True)
                    
                    self.add_log("INFO", f"Loaded built-in dataset: {selected_dataset}")
                    st.rerun()
                    
                except Exception as e:
                    st.session_state.data_loading_error = str(e)
                    st.error(f"❌ Error loading dataset: {e}")
                    self.add_log("ERROR", f"Failed to load dataset: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📁 Upload CSV Dataset")
            
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload a CSV file with features and a target column"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ File uploaded: {uploaded_file.name}")
                    
                    # Data preview
                    st.markdown("### 📋 Data Preview")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Column selection
                    target_column = st.selectbox(
                        "🎯 Select Target Column",
                        options=df.columns.tolist()
                    )
                    
                    if st.button("📥 Process Dataset", type="primary", key="process_csv"):
                        try:
                            st.session_state.data_loading_error = None
                            
                            temp_path = f"temp_{uploaded_file.name}"
                            df.to_csv(temp_path, index=False)
                            
                            X, y = self.data_manager.load_csv_dataset(temp_path, target_column, [])
                            
                            dataset_name = f"csv_{uploaded_file.name}"
                            st.session_state.loaded_X = X
                            st.session_state.loaded_y = y
                            st.session_state.current_dataset = dataset_name
                            st.session_state.data_loaded = True
                            st.session_state.dataset_info = self.data_manager.get_dataset_info(dataset_name)
                            
                            st.success(f"✅ Dataset processed successfully!")
                            self.add_log("INFO", f"Loaded CSV dataset: {uploaded_file.name}")
                            
                            os.remove(temp_path)
                            st.rerun()
                            
                        except Exception as e:
                            st.session_state.data_loading_error = str(e)
                            st.error(f"❌ Error processing dataset: {e}")
                            self.add_log("ERROR", f"Failed to process dataset: {e}")
                
                except Exception as e:
                    st.session_state.data_loading_error = str(e)
                    st.error(f"❌ Error reading file: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📋 Sample Datasets")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🎓 Student Performance", type="primary", key="student"):
                    try:
                        st.session_state.data_loading_error = None
                        
                        X, y = self.data_manager.load_csv_dataset(
                            "sample_data/student_performance.csv", 
                            "final_grade"
                        )
                        
                        st.session_state.loaded_X = X
                        st.session_state.loaded_y = y
                        st.session_state.current_dataset = "csv_student_performance.csv"
                        st.session_state.data_loaded = True
                        st.session_state.dataset_info = self.data_manager.get_dataset_info("csv_student_performance.csv")
                        
                        st.success("✅ Student Performance dataset loaded!")
                        self.add_log("INFO", "Loaded Student Performance dataset")
                        st.rerun()
                    except Exception as e:
                        st.session_state.data_loading_error = str(e)
                        st.error(f"❌ Error: {e}")
                        self.add_log("ERROR", f"Failed to load dataset: {e}")
            
            with col2:
                if st.button("💳 Credit Fraud", type="primary", key="fraud"):
                    try:
                        st.session_state.data_loading_error = None
                        
                        X, y = self.data_manager.load_csv_dataset(
                            "sample_data/credit_card_fraud.csv", 
                            "is_fraud"
                        )
                        
                        st.session_state.loaded_X = X
                        st.session_state.loaded_y = y
                        st.session_state.current_dataset = "csv_credit_card_fraud.csv"
                        st.session_state.data_loaded = True
                        st.session_state.dataset_info = self.data_manager.get_dataset_info("csv_credit_card_fraud.csv")
                        
                        st.success("✅ Credit Card Fraud dataset loaded!")
                        self.add_log("INFO", "Loaded Credit Card Fraud dataset")
                        st.rerun()
                    except Exception as e:
                        st.session_state.data_loading_error = str(e)
                        st.error(f"❌ Error: {e}")
                        self.add_log("ERROR", f"Failed to load dataset: {e}")
            
            with col3:
                if st.button("👥 Customer Churn", type="primary", key="churn"):
                    try:
                        st.session_state.data_loading_error = None
                        
                        X, y = self.data_manager.load_csv_dataset(
                            "sample_data/customer_churn.csv", 
                            "churned"
                        )
                        
                        st.session_state.loaded_X = X
                        st.session_state.loaded_y = y
                        st.session_state.current_dataset = "csv_customer_churn.csv"
                        st.session_state.data_loaded = True
                        st.session_state.dataset_info = self.data_manager.get_dataset_info("csv_customer_churn.csv")
                        
                        st.success("✅ Customer Churn dataset loaded!")
                        self.add_log("INFO", "Loaded Customer Churn dataset")
                        st.rerun()
                    except Exception as e:
                        st.session_state.data_loading_error = str(e)
                        st.error(f"❌ Error: {e}")
                        self.add_log("ERROR", f"Failed to load dataset: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_clean_dashboard(self, server: CleanFederatedServer):
        """Render clean dashboard"""
        st.markdown('<div class="section-header">🎯 System Dashboard</div>', unsafe_allow_html=True)
        
        if not server:
            st.markdown('''
            <div class="card">
                <h3>🚀 Getting Started</h3>
                <p>Welcome to the Clean Perfect Federated Learning Dashboard!</p>
                <ol>
                    <li>📁 Load a dataset from the Data Management section</li>
                    <li>⚙️ Initialize your federated learning system</li>
                    <li>🎯 Start training to see beautiful visualizations</li>
                </ol>
            </div>
            ''', unsafe_allow_html=True)
            return
        
        if not server.round_history:
            st.markdown('''
            <div class="card">
                <h3>🎯 Ready to Train</h3>
                <p>Your system is initialized! Start training to see perfect metrics and visualizations.</p>
            </div>
            ''', unsafe_allow_html=True)
            return
        
        # Key metrics
        latest_round = server.round_history[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Accuracy", f"{latest_round.global_accuracy:.4f}")
        
        with col2:
            st.metric("📉 Loss", f"{latest_round.global_loss:.4f}")
        
        with col3:
            st.metric("🔄 Rounds", len(server.round_history))
        
        with col4:
            st.metric("👥 Clients", len(latest_round.participating_clients))
        
        # Charts
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy chart
            rounds = [r.round_num for r in server.round_history]
            accuracies = [r.global_accuracy for r in server.round_history]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rounds, y=accuracies,
                mode='lines+markers',
                name='Global Accuracy',
                line=dict(color='#10b981', width=4),
                marker=dict(size=8, color='#10b981')
            ))
            
            fig.update_layout(
                title='📈 Training Accuracy Progress',
                xaxis_title='Round',
                yaxis_title='Accuracy',
                yaxis=dict(range=[0, 1]),
                template='plotly_dark',
                font=dict(color='white'),
                title_font=dict(size=16, color='white'),
                paper_bgcolor='#1e293b',
                plot_bgcolor='#1e293b'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Loss chart
            losses = [r.global_loss for r in server.round_history]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rounds, y=losses,
                mode='lines+markers',
                name='Global Loss',
                line=dict(color='#ef4444', width=4),
                marker=dict(size=8, color='#ef4444')
            ))
            
            fig.update_layout(
                title='📉 Training Loss Progress',
                xaxis_title='Round',
                yaxis_title='Loss',
                template='plotly_dark',
                font=dict(color='white'),
                title_font=dict(size=16, color='white'),
                paper_bgcolor='#1e293b',
                plot_bgcolor='#1e293b'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_clean_settings(self):
        """Render clean settings section"""
        st.markdown('<div class="section-header">⚙️ System Settings</div>', unsafe_allow_html=True)
        
        # Data Management Section
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📁 Data Configuration")
        
        validation_result, validation_msg = self._validate_data_loaded()
        if not validation_result:
            st.error(f"⚠️ {validation_msg}")
            st.info("Please load a dataset first before configuring the system.")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        st.success(f"✅ {validation_msg}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # System Configuration
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔧 System Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_clients = st.slider("👥 Number of Clients", 3, 20, 10)
            distribution = st.selectbox(
                "📊 Data Distribution",
                options=["iid", "non_iid", "quantity_skew"],
                format_func=lambda x: {
                    "iid": "🔄 IID (Independent)",
                    "non_iid": "🎲 Non-IID (Skewed)",
                    "quantity_skew": "📈 Quantity Skewed"
                }[x]
            )
            privacy_ratio = st.slider("🔒 Privacy Ratio", 0.0, 1.0, 0.6)
        
        with col2:
            model_types = st.multiselect(
                "🤖 Model Types",
                options=["logistic_regression", "random_forest"],
                default=["logistic_regression"],
                format_func=lambda x: {
                    "logistic_regression": "📊 Logistic Regression",
                    "random_forest": "🌲 Random Forest"
                }[x]
            )
        
        # Initialize system button
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Initialize System", type="primary", key="init_system"):
                st.info("🔄 Initializing clean federated learning system...")
                
                try:
                    X = st.session_state.loaded_X
                    y = st.session_state.loaded_y
                    
                    # Create server
                    server = CleanFederatedServer(X.shape[1], st.session_state.current_dataset)
                    
                    # Distribute data to clients
                    client_data = self.data_manager.distribute_data_to_clients(
                        st.session_state.current_dataset, n_clients, distribution
                    )
                    
                    # Create clients
                    for i, (client_id, (client_X, client_y)) in enumerate(client_data.items()):
                        privacy_enabled = random.random() < privacy_ratio
                        noise_multiplier = random.uniform(0.1, 0.5) if privacy_enabled else 0.0
                        model_type = model_types[i % len(model_types)]
                        
                        client = CleanFederatedClient(
                            client_id=client_id,
                            X=client_X,
                            y=client_y,
                            privacy_enabled=privacy_enabled,
                            noise_multiplier=noise_multiplier,
                            model_type=model_type
                        )
                        
                        server.register_client(client)
                    
                    # Update session state
                    st.session_state.server = server
                    st.session_state.client_data = client_data
                    st.session_state.system_status = "Ready"
                    st.session_state.system_initialized = True
                    
                    st.success(f"✅ Clean system initialized with {n_clients} clients!")
                    self.add_log("INFO", f"System initialized with {n_clients} clients")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error initializing system: {e}")
                    self.add_log("ERROR", f"System initialization failed: {e}")
        
        with col2:
            if st.session_state.system_initialized:
                st.success("✅ System Ready for Training")
            else:
                st.warning("⚠️ System Not Initialized")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Training Configuration
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎯 Training Configuration")
        
        if not st.session_state.system_initialized:
            st.warning("⚠️ Please initialize the system first.")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_rounds = st.slider("🔄 Training Rounds", 1, 50, 10)
            participation_rate = st.slider("👥 Participation Rate", 0.5, 1.0, 0.8)
        
        with col2:
            if st.session_state.server:
                if st.button("▶️ Start Training", type="primary", key="start_training"):
                    self.run_clean_training_rounds(
                        st.session_state.server, n_rounds, participation_rate
                    )
                    st.success("✅ Clean training completed!")
                    st.rerun()
                
                if st.button("⏹️ Stop Training", key="stop_training"):
                    self.training_active = False
                    st.session_state.system_status = "Stopped"
                    self.add_log("WARNING", "Training stopped by user")
            else:
                st.info("Please initialize the system first")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # System Status
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 System Status")
        self.render_clean_status_overview()
        st.markdown('</div>', unsafe_allow_html=True)
    
    def run_clean_training_rounds(self, server: CleanFederatedServer, n_rounds: int, 
                               participation_rate: float = 0.8):
        """Run clean training rounds"""
        self.training_active = True
        st.session_state.system_status = "Training"
        st.session_state.training_completed = False
        st.session_state.total_rounds = n_rounds
        
        # Progress container
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for round_num in range(n_rounds):
                if not self.training_active:
                    break
                
                st.session_state.current_round = round_num + 1
                
                # Select participating clients
                all_clients = list(server.clients.keys())
                num_participating = max(3, int(len(all_clients) * participation_rate))
                participating_clients = random.sample(all_clients, num_participating)
                
                # Run training round
                round_info = server.run_training_round(participating_clients)
                
                if round_info:
                    self.add_log("INFO", 
                        f"Round {round_info.round_num}: {len(participating_clients)} clients, "
                        f"Accuracy: {round_info.global_accuracy:.4f}"
                    )
                
                # Update progress
                progress = (round_num + 1) / n_rounds
                st.session_state.training_progress = progress
                progress_bar.progress(progress)
                status_text.markdown(f"🎯 **Round {round_num + 1}/{n_rounds}** - Accuracy: `{round_info.global_accuracy:.4f}`")
                
                time.sleep(0.5)
        
        self.training_active = False
        st.session_state.system_status = "Completed"
        st.session_state.training_completed = True
        
        self.add_log("INFO", f"Clean training completed after {n_rounds} rounds")
        
        progress_bar.progress(1.0)
        status_text.markdown("🎉 **Training Completed Successfully!**")
    
    def run(self):
        """Run clean dashboard application"""
        
        # Clean header
        self.render_clean_header()
        
        # Clean sidebar navigation
        with st.sidebar:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); 
                        padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                <h2 style="color: white; margin: 0;">🧭 Navigation</h2>
            </div>
            """, unsafe_allow_html=True)
            
            page = st.selectbox(
                "Choose Section",
                ["📁 Data Management", "🎯 Dashboard", "⚙️ Settings"],
                key="navigation"
            )
        
        # Main content
        server = st.session_state.server
        
        if page == "📁 Data Management":
            self.render_clean_data_management()
        elif page == "🎯 Dashboard":
            self.render_clean_dashboard(server)
        elif page == "⚙️ Settings":
            self.render_clean_settings()
        
        # Auto-refresh option
        if st.sidebar.checkbox("🔄 Auto Refresh", value=False):
            time.sleep(2)
            st.rerun()

def main():
    """Main application entry point"""
    dashboard = CleanPerfectDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
