import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import make_classification, make_regression, load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
import time
import random
from datetime import datetime, timedelta
import threading
import queue

class FederatedClient:
    """Represents a federated learning client"""
    
    def __init__(self, client_id, dataset, model_type="logistic"):
        self.client_id = client_id
        self.dataset = dataset
        self.model_type = model_type
        self.local_model = self._create_model()
        self.accuracy_history = []
        self.loss_history = []
        self.participation_history = []
        self.last_update_time = None
        
    def _create_model(self):
        """Create a local model based on type"""
        if self.model_type == "logistic":
            return LogisticRegression(max_iter=100, random_state=42)
        elif self.model_type == "neural":
            return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42)
        elif self.model_type == "random_forest":
            return RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            return LogisticRegression(max_iter=100, random_state=42)
    
    def train_local(self, global_weights=None, epochs=1, learning_rate=0.01, privacy_noise=0.0):
        """Train model locally on client data"""
        X_train, X_test, y_train, y_test = self.dataset
        
        # Apply global weights if provided
        if global_weights is not None and hasattr(self.local_model, 'coef_'):
            if hasattr(self.local_model, 'coef_'):
                self.local_model.coef_ = global_weights['coef'].copy()
            if hasattr(self.local_model, 'intercept_'):
                self.local_model.intercept_ = global_weights['intercept'].copy()
        
        # Train locally
        self.local_model.fit(X_train, y_train)
        
        # Get model weights
        weights = self._get_model_weights()
        
        # Add privacy noise if specified
        if privacy_noise > 0:
            weights = self._add_privacy_noise(weights, privacy_noise)
        
        # Calculate metrics
        y_pred = self.local_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        try:
            y_proba = self.local_model.predict_proba(X_test)
            loss = log_loss(y_test, y_proba)
        except:
            loss = 0.0
        
        self.accuracy_history.append(accuracy)
        self.loss_history.append(loss)
        self.last_update_time = datetime.now()
        
        return weights, accuracy, loss
    
    def _get_model_weights(self):
        """Extract model weights"""
        weights = {}
        if hasattr(self.local_model, 'coef_'):
            weights['coef'] = self.local_model.coef_.copy()
        if hasattr(self.local_model, 'intercept_'):
            weights['intercept'] = self.local_model.intercept_.copy()
        return weights
    
    def _add_privacy_noise(self, weights, noise_level):
        """Add differential privacy noise to weights"""
        noisy_weights = {}
        for key, value in weights.items():
            if isinstance(value, np.ndarray):
                noise = np.random.normal(0, noise_level, value.shape)
                noisy_weights[key] = value + noise
            else:
                noise = np.random.normal(0, noise_level)
                noisy_weights[key] = value + noise
        return noisy_weights
    
    def update_model(self, global_weights):
        """Update local model with global weights"""
        if hasattr(self.local_model, 'coef_'):
            self.local_model.coef_ = global_weights['coef'].copy()
        if hasattr(self.local_model, 'intercept_'):
            self.local_model.intercept_ = global_weights['intercept'].copy()

class FederatedServer:
    """Represents the federated learning server"""
    
    def __init__(self, model_type="logistic"):
        self.model_type = model_type
        self.global_model = self._create_model()
        self.global_weights = None
        self.round_history = []
        self.accuracy_history = []
        self.loss_history = []
        self.client_contributions = {}
        
    def _create_model(self):
        """Create global model"""
        if self.model_type == "logistic":
            return LogisticRegression(max_iter=100, random_state=42)
        elif self.model_type == "neural":
            return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42)
        elif self.model_type == "random_forest":
            return RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            return LogisticRegression(max_iter=100, random_state=42)
    
    def federated_averaging(self, client_weights, client_contributions=None):
        """Perform federated averaging"""
        if not client_weights:
            return None
        
        # Initialize aggregated weights
        avg_weights = {}
        
        # Get keys from first client
        first_client = list(client_weights.keys())[0]
        for key in client_weights[first_client].keys():
            avg_weights[key] = np.zeros_like(client_weights[first_client][key])
        
        # Weighted averaging
        total_weight = sum(client_contributions.values()) if client_contributions else len(client_weights)
        
        for client_id, weights in client_weights.items():
            weight = client_contributions.get(client_id, 1.0) if client_contributions else 1.0
            for key in weights.keys():
                avg_weights[key] += weights[key] * weight
        
        # Normalize
        for key in avg_weights.keys():
            avg_weights[key] /= total_weight
        
        self.global_weights = avg_weights
        return avg_weights
    
    def evaluate_global_model(self, test_dataset):
        """Evaluate global model on test data"""
        if self.global_weights is None:
            return 0.0, 0.0
        
        # Update global model with current weights
        if hasattr(self.global_model, 'coef_'):
            self.global_model.coef_ = self.global_weights['coef'].copy()
        if hasattr(self.global_model, 'intercept_'):
            self.global_model.intercept_ = self.global_weights['intercept'].copy()
        
        X_test, y_test = test_dataset
        
        y_pred = self.global_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        try:
            y_proba = self.global_model.predict_proba(X_test)
            loss = log_loss(y_test, y_proba)
        except:
            loss = 0.0
        
        return accuracy, loss

class RealFederatedLearning:
    """Main federated learning system"""
    
    def __init__(self):
        self.clients = []
        self.server = None
        self.current_round = 0
        self.max_rounds = 10
        self.is_training = False
        self.training_logs = []
        self.selected_dataset = "Synthetic Classification"
        self.model_type = "logistic"
        self.learning_rate = 0.01
        self.privacy_noise = 0.1
        self.client_fraction = 1.0
        self.local_epochs = 1
        
        # Initialize session state
        if 'federated_state' not in st.session_state:
            st.session_state.federated_state = {
                'clients': [],
                'server': None,
                'current_round': 0,
                'max_rounds': 10,
                'is_training': False,
                'training_logs': [],
                'selected_dataset': "Synthetic Classification",
                'model_type': "logistic",
                'learning_rate': 0.01,
                'privacy_noise': 0.1,
                'client_fraction': 1.0,
                'local_epochs': 1,
                'round_history': [],
                'global_accuracy': [],
                'global_loss': []
            }
    
    def load_dataset(self, dataset_name):
        """Load and prepare dataset"""
        if dataset_name == "Synthetic Classification":
            X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
        elif dataset_name == "Synthetic Regression":
            X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
            # Convert to classification
            y = (y > np.median(y)).astype(int)
        elif dataset_name == "Iris":
            data = load_iris()
            X, y = data.data, data.target
        elif dataset_name == "Wine":
            data = load_wine()
            X, y = data.data, data.target
        elif dataset_name == "Breast Cancer":
            data = load_breast_cancer()
            X, y = data.data, data.target
        else:
            X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return X, y
    
    def create_clients(self, n_clients, dataset_name):
        """Create federated clients with distributed data"""
        X, y = self.load_dataset(dataset_name)
        
        # Split data among clients
        clients_data = []
        samples_per_client = len(X) // n_clients
        
        for i in range(n_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client if i < n_clients - 1 else len(X)
            
            client_X = X[start_idx:end_idx]
            client_y = y[start_idx:end_idx]
            
            # Split into train/test for each client
            X_train, X_test, y_train, y_test = train_test_split(
                client_X, client_y, test_size=0.2, random_state=42
            )
            
            clients_data.append((X_train, X_test, y_train, y_test))
        
        # Create client objects
        self.clients = []
        for i, data in enumerate(clients_data):
            client = FederatedClient(f"Client_{i+1}", data, self.model_type)
            self.clients.append(client)
        
        return self.clients
    
    def run_training_round(self):
        """Run a single federated learning round"""
        if not self.clients or not self.server:
            return
        
        # Select participating clients
        n_participating = max(1, int(len(self.clients) * self.client_fraction))
        participating_clients = random.sample(self.clients, n_participating)
        
        # Collect client updates
        client_weights = {}
        client_accuracies = {}
        client_losses = {}
        client_contributions = {}
        
        for client in participating_clients:
            weights, accuracy, loss = client.train_local(
                self.server.global_weights,
                epochs=self.local_epochs,
                learning_rate=self.learning_rate,
                privacy_noise=self.privacy_noise
            )
            
            client_weights[client.client_id] = weights
            client_accuracies[client.client_id] = accuracy
            client_losses[client.client_id] = loss
            client_contributions[client.client_id] = 1.0  # Equal contribution
            
            client.participation_history.append(self.current_round)
        
        # Perform federated averaging
        self.server.federated_averaging(client_weights, client_contributions)
        
        # Evaluate global model
        X, y = self.load_dataset(self.selected_dataset)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        global_accuracy, global_loss = self.server.evaluate_global_model((X_test, y_test))
        
        # Store round history
        round_data = {
            'round': self.current_round,
            'participating_clients': [c.client_id for c in participating_clients],
            'client_accuracies': client_accuracies,
            'client_losses': client_losses,
            'global_accuracy': global_accuracy,
            'global_loss': global_loss,
            'timestamp': datetime.now()
        }
        
        self.server.round_history.append(round_data)
        self.server.accuracy_history.append(global_accuracy)
        self.server.loss_history.append(global_loss)
        
        # Add training log
        log_entry = {
            'timestamp': datetime.now(),
            'level': 'INFO',
            'message': f"Round {self.current_round} completed. {len(participating_clients)} clients participated. Global accuracy: {global_accuracy:.3f}"
        }
        self.training_logs.append(log_entry)
        
        self.current_round += 1
        
        return round_data
    
    def start_training(self):
        """Start federated training"""
        if self.is_training:
            return
        
        self.is_training = True
        self.current_round = 0
        
        # Initialize server
        self.server = FederatedServer(self.model_type)
        
        # Create clients
        self.create_clients(10, self.selected_dataset)
        
        # Training loop
        while self.current_round < self.max_rounds and self.is_training:
            round_data = self.run_training_round()
            
            # Update session state
            st.session_state.federated_state.update({
                'current_round': self.current_round,
                'round_history': self.server.round_history,
                'global_accuracy': self.server.accuracy_history,
                'global_loss': self.server.loss_history,
                'training_logs': self.training_logs
            })
            
            # Simulate training time
            time.sleep(2)
        
        self.is_training = False
        st.session_state.federated_state['is_training'] = False
        
        # Add completion log
        log_entry = {
            'timestamp': datetime.now(),
            'level': 'SUCCESS',
            'message': f"Training completed! Final accuracy: {self.server.accuracy_history[-1]:.3f}"
        }
        self.training_logs.append(log_entry)
    
    def stop_training(self):
        """Stop federated training"""
        self.is_training = False
        st.session_state.federated_state['is_training'] = False
        
        log_entry = {
            'timestamp': datetime.now(),
            'level': 'WARNING',
            'message': "Training stopped by user"
        }
        self.training_logs.append(log_entry)
    
    def create_dashboard(self):
        """Create the Streamlit dashboard"""
        st.set_page_config(
            page_title="Real Federated Learning Platform",
            page_icon="🔗",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
            min-height: 100vh;
        }
        
        .main .block-container {
            max-width: 100% !important;
            padding: 1rem !important;
        }
        
        .metric-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.8) 100%);
            backdrop-filter: blur(10px);
            border-radius: 1rem;
            padding: 1.5rem;
            border: 1px solid rgba(96, 165, 250, 0.3);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
            margin: 0.5rem;
            width: 100%;
            box-sizing: border-box;
        }
        
        .chart-container {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.8) 100%);
            backdrop-filter: blur(10px);
            border-radius: 1rem;
            padding: 1.5rem;
            border: 1px solid rgba(96, 165, 250, 0.3);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
            margin: 0.5rem;
            width: 100%;
            box-sizing: border-box;
        }
        
        .sidebar-section {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(10px);
            border-radius: 1rem;
            padding: 1rem;
            border: 1px solid rgba(96, 165, 250, 0.2);
            margin: 0.5rem 0;
            width: 100%;
            box-sizing: border-box;
        }
        
        .log-entry {
            background: rgba(30, 41, 59, 0.6);
            border-radius: 0.5rem;
            padding: 0.75rem;
            margin: 0.25rem 0;
            border-left: 3px solid #3b82f6;
            width: 100%;
            box-sizing: border-box;
        }
        
        .log-warning {
            border-left-color: #f59e0b;
        }
        
        .log-error {
            border-left-color: #ef4444;
        }
        
        .log-success {
            border-left-color: #10b981;
        }
        
        /* Fix layout issues */
        .stColumns {
            width: 100% !important;
        }
        
        .stColumn {
            width: 100% !important;
            padding: 0.5rem !important;
            box-sizing: border-box;
        }
        
        .stPlotlyChart {
            width: 100% !important;
        }
        
        .element-container {
            width: 100% !important;
            max-width: 100% !important;
        }
        
        .streamlit-container {
            width: 100% !important;
            max-width: 100% !important;
        }
        
        /* Fix sidebar */
        .css-1d391kg {
            width: 300px !important;
        }
        
        .css-1lcbmhc {
            width: 100% !important;
        }
        
        /* Fix tabs */
        .stTabs [data-baseweb="tab-list"] {
            width: 100% !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            flex: 1 !important;
            min-width: 100px !important;
        }
        
        /* Fix dataframes */
        .dataframe {
            width: 100% !important;
            overflow-x: auto !important;
        }
        
        .dataframe table {
            width: 100% !important;
            min-width: 600px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 3rem; font-weight: 900; background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #2563eb 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        🔗 Real Federated Learning Platform
        </h1>
        <p style='color: #94a3b8; font-size: 1.1rem; margin-top: 0.5rem;'>
        Distributed Machine Learning with Privacy Protection
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar Configuration
        with st.sidebar:
            st.markdown("### ⚙️ Configuration")
            
            # Dataset Selection
            datasets = ["Synthetic Classification", "Synthetic Regression", "Iris", "Wine", "Breast Cancer"]
            self.selected_dataset = st.selectbox(
                "Dataset",
                datasets,
                index=datasets.index(st.session_state.federated_state.get('selected_dataset', "Synthetic Classification"))
            )
            
            # Model Selection
            models = ["logistic", "neural", "random_forest"]
            self.model_type = st.selectbox(
                "Model Type",
                models,
                index=models.index(st.session_state.federated_state.get('model_type', "logistic"))
            )
            
            # Training Parameters
            st.markdown("#### 📊 Training Parameters")
            self.max_rounds = st.slider("Communication Rounds", 1, 50, st.session_state.federated_state.get('max_rounds', 10))
            self.client_fraction = st.slider("Client Fraction", 0.1, 1.0, st.session_state.federated_state.get('client_fraction', 1.0), 0.1)
            self.local_epochs = st.slider("Local Epochs", 1, 10, st.session_state.federated_state.get('local_epochs', 1))
            self.learning_rate = st.slider("Learning Rate", 0.001, 0.1, st.session_state.federated_state.get('learning_rate', 0.01), 0.001)
            
            # Privacy Settings
            st.markdown("#### 🔒 Privacy Settings")
            self.privacy_noise = st.slider("Privacy Noise Level", 0.0, 1.0, st.session_state.federated_state.get('privacy_noise', 0.1), 0.01)
            
            # Training Control
            st.markdown("#### 🚀 Training Control")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Start Training", type="primary", disabled=st.session_state.federated_state.get('is_training', False)):
                    # Update session state
                    st.session_state.federated_state.update({
                        'selected_dataset': self.selected_dataset,
                        'model_type': self.model_type,
                        'max_rounds': self.max_rounds,
                        'client_fraction': self.client_fraction,
                        'local_epochs': self.local_epochs,
                        'learning_rate': self.learning_rate,
                        'privacy_noise': self.privacy_noise,
                        'is_training': True
                    })
                    
                    # Start training in background
                    training_thread = threading.Thread(target=self.start_training)
                    training_thread.daemon = True
                    training_thread.start()
            
            with col2:
                if st.button("Stop Training", disabled=not st.session_state.federated_state.get('is_training', False)):
                    self.stop_training()
        
        # Main Content
        # Get current state
        current_round = st.session_state.federated_state.get('current_round', 0)
        max_rounds = st.session_state.federated_state.get('max_rounds', 10)
        is_training = st.session_state.federated_state.get('is_training', False)
        round_history = st.session_state.federated_state.get('round_history', [])
        global_accuracy = st.session_state.federated_state.get('global_accuracy', [])
        global_loss = st.session_state.federated_state.get('global_loss', [])
        training_logs = st.session_state.federated_state.get('training_logs', [])
        
        # Status Metrics
        st.markdown("### 📊 System Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
            <h3 style='color: #3b82f6; font-size: 2rem; margin: 0;'>{current_round}</h3>
            <p style='color: #94a3b8; margin: 0;'>Current Round</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            status = "🟢 Training" if is_training else "🔴 Idle"
            st.markdown(f"""
            <div class='metric-card'>
            <h3 style='color: #10b981; font-size: 2rem; margin: 0;'>{status}</h3>
            <p style='color: #94a3b8; margin: 0;'>System Status</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            accuracy = global_accuracy[-1] if global_accuracy else 0.0
            st.markdown(f"""
            <div class='metric-card'>
            <h3 style='color: #60a5fa; font-size: 2rem; margin: 0;'>{accuracy:.3f}</h3>
            <p style='color: #94a3b8; margin: 0;'>Global Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            loss = global_loss[-1] if global_loss else 0.0
            st.markdown(f"""
            <div class='metric-card'>
            <h3 style='color: #f59e0b; font-size: 2rem; margin: 0;'>{loss:.3f}</h3>
            <p style='color: #94a3b8; margin: 0;'>Global Loss</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Progress Bar
        if max_rounds > 0:
            progress = current_round / max_rounds
            st.progress(progress, text=f"Training Progress: {current_round}/{max_rounds} rounds")
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "📈 Analytics", "👥 Clients", "🔒 Privacy", "📝 Logs"])
        
        with tab1:
            st.markdown("### 📊 Training Dashboard")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                if global_accuracy:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(1, len(global_accuracy) + 1)),
                        y=global_accuracy,
                        mode='lines+markers',
                        name='Global Accuracy',
                        line=dict(color='#3b82f6', width=3),
                        marker=dict(color='#3b82f6', size=8)
                    ))
                    fig.update_layout(
                        title="📈 Global Accuracy Over Rounds",
                        xaxis_title="Round",
                        yaxis_title="Accuracy",
                        template="plotly_dark",
                        height=400,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                if global_loss:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(1, len(global_loss) + 1)),
                        y=global_loss,
                        mode='lines+markers',
                        name='Global Loss',
                        line=dict(color='#ef4444', width=3),
                        marker=dict(color='#ef4444', size=8)
                    ))
                    fig.update_layout(
                        title="📉 Global Loss Over Rounds",
                        xaxis_title="Round",
                        yaxis_title="Loss",
                        template="plotly_dark",
                        height=400,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### 📈 Advanced Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                if round_history:
                    # Client participation heatmap
                    participation_data = []
                    for round_data in round_history[-10:]:  # Last 10 rounds
                        for client_id in round_data['participating_clients']:
                            participation_data.append({
                                'Round': round_data['round'],
                                'Client': client_id,
                                'Participated': 1
                            })
                    
                    if participation_data:
                        df_participation = pd.DataFrame(participation_data)
                        fig = px.density_heatmap(
                            df_participation,
                            x='Round',
                            y='Client',
                            z='Participated',
                            title="🔥 Client Participation Heatmap",
                            template="plotly_dark",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                if round_history:
                    # Client accuracy distribution
                    client_accuracies = []
                    for round_data in round_history[-5:]:  # Last 5 rounds
                        for client_id, acc in round_data['client_accuracies'].items():
                            client_accuracies.append({
                                'Client': client_id,
                                'Accuracy': acc,
                                'Round': round_data['round']
                            })
                    
                    if client_accuracies:
                        df_accuracies = pd.DataFrame(client_accuracies)
                        fig = px.box(
                            df_accuracies,
                            x='Client',
                            y='Accuracy',
                            title="📊 Client Accuracy Distribution",
                            template="plotly_dark",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### 👥 Client Performance")
            
            if round_history:
                # Create client performance table
                client_data = {}
                for round_data in round_history:
                    for client_id, acc in round_data['client_accuracies'].items():
                        if client_id not in client_data:
                            client_data[client_id] = []
                        client_data[client_id].append(acc)
                
                # Calculate statistics
                client_stats = []
                for client_id, accuracies in client_data.items():
                    client_stats.append({
                        'Client': client_id,
                        'Rounds Participated': len(accuracies),
                        'Average Accuracy': np.mean(accuracies),
                        'Best Accuracy': np.max(accuracies),
                        'Latest Accuracy': accuracies[-1] if accuracies else 0.0,
                        'Improvement': accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0.0
                    })
                
                df_clients = pd.DataFrame(client_stats)
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.dataframe(
                    df_clients.round(3),
                    use_container_width=True,
                    hide_index=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Client performance chart
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig = go.Figure()
                
                for client_id in list(client_data.keys())[:5]:  # Show first 5 clients
                    fig.add_trace(go.Scatter(
                        x=list(range(1, len(client_data[client_id]) + 1)),
                        y=client_data[client_id],
                        mode='lines+markers',
                        name=client_id,
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    title="📈 Client Accuracy Progression",
                    xaxis_title="Round",
                    yaxis_title="Accuracy",
                    template="plotly_dark",
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            st.markdown("### 🔒 Privacy Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                
                # Privacy budget consumption
                privacy_budget = [self.privacy_noise * (i + 1) for i in range(len(global_accuracy))]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(privacy_budget) + 1)),
                    y=privacy_budget,
                    mode='lines+markers',
                    name='Privacy Budget Used',
                    line=dict(color='#8b5cf6', width=3),
                    marker=dict(color='#8b5cf6', size=8),
                    fill='tonexty'
                ))
                
                fig.update_layout(
                    title="🔒 Privacy Budget Consumption",
                    xaxis_title="Round",
                    yaxis_title="Privacy Noise Level",
                    template="plotly_dark",
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                
                # Privacy vs Accuracy trade-off
                if len(global_accuracy) > 1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=global_accuracy,
                        y=global_loss,
                        mode='lines+markers',
                        name='Privacy-Accuracy Trade-off',
                        line=dict(color='#06b6d4', width=3),
                        marker=dict(color='#06b6d4', size=8)
                    ))
                    
                    fig.update_layout(
                        title="⚖️ Privacy vs Accuracy Trade-off",
                        xaxis_title="Accuracy",
                        yaxis_title="Loss",
                        template="plotly_dark",
                        height=400,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Privacy metrics
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown(f"""
            **Privacy Configuration:**
            - **Noise Level:** {self.privacy_noise:.3f}
            - **Client Fraction:** {self.client_fraction:.1f}
            - **Differential Privacy:** {'✅ Enabled' if self.privacy_noise > 0 else '❌ Disabled'}
            - **Privacy Budget Remaining:** {max(0, 1.0 - self.privacy_noise * len(global_accuracy)):.3f}
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab5:
            st.markdown("### 📝 Training Logs")
            
            # Display logs
            log_container = st.container()
            
            for log in reversed(training_logs[-20:]):  # Show last 20 logs
                log_class = ""
                if log['level'] == 'ERROR':
                    log_class = "log-error"
                elif log['level'] == 'WARNING':
                    log_class = "log-warning"
                elif log['level'] == 'SUCCESS':
                    log_class = "log-success"
                
                st.markdown(f"""
                <div class='log-entry {log_class}'>
                <strong>{log['timestamp'].strftime('%H:%M:%S')}</strong> - 
                <span style='color: #3b82f6;'>[{log['level']}]</span> - 
                {log['message']}
                </div>
                """, unsafe_allow_html=True)
        
        # Auto-refresh
        if is_training:
            time.sleep(1)
            st.rerun()

def main():
    """Main function"""
    app = RealFederatedLearning()
    app.create_dashboard()

if __name__ == "__main__":
    main()
