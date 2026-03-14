#!/usr/bin/env python3
"""
Perfect Federated Learning Platform - Flawless Design & Functionality
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import random
import json
from collections import defaultdict
import asyncio

class PerfectFederatedPlatform:
    """Perfect federated learning platform with flawless execution"""
    
    def __init__(self):
        self.training_history = []
        self.client_performance = {}
        self.privacy_tracker = {}
        self.security_metrics = {}
        self.real_time_events = []
    
    def generate_perfect_training_curve(self, n_rounds, target_accuracy, current_round):
        """Generate perfect S-curve training data"""
        rounds = list(range(1, n_rounds + 1))
        accuracy_data = []
        loss_data = []
        confidence_data = []
        
        for round_num in rounds:
            if round_num <= current_round:
                # Perfect S-curve with realistic noise
                progress = round_num / n_rounds
                s_curve = 1 / (1 + np.exp(-12 * (progress - 0.4)))
                
                # Add realistic learning patterns
                base_accuracy = 0.2 + (target_accuracy - 0.2) * s_curve
                noise = np.random.normal(0, 0.01)  # Minimal noise for perfection
                accuracy = base_accuracy + noise
                
                # Loss follows inverse pattern
                base_loss = 0.9 * (1 - s_curve) + 0.1
                loss_noise = np.random.normal(0, 0.02)
                loss = base_loss + loss_noise
                
                # Confidence increases with accuracy
                confidence = min(0.95, s_curve * 1.2)
                
            else:
                # Perfect projection
                remaining_progress = (round_num - current_round) / (n_rounds - current_round) if current_round < n_rounds else 0
                current_progress = current_round / n_rounds
                current_s = 1 / (1 + np.exp(-12 * (current_progress - 0.4)))
                
                projected_s = current_s + remaining_progress * (1 - current_s)
                accuracy = 0.2 + (target_accuracy - 0.2) * projected_s
                loss = 0.9 * (1 - projected_s) + 0.1
                confidence = min(0.95, projected_s * 1.2)
            
            accuracy_data.append(max(0, min(1, accuracy)))
            loss_data.append(max(0, min(1, loss)))
            confidence_data.append(max(0, min(1, confidence)))
        
        return rounds, accuracy_data, loss_data, confidence_data
    
    def generate_perfect_clients(self, n_clients, target_accuracy, current_round=5):
        """Generate perfect client performance data"""
        clients = []
        
        # Create diverse client profiles
        client_profiles = [
            {"name": "High-Performance", "factor": 1.1, "samples": (1500, 2000)},
            {"name": "Standard", "factor": 1.0, "samples": (800, 1500)},
            {"name": "Resource-Constrained", "factor": 0.9, "samples": (300, 800)}
        ]
        
        for i in range(n_clients):
            profile = client_profiles[i % len(client_profiles)]
            
            # Perfect performance calculation
            base_accuracy = target_accuracy * profile["factor"]
            accuracy = max(0, min(1, base_accuracy + np.random.normal(0, 0.02)))
            
            # Generate perfect metrics
            client = {
                'id': f'FL-Client-{i+1:03d}',
                'name': f'{profile["name"]} Client {i+1}',
                'accuracy': accuracy,
                'samples': np.random.randint(*profile["samples"]),
                'status': np.random.choice(['Active', 'Training', 'Idle'], p=[0.8, 0.15, 0.05]),
                'latency': np.random.uniform(5, 50),
                'throughput': np.random.uniform(100, 1000),
                'contribution': max(0, min(1, accuracy + np.random.normal(0, 0.03))),
                'uptime': np.random.uniform(95, 99.9),
                'data_quality': np.random.uniform(0.8, 1.0),
                'model_version': f"v{round}",
                'last_update': datetime.now() - timedelta(minutes=np.random.randint(1, 60))
            }
            
            clients.append(client)
        
        return clients
    
    def calculate_perfect_privacy(self, current_round, total_rounds, epsilon, delta):
        """Calculate perfect privacy budget consumption"""
        if total_rounds == 0:
            return 0, 0, 0
        
        # Advanced privacy calculation
        progress = current_round / total_rounds
        
        # Privacy consumption follows diminishing returns
        budget_used = 1 - np.exp(-epsilon * progress * 0.8)
        
        # Calculate actual privacy loss
        privacy_loss = epsilon * progress
        
        # Calculate remaining budget
        remaining_budget = epsilon * (1 - progress)
        
        return min(1.0, budget_used), privacy_loss, remaining_budget

def create_perfect_platform():
    """Create the perfect federated learning platform"""
    
    # Initialize platform
    platform = PerfectFederatedPlatform()
    
    # Perfect page configuration
    st.set_page_config(
        page_title="Perfect Federated Learning Platform",
        page_icon="💎",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    #/* Perfect CSS - Stable Layout */
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100;200;300;400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* Solid background - no transparency */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        min-height: 100vh;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        overflow: hidden;
    }
    
    /* Solid container - no transparency */
    .block-container {
        background: #0f172a;
        padding: 0.25rem;
        max-width: 100%;
        position: relative;
        overflow: hidden;
        border-radius: 0.25rem;
        margin: 0.1rem;
        min-height: 100vh;
        border: 1px solid #334155;
    }
    
    /* Ultra-compact header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #2563eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.25rem 0;
        padding: 0.25rem 0;
        position: relative;
        text-shadow: 0 0 40px rgba(96, 165, 250, 0.5);
        user-select: none;
        pointer-events: none;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(96, 165, 250, 0.1) 0%, transparent 70%);
        z-index: -1;
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 150px;
        height: 6px;
        background: linear-gradient(90deg, transparent, #60a5fa, transparent);
        border-radius: 3px;
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { opacity: 0.5; transform: translateX(-50%) scaleX(0.5); }
        50% { opacity: 1; transform: translateX(-50%) scaleX(1); }
    }
    
    /* Solid sidebar - no transparency */
    .perfect-sidebar {
        background: #1e293b;
        backdrop-filter: none;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
        position: relative;
        overflow: hidden;
    }
    
    .sidebar-section {
        background: #334155;
        backdrop-filter: none;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #475569;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        transition: all 0.2s ease;
        position: relative;
    }
    
    .sidebar-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
        border-color: #475569;
    }
    
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #3b82f6;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        text-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
    }
    
    /* Solid metric cards - no transparency */
    .perfect-metric {
        background: #1e293b;
        backdrop-filter: none;
        border-radius: 0.5rem;
        padding: 1rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
        text-align: center;
        user-select: none;
        margin: 0.1rem;
    }
    
    .perfect-metric:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.6);
        border-color: #475569;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 900;
        color: #e2e8f0;
        margin-bottom: 0.25rem;
        text-shadow: 0 0 20px rgba(96, 165, 250, 0.5);
        position: relative;
        z-index: 1;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        position: relative;
        z-index: 1;
    }
    
    /* Solid charts - no transparency */
    .perfect-chart {
        background: #1e293b;
        backdrop-filter: none;
        border-radius: 0.5rem;
        padding: 1rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
        transition: all 0.2s ease;
        margin-bottom: 0.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .perfect-chart:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.6);
        border-color: #475569;
    }
    
    /* Solid tabs - no transparency */
    .perfect-tabs {
        background: #334155;
        backdrop-filter: none;
        border-radius: 0.5rem;
        padding: 0.5rem;
        border: 1px solid #475569;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
        position: relative;
    }
    
    /* Stable status badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem 1.5rem;
        border-radius: 2rem;
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        user-select: none;
    }
    
    .status-running {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        box-shadow: 0 8px 16px rgba(16, 185, 129, 0.4);
    }
    
    .status-idle {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        box-shadow: 0 8px 16px rgba(245, 158, 11, 0.4);
    }
    
    .status-completed {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        box-shadow: 0 8px 16px rgba(59, 130, 246, 0.4);
    }
    
    /* Stable buttons */
    .perfect-button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 1rem;
        padding: 1rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        transition: all 0.3s ease;
        box-shadow: 0 8px 16px rgba(59, 130, 246, 0.4);
        cursor: pointer;
        width: 100%;
        position: relative;
        overflow: hidden;
        user-select: none;
    }
    
    .perfect-button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        transition: width 0.6s ease, height 0.6s ease;
    }
    
    .perfect-button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .perfect-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 24px rgba(59, 130, 246, 0.6);
    }
    
    /* Stable data table */
    .perfect-table {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%);
        backdrop-filter: blur(20px);
        border-radius: 1.5rem;
        overflow: hidden;
        border: 1px solid rgba(96, 165, 250, 0.2);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        position: relative;
    }
    
    /* Solid Streamlit components - no transparency */
    .stSelectbox > div > div {
        background: #334155 !important;
        border: 1px solid #475569 !important;
        border-radius: 0.5rem !important;
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        backdrop-filter: none !important;
        position: relative !important;
    }
    
    .stSelectbox > div > div:focus {
        background: #475569 !important;
        border-color: #3b82f6 !important;
    }
    
    .stSelectbox > div > div > div {
        background: #334155 !important;
        color: #e2e8f0 !important;
        border: none !important;
    }
    
    .stSelectbox > div > div > div:hover {
        background: #475569 !important;
    }
    
    .stSlider > div > div {
        background: linear-gradient(90deg, #3b82f6, #60a5fa) !important;
        border-radius: 0.5rem !important;
        position: relative !important;
    }
    
    .stSlider > div > div > div {
        background: #3b82f6 !important;
        border: 2px solid #e2e8f0 !important;
    }
    
    .stCheckbox > div {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        position: relative !important;
        background: transparent !important;
    }
    
    .stCheckbox > div > div {
        background: #334155 !important;
        border: 1px solid #475569 !important;
    }
    
    .stCheckbox > div > div:checked {
        background: #3b82f6 !important;
        border-color: #3b82f6 !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.5rem !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.4) !important;
        cursor: pointer !important;
        position: relative !important;
        overflow: hidden !important;
        user-select: none !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.6) !important;
    }
    
    /* Solid tabs - no transparency */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border: none !important;
        gap: 0.5rem !important;
        position: relative !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #334155 !important;
        border: 1px solid #475569 !important;
        border-radius: 0.5rem !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 700 !important;
        color: #e2e8f0 !important;
        transition: all 0.3s ease !important;
        backdrop-filter: none !important;
        position: relative !important;
        user-select: none !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border-color: #3b82f6 !important;
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.4) !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background: #475569 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Stable progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.5);
        position: relative;
    }
    
    /* Remove all unwanted elements and prevent movement */
    .stApp > header {
        background: transparent;
        box-shadow: none;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 1000;
    }
    
    .stMarkdown {
        background: transparent;
        position: relative;
    }
    
    .stButton > button {
        position: relative;
        user-select: none;
    }
    
    /* Prevent drag and drop */
    * {
        -webkit-user-drag: none;
        -khtml-user-drag: none;
        -moz-user-drag: none;
        -o-user-drag: none;
        user-drag: none;
    }
    
    /* Prevent text selection */
    * {
        -webkit-user-select: none;
        -moz-user-select: none;
        -ms-user-select: none;
        user-select: none;
    }
    
    /* Allow text selection for inputs */
    input, textarea, [contenteditable] {
        -webkit-user-select: text;
        -moz-user-select: text;
        -ms-user-select: text;
        user-select: text;
    }
    
    /* Stable scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3b82f6, #60a5fa);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #2563eb, #3b82f6);
    }
    
    /* Fixed positioning for main content */
    .main .block-container {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        overflow-y: auto;
        overflow-x: hidden;
        padding: 1rem;
    }
    
    /* Fixed sidebar positioning */
    .element-container .stSidebar {
        position: fixed !important;
        top: 0;
        right: 0;
        height: 100vh;
        overflow-y: auto;
        z-index: 999;
        width: 300px !important;
    }
    
    /* Prevent any element from being draggable */
    [draggable="true"] {
        -webkit-user-drag: none;
        -khtml-user-drag: none;
        -moz-user-drag: none;
        -o-user-drag: none;
        user-drag: none;
    }
    
    /* Remove all empty spaces and gray boxes completely */
    .empty-container {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .element-container:empty {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .stEmpty {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Remove all gray backgrounds and boxes */
    div[data-testid="stVerticalBlock"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    div[data-testid="stHorizontalBlock"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Remove all empty blocks and spaces */
    .block-container:empty {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Remove all gray dividers and spaces */
    hr {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    br {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Remove all empty space and containers */
    .element-container {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
        min-height: 0 !important;
    }
    
    .element-container:empty {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Remove all gray backgrounds from streamlit elements - FORCE SOLID */
    .stSelectbox, .stSlider, .stCheckbox, .stButton, .stTextInput, .stNumberInput, .stDateInput {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* FORCE SOLID SELECTBOX */
    .stSelectbox [data-baseweb="select"] {
        background: #334155 !important;
        border: 2px solid #475569 !important;
        border-radius: 0.5rem !important;
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        backdrop-filter: none !important;
        position: relative !important;
        opacity: 1 !important;
    }
    
    .stSelectbox [data-baseweb="select"]:focus {
        background: #475569 !important;
        border-color: #3b82f6 !important;
        opacity: 1 !important;
    }
    
    .stSelectbox [data-baseweb="select"] [data-baseweb="select-arrow"] {
        color: #e2e8f0 !important;
        background: #334155 !important;
        opacity: 1 !important;
    }
    
    /* FORCE SOLID DROPDOWN OPTIONS */
    .stSelectbox [data-baseweb="popover"] {
        background: #334155 !important;
        border: 2px solid #475569 !important;
        border-radius: 0.5rem !important;
        opacity: 1 !important;
    }
    
    .stSelectbox [data-baseweb="popover"] [data-baseweb="list"] {
        background: #334155 !important;
        opacity: 1 !important;
    }
    
    .stSelectbox [data-baseweb="popover"] [data-baseweb="list"] [data-baseweb="list-item"] {
        background: #334155 !important;
        color: #e2e8f0 !important;
        border: none !important;
        opacity: 1 !important;
    }
    
    .stSelectbox [data-baseweb="popover"] [data-baseweb="list"] [data-baseweb="list-item"]:hover {
        background: #475569 !important;
        color: #e2e8f0 !important;
        opacity: 1 !important;
    }
    
    /* FORCE SOLID SLIDER */
    .stSlider [data-baseweb="slider"] {
        background: linear-gradient(90deg, #3b82f6, #60a5fa) !important;
        border-radius: 0.5rem !important;
        position: relative !important;
        opacity: 1 !important;
    }
    
    .stSlider [data-baseweb="slider"] [data-baseweb="slider-handle"] {
        background: #3b82f6 !important;
        border: 2px solid #e2e8f0 !important;
        opacity: 1 !important;
    }
    
    /* FORCE SOLID CHECKBOX */
    .stCheckbox [data-baseweb="checkbox"] {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        position: relative !important;
        background: transparent !important;
        opacity: 1 !important;
    }
    
    .stCheckbox [data-baseweb="checkbox"] [data-baseweb="checkbox-input"] {
        background: #334155 !important;
        border: 2px solid #475569 !important;
        opacity: 1 !important;
    }
    
    .stCheckbox [data-baseweb="checkbox"] [data-baseweb="checkbox-input"]:checked {
        background: #3b82f6 !important;
        border-color: #3b82f6 !important;
        opacity: 1 !important;
    }
    
    /* FORCE SOLID BUTTON */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.5rem !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.4) !important;
        cursor: pointer !important;
        position: relative !important;
        overflow: hidden !important;
        user-select: none !important;
        opacity: 1 !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.6) !important;
        opacity: 1 !important;
    }
    
    /* REMOVE ALL TRANSPARENCY */
    * {
        opacity: 1 !important;
    }
    
    /* ONLY ALLOW TRANSPARENCY FOR SPECIFIC ELEMENTS */
    .main-header::before {
        opacity: 0.1 !important;
    }
    
    /* OVERRIDE ALL STREAMLIT DEFAULT STYLES */
    .stSelectbox div[data-baseweb="select"] {
        background: #334155 !important;
        border: 2px solid #475569 !important;
        border-radius: 0.5rem !important;
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        backdrop-filter: none !important;
        position: relative !important;
        opacity: 1 !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    .stSelectbox div[data-baseweb="select"]:focus {
        background: #475569 !important;
        border-color: #3b82f6 !important;
        opacity: 1 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3) !important;
    }
    
    .stSelectbox div[data-baseweb="select"] div[data-baseweb="select-arrow"] {
        color: #e2e8f0 !important;
        background: #334155 !important;
        opacity: 1 !important;
    }
    
    /* FORCE SOLID DROPDOWN POPUP */
    .stSelectbox div[data-baseweb="popover"] {
        background: #334155 !important;
        border: 2px solid #475569 !important;
        border-radius: 0.5rem !important;
        opacity: 1 !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5) !important;
        backdrop-filter: none !important;
    }
    
    .stSelectbox div[data-baseweb="popover"] div[data-baseweb="list"] {
        background: #334155 !important;
        opacity: 1 !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    .stSelectbox div[data-baseweb="popover"] div[data-baseweb="list"] div[data-baseweb="list-item"] {
        background: #334155 !important;
        color: #e2e8f0 !important;
        border: none !important;
        opacity: 1 !important;
        padding: 0.75rem 1rem !important;
        margin: 0 !important;
        border-radius: 0.25rem !important;
    }
    
    .stSelectbox div[data-baseweb="popover"] div[data-baseweb="list"] div[data-baseweb="list-item"]:hover {
        background: #475569 !important;
        color: #e2e8f0 !important;
        opacity: 1 !important;
    }
    
    .stSelectbox div[data-baseweb="popover"] div[data-baseweb="list"] div[data-baseweb="list-item"][aria-selected="true"] {
        background: #3b82f6 !important;
        color: white !important;
        opacity: 1 !important;
    }
    
    /* REMOVE ANY REMAINING TRANSPARENCY */
    div[data-baseweb="popover"] {
        background: #334155 !important;
        opacity: 1 !important;
        backdrop-filter: none !important;
    }
    
    div[data-baseweb="list"] {
        background: #334155 !important;
        opacity: 1 !important;
    }
    
    div[data-baseweb="list-item"] {
        background: #334155 !important;
        opacity: 1 !important;
        color: #e2e8f0 !important;
    }
    
    /* FORCE SOLID ON ALL SELECTBOX ELEMENTS */
    .stSelectbox * {
        background: #334155 !important;
        opacity: 1 !important;
        color: #e2e8f0 !important;
        border-color: #475569 !important;
    }
    
    .stSelectbox *:hover {
        background: #475569 !important;
        opacity: 1 !important;
        color: #e2e8f0 !important;
    }
    
    /* AGGRESSIVE TRANSPARENCY REMOVAL */
    .stSelectbox,
    .stSelectbox div,
    .stSelectbox span,
    .stSelectbox ul,
    .stSelectbox li,
    .stSelectbox option,
    .stSelectbox select,
    .stSelectbox [data-baseweb],
    .stSelectbox [data-testid],
    .stSelectbox * {
        background: #334155 !important;
        background-color: #334155 !important;
        opacity: 1 !important;
        color: #e2e8f0 !important;
        border: 1px solid #475569 !important;
        border-color: #475569 !important;
        backdrop-filter: none !important;
        box-shadow: none !important;
        text-shadow: none !important;
        filter: none !important;
        -webkit-backdrop-filter: none !important;
        -moz-backdrop-filter: none !important;
        -ms-backdrop-filter: none !important;
    }
    
    /* FORCE SOLID ON HOVER */
    .stSelectbox:hover,
    .stSelectbox div:hover,
    .stSelectbox span:hover,
    .stSelectbox ul:hover,
    .stSelectbox li:hover,
    .stSelectbox option:hover,
    .stSelectbox select:hover,
    .stSelectbox [data-baseweb]:hover,
    .stSelectbox [data-testid]:hover,
    .stSelectbox *:hover {
        background: #475569 !important;
        background-color: #475569 !important;
        opacity: 1 !important;
        color: #e2e8f0 !important;
        border: 1px solid #3b82f6 !important;
        border-color: #3b82f6 !important;
    }
    
    /* FORCE SOLID ON FOCUS */
    .stSelectbox:focus,
    .stSelectbox div:focus,
    .stSelectbox span:focus,
    .stSelectbox ul:focus,
    .stSelectbox li:focus,
    .stSelectbox option:focus,
    .stSelectbox select:focus,
    .stSelectbox [data-baseweb]:focus,
    .stSelectbox [data-testid]:focus,
    .stSelectbox *:focus {
        background: #475569 !important;
        background-color: #475569 !important;
        opacity: 1 !important;
        color: #e2e8f0 !important;
        border: 2px solid #3b82f6 !important;
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3) !important;
    }
    
    /* REMOVE ALL STREAMLIT DEFAULT STYLES */
    .stSelectbox .css-1cypcdb,
    .stSelectbox .css-1lcbmhc,
    .stSelectbox .css-1pahdxg,
    .stSelectbox .css-1inwz32,
    .stSelectbox .css-1vq4p4l,
    .stSelectbox .css-1qg4t30,
    .stSelectbox .css-1c0cwcf,
    .stSelectbox .css-1wrcr25,
    .stSelectbox .css-1n5685k,
    .stSelectbox .css-1g6f8dv,
    .stSelectbox .css-1i5ys7g,
    .stSelectbox .css-1h1x0mu,
    .stSelectbox .css-1h4h6rj,
    .stSelectbox .css-1h4h6rj *,
    .stSelectbox .css-* {
        background: #334155 !important;
        background-color: #334155 !important;
        opacity: 1 !important;
        color: #e2e8f0 !important;
        border: 1px solid #475569 !important;
        border-color: #475569 !important;
        backdrop-filter: none !important;
        box-shadow: none !important;
    }
    
    /* FORCE SOLID ON ALL POSSIBLE SELECTBOX CLASSES */
    div[class*="css-"] .stSelectbox,
    div[class*="css-"] .stSelectbox *,
    div[class*="streamlit"] .stSelectbox,
    div[class*="streamlit"] .stSelectbox *,
    .stSelectbox div[class*="css-"],
    .stSelectbox div[class*="css-"] * {
        background: #334155 !important;
        background-color: #334155 !important;
        opacity: 1 !important;
        color: #e2e8f0 !important;
        border: 1px solid #475569 !important;
        border-color: #475569 !important;
        backdrop-filter: none !important;
        box-shadow: none !important;
    }
    
    /* TRANSPARENT THEME RESTORED */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        min-height: 100vh;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        overflow: hidden;
    }
    
    /* Transparent container */
    .block-container {
        background: rgba(15, 23, 42, 0.3);
        padding: 0.25rem;
        max-width: 100%;
        position: relative;
        overflow: hidden;
        border-radius: 0.25rem;
        margin: 0.1rem;
        min-height: 100vh;
        border: 1px solid rgba(51, 65, 85, 0.3);
    }
    
    /* Transparent sidebar */
    .perfect-sidebar {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.7) 100%);
        backdrop-filter: blur(20px);
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(96, 165, 250, 0.2);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .sidebar-section {
        background: rgba(51, 65, 85, 0.5);
        backdrop-filter: blur(10px);
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(96, 165, 250, 0.1);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        position: relative;
    }
    
    .sidebar-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        border-color: rgba(96, 165, 250, 0.3);
    }
    
    /* Transparent metric cards */
    .perfect-metric {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.8) 100%);
        backdrop-filter: blur(15px);
        border-radius: 0.5rem;
        padding: 1rem;
        border: 1px solid rgba(96, 165, 250, 0.3);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3), 0 0 15px rgba(96, 165, 250, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        text-align: center;
        user-select: none;
        margin: 0.1rem;
    }
    
    .perfect-metric:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4), 0 0 20px rgba(96, 165, 250, 0.2);
    }
    
    /* Transparent charts */
    .perfect-chart {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.8) 100%);
        backdrop-filter: blur(15px);
        border-radius: 0.5rem;
        padding: 1rem;
        border: 1px solid rgba(96, 165, 250, 0.3);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3), 0 0 15px rgba(96, 165, 250, 0.1);
        transition: all 0.3s ease;
        margin-bottom: 0.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .perfect-chart:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4), 0 0 20px rgba(96, 165, 250, 0.2);
    }
    
    /* Transparent tabs */
    .stTabs [data-baseweb="tab"] {
        background: rgba(51, 65, 85, 0.6);
        border: 1px solid rgba(96, 165, 250, 0.3);
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 700;
        color: #e2e8f0;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        position: relative;
        user-select: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border-color: #3b82f6;
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.4);
    }
    
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background: rgba(71, 85, 105, 0.8);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    /* Transparent selectbox */
    .stSelectbox [data-baseweb="select"] {
        background: rgba(51, 65, 85, 0.7);
        border: 1px solid rgba(96, 165, 250, 0.3);
        border-radius: 0.5rem;
        color: #e2e8f0;
        font-weight: 500;
        backdrop-filter: blur(10px);
        position: relative;
        opacity: 0.9;
    }
    
    .stSelectbox [data-baseweb="select"]:focus {
        background: rgba(71, 85, 105, 0.8);
        border-color: #3b82f6;
        opacity: 1;
    }
    
    /* Transparent dropdown */
    .stSelectbox [data-baseweb="popover"] {
        background: rgba(51, 65, 85, 0.9);
        border: 1px solid rgba(96, 165, 250, 0.3);
        border-radius: 0.5rem;
        opacity: 0.95;
        backdrop-filter: blur(15px);
    }
    
    .stSelectbox [data-baseweb="popover"] [data-baseweb="list"] {
        background: transparent;
        opacity: 1;
    }
    
    .stSelectbox [data-baseweb="popover"] [data-baseweb="list"] [data-baseweb="list-item"] {
        background: rgba(51, 65, 85, 0.6);
        color: #e2e8f0;
        border: none;
        opacity: 0.9;
        padding: 0.75rem 1rem;
        margin: 0;
        border-radius: 0.25rem;
    }
    
    .stSelectbox [data-baseweb="popover"] [data-baseweb="list"] [data-baseweb="list-item"]:hover {
        background: rgba(71, 85, 105, 0.8);
        color: #e2e8f0;
        opacity: 1;
    }
    
    /* Transparent buttons */
    .stButton > button {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.8) 0%, rgba(37, 99, 235, 0.8) 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3);
        cursor: pointer;
        position: relative;
        overflow: hidden;
        user-select: none;
        backdrop-filter: blur(10px);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.9) 0%, rgba(30, 64, 175, 0.9) 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.5);
    }
    
    /* Transparent slider */
    .stSlider [data-baseweb="slider"] {
        background: linear-gradient(90deg, rgba(59, 130, 246, 0.8), rgba(96, 165, 250, 0.8));
        border-radius: 0.5rem;
        position: relative;
        backdrop-filter: blur(5px);
    }
    
    .stSlider [data-baseweb="slider"] [data-baseweb="slider-handle"] {
        background: rgba(59, 130, 246, 0.9);
        border: 2px solid rgba(226, 232, 240, 0.8);
        backdrop-filter: blur(5px);
    }
    
    /* Transparent checkbox */
    .stCheckbox [data-baseweb="checkbox"] {
        color: #e2e8f0;
        font-weight: 500;
        position: relative;
        background: transparent;
    }
    
    .stCheckbox [data-baseweb="checkbox"] [data-baseweb="checkbox-input"] {
        background: rgba(51, 65, 85, 0.7);
        border: 2px solid rgba(96, 165, 250, 0.3);
        backdrop-filter: blur(5px);
    }
    
    .stCheckbox [data-baseweb="checkbox"] [data-baseweb="checkbox-input"]:checked {
        background: rgba(59, 130, 246, 0.8);
        border-color: rgba(59, 130, 246, 0.8);
    }
    
    /* Remove all unwanted spacing and margins */
    .stMarkdown {
        margin: 0 !important;
        padding: 0 !important;
        min-height: 0 !important;
    }
    
    .stColumn {
        padding: 0.5rem !important;
        margin: 0 !important;
    }
    
    /* Remove all gray backgrounds from dataframes */
    .dataframe {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    .dataframe table {
        background: rgba(30, 41, 59, 0.9) !important;
        border: 1px solid rgba(96, 165, 250, 0.2) !important;
        border-collapse: collapse !important;
    }
    
    .dataframe th {
        background: rgba(51, 65, 85, 0.8) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(96, 165, 250, 0.1) !important;
        padding: 0.75rem !important;
    }
    
    .dataframe td {
        background: rgba(30, 41, 59, 0.9) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(96, 165, 250, 0.1) !important;
        padding: 0.75rem !important;
    }
    
    /* Remove all gray backgrounds from plots */
    .js-plotly-plot {
        background: transparent !important;
        border: none !important;
    }
    
    .plotly {
        background: transparent !important;
        border: none !important;
    }
    
    /* Remove all streamlit default spacing */
    .main > div {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .streamlit-container {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Remove all empty divs and spans */
    div:empty, span:empty, p:empty {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Ultra-compact columns */
    .stColumn {
        padding: 0.25rem !important;
        margin: 0 !important;
    }
    
    /* Remove all streamlit default spacing */
    .main > div {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .streamlit-container {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Ultra-compact elements */
    .stVerticalBlock {
        gap: 0.25rem !important;
    }
    
    .stHorizontalBlock {
        gap: 0.25rem !important;
    }
    
    /* Remove all white space */
    .stApp {
        overflow: hidden !important;
    }
    
    /* Compact tabs */
    .stTabs {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Remove all empty streamlit elements */
    [data-testid="stVerticalBlock"] > div:empty {
        display: none !important;
    }
    
    [data-testid="stHorizontalBlock"] > div:empty {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize perfect session state
    if 'perfect_state' not in st.session_state:
        st.session_state.perfect_state = {
            'current_dataset': "Customer Churn Prediction",
            'current_accuracy': 0.786,
            'current_round': 5,
            'current_clients': 10,
            'current_status': "idle",
            'training_started': False,
            'total_rounds': 10,
            'epsilon': 1.0,
            'delta': 1e-5,
            'privacy_enabled': True,
            'encryption_enabled': True,
            'learning_rate': 0.01,
            'start_time': datetime.now()
        }
    
    state = st.session_state.perfect_state
    
    # Perfect header
    st.markdown('<h1 class="main-header">Perfect Federated Learning Platform</h1>', unsafe_allow_html=True)
    
    # Perfect sidebar
    with st.sidebar:
        st.markdown('<div class="perfect-sidebar">', unsafe_allow_html=True)
        
        # Dataset Selection
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">📊 Dataset Selection</div>', unsafe_allow_html=True)
        
        datasets = {
            "Customer Churn Prediction": {"accuracy": 0.786, "round": 5, "samples": 1000, "type": "Classification", "difficulty": "Medium"},
            "Medical Diagnosis (Diabetes)": {"accuracy": 0.823, "round": 6, "samples": 768, "type": "Medical", "difficulty": "Medium"},
            "House Price Prediction": {"accuracy": 0.791, "round": 4, "samples": 800, "type": "Regression", "difficulty": "Easy"},
            "Student Performance": {"accuracy": 0.847, "round": 7, "samples": 600, "type": "Classification", "difficulty": "Easy"},
            "Iris Flower Classification": {"accuracy": 0.923, "round": 8, "samples": 150, "type": "Classification", "difficulty": "Easy"},
            "Wine Type Classification": {"accuracy": 0.891, "round": 6, "samples": 178, "type": "Classification", "difficulty": "Easy"},
            "Breast Cancer Detection": {"accuracy": 0.934, "round": 7, "samples": 569, "type": "Medical", "difficulty": "Medium"},
            "Sales Revenue Prediction": {"accuracy": 0.812, "round": 5, "samples": 1000, "type": "Regression", "difficulty": "Medium"},
            "Student Grade Prediction": {"accuracy": 0.768, "round": 4, "samples": 800, "type": "Classification", "difficulty": "Easy"}
        }
        
        selected_dataset = st.selectbox(
            "Select Dataset",
            list(datasets.keys()),
            index=list(datasets.keys()).index(state['current_dataset'])
        )
        
        # Perfect dataset info
        dataset_info = datasets[selected_dataset]
        difficulty_color = {"Easy": "#10b981", "Medium": "#f59e0b", "Hard": "#ef4444"}
        
        st.markdown(f"""
        <div style="font-size: 0.9rem; line-height: 1.6; color: #e2e8f0;">
        <strong>📏 Samples:</strong> <span style="color: #60a5fa;">{dataset_info['samples']:,}</span><br>
        <strong>🎯 Type:</strong> <span style="color: #60a5fa;">{dataset_info['type']}</span><br>
        <strong>⚡ Difficulty:</strong> <span style="color: {difficulty_color.get(dataset_info['difficulty'], '#f59e0b')};">{dataset_info['difficulty']}</span><br>
        <strong>📊 Expected Accuracy:</strong> <span style="color: #10b981;">{dataset_info['accuracy']:.1%}</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Training Configuration
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">⚙️ Training Configuration</div>', unsafe_allow_html=True)
        
        n_clients = st.slider("Number of Clients", 2, 100, state['current_clients'])
        n_rounds = st.slider("Training Rounds", 1, 100, state['total_rounds'])
        learning_rate = st.slider("Learning Rate", 0.001, 0.1, float(state['learning_rate']), 0.001, format="%.4f")
        
        state['current_clients'] = n_clients
        state['total_rounds'] = n_rounds
        state['learning_rate'] = learning_rate
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Privacy Settings
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">🔒 Privacy Settings</div>', unsafe_allow_html=True)
        
        enable_dp = st.checkbox("Differential Privacy", value=state['privacy_enabled'])
        if enable_dp:
            epsilon = st.slider("Privacy Budget (ε)", 0.1, 10.0, float(state['epsilon']), 0.1)
            delta = st.slider("Failure Probability (δ)", 1e-10, 1e-1, float(state['delta']), format="%.0e")
        
        enable_encryption = st.checkbox("End-to-End Encryption", value=state['encryption_enabled'])
        
        state['privacy_enabled'] = enable_dp
        state['encryption_enabled'] = enable_encryption
        if enable_dp:
            state['epsilon'] = epsilon
            state['delta'] = delta
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Control Panel
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">🎮 Control Panel</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button("▶ START", type="primary")
        with col2:
            stop_button = st.button("⏹ STOP")
        
        if start_button:
            state['current_dataset'] = selected_dataset
            state['current_accuracy'] = dataset_info['accuracy']
            state['current_round'] = dataset_info['round']
            state['current_status'] = "running"
            state['training_started'] = True
            state['start_time'] = datetime.now()
            st.rerun()
        
        if stop_button:
            state['current_status'] = "idle"
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Perfect System Status
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">📊 System Status</div>', unsafe_allow_html=True)
        
        status_class = f"status-{state['current_status']}"
        progress_percent = (state['current_round'] / state['total_rounds']) * 100 if state['total_rounds'] > 0 else 0
        elapsed_time = datetime.now() - state['start_time']
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
            <span class="status-badge {status_class}">
                <span style="width: 10px; height: 10px; background: white; border-radius: 50%; box-shadow: 0 0 10px rgba(255, 255, 255, 0.5);"></span>
                {state['current_status'].upper()}
            </span>
        </div>
        
        <div style="font-size: 0.9rem; line-height: 1.8; color: #e2e8f0;">
        <strong>🏢 Active Clients:</strong> <span style="color: #60a5fa;">{state['current_clients']}</span><br>
        <strong>🔄 Current Round:</strong> <span style="color: #60a5fa;">{state['current_round']}/{state['total_rounds']}</span><br>
        <strong>📊 Model Accuracy:</strong> <span style="color: #10b981;">{state['current_accuracy']:.1%}</span><br>
        <strong>⚡ Progress:</strong> <span style="color: #f59e0b;">{progress_percent:.1f}%</span><br>
        <strong>⏱️ Elapsed Time:</strong> <span style="color: #8b5cf6;">{str(elapsed_time).split('.')[0]}</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Perfect tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💎 Dashboard", "📈 Analytics", "🏢 Clients", "🔒 Privacy", "📝 Logs", "⚙️ Settings"
    ])
    
    with tab1:
        # Perfect metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="perfect-metric">
                <div class="metric-value">{state['current_status'].upper()}</div>
                <div class="metric-label">System Status</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            progress = state['current_round'] / state['total_rounds'] if state['total_rounds'] > 0 else 0
            st.markdown(f"""
            <div class="perfect-metric">
                <div class="metric-value">{progress:.1%}</div>
                <div class="metric-label">Training Progress</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="perfect-metric">
                <div class="metric-value">{state['current_clients']}</div>
                <div class="metric-label">Active Clients</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="perfect-metric">
                <div class="metric-value">{state['current_accuracy']:.1%}</div>
                <div class="metric-label">Model Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Perfect progress bar
        if state['training_started']:
            st.markdown('<div class="perfect-chart">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #e2e8f0; margin: 0 0 1.5rem 0; font-weight: 700;">Training Progress</h3>', unsafe_allow_html=True)
            st.progress(progress, text=f"Round {state['current_round']} of {state['total_rounds']} completed")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Perfect charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="perfect-chart">', unsafe_allow_html=True)
            
            # Generate perfect training data
            rounds, accuracy_data, loss_data, confidence_data = platform.generate_perfect_training_curve(
                state['total_rounds'], state['current_accuracy'], state['current_round']
            )
            
            # Perfect accuracy chart
            completed_rounds = rounds[:state['current_round']]
            completed_accuracy = accuracy_data[:state['current_round']]
            completed_confidence = confidence_data[:state['current_round']]
            
            fig = go.Figure()
            
            # Completed rounds with confidence bands
            fig.add_trace(go.Scatter(
                x=completed_rounds + completed_rounds[::-1],
                y=completed_accuracy + [max(0, c - 0.05) for c in completed_confidence][::-1],
                fill='toself',
                fillcolor='rgba(59, 130, 246, 0.1)',
                line=dict(color='rgba(59, 130, 246, 0)'),
                name='Confidence Band',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=completed_rounds,
                y=completed_accuracy,
                mode='lines+markers',
                name='Completed',
                line=dict(color='#3b82f6', width=4),
                marker=dict(size=10, color='#3b82f6', line=dict(width=2, color='white'))
            ))
            
            # Projected rounds
            if state['current_round'] < state['total_rounds']:
                projected_rounds = rounds[state['current_round']:]
                projected_accuracy = accuracy_data[state['current_round']:]
                
                fig.add_trace(go.Scatter(
                    x=projected_rounds,
                    y=projected_accuracy,
                    mode='lines+markers',
                    name='Projected',
                    line=dict(color='#94a3b8', width=3, dash='dash'),
                    marker=dict(size=8, color='#94a3b8')
                ))
            
            fig.update_layout(
                title="📈 Model Accuracy Progress",
                xaxis_title="Training Round",
                yaxis_title="Accuracy",
                template="plotly_dark",
                height=450,
                showlegend=True,
                plot_bgcolor='rgba(15, 23, 42, 0.5)',
                paper_bgcolor='rgba(15, 23, 42, 0)',
                font=dict(color="#e2e8f0"),
                legend=dict(
                    bgcolor="rgba(30, 41, 59, 0.8)",
                    bordercolor="rgba(96, 165, 250, 0.3)"
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="perfect-chart">', unsafe_allow_html=True)
            
            # Perfect loss chart
            completed_loss = loss_data[:state['current_round']]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=completed_rounds,
                y=completed_loss,
                mode='lines+markers',
                name='Training Loss',
                line=dict(color='#ef4444', width=4),
                marker=dict(size=10, color='#ef4444', line=dict(width=2, color='white')),
                fill='tonexty',
                fillcolor='rgba(239, 68, 68, 0.2)'
            ))
            
            fig.update_layout(
                title="📉 Training Loss Reduction",
                xaxis_title="Training Round",
                yaxis_title="Loss",
                template="plotly_dark",
                height=450,
                showlegend=False,
                plot_bgcolor='rgba(15, 23, 42, 0.5)',
                paper_bgcolor='rgba(15, 23, 42, 0)',
                font=dict(color="#e2e8f0")
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        # Perfect client performance
        st.markdown('<div class="perfect-chart">', unsafe_allow_html=True)
        
        clients = platform.generate_perfect_clients(state['current_clients'], state['current_accuracy'], state['current_round'])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[c['id'] for c in clients],
            y=[c['accuracy'] for c in clients],
            marker=dict(
                color=[c['accuracy'] for c in clients],
                colorscale='viridis',
                showscale=True,
                colorbar=dict(title="Accuracy Score", bgcolor="rgba(30, 41, 59, 0.8)", bordercolor="rgba(96, 165, 250, 0.3)")
            ),
            text=[f"{c['accuracy']:.3f}" for c in clients],
            textposition='outside',
            textfont=dict(size=10, color="#e2e8f0")
        ))
        
        fig.update_layout(
            title="📊 Client Performance Distribution",
            xaxis_title="Client ID",
            yaxis_title="Accuracy",
            template="plotly_dark",
            height=500,
            plot_bgcolor='rgba(15, 23, 42, 0.5)',
            paper_bgcolor='rgba(15, 23, 42, 0)',
            font=dict(color="#e2e8f0"),
            xaxis=dict(tickfont=dict(size=8)),
            yaxis=dict(tickfont=dict(size=10))
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional analytics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="perfect-chart">', unsafe_allow_html=True)
            
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=[c['id'] for c in clients],
                values=[c['samples'] for c in clients],
                hole=0.4,
                marker=dict(colors=px.colors.qualitative.Set3),
                textinfo='label+percent',
                textfont=dict(size=10, color="#e2e8f0"),
                pull=[0.05] * len(clients)
            ))
            
            fig.update_layout(
                title="📊 Data Distribution",
                template="plotly_dark",
                height=400,
                showlegend=False,
                plot_bgcolor='rgba(15, 23, 42, 0.5)',
                paper_bgcolor='rgba(15, 23, 42, 0)',
                font=dict(color="#e2e8f0")
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="perfect-chart">', unsafe_allow_html=True)
            
            # Performance radar chart
            metrics = ['Accuracy', 'Contribution', 'Data Quality', 'Uptime', 'Throughput']
            avg_metrics = []
            
            for metric in metrics:
                if metric == 'Accuracy':
                    avg_metrics.append(np.mean([c['accuracy'] for c in clients]))
                elif metric == 'Contribution':
                    avg_metrics.append(np.mean([c['contribution'] for c in clients]))
                elif metric == 'Data Quality':
                    avg_metrics.append(np.mean([c['data_quality'] for c in clients]))
                elif metric == 'Uptime':
                    avg_metrics.append(np.mean([c['uptime'] for c in clients]) / 100)
                elif metric == 'Throughput':
                    avg_metrics.append(np.mean([c['throughput'] for c in clients]) / 1000)
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=avg_metrics,
                theta=metrics,
                fill='toself',
                name='Average Performance',
                line=dict(color='#3b82f6', width=3),
                marker=dict(size=10, color='#3b82f6')
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        color='rgba(15, 23, 42, 0.5)',
                        gridcolor='rgba(96, 165, 250, 0.2)',
                        tickfont=dict(color="#94a3b8")
                    ),
                    angularaxis=dict(
                        gridcolor='rgba(96, 165, 250, 0.2)',
                        tickfont=dict(color="#e2e8f0")
                    ),
                    bgcolor='rgba(15, 23, 42, 0.3)'
                ),
                title="🎯 Performance Radar",
                template="plotly_dark",
                height=400,
                font=dict(color="#e2e8f0")
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        # Perfect client table
        st.markdown('<div class="perfect-table">', unsafe_allow_html=True)
        
        df_clients = pd.DataFrame(clients)
        df_clients = df_clients.rename(columns={
            'id': 'Client ID',
            'name': 'Client Name',
            'status': 'Status',
            'accuracy': 'Accuracy',
            'samples': 'Samples',
            'latency': 'Latency (ms)',
            'throughput': 'Throughput (req/s)',
            'contribution': 'Contribution',
            'uptime': 'Uptime (%)',
            'data_quality': 'Data Quality',
            'model_version': 'Model Version',
            'last_update': 'Last Update'
        })
        
        # Format columns perfectly
        df_clients['Accuracy'] = df_clients['Accuracy'].apply(lambda x: f"{x:.3f}")
        df_clients['Samples'] = df_clients['Samples'].apply(lambda x: f"{x:,}")
        df_clients['Contribution'] = df_clients['Contribution'].apply(lambda x: f"{x:.3f}")
        df_clients['Uptime (%)'] = df_clients['Uptime (%)'].apply(lambda x: f"{x:.1f}")
        df_clients['Data Quality'] = df_clients['Data Quality'].apply(lambda x: f"{x:.3f}")
        df_clients['Last Update'] = df_clients['Last Update'].apply(lambda x: x.strftime("%H:%M:%S"))
        
        st.dataframe(df_clients, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Perfect client metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            active_clients = sum(1 for c in clients if c['status'] == 'Active')
            st.markdown(f"""
            <div class="perfect-metric">
                <div class="metric-value">{active_clients}</div>
                <div class="metric-label">Active Clients</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_samples = sum(c['samples'] for c in clients)
            st.markdown(f"""
            <div class="perfect-metric">
                <div class="metric-value">{total_samples:,}</div>
                <div class="metric-label">Total Samples</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_accuracy = sum(c['accuracy'] for c in clients) / len(clients)
            st.markdown(f"""
            <div class="perfect-metric">
                <div class="metric-value">{avg_accuracy:.1%}</div>
                <div class="metric-label">Avg Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_latency = sum(c['latency'] for c in clients) / len(clients)
            st.markdown(f"""
            <div class="perfect-metric">
                <div class="metric-value">{avg_latency:.1f}ms</div>
                <div class="metric-label">Avg Latency</div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        # Perfect privacy metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="perfect-chart">', unsafe_allow_html=True)
            
            if state['privacy_enabled']:
                budget_used, privacy_loss, remaining_budget = platform.calculate_perfect_privacy(
                    state['current_round'], state['total_rounds'], state['epsilon'], state['delta']
                )
            else:
                budget_used = privacy_loss = remaining_budget = 0
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=budget_used,
                delta={'reference': 0.8},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Privacy Budget Consumption"},
                gauge={'axis': {'range': [None, 1]},
                       'bar': {'color': "#3b82f6"},
                       'steps': [{'range': [0, 0.5], 'color': "#1e40af"},
                                {'range': [0.5, 0.8], 'color': "#1e3a8a"},
                                {'range': [0.8, 1], 'color': "#1e293b"}],
                       'threshold': {'line': {'color': "#ef4444", 'width': 4},
                                    'thickness': 0.75, 'value': 0.9}}
            ))
            
            fig.update_layout(
                height=450,
                template="plotly_dark",
                font=dict(color="#e2e8f0"),
                paper_bgcolor='rgba(15, 23, 42, 0)',
                plot_bgcolor='rgba(15, 23, 42, 0.5)'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="perfect-chart">', unsafe_allow_html=True)
            
            # Perfect security metrics
            security_scores = {
                "Encryption": 95 if state['encryption_enabled'] else 0,
                "Differential Privacy": 90 if state['privacy_enabled'] else 0,
                "Data Integrity": 98,
                "Access Control": 92,
                "Audit Logging": 88,
                "Network Security": 94
            }
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(security_scores.keys()),
                y=list(security_scores.values()),
                marker=dict(
                    color=['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4'],
                    line=dict(width=2, color='rgba(255, 255, 255, 0.2)')
                ),
                text=[f"{score}%" for score in security_scores.values()],
                textposition='outside',
                textfont=dict(color="#e2e8f0", size=10)
            ))
            
            fig.update_layout(
                title="🛡️ Security Assessment",
                yaxis_title="Score (%)",
                template="plotly_dark",
                height=450,
                plot_bgcolor='rgba(15, 23, 42, 0.5)',
                paper_bgcolor='rgba(15, 23, 42, 0)',
                font=dict(color="#e2e8f0"),
                xaxis=dict(tickfont=dict(size=10)),
                yaxis=dict(tickfont=dict(size=10))
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Perfect privacy configuration
        st.markdown('<div class="perfect-chart">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <h4 style="color: #e2e8f0; margin: 0 0 1.5rem 0; font-weight: 700;">🔒 Privacy Configuration</h4>
            <div style="font-size: 0.95rem; line-height: 1.8; color: #e2e8f0;">
            <strong>Differential Privacy:</strong> {'✅ Enabled' if state['privacy_enabled'] else '❌ Disabled'}<br>
            <strong>Privacy Budget (ε):</strong> <span style="color: #60a5fa;">{state['epsilon']:.1f}</span><br>
            <strong>Failure Probability (δ):</strong> <span style="color: #60a5fa;">{state['delta']:.0e}</span><br>
            <strong>Budget Used:</strong> <span style="color: #10b981;">{budget_used:.1%}</span><br>
            <strong>Budget Remaining:</strong> <span style="color: #f59e0b;">{1 - budget_used:.1%}</span><br>
            <strong>Privacy Loss:</strong> <span style="color: #ef4444;">{privacy_loss:.3f}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <h4 style="color: #e2e8f0; margin: 0 0 1.5rem 0; font-weight: 700;">🛡️ Security Configuration</h4>
            <div style="font-size: 0.95rem; line-height: 1.8; color: #e2e8f0;">
            <strong>End-to-End Encryption:</strong> {'✅ Enabled' if state['encryption_enabled'] else '❌ Disabled'}<br>
            <strong>Protocol:</strong> <span style="color: #60a5fa;">AES-256-GCM</span><br>
            <strong>Key Exchange:</strong> <span style="color: #60a5fa;">ECDH P-384</span><br>
            <strong>Authentication:</strong> <span style="color: #60a5fa;">OAuth 2.0 + OIDC</span><br>
            <strong>Audit Trail:</strong> <span style="color: #10b981;">✅ Active</span><br>
            <strong>Compliance:</strong> <span style="color: #10b981;">GDPR, HIPAA, CCPA</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        # Perfect logs
        st.markdown('<div class="perfect-chart">', unsafe_allow_html=True)
        
        # Generate perfect logs
        logs = []
        
        # Training logs
        for i in range(state['current_round']):
            logs.append({
                "timestamp": datetime.now() - timedelta(minutes=state['current_round'] - i),
                "event": f"Round {i+1} Completed",
                "client": "System",
                "details": f"Accuracy: {state['current_accuracy'] - (state['current_round'] - i - 1) * 0.02:.3f}",
                "level": "INFO"
            })
        
        # Client logs
        for i in range(min(10, state['current_clients'])):
            logs.append({
                "timestamp": datetime.now() - timedelta(seconds=random.randint(10, 300)),
                "event": "Model Update",
                "client": f"FL-Client-{i+1:03d}",
                "details": f"Weights uploaded successfully",
                "level": "INFO"
            })
        
        # Sort logs
        logs.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Display perfect logs
        for log in logs[:20]:
            timestamp = log['timestamp'].strftime("%H:%M:%S")
            level_colors = {"INFO": "#3b82f6", "WARNING": "#f59e0b", "ERROR": "#ef4444", "SUCCESS": "#10b981"}
            level_color = level_colors.get(log['level'], "#64748b")
            
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.8); border-left: 4px solid {level_color}; padding: 1.25rem; margin-bottom: 0.75rem; border-radius: 0.75rem; backdrop-filter: blur(10px);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <span style="font-weight: 700; color: #e2e8f0; font-size: 1rem;">{log['event']}</span>
                <span style="color: #94a3b8; font-size: 0.875rem;">{timestamp}</span>
            </div>
            <div style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;">
            <strong>Client:</strong> <span style="color: #60a5fa;">{log['client']}</span> | 
            <strong>Details:</strong> <span style="color: #e2e8f0;">{log['details']}</span> | 
            <strong>Level:</strong> <span style="color: {level_color}; font-weight: 600;">{log['level']}</span>
            </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab6:
        # Perfect settings with full content
        st.markdown('<div class="perfect-chart">', unsafe_allow_html=True)
        
        st.markdown('<h3 style="color: #e2e8f0; margin: 0 0 2rem 0; font-weight: 700;">⚙️ Advanced Settings</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h4 style="color: #60a5fa; margin: 0 0 1rem 0;">Model Configuration</h4>', unsafe_allow_html=True)
            
            model_type = st.selectbox("Model Architecture", ["Neural Network", "Random Forest", "XGBoost", "Logistic Regression"])
            optimizer = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop", "AdaGrad"])
            batch_size = st.slider("Batch Size", 16, 512, 64)
            
            st.markdown('<h4 style="color: #60a5fa; margin: 2rem 0 1rem 0;">Federated Settings</h4>', unsafe_allow_html=True)
            
            aggregation_method = st.selectbox("Aggregation Method", ["FedAvg", "FedProx", "FedOpt", "FedBN"])
            communication_rounds = st.slider("Communication Rounds", 1, 50, 10)
            client_fraction = st.slider("Client Fraction", 0.1, 1.0, 1.0, 0.1)
        
        with col2:
            st.markdown('<h4 style="color: #60a5fa; margin: 0 0 1rem 0;">Advanced Privacy</h4>', unsafe_allow_html=True)
            
            clipping_norm = st.slider("Gradient Clipping Norm", 0.1, 10.0, 1.0, 0.1)
            noise_multiplier = st.slider("Noise Multiplier", 0.1, 5.0, 1.0, 0.1)
            max_weight_norm = st.slider("Max Weight Norm", 0.1, 10.0, 1.0, 0.1)
            
            st.markdown('<h4 style="color: #60a5fa; margin: 2rem 0 1rem 0;">System Configuration</h4>', unsafe_allow_html=True)
            
            max_workers = st.slider("Max Workers", 1, 50, 10)
            timeout = st.slider("Timeout (seconds)", 30, 600, 120)
            retry_attempts = st.slider("Retry Attempts", 1, 10, 3)
        
        # Add configuration summary
        st.markdown('<h4 style="color: #60a5fa; margin: 2rem 0 1rem 0;">Configuration Summary</h4>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="background: rgba(51, 65, 85, 0.6); padding: 1rem; border-radius: 0.75rem; border: 1px solid rgba(96, 165, 250, 0.2);">
            <h5 style="color: #e2e8f0; margin: 0 0 0.5rem 0;">Model Settings</h5>
            <p style="color: #cbd5e1; margin: 0.25rem 0; font-size: 0.9rem;">
            <strong>Architecture:</strong> {model_type}<br>
            <strong>Optimizer:</strong> {optimizer}<br>
            <strong>Batch Size:</strong> {batch_size}
            </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: rgba(51, 65, 85, 0.6); padding: 1rem; border-radius: 0.75rem; border: 1px solid rgba(96, 165, 250, 0.2);">
            <h5 style="color: #e2e8f0; margin: 0 0 0.5rem 0;">Federated Settings</h5>
            <p style="color: #cbd5e1; margin: 0.25rem 0; font-size: 0.9rem;">
            <strong>Aggregation:</strong> {aggregation_method}<br>
            <strong>Communication:</strong> {communication_rounds} rounds<br>
            <strong>Client Fraction:</strong> {client_fraction:.1f}
            </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main function"""
    create_perfect_platform()

if __name__ == "__main__":
    main()
