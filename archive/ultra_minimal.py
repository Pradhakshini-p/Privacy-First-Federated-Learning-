#!/usr/bin/env python3
"""
Ultra-Minimal Dashboard
Just visuals, almost no text
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import random
import time

st.set_page_config(page_title="FL", page_icon="🔐", layout="wide")

# Hide streamlit elements
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Data
def get_data():
    return {
        'round': random.randint(1, 10),
        'accuracy': random.uniform(0.7, 0.95),
        'loss': random.uniform(0.1, 0.5),
        'clients': random.randint(2, 5),
        'privacy': random.uniform(0.0, 1.0),
        'training': random.choice([True, False])
    }

data = get_data()

# Top row - metrics
col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("", data['round'], "R")
with col2: st.metric("", f"{data['accuracy']:.2f}", "ACC")
with col3: st.metric("", f"{data['loss']:.2f}", "LOSS")
with col4: st.metric("", data['clients'], "C")
with col5: st.metric("", f"{data['privacy']:.1%}", "P")

# Status
col1, col2 = st.columns(2)

with col1:
    if data['training']:
        st.success("🟢")
    else:
        st.warning("⏸️")
    
    st.progress(data['round'] / 10)

with col2:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=data['privacy'] * 100,
        gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "darkblue"}}
    ))
    fig.update_layout(height=150, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

# Charts
col1, col2 = st.columns(2)

with col1:
    rounds = list(range(1, data['round'] + 1))
    acc = [0.7 + (i * 0.025) + random.uniform(-0.02, 0.02) for i in rounds]
    
    fig = px.line(x=rounds, y=acc, markers=True)
    fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    loss = [0.5 - (i * 0.04) + random.uniform(-0.02, 0.02) for i in rounds]
    
    fig = px.line(x=rounds, y=loss, markers=True, color_discrete_sequence=['red'])
    fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

# Controls
col1, col2, col3 = st.columns(3)
with col1: 
    if st.button("▶️", type="primary"): st.rerun()
with col2: 
    if st.button("⏹️"): st.rerun()
with col3: 
    if st.button("🔄"): st.rerun()

time.sleep(3)
st.rerun()
