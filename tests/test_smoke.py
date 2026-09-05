"""Smoke tests for Privacy-First Federated Learning Pipeline."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import DiabetesDataLoader
from src.model import create_model, DiabetesMLP


def test_dataset_loads():
    loader = DiabetesDataLoader()
    df = loader.load_diabetes_data()
    assert len(df) == 768
    assert "Outcome" in df.columns


def test_data_preprocessing():
    loader = DiabetesDataLoader()
    df = loader.load_diabetes_data()
    X, y = loader.preprocess_data(df)
    assert X.shape[1] == 8
    assert len(y) == len(X)


def test_hospital_silos():
    loader = DiabetesDataLoader()
    df = loader.load_diabetes_data()
    X, y = loader.preprocess_data(df)
    silos = loader.create_data_silos(X, y, n_silos=3)
    assert "hospital_1" in silos
    assert "hospital_2" in silos
    assert "hospital_3" in silos
    assert "global_test" in silos


def test_model_forward():
    model = DiabetesMLP(input_dim=8)
    x = torch.randn(16, 8)
    out = model(x)
    assert out.shape == (16, 2)


def test_create_model_factory():
    mlp = create_model("mlp", input_dim=8)
    cnn = create_model("cnn", input_dim=8)
    assert mlp(torch.randn(4, 8)).shape == (4, 2)
    assert cnn(torch.randn(4, 8)).shape == (4, 2)


def test_dataloaders():
    loader = DiabetesDataLoader()
    df = loader.load_diabetes_data()
    X, y = loader.preprocess_data(df)
    silos = loader.create_data_silos(X, y, n_silos=3)
    loaders = loader.create_dataloaders(silos["hospital_1"], batch_size=16)
    batch = next(iter(loaders["train"]))
    assert batch[0].shape[1] == 8
