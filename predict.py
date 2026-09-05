#!/usr/bin/env python3
"""Standalone CLI for diabetes prediction using trained models."""

import argparse
import json
import os
import sys

import pandas as pd
import torch

from src.data import DiabetesDataLoader
from src.model import create_model

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]


def load_model_and_scaler(model_type="mlp"):
    model_path = "models/global_model.pth"
    if not os.path.exists(model_path):
        model_path = "models/centralized_model.pth"

    if not os.path.exists(model_path):
        print("No trained model found. Run: python launch.py --mode full")
        sys.exit(1)

    model = create_model(model_type=model_type, input_dim=8)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    data_loader = DiabetesDataLoader()
    df = data_loader.load_diabetes_data()
    data_loader.preprocess_data(df)

    return model, data_loader.scaler, model_path


def predict_features(model, scaler, features: dict):
    input_df = pd.DataFrame([features])[FEATURES]
    input_scaled = scaler.transform(input_df)
    input_tensor = torch.FloatTensor(input_scaled)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][prediction].item()

    return {
        "prediction": int(prediction),
        "prediction_label": "Diabetes" if prediction == 1 else "No Diabetes",
        "confidence": float(confidence),
        "probabilities": {
            "No Diabetes": float(probabilities[0][0]),
            "Diabetes": float(probabilities[0][1]),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Predict diabetes risk from patient features")
    parser.add_argument("--json", type=str, help="JSON string or path to JSON file with features")
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "cnn"])
    args = parser.parse_args()

    model, scaler, model_path = load_model_and_scaler(args.model)
    print(f"Loaded model: {model_path}")

    if args.json:
        if os.path.isfile(args.json):
            with open(args.json, "r", encoding="utf-8") as f:
                features = json.load(f)
        else:
            features = json.loads(args.json)
    else:
        features = {
            "Pregnancies": 6,
            "Glucose": 148,
            "BloodPressure": 72,
            "SkinThickness": 35,
            "Insulin": 0,
            "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627,
            "Age": 50,
        }
        print("Using sample patient data (pass --json for custom input)")

    result = predict_features(model, scaler, features)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
