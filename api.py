#!/usr/bin/env python3
"""
Minimal Flask API for Diabetes Prediction Inference
Provides REST endpoints for model predictions
"""

from flask import Flask, request, jsonify
import torch
import numpy as np
import pandas as pd
import logging
import os
from datetime import datetime

# Import our custom modules
from src.model import create_model
from src.data import DiabetesDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and data
model = None
device = None
scaler = None
feature_columns = None


def load_model_and_scaler():
    """Load the trained model and data scaler"""
    global model, device, scaler, feature_columns

    try:
        # Load model
        model = create_model(model_type="mlp", input_dim=8)
        model_path = "models/global_model.pth"

        # Try federated model first, fallback to centralized
        if not os.path.exists(model_path):
            model_path = "models/centralized_model.pth"
            logger.info("Using centralized model for inference")

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            device = torch.device("cpu")
            logger.info(f"✅ Model loaded from {model_path}")
        else:
            logger.error(f"❌ No model found. Please train a model first.")
            return False

        # Load scaler and feature columns from data
        data_loader = DiabetesDataLoader()
        df = data_loader.load_diabetes_data()
        X, y = data_loader.preprocess_data(df)
        scaler = data_loader.scaler
        feature_columns = data_loader.feature_columns

        logger.info("✅ Model and scaler loaded successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    })


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model metadata"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    return jsonify({
        "model_type": "mlp",
        "input_features": feature_columns,
        "num_features": len(feature_columns),
        "output_classes": 2,
        "class_names": ["No Diabetes", "Diabetes"],
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make prediction on patient data

    Expected JSON format:
    {
        "features": {
            "Pregnancies": 6,
            "Glucose": 148,
            "BloodPressure": 72,
            "SkinThickness": 35,
            "Insulin": 0,
            "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627,
            "Age": 50
        }
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.get_json()

        if not data or 'features' not in data:
            return jsonify({"error": "Missing 'features' in request body"}), 400

        features = data['features']

        # Convert to DataFrame with correct column order
        input_df = pd.DataFrame([features])

        # Ensure all required features are present
        missing_features = set(feature_columns) - set(input_df.columns)
        if missing_features:
            return jsonify({
                "error": f"Missing features: {missing_features}",
                "required_features": feature_columns
            }), 400

        # Reorder columns to match training data
        input_df = input_df[feature_columns]

        # Scale features
        input_scaled = scaler.transform(input_df)
        input_tensor = torch.FloatTensor(input_scaled)

        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][prediction].item()

        result = {
            "prediction": int(prediction),
            "prediction_label": "Diabetes" if prediction == 1 else "No Diabetes",
            "confidence": float(confidence),
            "probabilities": {
                "No Diabetes": float(probabilities[0][0]),
                "Diabetes": float(probabilities[0][1])
            },
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"Prediction: {result['prediction_label']} (confidence: {confidence:.4f})")

        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Make predictions on multiple patient records

    Expected JSON format:
    {
        "features": [
            {"Pregnancies": 6, "Glucose": 148, ...},
            {"Pregnancies": 1, "Glucose": 85, ...},
            ...
        ]
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.get_json()

        if not data or 'features' not in data:
            return jsonify({"error": "Missing 'features' in request body"}), 400

        features_list = data['features']

        # Convert to DataFrame
        input_df = pd.DataFrame(features_list)

        # Ensure all required features are present
        missing_features = set(feature_columns) - set(input_df.columns)
        if missing_features:
            return jsonify({
                "error": f"Missing features: {missing_features}",
                "required_features": feature_columns
            }), 400

        # Reorder columns
        input_df = input_df[feature_columns]

        # Scale features
        input_scaled = scaler.transform(input_df)
        input_tensor = torch.FloatTensor(input_scaled)

        # Make predictions
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1).tolist()
            confidences = probabilities.max(dim=1).values.tolist()

        results = []
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            results.append({
                "index": i,
                "prediction": int(pred),
                "prediction_label": "Diabetes" if pred == 1 else "No Diabetes",
                "confidence": float(conf),
                "probabilities": {
                    "No Diabetes": float(probabilities[i][0]),
                    "Diabetes": float(probabilities[i][1])
                }
            })

        return jsonify({
            "predictions": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": str(e)}), 500


def main():
    """Main function to start the API server"""
    import argparse

    parser = argparse.ArgumentParser(description="Start Flask inference API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Load model before starting server
    if not load_model_and_scaler():
        logger.error("Failed to load model. Exiting.")
        return

    logger.info("=" * 60)
    logger.info("🚀 Starting Flask Inference API")
    logger.info("=" * 60)
    logger.info(f"🌐 Server running at http://{args.host}:{args.port}")
    logger.info(f"📊 Endpoints:")
    logger.info(f"   GET  /health - Health check")
    logger.info(f"   GET  /model_info - Model metadata")
    logger.info(f"   POST /predict - Single prediction")
    logger.info(f"   POST /predict_batch - Batch predictions")
    logger.info("=" * 60)

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
