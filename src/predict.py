from pathlib import Path
import joblib
import pandas as pd


# Load the trained model and threshold
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "fraud_model.pkl"

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
threshold = model_data["threshold"]


def predict_fraud(transaction):
    """
    Predict whether a transaction is fraudulent.
    """

    feature_order = [
        "Time",
        *[f"V{i}" for i in range(1, 29)],
        "Amount"
    ]

    df = pd.DataFrame([transaction])

    # Force the same feature order used during training
    df = df[feature_order]

    probability = model.predict_proba(df)[0, 1]

    prediction = int(probability >= threshold)

    return {
        "fraud_probability": probability,
        "prediction": prediction
    }