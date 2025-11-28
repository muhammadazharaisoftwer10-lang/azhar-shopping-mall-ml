import joblib
import os
import pandas as pd
from .features import make_features
from .train import train_all

def load_model(shop, models_dir="models"):
    """
    Load a pre-trained model for a given shop.
    If model is missing or corrupt, retrain all models automatically.
    """
    model_path = os.path.join(models_dir, f"model_{shop}.pkl")
    
    # If model missing or size zero, retrain
    if (not os.path.exists(model_path)) or os.path.getsize(model_path) == 0:
        print(f"Model for {shop} missing or corrupt. Retraining all models...")
        train_all(models_dir=models_dir)  # trains all shops
        if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
            raise FileNotFoundError(f"Model for {shop} could not be created after training.")

    # Try loading, if fails due to EOFError, retrain
    try:
        return joblib.load(model_path)
    except EOFError:
        print(f"Model file {model_path} corrupt. Retraining...")
        train_all(models_dir=models_dir)
        return joblib.load(model_path)

def predict_sales(model, date, footfall, advertising_spend, events):
    """
    Predict sales for a single row.
    
    Parameters:
    - model: trained sklearn model
    - date: datetime or str
    - footfall: int
    - advertising_spend: int
    - events: 0 or 1
    """
    df = pd.DataFrame({
        "date": [pd.to_datetime(date)],
        "footfall": [int(footfall)],
        "advertising_spend": [int(advertising_spend)],
        "events": [int(events)],
    })
    
    # Add date features
    df = make_features(df)
    
    # Select features for prediction
    X = df[["footfall", "advertising_spend", "events", "day", "month", "weekday"]]
    
    # Predict
    pred = model.predict(X)[0]
    return float(round(pred, 2))
