import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
import joblib
from .features import make_features
from .data_gen import generate_multi_shop_data

def train_all(csv_path="data/sales_multi.csv", models_dir="models"):
    """
    Train one RandomForest model per shop.
    Returns dictionary of shop -> saved model path.
    """
    # Regenerate data if missing or empty
    if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
        print("CSV missing/empty — regenerating data...")
        generate_multi_shop_data(save_path=csv_path)

    df = pd.read_csv(csv_path, parse_dates=["date"])  # read full multi-shop dataset

    # If CSV read empty, regenerate
    if df.empty:
        print("CSV empty after read — regenerating data...")
        generate_multi_shop_data(save_path=csv_path)
        df = pd.read_csv(csv_path, parse_dates=["date"])

    # Add date-based features
    df = make_features(df)

    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)

    shops = df["shop"].unique()
    models = {}

    # Train one model per shop
    for shop in shops:
        df_shop = df[df["shop"] == shop]
        if df_shop.empty:
            continue

        X = df_shop[["footfall", "advertising_spend", "events", "day", "month", "weekday"]]
        y = df_shop["sales"]

        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X, y)

        model_path = os.path.join(models_dir, f"model_{shop}.pkl")
        joblib.dump(model, model_path)
        models[shop] = model_path

    return models
