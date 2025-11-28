import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_multi_shop_data(days=365, shops=None, save_path="data/sales_multi.csv"):

    # Default shops list
    if shops is None:
        shops = ["Clothing", "Electronics", "FoodCourt", "Shoes"]

    # Ensure folder exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Generate dates
    dates = [datetime.today() - timedelta(days=x) for x in range(days)]
    dates.reverse()

    rows = []

    # Generate data for each shop per day
    for d in dates:
        for shop in shops:

            # Base traffic by shop type
            base = {
                "Clothing": 800,
                "Electronics": 400,
                "FoodCourt": 1200,
                "Shoes": 600,
            }.get(shop, 500)

            footfall = np.random.randint(int(base * 0.5), int(base * 1.5))
            advertising = np.random.randint(2000, 50000)
            events = np.random.choice([0, 1], p=[0.9, 0.1])

            sales = int(
                footfall * np.random.uniform(30, 120)
                + advertising * np.random.uniform(0.5, 3.0)
                + events * np.random.randint(5000, 20000)
            )

            rows.append({
                "date": d,
                "shop": shop,
                "footfall": footfall,
                "advertising_spend": advertising,
                "events": events,
                "sales": sales,
            })

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)

    if df.empty:
        raise ValueError("Generated data is empty")

    return df
