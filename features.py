import pandas as pd

def make_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["weekday"] = df["date"].dt.weekday
    return df
