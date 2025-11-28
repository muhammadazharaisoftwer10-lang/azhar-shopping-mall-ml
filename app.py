import streamlit as st
import pandas as pd
import os
import sys
import plotly.express as px

# ------------------------
# Add src folder to Python path
# ------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import modules from src
from data_gen import generate_multi_shop_data
from features import make_features
from train import train_all
from model_utils import load_model, predict_sales

# ------------------------
# Config
# ------------------------
DATA_PATH = "data/sales_multi.csv"
MODELS_DIR = "models"
SHOPS = ["Clothing", "Electronics", "FoodCourt", "Shoes"]

# ------------------------
# Streamlit Page Config
# ------------------------
st.set_page_config(
    page_title="AZHAR SHOPPING MALL SUKKUR",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------
# Stylish Header
# ------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Pacifico&family=Roboto:wght@400;700&display=swap');

h1 {
    font-family: 'Pacifico', cursive;
    font-size: 64px;
    background: linear-gradient(to right, #ff4b4b, #ffa500, #1f77b4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    margin-bottom: 0;
}

h3 {
    font-family: 'Roboto', sans-serif;
    font-size: 22px;
    text-align: center;
    color: #555555;
    margin-top: 0;
}
</style>

<h1>🛍️ AZHAR SHOPPING MALL SUKKUR</h1>
<h3>Multi-Shop ML Dashboard</h3>
""", unsafe_allow_html=True)

# ------------------------
# Generate CSV if missing
# ------------------------
if (not os.path.exists(DATA_PATH)) or os.path.getsize(DATA_PATH) == 0:
    st.info("Generating multi-shop 1-year data...")
    generate_multi_shop_data(shops=SHOPS, save_path=DATA_PATH)

# ------------------------
# Train missing models
# ------------------------
missing_models = [s for s in SHOPS if not os.path.exists(os.path.join(MODELS_DIR, f"model_{s}.pkl"))]
if missing_models:
    st.info(f"Training models for: {', '.join(missing_models)} ...")
    train_all(csv_path=DATA_PATH, models_dir=MODELS_DIR)

# ------------------------
# Load data
# ------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, parse_dates=["date"])
    return df

df = load_data(DATA_PATH)

# ------------------------
# Sidebar filters & prediction
# ------------------------
st.sidebar.header("Filters & Prediction")
shop_filter = st.sidebar.multiselect("Select shops", SHOPS, default=SHOPS)
start_date, end_date = st.sidebar.date_input(
    "Date range",
    [df["date"].min().date(), df["date"].max().date()]
)

st.sidebar.subheader("Predict Sales (per shop)")
pred_shop = st.sidebar.selectbox("Shop", SHOPS)
pred_date = st.sidebar.date_input("Prediction Date", value=df["date"].max().date())
pred_footfall = st.sidebar.number_input("Footfall", min_value=0, value=1000)
pred_ad = st.sidebar.number_input("Advertising Spend", min_value=0, value=20000)
pred_event = st.sidebar.selectbox("Event?", [0, 1])

if st.sidebar.button("Predict"):
    try:
        model = load_model(pred_shop, models_dir=MODELS_DIR)
        pred_val = predict_sales(model, pred_date, pred_footfall, pred_ad, pred_event)
        st.sidebar.success(f"Predicted sales for {pred_shop} on {pred_date}: PKR {int(pred_val):,}")
    except Exception as e:
        st.sidebar.error(f"Prediction failed: {e}")

# ------------------------
# Main layout
# ------------------------
st.markdown("### Overview")
col1, col2 = st.columns([2, 1])

mask = (
    (df["date"] >= pd.to_datetime(start_date)) &
    (df["date"] <= pd.to_datetime(end_date)) &
    (df["shop"].isin(shop_filter))
)
df_view = df.loc[mask]

# ------------------------
# Left column: Sales & Footfall charts
# ------------------------
with col1:
    if df_view.empty:
        st.warning("No data for selected filters")
    else:
        agg = df_view.groupby("date").agg({"sales": "sum", "footfall": "sum"}).reset_index()
        
        fig_sales = px.line(
            agg,
            x="date",
            y="sales",
            title="🏷️ Total Sales Over Time",
            labels={"sales": "Total Sales (PKR)", "date": "Date"},
            template="plotly_dark",
            color_discrete_sequence=["#ff4b4b"]
        )
        st.plotly_chart(fig_sales, use_container_width=True)

        fig_footfall = px.bar(
            agg,
            x="date",
            y="footfall",
            title="👥 Total Footfall Over Time",
            labels={"footfall": "Footfall", "date": "Date"},
            template="plotly_dark",
            color_discrete_sequence=["#1f77b4"]
        )
        st.plotly_chart(fig_footfall, use_container_width=True)

# ------------------------
# Right column: Stylish KPIs
# ------------------------
with col2:
    if not df_view.empty:
        total_sales = int(df_view["sales"].sum())
        avg_daily = int(df_view.groupby("date")["sales"].sum().mean())
        total_footfall = int(df_view["footfall"].sum())
    else:
        total_sales = avg_daily = total_footfall = 0

    st.markdown(f"<h4 style='color:#ff4b4b;font-weight:bold;'>💰 Total Sales: PKR {total_sales:,}</h4>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color:#1f77b4;font-weight:bold;'>📅 Avg Daily Sales: PKR {avg_daily:,}</h4>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color:#ffa500;font-weight:bold;'>👥 Total Footfall: {total_footfall:,}</h4>", unsafe_allow_html=True)

st.markdown("---")

# ------------------------
# Per-shop breakdown
# ------------------------
st.markdown("### Per-Shop Breakdown")
shop_agg = df_view.groupby(["shop", "date"]).agg({"sales": "sum", "footfall": "sum"}).reset_index()

for shop in shop_filter:
    sdata = shop_agg[shop_agg["shop"] == shop]
    if sdata.empty:
        continue
    st.markdown(f"#### 🏬 {shop}")
    
    fig_shop_sales = px.line(
        sdata,
        x="date",
        y="sales",
        title=f"{shop} Sales Over Time",
        labels={"sales": "Sales (PKR)", "date": "Date"},
        template="plotly_dark",
        color_discrete_sequence=["#ff4b4b"]
    )
    st.plotly_chart(fig_shop_sales, use_container_width=True)
    
    fig_shop_footfall = px.bar(
        sdata,
        x="date",
        y="footfall",
        title=f"{shop} Footfall Over Time",
        labels={"footfall": "Footfall", "date": "Date"},
        template="plotly_dark",
        color_discrete_sequence=["#1f77b4"]
    )
    st.plotly_chart(fig_shop_footfall, use_container_width=True)

# ------------------------
# Raw data table
# ------------------------
with st.expander("Show Raw Data"):
    st.dataframe(df_view)

# ------------------------
# Footer
# ------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;font-size:14px;'>Made with ❤️ — AZHAR SHOPPING MALL SUKKUR</p>", unsafe_allow_html=True)
