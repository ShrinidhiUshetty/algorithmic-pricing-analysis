from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------
# Configuration and data
# ------------------------------------------------------------
st.set_page_config(page_title="Algorithmic Pricing Analytics", page_icon="📊", layout="wide")
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_INTERVENTION = pd.Timestamp("2024-01-31")

@st.cache_data
def load_data():
    regional = pd.read_csv(DATA_DIR / "regional_sales_simulated.csv")
    synthetic = pd.read_csv(DATA_DIR / "synthetic_control_results.csv")
    regional["Date"] = pd.to_datetime(regional["Date"], errors="coerce")
    synthetic["Date"] = pd.to_datetime(synthetic["Date"], errors="coerce")
    regional = regional.dropna(subset=["Date", "Region", "Price", "Sales"]).copy()
    synthetic = synthetic.dropna(subset=["Date", "Bengaluru", "Synthetic_Bengaluru"]).copy()
    return regional, synthetic

try:
    df, sc = load_data()
except Exception as exc:
    st.error(f"Could not load project datasets: {exc}")
    st.stop()

required_regional = {"Date", "Region", "Price", "Sales"}
required_sc = {"Date", "Bengaluru", "Synthetic_Bengaluru"}
if not required_regional.issubset(df.columns) or not required_sc.issubset(sc.columns):
    st.error("The CSV schema does not match the expected project datasets.")
    st.stop()

regions = sorted(df["Region"].unique().tolist())
min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
if DEFAULT_INTERVENTION > df["Date"].max():
    DEFAULT_INTERVENTION = df["Date"].median().normalize()

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.title("📊 Pricing Analytics")
page = st.sidebar.radio(
    "Workspace",
    ["Executive Overview", "Exploratory Analysis", "Causal Impact",
     "Model Diagnostics", "Pricing Insights", "Data Explorer"]
)
st.sidebar.divider()
st.sidebar.caption("Global filters")
date_range = st.sidebar.date_input("Analysis period", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date)
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_date, end_date = pd.Timestamp(min_date), pd.Timestamp(max_date)
selected_regions = st.sidebar.multiselect("Regions", regions, default=regions)
filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date) &
              (df["Region"].isin(selected_regions))].copy()
intervention_date = st.sidebar.date_input(
    "Price intervention date", value=DEFAULT_INTERVENTION.date(),
    min_value=min_date, max_value=max_date
)
intervention_date = pd.Timestamp(intervention_date)
st.sidebar.caption("Intervention assumption: Bengaluru price reduction from ₹100 to ₹90.")
st.sidebar.caption("All project datasets are simulated.")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def metric_delta(current, previous, suffix=""):
    if pd.isna(previous) or previous == 0:
        return "n/a"
    return f"{(current / previous - 1) * 100:+.1f}%{suffix}"

def empty_warning(message):
    st.warning(message)

def make_monthly(data):
    if data.empty:
        return data
    return (data.assign(Month=data["Date"].dt.to_period("M").dt.to_timestamp())
            .groupby(["Month", "Region"], as_index=False)
            .agg(Sales=("Sales", "sum"), Avg_Price=("Price", "mean"),
                 Avg_Daily_Sales=("Sales", "mean")))

# Synthetic-control outcomes
sc_view = sc[(sc["Date"] >= start_date) & (sc["Date"] <= end_date)].copy()
sc_view["Effect"] = sc_view["Bengaluru"] - sc_view["Synthetic_Bengaluru"]
post_sc = sc_view[sc_view["Date"] >= intervention_date]
pre_sc = sc_view[sc_view["Date"] < intervention_date]

st.title("Algorithmic Pricing Analysis")
