import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# ==============================
# Load Data
# ==============================
@st.cache_data
def load_data():
    #df = pd.read_csv("sql_servers_storage_dummy.csv", parse_dates=["snapshot_date"])
    df = pd.read_csv("sql_servers_storage_dummy.csv", parse_dates=["snapshot_date"])
    df = df.sort_values(by=["server_name", "snapshot_date"])
    return df

df = load_data()

# ==============================
# Sidebar Selections
# ==============================
st.sidebar.header("Configuration")

server_list = ["All Servers"] + sorted(df["server_name"].unique().tolist())
selected_server = st.sidebar.selectbox("Select Server", server_list)

model_choice = st.sidebar.multiselect(
    "Select Forecast Model(s)",
    ["Prophet", "ARIMA"],
    default=["Prophet"]
)

graph_type = st.sidebar.selectbox(
    "Select Graph Type",
    ["Line", "Bar", "Area"],
    index=0
)

forecast_periods = st.sidebar.slider(
    "Months to Forecast",
    min_value=1,
    max_value=36,
    value=24
)

# ==============================
# Data Prep
# ==============================
if selected_server != "All Servers":
    data = df[df["server_name"] == selected_server]
else:
    data = df.groupby("snapshot_date")["storage_used_gb"].sum().reset_index()
    data["server_name"] = "All Servers"

data = data.rename(columns={"snapshot_date": "ds", "storage_used_gb": "y"})

# ==============================
# Function to Calculate MAPE
# ==============================
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# ==============================
# Prophet Forecast
# ==============================
def run_prophet(df, periods):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='D')
    forecast = model.predict(future)
    return forecast

# ==============================
# ARIMA Forecast
# ==============================
def run_arima(df, periods):
    df_arima = df.set_index("ds")["y"]
    model = ARIMA(df_arima, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    forecast_df = pd.DataFrame({"ds": pd.date_range(start=df["ds"].max() + pd.Timedelta(days=1), periods=periods), "yhat": forecast})
    return forecast_df

# ==============================
# Run Selected Models
# ==============================
fig = go.Figure()
metrics_data = []

# Plot historical data
fig.add_trace(go.Scatter(
    x=data["ds"], y=data["y"],
    mode="lines+markers",
    name="Historical",
    line=dict(color="blue")
))

if "Prophet" in model_choice:
    forecast_prophet = run_prophet(data[["ds", "y"]], forecast_periods * 30)
    fig.add_trace(go.Scatter(
        x=forecast_prophet["ds"], y=forecast_prophet["yhat"],
        mode="lines",
        name="Prophet Forecast",
        line=dict(color="green", dash="solid")
    ))
    mae = mean_absolute_error(data["y"], forecast_prophet.loc[:len(data)-1, "yhat"])
    rmse = math.sqrt(mean_squared_error(data["y"], forecast_prophet.loc[:len(data)-1, "yhat"]))
    mape = mean_absolute_percentage_error(data["y"], forecast_prophet.loc[:len(data)-1, "yhat"])
    metrics_data.append(["Prophet", mae, rmse, mape])

if "ARIMA" in model_choice:
    forecast_arima = run_arima(data, forecast_periods * 30)
    fig.add_trace(go.Scatter(
        x=forecast_arima["ds"], y=forecast_arima["yhat"],
        mode="lines",
        name="ARIMA Forecast",
        line=dict(color="orange", dash="dot")
    ))
    y_true = data["y"][-len(forecast_arima):]
    y_pred = forecast_arima["yhat"][:len(y_true)]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    metrics_data.append(["ARIMA", mae, rmse, mape])

# ==============================
# Graph Type Control
# ==============================
if graph_type == "Bar":
    for trace in fig.data:
        trace.update(mode=None, type="bar")
elif graph_type == "Area":
    for trace in fig.data:
        trace.update(fill="tozeroy")

# ==============================
# Display
# ==============================
st.title("📊 SQL Server Storage Capacity Planning Dashboard")

st.plotly_chart(fig, use_container_width=True)

# Metrics Table
if metrics_data:
    metrics_df = pd.DataFrame(metrics_data, columns=["Model", "MAE", "RMSE", "MAPE"])
    st.subheader("📈 Model Performance Metrics")
    st.dataframe(metrics_df)

