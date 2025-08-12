# capacity_planning_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==== Load Data ====
@st.cache_data
def load_data():
    df = pd.read_csv("sql_servers_storage_dummy.csv", parse_dates=["Date"])
    return df

df = load_data()

# ==== Streamlit UI ====
st.title("📊 SQL Server Capacity Planning Dashboard")

servers = df["Server"].unique()
selected_server = st.selectbox("Select Server", ["All Servers"] + list(servers))

graph_type = st.selectbox("Select Graph Type", ["Line", "Bar", "Area"])
selected_models = st.multiselect("Select Models to Display", ["Prophet", "ARIMA"], default=["Prophet", "ARIMA"])

# ==== Filter Data ====
if selected_server != "All Servers":
    data = df[df["Server"] == selected_server]
else:
    data = df.groupby("Date")["Storage_Used_GB"].sum().reset_index()
    data["Server"] = "All Servers"

# ==== Forecast Function ====
def prophet_forecast(data, periods=24):
    dfp = data.rename(columns={"Date": "ds", "Storage_Used_GB": "y"})
    model = Prophet()
    model.fit(dfp)
    future = model.make_future_dataframe(periods=periods, freq="MS")
    forecast = model.predict(future)
    return forecast[["ds", "yhat"]]

def arima_forecast(data, periods=24):
    series = data.set_index("Date")["Storage_Used_GB"]
    model = ARIMA(series, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    forecast_dates = pd.date_range(start=series.index[-1] + pd.DateOffset(months=1), periods=periods, freq="MS")
    return pd.DataFrame({"ds": forecast_dates, "yhat": forecast.values})

# ==== Forecast Models ====
historical = data.copy()

prophet_df = arima_df = None
if "Prophet" in selected_models:
    prophet_df = prophet_forecast(data)
if "ARIMA" in selected_models:
    arima_df = arima_forecast(data)

# ==== Plotting ====
fig = go.Figure()

# Historical usage
if graph_type == "Line":
    fig.add_trace(go.Scatter(x=historical["Date"], y=historical["Storage_Used_GB"], name="Historical Usage", line=dict(color="blue", width=2)))
elif graph_type == "Bar":
    fig.add_trace(go.Bar(x=historical["Date"], y=historical["Storage_Used_GB"], name="Historical Usage", marker_color="blue"))
elif graph_type == "Area":
    fig.add_trace(go.Scatter(x=historical["Date"], y=historical["Storage_Used_GB"], fill="tozeroy", name="Historical Usage", line=dict(color="blue")))

# Prophet forecast
if prophet_df is not None:
    if graph_type == "Line":
        fig.add_trace(go.Scatter(x=prophet_df["ds"], y=prophet_df["yhat"], name="Prophet Forecast", line=dict(color="green", width=2)))
    elif graph_type == "Bar":
        fig.add_trace(go.Bar(x=prophet_df["ds"], y=prophet_df["yhat"], name="Prophet Forecast", marker_color="green"))
    elif graph_type == "Area":
        fig.add_trace(go.Scatter(x=prophet_df["ds"], y=prophet_df["yhat"], fill="tozeroy", name="Prophet Forecast", line=dict(color="green")))

# ARIMA forecast
if arima_df is not None:
    if graph_type == "Line":
        fig.add_trace(go.Scatter(x=arima_df["ds"], y=arima_df["yhat"], name="ARIMA Forecast", line=dict(color="red", width=2)))
    elif graph_type == "Bar":
        fig.add_trace(go.Bar(x=arima_df["ds"], y=arima_df["yhat"], name="ARIMA Forecast", marker_color="red"))
    elif graph_type == "Area":
        fig.add_trace(go.Scatter(x=arima_df["ds"], y=arima_df["yhat"], fill="tozeroy", name="ARIMA Forecast", line=dict(color="red")))

# Capacity limit line (example: 1000 GB for All Servers, 500 GB for individual)
capacity_limit = 1000 if selected_server == "All Servers" else 500
fig.add_trace(go.Scatter(x=historical["Date"], y=[capacity_limit] * len(historical), name="Capacity Limit", line=dict(color="red", dash="dash")))

fig.update_layout(title=f"Storage Usage Forecast - {selected_server}", xaxis_title="Date", yaxis_title="Storage Used (GB)")

st.plotly_chart(fig, use_container_width=True)

# ==== Metrics Table ====
st.subheader("📈 Forecast Performance (on historical data)")
metrics = []
train = historical[:-12]  # last year as test
test = historical[-12:]

if "Prophet" in selected_models:
    prophet_hist = prophet_forecast(train, periods=12)
    mae = mean_absolute_error(test["Storage_Used_GB"], prophet_hist["yhat"][-12:])
    rmse = np.sqrt(mean_squared_error(test["Storage_Used_GB"], prophet_hist["yhat"][-12:]))
    mape = np.mean(np.abs((test["Storage_Used_GB"] - prophet_hist["yhat"][-12:]) / test["Storage_Used_GB"])) * 100
    metrics.append(["Prophet", mae, rmse, mape])

if "ARIMA" in selected_models:
    arima_hist = arima_forecast(train, periods=12)
    mae = mean_absolute_error(test["Storage_Used_GB"], arima_hist["yhat"])
    rmse = np.sqrt(mean_squared_error(test["Storage_Used_GB"], arima_hist["yhat"]))
    mape = np.mean(np.abs((test["Storage_Used_GB"] - arima_hist["yhat"]) / test["Storage_Used_GB"])) * 100
    metrics.append(["ARIMA", mae, rmse, mape])

if metrics:
    metrics_df = pd.DataFrame(metrics, columns=["Model", "MAE", "RMSE", "MAPE (%)"])
    st.dataframe(metrics_df)
