# app_capacity_planning_compare.py
import streamlit as st
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# =======================
# Page Config
# =======================
st.set_page_config(
    page_title="SQL Server Capacity Planning",
    page_icon="💾",
    layout="wide"
)

st.title("💾 SQL Server Capacity Planning Dashboard")
st.markdown("Compare **Prophet** and **ARIMA** forecasting models for SQL Server storage usage with accuracy metrics.")

# =======================
# Load Data
# =======================
@st.cache_data
def load_data():
    df = pd.read_csv("sql_servers_storage_dummy.csv")
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df

df = load_data()

# =======================
# Sidebar Controls
# =======================
servers = sorted(df["server_name"].unique())
selected_server = st.sidebar.selectbox("Select a Server", servers)

forecast_years = st.sidebar.slider("Forecast Horizon (Years)", 1, 5, 2)
capacity_limit = st.sidebar.number_input("Capacity Limit (GB)", min_value=100, value=800)
validation_months = st.sidebar.slider("Validation Period (Months)", 1, 12, 3)

# =======================
# Prophet Forecast Function
# =======================
def forecast_prophet(server_df, forecast_days=365):
    prophet_df = server_df.rename(columns={"snapshot_date": "ds", "storage_used_gb": "y"})
    model = Prophet()
    model.fit(prophet_df[["ds", "y"]])
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    return forecast[["ds", "yhat"]]

# =======================
# ARIMA Forecast Function
# =======================
def forecast_arima(server_df, forecast_days=365):
    ts = server_df.set_index("snapshot_date")["storage_used_gb"]
    model = ARIMA(ts, order=(2, 1, 2))
    model_fit = model.fit()
    forecast_result = model_fit.get_forecast(steps=forecast_days)
    forecast_mean = forecast_result.predicted_mean
    forecast_index = pd.date_range(ts.index[-1] + timedelta(days=1), periods=forecast_days)
    return pd.DataFrame({"ds": forecast_index, "yhat": forecast_mean.values})

# =======================
# Accuracy Calculation
# =======================
def calculate_accuracy(train_df, test_df, model_func):
    forecast_days = len(test_df)
    forecast_df = model_func(train_df, forecast_days)
    forecast_df = forecast_df.set_index("ds").loc[test_df["snapshot_date"]]
    y_true = test_df["storage_used_gb"].values
    y_pred = forecast_df["yhat"].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape

# =======================
# Main Logic
# =======================
server_df = df[df["server_name"] == selected_server].copy()
server_df.sort_values("snapshot_date", inplace=True)

# Split into train/test for validation
validation_days = validation_months * 30
train_df = server_df.iloc[:-validation_days]
test_df = server_df.iloc[-validation_days:]

# Calculate accuracy
prophet_mae, prophet_rmse, prophet_mape = calculate_accuracy(train_df, test_df, forecast_prophet)
arima_mae, arima_rmse, arima_mape = calculate_accuracy(train_df, test_df, forecast_arima)

# Forecast for future
forecast_days = forecast_years * 365
prophet_forecast = forecast_prophet(server_df, forecast_days)
arima_forecast = forecast_arima(server_df, forecast_days)

# =======================
# Display Accuracy Table
# =======================
st.subheader("📏 Model Accuracy (Validation Period)")
accuracy_df = pd.DataFrame({
    "Model": ["Prophet", "ARIMA"],
    "MAE": [prophet_mae, arima_mae],
    "RMSE": [prophet_rmse, arima_rmse],
    "MAPE (%)": [prophet_mape, arima_mape]
})
st.table(accuracy_df)

# =======================
# Capacity Breach Check
# =======================
for model_name, forecast_df in [("Prophet", prophet_forecast), ("ARIMA", arima_forecast)]:
    breach_rows = forecast_df[forecast_df["yhat"] >= capacity_limit]
    if not breach_rows.empty:
        breach_date = breach_rows.iloc[0]["ds"].date()
        st.markdown(
            f"<div style='background-color:#ffcccc;padding:10px;border-radius:5px;'>"
            f"🚨 [{model_name}] Capacity Breach Alert: Expected to exceed {capacity_limit} GB on {breach_date}."
            f"</div>",
            unsafe_allow_html=True
        )

# =======================
# Plot Comparison
# =======================
st.subheader(f"📊 Forecast Comparison for {selected_server}")
fig = go.Figure()

# Historical
fig.add_trace(go.Scatter(x=server_df["snapshot_date"], y=server_df["storage_used_gb"],
                         mode='lines', name='Historical Usage'))

# Prophet forecast
fig.add_trace(go.Scatter(x=prophet_forecast["ds"], y=prophet_forecast["yhat"],
                         mode='lines', name='Prophet Forecast'))

# ARIMA forecast
fig.add_trace(go.Scatter(x=arima_forecast["ds"], y=arima_forecast["yhat"],
                         mode='lines', name='ARIMA Forecast'))

# Capacity limit line
fig.add_trace(go.Scatter(x=pd.concat([server_df["snapshot_date"], prophet_forecast["ds"]]),
                         y=[capacity_limit] * (len(server_df) + len(prophet_forecast)),
                         mode='lines', name='Capacity Limit',
                         line=dict(color='red', dash='dash')))

fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Storage Used (GB)")
st.plotly_chart(fig, use_container_width=True)
