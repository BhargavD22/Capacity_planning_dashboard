import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(page_title="Capacity Planning Dashboard", layout="wide")

st.title("📊 Capacity Planning Dashboard")
st.markdown("Upload your storage usage data and compare **Prophet** vs **ARIMA** forecasts.")

# -------------------------------
# File uploader
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"],
    help="CSV must have 'snapshot_date', 'server_name', 'storage_used_gb' columns"
)

@st.cache_data
def load_data(file):
    return pd.read_csv(file, parse_dates=["snapshot_date"])

if uploaded_file is not None:
    # Load Data
    df = load_data(uploaded_file)

    # -------------------------------
    # Sidebar filters
    # -------------------------------
    st.sidebar.header("⚙️ Configuration")

    server_list = ["All Servers"] + sorted(df["server_name"].unique())
    selected_server = st.sidebar.selectbox("Select Server", server_list)
    forecast_periods = st.sidebar.slider("Forecast Months", 1, 36, 12)
    capacity_limit = st.sidebar.slider("Set Capacity Limit (GB)", 500, 10000, 5000)

    # -------------------------------
    # Filter data
    # -------------------------------
    if selected_server != "All Servers":
        df_filtered = df[df["server_name"] == selected_server]
    else:
        df_filtered = df.groupby("snapshot_date", as_index=False)["storage_used_gb"].sum()
        df_filtered["server_name"] = "All Servers"

    # Prepare data
    df_filtered = df_filtered.rename(columns={"snapshot_date": "ds", "storage_used_gb": "y"})
    df_filtered = df_filtered.sort_values("ds")

    # -------------------------------
    # Prophet Forecast
    # -------------------------------
    prophet_model = Prophet()
    prophet_model.fit(df_filtered)
    future_prophet = prophet_model.make_future_dataframe(periods=forecast_periods, freq='M')
    forecast_prophet = prophet_model.predict(future_prophet)

    # -------------------------------
    # ARIMA Forecast
    # -------------------------------
    df_arima = df_filtered.set_index("ds")["y"]
    arima_model = ARIMA(df_arima, order=(1, 1, 1))
    arima_fit = arima_model.fit()
    forecast_values_arima = arima_fit.forecast(steps=forecast_periods)
    forecast_arima = pd.DataFrame({
        "ds": pd.date_range(start=df_filtered["ds"].max(), periods=forecast_periods + 1, freq='M')[1:],
        "yhat": forecast_values_arima
    })

    # -------------------------------
    # Combine for plotting
    # -------------------------------
    fig = go.Figure()

    # Actual data
    fig.add_trace(go.Scatter(
        x=df_filtered["ds"], y=df_filtered["y"],
        mode="lines+markers", name="Actual", line=dict(color="black")
    ))

    # Prophet forecast
    fig.add_trace(go.Scatter(
        x=forecast_prophet["ds"], y=forecast_prophet["yhat"],
        mode="lines", name="Prophet Forecast", line=dict(color="blue")
    ))

    # ARIMA forecast
    fig.add_trace(go.Scatter(
        x=forecast_arima["ds"], y=forecast_arima["yhat"],
        mode="lines", name="ARIMA Forecast", line=dict(color="orange")
    ))

    # Capacity line
    fig.add_trace(go.Scatter(
        x=pd.date_range(start=df_filtered["ds"].min(),
                        end=forecast_prophet["ds"].max(), freq="M"),
        y=[capacity_limit] * len(pd.date_range(start=df_filtered["ds"].min(),
                                               end=forecast_prophet["ds"].max(), freq="M")),
        mode="lines", name="Capacity Limit",
        line=dict(color="red", dash="dash")
    ))

    fig.update_layout(title=f"{selected_server} - Prophet vs ARIMA Forecast",
                      xaxis_title="Date", yaxis_title="Storage Used (GB)")

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # Error Metrics (on historical overlap)
    # -------------------------------
    # Prophet errors
    prophet_actual = df_filtered["y"].values
    prophet_pred = forecast_prophet.iloc[:len(df_filtered)]["yhat"].values
    prophet_mae = mean_absolute_error(prophet_actual, prophet_pred)
    prophet_rmse = np.sqrt(mean_squared_error(prophet_actual, prophet_pred))

    # ARIMA errors
    arima_actual = df_filtered["y"].values[-len(forecast_arima):] if len(forecast_arima) < len(df_filtered) else df_filtered["y"].values
    arima_pred = forecast_arima["yhat"].values[:len(arima_actual)]
    arima_mae = mean_absolute_error(arima_actual, arima_pred)
    arima_rmse = np.sqrt(mean_squared_error(arima_actual, arima_pred))

    st.subheader("📏 Model Performance (Historical)")
    st.write(f"**Prophet** → MAE: `{prophet_mae:.2f}`, RMSE: `{prophet_rmse:.2f}`")
    st.write(f"**ARIMA** → MAE: `{arima_mae:.2f}`, RMSE: `{arima_rmse:.2f}`")

else:
    st.info("Please upload a CSV file to proceed.")

