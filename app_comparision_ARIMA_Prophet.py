# app_comparison_ARIMA_Prophet_with_capacity.py
import streamlit as st
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

st.set_page_config(page_title="Server Storage Forecast", layout="wide")

# ================= Load CSV =================
@st.cache_data
def load_data():
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=["snapshot_date"])
        return df
    return None

df = load_data()

if df is not None:
    st.sidebar.header("Controls")

    # Forecast Model Choice
    model_choice = st.sidebar.selectbox("Select Model", ["Prophet", "ARIMA"])

    # Graph Choice
    chart_choice = st.sidebar.selectbox(
        "Select Graph Type", ["Line Chart", "Bar Chart"]
    )

    # Capacity Limit
    capacity_limit = st.sidebar.slider(
        "Set Capacity Limit (GB)", 
        min_value=0, 
        max_value=int(df["storage_used_gb"].max() * 2), 
        value=int(df["storage_used_gb"].max())
    )

    # Server Selection
    servers = df["server_name"].unique().tolist()
    server_choice = st.sidebar.multiselect("Select Servers", servers, default=servers[0])

    # Forecast Period
    forecast_periods = st.sidebar.slider(
        "Months to Forecast", min_value=1, max_value=36, value=12
    )

    # ================= Forecast Function =================
    def forecast_prophet(data, periods):
        df_train = data.rename(columns={"snapshot_date": "ds", "storage_used_gb": "y"})
        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=periods, freq="MS")
        forecast = m.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    def forecast_arima(data, periods):
        df_train = data.set_index("snapshot_date")["storage_used_gb"]
        model = ARIMA(df_train, order=(1, 1, 1))
        model_fit = model.fit()
        forecast_vals = model_fit.forecast(steps=periods)
        forecast_df = pd.DataFrame({
            "ds": pd.date_range(start=df_train.index[-1] + pd.offsets.MonthBegin(), periods=periods, freq="MS"),
            "yhat": forecast_vals
        })
        forecast_df["yhat_lower"] = forecast_df["yhat"] * 0.95
        forecast_df["yhat_upper"] = forecast_df["yhat"] * 1.05
        return forecast_df

    # ================= Plot Function =================
    def plot_forecast(original, forecast, server):
        fig = go.Figure()

        # Historical Data
        fig.add_trace(go.Scatter(
            x=original["snapshot_date"], y=original["storage_used_gb"],
            mode="lines+markers", name="Historical", line=dict(color="blue")
        ))

        # Forecast Data
        fig.add_trace(go.Scatter(
            x=forecast["ds"], y=forecast["yhat"],
            mode="lines+markers", name="Forecast", line=dict(color="green")
        ))

        # Capacity Limit Line
        fig.add_trace(go.Scatter(
            x=pd.concat([original["snapshot_date"], forecast["ds"]]),
            y=[capacity_limit] * (len(original) + len(forecast)),
            mode="lines", name="Capacity Limit", line=dict(color="red", dash="dash")
        ))

        # Highlight points above capacity
        breach_points = forecast[forecast["yhat"] > capacity_limit]
        if not breach_points.empty:
            fig.add_trace(go.Scatter(
                x=breach_points["ds"], y=breach_points["yhat"],
                mode="markers", name="Breach", marker=dict(color="red", size=10, symbol="circle")
            ))

        fig.update_layout(
            title=f"Forecast for {server}",
            xaxis_title="Date", yaxis_title="Storage Used (GB)",
            legend=dict(orientation="h")
        )

        return fig

    # ================= Main Loop =================
    if len(server_choice) == 1:
        server = server_choice[0]
        df_server = df[df["server_name"] == server].copy()

        if model_choice == "Prophet":
            forecast_df = forecast_prophet(df_server, forecast_periods)
        else:
            forecast_df = forecast_arima(df_server, forecast_periods)

        fig = plot_forecast(df_server, forecast_df, server)
        st.plotly_chart(fig, use_container_width=True)

    else:
        combined_fig = go.Figure()
        for server in server_choice:
            df_server = df[df["server_name"] == server].copy()

            if model_choice == "Prophet":
                forecast_df = forecast_prophet(df_server, forecast_periods)
            else:
                forecast_df = forecast_arima(df_server, forecast_periods)

            combined_fig.add_trace(go.Scatter(
                x=forecast_df["ds"], y=forecast_df["yhat"],
                mode="lines", name=f"{server} Forecast"
            ))

        combined_fig.add_trace(go.Scatter(
            x=pd.date_range(df["snapshot_date"].min(), df["snapshot_date"].max() + pd.DateOffset(months=forecast_periods)),
            y=[capacity_limit] * (len(pd.date_range(df["snapshot_date"].min(), df["snapshot_date"].max() + pd.DateOffset(months=forecast_periods)))),
            mode="lines", name="Capacity Limit", line=dict(color="red", dash="dash")
        ))

        st.plotly_chart(combined_fig, use_container_width=True)

else:
    st.warning("Please upload a CSV file to proceed.")
