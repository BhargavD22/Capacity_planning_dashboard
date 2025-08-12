import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# =======================
# 1. Load Data
# =======================
@st.cache_data
def load_data():
    # For GitHub hosting:
    # url = "https://raw.githubusercontent.com/<your-username>/<repo-name>/main/sql_servers_storage_dummy.csv"
    # return pd.read_csv(url, parse_dates=["snapshot_date"])

    # Local run:
    return pd.read_csv("sql_servers_storage_dummy.csv", parse_dates=["snapshot_date"])

df = load_data()

# =======================
# 2. Sidebar Controls
# =======================
st.sidebar.title("⚙️ Controls")
forecast_years = st.sidebar.slider("Forecast Horizon (Years)", 1, 5, 2)
capacity_limit = st.sidebar.number_input("Capacity Limit per Server (GB)", min_value=100, max_value=2000, value=1000)
show_confidence = st.sidebar.checkbox("Show Confidence Intervals", value=True)

# =======================
# 3. Forecast for all servers
# =======================
forecasts = []
fig = go.Figure()

for server in df["server_name"].unique():
    server_data = df[df["server_name"] == server].copy()
    server_data = server_data.rename(columns={"snapshot_date": "ds", "storage_used_gb": "y"})

    # Fit model
    m = Prophet()
    m.fit(server_data)

    # Forecast
    future = m.make_future_dataframe(periods=forecast_years * 365)
    forecast = m.predict(future)
    forecast["server_name"] = server
    forecasts.append(forecast)

    # Plot actual
    fig.add_trace(go.Scatter(
        x=server_data["ds"],
        y=server_data["y"],
        mode="lines",
        name=f"{server} Actual"
    ))

    # Plot forecast
    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat"],
        mode="lines",
        name=f"{server} Forecast"
    ))

    # Confidence intervals
    if show_confidence:
        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_upper"],
            mode="lines",
            line=dict(dash="dot"),
            name=f"{server} Upper Bound"
        ))
        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_lower"],
            mode="lines",
            line=dict(dash="dot"),
            name=f"{server} Lower Bound"
        ))

# =======================
# 4. Capacity Line
# =======================
fig.add_trace(go.Scatter(
    x=df["snapshot_date"],
    y=[capacity_limit] * len(df["snapshot_date"]),
    mode="lines",
    line=dict(color="red", dash="dash"),
    name="Capacity Limit"
))

fig.update_layout(
    title="📊 Multi-Server Storage Capacity Forecast",
    xaxis_title="Date",
    yaxis_title="Storage Used (GB)",
    template="plotly_white",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# =======================
# 5. Servers Approaching Capacity
# =======================
all_forecasts = pd.concat(forecasts)

# Filter only forecasted period (exclude historical)
future_forecasts = all_forecasts[all_forecasts["ds"] > df["snapshot_date"].max()]

# Check if any predicted usage exceeds capacity
alerts = future_forecasts.groupby("server_name").apply(
    lambda g: g[g["yhat"] >= capacity_limit].head(1)
).reset_index(drop=True)

st.subheader("⚠️ Servers Projected to Exceed Capacity")
if alerts.empty:
    st.success("No servers expected to exceed the capacity limit in the forecast horizon.")
else:
    st.dataframe(alerts[["server_name", "ds", "yhat"]].rename(columns={
        "server_name": "Server",
        "ds": "Date",
        "yhat": "Forecasted Usage (GB)"
    }))
