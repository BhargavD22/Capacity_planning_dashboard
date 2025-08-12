# app.py
import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

# =======================
# 1. Page Config
# =======================
st.set_page_config(
    page_title="SQL Server Capacity Planning Dashboard",
    page_icon="💾",
    layout="wide"
)

st.title("💾 SQL Server Capacity Planning Dashboard")
st.markdown("Monitor storage trends and forecast future usage for your on-prem SQL servers.")

# =======================
# 2. Load Data
# =======================
@st.cache_data
def load_data():
    df = pd.read_csv("sql_servers_storage_dummy.csv")
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df

df = load_data()

# =======================
# 3. Sidebar Controls
# =======================
servers = ["All Servers"] + sorted(df["server_name"].unique())
selected_server = st.sidebar.selectbox("Select a Server", servers)

forecast_years = st.sidebar.slider("Forecast Horizon (Years)", 1, 5, 2)
capacity_limit = st.sidebar.number_input("Capacity Limit (GB)", min_value=100, value=800)

# =======================
# 4. Single Server View & Breach Warning
# =======================
if selected_server != "All Servers":
    data_to_forecast = [selected_server]

    # Breach date calculation
    server_data = df[df["server_name"] == selected_server].copy()
    server_data = server_data.rename(columns={"snapshot_date": "ds", "storage_used_gb": "y"})

    m_temp = Prophet()
    m_temp.fit(server_data)
    future_temp = m_temp.make_future_dataframe(periods=forecast_years * 365)
    forecast_temp = m_temp.predict(future_temp)

    breach_rows = forecast_temp[forecast_temp["yhat"] >= capacity_limit]
    if not breach_rows.empty:
        breach_date = breach_rows.iloc[0]["ds"].date()
        st.markdown(
            f"<div style='background-color:#ffcccc;padding:10px;border-radius:5px;'>"
            f"🚨 <b>Capacity Breach Alert:</b> Forecast shows this server will exceed "
            f"<b>{capacity_limit} GB</b> on <b>{breach_date}</b>."
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='background-color:#ccffcc;padding:10px;border-radius:5px;'>"
            f"✅ <b>Safe:</b> No capacity breach expected within forecast horizon."
            f"</div>",
            unsafe_allow_html=True
        )

else:
    data_to_forecast = df["server_name"].unique()

# =======================
# 5. Forecast & Charts
# =======================
for server in data_to_forecast:
    st.subheader(f"📊 {server} - Storage Usage Forecast")

    server_df = df[df["server_name"] == server].copy()
    server_df = server_df.rename(columns={"snapshot_date": "ds", "storage_used_gb": "y"})

    m = Prophet()
    m.fit(server_df)
    future = m.make_future_dataframe(periods=forecast_years * 365)
    forecast = m.predict(future)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=server_df["ds"], y=server_df["y"], mode='lines', name='Historical Usage'))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode='lines', name='Forecast'))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=[capacity_limit]*len(forecast), 
                             mode='lines', name='Capacity Limit', line=dict(color='red', dash='dash')))
    fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Storage Used (GB)")
    st.plotly_chart(fig, use_container_width=True)
