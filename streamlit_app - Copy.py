import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(
    page_title="Azure VM Optimization Dashboard",
    layout="wide"
)

st.title("Cloud Workload Optimization Dashboard")

# ------------------------------------------------------------------
# Load Data
# ------------------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_parquet("C:\\Users\\adity\\Walsh\\MS_Capstone\\Git\\data\\processed\\hourly_n2000.parquet")

df = load_data()


# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.header("Controls")

vm_list = sorted(df["vm_id"].unique())

selected_vm = st.sidebar.selectbox(
    "Select VM",
    vm_list
)

page = st.sidebar.radio(
    "Module",
    [
        "Overview",
        "Forecasting (RQ1)",
        "Clustering Metrics (RQ3)",
        "Rightsizing (RQ5)"
    ]
)

vm_df = df[df["vm_id"] == selected_vm].copy()

# ======================================================
# OVERVIEW
# ======================================================

if page == "Overview":

    st.header("Environment Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "VM Count",
        len(df["vm_id"].unique())
    )

    col2.metric(
        "Avg CPU",
        f"{df['avg_cpu_mean'].mean():.2f}%"
    )

    col3.metric(
        "Peak CPU",
        f"{df['avg_cpu_max'].max():.2f}%"
    )

    col4.metric(
        "Records",
        f"{len(df):,}"
    )

    st.subheader("Average CPU Distribution")

    fig = px.histogram(
        df,
        x="avg_cpu_mean",
        nbins=30
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# RQ1 FORECASTING
# ======================================================

elif page == "Forecasting (RQ1)":

    st.header("RQ1 - CPU Forecasting")

    st.subheader(f"VM {selected_vm}")

    fig = px.line(
        vm_df,
        x="hour_index",
        y="avg_cpu_mean",
        labels={
            "hour_index": "Hour",
            "avg_cpu_mean": "Average CPU (%)"
        },
        title="Historical CPU Utilization"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Simple moving-average forecast for demo
    forecast_window = 24

    rolling_avg = (
        vm_df["avg_cpu_mean"]
        .rolling(window=forecast_window)
        .mean()
    )

    forecast_value = rolling_avg.iloc[-1]

    future_hours = list(
        range(
            vm_df["hour_index"].max() + 1,
            vm_df["hour_index"].max() + 25
        )
    )

    forecast_df = pd.DataFrame({
        "hour_index": future_hours,
        "forecast_cpu": forecast_value
    })

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=vm_df["hour_index"],
            y=vm_df["avg_cpu_mean"],
            mode="lines",
            name="Actual"
        )
    )

    fig2.add_trace(
        go.Scatter(
            x=forecast_df["hour_index"],
            y=forecast_df["forecast_cpu"],
            mode="lines",
            name="Forecast",
            line=dict(dash="dash")
        )
    )

    fig2.update_layout(
        title="Next 24 Hour CPU Forecast"
    )

    st.plotly_chart(fig2, use_container_width=True)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average CPU",
        f"{vm_df['avg_cpu_mean'].mean():.2f}%"
    )

    col2.metric(
        "Maximum CPU",
        f"{vm_df['avg_cpu_max'].max():.2f}%"
    )

    col3.metric(
        "Forecast CPU",
        f"{forecast_value:.2f}%"
    )

# ======================================================
# RQ3
# ======================================================

elif page == "Clustering Metrics (RQ3)":

    st.header("RQ3 - VM Workload Characteristics")

    feature_df = (
        df.groupby("vm_id")
        .agg({
            "avg_cpu_mean": "mean",
            "avg_cpu_max": "max",
            "avg_cpu_std": "mean"
        })
        .reset_index()
    )

    fig = px.scatter(
        feature_df,
        x="avg_cpu_mean",
        y="avg_cpu_std",
        color="avg_cpu_max",
        hover_data=["vm_id"],
        labels={
            "avg_cpu_mean": "Average CPU",
            "avg_cpu_std": "CPU Variability"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        """
        Interpretation:
        • Bottom-left = Idle VMs
        • Bottom-right = Consistently Busy VMs
        • Top-left = Bursty VMs
        • Top-right = Highly Variable Busy VMs
        """
    )

# ======================================================
# RQ5 RIGHTSIZING
# ======================================================

elif page == "Rightsizing (RQ5)":

    st.header("RQ5 - Rightsizing Recommendation")

    p95_cpu = vm_df["avg_cpu_max"].quantile(0.95)

    avg_cpu = vm_df["avg_cpu_mean"].mean()

    peak_cpu = vm_df["avg_cpu_max"].max()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average CPU",
        f"{avg_cpu:.2f}%"
    )

    col2.metric(
        "P95 CPU",
        f"{p95_cpu:.2f}%"
    )

    col3.metric(
        "Peak CPU",
        f"{peak_cpu:.2f}%"
    )

    if p95_cpu < 40:
        recommendation = "✅ Downsizing Recommended"
        savings = "High Potential Savings"

    elif p95_cpu < 70:
        recommendation = "✅ Keep Current Size"
        savings = "Optimally Sized"

    else:
        recommendation = "⚠ Consider Larger VM"
        savings = "Risk of Resource Saturation"

    st.success(recommendation)

    st.write(f"Assessment: {savings}")

    fig = px.box(
        vm_df,
        y="avg_cpu_mean",
        title="CPU Utilization Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# RAW DATA
# ======================================================

with st.expander("Show Raw Data"):
    st.dataframe(vm_df)