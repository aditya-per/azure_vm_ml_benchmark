import streamlit as st
import pandas as pd
import numpy as np
import joblib

import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Cloud Workload Optimization Dashboard",
    page_icon="☁",
    layout="wide"
)

# ======================================================
# CONFIG
# ======================================================

DATA_PATH = "data/processed/hourly_n2000.parquet"

MODEL_PATH = "notebooks/saved_models/rq1_xgboost_model.pkl"
FEATURE_PATH = "notebooks/saved_models/rq1_feature_columns.pkl"

CLUSTER_FILE = "notebooks/saved_models/vm_cluster_labels.csv"
PROFILE_FILE = "notebooks/saved_models/cluster_profiles.csv"

# ======================================================
# RQ3 CLUSTER LABELS
# ======================================================

CLUSTER_NAMES = {
    0: "Idle / Very Low Utilization",
    1: "Low Utilization",
    2: "High Utilization",
    3: "Steady Moderate",
    4: "Bursty / Variable"
}

CLUSTER_DESCRIPTIONS = {
    0: "Most CPU remains idle. Strong candidate for rightsizing.",
    1: "Low average utilization with occasional workload activity.",
    2: "Sustained high utilization. Monitor capacity carefully.",
    3: "Stable predictable workloads with consistent behavior.",
    4: "Variable workloads with frequent spikes and bursts."
}

# ======================================================
# LOAD DATA
# ======================================================

@st.cache_data
def load_data():
    return pd.read_parquet(DATA_PATH)

@st.cache_data
def load_cluster_labels():
    return pd.read_csv(CLUSTER_FILE)

@st.cache_data
def load_cluster_profiles():
    return pd.read_csv(PROFILE_FILE)

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_PATH)
    return model, feature_cols

df = load_data()
cluster_df = load_cluster_labels()
profile_df = load_cluster_profiles()
xgb_model, feature_cols = load_model()

# ======================================================
# FEATURE ENGINEERING
# ======================================================

def create_features(data):

    data = data.copy().sort_values("hour_index")

    cpu = data["avg_cpu_mean"]

    for lag in [1,2,3,6,12,24,48,168]:
        data[f"lag_{lag}"] = cpu.shift(lag)

    for window in [3,6,12,24,168]:

        data[f"roll_mean_{window}"] = (
            cpu.shift(1).rolling(window).mean()
        )

        data[f"roll_std_{window}"] = (
            cpu.shift(1).rolling(window).std()
        )

        data[f"roll_min_{window}"] = (
            cpu.shift(1).rolling(window).min()
        )

        data[f"roll_max_{window}"] = (
            cpu.shift(1).rolling(window).max()
        )

    data["hour_sin"] = np.sin(
        2*np.pi*data["hour_of_day"]/24
    )

    data["hour_cos"] = np.cos(
        2*np.pi*data["hour_of_day"]/24
    )

    data["day_cycle_sin"] = np.sin(
        2*np.pi*data["day_of_cycle"]/7
    )

    data["day_cycle_cos"] = np.cos(
        2*np.pi*data["day_of_cycle"]/7
    )

    data["hourly_range"] = (
        data["avg_cpu_max"] -
        data["avg_cpu_min"]
    )

    data["peak_to_mean"] = (
        data["avg_cpu_max"] /
        (data["avg_cpu_mean"] + 1e-6)
    )

    return data


# ======================================================
# RECURSIVE FORECAST
# ======================================================

def forecast_future_cpu(
    history_df,
    model,
    feature_cols,
    horizon_hours
):

    history = history_df.copy()

    forecasts = []

    for _ in range(horizon_hours):

        feature_df = create_features(history)

        latest = feature_df.iloc[-1:]

        X = latest[feature_cols].fillna(0)

        pred = float(model.predict(X)[0])

        pred = np.clip(pred, 0, 100)

        next_hour = int(
            history["hour_index"].max() + 1
        )

        new_row = history.iloc[-1].copy()

        new_row["hour_index"] = next_hour
        new_row["avg_cpu_mean"] = pred
        new_row["avg_cpu_max"] = pred
        new_row["avg_cpu_min"] = pred
        new_row["avg_cpu_std"] = 0

        forecasts.append(
            {
                "hour_index": next_hour,
                "forecast_cpu": pred
            }
        )

        history = pd.concat(
            [
                history,
                pd.DataFrame([new_row])
            ],
            ignore_index=True
        )

    return pd.DataFrame(forecasts)

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("Controls")

selected_vm = st.sidebar.selectbox(
    "Select VM",
    sorted(df["vm_id"].unique())
)

page = st.sidebar.radio(
    "Dashboard Section",
    [
        "Overview",
        "Forecasting (RQ1)",
        "Workload Archetypes (RQ3)",
        "Rightsizing (RQ4)"
    ]
)

vm_df = df[
    df["vm_id"] == selected_vm
].copy()

# ======================================================
# OVERVIEW
# ======================================================

if page == "Overview":

    st.title("Cloud Workload Optimization Dashboard")

    total_vms = df["vm_id"].nunique()

    avg_cpu = df["avg_cpu_mean"].mean()

    peak_cpu = df["avg_cpu_max"].max()

    p95_cpu = df["avg_cpu_max"].quantile(0.95)

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "VM Count",
        total_vms
    )

    c2.metric(
        "Average CPU",
        f"{avg_cpu:.2f}%"
    )

    c3.metric(
        "Peak CPU",
        f"{peak_cpu:.2f}%"
    )

    c4.metric(
        "P95 CPU",
        f"{p95_cpu:.2f}%"
    )

    st.subheader("CPU Distribution")

    fig = px.histogram(
        df,
        x="avg_cpu_mean",
        nbins=40
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Cluster Distribution")

    cluster_counts = cluster_df["cluster"].value_counts().sort_index()

    cluster_plot = pd.DataFrame({
        "Cluster":[CLUSTER_NAMES[x] for x in cluster_counts.index],
        "Count":cluster_counts.values
    })

    fig2 = px.pie(
        cluster_plot,
        names="Cluster",
        values="Count"
    )

    st.plotly_chart(fig2, use_container_width=True)

# ======================================================
# RQ1
# ======================================================

elif page == "Forecasting (RQ1)":

    st.header("RQ1 - XGBoost CPU Forecasting")

    forecast_days = st.slider(
        "Forecast Horizon (Days)",
        1,
        15,
        7
    )

    history = (
        vm_df
        .sort_values("hour_index")
        .tail(360)
        .copy()
    )

    forecast_df = forecast_future_cpu(
        history,
        xgb_model,
        feature_cols,
        forecast_days * 24
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history["hour_index"],
            y=history["avg_cpu_mean"],
            name="Historical CPU"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_df["hour_index"],
            y=forecast_df["forecast_cpu"],
            name="Forecast",
            line=dict(
                color="red",
                dash="dash"
            )
        )
    )

    fig.update_layout(
        title=f"VM {selected_vm} CPU Forecast"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    forecast_peak = forecast_df["forecast_cpu"].max()
    forecast_mean = forecast_df["forecast_cpu"].mean()

    c1,c2,c3 = st.columns(3)

    c1.metric(
        "Forecast Peak CPU",
        f"{forecast_peak:.2f}%"
    )

    c2.metric(
        "Forecast Average CPU",
        f"{forecast_mean:.2f}%"
    )

    c3.metric(
        "Forecast Horizon",
        f"{forecast_days} Days"
    )

elif page == "Workload Archetypes (RQ3)":

    st.header("RQ3 - Workload Archetype")

    cluster_df["vm_id"] = (
        cluster_df["vm_id"].astype(int)
    )

    match = cluster_df[
        cluster_df["vm_id"] == selected_vm
    ]

    if match.empty:

        st.error(
            f"No cluster assigned to VM {selected_vm}"
        )

        st.stop()

    vm_cluster = int(
        match["cluster"].iloc[0]
    )

    st.success(
        f"Cluster {vm_cluster}: "
        f"{CLUSTER_NAMES[vm_cluster]}"
    )

    st.info(
        CLUSTER_DESCRIPTIONS[vm_cluster]
    )
    
# ======================================================
# RQ4 CAPACITY PLANNING / RIGHTSIZING
# ======================================================

elif page == "Capacity Planning (RQ4)":

    st.header("RQ4 - Capacity Planning Recommendation")

    # Current utilization metrics
    avg_cpu = vm_df["avg_cpu_mean"].mean()

    p95_cpu = vm_df["avg_cpu_max"].quantile(0.95)

    peak_cpu = vm_df["avg_cpu_max"].max()

    # Cluster lookup
    cluster_match = cluster_df[
        cluster_df["vm_id"] == int(selected_vm)
    ]

    if not cluster_match.empty:

        cluster_id = int(
            cluster_match["cluster"].iloc[0]
        )

        cluster_name = CLUSTER_NAMES.get(
            cluster_id,
            f"Cluster {cluster_id}"
        )

    else:

        cluster_name = "Unknown"

    # Forecast next 7 days
    history = (
        vm_df
        .sort_values("hour_index")
        .tail(360)
        .copy()
    )

    forecast_df = forecast_future_cpu(
        history,
        xgb_model,
        feature_cols,
        7 * 24
    )

    forecast_avg = (
        forecast_df["forecast_cpu"].mean()
    )

    forecast_peak = (
        forecast_df["forecast_cpu"].max()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Average CPU",
        f"{avg_cpu:.1f}%"
    )

    col2.metric(
        "P95 CPU",
        f"{p95_cpu:.1f}%"
    )

    col3.metric(
        "Forecast Mean CPU",
        f"{forecast_avg:.1f}%"
    )

    col4.metric(
        "Forecast Peak CPU",
        f"{forecast_peak:.1f}%"
    )

    st.subheader("Workload Archetype")

    st.info(
        f"{cluster_name}"
    )

    # Recommendation engine

    if forecast_peak < 40:

        recommendation = (
            "✅ Downsizing Recommended"
        )

        rationale = (
            "Forecasted utilization remains low."
        )

        priority = "High"

    elif forecast_peak < 70:

        recommendation = (
            "✅ Current VM Size is Appropriate"
        )

        rationale = (
            "Forecast remains within "
            "healthy utilization range."
        )

        priority = "Low"

    else:

        recommendation = (
            "⚠ Consider Upsizing"
        )

        rationale = (
            "Forecast indicates sustained "
            "high utilization."
        )

        priority = "High"

    st.success(recommendation)

    st.write(
        f"Reason: {rationale}"
    )

    st.write(
        f"Priority Level: {priority}"
    )

    # Forecast chart

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history["hour_index"],
            y=history["avg_cpu_mean"],
            name="Historical CPU"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_df["hour_index"],
            y=forecast_df["forecast_cpu"],
            name="Forecast",
            line=dict(
                color="red",
                dash="dash"
            )
        )
    )

    fig.update_layout(
        title="7-Day Capacity Forecast"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Executive summary

    st.subheader(
        "Executive Recommendation"
    )

    st.markdown(
        f"""
        **VM ID:** {selected_vm}

        **Workload Type:** {cluster_name}

        **Forecast Peak CPU:** {forecast_peak:.1f}%

        **Recommendation:** {recommendation}

        **Business Impact:** {rationale}
        """
    )