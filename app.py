import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="6G Smart Factory Analytics",
    layout="wide"
)

st.title("Manufacturing Process Health and Operational Efficiency Analysis")
st.subheader("6G-Enabled Smart Factory Diagnostic Dashboard")

# =========================
# LOAD DATA
# =========================


@st.cache_data
def load_data():
    df = pd.read_csv("Thales_Group_Manufacturing.csv")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Create datetime
    if "Date" in df.columns and "Timestamp" in df.columns:
        df["Datetime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Timestamp"].astype(str),
            errors="coerce"
        )
    elif "Date" in df.columns and "Time" in df.columns:
        df["Datetime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            errors="coerce"
        )

    df = df.drop_duplicates()
    df = df.dropna()

    return df


df = load_data()

# =========================
# SIDEBAR FILTERS
# =========================

st.sidebar.header("Dashboard Filters")

machines = st.sidebar.multiselect(
    "Select Machine",
    options=sorted(df["Machine_ID"].unique()),
    default=sorted(df["Machine_ID"].unique())[:5]
)

modes = st.sidebar.multiselect(
    "Select Operation Mode",
    options=sorted(df["Operation_Mode"].unique()),
    default=sorted(df["Operation_Mode"].unique())
)

filtered_df = df[
    (df["Machine_ID"].isin(machines)) &
    (df["Operation_Mode"].isin(modes))
]

# =========================
# KPI CALCULATIONS
# =========================

filtered_df["Machine_Health_Index"] = (
    0.35 * (100 - filtered_df["Temperature_C"]) +
    0.25 * (100 - filtered_df["Vibration_Hz"] * 10) +
    0.25 * (100 - filtered_df["Power_Consumption_kW"] * 5) +
    0.15 * filtered_df["Predictive_Maintenance_Score"]
)

filtered_df["Machine_Health_Index"] = filtered_df["Machine_Health_Index"].clip(
    0, 100)

filtered_df["Defect_Density_Score"] = (
    filtered_df["Quality_Control_Defect_Rate_%"] /
    filtered_df["Production_Speed_units_per_hr"]
) * 1000

filtered_df["Error_Frequency_Index"] = filtered_df["Error_Rate_%"]

avg_health = filtered_df["Machine_Health_Index"].mean()
avg_speed = filtered_df["Production_Speed_units_per_hr"].mean()
avg_defect = filtered_df["Quality_Control_Defect_Rate_%"].mean()
avg_error = filtered_df["Error_Rate_%"].mean()

# =========================
# FACTORY HEALTH OVERVIEW
# =========================

st.header("1. Factory Health Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Average Machine Health Index", f"{avg_health:.2f}")
col2.metric("Average Production Speed", f"{avg_speed:.2f} units/hr")
col3.metric("Average Defect Rate", f"{avg_defect:.2f}%")
col4.metric("Average Error Rate", f"{avg_error:.2f}%")

fig_eff = px.pie(
    filtered_df,
    names="Efficiency_Status",
    title="Efficiency Status Distribution"
)
st.plotly_chart(fig_eff, use_container_width=True)

# =========================
# SENSOR HEALTH ANALYSIS
# =========================

st.header("2. Machine-Level Sensor Health Analysis")

col1, col2 = st.columns(2)

fig_temp = px.box(
    filtered_df,
    x="Operation_Mode",
    y="Temperature_C",
    color="Efficiency_Status",
    title="Temperature by Operation Mode"
)
col1.plotly_chart(fig_temp, use_container_width=True)

fig_vib = px.box(
    filtered_df,
    x="Operation_Mode",
    y="Vibration_Hz",
    color="Efficiency_Status",
    title="Vibration by Operation Mode"
)
col2.plotly_chart(fig_vib, use_container_width=True)

fig_power = px.scatter(
    filtered_df,
    x="Power_Consumption_kW",
    y="Production_Speed_units_per_hr",
    color="Efficiency_Status",
    hover_data=["Machine_ID", "Operation_Mode"],
    title="Power Consumption vs Production Speed"
)
st.plotly_chart(fig_power, use_container_width=True)

# =========================
# MACHINE HEALTH SCORECARD
# =========================

st.header("3. Machine Health Scorecard")

machine_summary = filtered_df.groupby("Machine_ID").agg(
    Avg_Temperature=("Temperature_C", "mean"),
    Avg_Vibration=("Vibration_Hz", "mean"),
    Avg_Power=("Power_Consumption_kW", "mean"),
    Avg_Health_Index=("Machine_Health_Index", "mean"),
    Avg_Production_Speed=("Production_Speed_units_per_hr", "mean"),
    Avg_Defect_Rate=("Quality_Control_Defect_Rate_%", "mean"),
    Avg_Error_Rate=("Error_Rate_%", "mean")
).reset_index()

st.dataframe(machine_summary, use_container_width=True)

fig_health = px.bar(
    machine_summary,
    x="Machine_ID",
    y="Avg_Health_Index",
    title="Machine-wise Health Index",
    color="Avg_Health_Index"
)
st.plotly_chart(fig_health, use_container_width=True)

# =========================
# PRODUCTION & QUALITY PANEL
# =========================

st.header("4. Production and Quality Diagnostics")

col1, col2 = st.columns(2)

fig_prod_defect = px.scatter(
    filtered_df,
    x="Production_Speed_units_per_hr",
    y="Quality_Control_Defect_Rate_%",
    color="Efficiency_Status",
    hover_data=["Machine_ID", "Operation_Mode"],
    title="Production Speed vs Defect Rate"
)
col1.plotly_chart(fig_prod_defect, use_container_width=True)

fig_error = px.scatter(
    filtered_df,
    x="Vibration_Hz",
    y="Error_Rate_%",
    color="Efficiency_Status",
    hover_data=["Machine_ID", "Operation_Mode"],
    title="Vibration vs Error Rate"
)
col2.plotly_chart(fig_error, use_container_width=True)

fig_temp_defect = px.scatter(
    filtered_df,
    x="Temperature_C",
    y="Quality_Control_Defect_Rate_%",
    color="Efficiency_Status",
    hover_data=["Machine_ID", "Operation_Mode"],
    title="Temperature vs Defect Rate"
)
st.plotly_chart(fig_temp_defect, use_container_width=True)

# =========================
# NETWORK PERFORMANCE
# =========================

st.header("5. 6G Network Performance Analysis")

col1, col2 = st.columns(2)

fig_latency = px.box(
    filtered_df,
    x="Efficiency_Status",
    y="Network_Latency_ms",
    color="Efficiency_Status",
    title="Network Latency by Efficiency Status"
)
col1.plotly_chart(fig_latency, use_container_width=True)

fig_packet = px.box(
    filtered_df,
    x="Efficiency_Status",
    y="Packet_Loss_%",
    color="Efficiency_Status",
    title="Packet Loss by Efficiency Status"
)
col2.plotly_chart(fig_packet, use_container_width=True)

# =========================
# EFFICIENCY DIAGNOSTICS
# =========================

st.header("6. Efficiency Diagnostics View")

eff_machine = filtered_df.groupby(
    ["Machine_ID", "Efficiency_Status"]
).size().reset_index(name="Count")

fig_machine_eff = px.bar(
    eff_machine,
    x="Machine_ID",
    y="Count",
    color="Efficiency_Status",
    title="Machine-wise Efficiency Status Breakdown"
)
st.plotly_chart(fig_machine_eff, use_container_width=True)

eff_mode = filtered_df.groupby(
    ["Operation_Mode", "Efficiency_Status"]
).size().reset_index(name="Count")

fig_mode_eff = px.bar(
    eff_mode,
    x="Operation_Mode",
    y="Count",
    color="Efficiency_Status",
    title="Operation Mode vs Efficiency Status"
)
st.plotly_chart(fig_mode_eff, use_container_width=True)

# =========================
# CORRELATION HEATMAP
# =========================

st.header("7. Cross-Metric Correlation Analysis")

numeric_cols = [
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Quality_Control_Defect_Rate_%",
    "Production_Speed_units_per_hr",
    "Predictive_Maintenance_Score",
    "Error_Rate_%",
    "Machine_Health_Index",
    "Defect_Density_Score"
]

corr = filtered_df[numeric_cols].corr()

fig_corr = px.imshow(
    corr,
    text_auto=True,
    title="Correlation Heatmap of Smart Factory Metrics"
)
st.plotly_chart(fig_corr, use_container_width=True)

# =========================
# INSIGHTS & RECOMMENDATIONS
# =========================

st.header("8. Key Insights and Recommendations")

low_eff = filtered_df[filtered_df["Efficiency_Status"] == "Low"]

if not low_eff.empty:
    worst_machine = low_eff["Machine_ID"].value_counts().idxmax()
else:
    worst_machine = "No low-efficiency machine found"

st.markdown(f"""
### Key Findings

1. The dashboard shows the overall distribution of **High, Medium, and Low efficiency machines**.
2. Machine health is evaluated using temperature, vibration, power consumption, and predictive maintenance score.
3. Higher temperature and vibration patterns can indicate early degradation.
4. Power consumption is compared with production speed to identify inefficient machines.
5. Defect rate and error rate are analyzed to detect quality bottlenecks.
6. The machine most frequently appearing in low-efficiency status is **{worst_machine}**.

### Recommendations

- Monitor machines with low health index more frequently.
- Investigate machines with high vibration and high error rates.
- Reduce unnecessary power consumption in machines with low production speed.
- Prioritize maintenance for machines with poor predictive maintenance scores.
- Use 6G network latency and packet loss monitoring to improve real-time factory visibility.
""")

# =========================
# DOWNLOAD CLEANED DATA
# =========================

st.header("9. Download Processed Dataset")

csv = filtered_df.to_csv(index=False)

st.download_button(
    label="Download Filtered Processed Data",
    data=csv,
    file_name="processed_smart_factory_data.csv",
    mime="text/csv"
)
