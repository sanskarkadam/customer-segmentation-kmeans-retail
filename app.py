"""A small Streamlit dashboard for exploring the saved K-Means results."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "Mall_Customers.csv"
RESULTS_PATH = PROJECT_DIR / "outputs" / "customers_with_clusters.csv"

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")
st.title("Customer Segmentation with K-Means")
st.write(
    "Explore groups formed from customer age, annual income, and spending score. "
    "These are descriptive patterns, not proof of customer motivation or future behaviour."
)

if not DATA_PATH.exists():
    st.error("Dataset not found. Add Mall_Customers.csv to the data folder; see README.md.")
    st.stop()
if not RESULTS_PATH.exists():
    with st.spinner("Finding customer groups for this dataset..."):
        try:
            from main import main as create_cluster_results

            create_cluster_results()
        except Exception as error:
            st.error(f"Could not prepare the customer groups: {error}")
            st.stop()

customer_data = pd.read_csv(RESULTS_PATH)
segment_summary = customer_data.groupby("Segment").agg(
    Customer_Count=("CustomerID", "count"),
    Average_Age=("Age", "mean"),
    Average_Income=("Annual Income (k$)", "mean"),
    Average_Spending_Score=("Spending Score (1-100)", "mean"),
).round(1)

st.subheader("Dataset overview")
metric_columns = st.columns(3)
metric_columns[0].metric("Customers", f"{len(customer_data):,}")
metric_columns[1].metric("Features used", "3")
metric_columns[2].metric("Segments", customer_data["Cluster"].nunique())
st.caption("The original dataset has five columns. CustomerID is kept for reference but excluded from clustering.")

selected_segments = st.multiselect(
    "Choose segments to display",
    options=sorted(customer_data["Segment"].unique()),
    default=sorted(customer_data["Segment"].unique()),
)
filtered_data = customer_data[customer_data["Segment"].isin(selected_segments)]

left_column, right_column = st.columns([1.5, 1])
with left_column:
    figure = px.scatter(
        filtered_data,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color="Segment",
        hover_data=["CustomerID", "Gender", "Age"],
        title="Annual income vs spending score",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    st.plotly_chart(figure, use_container_width=True)
with right_column:
    st.subheader("Segment statistics")
    st.dataframe(segment_summary.loc[segment_summary.index.isin(selected_segments)], use_container_width=True)
    st.subheader("Customer counts")
    st.bar_chart(segment_summary.loc[segment_summary.index.isin(selected_segments), "Customer_Count"])

st.subheader("Possible business actions")
for segment_name in selected_segments:
    row = segment_summary.loc[segment_name]
    if row["Average_Income"] >= customer_data["Annual Income (k$)"].median() and row["Average_Spending_Score"] >= customer_data["Spending Score (1-100)"].median():
        action = "Consider loyalty benefits or early access, then measure whether engagement changes."
    elif row["Average_Income"] >= customer_data["Annual Income (k$)"].median():
        action = "Test relevant product discovery or personalized offers; avoid assuming price sensitivity from this data alone."
    elif row["Average_Spending_Score"] >= customer_data["Spending Score (1-100)"].median():
        action = "Test affordable bundles or rewards and track response before expanding the offer."
    else:
        action = "Consider broad value messaging and convenient promotions, then compare campaign results."
    st.markdown(f"**{segment_name}** ({int(row['Customer_Count'])} customers): {action}")

st.subheader("Customer records")
st.dataframe(filtered_data, use_container_width=True, hide_index=True)
st.download_button(
    "Download filtered customer records (CSV)",
    data=filtered_data.to_csv(index=False).encode("utf-8"),
    file_name="customers_with_clusters.csv",
    mime="text/csv",
)
st.caption("The plot shows only income and spending. Clustering also uses age, so the two-dimensional view cannot show every distance or group boundary.")
