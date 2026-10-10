"""Customer segmentation using K-Means on the Mall Customers dataset."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "Mall_Customers.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
FEATURE_COLUMNS = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
RANDOM_STATE = 42


def load_customer_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Download Mall_Customers.csv from the source linked in README.md "
            "and place it in the data folder."
        )
    customer_data = pd.read_csv(DATA_PATH)
    required_columns = set(FEATURE_COLUMNS + ["CustomerID"])
    if not required_columns.issubset(customer_data.columns):
        missing_columns = sorted(required_columns - set(customer_data.columns))
        raise ValueError(f"The CSV is missing required columns: {missing_columns}")
    return customer_data


def prepare_data(customer_data):
    customer_data = customer_data.drop_duplicates().copy()
    if customer_data[FEATURE_COLUMNS].isnull().any().any():
        customer_data[FEATURE_COLUMNS] = customer_data[FEATURE_COLUMNS].fillna(
            customer_data[FEATURE_COLUMNS].median()
        )
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(customer_data[FEATURE_COLUMNS])
    return customer_data, scaled_features


def save_eda_charts(customer_data):
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, column, title in zip(
        axes.flat[:3], FEATURE_COLUMNS,
        ["Age distribution", "Annual income distribution", "Spending score distribution"],
    ):
        sns.histplot(data=customer_data, x=column, kde=True, ax=axis, color="#4385a8")
        axis.set_title(title)
    sns.scatterplot(
        data=customer_data, x="Annual Income (k$)", y="Spending Score (1-100)",
        hue="Gender" if "Gender" in customer_data else None,
        ax=axes[1, 0], palette="Set1", s=50,
    )
    axes[1, 0].set_title("Annual income vs spending score")
    if "Gender" in customer_data:
        sns.countplot(
            data=customer_data, x="Gender", hue="Gender", legend=False,
            ax=axes[1, 1], palette="Set2",
        )
    else:
        axes[1, 1].text(0.5, 0.5, "Gender column is not available", ha="center")
    axes[1, 1].set_title("Customer count by gender")
    sns.heatmap(
        customer_data.select_dtypes(include="number").corr(), annot=True,
        cmap="coolwarm", center=0, ax=axes[1, 2], fmt=".2f",
    )
    axes[1, 2].set_title("Correlation between numeric columns")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "eda_overview.png", dpi=160)
    plt.close(figure)


def choose_cluster_count(scaled_features, row_count):
    tested_k = list(range(1, min(10, row_count - 1) + 1))
    inertia_values = []
    for cluster_count in tested_k:
        model = KMeans(n_clusters=cluster_count, random_state=RANDOM_STATE, n_init=10)
        model.fit(scaled_features)
        inertia_values.append(float(model.inertia_))

    silhouette_values = {}
    for cluster_count in range(2, min(10, row_count - 1) + 1):
        model = KMeans(n_clusters=cluster_count, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(scaled_features)
        silhouette_values[cluster_count] = float(silhouette_score(scaled_features, labels))

    # The strongest silhouette is a useful starting point; check the elbow too.
    selected_k = max(silhouette_values, key=silhouette_values.get)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(tested_k, inertia_values, marker="o", color="#4385a8")
    axes[0].set(title="Elbow method", xlabel="Number of clusters (K)", ylabel="Inertia")
    axes[0].set_xticks(tested_k)
    axes[1].plot(list(silhouette_values), list(silhouette_values.values()), marker="o", color="#d47a38")
    axes[1].set(title="Silhouette scores", xlabel="Number of clusters (K)", ylabel="Silhouette score")
    axes[1].set_xticks(list(silhouette_values))
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "cluster_selection.png", dpi=160)
    plt.close(figure)
    return selected_k, tested_k, inertia_values, silhouette_values


def describe_segments(summary):
    income_midpoint = summary["Annual Income (k$)"].median()
    spending_midpoint = summary["Spending Score (1-100)"].median()
    names = {}
    for cluster_id, row in summary.iterrows():
        high_income = row["Annual Income (k$)"] >= income_midpoint
        high_spending = row["Spending Score (1-100)"] >= spending_midpoint
        if high_income and high_spending:
            name = "High-income, high-spending"
        elif high_income:
            name = "High-income, lower-spending"
        elif high_spending:
            name = "Lower-income, higher-spending"
        else:
            name = "Lower-income, lower-spending"
        names[cluster_id] = name
    return names


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    customer_data = load_customer_data()
    print("First five rows:\n", customer_data.head(), "\n")
    print("Shape:", customer_data.shape)
    print("Columns:", customer_data.columns.tolist())
    print("Data types:\n", customer_data.dtypes)
    print("Missing values:\n", customer_data.isna().sum())
    print("Duplicate rows:", int(customer_data.duplicated().sum()))

    customer_data, scaled_features = prepare_data(customer_data)
    save_eda_charts(customer_data)
    selected_k, tested_k, inertia_values, silhouette_values = choose_cluster_count(
        scaled_features, len(customer_data)
    )
    final_model = KMeans(n_clusters=selected_k, random_state=RANDOM_STATE, n_init=10)
    customer_data["Cluster"] = final_model.fit_predict(scaled_features)

    summary = customer_data.groupby("Cluster").agg(
        Customer_Count=("CustomerID", "count"),
        Average_Age=("Age", "mean"),
        **{"Annual Income (k$)": ("Annual Income (k$)", "mean")},
        **{"Spending Score (1-100)": ("Spending Score (1-100)", "mean")},
    )
    segment_names = describe_segments(summary)
    summary.insert(0, "Segment", pd.Series(segment_names))
    summary = summary.round(1)
    customer_data["Segment"] = customer_data["Cluster"].map(segment_names)
    customer_data.to_csv(OUTPUT_DIR / "customers_with_clusters.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "segment_summary.csv")

    figure, axis = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=customer_data, x="Annual Income (k$)", y="Spending Score (1-100)",
        hue="Segment", palette="tab10", s=70, ax=axis,
    )
    centroids = customer_data.groupby("Cluster")[["Annual Income (k$)", "Spending Score (1-100)"]].mean()
    axis.scatter(centroids.iloc[:, 0], centroids.iloc[:, 1], marker="X", s=220,
                 c="black", edgecolor="white", linewidth=1, label="Cluster centres")
    axis.set_title(f"Customer segments (K={selected_k})")
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "customer_segments.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    results = {
        "selected_k": selected_k,
        "silhouette_scores": silhouette_values,
        "tested_k": tested_k,
        "inertia": inertia_values,
        "segment_names": {str(key): value for key, value in segment_names.items()},
    }
    (OUTPUT_DIR / "analysis_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSelected K: {selected_k}")
    print("Silhouette scores:", silhouette_values)
    print("\nSegment summary:\n", summary)
    print(f"\nCharts and CSV results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}")
