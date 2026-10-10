# Customer Segmentation with K-Means (Retail Dataset)

A beginner-friendly project that groups customers from the Mall Customers dataset by age, annual income, and spending score. It includes data checks, exploratory charts, K-Means evaluation, descriptive segment summaries, CSV exports, and a small Streamlit dashboard.

## Project structure

```text
customer-segmentation/
├── data/Mall_Customers.csv
├── outputs/                 # generated charts, summaries, and customer labels
├── app.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

`main.py` performs the analysis and writes generated files to `outputs/`. `app.py` reads the saved labeled CSV, so run the analysis once before starting the dashboard.

## Dataset

The included `data/Mall_Customers.csv` is the publicly shared Mall Customers teaching dataset. It contains `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, and `Spending Score (1-100)`. The copy used here is available in the [tirthajyoti Machine-Learning-with-Python repository](https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Datasets/Mall_Customers.csv). If the file is missing, download the CSV from that page and save it at `data/Mall_Customers.csv`. The project does not create substitute or synthetic customer records.

## Setup

From the project folder, create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can use the environment's Python directly with `.venv\Scripts\python.exe`.

## Run the analysis

```powershell
python main.py
```

The program prints a data overview and selected K, then saves:

- `outputs/eda_overview.png`: distributions, income/spending scatter, gender counts, and numeric correlation heatmap.
- `outputs/cluster_selection.png`: inertia curve and silhouette scores for candidate K values.
- `outputs/customer_segments.png`: colored customer groups and their centers on the income/spending axes.
- `outputs/customers_with_clusters.csv`: original customer records with `Cluster` and descriptive `Segment` columns.
- `outputs/segment_summary.csv`: customer count and average age, income, and spending score for each cluster.
- `outputs/analysis_results.json`: the K-selection values used in the run.

## Run the dashboard

After `main.py` succeeds, run:

```powershell
streamlit run app.py
```

Use the segment filter to explore groups. The table can be downloaded as CSV. Recommendations are starting points for experiments; validate them with campaign results.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository. Include the `data/Mall_Customers.csv` file and `requirements.txt`; do not upload `.venv`.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and connect your GitHub account.
3. Choose **Create app**, select the repository and branch, and set the app file to `app.py`. In advanced settings, select Python 3.13 to match the development environment.
4. Deploy. On its first launch, the app runs `main.py` to create the cluster results if they are not already present.

The repository and deployed dashboard are public. Do not add private data or credentials to the project.

## How the analysis works

1. Pandas loads the CSV and reports its first rows, shape, columns, types, missing values, and duplicate rows. Exact duplicate rows are removed. Missing numeric feature values, if any, are filled with their column median.
2. `CustomerID` is retained for identifying records but left out of clustering because an identifier does not describe shopping behaviour. The clustering uses age, annual income, and spending score.
3. `StandardScaler` puts these features on comparable scales. K-Means uses distances, so a feature with a larger numeric range could otherwise dominate the grouping.
4. The Elbow Method plots inertia for K from 1 to 10 (or fewer when the dataset is small). Inertia always falls as K increases, so look for where its drop starts to slow. Silhouette scores are computed for K from 2 to 10; higher scores indicate better separation in this feature space. The script uses the highest silhouette score as a reproducible starting choice. In a real analysis, inspect both plots and whether the groups are useful and understandable before settling on K.
5. K-Means is fitted with `random_state=42` and `n_init=10`. The cluster labels are attached to a copy of the cleaned dataset and exported.
6. Segment names are assigned from each cluster's average income and spending score relative to the cluster averages' medians. They describe observed patterns only. The clusters cannot establish customer motivations or predict future purchases.

The segment chart displays income and spending score in two dimensions, while clustering uses three features including age. It is a helpful view, but overlapping points on this chart can still differ by age and their three-dimensional distances.

## Results from the included dataset

The included CSV has 200 rows and no missing values or exact duplicate rows. With the current code and `random_state=42`, the best silhouette score among the tested values was at K=6 (about 0.428). The script chooses K by that score as a starting point; the elbow chart and practical usefulness should still be considered.

| Segment description | Customers | Average age | Average income (k$) | Average spending score |
|---|---:|---:|---:|---:|
| Lower-income, higher-spending | 45 | 56.3 | 54.3 | 49.1 |
| High-income, lower-spending | 39 | 26.8 | 57.1 | 48.1 |
| High-income, lower-spending | 33 | 41.9 | 88.9 | 17.0 |
| High-income, high-spending | 39 | 32.7 | 86.5 | 82.1 |
| Lower-income, higher-spending | 23 | 25.0 | 25.3 | 77.6 |
| Lower-income, lower-spending | 21 | 45.5 | 26.3 | 19.4 |

Names can repeat because clusters are named from income and spending levels, while age can further separate customers into different clusters. For example, the two high-income/lower-spending groups differ in their average income, spending score, and age. A business could test loyalty offers with the high-income/high-spending group, and test value-focused promotions with lower-income/lower-spending customers. These are hypotheses from summary data, not claims about individual motives; campaign response should be measured.

## Reading the charts

- The three histograms show how age, income, and spending score are distributed.
- The income-versus-spending chart helps reveal broad customer patterns, optionally separated by gender.
- The gender chart shows the row count by gender; it does not imply gender causes a spending pattern.
- The correlation heatmap summarizes linear relationships between numeric columns. Customer ID is present in the heatmap but should not be interpreted as a meaningful customer characteristic.
- The segment plot and summary table show how the selected groups differ on the measured features.

## Limitations

This is a small, commonly used teaching dataset, not a full transaction history. Its spending score is supplied by the dataset and is not a direct measure of future revenue. K-Means favors roughly compact groups, depends on the chosen features and K, and can change when data or preprocessing changes. Use the results to form testable marketing hypotheses, not as proof of why a person buys or what they will do next.

## Short project summary

This project applies K-Means clustering to group Mall Customers using age, annual income, and spending score. It checks and scales the data, compares cluster counts with inertia and silhouette score, visualizes the resulting groups, and exports cluster labels and summary statistics. A Streamlit dashboard makes the results easier to explore and download.

## Viva questions and simple answers

1. **What is clustering?** It is an unsupervised learning task that groups records by similarity without pre-existing target labels.
2. **How does K-Means work?** It assigns each record to a nearby center, recalculates centers as group averages, and repeats until assignments settle.
3. **Why scale the features?** K-Means uses distances, so scaling prevents a feature's larger units or range from overwhelming the others.
4. **Why exclude CustomerID?** It is an identifier rather than a measure of customer behaviour, so its numeric value would add meaningless distance.
5. **What does inertia measure?** The total squared distance between records and their assigned cluster centers; lower values mean tighter groups, but inertia decreases as K grows.
6. **What is the Elbow Method?** It plots inertia across K and looks for a bend where adding more groups gives smaller improvements.
7. **What does silhouette score tell us?** It compares how close a record is to its own group against other groups; higher average scores generally suggest better-separated clusters.
8. **Why not choose K from one score alone?** A score is one guide; interpretability, the elbow curve, and whether groups help the business also matter.
9. **What is a limitation of the cluster scatter plot?** It shows income and spending only, while this project's clustering also includes age.
10. **How can a business use these segments?** It can design small, measurable campaign tests for different observed groups and evaluate the results.
