# Customer Segmentation with K-Means

A small Python project that groups customers from the Mall Customers dataset by age, annual income, and spending score. It includes exploratory charts, K-Means evaluation, segment summaries, and a Streamlit dashboard.

## Project files

```text
├── data/
│   └── Mall_Customers.csv
├── outputs/                 # charts and result files created by main.py
├── app.py                   # Streamlit dashboard
├── main.py                  # data checks and K-Means analysis
├── requirements.txt
└── README.md
```

The source CSV is the publicly shared teaching dataset available in the [tirthajyoti Machine-Learning-with-Python repository](https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Datasets/Mall_Customers.csv). It contains 200 records and the columns `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, and `Spending Score (1-100)`. The project uses the real CSV and does not generate replacement data.

## Run on Windows

Open Command Prompt in this folder and enter each command separately:

```cmd
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python main.py
```

To open the dashboard after the analysis completes:

```cmd
streamlit run app.py
```

The browser address will appear in the terminal, usually `http://localhost:8501`. In PowerShell, activate the environment with `.venv\Scripts\Activate.ps1` instead of the `call` command.

If you move the project to another folder, recreate the virtual environment there. Windows virtual environments can refer to their original folder. For example:

```cmd
python -m venv --clear .venv
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

## What the analysis does

1. Loads the CSV and shows its first rows, shape, columns, data types, missing values, and duplicate count.
2. Removes exact duplicate rows and fills missing values in the clustering features with their column median.
3. Keeps `CustomerID` for identifying customers but leaves it out of clustering. The features are age, annual income, and spending score.
4. Scales those features with `StandardScaler`. K-Means compares distances, so scaling keeps a feature with larger numbers from dominating the result just because of its units.
5. Compares K values from 1 to 10 using inertia and from 2 to 10 using silhouette score. The script selects the K with the highest silhouette score as a starting point; the elbow curve and usefulness of the groups should also be considered.
6. Fits K-Means with `random_state=42` and `n_init=10`, adds the resulting labels, and summarizes each group.

## Results from the included CSV

The included file has 200 customers, no missing values, and no exact duplicate rows. With the current code, the highest silhouette score in the tested range is **0.4284 at K=6**. The script uses six clusters for this dataset. A silhouette score is a comparison aid, not proof that these are the only useful groups.

| Segment description | Customers | Average age | Average income (k$) | Average spending score |
|---|---:|---:|---:|---:|
| Lower-income, higher-spending | 45 | 56.3 | 54.3 | 49.1 |
| High-income, lower-spending | 39 | 26.8 | 57.1 | 48.1 |
| High-income, lower-spending | 33 | 41.9 | 88.9 | 17.0 |
| High-income, high-spending | 39 | 32.7 | 86.5 | 82.1 |
| Lower-income, higher-spending | 23 | 25.0 | 25.3 | 77.6 |
| Lower-income, lower-spending | 21 | 45.5 | 26.3 | 19.4 |

Some names repeat because names describe broad income and spending levels, while age can still split those customers into separate clusters. The groups describe observed patterns; they do not establish why someone shops or predict what they will buy.

## Files created by `main.py`

- `outputs/eda_overview.png`: feature distributions, income versus spending, gender counts, and a numeric correlation heatmap.
- `outputs/cluster_selection.png`: the elbow curve and silhouette scores.
- `outputs/customer_segments.png`: customer groups on the income and spending axes, with group centers.
- `outputs/customers_with_clusters.csv`: customer records with `Cluster` and `Segment` labels.
- `outputs/segment_summary.csv`: group counts and average age, income, and spending score.
- `outputs/analysis_results.json`: K-selection values from the run.

The two-dimensional segment chart shows income and spending only. Age is also used by K-Means, so the chart cannot show every difference between groups.

## Streamlit dashboard

Run `streamlit run app.py` after `main.py`. The dashboard has customer counts and group averages, a segment filter, an interactive scatter plot, business suggestions, and a CSV download. If the generated cluster file is missing, the app runs `main.py` on startup to create it.

## Deploy on Streamlit Community Cloud

This project is in the public repository [sanskarkadam/customer-segmentation-kmeans-retail](https://github.com/sanskarkadam/customer-segmentation-kmeans-retail). To deploy it:

1. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
2. Choose **Create app** and select repository `sanskarkadam/customer-segmentation-kmeans-retail`.
3. Select branch `main` and app file `app.py`.
4. In **Advanced settings**, select Python 3.13, then choose **Deploy**.

The repository and deployed app are public. The dataset is included; do not add private data or credentials. Community Cloud installs the packages listed in `requirements.txt`.

## Limitations

This is a small teaching dataset, not a complete transaction history. The supplied spending score is not the same as actual revenue or a forecast. K-Means depends on the features, scaling, and chosen K, and tends to favor compact groups. Treat suggested marketing actions as ideas to test and measure, not conclusions about individual customers.

## Short project summary

This project uses K-Means to group 200 Mall Customers by age, annual income, and spending score. It checks and scales the data, compares candidate cluster counts with inertia and silhouette score, visualizes six groups, and exports customer labels and group statistics. A Streamlit dashboard lets users explore and download the results.

## Viva questions

1. **What is clustering?** An unsupervised learning method that groups similar records without known target labels.
2. **How does K-Means work?** It assigns records to nearby centers, recalculates each center as the group average, and repeats.
3. **Why scale the features?** K-Means uses distances, so scaling prevents a feature's larger numeric range from dominating.
4. **Why leave out `CustomerID`?** It identifies a record but does not describe customer behaviour.
5. **What is inertia?** The sum of squared distances from records to their assigned cluster centers.
6. **What is the Elbow Method?** A plot of inertia across K values; a bend can show where extra clusters bring smaller gains.
7. **What does silhouette score measure?** How well each record fits its own group compared with nearby groups; higher average scores often mean clearer separation.
8. **Why not choose K using only silhouette score?** Interpretability and the elbow curve also matter; the largest score does not guarantee the most useful groups.
9. **What does the segment chart leave out?** It shows income and spending, but clustering also uses age.
10. **How might a retailer use the groups?** To try different promotions with different groups and measure the results.
