import os
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import warnings

# ==============================================
# Settings
# ==============================================
csv_file = "nba_player_stats_clean.csv"
output_file = "silhouette_combinations.txt"
cluster_sizes = [2, 3, 4, 5]  # cluster sizes to test

# Agglomerative clustering settings
linkages = ["ward", "average", "complete", "single"]
metrics = ["euclidean", "manhattan", "cosine"]  # only used if linkage != "ward"

# ==============================================
# Load CSV
# ==============================================
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    print(f"Lade Daten aus CSV: {csv_file}")
else:
    raise FileNotFoundError(f"CSV {csv_file} nicht gefunden.")

# ==============================================
# Features to test
# ==============================================
features = [
    'PCT_STL_Usage','PCT_BLK_Usage',
    'PCT_PTS_FB_Scoring', 'PCT_PTS_2PT_MR_Scoring',
    'PCT_AST_Usage', 
    'PCT_DREB_Usage','PCT_OREB_Usage',
    'PCT_PF_Usage', 'PCT_PFD_Usage', 'PCT_TOV_Usage',
    'PCT_PTS_FT_Scoring', 'PCT_PTS_2PT_Scoring',  'PCT_PTS_3PT_Scoring','PCT_FTM_Usage',
]

features = [
    'PCT_STL_Usage', 'PCT_PTS_FB_Scoring',
    'PCT_PTS_2PT_MR_Scoring',
    'PCT_AST_Usage', 'PCT_BLK_Usage',
    'PCT_DREB_Usage', 'PCT_PTS_FT_Scoring', 'PCT_PF_Usage',
    'PCT_PTS_2PT_Scoring', 'PCT_OREB_Usage', 'PCT_PTS_3PT_Scoring',
    'PCT_TOV_Usage',
    'PCT_FG3M_Usage', 'PCT_PFD_Usage', 'PCT_FTM_Usage',
    'USG_PCT_Advanced','PCT_FGM_Usage',
    'OPP_PTS_OFF_TOV_Defense_AVG', 'OPP_PTS_FB_Defense_AVG',
    'OPP_PTS_2ND_CHANCE_Defense_AVG', 'OPP_PTS_PAINT_Defense_AVG',
]
features = [f for f in features if f in df.columns]

# ==============================================
# Compute correlations and order features
# ==============================================
numeric_df = df[features].fillna(0)
corr_matrix = numeric_df.corr().abs()
avg_corr = corr_matrix.mean().sort_values()  # least correlated first
ordered_features = avg_corr.index.tolist()

print("Ordered features from least to most correlated:")
print(ordered_features)

# ==============================================
# Iterate over cluster sizes, algorithms, and features
# ==============================================
results = []

for n_features in range(2, len(ordered_features) + 1):
    selected_features = ordered_features[:n_features]
    selected_features = features
    X = numeric_df[selected_features]

    # --- KMeans ---
    for k in cluster_sizes:
        kmeans = KMeans(n_clusters=k, n_init=50, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        counts = pd.Series(labels).value_counts().to_dict()  # cluster distribution
        results.append(("KMeans", score, k, selected_features, counts))

    # --- Agglomerative Clustering ---
    for k in cluster_sizes:
        for linkage in linkages:
            if linkage == "ward":
                model = AgglomerativeClustering(n_clusters=k, linkage="ward")
                labels = model.fit_predict(X)
                score = silhouette_score(X, labels)
                counts = pd.Series(labels).value_counts().to_dict()
                results.append((f"Agglomerative-{linkage}", score, k, selected_features, counts))
            else:
                for metric in metrics:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            model = AgglomerativeClustering(
                                n_clusters=k, linkage=linkage, metric=metric
                            )
                            labels = model.fit_predict(X)
                            score = silhouette_score(X, labels)
                            counts = pd.Series(labels).value_counts().to_dict()
                            results.append((f"Agglomerative-{linkage}-{metric}", score, k, selected_features, counts))
                        except Exception as e:
                            print(f"Skipping Agglomerative {linkage}-{metric}: {e}")
    break

# ==============================================
# Save all combinations to TXT (sorted by score)
# ==============================================
results_sorted = sorted(results, key=lambda x: x[1], reverse=True)

with open(output_file, "w") as f:
    f.write("Algorithm | Silhouette Score | Cluster Size | Features | Cluster Distribution\n")
    f.write("="*160 + "\n")
    for algo, score, k, feats, counts in results_sorted:
        f.write(f"{algo} | {score:.4f} | {k} | {feats} | {counts}\n")

# ==============================================
# Show best Silhouette coefficient
# ==============================================
best_algo, best_score, best_k, best_features, best_counts = results_sorted[0]
print(f"\nBest Silhouette Score: {best_score:.4f}")
print(f"Algorithm: {best_algo}")
print(f"Cluster size: {best_k}")
print(f"Features: {best_features}")
print(f"Cluster distribution: {best_counts}")

with open(output_file, "a") as f:
    f.write("\n\nBest Silhouette Score:\n")
    f.write(f"{best_algo} | {best_score:.4f} | {best_k} | {best_features} | {best_counts}\n")

print(f"All combinations saved to {output_file}")
