# ==============================================
# Refactored Advanced Clustering Analysis
# ==============================================
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression

from nba_api.stats.endpoints import leaguestandings
from nba_api.stats.static import teams

# ==============================================
# Einstellungen
# ==============================================
csv_file = "nba_player_stats_2024_25.csv"
cluster_size = 3
output_dir = "advanced_outputs"
os.makedirs(output_dir, exist_ok=True)
output_txt = os.path.join(output_dir, "analysis_results.txt")

# ==============================================
# 1. CSV laden
# ==============================================
if os.path.exists(csv_file):
    print(f"Lade Daten aus CSV: {csv_file}")
    filtered_players = pd.read_csv(csv_file)
else:
    raise FileNotFoundError(f"CSV {csv_file} nicht gefunden. Bitte zuerst die Daten über die API abrufen.")

# ==============================================
# 2. Features für Clustering
# ==============================================
features = [
    'PCT_STL_Usage', 'PCT_PTS_FB_Scoring',
    'PCT_PTS_2PT_MR_Scoring', 'PCT_AST_Usage',
    'PCT_BLK_Usage', 'PCT_DREB_Usage', 'PCT_PTS_FT_Scoring',
    'PCT_PF_Usage', 'PCT_PTS_2PT_Scoring', 'PCT_OREB_Usage',
    'PCT_PTS_3PT_Scoring', 'PCT_TOV_Usage',
    'PCT_FG3M_Usage', 'PCT_PFD_Usage', 'PCT_FTM_Usage',
    'USG_PCT_Advanced', 'PCT_FGM_Usage',
    'OPP_PTS_OFF_TOV_Defense_AVG', 'OPP_PTS_FB_Defense_AVG',
    'OPP_PTS_2ND_CHANCE_Defense_AVG', 'OPP_PTS_PAINT_Defense_AVG',
]
missing = [f for f in features if f not in filtered_players.columns]
if missing:
    raise ValueError(f"Folgende Spalten fehlen im DataFrame: {missing}")

# Scale average defense metrics
avg_metrics = [
    'OPP_PTS_OFF_TOV_Defense_AVG', 'OPP_PTS_FB_Defense_AVG',
    'OPP_PTS_2ND_CHANCE_Defense_AVG', 'OPP_PTS_PAINT_Defense_AVG'
]
scaler = MinMaxScaler()
filtered_players[avg_metrics] = scaler.fit_transform(filtered_players[avg_metrics])

X = filtered_players[features].fillna(0)

# ==============================================
# 3. KMeans Clustering
# ==============================================
kmeans = KMeans(init="k-means++", n_init=50, n_clusters=cluster_size, random_state=42)
filtered_players["cluster"] = kmeans.fit_predict(X)

# ==============================================
# 4. Cluster-Summary
# ==============================================
cluster_summary = filtered_players.groupby("cluster").median(numeric_only=True)[features]

# ==============================================
# 5. Silhouette Score
# ==============================================
sil_score = silhouette_score(X, filtered_players["cluster"])
print(f"Silhouette Coefficient: {sil_score:.4f}")

# ==============================================
# 6. Minuten pro Team/Cluster
# ==============================================
team_cluster_minutes = (
    filtered_players.groupby(["TEAM_ABBREVIATION", "cluster"])["MIN_AVG"]
    .sum()
    .unstack(fill_value=0)
)

# ==============================================
# 7. Regression Siege ~ Cluster-Minuten
# ==============================================
standings = leaguestandings.LeagueStandings(
    season="2024-25", league_id="00"
).get_data_frames()[0]

nba_teams = teams.get_teams()
team_map = {t["id"]: t["abbreviation"] for t in nba_teams}
standings["TEAM_ABBREVIATION"] = standings["TeamID"].map(team_map)
team_wins = standings[["TEAM_ABBREVIATION", "WINS"]]

merged = team_cluster_minutes.merge(team_wins, on="TEAM_ABBREVIATION", how="inner")
X_reg = merged.drop(columns=["WINS", "TEAM_ABBREVIATION"]).copy()
X_reg.columns = X_reg.columns.astype(str)
y_reg = merged["WINS"]

reg = LinearRegression()
reg.fit(X_reg, y_reg)
coefficients = pd.Series(reg.coef_, index=X_reg.columns).sort_values(ascending=False)

# ==============================================
# 8. Dataset-Info
# ==============================================
dataset_info = f"""
=== NBA 2024-25 Players Dataset Info ===

Total players: {len(filtered_players)}
Teams: {filtered_players['TEAM_ABBREVIATION'].nunique()}
Features for clustering: {len(features)}

Columns:
{", ".join(filtered_players.columns)}
"""

# ==============================================
# 9. Save ALL results to one TXT
# ==============================================
with open(output_txt, "w") as f:
    f.write(f"Silhouette Score: {sil_score:.4f}\n\n")

    f.write("=== Cluster Summary (Medianwerte) ===\n")
    f.write(str(cluster_summary))
    f.write("\n\n")

    f.write("=== Team Cluster Minutes ===\n")
    f.write(str(team_cluster_minutes))
    f.write("\n\n")

    f.write("=== Regression Siege ~ Cluster-Minuten ===\n")
    f.write(str(coefficients))
    f.write(f"\n\nIntercept (Basis-Siege): {reg.intercept_:.4f}\n\n")

    f.write(dataset_info)

print(f"Alle Ergebnisse gespeichert in {output_txt}")

# ==============================================
# 10. Plots (saved as PNG)
# ==============================================
# Cluster sizes
plt.figure(figsize=(8, 5))
sns.barplot(
    x=filtered_players["cluster"].value_counts().sort_index().index,
    y=filtered_players["cluster"].value_counts().sort_index().values,
    palette="viridis"
)
plt.title("Cluster-Größen (Spieleranzahl)", fontsize=14)
plt.xlabel("Cluster")
plt.ylabel("Anzahl Spieler")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cluster_sizes.png"))
plt.close()

# Scatterplot example (USG_PCT_Advanced vs PCT_AST_Usage)
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=filtered_players,
    x="USG_PCT_Advanced", y="PCT_AST_Usage",
    hue="cluster", palette="viridis", alpha=0.7
)
plt.title("Cluster nach USG_PCT_Advanced & PCT_AST_Usage", fontsize=14)
plt.xlabel("USG_PCT_Advanced")
plt.ylabel("PCT_AST_Usage")
plt.legend(title="Cluster")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cluster_scatter.png"))
plt.close()

print("Plots gespeichert: cluster_sizes.png & cluster_scatter.png")
