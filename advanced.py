import os
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from nba_api.stats.endpoints import leaguestandings
from nba_api.stats.static import teams
from sklearn.linear_model import LinearRegression
import seaborn as sns

# ==============================================
# Einstellungen
# ==============================================
csv_file = "nba_player_stats_clean.csv"
cluster_size = 3

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
missing = [f for f in features if f not in filtered_players.columns]
if missing:
    raise ValueError(f"Folgende Spalten fehlen im DataFrame: {missing}")

avg_metrics = [
    'OPP_PTS_OFF_TOV_Defense_AVG', 'OPP_PTS_FB_Defense_AVG',
    'OPP_PTS_2ND_CHANCE_Defense_AVG', 'OPP_PTS_PAINT_Defense_AVG'
]

scaler = MinMaxScaler()
filtered_players[avg_metrics] = scaler.fit_transform(filtered_players[avg_metrics])

X = filtered_players[features].fillna(0)

# ==============================================
# 4. KMeans Clustering
# ==============================================
kmeans = KMeans(init="k-means++", n_init=50, n_clusters=cluster_size, random_state=42)
filtered_players["cluster"] = kmeans.fit_predict(X)

# ==============================================
# 5. Cluster-Summary
# ==============================================
cluster_summary = (
    filtered_players.groupby("cluster")
    .median(numeric_only=True)[features]
)

# ==============================================
# 6. Ausgabe-Verzeichnisse
# ==============================================
output_dir = "advanced_outputs"
os.makedirs(output_dir, exist_ok=True)

# ==============================================
# 7. Silhouette Score speichern
# ==============================================
sil_score = silhouette_score(X, filtered_players["cluster"])
print(f"Silhouette Coefficient: {sil_score:.4f}")
with open(os.path.join(output_dir, "silhouette_score.txt"), "w") as f:
    f.write(f"Silhouette Coefficient for {cluster_size} clusters: {sil_score:.4f}\n")

# ==============================================
# 8. Cluster-Profile & Spielerlisten
# ==============================================


with open(os.path.join(output_dir, "cluster_summary.txt"), "w") as f:
    f.write("Cluster-Profile (Medianwerte mit Height & Weight):\n")
    f.write(str(cluster_summary))
    f.write("\n")

for cid in range(cluster_size):
    cluster_players = filtered_players[filtered_players["cluster"] == cid]
    file_path = os.path.join(output_dir, f"cluster_{cid}_players.txt")

    cluster_medians = cluster_players[features].median()

    with open(file_path, "w") as f:
        f.write(f"Cluster {cid} - Alle Spieler:\n\n")
        f.write(cluster_players[features].to_string(index=False))
        f.write("\n\n--- Medianwerte ---\n")
        for col, val in cluster_medians.items():
            f.write(f"{col}: {val}\n")
    print(f"Spieler von Cluster {cid} gespeichert in {file_path}")

# ==============================================
# 9. Clustergrößen-Plot
# ==============================================
filtered_players["cluster"].value_counts().sort_index().plot(
    kind="bar", title="Cluster-Größen"
)
plt.xlabel("Cluster")
plt.ylabel("Anzahl Spieler")
plt.show()

# ==============================================
# 10. Minuten pro Team/Cluster
# ==============================================
team_cluster_minutes = (
    filtered_players.groupby(["TEAM_ABBREVIATION", "cluster"])["MIN_AVG"]
    .sum()
    .unstack(fill_value=0)
)

with open(os.path.join(output_dir, "team_cluster_counts.txt"), "w") as f:
    for team, clusters in team_cluster_minutes.iterrows():
        f.write(f"\nTeam {team}:\n")
        for cid, minutes in clusters.items():
            f.write(f"  Cluster {cid}: {minutes} Minuten\n")

# ==============================================
# 11. Regression Siege ~ Cluster-Minuten
# ==============================================
standings = leaguestandings.LeagueStandings(season="2024-25", league_id="00").get_data_frames()[0]
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
with open(os.path.join(output_dir, "regression_results.txt"), "w") as f:
    f.write("--- Einfluss der Cluster auf Siege (Linear Regression) ---\n")
    f.write(str(coefficients))
    f.write(f"\n\nIntercept (Basis-Siege): {reg.intercept_}")

# ==============================================
# 12. Dataset-Info
# ==============================================
info_path = os.path.join(output_dir, "dataset_info.txt")
with open(info_path, "w") as f:
    f.write("=== NBA 2024-25 Players Dataset Info ===\n\n")
    f.write(f"Total players: {len(filtered_players)}\n")
    f.write(f"Teams: {filtered_players['TEAM_ABBREVIATION'].nunique()}\n")
    f.write(f"Features for clustering: {len(features)}\n")
    f.write("\nColumns:\n")
    f.write(", ".join(filtered_players.columns))

# ==============================================
# 13. Scatterplots
# ==============================================


