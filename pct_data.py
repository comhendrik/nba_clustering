import os
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score
from nba_api.stats.endpoints import leaguestandings
from nba_api.stats.static import teams
from sklearn.linear_model import LinearRegression
import seaborn as sns

# ==============================================
# Einstellungen
# ==============================================
csv_file = "nba_player_stats_2024_25.csv"
cluster_size = 3
output_dir = "pct_outputs"
os.makedirs(output_dir, exist_ok=True)

report_path = os.path.join(output_dir, "pct_analysis_report.txt")
with open(report_path, "w") as report:
    report.write("=== NBA 2024-25 Advanced Percentages Cluster & Regression Analysis ===\n\n")

# ==============================================
# 1. CSV laden
# ==============================================
if os.path.exists(csv_file):
    print(f"Lade Daten aus CSV: {csv_file}")
    filtered_players = pd.read_csv(csv_file)
else:
    raise FileNotFoundError(f"CSV {csv_file} nicht gefunden. Bitte zuerst die Daten abrufen.")

# ==============================================
# 2. Features für Clustering
# ==============================================
features = [
    'USG_PCT_Usage', 'PCT_FGM_Usage', 'PCT_FG3M_Usage', 'PCT_FTM_Usage',
    'PCT_OREB_Usage', 'PCT_DREB_Usage', 'PCT_AST_Usage', 'PCT_TOV_Usage',
    'PCT_STL_Usage', 'PCT_BLK_Usage', 'PCT_PF_Usage', 'PCT_PFD_Usage',
]
missing = [f for f in features if f not in filtered_players.columns]
if missing:
    raise ValueError(f"Fehlende Spalten im DataFrame: {missing}")

X = filtered_players[features].fillna(0)

feature_map = {
    'USG_PCT_Usage': 1,          # Placeholder, can be set based on usage calculation
    'PCT_FGM_Usage': 0.666284,
    'PCT_FG3M_Usage': 0.403160,
    'PCT_FTM_Usage': 0.211148,
    'PCT_OREB_Usage': 0.101541,
    'PCT_DREB_Usage': 0.454771,
    'PCT_AST_Usage': 0.302946,
    'PCT_TOV_Usage': 0.680298,
    'PCT_STL_Usage': 0.334396,
    'PCT_BLK_Usage': 0.205620,
    'PCT_PF_Usage': 0.205090,
    'PCT_PFD_Usage': 0.123648
}

feature_weights = {f"{k}": v for k, v in feature_map.items()}
weights_array = np.array([feature_weights[f] for f in features])
X_weighted = X * weights_array

# ==============================================
# 3. KMeans Clustering
# ==============================================
kmeans = KMeans(init="k-means++", n_init=50, n_clusters=cluster_size, random_state=42)
filtered_players["cluster"] = kmeans.fit_predict(X_weighted)

# ==============================================
# 4. Cluster-Summary
# ==============================================
cluster_summary = (
    filtered_players.groupby("cluster")
    .median(numeric_only=True)[features]
)

with open(report_path, "a") as report:
    report.write("=== Cluster-Summary (Medianwerte) ===\n")
    report.write(str(cluster_summary))
    report.write("\n\n")

# ==============================================
# 5. Silhouette Score
# ==============================================
sil_score = silhouette_score(X, filtered_players["cluster"])
print(f"Silhouette Coefficient: {sil_score:.4f}")
with open(report_path, "a") as report:
    report.write(f"Silhouette Coefficient: {sil_score:.4f}\n\n")

# ==============================================
# 6. Clustergrößen-Plot
# ==============================================
plt.figure(figsize=(6, 4))
filtered_players["cluster"].value_counts().sort_index().plot(
    kind="bar", title="Cluster-Größen", color="lightgreen", edgecolor="black"
)
plt.xlabel("Cluster")
plt.ylabel("Anzahl Spieler")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cluster_sizes.png"), dpi=300)
plt.close()

# ==============================================
# 7. Minuten pro Team/Cluster
# ==============================================
team_cluster_minutes = (
    filtered_players.groupby(["TEAM_ABBREVIATION", "cluster"])["MIN_AVG"]
    .sum()
    .unstack(fill_value=0)
)

with open(report_path, "a") as report:
    report.write("=== Minuten pro Team und Cluster ===\n")
    for team, clusters in team_cluster_minutes.iterrows():
        report.write(f"\nTeam {team}:\n")
        for cid, minutes in clusters.items():
            report.write(f"  Cluster {cid}: {minutes:.1f} Minuten\n")
    report.write("\n\n")

# ==============================================
# 8. Regression Siege ~ Cluster-Minuten
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
y_pred = reg.predict(X_reg)

# Regression metrics
r2 = r2_score(y_reg, y_pred)
n, p = X_reg.shape
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
mae = mean_absolute_error(y_reg, y_pred)
mse = mean_squared_error(y_reg, y_pred)
rmse = np.sqrt(mse)

coefficients = pd.Series(reg.coef_, index=X_reg.columns).sort_values(ascending=False)

with open(report_path, "a") as report:
    report.write("=== Regression: Siege ~ Cluster-Minuten ===\n")
    report.write("Koeffizienten:\n")
    report.write(str(coefficients))
    report.write(f"\nIntercept (Basis-Siege): {reg.intercept_:.3f}\n")
    report.write(f"R²: {r2:.4f}\n")
    report.write(f"Adjusted R²: {adj_r2:.4f}\n")
    report.write(f"MAE: {mae:.4f}\n")
    report.write(f"MSE: {mse:.4f}\n")
    report.write(f"RMSE: {rmse:.4f}\n\n")

# Regression plot
plt.figure(figsize=(7, 5))
sns.regplot(x=y_reg, y=y_pred, ci=None, line_kws={"color": "red", "lw": 2})
plt.xlabel("Tatsächliche Siege")
plt.ylabel("Vorhergesagte Siege")
plt.title("Regression: Tatsächliche vs. Vorhergesagte Siege")
plt.annotate(
    f"R² = {r2:.3f}\nAdj. R² = {adj_r2:.3f}\nRMSE = {rmse:.2f}",
    xy=(0.05, 0.85),
    xycoords="axes fraction",
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "regression_fit.png"), dpi=300)
plt.close()

# ==============================================
# 9. Dataset-Info
# ==============================================
with open(report_path, "a") as report:
    report.write("=== Dataset Info ===\n")
    report.write(f"Total players: {len(filtered_players)}\n")
    report.write(f"Teams: {filtered_players['TEAM_ABBREVIATION'].nunique()}\n")
    report.write(f"Features for clustering: {len(features)}\n")
    report.write("\nColumns:\n")
    report.write(", ".join(filtered_players.columns))
    report.write("\n\n")

# ==============================================
# 10. Scatterplots
# ==============================================
plot_features = [
    ("USG_PCT_Usage", "PCT_PTS_Usage"),
    ("PCT_AST_Usage", "PCT_TOV_Usage"),
    ("PCT_OREB_Usage", "PCT_DREB_Usage"),
    ("PCT_STL_Usage", "PCT_BLK_Usage"),
]
for x_feat, y_feat in plot_features:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=filtered_players,
        x=x_feat,
        y=y_feat,
        hue="cluster",
        palette="tab10",
        s=80,
        alpha=0.7,
    )
    plt.title(f"Clustervergleich: {x_feat} vs {y_feat}")
    plt.xlabel(x_feat)
    plt.ylabel(y_feat)
    plt.legend(title="Cluster")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"cluster_plot_{x_feat}_{y_feat}.png"), dpi=300)
    plt.close()

print(f"\nAnalyse abgeschlossen. Ergebnisse gespeichert in '{report_path}' und Plots in '{output_dir}'.")
