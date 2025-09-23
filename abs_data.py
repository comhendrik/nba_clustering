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
mode = "AVG"     # <--- z.B. PER_36, PER_40, per35
cluster_size = 3
output_dir = "cluster_outputs"
os.makedirs(output_dir, exist_ok=True)

report_path = os.path.join(output_dir, "analysis_report.txt")
with open(report_path, "w") as report:
    report.write("=== NBA 2024-25 Cluster & Regression Analysis ===\n\n")

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
base_cols = ["OREB", "DREB", "AST", "TOV", "STL", "BLK", "PF", "PFD", "PTS"]
features = [f"{col}_{mode}" for col in base_cols]

missing = [f for f in features if f not in filtered_players.columns]
if missing:
    raise ValueError(f"Fehlende Spalten im DataFrame: {missing}")

X = filtered_players[features].fillna(0)

# ==============================================
# 3. Skalierung & Gewichtung
# ==============================================
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

base_feature_weights = {
    "OREB": 0.1, "DREB": 0.45, "AST": 0.3, "TOV": 0.68,
    "STL": 0.33, "BLK": 0.2, "PF": 0.2, "PFD": 0.12, "PTS": 0.73,
}
feature_weights = {f"{k}_{mode}": v for k, v in base_feature_weights.items()}
weights_array = np.array([feature_weights[f] for f in features])
X_weighted = X_scaled * weights_array

# ==============================================
# 4. KMeans Clustering
# ==============================================
kmeans = KMeans(init="k-means++", n_init=50, n_clusters=cluster_size, random_state=42)
filtered_players["cluster"] = kmeans.fit_predict(X_weighted)

# ==============================================
# 5. Cluster-Summary
# ==============================================
static_features = ["AGE", "MIN_AVG",]
stats = ["REB", "AST", "TOV", "STL", "BLK", "PF", "PFD", "PTS"]
important_features = static_features + [f"{s}_{mode}" for s in stats]

cluster_summary = (
    filtered_players.groupby("cluster")
    .median(numeric_only=True)[important_features]
)

with open(report_path, "a") as report:
    report.write("=== Cluster-Summary (Medianwerte) ===\n")
    report.write(str(cluster_summary))
    report.write("\n\n")

# ==============================================
# 6. Silhouette Score
# ==============================================
sil_score = silhouette_score(X_scaled, filtered_players["cluster"])
print(f"Silhouette Coefficient: {sil_score:.4f}")
with open(report_path, "a") as report:
    report.write(f"Silhouette Coefficient: {sil_score:.4f}\n\n")

# ==============================================
# 7. Clustergrößen-Plot
# ==============================================
plt.figure(figsize=(6, 4))
filtered_players["cluster"].value_counts().sort_index().plot(
    kind="bar", title="Cluster-Größen", color="skyblue", edgecolor="black"
)
plt.xlabel("Cluster")
plt.ylabel("Anzahl Spieler")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cluster_sizes.png"), dpi=300)
plt.close()

# ==============================================
# 8. Minuten pro Team/Cluster
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
# 9. Regression Siege ~ Cluster-Minuten
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
# 10. Dataset-Info
# ==============================================
with open(report_path, "a") as report:
    report.write("=== Dataset Info ===\n")
    report.write(f"Total players: {len(filtered_players)}\n")
    report.write(f"Teams: {filtered_players['TEAM_ABBREVIATION'].nunique()}\n")
    report.write(f"Features for clustering: {len(features)}\n")
    report.write(f"Median AGE: {filtered_players['AGE'].median()}\n")
    report.write(f"Median MIN_AVG: {filtered_players['MIN_AVG'].median()}\n")
    report.write(f"Median PTS_{mode}: {filtered_players[f'PTS_{mode}'].median()}\n")
    report.write(f"Median REB_{mode}: {filtered_players[f'REB_{mode}'].median()}\n")
    report.write(f"Median AST_{mode}: {filtered_players[f'AST_{mode}'].median()}\n")
    report.write("\n\n")

print(f"\nAnalyse abgeschlossen. Ergebnisse gespeichert in '{report_path}' und Plots in '{output_dir}'.")
