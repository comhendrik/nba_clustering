import os
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

csv_file = "nba_2024_25_players.csv"

# -----------------------------
# 1. Prüfen, ob CSV existiert
# -----------------------------
if os.path.exists(csv_file):
    print(f"Lade Daten aus CSV: {csv_file}")
    filtered_players = pd.read_csv(csv_file)
else:
    raise FileNotFoundError(f"CSV {csv_file} nicht gefunden. Bitte zuerst die Daten über die API abrufen.")


# -----------------------------
# 3. K-Means Clusteranalyse
# -----------------------------
features = [
    'AGE', 'GP', 'W', 'L', 'W_PCT',
    'MIN_AVG', 'FGM_AVG', 'FGA_AVG', 'FG_PCT',
    'FG3M_AVG', 'FG3A_AVG', 'FG3_PCT',
    'FTM_AVG', 'FTA_AVG', 'FT_PCT',
    'OREB_AVG', 'DREB_AVG', 'REB_AVG', 'AST_AVG',
    'TOV_AVG', 'STL_AVG', 'BLK_AVG', 'BLKA_AVG',
    'PF_AVG', 'PFD_AVG', 'PTS_AVG', 'PLUS_MINUS_AVG',
    'NBA_FANTASY_PTS', 'DD2', 'TD3',
    'GP_RANK', 'W_RANK', 'L_RANK', 'W_PCT_RANK', 'MIN_RANK',
    'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK', 'FG3M_RANK', 'FG3A_RANK',
    'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK', 'FT_PCT_RANK',
    'OREB_RANK', 'DREB_RANK', 'REB_RANK', 'AST_RANK',
    'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK',
    'PF_RANK', 'PFD_RANK', 'PTS_RANK', 'PLUS_MINUS_RANK',
    'NBA_FANTASY_PTS_RANK', 'DD2_RANK', 'TD3_RANK',
    'TEAM_COUNT',
    'HEIGHT',
    'WEIGHT'
]

# Prüfen, ob alle Features existieren
missing_features = [f for f in features if f not in filtered_players.columns]
if missing_features:
    raise ValueError(f"Folgende Features fehlen im DataFrame: {missing_features}")

X = filtered_players[features].fillna(0)

cluster_size = 5
kmeans = KMeans(n_clusters=cluster_size, random_state=42)
filtered_players['cluster'] = kmeans.fit_predict(X)

# -----------------------------
# Cluster-Profile berechnen (Medianwerte)
# -----------------------------
important_features = ['AGE', 'REB_AVG', 'AST_AVG',
    'TOV_AVG', 'STL_AVG', 'BLK_AVG',
    'PF_AVG', 'PFD_AVG', 'PTS_AVG', 'PLUS_MINUS_AVG',
    'HEIGHT', 'WEIGHT']

cluster_summary = filtered_players.groupby("cluster").median(numeric_only=True)[important_features]

# -----------------------------
# Cluster-Ausgaben in Textdateien speichern
# -----------------------------
output_dir = "cluster_outputs"
os.makedirs(output_dir, exist_ok=True)

# Cluster Profile
with open(os.path.join(output_dir, "cluster_summary.txt"), "w") as f:
    f.write("Cluster-Profile (Medianwerte mit Height & Weight):\n")
    f.write(str(cluster_summary))
    f.write("\n")

# Alle Spieler pro Cluster
player_columns = ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'AGE', 'REB_AVG', 'AST_AVG',
                  'TOV_AVG', 'STL_AVG', 'BLK_AVG',
                  'PF_AVG', 'PFD_AVG', 'PTS_AVG', 'PLUS_MINUS_AVG',
                  'HEIGHT', 'WEIGHT']

for cluster_id in range(cluster_size):
    cluster_players = filtered_players[filtered_players['cluster'] == cluster_id]
    file_path = os.path.join(output_dir, f"cluster_{cluster_id}_players.txt")
    with open(file_path, "w") as f:
        f.write(f"Cluster {cluster_id} - Alle Spieler:\n\n")
        f.write(cluster_players[player_columns].to_string(index=False))
    print(f"Spieler von Cluster {cluster_id} gespeichert in {file_path}")


# -----------------------------
# 5. Clustergrößen visualisieren
# -----------------------------
filtered_players['cluster'].value_counts().sort_index().plot(
    kind='bar', title='Cluster-Größen'
)
plt.xlabel('Cluster')
plt.ylabel('Anzahl Spieler')
plt.show()

# -----------------------------
# 6. Spieler pro Team und Cluster zählen
# -----------------------------
team_cluster_counts = (
    filtered_players
    .groupby(["TEAM_ABBREVIATION", "cluster"])
    .size()
    .unstack(fill_value=0)
)

# Dictionary-Ausgabe in Textdatei
with open(os.path.join(output_dir, "team_cluster_counts.txt"), "w") as f:
    for team, clusters in team_cluster_counts.iterrows():
        f.write(f"\nTeam {team}:\n")
        for cluster_id, count in clusters.items():
            f.write(f"  Cluster {cluster_id}: {count} Spieler\n")

# -----------------------------
# 7. Lineare Regression
# -----------------------------
from nba_api.stats.endpoints import leaguestandings
from nba_api.stats.static import teams
from sklearn.linear_model import LinearRegression

standings = leaguestandings.LeagueStandings(
    season="2024-25", league_id="00"
).get_data_frames()[0]

# Mapping TeamID -> Abkürzung
nba_teams = teams.get_teams()
team_map = {t['id']: t['abbreviation'] for t in nba_teams}
standings['TEAM_ABBREVIATION'] = standings['TeamID'].map(team_map)
team_wins = standings[['TEAM_ABBREVIATION', 'WINS']]

# Merge
merged = team_cluster_counts.merge(
    team_wins,
    on="TEAM_ABBREVIATION",
    how="inner"
)

X = merged.drop(columns=["WINS", "TEAM_ABBREVIATION"]).copy()
X.columns = X.columns.astype(str)
y = merged["WINS"]

reg = LinearRegression()
reg.fit(X, y)
coefficients = pd.Series(reg.coef_, index=X.columns).sort_values(ascending=False)

# Ergebnisse in Datei speichern
with open(os.path.join(output_dir, "regression_results.txt"), "w") as f:
    f.write("--- Einfluss der Cluster auf Siege (Linear Regression) ---\n")
    f.write(str(coefficients))
    f.write("\n\nIntercept (Basis-Siege): " + str(reg.intercept_))

print(f"\nAlle Ergebnisse in Ordner '{output_dir}' gespeichert!")
