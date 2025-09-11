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


from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# 3. K-Means Clusteranalyse
# -----------------------------
features = [
    'REB_AVG', 'AST_AVG',
    'TOV_AVG', 'STL_AVG', 'BLK_AVG',
    'PF_AVG', 'PFD_AVG', 'PTS_AVG',
    'HEIGHT',
    'WEIGHT'
]

# Prüfen, ob alle Features existieren
missing_features = [f for f in features if f not in filtered_players.columns]
if missing_features:
    raise ValueError(f"Folgende Features fehlen im DataFrame: {missing_features}")



# Danach Features auswählen
X = filtered_players[features].fillna(0)


# -----------------------------
# Standardisierung (Z-Transformation)
# -----------------------------
# Create a scaler that transforms data to the range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

cluster_size = 4
kmeans = KMeans(n_clusters=cluster_size, random_state=42)
filtered_players['cluster'] = kmeans.fit_predict(X_scaled)




# -----------------------------
# Cluster-Profile berechnen (Medianwerte)
# -----------------------------
important_features = ['AGE', 'MIN_AVG', 'REB_AVG', 'AST_AVG',
    'TOV_AVG', 'STL_AVG', 'BLK_AVG',
    'PF_AVG', 'PFD_AVG', 'PTS_AVG', 'PLUS_MINUS_AVG',
    'HEIGHT', 'WEIGHT']

cluster_summary = filtered_players.groupby("cluster").median(numeric_only=True)[important_features]

# -----------------------------
# Cluster-Ausgaben in Textdateien speichern
# -----------------------------
output_dir = "cluster_outputs"
os.makedirs(output_dir, exist_ok=True)


from sklearn.metrics import silhouette_score

# -----------------------------
# Silhouette Coefficient berechnen
# -----------------------------
# X_scaled = die standardisierten Features, die für KMeans verwendet wurden
sil_score = silhouette_score(X_scaled, filtered_players['cluster'])
print(f"Silhouette Coefficient: {sil_score:.4f}")

# Optional: in eine Textdatei speichern
silhouette_file = os.path.join(output_dir, "silhouette_score.txt")
with open(silhouette_file, "w") as f:
    f.write(f"Silhouette Coefficient for {cluster_size} clusters: {sil_score:.4f}\n")

print(f"Silhouette score saved in {silhouette_file}")


# Cluster Profile
with open(os.path.join(output_dir, "cluster_summary.txt"), "w") as f:
    f.write("Cluster-Profile (Medianwerte mit Height & Weight):\n")
    f.write(str(cluster_summary))
    f.write("\n")

# Alle Spieler pro Cluster
# Alle Spieler pro Cluster inkl. Medianwerte
player_columns = ['PLAYER_NAME', 'TEAM_ABBREVIATION','MIN_AVG', 'AGE', 'REB_AVG', 'AST_AVG',
                  'TOV_AVG', 'STL_AVG', 'BLK_AVG',
                  'PF_AVG', 'PFD_AVG', 'PTS_AVG', 'PLUS_MINUS_AVG',
                  'HEIGHT', 'WEIGHT']

for cluster_id in range(cluster_size):
    cluster_players = filtered_players[filtered_players['cluster'] == cluster_id]
    file_path = os.path.join(output_dir, f"cluster_{cluster_id}_players.txt")
    
    # Medianwerte für diesen Cluster berechnen
    cluster_medians = cluster_players[player_columns[2:]].median()  # skip PLAYER_NAME and TEAM_ABBREVIATION

    with open(file_path, "w") as f:
        f.write(f"Cluster {cluster_id} - Alle Spieler:\n\n")
        f.write(cluster_players[player_columns].to_string(index=False))
        f.write("\n\n--- Medianwerte für Cluster ---\n")
        for col, val in cluster_medians.items():
            f.write(f"{col}: {val}\n")
    
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
# 6. Spieler pro Team und Cluster zählen -> nach Minuten summieren
# -----------------------------
team_cluster_minutes = (
    filtered_players
    .groupby(["TEAM_ABBREVIATION", "cluster"])["MIN_AVG"]   # <- aggregate minutes
    .sum()
    .unstack(fill_value=0)
)
# Dictionary-Ausgabe in Textdatei
with open(os.path.join(output_dir, "team_cluster_counts.txt"), "w") as f:
    for team, clusters in team_cluster_minutes.iterrows():
        f.write(f"\nTeam {team}:\n")
        for cluster_id, minutes in clusters.items():
            f.write(f"  Cluster {cluster_id}: {minutes} Minuten\n")

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
merged = team_cluster_minutes.merge(
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

# -----------------------------
# 0b. Dataset Info speichern
# -----------------------------
dataset_info_path = os.path.join(output_dir, "dataset_info.txt")

with open(dataset_info_path, "w") as f:
    f.write("=== NBA 2024-25 Players Dataset Info ===\n\n")
    f.write(f"Total number of players: {len(filtered_players)}\n")
    f.write(f"Number of teams: {filtered_players['TEAM_ABBREVIATION'].nunique()}\n")
    f.write(f"Number of features used for clustering: {len(features)}\n")
    f.write(f"Median AGE: {filtered_players['AGE'].median()}\n")
    f.write(f"Median MIN_AVG: {filtered_players['MIN_AVG'].median()}\n")
    f.write(f"Median PTS_AVG: {filtered_players['PTS_AVG'].median()}\n")
    f.write(f"Median REB_AVG: {filtered_players['REB_AVG'].median()}\n")
    f.write(f"Median AST_AVG: {filtered_players['AST_AVG'].median()}\n")
    f.write("\nColumns in dataset:\n")
    f.write(", ".join(filtered_players.columns))
    
print(f"Dataset info saved in {dataset_info_path}")
