import pandas as pd
import time
from nba_api.stats.endpoints import synergyplaytypes
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def get_player_playtype_frequencies(season: str, playtypes: list):
    dfs = []
    all_players = set()

    for type_group in ["Offensive"]:  # usually just offensive for player archetypes
        for pt in playtypes:
            ep = synergyplaytypes.SynergyPlayTypes(
                league_id="00",
                play_type_nullable=pt,
                player_or_team_abbreviation="P",  # player-level
                season=season,
                season_type_all_star="Regular Season",
                type_grouping_nullable=type_group
            )

            df_pt = ep.get_data_frames()[0]
            if df_pt.empty:
                continue

            df2 = df_pt[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "POSS_PCT"]].copy()
            df2["POSS_PCT"] = df2["POSS_PCT"] * 100
            col_name = pt.lower() + "_" + type_group.lower()
            df2 = df2.rename(columns={"POSS_PCT": col_name})

            dfs.append(df2)
            all_players.update(df2["PLAYER_ID"].unique())
            time.sleep(1)

    # Merge all DataFrames on PLAYER_ID, PLAYER_NAME, TEAM_ABBREVIATION
    df_final = pd.DataFrame({"PLAYER_ID": list(all_players)})
    player_map = pd.concat(dfs).drop_duplicates(subset="PLAYER_ID")[
        ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"]
    ]
    df_final = df_final.merge(player_map, on="PLAYER_ID", how="left")

    for df in dfs:
        df_final = df_final.merge(df, on=["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"], how="left")

    df_final = df_final.fillna(0)
    return df_final

playtypes = ["Isolation", "Cut", "Spotup", "Postup", "Handoff", "Transition",
             "PRBallHandler", "PRRollman", "OffScreen", "Misc"]

df_players = get_player_playtype_frequencies("2024-25", playtypes)

feature_cols = [col for col in df_players.columns if col not in ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "POSS_PCT"]]

X = df_players[feature_cols].values

best_score = -1
best_k = None
best_labels = None

for k in range(2, 7 + 1):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"Silhouette score for k={k}: {score:.4f}")
    if score > best_score:
        best_score = score
        best_k = k
        best_labels = labels

df_players["Cluster"] = best_labels
print(f"Best cluster size based on silhouette coefficient: {best_k}")


distribution = df_players['Cluster'].value_counts().sort_index()
print("Distribution (Player per Cluster):")
print(distribution)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

# Plot clusters
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=df_players['Cluster'], cmap='tab10', s=50, alpha=0.8)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Spieler Cluster Visualisierung nach Playtype-Frequenzen")
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.grid(True)
plt.show()

