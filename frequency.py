import time
import pandas as pd
from nba_api.stats.endpoints import synergyplaytypes, leaguestandingsv3
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ------------------------
# 1. Collect team playtype percentiles
# ------------------------
def get_team_playtype_percentiles(season: str, playtypes: list):
    dfs = []
    all_teams = set()

    for type_group in ["Offensive"]:
        for pt in playtypes:
            ep = synergyplaytypes.SynergyPlayTypes(
                league_id="00",
                play_type_nullable=pt,
                player_or_team_abbreviation="T",  # team-level
                season=season,
                season_type_all_star="Regular Season",
                type_grouping_nullable=type_group
            )

            df_pt = ep.get_data_frames()[0]
            if df_pt.empty:
                continue
            
            df2 = df_pt[["TEAM_ID", "TEAM_ABBREVIATION", "POSS_PCT"]].copy()
            df2["POSS_PCT"] = df2["POSS_PCT"] * 100
            col_name = pt.lower() + "_" + type_group.lower()
            df2 = df2.rename(columns={"POSS_PCT": col_name})

            dfs.append(df2)
            all_teams.update(df2["TEAM_ID"].unique())
            time.sleep(1)

    # Merge all DataFrames on TEAM_ID and TEAM_ABBREVIATION
    df_final = pd.DataFrame({"TEAM_ID": list(all_teams)})
    abbrev_map = pd.concat(dfs).drop_duplicates(subset="TEAM_ID")[["TEAM_ID", "TEAM_ABBREVIATION"]]
    df_final = df_final.merge(abbrev_map, on="TEAM_ID", how="left")

    for df in dfs:
        df_final = df_final.merge(df, on=["TEAM_ID", "TEAM_ABBREVIATION"], how="left")

    df_final = df_final.fillna(0)
    return df_final

# ------------------------
# 2. Get team wins
# ------------------------
from nba_api.stats.endpoints import leaguedashteamstats
import pandas as pd

def get_team_offrtg(season: str):
    ep = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        league_id_nullable="00",
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced"  # includes OFFRTG/DEFRTG/NETRTG
    )

    df = ep.get_data_frames()[0]

    print(df.columns)

    # Add ranking
    df["OffensiveRank"] = df["OFF_RATING"].rank(ascending=False, method="min").astype(int)

    return df[["TEAM_ID", "TEAM_NAME", "OFF_RATING", "DEF_RATING"]]


# ------------------------
# 3. Build dataset for regression
# ------------------------
season = "2024-25"
playtypes = ["Isolation", "Cut", "Spotup", "Postup", "Handoff", "Transition",
             "PRBallHandler", "PRRollman", "OffScreen", "Misc"]

df_percentiles = get_team_playtype_percentiles(season, playtypes)
df_wins = get_team_offrtg(season)

df = pd.merge(df_percentiles, df_wins, on=["TEAM_ID"], how="left")

# Sum of all playtype columns


# ------------------------
# 4. Linear regression using scikit-learn
# ------------------------
feature_cols = [col for col in df.columns if col not in ["TEAM_ABBREVIATION","TEAM_ID", "TEAM_NAME", "OFF_RATING", "DEF_RATING", "OffensiveRank"]]
print(df.columns)
df["Playtype_Sum"] = df[feature_cols].sum(axis=1)

print(df.head())
X = df[feature_cols]
y = df["OFF_RATING"]

model = LinearRegression(fit_intercept=False)
model.fit(X, y)
y_pred = model.predict(X)

# ------------------------
# 5. Evaluate model
# ------------------------
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")
print(f"MSE: {mse:.2f}")

print("\nKoeffizienten (descending by value):")
for feature, coef in sorted(zip(feature_cols, model.coef_), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")



# ------------------------
# 6. Scatterplot Actual vs Predicted
# ------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y, y=y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
plt.xlabel("Actual Wins")
plt.ylabel("Predicted Wins")
plt.title("Actual vs Predicted Wins (Team-level)")
plt.grid(True)
plt.show()

# ------------------------
# 7. Residual plot
# ------------------------
residuals = y - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=15)
plt.xlabel("Residual (Actual – Predicted)")
plt.title("Distribution of Residuals")
plt.grid(True)
plt.show()


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------
# Features for clustering
# ------------------------
X_features = df[feature_cols]

# ------------------------
# Test different cluster sizes and compute silhouette
# ------------------------
sil_scores = []
cluster_range = range(2, 8)  # 2 to 7 clusters

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_features)
    sil = silhouette_score(X_features, labels)
    sil_scores.append(sil)
    print(f"Clusters: {k}, Silhouette Score: {sil:.4f}")

best_k = cluster_range[sil_scores.index(max(sil_scores))]
print(f"\nBest number of clusters: {best_k}")

# ------------------------
# Fit KMeans with best cluster
# ------------------------
kmeans = KMeans(n_clusters=best_k, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_features)

# ------------------------
# PCA for visualization
# ------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_features)
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

print("Explained variance by PCA components:", pca.explained_variance_ratio_)

# ------------------------
# Plot teams using PCA components
# ------------------------
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x="PCA1",
    y="PCA2",
    hue="Cluster",
    data=df,
    palette="tab10",
    s=200,
    legend="full"
)

# Annotate team abbreviations
for i, row in df.iterrows():
    plt.text(row["PCA1"] + 0.01, row["PCA2"] + 0.01, row["TEAM_ABBREVIATION"], fontsize=9)

plt.title(f"NBA Teams Clustered by Playtype Percentiles (k={best_k})")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()


