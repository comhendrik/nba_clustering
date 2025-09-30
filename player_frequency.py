import pandas as pd
import time
from nba_api.stats.endpoints import synergyplaytypes
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from nba_api.stats.endpoints import leaguestandings
from nba_api.stats.static import teams

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



###Linear Regression

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Assume df_players_clustered has columns: PLAYER_ID, TEAM_ABBREVIATION, Cluster
# and df_team_wins has TEAM_ABBREVIATION, WINS

# -----------------------------
# Step 1: Aggregate cluster frequencies per team
# -----------------------------
df_players['count'] = 1  # helper column for counts
cluster_counts = df_players.groupby(['TEAM_ABBREVIATION', 'Cluster'])['count'].sum().unstack(fill_value=0)


standings = leaguestandings.LeagueStandings(season="2024-25", league_id="00").get_data_frames()[0]
nba_teams = teams.get_teams()
team_map = {t["id"]: t["abbreviation"] for t in nba_teams}
standings["TEAM_ABBREVIATION"] = standings["TeamID"].map(team_map)
team_wins = standings[["TEAM_ABBREVIATION", "WINS"]]

merged = cluster_counts.merge(team_wins, on="TEAM_ABBREVIATION", how="inner")
X_reg = merged.drop(columns=["WINS", "TEAM_ABBREVIATION"]).copy()
X_reg.columns = X_reg.columns.astype(str)
y_reg = merged["WINS"]

# -----------------------------
# Step 2: Fit regression
# -----------------------------

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def polynomial_regression_analysis(X_reg, y_reg, max_degree=4, fit_intercept=False):
    results = []

    for degree in range(1, max_degree + 1):
        print(f"\n--- Polynomial Degree: {degree} ---")
        
        # -----------------------------
        # Step 1: Transform features
        # -----------------------------
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X_reg)
        feature_names = poly.get_feature_names_out(X_reg.columns)  

        # -----------------------------
        # Step 2: Fit model
        # -----------------------------
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(X_poly, y_reg)

        # -----------------------------
        # Step 3: Predictions & metrics
        # -----------------------------
        y_pred = model.predict(X_poly)
        mae = mean_absolute_error(y_reg, y_pred)
        r2 = r2_score(y_reg, y_pred)
        mse = mean_squared_error(y_reg, y_pred)
        print(f"MAE: {mae:.2f}, R²: {r2:.2f}, MSE: {mse:.2f}")

        # Coefficients
        for name, coef in zip(feature_names, model.coef_):
            print(f"{name}: {coef:.4f}")
        print(f"Intercept: {model.intercept_:.4f}")

        # Save results
        results.append({
            "degree": degree,
            "model": model,
            "X_poly": X_poly,
            "y_pred": y_pred,
            "mae": mae,
            "r2": r2,
            "mse": mse,
            "feature_names": feature_names
        })

        # -----------------------------
        # Step 4: Plot predicted vs actual wins
        # -----------------------------
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=y_reg, y=y_pred)
        plt.plot([y_reg.min(), y_reg.max()], [y_reg.min(), y_reg.max()], "r--", lw=2)
        plt.xlabel("Actual Wins")
        plt.ylabel("Predicted Wins")
        plt.title(f"Degree {degree} Polynomial: Actual vs Predicted Wins")
        plt.grid(True)
        plt.show()

    return results

results = polynomial_regression_analysis(X_reg, y_reg, max_degree=4)

