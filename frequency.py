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
def get_team_playtype_percentiles(season: str, playtypes: list, type_groups: list, season_type: str):
    dfs = []
    all_teams = set()

    for type_group in type_groups:
        for pt in playtypes:
            ep = synergyplaytypes.SynergyPlayTypes(
                league_id="00",
                play_type_nullable=pt,
                player_or_team_abbreviation="T",  # team-level
                season=season,
                season_type_all_star=season_type,
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

def get_team_rtg(season: str, season_type: str):
    ep = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        league_id_nullable="00",
        season_type_all_star=season_type,
        measure_type_detailed_defense="Advanced"  # includes OFFRTG/DEFRTG/NETRTG
    )

    df = ep.get_data_frames()[0]

    # Add ranking
    df["OffensiveRank"] = df["OFF_RATING"].rank(ascending=False, method="min").astype(int)

    df["DefensiveRank"] = df["DEF_RATING"].rank(ascending=False, method="min").astype(int)

    df["NetRank"] = df["NET_RATING"].rank(ascending=False, method="min").astype(int)

    return df[["TEAM_ID", "TEAM_NAME", "OFF_RATING", "DEF_RATING", "NET_RATING", "W"]]


# ------------------------
# 3. Build dataset for regression
# ------------------------
season = "2024-25"
playtypes = ["Isolation", "Cut", "Spotup", "Postup", "Handoff", "Transition",
             "PRBallHandler", "PRRollman", "OffScreen", "Misc"]

import os
import pandas as pd

# Filenames for CSVs
csv_files = {
    "df_percentiles_offensive_rs": "df_percentiles_offensive_rs.csv",
    "df_percentiles_defensive_rs": "df_percentiles_defensive_rs.csv",
    "df_percentiles_offensive_defensive_rs": "df_percentiles_offensive_defensive_rs.csv",
    "df_ratings_rs": "df_ratings_rs.csv",
    "df_percentiles_offensive_po": "df_percentiles_offensive_po.csv",
    "df_percentiles_defensive_po": "df_percentiles_defensive_po.csv",
    "df_percentiles_offensive_defensive_po": "df_percentiles_offensive_defensive_po.csv",
    "df_ratings_po": "df_ratings_po.csv"
}

# Function to load or create CSV
def load_or_create_df(name, create_func):
    if os.path.exists(csv_files[name]):
        print(f"Loading {name} from CSV...")
        return pd.read_csv(csv_files[name], index_col=0)  # assuming first column is index
    else:
        print(f"CSV for {name} not found. Generating DataFrame...")
        df = create_func()
        df.to_csv(csv_files[name])
        return df

# Define functions to generate data
def create_df_percentiles_offensive(season_type: str):
    return get_team_playtype_percentiles(season, playtypes, ["Offensive", "Defensive"], season_type)

def create_df_percentiles_defensive(season_type: str):
    return get_team_playtype_percentiles(season, playtypes, ["Defensive"], season_type)

def create_df_percentiles_offensive_defensive(season_type: str):
    return get_team_playtype_percentiles(season, playtypes, ["Offensive", "Defensive"], season_type)

def create_df_ratings(season_type: str):
    return get_team_rtg(season, season_type)

#TODO get ratings for playoff stats, win counts or round, test both and compare both of them with each other

# Load or create each DataFrame
df_percentiles_offensive_rs = load_or_create_df("df_percentiles_offensive_rs", lambda: create_df_percentiles_offensive("Regular Season"))
df_percentiles_defensive_rs = load_or_create_df("df_percentiles_defensive_rs", lambda: create_df_percentiles_defensive("Regular Season"))
df_percentiles_offensive_defensive_rs = load_or_create_df("df_percentiles_offensive_defensive_rs", lambda: create_df_percentiles_offensive_defensive("Regular Season"))
df_ratings_rs = load_or_create_df("df_ratings_rs", lambda: create_df_ratings("Regular Season"))
df_percentiles_offensive_po = load_or_create_df("df_percentiles_offensive_po", lambda: create_df_percentiles_offensive("Playoffs"))
df_percentiles_defensive_po = load_or_create_df("df_percentiles_defensive_po", lambda: create_df_percentiles_defensive("Playoffs"))
df_percentiles_offensive_defensive_po = load_or_create_df("df_percentiles_offensive_defensive_po", lambda: create_df_percentiles_offensive_defensive("Playoffs"))
df_ratings_po = load_or_create_df("df_ratings_po", lambda: create_df_ratings("Playoffs"))

def running_regression(df_percentiles, df_rating, goal_rating, season_type):
    df = pd.merge(df_percentiles, df_rating, on=["TEAM_ID"], how="left")

    # Sum of all playtype columns


    # ------------------------
    # 4. Linear regression using scikit-learn
    # ------------------------
    feature_cols = [col for col in df.columns if col not in ["TEAM_ABBREVIATION","TEAM_ID", "TEAM_NAME", "OFF_RATING", "DEF_RATING", "NET_RATING", "OffensiveRank", "DefensiveRank", "NetRank", "W"]]

    X = df[feature_cols]
    y = df[goal_rating]

    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    y_pred = model.predict(X)

    # ------------------------
    # 5. Evaluate model
    # ------------------------
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    # ------------------------
    # 2. Output folder
    # ------------------------
    output_folder = "model_outputs"
    os.makedirs(output_folder, exist_ok=True)

    # ------------------------
    # 3. Save metrics & coefficients to a text file
    # ------------------------
    txt_file = os.path.join(output_folder, f"metrics_and_coefficients_{goal_rating}_{season_type}.txt")
    with open(txt_file, "w") as f:
        f.write(f"MAE: {mae:.2f}\n")
        f.write(f"R²: {r2:.2f}\n")
        f.write(f"MSE: {mse:.2f}\n\n")
        
        f.write("Koeffizienten (descending by value):\n")
        for feature, coef in sorted(zip(feature_cols, model.coef_), key=lambda x: x[1], reverse=True):
            f.write(f"{feature}: {coef:.4f}\n")
        f.write(f"Intercept: {model.intercept_:.4f}\n")

    print(f"Metrics and coefficients saved to {txt_file}")

    # ------------------------
    # 4. Scatterplot: Actual vs Predicted
    # ------------------------
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y, y=y_pred)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
    plt.xlabel("Actual Wins")
    plt.ylabel("Predicted Wins")
    plt.title("Actual vs Predicted Wins (Team-level)")
    plt.grid(True)
    scatter_file = os.path.join(output_folder, f"actual_vs_predicted_{goal_rating}_{season_type}.png")
    plt.savefig(scatter_file)
    plt.close()
    print(f"Scatterplot saved to {scatter_file}")

    # ------------------------
    # 5. Residual plot
    # ------------------------
    residuals = y - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=15)
    plt.xlabel("Residual (Actual – Predicted)")
    plt.title("Distribution of Residuals")
    plt.grid(True)
    residual_file = os.path.join(output_folder, f"residuals_distribution_{goal_rating}_{season_type}.png")
    plt.savefig(residual_file)
    plt.close()
    print(f"Residual plot saved to {residual_file}")

def running_correlation(df, season_type):
    ## correlation

    output_folder = "model_outputs"
    os.makedirs(output_folder, exist_ok=True)
    cols = ["OFF_RATING", "DEF_RATING", "NET_RATING","W"]

    # Compute correlation matrix
    corr_matrix = df[cols].corr()

    # ------------------------
    # 3. Save correlation matrix as CSV
    # ------------------------
    corr_csv_file = os.path.join(output_folder, "correlation_matrix.csv")
    corr_matrix.to_csv(corr_csv_file)
    print(f"Correlation matrix saved to {corr_csv_file}")

    # ------------------------
    # 4. Save correlation heatmap as PNG
    # ------------------------
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    corr_png_file = os.path.join(output_folder, f"correlation_matrix_{season_type}.png")
    plt.savefig(corr_png_file)
    plt.close()
    print(f"Correlation heatmap saved to {corr_png_file}")

running_regression(df_percentiles_offensive_rs, df_ratings_rs, "OFF_RATING", "Regular Season")
running_regression(df_percentiles_defensive_rs, df_ratings_rs, "DEF_RATING", "Regular Season")
running_regression(df_percentiles_offensive_defensive_rs, df_ratings_rs, "NET_RATING", "Regular Season")
running_correlation(df_ratings_rs, "Regular Season")


running_regression(df_percentiles_offensive_po, df_ratings_po, "OFF_RATING", "Playoffs")
running_regression(df_percentiles_defensive_po, df_ratings_po, "DEF_RATING", "Playoffs")
running_regression(df_percentiles_offensive_defensive_po, df_ratings_po, "NET_RATING", "Playoffs")
running_correlation(df_ratings_po, "Playoffs")





