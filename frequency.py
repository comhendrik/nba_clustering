import time
import pandas as pd
from nba_api.stats.endpoints import synergyplaytypes, leaguestandingsv3
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

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
    return get_team_playtype_percentiles(season, playtypes, ["Offensive"], season_type)

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

import os
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import os
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def running_regression(df_percentiles, df_rating, goal_rating, season_type):
    # Merge datasets
    df = pd.merge(df_percentiles, df_rating, on=["TEAM_ID"], how="left")

    # Select features
    exclude_cols = ["TEAM_ABBREVIATION","TEAM_ID", "TEAM_NAME", "OFF_RATING", "DEF_RATING",
                    "NET_RATING", "OffensiveRank", "DefensiveRank", "NetRank", "W"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols]
    y = df[goal_rating]

    # Add constant for intercept
    X_sm = sm.add_constant(X)

    # ------------------------
    # 1. Initial regression
    # ------------------------
    sm_model = sm.OLS(y, X_sm).fit()
    sm_y_pred = sm_model.predict(X_sm)

    sm_mae = mean_absolute_error(y, sm_y_pred)
    sm_mse = mean_squared_error(y, sm_y_pred)
    sm_r2 = r2_score(y, sm_y_pred)

    output_folder = "frequency/regression_outputs"
    os.makedirs(output_folder, exist_ok=True)

    plt.figure(figsize=(8,6))
    plt.scatter(y, sm_y_pred, color='blue')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Predicted vs Actual {goal_rating} ({season_type})")
    plt.tight_layout()
    plot_file = os.path.join(output_folder, f"predicted_vs_actual_initial_{goal_rating}_{season_type}.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Predicted vs actual plot saved to {plot_file}")

    # ------------------------
    # 2. Backward elimination (p > 0.05)
    # ------------------------
    X_be = X_sm.copy()
    iteration = 1
    while True:
        model_be = sm.OLS(y, X_be).fit()
        pvals = model_be.pvalues.drop('const', errors='ignore')
        if len(pvals) == 0 or (pvals <= 0.05).all():
            break
        # Remove variable with largest p-value
        worst_var = pvals.idxmax()
        X_be = X_be.drop(columns=[worst_var])
        iteration += 1

    sm_model_sig = sm.OLS(y, X_be).fit()
    sm_y_pred_sig = sm_model_sig.predict(X_be)

    sm_mae_sig = mean_absolute_error(y, sm_y_pred_sig)
    sm_mse_sig = mean_squared_error(y, sm_y_pred_sig)
    sm_r2_sig = r2_score(y, sm_y_pred_sig)

    # ------------------------
    # 3. Output folder & text file
    # ------------------------
    
    txt_file = os.path.join(output_folder, f"metrics_and_coefficients_{goal_rating}_{season_type}.txt")

    with open(txt_file, "w") as f:
        # Initial regression
        f.write("=== Initial Regression (all variables) ===\n")
        f.write(f"MAE: {sm_mae:.2f}\nR²: {sm_r2:.2f}\nMSE: {sm_mse:.2f}\n")
        f.write(f"F-Statistik: {sm_model.fvalue:.3f}, p-Wert: {sm_model.f_pvalue:.4f}\n\n")
        f.write("Koeffizienten:\n")
        for var, coef, pval, tval in zip(sm_model.params.index, sm_model.params.values,
                                         sm_model.pvalues.values, sm_model.tvalues.values):
            f.write(f"{var}: {coef:.4f}, p={pval:.4f}, t={tval:.2f}\n")

        # Regression after backward elimination
        f.write("\n=== Regression after Backward Elimination (p <= 0.05) ===\n")
        f.write(f"MAE: {sm_mae_sig:.2f}\nR²: {sm_r2_sig:.2f}\nMSE: {sm_mse_sig:.2f}\n")
        f.write(f"F-Statistik: {sm_model_sig.fvalue:.3f}, p-Wert: {sm_model_sig.f_pvalue:.4f}\n\n")
        f.write("Koeffizienten:\n")
        for var, coef, pval, tval in zip(sm_model_sig.params.index, sm_model_sig.params.values,
                                         sm_model_sig.pvalues.values, sm_model_sig.tvalues.values):
            f.write(f"{var}: {coef:.4f}, p={pval:.4f}, t={tval:.2f}\n")

    print(f"Metrics and coefficients saved to {txt_file}")

    # ------------------------
    # 4. Plot predicted vs actual for final model
    # ------------------------
    plt.figure(figsize=(8,6))
    plt.scatter(y, sm_y_pred_sig, color='blue')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Predicted vs Actual {goal_rating} ({season_type})")
    plt.tight_layout()
    plot_file = os.path.join(output_folder, f"predicted_vs_actual_{goal_rating}_{season_type}.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Predicted vs actual plot saved to {plot_file}")

    # ------------------------
    # 5. Residual plot
    # ------------------------
    residuals = y - sm_y_pred_sig
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

    output_folder = "frequency/correlation_outputs"
    os.makedirs(output_folder, exist_ok=True)
    

    exclude_cols = ["TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME"]  # example columns

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=["number"])

    # Remove specific columns
    numeric_df = numeric_df.drop(columns=[col for col in exclude_cols if col in numeric_df.columns])

    # Compute correlation matrix
    corr_matrix = numeric_df.corr()

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


#running_regression(df_percentiles_offensive_po, df_ratings_po, "OFF_RATING", "Playoffs")
#running_regression(df_percentiles_defensive_po, df_ratings_po, "DEF_RATING", "Playoffs")
#running_regression(df_percentiles_offensive_defensive_po, df_ratings_po, "NET_RATING", "Playoffs")
#running_correlation(df_ratings_po, "Playoffs")


running_correlation(df_percentiles_defensive_rs, "Defensive Regular Season")
running_correlation(df_percentiles_offensive_rs, "Offensive Regular Season")
running_correlation(df_percentiles_offensive_defensive_rs, "Offensive and Defensive Regular Season")





