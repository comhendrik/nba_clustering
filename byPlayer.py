import time
from nba_api.stats.endpoints import synergyplaytypes, leaguestandingsv3
import pandas as pd
import statsmodels.api as sm

def get_team_playtype_records(season: str, playtypes: list):
    dfs = []
    for type_group in ["Offensive", "Defensive"]:
        for pt in playtypes:
            ep = synergyplaytypes.SynergyPlayTypes(
                league_id="00",
                play_type_nullable=pt,
                player_or_team_abbreviation="P",
                season=season,
                season_type_all_star="Regular Season",
                type_grouping_nullable=type_group
            )
            df_pt = ep.get_data_frames()[0]
            df2 = df_pt[["SEASON_ID", "TEAM_ID", "PLAY_TYPE"]].copy()
            df2["PLAY_TYPE"] = df2["PLAY_TYPE"].str.lower() + "_" + type_group.lower()
            # Rename PLAY_TYPE column to differentiate
            dfs.append(df2)
            time.sleep(1)  # to avoid hitting rate limits

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all

def get_team_wins(season: str):
    ep = leaguestandingsv3.LeagueStandingsV3(
        season = season,
        league_id = "00",
        season_type = "Regular Season"
    )
    df = ep.get_data_frames()[0]
    df = df.rename(columns={"TeamID": "TEAM_ID", "SeasonID": "SEASON_ID"})
    return df[["SEASON_ID", "TEAM_ID", "WINS"]]

def build_dataset(season: str, playtypes: list):
    df_all = get_team_playtype_records(season, playtypes)
    counts = df_all.groupby(["SEASON_ID", "TEAM_ID", "PLAY_TYPE"]).size().reset_index(name="PLAY_COUNT")
    counts_wide = counts.pivot_table(
        index=["SEASON_ID", "TEAM_ID"],
        columns="PLAY_TYPE",
        values="PLAY_COUNT",
        fill_value=0
    ).reset_index()

    
    df_wins = get_team_wins(season)
    df_merge = pd.merge(counts_wide, df_wins, on=["SEASON_ID", "TEAM_ID"])

    df_merge.columns = df_merge.columns.str.lower()

    return df_merge

def run_regression(df, playtypes: list):
    # Assuming playtypes exactly match the columns in df
    feature_cols = playtypes  # e.g. ["Isolation", "Cut", "Spotup", ...]
    X = df[feature_cols].astype(float)  # convert counts to numeric
    y = df["WINS"].astype(float)
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

if __name__ == '__main__':
    season = "2024-25"
    playtypes = ["Isolation", "Cut", "Spotup", "Postup", "Handoff", "Transition", "PRBallHandler", "PRRollman", "OffScreen", "Misc"]

    # Convert playtypes to uppercase
    type_groups = ["offensive", "defensive"]
    playtypes_reg = ['isolation_offensive', 'isolation_defensive', 'cut_offensive', 'spotup_offensive', 'spotup_defensive', 'postup_offensive', 'postup_defensive', 'handoff_offensive','handoff_defensive', 'transition_offensive', 'prballhandler_offensive', 'prballhandler_defensive','prrollman_offensive', 'prrollman_defensive', 'offscreen_offensive', 'offscreen_defensive', 'misc_offensive']
    df = build_dataset(season, playtypes)
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # Assuming df is your DataFrame with play type counts and 'WINS' as the target variable
    # Ensure all column names are lowercase

    # Define feature columns (play types) and target variable
    X = df[playtypes_reg]
    y = df['wins']

    # Initialize and train the linear regression model
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    # Make predictions on the test set
    y_pred = model.predict(X)

    # Evaluate the model
    # Metriken berechnen
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y , y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.2f}")
    print(f"MSE: {mse:.2f}")

    # Koeffizienten und Intercept anzeigen
    # Ensure playtypes_reg and model.coef_ are aligned
    print("Koeffizienten:")
    for feature, coef in sorted(zip(playtypes_reg, model.coef_), key=lambda x: playtypes_reg.index(x[0])):
        print(f"{feature}: {coef:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")



    import seaborn as sns

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y, y=y_pred)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
    plt.xlabel("Actual Wins")
    plt.ylabel("Predicted Wins")
    plt.title("Actual vs Predicted Wins (Test Set)")
    plt.grid(True)
    plt.show()

    # Plot residuals
    residuals = y - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=15)
    plt.xlabel("Residual (Actual – Predicted)")
    plt.title("Distribution of Residuals")
    plt.grid(True)
    plt.show()

from scipy.optimize import minimize

# Modellkoeffizienten aus LinearRegression
coefs = model.coef_
intercept = model.intercept_

def wins_to_maximize(x):
    # x = array der Playtypes counts
    return -(intercept + sum(coefs[i] * x[i] for i in range(len(x))))  # negative für Minimierer

# Startwert (aktuelle Playtype Counts)
x0 = X.iloc[0].values

res = minimize(wins_to_maximize, x0, bounds=[(0, 100)]*len(x0))  # z.B. 0-100 Spielzüge
print("Optimale Playtype Counts:", res.x)


