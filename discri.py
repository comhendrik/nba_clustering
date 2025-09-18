import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from nba_api.stats.endpoints import leaguedashplayerstats

# ==============================================
# 1. Example players dataset (0–1 fractions)
# ==============================================

player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
    season="2023-24",
    league_id_nullable="00",
    measure_type_detailed_defense="Usage",
    rank="Y",
    season_type_all_star="Regular Season"
).get_data_frames()[0]

# ==============================================
# 2. Features for segmentation
# ==============================================
features = ['PCT_OREB', 'PCT_DREB', 'PCT_AST', 'PCT_TOV', 
            'PCT_STL', 'PCT_BLK', 'PCT_PTS', 'PCT_FG3M', 'PCT_FG3A']

# ==============================================
# 3. Compute 4th-quartile thresholds
# ==============================================
quartile_thresholds = {col: player_stats[col].quantile(0.60) for col in features}

# ==============================================
# 4. Multi-label segmentation
# ==============================================
def segment_player_multilabel(row):
    labels = []

    # --- Specific archetypes ---
    if row["PCT_OREB"] >= quartile_thresholds["PCT_OREB"] or row["PCT_DREB"] >= quartile_thresholds["PCT_DREB"]:
        labels.append("Rebounder/Big")

    if row["PCT_AST"] >= quartile_thresholds["PCT_AST"] and row["PCT_PTS"] < quartile_thresholds["PCT_PTS"]:
        labels.append("Playmaker")

    if (
        row["PCT_PTS"] >= quartile_thresholds["PCT_PTS"] / 2
        and (
            row["PCT_FG3M"] >= quartile_thresholds["PCT_FG3M"]
            or row["PCT_FG3A"] >= quartile_thresholds["PCT_FG3A"]
        )
    ):
        labels.append("3&D/Stretch")

    # --- General archetypes ---
    offensive = (
        row["PCT_PTS"] >= quartile_thresholds["PCT_PTS"]
        or row["PCT_AST"] >= quartile_thresholds["PCT_AST"]
    )
    defensive = (
        row["PCT_STL"] >= quartile_thresholds["PCT_STL"]
        or row["PCT_BLK"] >= quartile_thresholds["PCT_BLK"]
        or row["PCT_DREB"] >= quartile_thresholds["PCT_DREB"]
    )

    if offensive and defensive:
        labels.append("Two-Way")
    elif offensive:
        labels.append("Offensive")
    elif defensive:
        labels.append("Defensive")

    if not labels:
        labels.append("Other")

    return labels

# Step 1: assign the list of labels
player_stats["Archetype"] = player_stats.apply(segment_player_multilabel, axis=1)

# Step 2: explode so each label becomes its own row
player_stats_expanded = player_stats.explode("Archetype").reset_index(drop=True)


# ==============================================
# 2. Train LDA
# ==============================================

X = player_stats_expanded[features]
y = player_stats_expanded["Archetype"]
print(y.head())

lda = LinearDiscriminantAnalysis()
lda.fit(X, y)


import os
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from nba_api.stats.endpoints import leaguestandings
from nba_api.stats.static import teams
from sklearn.linear_model import LinearRegression
import seaborn as sns

# ==============================================
# Einstellungen
# ==============================================
csv_file = "nba_2024_25_players_advanced.csv"

# ==============================================
# 1. CSV laden
# ==============================================
if os.path.exists(csv_file):
    print(f"Lade Daten aus CSV: {csv_file}")
    df = pd.read_csv(csv_file)
else:
    raise FileNotFoundError(f"CSV {csv_file} nicht gefunden. Bitte zuerst die Daten über die API abrufen.")

df["Archetype"] = lda.predict(df[features].fillna(0))

import os

# Make sure output directory exists
output_dir = "discri_outputs"
os.makedirs(output_dir, exist_ok=True)

# Path for output TXT
output_path = os.path.join(output_dir, "player_archetypes.txt")

# Save player names and predicted archetypes
with open(output_path, "w") as f:
    for idx, row in df.iterrows():
        f.write(f"{row['PLAYER_NAME']}: {row['Archetype']}\n")

print(f"All player archetypes saved to {output_path}")


# ==============================================
# 5. Create binary indicator columns for each type
# ==============================================
all_types = ["Rebounder/Big", "Playmaker", "3&D/Stretch", "Two-Way", "Offensive", "Defensive", "Other"]
for t in all_types:
    df[t] = df['Archetype'].apply(lambda x: int(t in x))

# ==============================================
# 6. Sum minutes per team and segment
# ==============================================
# Multiply each player's binary type indicators by their minutes first
minutes_weighted = df[all_types].multiply(df["MIN_AVG"], axis=0)

# Then sum per team
team_segment_minutes = df[["TEAM_ABBREVIATION"]].join(minutes_weighted).groupby("TEAM_ABBREVIATION").sum()
team_minutes_path = os.path.join(output_dir, "team_segment_minutes.txt")
team_segment_minutes.to_csv(team_minutes_path)
print(f"Team segment minutes saved to {team_minutes_path}")

# ==============================================
# 7. Regression Wins ~ Segment Minutes
# ==============================================
standings = leaguestandings.LeagueStandings(season="2024-25", league_id="00").get_data_frames()[0]
nba_teams = teams.get_teams()
team_map = {t["id"]: t["abbreviation"] for t in nba_teams}
standings["TEAM_ABBREVIATION"] = standings["TeamID"].map(team_map)
team_wins = standings[["TEAM_ABBREVIATION", "WINS"]]

merged = team_segment_minutes.merge(team_wins, on="TEAM_ABBREVIATION", how="inner")
X_reg = merged.drop(columns=["WINS", "TEAM_ABBREVIATION"])
y_reg = merged["WINS"]

reg = LinearRegression()
reg.fit(X_reg, y_reg)
coefficients = pd.Series(reg.coef_, index=X_reg.columns).sort_values(ascending=False)
regression_path = os.path.join(output_dir, "regression_results.txt")
with open(regression_path, "w") as f:
    f.write("--- Einfluss der Segmente auf Siege (Linear Regression) ---\n")
    f.write(str(coefficients))
    f.write(f"\nIntercept: {reg.intercept_}")
print(f"Regression results saved to {regression_path}")

# ==============================================
# 8. Segment counts plot
# ==============================================
type_counts = {t: df[t].sum() for t in all_types}
pd.Series(type_counts).sort_values().plot(kind="bar", title="Player Types (multi-label)")
plt.xlabel("Player Type")
plt.ylabel("Total Minutes Counted")
plt.show()

