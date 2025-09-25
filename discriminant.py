import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nba_api.stats.endpoints import leaguedashplayerstats, leaguestandings
from nba_api.stats.static import teams
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error
)
import statsmodels.api as sm

# ==============================================
# Einstellungen
# ==============================================
season = "2023-24"
season_eval = "2024-25"   # Für Regression mit Siegen
output_dir = "discri_outputs"
os.makedirs(output_dir, exist_ok=True)

report_path = os.path.join(output_dir, "discriminant_analysis_report.txt")
with open(report_path, "w") as report:
    report.write(f"=== NBA {season} Player Archetype Segmentation (Usage-based) ===\n\n")

# ==============================================
# 1. Spieler-Daten laden (mit Usage-Metriken)
# ==============================================
player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
    season=season,
    league_id_nullable="00",
    measure_type_detailed_defense="Usage",
    rank="Y",
    season_type_all_star="Regular Season"
).get_data_frames()[0]

# Features mit PCT_* (alle Usage-basiert)
base_metrics = [
    'PCT_OREB', 'PCT_DREB', 'PCT_AST', 'PCT_TOV',
    'PCT_STL', 'PCT_BLK', 'PCT_PTS', 'PCT_FG3M', 'PCT_FGM'
]



# ==============================================
# 3. Rename columns in CSV (from *_USAGE → PCT_*)
# ==============================================
rename_map = {f"{m}": f"{m}_Usage" for m in base_metrics}
player_stats = player_stats.rename(columns=rename_map)

prediction_features = [f"{m}_Usage" for m in base_metrics]

# ==============================================
# 2. Quartil-Schwellen berechnen
# ==============================================
quartile_thresholds = {col: player_stats[col].quantile(0.75) for col in prediction_features}

# ==============================================
# 3. Multi-Label Archetype-Zuweisung
# ==============================================
def segment_player_multilabel(row):
    labels = []

    # --- Spezifische Archetypen ---
    if row["PCT_FG3M_Usage"] >= quartile_thresholds["PCT_FG3M_Usage"]:
        labels.append("Shooter")

    if row["PCT_FGM_Usage"] >= quartile_thresholds["PCT_FGM_Usage"]:
        labels.append("Slasher")

    if (
        row["PCT_STL_Usage"] >= quartile_thresholds["PCT_STL_Usage"]
        or row["PCT_BLK_Usage"] >= quartile_thresholds["PCT_BLK_Usage"]
    ):
        labels.append("Defensive")

    if (
        row["PCT_OREB_Usage"] >= quartile_thresholds["PCT_OREB_Usage"]
        or row["PCT_DREB_Usage"] >= quartile_thresholds["PCT_DREB_Usage"]
    ):
        labels.append("Rebounder/Big")

    if row["PCT_AST_Usage"] >= quartile_thresholds["PCT_AST_Usage"]:
        labels.append("Playmaker")

    if not labels:
        labels.append("Other")

    # --- Smart collapse to single label ---
    if len(labels) > 1:
        # Example rule: Shooter + Slasher = Offensive
        if "Shooter" in labels and "Slasher" in labels:
            return ["Offensive"]

        # Example: Defensive + Rebounder/Big = Anchor
        if "Defensive" in labels and "Rebounder/Big" in labels:
            return ["Anchor"]

        # Example: Playmaker + Shooter = Offensive Playmaker
        if ("Playmaker" in labels and "Shooter" in labels) or ("Playmaker" in labels and "Slasher" in labels):
            return ["Offensive Playmaker"]

        # Example: Playmaker + Defensive = Two-Way
        if ("Shooter" in labels and "Defensive" in labels) or ("Slasher" in labels and "Defensive" in labels):
            return ["Two-Way"]

        # Fallback: if no smart combo defined → take the first label
        return [labels[0]]

    return labels


# Anwenden
player_stats["Archetype"] = player_stats.apply(segment_player_multilabel, axis=1)
player_stats_expanded = player_stats.explode("Archetype").reset_index(drop=True)

# ==============================================
# 4. LDA Modell trainieren
# ==============================================
X = player_stats_expanded[prediction_features].fillna(0)
y = player_stats_expanded["Archetype"]

lda = LinearDiscriminantAnalysis()
lda.fit(X, y)

with open(report_path, "a") as report:
    report.write("=== Trainierte Archetypen (Beispiele) ===\n")
    report.write(player_stats_expanded[["PLAYER_NAME", "Archetype"]].head(15).to_string(index=False))
    report.write("\n\n")

# ==============================================
# 5. Archetypen für Advanced Dataset (2024-25) vorhersagen
# ==============================================
csv_file = "nba_player_stats_2024_25.csv"
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"{csv_file} nicht gefunden. Bitte zuerst API-Daten abrufen.")

df = pd.read_csv(csv_file)
df["Archetype"] = lda.predict(df[prediction_features].fillna(0))

# ==============================================
# 6. Archetype Grouping
# ==============================================
team_archetype_minutes = (
    df.groupby(["TEAM_ABBREVIATION", "Archetype"])["MIN_AVG"]
    .sum()
    .unstack(fill_value=0)
)

with open(report_path, "a") as report:
    report.write("=== Minuten pro Team & Archetyp ===\n")
    report.write(team_archetype_minutes.to_string())
    report.write("\n\n")

# ==============================================
# 8. Regression Siege ~ Archetyp-Minuten
# ==============================================
standings = leaguestandings.LeagueStandings(season=season_eval, league_id="00").get_data_frames()[0]
nba_teams = teams.get_teams()
team_map = {t["id"]: t["abbreviation"] for t in nba_teams}
standings["TEAM_ABBREVIATION"] = standings["TeamID"].map(team_map)
team_wins = standings[["TEAM_ABBREVIATION", "WINS"]]

merged = team_archetype_minutes.merge(team_wins, on="TEAM_ABBREVIATION", how="inner")
X_reg = merged.drop(columns=["WINS", "TEAM_ABBREVIATION"])
y_reg = merged["WINS"]

reg = LinearRegression()
reg.fit(X_reg, y_reg)
y_pred = reg.predict(X_reg)

# Metriken
r2 = r2_score(y_reg, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_reg) - 1) / (len(y_reg) - X_reg.shape[1] - 1)
mse = mean_squared_error(y_reg, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_reg, y_pred)

coefficients = pd.Series(reg.coef_, index=X_reg.columns).sort_values(ascending=False)

with open(report_path, "a") as report:
    report.write("=== Regression: Siege ~ Archetyp-Minuten ===\n")
    report.write(coefficients.to_string())
    report.write(f"\nIntercept: {reg.intercept_:.3f}\n")
    report.write(f"R²: {r2:.4f}, Adjusted R²: {adj_r2:.4f}\n")
    report.write(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}\n\n")

# Regression Fit Plot
plt.figure(figsize=(7, 5))
sns.regplot(x=y_reg, y=y_pred, ci=None, line_kws={"color": "red"})
plt.xlabel("Tatsächliche Siege")
plt.ylabel("Vorhergesagte Siege")
plt.title("Regression: Siege ~ Archetyp-Minuten")
plt.annotate(f"R²={r2:.3f}, Adj.R²={adj_r2:.3f}\nRMSE={rmse:.2f}",
             xy=(0.05, 0.85), xycoords="axes fraction",
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "regression_fit.png"), dpi=300)
plt.close()

# ==============================================
# 9. Segmentverteilung Plot
# ==============================================

archetype_counts = df["Archetype"].value_counts().reset_index()
archetype_counts.columns = ["Archetype", "Count"]  # fix column names

archetype_counts.set_index("Archetype")["Count"].sort_values().plot(
    kind="bar",
    color="skyblue",
    edgecolor="black",
    figsize=(8, 5),
    title="Archetype Counts"
)
plt.xlabel("Archetyp")
plt.ylabel("Spieleranzahl")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "segment_distribution.png"), dpi=300)
plt.close()

print(f"Analyse abgeschlossen. Alle Ergebnisse in '{report_path}' und Plots in '{output_dir}' gespeichert.")
