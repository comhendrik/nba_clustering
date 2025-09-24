# -----------------------------
# 1. Imports
# -----------------------------
from nba_api.stats.endpoints import leaguedashteamstats
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# -----------------------------
# 2. Output directory & date
# -----------------------------
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# -----------------------------
# 3. Team-Stats abrufen (Saison 2024-25, NBA)
# -----------------------------
team_stats = leaguedashteamstats.LeagueDashTeamStats(
    season="2024-25",
    league_id_nullable="00"  # NBA
).get_data_frames()[0]

# -----------------------------
# 4. Pro-Spiel Averages berechnen
# -----------------------------

avg_cols = ["OREB", "DREB", "AST", "TOV", "STL", "BLK", "PF", "PFD", "FGM", "FG3M", "FTM"]
for col in avg_cols:
    team_stats[f"{col}_AVG"] = team_stats[col] / team_stats["GP"]

# -----------------------------
# 5. Relevante Features auswählen
# -----------------------------
features = ['TEAM_NAME', 'GP', 'W', 'L', 'W_PCT'] + [f"{c}_AVG" for c in avg_cols]
team_stats_avg = team_stats[features]

print("\n--- Team Stats mit pro-Spiel Averages ---")
print(team_stats_avg.head())

# -----------------------------
# 6. Korrelation mit Teamerfolg (Siege)
# -----------------------------
correlation = team_stats_avg[[f"{c}_AVG" for c in avg_cols] + ['W']].corr()['W'].sort_values(ascending=False)
print("\nKorrelation der Stats mit Siegen:")
print(correlation)

# -----------------------------
# 7. Visualisierung & Speichern
# -----------------------------
plt.figure(figsize=(10,6))
sns.barplot(x=correlation.index, y=correlation.values)
plt.xticks(rotation=45)
plt.title("Korrelation pro-Spiel Stats vs. Siege (2024-25 NBA)")
plt.ylabel("Korrelationskoeffizient")

plot_path = output_dir / f"correlation_plot.png"
plt.savefig(plot_path, bbox_inches="tight", dpi=300)
plt.close()

print(f"\nPlot saved to {plot_path}")

