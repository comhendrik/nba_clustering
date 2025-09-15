# -----------------------------
# 1. Imports
# -----------------------------
from nba_api.stats.endpoints import leaguedashteamstats
import pandas as pd

# -----------------------------
# 2. Team-Stats abrufen (Saison 2024-25, NBA)
# -----------------------------
team_stats = leaguedashteamstats.LeagueDashTeamStats(
    season="2024-25",
    league_id_nullable="00"  # NBA
).get_data_frames()[0]

# -----------------------------
# 3. Pro-Spiel Averages berechnen
# -----------------------------
# Die Stats sind oft schon totals und GP = games played
avg_cols = ['OREB', 'DREB', 'AST', 'TOV', 'STL', 'BLK', 'PF', 'PFD', 'PTS']
for col in avg_cols:
    team_stats[f"{col}_AVG"] = team_stats[col] / team_stats["GP"]

# -----------------------------
# 4. Relevante Features auswählen
# -----------------------------
features = ['TEAM_NAME', 'GP', 'W', 'L', 'W_PCT'] + [f"{c}_AVG" for c in avg_cols]
team_stats_avg = team_stats[features]

print("\n--- Team Stats mit pro-Spiel Averages ---")
print(team_stats_avg.head())

# -----------------------------
# 5. Korrelation mit Teamerfolg (Siege)
# -----------------------------
correlation = team_stats_avg[[f"{c}_AVG" for c in avg_cols] + ['W']].corr()['W'].sort_values(ascending=False)
print("\nKorrelation der Stats mit Siegen:")
print(correlation)

# -----------------------------
# 6. Optional: Visualisierung
# -----------------------------
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.barplot(x=correlation.index, y=correlation.values)
plt.xticks(rotation=45)
plt.title("Korrelation pro-Spiel Stats vs. Siege (2024-25 NBA)")
plt.ylabel("Korrelationskoeffizient")
plt.show()
