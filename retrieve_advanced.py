from nba_api.stats.endpoints import leaguedashplayerstats

player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
    season="2024-25",
    league_id_nullable="00",
    measure_type_detailed_defense="Usage",
    rank="Y",
    season_type_all_star="Regular Season"
).get_data_frames()[0]

print(player_stats.columns)

# -----------------------------
# 5. Features definieren
# -----------------------------
features = ['PLAYER_ID', 'PLAYER_NAME', 'NICKNAME', 'TEAM_ID', 'TEAM_ABBREVIATION',
       'AGE', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'USG_PCT', 'PCT_FGM', 'PCT_FGA',
       'PCT_FG3M', 'PCT_FG3A', 'PCT_FTM', 'PCT_FTA', 'PCT_OREB', 'PCT_DREB',
       'PCT_REB', 'PCT_AST', 'PCT_TOV', 'PCT_STL', 'PCT_BLK', 'PCT_BLKA',
       'PCT_PF', 'PCT_PFD', 'PCT_PTS', 'MIN_AVG']

filtered_players = (
    player_stats
    .sort_values(["TEAM_ABBREVIATION", "GP"], ascending=[True, False])
    .groupby("TEAM_ABBREVIATION", group_keys=False)
    .head(10)          # Top 10 nach GP je Team
    .copy()
)

avg_cols = [
    'MIN']

for col in avg_cols:
    filtered_players[f"{col}_AVG"] = filtered_players[col] / filtered_players["GP"]


X = filtered_players[features].fillna(0)


print(X.columns)

print("\n--- DataFrame mit Features ---")
print(X.head())

# -----------------------------
# 6. DataFrame als CSV speichern
# -----------------------------
csv_file = "nba_2024_25_players_advanced.csv"
X.to_csv(csv_file, index=False)
print(f"DataFrame gespeichert als: {csv_file}")