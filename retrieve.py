# -----------------------------
# 1. Imports
# -----------------------------
from nba_api.stats.endpoints import leaguedashplayerstats, commonplayerinfo
from nba_api.stats.static import players
import pandas as pd
import time

# -----------------------------
# 2. Spieler-Stats abrufen (Saison 2024-25, NBA only)
# -----------------------------
player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
    season="2024-25",
    league_id_nullable="00"  # 00 = NBA, 10 = WNBA, 20 = G-League
).get_data_frames()[0]

# Nur aktive Spieler mit >= 20 Spielen
filtered_players = player_stats[player_stats["GP"] >= 20].copy()

# -----------------------------
# 3. Körperliche Attribute ergänzen
# -----------------------------
heights, weights = [], []


i = 0
for pid in filtered_players["PLAYER_ID"]:
    try:
        print(f"Verarbeite Spieler {i+1}/{len(filtered_players)} mit PLAYER_ID {pid}")
        i += 1
        info = commonplayerinfo.CommonPlayerInfo(player_id=pid).get_data_frames()[0]
        height_str = info.at[0, "HEIGHT"]  # z.B. "6-7"
        weight = info.at[0, "WEIGHT"]

        # Height von "6-7" in Inches konvertieren
        if isinstance(height_str, str) and "-" in height_str:
            feet, inches = height_str.split("-")
            height_in = int(feet) * 12 + int(inches)
        else:
            height_in = None

        heights.append(height_in)
        weights.append(weight)

        time.sleep(0.3)  # API-Rate Limit

    except Exception as e:
        print(f"Fehler bei PLAYER_ID {pid}: {e}")
        heights.append(None)
        weights.append(None)

filtered_players["HEIGHT"] = heights
filtered_players["WEIGHT"] = weights

# -----------------------------
# 4. Averages berechnen
# -----------------------------
avg_cols = [
    'MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
    'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA',
    'PF', 'PFD', 'PTS', 'PLUS_MINUS'
]

for col in avg_cols:
    filtered_players[f"{col}_AVG"] = filtered_players[col] / filtered_players["GP"]

# -----------------------------
# 5. Features definieren
# -----------------------------
features = [
    'PLAYER_NAME', 'TEAM_ABBREVIATION',
    'AGE', 'GP', 'W', 'L', 'W_PCT',
    'MIN_AVG', 'FGM_AVG', 'FGA_AVG', 'FG_PCT',
    'FG3M_AVG', 'FG3A_AVG', 'FG3_PCT',
    'FTM_AVG', 'FTA_AVG', 'FT_PCT',
    'OREB_AVG', 'DREB_AVG', 'REB_AVG', 'AST_AVG',
    'TOV_AVG', 'STL_AVG', 'BLK_AVG', 'BLKA_AVG',
    'PF_AVG', 'PFD_AVG', 'PTS_AVG', 'PLUS_MINUS_AVG',
    'NBA_FANTASY_PTS', 'DD2', 'TD3',
    'GP_RANK', 'W_RANK', 'L_RANK', 'W_PCT_RANK', 'MIN_RANK',
    'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK', 'FG3M_RANK', 'FG3A_RANK',
    'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK', 'FT_PCT_RANK',
    'OREB_RANK', 'DREB_RANK', 'REB_RANK', 'AST_RANK',
    'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK',
    'PF_RANK', 'PFD_RANK', 'PTS_RANK', 'PLUS_MINUS_RANK',
    'NBA_FANTASY_PTS_RANK', 'DD2_RANK', 'TD3_RANK',
    'TEAM_COUNT',
    'HEIGHT',
    'WEIGHT'
]

X = filtered_players[features].fillna(0)

print("\n--- DataFrame mit Features ---")
print(X.head())

# -----------------------------
# 6. DataFrame als CSV speichern
# -----------------------------
csv_file = "nba_2024_25_players.csv"
X.to_csv(csv_file, index=False)
print(f"DataFrame gespeichert als: {csv_file}")
