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

# Für jedes Team die 10 Spieler mit den meisten Spielen auswählen
filtered_players = (
    player_stats
    .sort_values(["TEAM_ABBREVIATION", "GP"], ascending=[True, False])
    .groupby("TEAM_ABBREVIATION", group_keys=False)
    .head(10)          # Top 10 nach GP je Team
    .copy()
)


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
    'PF', 'PFD', 'PTS'
]

for col in avg_cols:
    filtered_players[f"{col}_AVG"] = filtered_players[col] / filtered_players["GP"]

rate_cols = [
    'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
    'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL',
    'BLK', 'BLKA', 'PF', 'PFD', 'PTS'
]

# pro Spieler Minuten abrufen
minutes = filtered_players['MIN']

for col in rate_cols:
    filtered_players[f"{col}_PER48"] = (filtered_players[col] / minutes) * 48
    filtered_players[f"{col}_PER40"] = (filtered_players[col] / minutes) * 40
    filtered_players[f"{col}_PER36"] = (filtered_players[col] / minutes) * 36


# -----------------------------
# 5. Features definieren
# -----------------------------
features = [
    "PLAYER_NAME", "TEAM_ABBREVIATION",
    "AGE", "GP", "W", "L", "W_PCT",
    "MIN_AVG",
    "FGM_AVG", "FGM_PER48", "FGM_PER40", "FGM_PER36",
    "FGA_AVG", "FGA_PER48", "FGA_PER40", "FGA_PER36",
    "FG_PCT",
    "FG3M_AVG", "FG3M_PER48", "FG3M_PER40", "FG3M_PER36",
    "FG3A_AVG", "FG3A_PER48", "FG3A_PER40", "FG3A_PER36",
    "FG3_PCT",
    "FTM_AVG", "FTM_PER48", "FTM_PER40", "FTM_PER36",
    "FTA_AVG", "FTA_PER48", "FTA_PER40", "FTA_PER36",
    "FT_PCT",
    "OREB_AVG", "OREB_PER48", "OREB_PER40", "OREB_PER36",
    "DREB_AVG", "DREB_PER48", "DREB_PER40", "DREB_PER36",
    "REB_AVG", "REB_PER48", "REB_PER40", "REB_PER36",
    "AST_AVG", "AST_PER48", "AST_PER40", "AST_PER36",
    "TOV_AVG", "TOV_PER48", "TOV_PER40", "TOV_PER36",
    "STL_AVG", "STL_PER48", "STL_PER40", "STL_PER36",
    "BLK_AVG", "BLK_PER48", "BLK_PER40", "BLK_PER36",
    "BLKA_AVG", "BLKA_PER48", "BLKA_PER40", "BLKA_PER36",
    "PF_AVG", "PF_PER48", "PF_PER40", "PF_PER36",
    "PFD_AVG", "PFD_PER48", "PFD_PER40", "PFD_PER36",
    "PTS_AVG", "PTS_PER48", "PTS_PER40", "PTS_PER36",
    "NBA_FANTASY_PTS", "DD2", "TD3",
    "GP_RANK", "W_RANK", "L_RANK", "W_PCT_RANK", "MIN_RANK",
    "FGM_RANK", "FGA_RANK", "FG_PCT_RANK", "FG3M_RANK", "FG3A_RANK",
    "FG3_PCT_RANK", "FTM_RANK", "FTA_RANK", "FT_PCT_RANK",
    "OREB_RANK", "DREB_RANK", "REB_RANK", "AST_RANK",
    "TOV_RANK", "STL_RANK", "BLK_RANK", "BLKA_RANK",
    "PF_RANK", "PFD_RANK", "PTS_RANK", "PLUS_MINUS_RANK",
    "NBA_FANTASY_PTS_RANK", "DD2_RANK", "TD3_RANK",
    "TEAM_COUNT",
    "HEIGHT",
    "WEIGHT",
]


X = filtered_players[features].fillna(0)

print("\n--- DataFrame mit Features ---")
print(X.head())

# -----------------------------
# 6. DataFrame als CSV speichern
# -----------------------------
csv_file = "nba_2024_25_players_per.csv"
X.to_csv(csv_file, index=False)
print(f"DataFrame gespeichert als: {csv_file}")
