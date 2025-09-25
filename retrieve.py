from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd
from functools import reduce

# ----- Settings -----
SEASON = "2022-23"
SEASON_TYPE = "Regular Season"
OUTPUT_FILE = "nba_player_stats_2022_23.csv"

# ----- Measure types -----
measure_types = ["Base", "Advanced", "Misc", "Scoring", "Usage", "Defense"]

# ----- Fetch all measure types -----
dfs = []

for measure in measure_types:
    print(f"Fetching measure type: {measure}")
    df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=SEASON,
        season_type_all_star=SEASON_TYPE,
        measure_type_detailed_defense=measure
    ).get_data_frames()[0]

    # Keep PLAYER_ID and PLAYER_NAME without suffix; add suffix to other columns only
    if measure != "Base":
        suffix = f"_{measure}"
        df = df.rename(columns={col: f"{col}{suffix}" for col in df.columns if col not in ["PLAYER_ID", "PLAYER_NAME"]})
    
    if "MIN" in df.columns and "TEAM_ID" in df.columns:
        df = df.groupby("TEAM_ID", group_keys=False).apply(lambda x: x.nlargest(10, "MIN"))

    dfs.append(df)

# ----- Merge all DataFrames on PLAYER_ID and PLAYER_NAME -----
merged_df = reduce(lambda left, right: pd.merge(left, right, on=["PLAYER_ID", "PLAYER_NAME"]), dfs)



avg_metrics = [
    "MIN",
    "FG3M",
    "FGM",
    "FTM",
    "OREB",
    "DREB",
    "AST",
    "TOV",
    "STL",
    "BLK",
    "PF",
    "PFD",
    "PTS_OFF_TOV_Misc",
    "PTS_2ND_CHANCE_Misc",
    "PTS_FB_Misc",
    "PTS_PAINT_Misc",
    "OPP_PTS_OFF_TOV_Misc",
    "OPP_PTS_2ND_CHANCE_Misc",
    "OPP_PTS_FB_Misc",
    "OPP_PTS_PAINT_Misc",
    "OPP_PTS_OFF_TOV_Defense", "OPP_PTS_FB_Defense", 
    "OPP_PTS_2ND_CHANCE_Defense", "OPP_PTS_PAINT_Defense", 
    "POSS_Advanced",
    "PTS",
    "REB"
]

for col in avg_metrics:
    if col in merged_df.columns:
        merged_df[col + "_AVG"] = merged_df.apply(
            lambda row: row[col] / row["GP"] if row["GP"] > 0 else 0, axis=1
        )


# ----- Save to CSV -----
merged_df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved {len(merged_df.columns)+1} metrics (including MIN_AVG) to {OUTPUT_FILE}")
