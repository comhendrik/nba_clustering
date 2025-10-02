
from nba_api.stats.endpoints import leaguedashteamstats
ep = leaguedashteamstats.LeagueDashTeamStats(
    season="2024-25",
    league_id_nullable="00",
    season_type_all_star="Playoffs",
    measure_type_detailed_defense="Advanced"  # includes OFFRTG/DEFRTG/NETRTG
)

df = ep.get_data_frames()[0]

print(df.columns)