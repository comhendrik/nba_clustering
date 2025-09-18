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
cluster_size = 3

# ==============================================
# 1. CSV laden
# ==============================================
if os.path.exists(csv_file):
    print(f"Lade Daten aus CSV: {csv_file}")
    filtered_players = pd.read_csv(csv_file)
else:
    raise FileNotFoundError(f"CSV {csv_file} nicht gefunden. Bitte zuerst die Daten über die API abrufen.")

# ==============================================
# 2. Features für Clustering
# ==============================================
features = ['USG_PCT', 'PCT_FGM',
       'PCT_FG3M', 'PCT_FTM', 'PCT_OREB', 'PCT_DREB', 'PCT_AST', 'PCT_TOV', 'PCT_STL', 'PCT_BLK',
       'PCT_PF', 'PCT_PFD', 'PCT_PTS']

import seaborn as sns
import matplotlib.pyplot as plt

# Wähle nur die vorhandenen Features
available_features = [col for col in features if col in filtered_players.columns]

plt.figure(figsize=(14, 6))
sns.boxplot(data=filtered_players[available_features])
plt.title("Boxplots aller Features")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


