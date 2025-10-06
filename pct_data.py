import os
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score
from nba_api.stats.endpoints import leaguestandings
from nba_api.stats.static import teams
from sklearn.linear_model import LinearRegression
import seaborn as sns

import statsmodels.api as sm

def running_regression(df_to_merge, df_rating, goal_rating, season_type):
    # Merge datasets
    merged = df_to_merge.merge(df_rating, on="TEAM_ABBREVIATION", how="inner")
    X_reg = merged.drop(columns=["WINS", "TEAM_ABBREVIATION"]).copy()
    X_reg.columns = X_reg.columns.astype(str)
    y_reg = merged["WINS"]

    # ------------------------
    # 1. Initial regression
    # ------------------------
    sm_model = sm.OLS(y_reg, X_reg).fit()
    sm_y_pred = sm_model.predict(X_reg)

    sm_mae = mean_absolute_error(y_reg, sm_y_pred)
    sm_mse = mean_squared_error(y_reg, sm_y_pred)
    sm_r2 = r2_score(y_reg, sm_y_pred)

    output_folder = "percentage/regression_outputs"
    os.makedirs(output_folder, exist_ok=True)

    plt.figure(figsize=(8,6))
    plt.scatter(y_reg, sm_y_pred, color='blue')
    plt.plot([y_reg.min(), y_reg.max()], [y_reg.min(), y_reg.max()], 'r--', linewidth=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Predicted vs Actual {goal_rating} ({season_type})")
    plt.tight_layout()
    plot_file = os.path.join(output_folder, f"predicted_vs_actual_initial_{goal_rating}_{season_type}.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Predicted vs actual plot saved to {plot_file}")

    # ------------------------
    # 2. Backward elimination (p > 0.05)
    # ------------------------
    X_be = X_reg.copy()
    iteration = 1
    while True:
        model_be = sm.OLS(y_reg, X_be).fit()
        pvals = model_be.pvalues.drop('const', errors='ignore')
        if len(pvals) == 0 or (pvals <= 0.05).all():
            break
        # Remove variable with largest p-value
        worst_var = pvals.idxmax()
        X_be = X_be.drop(columns=[worst_var])
        iteration += 1

    sm_model_sig = sm.OLS(y_reg, X_be).fit()
    sm_y_pred_sig = sm_model_sig.predict(X_be)

    sm_mae_sig = mean_absolute_error(y_reg, sm_y_pred_sig)
    sm_mse_sig = mean_squared_error(y_reg, sm_y_pred_sig)
    sm_r2_sig = r2_score(y_reg, sm_y_pred_sig)

    # ------------------------
    # 3. Output folder & text file
    # ------------------------
    
    txt_file = os.path.join(output_folder, f"metrics_and_coefficients_{goal_rating}_{season_type}.txt")

    with open(txt_file, "w") as f:
        # Initial regression
        f.write("=== Initial Regression (all variables) ===\n")
        f.write(f"MAE: {sm_mae:.2f}\nR²: {sm_r2:.2f}\nMSE: {sm_mse:.2f}\n")
        f.write(f"F-Statistik: {sm_model.fvalue:.3f}, p-Wert: {sm_model.f_pvalue:.4f}\n\n")
        f.write("Koeffizienten:\n")
        for var, coef, pval, tval in zip(sm_model.params.index, sm_model.params.values,
                                         sm_model.pvalues.values, sm_model.tvalues.values):
            f.write(f"{var}: {coef:.4f}, p={pval:.4f}, t={tval:.2f}\n")

        # Regression after backward elimination
        f.write("\n=== Regression after Backward Elimination (p <= 0.05) ===\n")
        f.write(f"MAE: {sm_mae_sig:.2f}\nR²: {sm_r2_sig:.2f}\nMSE: {sm_mse_sig:.2f}\n")
        f.write(f"F-Statistik: {sm_model_sig.fvalue:.3f}, p-Wert: {sm_model_sig.f_pvalue:.4f}\n\n")
        f.write("Koeffizienten:\n")
        for var, coef, pval, tval in zip(sm_model_sig.params.index, sm_model_sig.params.values,
                                         sm_model_sig.pvalues.values, sm_model_sig.tvalues.values):
            f.write(f"{var}: {coef:.4f}, p={pval:.4f}, t={tval:.2f}\n")

    print(f"Metrics and coefficients saved to {txt_file}")

    # ------------------------
    # 4. Plot predicted vs actual for final model
    # ------------------------
    plt.figure(figsize=(8,6))
    plt.scatter(y_reg, sm_y_pred_sig, color='blue')
    plt.plot([y_reg.min(), y_reg.max()], [y_reg.min(), y_reg.max()], 'r--', linewidth=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Predicted vs Actual {goal_rating} ({season_type})")
    plt.tight_layout()
    plot_file = os.path.join(output_folder, f"predicted_vs_actual_{goal_rating}_{season_type}.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Predicted vs actual plot saved to {plot_file}")

    # ------------------------
    # 5. Residual plot
    # ------------------------
    residuals = y_reg - sm_y_pred_sig
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=15)
    plt.xlabel("Residual (Actual – Predicted)")
    plt.title("Distribution of Residuals")
    plt.grid(True)
    residual_file = os.path.join(output_folder, f"residuals_distribution_{goal_rating}_{season_type}.png")
    plt.savefig(residual_file)
    plt.close()
    print(f"Residual plot saved to {residual_file}")


def running_correlation(df, season_type):
    ## correlation

    output_folder = "percentage/correlation_outputs"
    os.makedirs(output_folder, exist_ok=True)
    

    exclude_cols = ["TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME"]  # example columns

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=["number"])

    # Remove specific columns
    numeric_df = numeric_df.drop(columns=[col for col in exclude_cols if col in numeric_df.columns])

    # Compute correlation matrix
    corr_matrix = numeric_df.corr()

    # ------------------------
    # 4. Save correlation heatmap as PNG
    # ------------------------
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    corr_png_file = os.path.join(output_folder, f"correlation_matrix_{season_type}.png")
    plt.savefig(corr_png_file)
    plt.close()
    print(f"Correlation heatmap saved to {corr_png_file}")


# ==============================================
# Einstellungen
# ==============================================
csv_file = "nba_player_stats_2024_25.csv"
cluster_size = 3
output_dir = "percentage/extended"
os.makedirs(output_dir, exist_ok=True)

report_path = os.path.join(output_dir, "pct_analysis_report.txt")
with open(report_path, "w") as report:
    report.write("=== NBA 2024-25 Advanced Percentages Cluster & Regression Analysis ===\n\n")

# ==============================================
# 1. CSV laden
# ==============================================
if os.path.exists(csv_file):
    print(f"Lade Daten aus CSV: {csv_file}")
    filtered_players = pd.read_csv(csv_file)
else:
    raise FileNotFoundError(f"CSV {csv_file} nicht gefunden. Bitte zuerst die Daten abrufen.")

# ==============================================
# 2. Features für Clustering
# ==============================================
features = [
    'USG_PCT_Usage', 'PCT_FGM_Usage', 'PCT_FG3M_Usage', 'PCT_FTM_Usage',
    'PCT_OREB_Usage', 'PCT_DREB_Usage', 'PCT_AST_Usage', 'PCT_TOV_Usage',
    'PCT_STL_Usage', 'PCT_BLK_Usage', 'PCT_PF_Usage', 'PCT_PFD_Usage',
]
missing = [f for f in features if f not in filtered_players.columns]
if missing:
    raise ValueError(f"Fehlende Spalten im DataFrame: {missing}")

X = filtered_players[features].fillna(0)

feature_map = {
    'USG_PCT_Usage': 1,          # Placeholder, can be set based on usage calculation
    'PCT_FGM_Usage': 0.666284,
    'PCT_FG3M_Usage': 0.403160,
    'PCT_FTM_Usage': 0.211148,
    'PCT_OREB_Usage': 0.101541,
    'PCT_DREB_Usage': 0.454771,
    'PCT_AST_Usage': 0.302946,
    'PCT_TOV_Usage': 0.680298,
    'PCT_STL_Usage': 0.334396,
    'PCT_BLK_Usage': 0.205620,
    'PCT_PF_Usage': 0.205090,
    'PCT_PFD_Usage': 0.123648
}

feature_weights = {f"{k}": v for k, v in feature_map.items()}
weights_array = np.array([feature_weights[f] for f in features])
X_weighted = X * weights_array

# ==============================================
# 3. KMeans Clustering
# ==============================================
kmeans = KMeans(init="k-means++", n_init=50, n_clusters=cluster_size, random_state=42)
filtered_players["cluster"] = kmeans.fit_predict(X_weighted)

# ==============================================
# 4. Cluster-Summary
# ==============================================
cluster_summary = (
    filtered_players.groupby("cluster")
    .median(numeric_only=True)[features]
)

with open(report_path, "a") as report:
    report.write("=== Cluster-Summary (Medianwerte) ===\n")
    report.write(str(cluster_summary))
    report.write("\n\n")

# ==============================================
# 5. Silhouette Score
# ==============================================
sil_score = silhouette_score(X, filtered_players["cluster"])
print(f"Silhouette Coefficient: {sil_score:.4f}")
with open(report_path, "a") as report:
    report.write(f"Silhouette Coefficient: {sil_score:.4f}\n\n")

# ==============================================
# 6. Clustergrößen-Plot
# ==============================================
plt.figure(figsize=(6, 4))
filtered_players["cluster"].value_counts().sort_index().plot(
    kind="bar", title="Cluster-Größen", color="lightgreen", edgecolor="black"
)
plt.xlabel("Cluster")
plt.ylabel("Anzahl Spieler")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cluster_sizes.png"), dpi=300)
plt.close()

# ==============================================
# 7. Minuten pro Team/Cluster
# ==============================================
team_cluster_minutes = (
    filtered_players.groupby(["TEAM_ABBREVIATION", "cluster"])["MIN_AVG"]
    .sum()
    .unstack(fill_value=0)
)

with open(report_path, "a") as report:
    report.write("=== Minuten pro Team und Cluster ===\n")
    for team, clusters in team_cluster_minutes.iterrows():
        report.write(f"\nTeam {team}:\n")
        for cid, minutes in clusters.items():
            report.write(f"  Cluster {cid}: {minutes:.1f} Minuten\n")
    report.write("\n\n")

# ==============================================
# 8. Regression Siege ~ Cluster-Minuten
# ==============================================
standings = leaguestandings.LeagueStandings(season="2024-25", league_id="00").get_data_frames()[0]
nba_teams = teams.get_teams()
team_map = {t["id"]: t["abbreviation"] for t in nba_teams}
standings["TEAM_ABBREVIATION"] = standings["TeamID"].map(team_map)
team_wins = standings[["TEAM_ABBREVIATION", "WINS"]]



running_regression(team_cluster_minutes, team_wins, "WINS", "Regular Season")

running_correlation(team_cluster_minutes, "Cluster")
# ==============================================
# 9. Dataset-Info
# ==============================================
with open(report_path, "a") as report:
    report.write("=== Dataset Info ===\n")
    report.write(f"Total players: {len(filtered_players)}\n")
    report.write(f"Teams: {filtered_players['TEAM_ABBREVIATION'].nunique()}\n")
    report.write(f"Features for clustering: {len(features)}\n")
    report.write("\nColumns:\n")
    report.write(", ".join(filtered_players.columns))
    report.write("\n\n")

# ==============================================
# 10. Scatterplots
# ==============================================
plot_features = [
    ("USG_PCT_Usage", "PCT_PTS_Usage"),
    ("PCT_AST_Usage", "PCT_TOV_Usage"),
    ("PCT_OREB_Usage", "PCT_DREB_Usage"),
    ("PCT_STL_Usage", "PCT_BLK_Usage"),
]
for x_feat, y_feat in plot_features:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=filtered_players,
        x=x_feat,
        y=y_feat,
        hue="cluster",
        palette="tab10",
        s=80,
        alpha=0.7,
    )
    plt.title(f"Clustervergleich: {x_feat} vs {y_feat}")
    plt.xlabel(x_feat)
    plt.ylabel(y_feat)
    plt.legend(title="Cluster")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"cluster_plot_{x_feat}_{y_feat}.png"), dpi=300)
    plt.close()

print(f"\nAnalyse abgeschlossen. Ergebnisse gespeichert in '{report_path}' und Plots in '{output_dir}'.")
