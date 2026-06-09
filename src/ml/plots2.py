import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_advanced_plots(data_path: str, output_img_dir: str):
    """Generates advanced behavioral charts, stability segments, and long-term cyclic time visualizations."""
    os.makedirs(output_img_dir, exist_ok=True)
    
    # 1. Ingest Data Matrix
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract structural analytics helpers on the fly to match original notebook variables
    movement_profile = df.groupby('song')['rank_change'].std().reset_index()
    movement_profile.columns = ['song', 'movement_volatility']

    entry_exit = df.groupby('song').agg(
        entry_date=('date', 'first'),
        exit_date=('date', 'last'),
        entry_rank=('position', 'first'),
        exit_rank=('position', 'last'),
        entry_popularity=('popularity', 'first'),
        exit_popularity=('popularity', 'last')
    ).reset_index()
    entry_exit['rank_change'] = entry_exit['exit_rank'] - entry_exit['entry_rank']

    def _classify_boundary(row):
        if row['entry_rank'] <= 10 and row['exit_rank'] > 50:
            return "Viral Drop (fast rise, fast fall)"
        elif row['entry_rank'] > 50 and row['exit_rank'] <= 10:
            return "Slow Burner Hit"
        elif row['rank_change'] > 20:
            return "Strong Decline"
        elif row['rank_change'] < -10:
            return "Gaining Momentum"
        else:
            return "Stable Run"
    entry_exit['behavior'] = entry_exit.apply(_classify_boundary, axis=1)

    # =========================================================================
    # YOUR 12 ORIGINAL NOTEBOOK CHARTS
    # =========================================================================
    
    # Chart 1: Song Lifecycle Distribution
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, y='lifecycle', order=df['lifecycle'].value_counts().index)
    plt.title("Song Lifecycle Distribution")
    plt.xlabel("Number of Songs")
    plt.ylabel("Lifecycle Type")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "01_lifecycle_distribution.png"))
    plt.close()

    # Chart 2: Rank Stability vs Average Performance Scatter
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='rank_volatility', y='avg_rank', hue='lifecycle', alpha=0.7)
    plt.title("Rank Stability vs Average Performance")
    plt.xlabel("Rank Volatility (Stability ↓ better)")
    plt.ylabel("Average Rank (Performance ↑ better)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "02_stability_vs_performance.png"))
    plt.close()

    # Chart 3: Top Songs by Average Rank
    plt.figure(figsize=(12, 10))
    song_metrics = df.groupby('song', as_index=False)['avg_rank'].mean()
    top_songs = song_metrics.sort_values('avg_rank').head(50)
    sns.barplot(data=top_songs, x='avg_rank', y='song', palette='viridis')
    plt.title("Top Songs by Average Rank")
    plt.xlabel("Average Rank (lower is better)")
    plt.ylabel("Song")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "03_top_songs_avg_rank.png"))
    plt.close()

    # Chart 4: Volatility vs Decay Rate
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='rank_volatility', y='decay_rate', hue='lifecycle')
    plt.title("Volatility vs Decay Rate (Song Stability Risk)")
    plt.xlabel("Rank Volatility")
    plt.ylabel("Decay Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "04_volatility_vs_decay.png"))
    plt.close()

    # Chart 5: Daily Rank Movement Count
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='movement', order=df['movement'].value_counts().index, palette='Set2')
    plt.title("Daily Rank Movement Distribution")
    plt.xlabel("Movement Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "05_daily_rank_movement.png"))
    plt.close()

    # Chart 6: Distribution Histogram of Rank Changes
    plt.figure(figsize=(10, 5))
    sns.histplot(df['rank_change'].dropna(), bins=50, kde=True, color='purple')
    plt.title("Distribution of Rank Changes")
    plt.xlabel("Rank Change (negative = up, positive = down)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "06_rank_change_distribution.png"))
    plt.close()

    # Chart 7: Target Individual Track Historical Profile Line
    if not df.empty:
        song_name = df['song'].iloc[0]
        temp = df[df['song'] == song_name].sort_values('date')
        plt.figure(figsize=(14, 6))
        plt.plot(temp['date'], temp['position'], marker='o')
        plt.gca().invert_yaxis()
        plt.title(f"Rank Movement Over Time: {song_name}")
        plt.xlabel("Date")
        plt.ylabel("Position (lower is better)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_img_dir, "07_track_historical_timeline.png"))
        plt.close()

    # Chart 8: Top 20 Most Volatile Songs
    plt.figure(figsize=(10, 6))
    top_moving = movement_profile.sort_values('movement_volatility', ascending=False).head(20)
    sns.barplot(data=top_moving, x='movement_volatility', y='song', palette='viridis')
    plt.title("Top 20 Most Volatile Songs (Rank Movement)")
    plt.xlabel("Volatility (Higher = more unstable)")
    plt.ylabel("Song")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "08_top20_movement_volatility.png"))
    plt.close()

    # Chart 9: Entry vs Exit Rank Behavior Boundary Scatter
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=entry_exit, x='entry_rank', y='exit_rank', hue='behavior', palette='Set2')
    max_bound = max(entry_exit['entry_rank'].max(), entry_exit['exit_rank'].max()) if not entry_exit.empty else 50
    plt.plot([0, max_bound], [0, max_bound], linestyle='--', color='black')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.title("Entry vs Exit Rank Behavior")
    plt.xlabel("Entry Rank (lower = better)")
    plt.ylabel("Exit Rank (lower = better)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "09_entry_vs_exit_scatter.png"))
    plt.close()

    # Chart 10: Entry vs Exit Behavior Distribution Count
    plt.figure(figsize=(8, 5))
    sns.countplot(data=entry_exit, y='behavior', order=entry_exit['behavior'].value_counts().index, palette='viridis')
    plt.title("Entry vs Exit Behavior Distribution")
    plt.xlabel("Number of Songs")
    plt.ylabel("Behavior Type")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "10_entry_vs_exit_distribution.png"))
    plt.close()

    # Chart 11: Fast Risers vs Slow Decliners Momentum Map
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="rank_velocity", y="rank_acceleration", hue="segment", alpha=0.7)
    plt.title("Fast Risers vs Slow Decliners (Momentum Map)")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "11_momentum_map_anomalies.png"))
    plt.close()

    # Chart 12: Popularity Trend Split by Segmentation Lines
    plt.figure(figsize=(12, 6))
    trend = df.groupby(["date", "segment"])["rolling_7d_popularity"].mean().reset_index()
    sns.lineplot(data=trend, x="date", y="rolling_7d_popularity", hue="segment")
    plt.title("Popularity Trend: Fast Risers vs Slow Decliners")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "12_popularity_trend_segmentation.png"))
    plt.close()


    # =========================================================================
    # 🆕 3 NEW CYCLIC TIME-SERIES AND IMPROVEMENTS FOR 2027/2028 STRATEGY
    # =========================================================================
    
    # New Chart 13: Seasonality Verification (Month vs Popularity Heatmap)
    # Why: Validates whether certain months drive higher track streaming velocities
    plt.figure(figsize=(12, 6))
    df['month_extracted'] = df['date'].dt.month
    monthly_matrix = df.pivot_table(index='lifecycle', columns='month_extracted', values='popularity', aggfunc='mean').fillna(0)
    sns.heatmap(monthly_matrix, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Mean Popularity'})
    plt.title("Seasonal Track Footprint: Lifecycle vs Month Heatmap")
    plt.xlabel("Calendar Month Index")
    plt.ylabel("Track Classification Status")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "13_seasonality_heatmap_month.png"))
    plt.close()

    # New Chart 14: Annual Baseline Trajectory (Year Inflation vs Mean Performance Score)
    # Why: Visualizes baseline trend movement shifts across calendar years
    plt.figure(figsize=(10, 5))
    df['year_extracted'] = df['date'].dt.year
    sns.boxplot(data=df, x='year_extracted', y='performance_score', palette='Set3')
    plt.title("Annual Metric Baseline Distribution Over Time")
    plt.xlabel("Calendar Year")
    plt.ylabel("Performance Score Baseline")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "14_annual_baseline_inflation.png"))
    plt.close()

    # New Chart 15: Cyclic Coordinate Validation Scatterplot
    # Why: Confirms the month_sin/month_cos variables form a clean circle, proving mathematical seasonality continuity
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=df, x='month_sin', y='month_cos', hue='month_extracted', palette='hsv', alpha=0.5)
    plt.title("Validation Check: Trigonometric Cyclic Time Wheel")
    plt.xlabel("Month Sine Coordinate")
    plt.ylabel("Month Cosine Coordinate")
    plt.legend(title="Month Num", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "15_cyclic_trig_wheel.png"))
    plt.close()


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
    
    generate_advanced_plots(
        data_path=os.path.join(ROOT_DIR, "data/processed/final_dataset_merged_features.csv"),
        #analytics_path=os.path.join(ROOT_DIR, "data/processed/top_song_analytics.csv"),
        output_img_dir=os.path.join(ROOT_DIR, "data/processed/plots2")
    )
   