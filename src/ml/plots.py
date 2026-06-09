import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_and_save_plots(features_path: str, analytics_path: str, output_img_dir: str):
    os.makedirs(output_img_dir, exist_ok=True)
    df = pd.read_csv(features_path)
    song_metrics = pd.read_csv(analytics_path)
     # Sort structures to guarantee sequential plotting continuity
    top_songs = song_metrics.sort_values('performance_score', ascending=False)


    # 1. Longevity Barplot
    plt.figure(figsize=(14, 6))
    top10 = song_metrics.sort_values('days_on_chart', ascending=False).head(10)
    sns.barplot(data=top10, x='days_on_chart', y='song')
    plt.title("Top Songs by Longevity")
    plt.savefig(os.path.join(output_img_dir, "longevity_top10.png"))
    plt.close()


    #1.longetivity for all 50 songs 
    plt.figure(figsize=(14, 6))
    top50 = song_metrics.sort_values('days_on_chart', ascending=False).head(50)
    sns.barplot(data=top50, x='days_on_chart', y='song')
    plt.title("Top Songs by Longevity")
    plt.savefig(os.path.join(output_img_dir, "longevity_top50.png"))
    plt.close()

    # 2. Average Rank Barplot
    plt.figure(figsize=(12, 6))
    top_avg = song_metrics.sort_values('avg_rank').head(20)
    sns.barplot(data=top_avg, x='avg_rank', y='song')
    plt.title("Best Average Ranked Songs")
    plt.savefig(os.path.join(output_img_dir, "average_rank_top20.png"))
    plt.close()
    


    plt.figure(figsize=(12, 6))
    top_avg_50 = song_metrics.sort_values('avg_rank').head(50)
    sns.barplot(data=top_avg_50, x='avg_rank', y='song')
    plt.title("Best Average Ranked Songs")
    plt.savefig(os.path.join(output_img_dir, "average_rank_top50.png"))
    plt.close()
    
    print(f"Plots saved successfully inside {output_img_dir}/")



    song_name = "Last Night"

    temp = df[df['song'] == song_name].sort_values('date')

    plt.figure(figsize=(10,5))
    plt.plot(temp['date'], temp['position'], marker='o')

    plt.title(f"Rank Trend of {song_name}")
    plt.ylabel("Position (Lower is Better)")
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.gca().invert_yaxis()  # optional: rank 1 at top
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "lastnightsong_ranktrend.png"))

    #last night ranking decreases as i go from 5-2024 to 11-2025



    #3.volatility
    top_vol = song_metrics.sort_values(
    'rank_volatility',
    ascending=False
).head(20)

    plt.figure(figsize=(12,6))

    sns.barplot(
    data=top_vol,
    x='rank_volatility',
    y='song'
)

    plt.title("Most Volatile Songs")
    plt.savefig(os.path.join(output_img_dir, "songvsvolatility.png"))
    top_vol = song_metrics.sort_values(
    'rank_volatility',
    ascending=False
).head(50)

    plt.figure(figsize=(12,6))

    sns.barplot(
    data=top_vol,
    x='rank_volatility',
    y='song'
)

    plt.title("Most Volatile Songs")
    plt.savefig(os.path.join(output_img_dir, "allsongvolatile_chart.png"))
    # =========================================================================
    # 1. LONGEVITY CHARTS (Top 10 & Top 50)
    # =========================================================================
    # Top 10 Longevity
    plt.figure(figsize=(14, 6))
    top10_long = song_metrics.sort_values('days_on_chart', ascending=False).head(10)
    sns.barplot(data=top10_long, x='days_on_chart', y='song')
    plt.title("Top 10 Songs by Longevity")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "longevity_top10.png"))
    plt.close()

    # Top 50 Longevity
    plt.figure(figsize=(14, 12))
    top50_long = song_metrics.sort_values('days_on_chart', ascending=False).head(50)
    sns.barplot(data=top50_long, x='days_on_chart', y='song')
    plt.title("Top 50 Songs by Longevity")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "longevity_top50.png"))
    plt.close()

    # =========================================================================
    # 2. AVERAGE RANKING PERFORMANCE (Top 20 & Top 50)
    # =========================================================================
    # Top 20 Avg Rank
    plt.figure(figsize=(12, 6))
    top20_avg = song_metrics.sort_values('avg_rank').head(20)
    sns.barplot(data=top20_avg, x='avg_rank', y='song')
    plt.title("Best Average Ranked Songs (Top 20)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "average_rank_top20.png"))
    plt.close()

    # Top 50 Avg Rank
    plt.figure(figsize=(12, 12))
    top50_avg = song_metrics.sort_values('avg_rank').head(50)
    sns.barplot(data=top50_avg, x='avg_rank', y='song')
    plt.title("Best Average Ranked Songs (Top 50)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "average_rank_top50.png"))
    plt.close()

    # =========================================================================
    # 3. 7-DAY ROLLING POPULARITY TREND LINES (Top 2 & Top 50)
    # =========================================================================
    # Top 2 Popularity Lines
    plt.figure(figsize=(15, 7))
    top2_tracks = top_songs.head(2)['song']
    for s in top2_tracks:
        temp = df[df['song'] == s].sort_values('date')
        plt.plot(temp['date'], temp['rolling_7d_popularity'], label=s)
    plt.legend()
    plt.title("7-Day Popularity Trend (Top 2 Tracks)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "popularity_trend_top2.png"))
    plt.close()

    # Top 50 Popularity Lines
    plt.figure(figsize=(15, 8))
    top50_tracks = top_songs.head(50)['song']
    for s in top50_tracks:
        temp = df[df['song'] == s].sort_values('date')
        plt.plot(temp['date'], temp['rolling_7d_popularity'], alpha=0.5)
    plt.title("7-Day Popularity Trend (Top 50 Tracks)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "popularity_trend_top50.png"))
    plt.close()

    # =========================================================================
    # 4. VELOCITY LOGIC: CLIMBING VELOCITY SPEED (Top 10 & Top 50)
    # =========================================================================
    # Top 10 Fastest Climbing
    plt.figure(figsize=(10, 6))
    df.groupby('song')['rank_velocity'].mean().sort_values().head(10).plot(kind='barh')
    plt.title("Fastest Climbing Songs (Top 10)")
    plt.gca().invert_yaxis()  # Keeps fastest climb at the top
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "velocity_climbing_top10.png"))
    plt.close()

    # Top 50 Fastest Climbing
    plt.figure(figsize=(10, 12))
    df.groupby('song')['rank_velocity'].mean().sort_values().head(50).plot(kind='barh')
    plt.title("Fastest Climbing Songs (Top 50)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "velocity_climbing_top50.png"))
    plt.close()

    # =========================================================================
    # 5. MODEL DECAY & FATIGUE DEGRADATION TRAJECTORY
    # =========================================================================
    if not top_songs.empty:
        plt.figure(figsize=(12, 6))
        peak_track = top_songs.iloc[0]['song']
        temp = df[df['song'] == peak_track].sort_values('days_since_peak')
        plt.plot(temp['days_since_peak'], temp['position'])
        plt.gca().invert_yaxis()  # Rank 1 at topmost vertical bound
        plt.title(f"Decay Curve Lifecycle Fatigue - {peak_track}")
        plt.xlabel("Days Since Peak Position Achieved")
        plt.ylabel("Chart Position")
        plt.tight_layout()
        plt.savefig(os.path.join(output_img_dir, "fatigue_decay_curve.png"))
        plt.close()

    # =========================================================================
    # 6. ARTIST MOMENTUM DISTRIBUTIONS (Top 10 & Top 50)
    # =========================================================================
    # Calculate artist aggregation on the fly based on feature dataset
    artist_momentum = df.groupby(['artist', 'date'])['popularity'].mean().reset_index()
    artist_momentum['artist_momentum_30d'] = artist_momentum.groupby('artist')['popularity'].transform(
        lambda x: x.rolling(30, min_periods=1).mean()
    )
    
    # Top 10 Artist Momentum
    plt.figure(figsize=(12, 6))
    artist_momentum.groupby('artist')['artist_momentum_30d'].mean().sort_values(ascending=False).head(10).plot(kind='bar')
    plt.title("Top Artist Momentum (Top 10)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "artist_momentum_top10.png"))
    plt.close()

    # Top 50 Artist Momentum
    plt.figure(figsize=(14, 8))
    artist_momentum.groupby('artist')['artist_momentum_30d'].mean().sort_values(ascending=False).head(50).plot(kind='bar')
    plt.title("Top Artist Momentum (Top 50)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "artist_momentum_top50.png"))
    plt.close()

    # =========================================================================
    # 7. OVERALL POPULARITY TREND
    # =========================================================================
    plt.figure(figsize=(14, 5))
    overall_trend = df.groupby('date')['popularity'].mean()
    plt.plot(overall_trend.index, overall_trend.values, color='blue', linewidth=2)
    plt.gca().invert_yaxis()
    plt.title("Overall Popularity Trend (All Songs Combined)")
    plt.ylabel("Rolling Average Rank")
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "overall_popularity_trend.png"))
    plt.close()

    # =========================================================================
    # 8. SPECIFIC TRACK TRACKING: ROLLING TREND PROFILE ("Last Night")
    # =========================================================================
    target_song = "Last Night"
    if target_song in df['song'].values:
        plt.figure(figsize=(14, 5))
        temp = df[df['song'] == target_song].sort_values('date')
        plt.plot(temp['date'], temp['popularity'], marker='o', linewidth=2, color='green')
        plt.gca().invert_yaxis()
        plt.title(f"Rolling Trend Score Analysis: {target_song}")
        plt.ylabel("7-Day Rolling Avg Rank")
        plt.xlabel("Date")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_img_dir, "track_profile_last_night.png"))
        plt.close()

    # =========================================================================
    # 9. ADVANCED METRIC MULTIVARIATE SCATTERPLOT
    # =========================================================================
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=song_metrics,
        x='avg_rank',
        y='rank_volatility',
        hue='days_on_chart',
        size='days_on_chart',
        palette='viridis',
        sizes=(20, 300),
        alpha=0.7
    )
    plt.title("Song Stability vs System Performance Matrix")
    plt.xlabel("Average Chart Rank (Lower = Better)")
    plt.ylabel("Volatility (Higher Variance = More Unstable)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "stability_vs_performance.png"))
    plt.close()

    print(f"All target visual charts generated inside execution path: {output_img_dir}/")



# above analysis reveal 
# graccias por nada is  most stable 
#fluctuation range limit  min 17 max is 30 times 





if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
    
    generate_and_save_plots(
        features_path=os.path.join(ROOT_DIR, "data/processed/engineered_features.csv"),
        analytics_path=os.path.join(ROOT_DIR, "data/processed/top_song_analytics.csv"),
        output_img_dir=os.path.join(ROOT_DIR, "data/processed/plots")
    )
   