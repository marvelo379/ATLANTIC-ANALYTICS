import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_performance_plots(profiles_path: str, output_img_dir: str):
    """Generates all targeted figures for song performance, popularity metrics, and longevity matrices."""
    os.makedirs(output_img_dir, exist_ok=True)
    
    # Ingest analytical metrics
    stats_df = pd.read_csv(profiles_path)

    # 1. Songs with Longest Playlist Presence (Top 20 Barplot)
    plt.figure(figsize=(12, 6))
    top20_longest = stats_df.sort_values(by="days_on_chart", ascending=False).head(20)
    sns.barplot(data=top20_longest, y="song", x="days_on_chart", palette="viridis")
    plt.title("Top 20 Songs by Playlist Longevity Presence")
    plt.xlabel("Days on Chart")
    plt.ylabel("Song Name")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "song_longevity_top20.png"))
    plt.close()

    # 2. Playlist Presence Extension (Top 50 Barplot)
    plt.figure(figsize=(12, 12))
    top50_longest = stats_df.sort_values(by="days_on_chart", ascending=False).head(50)
    sns.barplot(data=top50_longest, y="song", x="days_on_chart", palette="viridis")
    plt.title("Top 50 Songs by Playlist Longevity Presence")
    plt.xlabel("Days on Chart")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "song_longevity_top50.png"))
    plt.close()

    # 3. Song Survival Distribution Countplot
    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=stats_df, 
        x="survival_group", 
        order=["Short Stay", "Medium Survivor", "Long Survivor"],
        palette="Set2"
    )
    plt.title("Song Survival Lifespan Distribution")
    plt.xlabel("Survival Cohort Group")
    plt.ylabel("Unique Track Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "song_survival_groups.png"))
    plt.close()

    # 4. Top 20 Songs by Average Popularity
    plt.figure(figsize=(12, 6))
    top20_pop = stats_df.sort_values(by="avg_popularity", ascending=False).head(20)
    sns.barplot(data=top20_pop, x="avg_popularity", y="song", palette="magma")
    plt.title("Top 20 Songs by Average Popularity Index")
    plt.xlabel("Average Popularity Score")
    plt.ylabel("Song Name")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "song_popularity_top20.png"))
    plt.close()

    # 5. Peak Strength vs Longevity Strength Strategic Matrix
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=stats_df,
        x='peak_z',
        y='longevity_z',
        hue='performance_segment',
        palette='Set2',
        s=120,
        alpha=0.8
    )
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.axvline(0, color='black', linewidth=1, linestyle='--')
    plt.title("Strategic Market Matrix: Peak Strength vs Longevity")
    plt.xlabel("Peak Chart Position Strength (Z-score)")
    plt.ylabel("Chart Longevity Track Strength (Z-score)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Market Quadrant", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "strategic_peak_vs_longevity_matrix.png"))
    plt.close()

    print(f"Success! Performance figures compiled inside folder paths: {output_img_dir}/")


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
    
    PROFILES_INPUT = os.path.join(ROOT_DIR, "data/processed/song_analytics_profiles.csv")
    OUTPUT_FOLDER = os.path.join(ROOT_DIR, "data/processed/plotsong")

    generate_performance_plots(profiles_path=PROFILES_INPUT, output_img_dir=OUTPUT_FOLDER)
