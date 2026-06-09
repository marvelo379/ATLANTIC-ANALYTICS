import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_artist_plots(profiles_path: str, output_img_dir: str):
    """Compiles market charts covering total chart footprints, productivity, and dominance distribution patterns."""
    os.makedirs(output_img_dir, exist_ok=True)
    
    # Ingest baseline stats profile
    stats_df = pd.read_csv(profiles_path)

    # 1. Top 10 Artists by Total Days on Playlist
    plt.figure(figsize=(12, 6))
    top10_days = stats_df.sort_values(by='total_days_on_playlist', ascending=False).head(10)
    sns.barplot(data=top10_days, x='total_days_on_playlist', y='artist_preprocessed', palette='viridis')
    plt.title("Top 10 Artists by Total Combined Chart Real Estate")
    plt.xlabel("Total Days across All Active Tracks")
    plt.ylabel("Artist Name")
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "artist_presence_top10.png"))
    plt.close()

    # 2. Artist Catalog Volume Output vs Average Staying Power
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=stats_df,
        x='unique_songs',
        y='avg_song_longevity',
        size='total_days_on_playlist',
        hue='dominance_score',
        palette='magma',
        sizes=(40, 400),
        alpha=0.8
    )
    plt.title("Artist Productivity vs Track Retention Lifecycle")
    plt.xlabel("Total Unique Hits Introduced to Chart")
    plt.ylabel("Average Lifespan Days per Track")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "artist_productivity_vs_retention.png"))
    plt.close()

    # 3. Top 20 Artists Ranked by Consolidated Dominance Index Score
    plt.figure(figsize=(12, 8))
    top20_dominant = stats_df.sort_values(by='dominance_score', ascending=False).head(20)
    sns.barplot(data=top20_dominant, x='dominance_score', y='artist_preprocessed', palette='rocket')
    plt.title("Top 20 Industry Dominant Artists (Consolidated Index)")
    plt.xlabel("Calculated Dominance Score Units")
    plt.ylabel("Artist Name")
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "artist_dominance_top20.png"))
    plt.close()

    print(f"Success! Career evaluation charts generated inside: {output_img_dir}/")


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
    
    PROFILES_INPUT = os.path.join(ROOT_DIR, "data/processed/artist_analytics_profiles.csv")
    OUTPUT_FOLDER = os.path.join(ROOT_DIR, "data/processed/plotartist")

    generate_artist_plots(profiles_path=PROFILES_INPUT, output_img_dir=OUTPUT_FOLDER)
