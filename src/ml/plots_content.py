import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_content_plots(profiles_path: str, output_img_dir: str):
    """Generates all targeted visual figures mapping track metadata factors to chart metrics."""
    os.makedirs(output_img_dir, exist_ok=True)
    
    # Ingest baseline stats profiles
    song_df = pd.read_csv(profiles_path)

    # 1. Explicit vs Non-Explicit: Hit Score Distribution (Boxplot)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=song_df, x='is_explicit', y='hit_score', palette='Set2', ax=ax)
    ax.set_title("Explicit vs Non-Explicit: Hit Score Footprint")
    ax.set_xlabel("Is Explicit Content Status (0 = No, 1 = Yes)")
    ax.set_ylabel("Hit Score Index Units")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "content_explicit_hit_score.png"))
    plt.close(fig)

    # 2. Explicit vs Non-Explicit: Popularity Distribution (Boxplot)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=song_df, x='is_explicit', y='avg_popularity', palette='Set3', ax=ax)
    ax.set_title("Explicit vs Non-Explicit: Popularity Distribution Profile")
    ax.set_xlabel("Is Explicit Content Status")
    ax.set_ylabel("Average Popularity Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "content_explicit_popularity.png"))
    plt.close(fig)

    # 3. Single vs Album Track Performance (Boxplot)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=song_df, x='album_type', y='hit_score', palette='Set2', ax=ax)
    ax.set_title("Release Format Footprint: Single vs Album Track Performance")
    ax.set_xlabel("Album Product Type Category")
    ax.set_ylabel("Calculated Hit Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "content_album_type_hit_score.png"))
    plt.close(fig)

    # 4. Popularity Distribution by Album Type (Violin Plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=song_df, x='album_type', y='avg_popularity', palette='Set3', ax=ax)
    ax.set_title("Popularity Density Variations by Product Format Type")
    ax.set_xlabel("Album Product Type Category")
    ax.set_ylabel("Average Popularity Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "content_album_type_popularity_violin.png"))
    plt.close(fig)

    # 5. Longevity Performance: Singles vs Album Tracks (Boxplot)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=song_df, x='album_type', y='days_on_chart', palette='coolwarm', ax=ax)
    ax.set_title("Retention Lifespan Profile: Singles vs Album Tracks")
    ax.set_xlabel("Album Product Type Category")
    ax.set_ylabel("Total Lifespan Days on Chart")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "content_album_type_longevity.png"))
    plt.close(fig)

    # 6. Song Run-Time Duration vs Popularity (Scatter with Regression Line)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=song_df, x='duration_min', y='avg_popularity', alpha=0.6, ax=ax)
    sns.regplot(data=song_df, x='duration_min', y='avg_popularity', scatter=False, color='red', ax=ax)
    ax.set_title("Continuous Trend Map: Track Duration vs Popularity Baseline")
    ax.set_xlabel("Track Duration (Minutes)")
    ax.set_ylabel("Average Popularity Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "content_duration_vs_popularity_trend.png"))
    plt.close(fig)

    # 7. Song Run-Time Duration vs Best Chart Position (Scatter with Regression Line)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=song_df, x='duration_min', y='best_rank', alpha=0.6, ax=ax)
    ax.invert_yaxis()  # Rank 1 at top boundary limit
    sns.regplot(data=song_df, x='duration_min', y='best_rank', scatter=False, color='red', ax=ax)
    ax.set_title("Continuous Trend Map: Track Duration vs Peak Chart Position")
    ax.set_xlabel("Track Duration (Minutes)")
    ax.set_ylabel("Best Chart Position Achieved")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "content_duration_vs_best_rank_trend.png"))
    plt.close(fig)

    # 8. Popularity Across Structured Song Duration Groups (Boxplot)
    fig, ax = plt.subplots(figsize=(10, 6))
    group_order = ['<2 min', '2-3 min', '3-4 min', '4-5 min', '5+ min']
    sns.boxplot(data=song_df, x='duration_group', y='avg_popularity', order=group_order, palette='Set2', ax=ax)
    ax.set_title("Popularity Variations Across Segmented Duration Cohort Groups")
    ax.set_xlabel("Duration Bracket Group")
    ax.set_ylabel("Average Popularity Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "content_duration_brackets_popularity.png"))
    plt.close(fig)

    # 9. Best Chart Position Across Duration Brackets (Boxplot)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=song_df, x='duration_group', y='best_rank', order=group_order, palette='coolwarm', ax=ax)
    ax.invert_yaxis()
    ax.set_title("Peak Position Distributions Across Segmented Duration Cohort Groups")
    ax.set_xlabel("Duration Bracket Group")
    ax.set_ylabel("Best Chart Position Achieved")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "content_duration_brackets_best_rank.png"))
    plt.close(fig)

    # 10. Album Volume Size vs Single Track Popularity (Scatter with Regression Line)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=song_df, x='total_tracks', y='avg_popularity', alpha=0.6, ax=ax)
    sns.regplot(data=song_df, x='total_tracks', y='avg_popularity', scatter=False, color='red', ax=ax)
    ax.set_title("Continuous Trend Map: Product Track Volume Size vs Popularity")
    ax.set_xlabel("Total Tracks Contained within Album Structure")
    ax.set_ylabel("Average Popularity Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "content_album_size_vs_popularity_trend.png"))
    plt.close(fig)

    # 11. Average Track Success Distributed per Individual Album Size (Lineplot)
    fig, ax = plt.subplots(figsize=(10, 6))
    grouped_tracks = song_df.groupby('total_tracks')['avg_popularity'].mean().reset_index()
    sns.lineplot(data=grouped_tracks, x='total_tracks', y='avg_popularity', marker='o', color='purple', ax=ax)
    ax.set_title("Catalog Footprint: Mean Track Success Volumes by Album Size")
    ax.set_xlabel("Total Tracks Contained within Album Structure")
    ax.set_ylabel("Mean Track Popularity Score")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "content_album_size_mean_line.png"))
    plt.close(fig)

    # 12. Volume Inventory Distribution of Ingested Album Product Formats (Histogram)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=song_df, x='total_tracks', bins=25, color='teal', edgecolor='black', alpha=0.7, ax=ax)
    ax.set_title("Inventory Distribution Density: Ingested Album Track Size Footprints")
    ax.set_xlabel("Total Tracks Contained within Album Structure")
    ax.set_ylabel("Unique Song Counts")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "content_album_size_distribution_hist.png"))
    plt.close(fig)

    # =========================================================================
    # --- NEW PLOTS INTEGRATING ADVANCED METRICS ---
    # =========================================================================

    # 13. Monetization Engine: Revenue Efficiency by Predicted Audience Behavior (Boxplot)
    fig, ax = plt.subplots(figsize=(12, 6))
    # Sort category order by median revenue to ensure sorted layout display
    behavior_order = song_df.groupby('predicted_audience_behavior')['estimated_revenue'].median().sort_values(ascending=False).index
    sns.boxplot(data=song_df, x='predicted_audience_behavior', y='estimated_revenue', order=behavior_order, palette='viridis', ax=ax)
    ax.set_title("Monetization Engine: Revenue Efficiency by Predicted Audience Behavior Cohort")
    ax.set_xlabel("Predicted Audience Behavior Cohort")
    ax.set_ylabel("Estimated Revenue ($)")
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "content_monetization_by_behavior.png"))
    plt.close(fig)

    # 14. Content Packaging Strategy: Market Volume Share Count (Barplot, sorted)
    fig, ax = plt.subplots(figsize=(10, 6))
    strategy_order = song_df['release_packaging_strategy'].value_counts().index
    sns.countplot(data=song_df, x='release_packaging_strategy', order=strategy_order, palette='magma', ax=ax)
    ax.set_title("Content Packaging Strategy: Release Structure Distribution")
    ax.set_xlabel("Release Packaging Strategy")
    ax.set_ylabel("Track Volume Counts")
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "content_packaging_strategy_distribution.png"))
    plt.close(fig)

    # 15. Forecast Drift: Performance Momentum by Trend Arrow Status (Boxplot)
    fig, ax = plt.subplots(figsize=(10, 6))
    trend_order = song_df.groupby('trend_arrow')['performance_score'].median().sort_values(ascending=False).index
    sns.boxplot(data=song_df, x='trend_arrow', y='performance_score', order=trend_order, palette='coolwarm', ax=ax)
    ax.set_title("Forecast Drift: Track Performance Score by Trend Direction Indicator")
    ax.set_xlabel("Trend Arrow Indicator Status")
    ax.set_ylabel("Performance Score Index")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "content_trend_momentum_performance.png"))
    plt.close(fig)

    print(f"Success! Content attribute charts generated inside folder paths: {output_img_dir}/")

    # 1. Total Financial Revenue Breakdown by Product Format (Sorted Bar Chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    rev_by_type = song_df.groupby('album_type')['estimated_revenue'].sum().sort_values(ascending=False).reset_index()
    
    sns.barplot(data=rev_by_type, x='album_type', y='estimated_revenue', palette='Blues_r', ax=ax)
    ax.set_title("Financial Performance: Total Revenue Breakdown by Album Product Format")
    ax.set_xlabel("Album Product Type Category")
    ax.set_ylabel("Total Estimated Revenue ($)")
    
    # Format currency with comma separations for enhanced professional presentation
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${int(x):,}"))
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "finance_revenue_by_album_type.png"))
    plt.close(fig)

    


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
    
    PROFILES_INPUT = os.path.join(ROOT_DIR, "data/processed/song_content_profiles.csv")
    OUTPUT_FOLDER = os.path.join(ROOT_DIR, "data/processed/plotcontent")

    generate_content_plots(profiles_path=PROFILES_INPUT, output_img_dir=OUTPUT_FOLDER)