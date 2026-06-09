import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_popularity_plots(data_path: str, corr_matrix_path: str, output_img_dir: str):
    """Compiles and renders statistical charts mapping out popularity distributions and competitive dynamics."""
    os.makedirs(output_img_dir, exist_ok=True)
    
    # Ingest source processing data structures
    df = pd.read_csv(data_path)
    corr_df = pd.read_csv(corr_matrix_path, index_col=0)

    # 1. Popularity Distribution Across Chart Ranks (Boxplot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='rank_group', y='popularity', palette='Set2')
    plt.title("Popularity Metric Distribution Variation Across Chart Brackets")
    plt.xlabel("Rank Tier Group")
    plt.ylabel("Popularity Index Units")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "popularity_boxplots.png"))
    plt.close()

    # 2. Popularity Density Distribution (Violin Plot)
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='rank_group', y='popularity', palette='Set3')
    plt.title("Popularity Probability Density Across Chart Brackets")
    plt.xlabel("Rank Tier Group")
    plt.ylabel("Popularity Density")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "popularity_violins.png"))
    plt.close()

    # 3. Volume Breakdown: Total Records per Rank Group (Countplot)
    plt.figure(figsize=(8, 5))
    group_order = ['Top 10 (1-10)', 'Top 20 (11-20)', 'Top 50 (21-50)', 'Others (>50)']
    sns.countplot(data=df, x='rank_group', order=group_order, palette='Set2')
    plt.title("Total Recording Entry Volume Count per Rank Bracket")
    plt.xlabel("Rank Tier Group")
    plt.ylabel("Number of Tracking Rows")
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "popularity_volume_counts.png"))
    plt.close()

    # 4. Average Historical Metric Value by Rank Group (Barplot)
    plt.figure(figsize=(8, 5))
    df.groupby('rank_group', observed=False)['avg_popularity'].mean().reindex(group_order).plot(
        kind='bar', color='skyblue', edgecolor='black', alpha=0.8
    )
    plt.title("Mean Historical Long-Term Popularity vs Rank Group")
    plt.xlabel("Rank Tier Group")
    plt.ylabel("Average Popularity Score")
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "popularity_mean_bars.png"))
    plt.close()

    # 5. 🆕 FEATURE IMPROVEMENT: Numerical Correlation Feature Matrix Heatmap
    # Why: Directly tests your notebook discovery that position is a complex function of velocity, competition, and trends.
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1.0, vmax=1.0, square=True)
    plt.title("Market Dynamics Matrix: Correlation Coefficient Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "popularity_dynamics_correlation_heatmap.png"))
    plt.close()

    print(f"Success! Popularity analytics figures compiled inside: {output_img_dir}/")


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
    
    DATA_INPUT = os.path.join(ROOT_DIR, "data/processed/unifiled_psa.csv")
    CORR_INPUT = os.path.join(ROOT_DIR, "data/processed/popularity_correlation_matrix.csv")
    OUTPUT_FOLDER = os.path.join(ROOT_DIR, "data/processed/plotpopularity")

    generate_popularity_plots(data_path=DATA_INPUT, corr_matrix_path=CORR_INPUT, output_img_dir=OUTPUT_FOLDER)
