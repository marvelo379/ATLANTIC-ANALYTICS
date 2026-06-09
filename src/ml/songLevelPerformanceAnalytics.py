import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class SongPerformanceAnalyzer:
    """Performs deep historical track longevity analysis, z-score survival profiles, and evergreen indexing."""
    
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.df = None
        self.song_stats = None

    def load_pipeline_data(self):
        """Loads data from the advanced step matrix."""
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file missing: {self.input_path}")
        
        logging.info(f"Ingesting structured feature matrix from {self.input_path}...")
        self.df = pd.read_csv(self.input_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Coerce baseline computational requirements
        self.df["days_on_chart"] = pd.to_numeric(self.df["days_on_chart"], errors="coerce").fillna(1)
        self.df["popularity"] = pd.to_numeric(self.df["popularity"], errors="coerce").fillna(0)
        return self.df

    def process_song_performance_metrics(self):
        """Builds granular track performance scoreboards, peak strengths, and robust longevity scores."""
        logging.info("Aggregating historical tracking rows into song-level benchmarks...")
        
        # 1. Base aggregations
        self.song_stats = self.df.groupby(['song', 'artist']).agg(
            peak_rank=('position', 'min'),
            avg_rank=('position', 'mean'),
            worst_rank=('position', 'max'),
            days_on_chart=('date', 'nunique'),
            avg_popularity=('popularity', 'mean'),
            max_popularity=('popularity', 'max'),
            min_popularity=('popularity', 'min')
        ).reset_index()

        # 2. Peak Strength (Scale: Lower rank number = Higher strength)
        max_rank = float(self.df['position'].max())
        min_rank = 1.0
        self.song_stats['peak_strength'] = (max_rank - self.song_stats['peak_rank']) / (max_rank - min_rank)

        # 3. Inject Volatility Profile safely
        vol_map = self.df.groupby(['song', 'artist'])['rank_volatility'].mean().reset_index()
        self.song_stats = self.song_stats.merge(vol_map, on=['song', 'artist'], how='left')
        self.song_stats['rank_volatility'] = self.song_stats['rank_volatility'].fillna(0.0)

        # 4. Corrected Longevity Score Formula using Min-Max Normalized Volatility
        vol_max = self.song_stats['rank_volatility'].max()
        norm_vol = self.song_stats['rank_volatility'] / vol_max if vol_max > 0 else 0
        self.song_stats['longevity_score'] = np.log1p(self.song_stats['days_on_chart']) * (1.0 - norm_vol)

        # 5. Classify Survival Groups
        def _classify_survival(days):
            if days >= 90: return "Long Survivor"
            return "Medium Survivor" if days >= 30 else "Short Stay"
        self.song_stats["survival_group"] = self.song_stats["days_on_chart"].apply(_classify_survival)

        # 6. Apply Scikit-Learn Z-Score Standardization for Strategic Matrixing
        logging.info("Applying StandardScaler profiles across peak vs longevity components...")
        scaler = StandardScaler()
        self.song_stats[['peak_z', 'longevity_z']] = scaler.fit_transform(
            self.song_stats[['peak_strength', 'longevity_score']]
        )

        # 7. Core Categorical Segmentation
        def _segment_hit(row):
            if row['peak_z'] >= 0 and row['longevity_z'] >= 0: return "Evergreen Hit"
            if row['peak_z'] >= 0 and row['longevity_z'] < 0: return "Viral Spike"
            return "Slow Burner" if row['peak_z'] < 0 and row['longevity_z'] >= 0 else "Low Performer"
        self.song_stats['performance_segment'] = self.song_stats.apply(_segment_hit, axis=1)

    def generate_final_unified_dataset(self, output_dir: str):
        """Merges engineered metrics back into time-series tracks and saves both deliverables."""
        os.makedirs(output_dir, exist_ok=True)

        # Map metrics back to original dataset records
        logging.info("Merging analytics back to main dataset...")
        unified_df = self.df.merge(
            self.song_stats[[
                'song', 'artist', 'peak_strength', 'longevity_score', 
                'peak_z', 'longevity_z', 'performance_segment', 'survival_group'
            ]],
            on=['song', 'artist'],
            how='left'
        )

        # Calculate Unified Hit Score
        pop_max = unified_df['popularity'].max()
        norm_pop = unified_df['popularity'] / pop_max if pop_max > 0 else 0
        unified_df['hit_score'] = (0.4 * unified_df['peak_z']) + (0.6 * unified_df['longevity_z']) + norm_pop

        # Export Files
        stats_path = os.path.join(output_dir, "song_analytics_profiles.csv")
        self.song_stats.to_csv(stats_path, index=False)
        
        unified_path = os.path.join(output_dir, "unified_feng.csv")
        unified_df.to_csv(unified_path, index=False)
        
        logging.info(f"Performance Scoreboard successfully saved to: {stats_path}")
        logging.info(f"Final pipeline dataset successfully saved to: {unified_path}")


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
    
    INPUT_DATA = os.path.join(ROOT_DIR, "data/processed/final_dataset_merged_features.csv")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "data/processed")

    analyzer = SongPerformanceAnalyzer(INPUT_DATA)
    analyzer.load_pipeline_data()
    analyzer.process_song_performance_metrics()
    analyzer.generate_final_unified_dataset(OUTPUT_DIR)
