# import os
import logging
import pandas as pd
import numpy as np
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ContentAttributeAnalyzer:
    """Aggregates and engineers advanced track properties including monetization metrics, 

    behavioral predictions, trend drift, and release strategy categorizations.
    """
    
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.df = None
        self.song_content_df = None

    def load_pipeline_data(self):
        """Loads data from the popularity analytics pipeline step."""
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input master file missing: {self.input_path}")
        
        logging.info(f"Ingesting master dataset from {self.input_path}...")
        self.df = pd.read_csv(self.input_path)
        return self.df

    def process_song_level_content(self):
        """Aggregates data to distinct track rows to calculate precise content weights."""
        logging.info("Building song-level profile matrices for structural metadata indexing...")
        
        # Base aggregation matching your initial pipeline
        self.song_content_df = self.df.groupby('song', as_index=False).agg(
            is_explicit=('is_explicit', 'max'),
            album_type=('album_type', 'first'),
            total_tracks=('total_tracks', 'mean'),
            duration_min=('duration_min', 'mean'),
            best_rank=('position', 'min'),
            avg_rank=('position', 'mean'),
            avg_popularity=('popularity', 'mean'),
            hit_score=('hit_score', 'mean'),
            performance_score=('performance_score', 'mean'),
            days_on_chart=('days_on_chart', 'max'),
            peak_strength=('peak_strength', 'mean'),
            longevity_score=('longevity_score', 'mean'),
            # Aggregating historical stream data if available for trend/monetization features
            total_streams=('streams', 'sum') if 'streams' in self.df.columns else ('popularity', 'sum') 
        )

    def engineer_duration_brackets(self):
        """Partitions continuous track durations into clean categorical brackets."""
        logging.info("Mapping track run-times to discrete duration segment boxes...")
        
        self.song_content_df['duration_group'] = pd.cut(
            self.song_content_df['duration_min'],
            bins=[0, 2, 3, 4, 5, float('inf')],
            labels=['<2 min', '2-3 min', '3-4 min', '4-5 min', '5+ min'],
            include_lowest=True
        )

    def engineer_monetization_and_behavior_features(self):
        """Engineers new features based on business logic, monetization rates, 

        and consumer behavioral alignment outlined in internal pipeline blueprints.
        """
        logging.info("Executing advanced behavioral mapping and monetization modeling...")

        # 1. Monetization Modeling (RPM = Revenue Per 1,000 Streams)
        # Clean: $4.20 RPM, Explicit: $3.20 RPM
        self.song_content_df['rpm_rate'] = np.where(self.song_content_df['is_explicit'] == 1, 3.20, 4.20)
        self.song_content_df['estimated_revenue'] = (self.song_content_df['total_streams'] / 1000) * self.song_content_df['rpm_rate']

        # 2. Consumer Behavioral & Demographic Alignment
        # Explicit + Short = High Gen-Z/Viral affinity. Clean + Standard length = Family/Passive Listening.
        conditions = [
            (self.song_content_df['is_explicit'] == 1) & (self.song_content_df['duration_min'] < 3.0),
            (self.song_content_df['is_explicit'] == 0) & (self.song_content_df['duration_min'].between(2.0, 4.0)),
            (self.song_content_df['duration_min'] >= 5.0)
        ]
        choices = ['Gen-Z Viral Segment', 'Family & Passive Playlist', 'High Skip-Risk / Long Session']
        self.song_content_df['predicted_audience_behavior'] = np.select(conditions, choices, default='General Pop/Standard')

        # 3. Content Packaging & Release Strategy Classification
        # Identifies structural strategies (e.g., 'Stream Baiting' via massive 25+ track albums)
        self.song_content_df['release_packaging_strategy'] = np.select(
            [
                (self.song_content_df['album_type'] == 'album') & (self.song_content_df['total_tracks'] >= 22),
                (self.song_content_df['album_type'] == 'album') & (self.song_content_df['total_tracks'] < 22),
                (self.song_content_df['album_type'] == 'single')
            ],
            ['Bloated Album (Stream-Optimized)', 'Standard Album Structure', 'Focused Single Release'],
            default='Other Strategy'
        )

        # 4. Trend Momentum & Forecast Drift Indicators
        # Combines performance metrics with charting velocity to predict upcoming ranking drift
        self.song_content_df['longevity_velocity_ratio'] = (
            self.song_content_df['longevity_score'] / (self.song_content_df['days_on_chart'] + 1)
        )
        
        # High score & high days implies stability; high score but low days indicates sharp upward momentum
        self.song_content_df['trend_arrow'] = np.select(
            [
                (self.song_content_df['best_rank'] <= 10) & (self.song_content_df['longevity_velocity_ratio'] > 1.5),
                (self.song_content_df['longevity_velocity_ratio'] < 0.5)
            ],
            ['Upward Momentum / Viral Blast', 'Fading / Downward Drift'],
            default='Stable Mainstay'
        )

    def export_content_dataset(self, output_dir: str):
        """Maps calculated features back to the tracking rows, safely merges profile attributes 
        without creating column duplicates or null gaps, and saves the final deliverables.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Baseline Save 1: Save the core profile lookups
        profile_output_path = os.path.join(output_dir, "song_content_profiles.csv")
        self.song_content_df.to_csv(profile_output_path, index=False)
        logging.info(f"Song Content Profiles lookup table saved to: {profile_output_path}")

        # Baseline Save 2: Original final complete dataset
        master_output_path = os.path.join(output_dir, "final_complete_dataset.csv")
        # Base broadcast mapping
        broadcast_cols = [
            'song', 'duration_group', 'rpm_rate', 'estimated_revenue', 
            'predicted_audience_behavior', 'release_packaging_strategy', 'trend_arrow'
        ]
        base_master_df = self.df.merge(
            self.song_content_df[broadcast_cols],
            on='song',
            how='left'
        )
        base_master_df.to_csv(master_output_path, index=False)

        # =========================================================================
        # NEW IMPLEMENTATION: SAFE MERGE FOR FINAL_COMPLETE_DATASET_2024_25
        # =========================================================================
        logging.info("Executing safe merge pipeline for 'final_complete_dataset_2024_25.csv'...")
        
        # 1. Standardize and copy active time-series tracking dataframe 
        time_series_df = self.df.copy()
        
        # 2. Prevent Column Collision (_x, _y suffixes): Identify intersecting columns
        # Drop intersecting columns from the profiles table except the join key ('song')
        overlapping_columns = [
            col for col in self.song_content_df.columns 
            if col in time_series_df.columns and col != 'song'
        ]
        safe_profiles_df = self.song_content_df.drop(columns=overlapping_columns)

        # 3. Perform the merge operation safely on the 'song' index key
        merged_2024_25_df = time_series_df.merge(safe_profiles_df, on='song', how='left')

        # 4. Integrity Scrubbing: Remove row duplicates and eliminate null tracking issues
        initial_row_count = len(merged_2024_25_df)
        
        # Drop rows with duplicate entries across the primary series indices
        merged_2024_25_df = merged_2024_25_df.drop_duplicates()
        
        # Ensure no target matching key contains empty or broken null data fields
        merged_2024_25_df = merged_2024_25_df.dropna(subset=['song'])
        
        final_row_count = len(merged_2024_25_df)
        logging.info(f"Scrubbing complete. Rows adjusted from {initial_row_count} to {final_row_count}.")

        # 5. Export clean dataset to disk
        target_2024_25_path = os.path.join(output_dir, "final_complete_dataset_2024_25.csv")
        merged_2024_25_df.to_csv(target_2024_25_path, index=False)
        
        logging.info(f"Master Safe Training Dataset exported to: {target_2024_25_path}")

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
    
    INPUT_DATA = os.path.join(ROOT_DIR, "data/processed/unifiled_psa.csv")
    PROCESSED_DIR = os.path.join(ROOT_DIR, "data/processed")

    analyzer = ContentAttributeAnalyzer(INPUT_DATA)
    analyzer.load_pipeline_data()
    analyzer.process_song_level_content()
    analyzer.engineer_duration_brackets()
    analyzer.engineer_monetization_and_behavior_features() # Executing the new features
    analyzer.export_content_dataset(PROCESSED_DIR)