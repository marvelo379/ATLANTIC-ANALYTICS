import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class PlaylistFeatureExtractor:
    """Calculates granular track metrics, rolling statistics, velocity, and performance scores."""
    
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.df = None
        self.song_metrics = None

    def load_processed_data(self):
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Processed file missing: {self.input_path}")
        self.df = pd.read_csv(self.input_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values(['song', 'date']).reset_index(drop=True)

    def extract_song_level_metrics(self):
        """Generates static aggregate track summary metrics."""
        self.df['duration_min'] = self.df['duration_ms'] / 60000.0
        
        self.song_metrics = (
            self.df.groupby('song')
            .agg(
                days_on_chart=('date', 'nunique'),
                avg_rank=('position', 'mean'),
                best_rank=('position', 'min'),
                worst_rank=('position', 'max'),
                rank_volatility=('position', 'std'),
                avg_popularity=('popularity', 'mean'),
                duration_min=('duration_min', 'mean')
            )
            .reset_index()
        )
        self.song_metrics['rank_volatility'] = self.song_metrics['rank_volatility'].fillna(0.0)

    def calculate_performance_scores(self):
        """Calculates normalized indexing bounds and the final performance metric."""
        sm = self.song_metrics
        sm['days_norm'] = sm['days_on_chart'] / sm['days_on_chart'].max()
        sm['rank_norm'] = 1.0 - (sm['avg_rank'] / 100.0)
        sm['volatility_norm'] = 1.0 - (sm['rank_volatility'] / sm['rank_volatility'].max())
        sm['best_rank_norm'] = 1.0 - (sm['best_rank'] / 100.0)
          #this formula decided after plot n it can be deviated too , guessed probability 
        sm['performance_score'] = (
            sm['days_norm'] * 0.35 +
            sm['rank_norm'] * 0.35 +
            sm['best_rank_norm'] * 0.20 +
            sm['volatility_norm'] * 0.10
        )

  #performance metric also range from 0 to 1, 1 strong performance , close 0 poor performnce 
    def calculate_rolling_and_time_features(self):
        """Computes rolling variations, velocities, fatigue profiles, and divergences."""
        # Rolling Averages
        self.df['rolling_7d_popularity'] = self.df.groupby('song')['popularity'].transform(lambda x: x.rolling(7, min_periods=1).mean())
        self.df['rolling_30d_popularity'] = self.df.groupby('song')['popularity'].transform(lambda x: x.rolling(30, min_periods=1).mean())

        # Velocity Metrics
        self.df['rank_velocity'] = self.df.groupby('song')['position'].diff()
        self.df['rank_acceleration'] = self.df.groupby('song')['rank_velocity'].diff()

        # Fatigue & Decay Modeling
        peak_rank = self.df.groupby('song')['position'].min().reset_index(name='peak_rank')
        peak_date = self.df.loc[self.df.groupby('song')['position'].idxmin()][['song', 'date']].rename(columns={'date': 'peak_date'})
        decay_df = peak_rank.merge(peak_date, on='song')#how quickly a song loses popularity after reaching its peak.
        #decay=currrank-peakrank/days since peak
        #THOUGHT: POST-PEAK TREND TRACK,AVG RANK AFTER PEAK,RETENTION SCORE,SURVIVAL MODELING,POPULARITY DROP 


        #peak_rank n peak_date :-->decay analysys,post peak trend, survival modeling,popularity drop calculate

        self.df = self.df.merge(decay_df, on='song')
        self.df['days_since_peak'] = (self.df['date'] - self.df['peak_date']).dt.days
        self.df['decay_rate'] = self.df['position'] - self.df['peak_rank']

        # Global Divergence
        #THOUGHT: RANK DIVERGENCE, POPULARITY DIVERGENCE , VOLATILITY
        global_pop = self.df.groupby('date')['popularity'].mean().reset_index(name='global_popularity')
        self.df = self.df.merge(global_pop, on='date')
        self.df['global_local_divergence'] = self.df['popularity'] - self.df['global_popularity']

    def export_features(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.df.to_csv(os.path.join(output_dir, "engineered_features.csv"), index=False)
        
        top_songs = self.song_metrics.sort_values('performance_score', ascending=False)
        top_songs.to_csv(os.path.join(output_dir, "top_song_analytics.csv"), index=False)
        logging.info("Feature matrices successfully stored.")

if __name__ == "__main__":
     # Finds the exact folder where features.py lives (src/ml/)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Navigates up two levels to the root directory (my-ml-saas/)
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
    
    # Safely stitches together the absolute system paths
    INPUT_FILE = os.path.join(ROOT_DIR, "data/processed/unified_dataingestion.csv")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "data/processed")

    extractor = PlaylistFeatureExtractor(INPUT_FILE)
    extractor.load_processed_data()
    extractor.extract_song_level_metrics()
    extractor.calculate_performance_scores()
    extractor.calculate_rolling_and_time_features()
    extractor.export_features(OUTPUT_DIR)
   