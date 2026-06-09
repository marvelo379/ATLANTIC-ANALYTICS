import os
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ArtistDominanceAnalyzer:
    """Computes career longevity weights, footprint metrics, and composite market dominance indices."""
    
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.df = None
        self.artist_stats = None

    def load_pipeline_data(self):
        """Loads data from the master song analytics pipeline step."""
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input feature file missing: {self.input_path}")
        
        logging.info(f"Ingesting master dataset from {self.input_path}...")
        self.df = pd.read_csv(self.input_path)
        
        # Enforce clean typing requirements
        if 'artist_preprocessed' not in self.df.columns:
            self.df['artist_preprocessed'] = self.df['artist'].str.strip()
            
        return self.df

    def process_artist_dominance_metrics(self):
        """Generates catalog footprints, longevity factors, and composite standard scores."""
        logging.info("Calculating artist footprint aggregations across multi-song profiles...")
        
        # 1. Isolate unique song records to evaluate true track lifespan volumes
        song_days = self.df[['artist_preprocessed', 'song', 'days_on_chart']].drop_duplicates()
        
        # 2. Base catalog footprint summaries
        self.artist_stats = song_days.groupby('artist_preprocessed').agg(
            total_days_on_playlist=('days_on_chart', 'sum'),
            unique_songs=('song', 'nunique')
        ).reset_index()

        # 3. Calculate True Average Hit Longevity per Artist
        self.artist_stats['avg_song_longevity'] = (
            self.artist_stats['total_days_on_playlist'] / self.artist_stats['unique_songs']
        )

        # 4. Standardize Components using Scikit-Learn StandardScaler
        logging.info("Fitting standard scaler matrices across catalog longevity and volume...")
        scaler = StandardScaler()
        self.artist_stats[['t_z', 's_z', 'l_z']] = scaler.fit_transform(
            self.artist_stats[['total_days_on_playlist', 'unique_songs', 'avg_song_longevity']]
        )

        # 5. Composite Weighted Dominance Score Formula
        # 50% Weight: Absolute Chart Presence Volume (Market Real Estate)
        # 20% Weight: Creative Prolific Output Volume (Unique Hits)
        # 30% Weight: Average Song Retention Strength (Staying Power)
        #THESES ARE ZSCORE , HOW THE AVG VAL DEVIATE FROM ACTUAL
        self.artist_stats['dominance_score'] = (
            (0.5 * self.artist_stats['t_z']) + 
            (0.2 * self.artist_stats['s_z']) + 
            (0.3 * self.artist_stats['l_z'])
        )
        #VALUE 1-,1 :NORMAL;  -2 ,2 COMMON, -3,3 WIDE , OUTLIER
        #neagtive below avg, 0 avg , greater than 0 dominated  singer , much higher or much lower value mean outlier 
        

    def generate_final_artist_dataset(self, output_dir: str):
        """Maps computed composite stats back to transaction tracking rows and archives files."""
        os.makedirs(output_dir, exist_ok=True)

        logging.info("Merging career dominance metrics back into master dataset framework...")
        unified_df = self.df.merge(
            self.artist_stats[[
                'artist_preprocessed', 'total_days_on_playlist', 
                'unique_songs', 'avg_song_longevity', 'dominance_score'
            ]],
            on='artist_preprocessed',
            how='left'
        )

        # Export Production Deliverables
        stats_path = os.path.join(output_dir, "artist_analytics_profiles.csv")
        self.artist_stats.to_csv(stats_path, index=False)
        
        master_path = os.path.join(output_dir, "unified_dts_artisteng.csv")
        unified_df.to_csv(master_path, index=False)
        
        logging.info(f"Artist Career Scoreboards saved successfully to: {stats_path}")
        logging.info(f"Final Pipeline Master Dataset saved successfully to: {master_path}")


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
    
    INPUT_DATA = os.path.join(ROOT_DIR, "data/processed/unified_feng.csv")
    PROCESSED_DIR = os.path.join(ROOT_DIR, "data/processed")

    analyzer = ArtistDominanceAnalyzer(INPUT_DATA)
    analyzer.load_pipeline_data()
    analyzer.process_artist_dominance_metrics()
    analyzer.generate_final_artist_dataset(PROCESSED_DIR)
