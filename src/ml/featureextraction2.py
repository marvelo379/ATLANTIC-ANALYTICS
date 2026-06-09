import os
import logging
import pandas as pd
import numpy as np

# Configure logging for clear pipeline status visibility
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MergedFeatureProcessor:
    """Merges time-series and aggregate matrices to engineer lifecycle and cyclic forecasting parameters."""
    
    def __init__(self, features_path: str, analytics_path: str):
        self.features_path = features_path
        self.analytics_path = analytics_path
        self.df = None
    def load_and_merge_datasets(self):
        """Loads both baseline tracking matrices and joins them cleanly without column collisions."""
        if not os.path.exists(self.features_path) or not os.path.exists(self.analytics_path):
            raise FileNotFoundError("Baseline features missing. Please execute src.ml.features first.")
        
        logging.info("Ingesting baseline datasets...")
        base_df = pd.read_csv(self.features_path)
        analytics_df = pd.read_csv(self.analytics_path)
        
        # Enforce column sorting alignment
        base_df['date'] = pd.to_datetime(base_df['date'])
        base_df = base_df.sort_values(['song', 'date']).reset_index(drop=True)

        logging.info("Merging time-series rows with song aggregate analytics...")
        
        # 🆕 FIX THE COLLISION: Identify overlapping columns (except 'song' which is our join key)
        overlapping_cols = list(set(base_df.columns) & set(analytics_df.columns))
        overlapping_cols.remove('song') # Keep 'song' since we need it to merge
        
        # Drop the duplicates from analytics_df before joining so they don't create _x and _y
        analytics_df_cleaned = analytics_df.drop(columns=overlapping_cols, errors='ignore')

        # Secure relational merge
        self.df = pd.merge(base_df, analytics_df_cleaned, on='song', how='left')
        return self.df
    

    # def load_and_merge_datasets(self):
    #     """Loads both baseline tracking matrices and joins them securely using the song column."""
    #     if not os.path.exists(self.features_path) or not os.path.exists(self.analytics_path):
    #         raise FileNotFoundError("Baseline features missing. Please execute src.ml.features first.")
        
    #     logging.info("Ingesting baseline datasets...")
    #     base_df = pd.read_csv(self.features_path)
    #     analytics_df = pd.read_csv(self.analytics_path)
        
    #     # Enforce column sorting alignment
    #     base_df['date'] = pd.to_datetime(base_df['date'])
    #     base_df = base_df.sort_values(['song', 'date']).reset_index(drop=True)

    #     logging.info("Merging time-series rows with song aggregate analytics...")
    #     # Relational merge ensures every row now possesses 'rank_volatility' and 'avg_rank'
    #     self.df = pd.merge(base_df, analytics_df, on='song', how='left')
    #     return self.df

    def engineer_lifecycle_and_movements(self):
        """Processes granular row vectors into discrete trend lifecycle categories."""
        logging.info("Evaluating rank_volatility fields to define behavioral categories...")

        # 1. Custom Song Classification Matrix Mapping
        #the formula constructed by using reallife spotify based formula.tiktok
        #THOUGHT: for accurate formula i would hv also used kmean cluster , dbscan to dervie formula
        #but because this dts is similar to spotify used hardcoded 
        def _classify_song(row):
            if row['rank_velocity'] > 0 and row['decay_rate'] < 0.3:
                return '🚀 Rising (Viral Growth)'
            elif row['rank_velocity'] < 0 and row['decay_rate'] > 0.7:
                return '📉 Fading Hit'
            elif row['rank_volatility'] < 10 and row['avg_rank'] < 50:
                return '🎯 Stable Hit'
            elif row['rank_volatility'] > 20:
                return '⚡ Volatile / Spike Track'
            else:
                return '🎧 Catalog / Steady'

        self.df['lifecycle'] = self.df.apply(_classify_song, axis=1)

        # 2. Daily Rank Delta Movement Tracking
        self.df['rank_change'] = self.df.groupby('song')['position'].diff()
        
        def _movement_type(x):
            if pd.isna(x):
                return "→ Stable"
            return "⬆ Rising" if x < 0 else ("⬇ Falling" if x > 0 else "→ Stable")

        self.df['movement'] = self.df['rank_change'].apply(_movement_type)

    def identify_momentum_anomalies(self):
        """Isolates high-impact market tracks (Fast Risers vs Slow Decliners)."""
        logging.info("Identifying momentum segments via conditional boundaries...")
        
        days_threshold = self.df['days_on_chart'].quantile(0.6)
        volatility_median = self.df['rank_volatility'].median()

        self.df["is_fast_riser"] = (
            (self.df["rank_velocity"] > 0) &
            (self.df["rank_acceleration"] > 0) &
            (self.df["rolling_7d_popularity"] > self.df["rolling_30d_popularity"]) &
            (self.df["days_on_chart"] <= days_threshold)
        )

        self.df["is_slow_decliner"] = (
            (self.df["rank_velocity"] < 0) &
            (self.df["decay_rate"] > 0) &
            (self.df["days_since_peak"] > 0) &
            (self.df["rolling_7d_popularity"] < self.df["rolling_30d_popularity"]) &
            (self.df["rank_volatility"] <= volatility_median)
        )

        self.df["segment"] = "Neutral"#default i labeleed my song as neutral 
        #if fastrise condition met then fill fast riser
        self.df.loc[self.df["is_fast_riser"], "segment"] = "Fast Riser"
        #loc[row_selected,col_selected]=filler ...
        self.df.loc[self.df["is_slow_decliner"], "segment"] = "Slow Decliner"

    def calculate_cyclic_time_features(self):
        """Transforms standard timeline attributes to cyclic variables for 2027/2028 predictions."""
        logging.info("Adding time-series cyclic coordinates to the merged dataset...")
        
        months = self.df['date'].dt.month
        #cyclic encoded:seasonsality prediction, (cos theta,sin theta coordinate of all point)
        self.df['month_sin'] = np.sin(2 * np.pi * months / 12.0)
        #2 pie m /12
        #THOUGHT: PLOT SEASONALITY POPULARITY , PERIODIC STREAMING PATTERN,RELEASE TIMING EFFECT, MARKET CYCLE
        #HOLIDAY HIT , SEASON VIRAL , YR VIRAL 
        self.df['month_cos'] = np.cos(2 * np.pi * months / 12.0)
        self.df['year_engineered'] = self.df['date'].dt.year + (self.df['date'].dt.dayofyear / 365.25)
          #time ttal converted with base year rather than analysing three parameter it has continuous 1 parameter to analyse with continuous values 
          #THOUGHT: TEMPORAL DRIFT , LONG TREND , 
    def export_final_dataset(self, output_dir: str):
        """Writes the consolidated data layout out to your processed local directory."""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "final_dataset_merged_features.csv")
        self.df.to_csv(output_path, index=False)
        logging.info(f"Consolidated feature dataset exported to: {output_path}")


if __name__ == "__main__":
    # Handle environment root path mapping
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
    
    BASE_FEATURES = os.path.join(ROOT_DIR, "data/processed/engineered_features.csv")
    BASE_ANALYTICS = os.path.join(ROOT_DIR, "data/processed/top_song_analytics.csv")
    PROCESSED_DIR = os.path.join(ROOT_DIR, "data/processed")

    # Run execution pipeline
    processor = MergedFeatureProcessor(BASE_FEATURES, BASE_ANALYTICS)
    processor.load_and_merge_datasets()
    processor.engineer_lifecycle_and_movements()
    processor.identify_momentum_anomalies()
    processor.calculate_cyclic_time_features()
    processor.export_final_dataset(PROCESSED_DIR)


#END PLAYLIST BASED ANALYTICS DRIVEN !!!!
