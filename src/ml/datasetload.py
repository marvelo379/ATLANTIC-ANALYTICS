import logging
import os
import pandas as pd

# Set up logging for production visibility
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class PlaylistDatasetProcessor:
    """Handles data ingestion, structural validation, and preprocessing for playlist snapshots."""
    
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """Loads the raw daily playlist snapshots from a CSV file."""
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found at: {self.input_path}")
        
        logging.info(f"Loading data from {self.input_path}...")
        self.df = pd.read_csv(self.input_path)
        return self.df

    def validate_data(self) -> bool:
        """Validates critical constraints like rank ranges and structural bounds."""
        if self.df is None:
            raise ValueError("Data has not been loaded. Call load_data() first.")

        # 1. Validate rank range (1–50)
        invalid_ranks = self.df[(self.df["position"] < 1) | (self.df["position"] > 50)]
        if not invalid_ranks.empty:
            logging.warning(f"Found {len(invalid_ranks)} entries with invalid positions outside [1, 50].")
            # Production fix: Drop out-of-bounds ranks if they exist
            self.df = self.df[(self.df["position"] >= 1) & (self.df["position"] <= 50)]
        else:
            logging.info("Validation passed: All rank ranges are well-formatted within {1,50}.")
            
        return True

    def process_data(self) -> pd.DataFrame:
        """Cleans duplicates, standardizes artist formats, and builds features."""
        if self.df is None:
            raise ValueError("Data has not been loaded. Call load_data() first.")

        # 1. Feature Engineering: Count original duplicates before dropping
        self.df['duplicate_per_song_date'] = (
            self.df.groupby(['song', 'date'])['song'].transform('count')
        )

        # 2. Handle duplicate song-date entries (Keep first occurrence)
        duplicate_count = self.df.duplicated(subset=['song', 'date']).sum()
        logging.info(f"Identified {duplicate_count} duplicate song-date entries. Dropping duplicates...")
        
        self.df = self.df.drop_duplicates(subset=['song', 'date'], keep='first')
        self.df = self.df.reset_index(drop=True)

        # 3. Standardize artist name formatting & Feature Generation
        if 'artist' in self.df.columns:
            self.df['artist_original'] = self.df['artist']
            
            # Remove extra whitespaces
            cleaned_artist = self.df['artist'].str.strip()
            # Standardize multi-artist spacing around '&'
            cleaned_artist = cleaned_artist.str.replace(r'\s*&\s*', ' & ', regex=True)
            # Collapse any double spaces down to a single space
            cleaned_artist = cleaned_artist.str.replace(r'\s+', ' ', regex=True)
            
            self.df['artist_preprocessed'] = cleaned_artist
            
            # Generate structured artist list
            self.df['artist_preprocessed_list'] = self.df['artist_preprocessed'].str.split(r'\s*&\s*')
        else:
            logging.error("Critical column 'artist' is missing from the dataset.")

        logging.info("Data preprocessing completed successfully.")
        return self.df

    def save_processed_data(self, output_path: str) -> None:
        """Saves the preprocessed, validated data back to the clean data directory."""
        if self.df is None:
            raise ValueError("No data available to save. Run process_data() first.")
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, index=False)
        logging.info(f"Cleaned dataset saved successfully to: {output_path}")

import os
# Execution wrapper for production pipeline integration
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.normpath(os.path.join(BASE_DIR, "../.."))
    RAW_DATA_PATH = os.path.normpath(os.path.join(
    REPO_ROOT, 
    "data/raw/Atlantic_United_States.csv"
                                ))

    PROCESSED_DATA_PATH = os.path.normpath(os.path.join(
    REPO_ROOT, 
    "data/processed/unified_dataingestion.csv"
         ))
    # In production, these paths will come from your config files (e.g., config/config.yaml)
    # RAW_DATA_PATH = "C:/Users//unified_mlproject1/data/raw/Atlantic_United_States.csv"
    
    # PROCESSED_DATA_PATH = "C:/Users//unified_mlproject1/data/processed/unified_dataingestion.csv"

    # Instantiate and run pipeline steps sequentially
    processor = PlaylistDatasetProcessor(input_path=RAW_DATA_PATH)
    processor.load_data()
    processor.validate_data()
    processor.process_data()
    processor.save_processed_data(output_path=PROCESSED_DATA_PATH)
