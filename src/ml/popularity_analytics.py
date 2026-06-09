import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class PopularityScoreAnalyzer:
    """Evaluates popularity-to-rank correlation variations and distribution properties across chart tiers."""
    
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.df = None
        self.correlation_matrix = None

    def load_pipeline_data(self):
        """Loads data from the artist analytics pipeline step."""
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input master file missing: {self.input_path}")
        
        logging.info(f"Ingesting master dataset from {self.input_path}...")
        self.df = pd.read_csv(self.input_path)
        return self.df

    def evaluate_correlations(self):
        """Computes statistical correlation coefficients to analyze popularity vs chart mechanics."""
        logging.info("Calculating Pearson and Spearman correlation matrices...")
        
        # 1. Direct Pearson vs Spearman checks
        pearson_val = self.df['popularity'].corr(self.df['position'], method='pearson')
        spearman_val = self.df['popularity'].corr(self.df['position'], method='spearman')
        
        logging.info(f"Baseline Position vs Popularity - Pearson: {pearson_val:.4f}, Spearman: {spearman_val:.4f}")

        # 2. Comprehensive feature correlation matrix relative to chart position
        target_cols = [
            'position', 'popularity', 'rank_velocity', 'rank_acceleration',
            'rolling_7d_popularity', 'global_popularity', 'rank_volatility', 'dominance_score'
        ]
        available_cols = [c for c in target_cols if c in self.df.columns]
        self.correlation_matrix = self.df[available_cols].corr()

    def engineer_rank_bracket_groups(self):
        """Divides track records into clear, non-overlapping performance rank brackets."""
        logging.info("Segmenting tracks into non-overlapping chart rank tier brackets...")
        
        # Explicit bounds prevent tracking entries from blending across multiple groups
        self.df['rank_group'] = pd.cut(
            self.df['position'],
            bins=[0, 10, 20, 50, float('inf')],
            labels=['Top 10 (1-10)', 'Top 20 (11-20)', 'Top 50 (21-50)', 'Others (>50)'],
            include_lowest=True
        )

        # Log out aggregate footprints for structural sanity checking
        summary = self.df.groupby('rank_group', observed=False)['popularity'].agg(['mean', 'median', 'count'])
        for idx, row in summary.iterrows():
            logging.info(f"Bracket {idx} -> Mean Popularity: {row['mean']:.2f}, Volume: {int(row['count'])}")

    def export_popularity_dataset(self, output_dir: str):
        """Writes the expanded popularity analytical datasets out to the local directory."""
        os.makedirs(output_dir, exist_ok=True)

        # Save the primary augmented time-series dataset
        master_output_path = os.path.join(output_dir, "unifiled_psa.csv")
        self.df.to_csv(master_output_path, index=False)
        
        # Save the raw correlation numerical matrix for configuration inspection
        corr_output_path = os.path.join(output_dir, "popularity_correlation_matrix.csv")
        self.correlation_matrix.to_csv(corr_output_path)
        
        logging.info(f"Popularity Time-Series Matrix successfully saved to: {master_output_path}")
        logging.info(f"Numerical Correlation Matrix summary saved to: {corr_output_path}")


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
    
    INPUT_DATA = os.path.join(ROOT_DIR, "data/processed/unified_dts_artisteng.csv")
    PROCESSED_DIR = os.path.join(ROOT_DIR, "data/processed")

    analyzer = PopularityScoreAnalyzer(INPUT_DATA)
    analyzer.load_pipeline_data()
    analyzer.evaluate_correlations()
    analyzer.engineer_rank_bracket_groups()
    analyzer.export_popularity_dataset(PROCESSED_DIR)
