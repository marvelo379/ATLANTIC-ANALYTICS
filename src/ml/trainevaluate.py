# import os
# import logging
# import joblib
# import numpy as np
# import pandas as pd
# from scipy.stats import norm
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.preprocessing import RobustScaler
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from catboost import CatBoostClassifier

# # Configure logging for full pipeline execution visibility
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# class MLForecastingPipeline:
#     """Production architecture driving regression trajectory forecasting and multi-horizon survival classification."""
    
#     def __init__(self, data_path: str, output_dir: str):
#         self.data_path = data_path
#         self.output_dir = output_dir
#         self.df = None
#         self.scaler = RobustScaler()

        
#         # Define clean operational features completely free of look-ahead target leaks
#         self.features = [
#             "position", "rank_velocity", "rank_acceleration", "rank_change",
#             "popularity", "global_popularity", "global_local_divergence", "popularity_rank_ratio",
#             "rolling_7d_popularity", "rolling_30d_popularity",
#             "position_roll_mean_7", "position_roll_std_7", "popularity_roll_mean_7", "popularity_roll_std_7",
#             "rank_lag_1", "rank_lag_2", "rank_lag_3", "rank_lag_7", "rank_lag_14", "rank_lag_30",
#             "days_on_chart", "days_since_peak", "decay_rate", "duration_min", "is_explicit", "total_tracks",
#             "dominance_score", "unique_songs", "day_of_week", "month", "quarter", "week_of_year",
#             "month_sin", "month_cos", "year_engineered"
#         ]

#     def load_and_clean_data(self):
#         print("loaded the file load n clean")
#         """Loads master datasets, formats timestamps, and isolates target fields safely."""
#         if not os.path.exists(self.data_path):
#             raise FileNotFoundError(f"Final merged dataset missing: {self.data_path}")
        
#         logging.info(f"Ingesting master dataset from {self.data_path}...")
#         self.df = pd.read_csv(self.data_path)
#         self.df["date"] = pd.to_datetime(self.df["date"])
#         self.df = self.df.sort_values(by=["song", "artist", "date"]).reset_index(drop=True)
        
#         # Eliminate structural mirror duplicates safely
#         self.df = self.df.drop_duplicates(subset=["song", "artist", "date"]).reset_index(drop=True)
        
#         # Explicitly protect the feature engine by purging only true systemic future leakage fields
#         leakage_to_purge = ["hit_score", "performance_score", "total_days_on_playlist", "avg_song_longevity"]
#         self.df = self.df.drop(columns=[c for c in leakage_to_purge if c in self.df.columns], errors="ignore")

#     def engineer_pipeline_features(self):
#         """Builds lags, rolling brackets, non-overlapping target shifts, and survival boundaries."""
#         logging.info("Engineering longitudinal timelines, lag arrays, and cyclic variables...")

#         # 1. Temporal Calendar Blocks
#         self.df["day_of_week"] = self.df["date"].dt.dayofweek
#         self.df["month"] = self.df["date"].dt.month
#         self.df["quarter"] = self.df["date"].dt.quarter
#         self.df["week_of_year"] = self.df["date"].dt.isocalendar().week.astype(int)
        
#         # Protect against Division-by-Zero errors
#         self.df["popularity_rank_ratio"] = self.df["popularity"] / (self.df["position"] + 1e-5)

#         # 2. Sequential Group-By Lags and Rolling Windows
#         grouped = self.df.groupby(["song", "artist"])
        
#         for lag in [1, 2, 3, 7, 14, 30]:
#             self.df[f"rank_lag_{lag}"] = grouped["position"].shift(lag)

#         # Vectorized rolling operations to prevent shape fragmentation
#         self.df["position_roll_mean_7"] = grouped["position"].transform(lambda x: x.rolling(7, min_periods=1).mean())
#         self.df["position_roll_std_7"] = grouped["position"].transform(lambda x: x.rolling(7, min_periods=1).std()).fillna(0)
#         self.df["popularity_roll_mean_7"] = grouped["popularity"].transform(lambda x: x.rolling(7, min_periods=1).mean())
#         self.df["popularity_roll_std_7"] = grouped["popularity"].transform(lambda x: x.rolling(7, min_periods=1).std()).fillna(0)

#         # 3. Targets: Regression (7-Day Rank Drift) & Multi-Horizon Longevity Status
#         self.df["target_rank_change_7d"] = grouped["position"].shift(-7) - self.df["position"]
        
#         # Multi-Horizon Survival Definition: Does the track survive in the chart (position <= 50) 30 days from now?
#         self.df["target_survives_30d"] = (grouped["position"].shift(-30).fillna(100) <= 50).astype(int)

#         # 4. Impute Autoregressive Padding safely without breaking long histories
#         cols_to_fill = [c for c in self.df.columns if c.startswith("rank_lag_") or "roll_" in c]
#         self.df[cols_to_fill] = grouped[cols_to_fill].ffill().bfill()
        
#         # Global fallback safeguard for single-appearance songs
#         global_median = self.df[self.features].median(numeric_only=True)
#         self.df[self.features] = self.df[self.features].fillna(global_median)

#         # Drop rows where forward regression window goals are mathematically uncomputable
#         self.df = self.df.dropna(subset=["target_rank_change_7d", "target_survives_30d"]).reset_index(drop=True)

#     def train_predictive_models(self):
#         """Executes Time-Series Splits, fits core estimators, and calculates P95 intervals."""
#         logging.info("Initializing modeling pipelines across historical splits...")
        
#         # Select active parameters
#         X = self.df[self.features].copy()
#         y_reg = self.df["target_rank_change_7d"].copy()
#         y_clf = self.df["target_survives_30d"].copy()

#         # Enforce temporal alignment via robust scaling coordinates
#         X_scaled = self.scaler.fit_transform(X)
#         X_scaled_df = pd.DataFrame(X_scaled, columns=self.features)

#         # Set up sequential Time-Series validation splitting
#         tscv = TimeSeriesSplit(n_splits=3)
#         for train_idx, val_idx in tscv.split(X_scaled_df):
#             X_train, X_val = X_scaled_df.iloc[train_idx], X_scaled_df.iloc[val_idx]
#             y_t_reg, y_v_reg = y_reg.iloc[train_idx], y_reg.iloc[val_idx]
#             y_t_clf, y_v_clf = y_clf.iloc[train_idx], y_clf.iloc[val_idx]

#         logging.info("Fitting LightGBM Regressor and CatBoost Classifier models...")
#         print("bkjndskjnwknjsw")
        
#         # Model 1: Trajectory Regression Estimator
#         reg_model = LGBMRegressor(n_estimators=150, learning_rate=0.05, random_state=42, verbose=-1)
#         reg_model.fit(X_train, y_t_reg, eval_set=[(X_val, y_v_reg)])

#         # Model 2: Multi-Horizon Longevity Classifier
#         clf_model = CatBoostClassifier(iterations=150, learning_rate=0.05, random_state=42, verbose=0)
#         clf_model.fit(X_train, y_t_clf, eval_set=[(X_val, y_v_clf)])

#         # 5. Compute P95 Standard Error Margins
#         val_predictions = reg_model.predict(X_val)
#         residuals = y_v_reg - val_predictions
#         residual_std = np.std(residuals)

#         # Archive artifacts to disk
#         os.makedirs(self.output_dir, exist_ok=True)
#         joblib.dump(reg_model, os.path.join(self.output_dir, "trajectory_regressor.pkl"))
#         joblib.dump(clf_model, os.path.join(self.output_dir, "longevity_classifier.pkl"))
#         joblib.dump(self.scaler, os.path.join(self.output_dir, "robust_scaler.pkl"))
#         np.save(os.path.join(self.output_dir, "residual_std.npy"), residual_std)
        
#         logging.info("Model artifacts and weights successfully archived.")
#         self.generate_dashboard_projections(reg_model, clf_model, residual_std)

#     def generate_dashboard_projections(self, reg, clf, res_std):
#         """Generates mock future projection frames for 2027/2028 dashboard UI testing."""
#         logging.info("Generating predictive 2027/2028 dashboard mock timeline matrices...")
        
#         # Build an extension mapping vector across a 2-year forecast horizon
#         future_dates = pd.date_range(start="2026-06-01", end="2028-12-31", freq="D")
#         future_df = pd.DataFrame({"date": future_dates})
        
#         # Backfill standard static attributes using current dataset parameters
#         future_df["month"] = future_df["date"].dt.month
#         future_df["year_engineered"] = future_df["date"].dt.year + (future_df["date"].dt.dayofyear / 365.25)
#         future_df["month_sin"] = np.sin(2 * np.pi * future_df["month"] / 12.0)
#         future_df["month_cos"] = np.cos(2 * np.pi * future_df["month"] / 12.0)
        
#         # Populate additional feature parameters with data baselines
#         for feat in self.features:
#             if feat not in future_df.columns:
#                 future_df[feat] = self.df[feat].median()

#         # Transform and predict across the future space
#         scaled_future = self.scaler.transform(future_df[self.features])
#         predictions = reg.predict(scaled_future)
#         probabilities = clf.predict_proba(scaled_future)[:, 1]

#         # Calculate P95 boundary values (Z = 1.96)
#         z_score = 1.96
#         future_df["predicted_rank_change"] = predictions
#         future_df["p95_lower_bound"] = predictions - (z_score * res_std)
#         future_df["p95_upper_bound"] = predictions + (z_score * res_std)
#         future_df["survival_probability_score"] = probabilities

#         # Save dashboard data frame out to disk
#         dashboard_output = os.path.join(self.output_dir, "dashboard_predictions_2027_2028.csv")
#         future_df.to_csv(dashboard_output, index=False)
#         logging.info(f"Dashboard backend frames successfully exported to: {dashboard_output}")


# if __name__ == "__main__":
#     # Configure workspace layout anchors
#     SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#     ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
    
#     INPUT_PATH = os.path.join(ROOT_DIR, "data/processed/final_complete_dataset.csv")
#     OUTPUT_PATH = os.path.join(ROOT_DIR, "data/processed")

#     pipeline = MLForecastingPipeline(INPUT_PATH, OUTPUT_PATH)
