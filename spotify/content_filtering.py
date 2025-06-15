import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from category_encoders.count import CountEncoder

from data_cleaning import prune_for_content_filtering

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Data & model paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

@dataclass(frozen=True, slots=True)
class Paths:
    cleaned_csv: Path = PROCESSED_DIR / "cleaned_data.csv"
    transformer_model: Path = MODELS_DIR / "transformer.joblib"
    transformed_output: Path = MODELS_DIR / "transformed_data.npz"

# Feature groups
FREQ_ENCODE_COLS: List[str] = ["year"]
OHE_COLS: List[str] = ["artist", "time_signature", "key"]
TFIDF_COL: str = "tags"
STANDARD_SCALE_COLS: List[str] = ["duration_ms", "loudness", "tempo"]
MINMAX_SCALE_COLS: List[str] = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence"
]

def train_feature_transformer(df: pd.DataFrame, save_path: Path) -> None:
    preprocessor = ColumnTransformer(
        transformers=[
            ("freq", CountEncoder(normalize=True, return_df=True), FREQ_ENCODE_COLS),
            ("ohe", OneHotEncoder(handle_unknown="ignore"), OHE_COLS),
            ("tfidf", TfidfVectorizer(max_features=85), TFIDF_COL),
            ("standard", StandardScaler(), STANDARD_SCALE_COLS),
            ("minmax", MinMaxScaler(), MINMAX_SCALE_COLS),
        ],
        remainder="passthrough",
        n_jobs=-1,
        verbose=False,
        force_int_remainder_cols=False,
    )
    preprocessor.fit(df)
    joblib.dump(preprocessor, save_path)
    logging.info(f"Transformer saved to {save_path}")

def transform_dataset(df: pd.DataFrame, transformer_path: Path) -> np.ndarray:
    preprocessor = joblib.load(transformer_path)
    return preprocessor.transform(df)

def save_transformed_array(array, path: Path) -> None:
    save_npz(path, array)
    logging.info(f"Transformed data saved to {path}")

def compute_similarity_scores(query_vec, matrix) -> np.ndarray:
    return cosine_similarity(query_vec, matrix)

def get_top_k_recommendations(
    query_name: str,
    query_artist: str,
    raw_df: pd.DataFrame,
    features_matrix,
    top_k: int = 10
) -> pd.DataFrame:
    query_name, query_artist = query_name.lower(), query_artist.lower()
    query_match = raw_df[
        (raw_df["name"] == query_name) & (raw_df["artist"] == query_artist)
    ]

    if query_match.empty:
        raise ValueError(f"Song '{query_name}' by '{query_artist}' not found.")

    query_idx = query_match.index[0]
    query_vec = features_matrix[query_idx].reshape(1, -1)

    sim_scores = compute_similarity_scores(query_vec, features_matrix)
    ranked_indices = np.argsort(sim_scores.ravel())[::-1]
    ranked_indices = [i for i in ranked_indices if i != query_idx][:top_k]

    recommendations = raw_df.iloc[ranked_indices][
        ["name", "artist", "spotify_preview_url"]
    ].reset_index(drop=True)

    return recommendations

def main():
    paths = Paths()

    logging.info(f"Loading cleaned data from {paths.cleaned_csv} …")
    df_raw = pd.read_csv(paths.cleaned_csv)
    df_cleaned = prune_for_content_filtering(df_raw)

    logging.info("Training feature transformer …")
    train_feature_transformer(df_cleaned, paths.transformer_model)

    logging.info("Transforming dataset …")
    transformed_matrix = transform_dataset(df_cleaned, paths.transformer_model)

    save_transformed_array(transformed_matrix, paths.transformed_output)

if __name__ == "__main__":
    main()