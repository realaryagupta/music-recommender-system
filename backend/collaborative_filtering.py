import pandas as pd
import dask.dataframe as dd 
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from sklearn.metrics.pairwise import cosine_similarity
import logging
import sys
from pathlib import Path

# ---------- Logging Setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

# ---------- File Paths ----------


TRACK_IDS_PATH = "../models/track_ids.npy"
FILTERED_SONGS_PATH = "../data/processed/collab_filtered_data.csv"
INTERACTION_MATRIX_PATH = "../models/interaction_matrix.npz"
SONGS_DATA_PATH = "../data/processed/cleaned_data.csv"
USER_HISTORY_PATH = "../data/raw/User Listening History.csv"

# ---------- Utility Functions ----------

def save_to_csv(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV."""
    logging.info(f"Saving DataFrame to {path}")
    df.to_csv(path, index=False)


def save_sparse_matrix(matrix: csr_matrix, path: str) -> None:
    """Save sparse matrix to file."""
    logging.info(f"Saving sparse matrix to {path}")
    save_npz(path, matrix)


def save_track_ids(track_ids: np.ndarray, path: str) -> None:
    """Save track IDs to a .npy file."""
    logging.info(f"Saving track IDs to {path}")
    np.save(path, track_ids, allow_pickle=True)


# ---------- Core Processing Functions ----------

def filter_songs_by_ids(songs_df: pd.DataFrame, track_ids: list, output_path: str) -> pd.DataFrame:
    """Filter songs dataset by track IDs and save to CSV."""
    logging.info("Filtering songs by track IDs...")
    filtered = songs_df[songs_df["track_id"].isin(track_ids)].sort_values("track_id").reset_index(drop=True)
    save_to_csv(filtered, output_path)
    logging.info(f"Filtered songs saved with {len(filtered)} entries.")
    return filtered


def create_interaction_matrix(history_df: dd.DataFrame, track_ids_path: str, matrix_path: str) -> csr_matrix:
    """Generate and save a sparse interaction matrix from user listening history."""
    logging.info("Creating interaction matrix...")
    df = history_df.copy()
    df['playcount'] = df['playcount'].astype(np.float64)
    df = df.categorize(columns=['user_id', 'track_id'])

    user_idx = df['user_id'].cat.codes
    track_idx = df['track_id'].cat.codes
    track_ids = df['track_id'].cat.categories.values

    save_track_ids(track_ids, track_ids_path)

    df = df.assign(user_idx=user_idx, track_idx=track_idx)

    logging.info("Grouping and computing playcounts...")
    grouped = df.groupby(['track_idx', 'user_idx'])['playcount'].sum().reset_index().compute()

    interaction = csr_matrix(
        (grouped['playcount'], (grouped['track_idx'], grouped['user_idx'])),
        shape=(grouped['track_idx'].nunique(), grouped['user_idx'].nunique())
    )

    save_sparse_matrix(interaction, matrix_path)
    logging.info("Interaction matrix created and saved.")
    return interaction


def collaborative_recommendation(song_name: str, artist_name: str, track_ids: np.ndarray,
                                 songs_df: pd.DataFrame, interaction_matrix: csr_matrix, k: int = 5) -> pd.DataFrame:
    """Generate top-k song recommendations using collaborative filtering."""
    logging.info(f"Generating recommendations for: {song_name} by {artist_name}")
    song_row = songs_df[
        (songs_df["name"].str.lower() == song_name.lower()) &
        (songs_df["artist"].str.lower() == artist_name.lower())
    ]

    if song_row.empty:
        logging.error("Song not found in dataset.")
        raise ValueError("Song not found in dataset.")

    input_track_id = song_row['track_id'].values.item()
    song_index = np.where(track_ids == input_track_id)[0].item()

    input_vector = interaction_matrix[song_index]
    similarity = cosine_similarity(input_vector, interaction_matrix).ravel()

    top_indices = np.argsort(similarity)[-k-1:][::-1]
    top_scores = similarity[top_indices]
    top_track_ids = track_ids[top_indices]

    scores_df = pd.DataFrame({"track_id": top_track_ids, "score": top_scores})

    recommended_songs = (
        songs_df[songs_df["track_id"].isin(top_track_ids)]
        .merge(scores_df, on="track_id")
        .sort_values("score", ascending=False)
        .drop(columns=["track_id", "score"])
        .reset_index(drop=True)
    )

    logging.info(f"Top {k} recommendations generated.")
    return recommended_songs


# ---------- Main Execution ----------

def main():
    logging.info("Starting pipeline...")

    try:
        user_history = dd.read_csv(USER_HISTORY_PATH)
        track_ids = user_history["track_id"].unique().compute().tolist()
        logging.info(f"Loaded user history with {len(track_ids)} unique track IDs.")

        songs_df = pd.read_csv(SONGS_DATA_PATH)
        filter_songs_by_ids(songs_df, track_ids, FILTERED_SONGS_PATH)

        create_interaction_matrix(user_history, TRACK_IDS_PATH, INTERACTION_MATRIX_PATH)

        logging.info("Pipeline completed successfully.")
    except Exception as e:
        logging.exception(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()
