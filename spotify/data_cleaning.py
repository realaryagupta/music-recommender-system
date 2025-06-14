import pandas as pd


def preprocess_tracks(df: pd.DataFrame) -> pd.DataFrame:
    # Remove duplicate tracks, keeping the first occurrence
    df = df.loc[~df.duplicated("track_id")]

    # Drop columns that are not required
    df = df.drop(columns=["genre", "spotify_id"])

    # Fill missing tag information with a descriptive placeholder
    df["tags"] = df["tags"].fillna("no_tags")

    # Enforce lowercase for selected text columns
    for col in ("name", "artist", "tags"):
        df[col] = df[col].str.lower()

    return df.reset_index(drop=True)


def prune_for_content_filtering(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["track_id", "name", "spotify_preview_url"])


def run_pipeline(path: str) -> None:
    raw_df = pd.read_csv(path)
    cleaned_df = preprocess_tracks(raw_df)
    cleaned_df.to_csv("../data/processed/cleaned_data.csv", index=False)


if __name__ == "__main__":
    # Path to the raw data file
    CSV_PATH = "../data/raw/Music Info.csv"

    run_pipeline(CSV_PATH)
