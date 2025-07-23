# backend/main.py
import pandas as pd
import numpy as np
from scipy.sparse import load_npz, csr_matrix
import joblib
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import difflib
import sys # Added for path debugging

# Import individual recommendation systems
# Assuming these are correctly imported and available in your environment
# from .hybrid_recommendations import HybridRecommenderSystem
# from .collaborative_filtering import collaborative_recommendation
# from .content_filtering import get_top_k_recommendations

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

# --- Define Paths ---
# Calculate PROJECT_ROOT dynamically, assuming main.py is in 'backend' subfolder
# It will go up one level from backend/: musiccccc/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- IMPORTANT DEBUGGING LOGS ---
logging.info(f"__file__: {Path(__file__).resolve()}")
logging.info(f"Calculated PROJECT_ROOT: {PROJECT_ROOT}")
logging.info(f"Current Working Directory (CWD): {Path.cwd()}")
# --- END IMPORTANT DEBUGGING LOGS ---

PATHS = {
    "cleaned_data": PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv",
    "collab_track_ids": PROJECT_ROOT / "models" / "track_ids.npy",
    "collab_matrix": PROJECT_ROOT / "models" / "interaction_matrix.npz",
    "content_transformer": PROJECT_ROOT / "models" / "transformer.joblib",
    "content_matrix": PROJECT_ROOT / "models" / "transformed_data.npz",
    "filtered_songs": PROJECT_ROOT / "data" / "processed" / "collab_filtered_data.csv"
}

# --- FastAPI App Initialization ---
app = FastAPI()

# --- CORS Middleware ---
# For debugging, allow all origins.
# In production, replace "*" with specific origins like your Vercel URLs.
origins = ["*"] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"], # Allow all headers
)

# --- Global variables for loaded models and data ---
songs_df = None
track_ids = None
interaction_matrix = None
transformer = None
tfidf_matrix = None
filtered_songs_df = None

@app.on_event("startup")
async def load_models_and_data():
    global songs_df, track_ids, interaction_matrix, transformer, tfidf_matrix, filtered_songs_df
    logging.info("Starting model and data loading process...")
    try:
        # --- ENHANCED PATH VALIDATION AND LOADING ---
        # Ensure all paths exist before attempting to load
        for key, path_obj in PATHS.items():
            if not path_obj.exists():
                logging.critical(f"CRITICAL ERROR: Required file not found for '{key}': {path_obj}")
                # Raise an exception to prevent the app from starting with missing data
                raise FileNotFoundError(f"Missing required data/model file: {path_obj}. Please run data preprocessing scripts.")

        songs_df = pd.read_csv(PATHS["cleaned_data"])
        logging.info(f"Loaded songs_df with {len(songs_df)} entries from {PATHS['cleaned_data']}")

        track_ids = np.load(PATHS["collab_track_ids"], allow_pickle=True)
        logging.info(f"Loaded track_ids with {len(track_ids)} entries from {PATHS['collab_track_ids']}")

        interaction_matrix = load_npz(PATHS["collab_matrix"])
        logging.info(f"Loaded interaction_matrix with shape {interaction_matrix.shape} from {PATHS['collab_matrix']}")

        transformer = joblib.load(PATHS["content_transformer"])
        logging.info(f"Loaded transformer from {PATHS['content_transformer']}")

        tfidf_matrix = load_npz(PATHS["content_matrix"])
        logging.info(f"Loaded tfidf_matrix with shape {tfidf_matrix.shape} from {PATHS['content_matrix']}")

        filtered_songs_df = pd.read_csv(PATHS["filtered_songs"])
        logging.info(f"Loaded filtered_songs_df with {len(filtered_songs_df)} entries from {PATHS['filtered_songs']}")

        logging.info("All models and data loaded successfully.")
    except Exception as e:
        logging.error(f"Error during model or data loading: {e}", exc_info=True)
        # Re-raise the exception to ensure FastAPI startup fails if data is missing
        raise RuntimeError(f"Application failed to start due to data/model loading error: {e}") from e


# --- Helper for Fuzzy Matching ---
def find_closest_song_match(query_song_name: str, query_artist_name: str, df: pd.DataFrame, n: int = 3, cutoff: float = 0.6):
    """
    Finds the closest song and artist matches in the DataFrame.
    Returns the exact match (if found), or a list of suggestions.
    """
    # Lowercase for robust matching
    query_song_name_lower = query_song_name.lower()
    query_artist_name_lower = query_artist_name.lower()

    # Try exact match first
    exact_match = df[
        (df['name'].str.lower() == query_song_name_lower) &
        (df['artist'].str.lower() == query_artist_name_lower)
    ]
    if not exact_match.empty:
        return {'status': 'found', 'song_name': exact_match['name'].iloc[0], 'artist_name': exact_match['artist'].iloc[0]}

    # If no exact match, try fuzzy matching
    all_songs = df['name'].str.lower().tolist()
    all_artists = df['artist'].str.lower().tolist()

    closest_songs = difflib.get_close_matches(query_song_name_lower, all_songs, n=n, cutoff=cutoff)
    closest_artists = difflib.get_close_matches(query_artist_name_lower, all_artists, n=n, cutoff=cutoff)

    suggestions = []
    # Combine song and artist suggestions to find plausible pairs
    for s_name in closest_songs:
        # Find artists associated with this suggested song name
        artists_for_s_name = df[df['name'].str.lower() == s_name]['artist'].str.lower().unique().tolist()
        for a_name in artists_for_s_name:
            # Check if the artist is also a close match to the query artist
            if difflib.SequenceMatcher(None, query_artist_name_lower, a_name).ratio() >= cutoff:
                original_song_name = df[(df['name'].str.lower() == s_name) & (df['artist'].str.lower() == a_name)]['name'].iloc[0]
                original_artist_name = df[(df['name'].str.lower() == s_name) & (df['artist'].str.lower() == a_name)]['artist'].iloc[0]
                suggestions.append({'song_name': original_song_name, 'artist_name': original_artist_name})
    
    # Remove duplicates from suggestions (based on song_name, artist_name pair)
    seen_suggestions = set()
    unique_suggestions = []
    for s in suggestions:
        item = (s['song_name'], s['artist']['name']) # Corrected 'artist' to 'artist_name' if needed based on your df structure
        if item not in seen_suggestions:
            unique_suggestions.append(s)
            seen_suggestions.add(item)

    if unique_suggestions:
        return {'status': 'suggestions', 'suggestions': unique_suggestions[:n]} # Limit to top N suggestions
    else:
        return {'status': 'not_found'}


# --- Pydantic Models for Request and Response ---
class RecommendationRequest(BaseModel):
    song_name: str
    artist_name: str
    recommendation_count: int = 5
    content_based_weight: float = 0.5

class SongRecommendation(BaseModel):
    name: str
    artist: str
    spotify_preview_url: str | None = None
    year: int | None = None
    duration_ms: int | None = None
    album: str | None = None
    tags: str | None = None

class RecommendationResponse(BaseModel):
    query_song: str
    query_artist: str
    found_match: bool
    suggested_matches: list[dict[str, str]] | None = None
    
    hybrid_recommendations: list[SongRecommendation] | None = None
    content_based_recommendations: list[SongRecommendation] | None = None
    collaborative_recommendations: list[SongRecommendation] | None = None

# --- API Endpoint ---

# Explicitly handle OPTIONS for /recommend
@app.options("/recommend")
async def options_recommend():
    """
    Handles OPTIONS preflight requests for the /recommend endpoint.
    CORSMiddleware should handle this, but adding it explicitly for debugging.
    """
    logging.info("Explicit OPTIONS /recommend endpoint hit!")
    return Response(status_code=200)


@app.post("/recommend", response_model=RecommendationResponse)
async def get_all_recommendations(request: RecommendationRequest):
    """
    Generates hybrid, content-based, and collaborative song recommendations.
    Includes fuzzy matching for input song/artist.
    """
    # Defensive check: ensure models are loaded. This helps catch if startup failed.
    if songs_df is None or track_ids is None or interaction_matrix is None or transformer is None or tfidf_matrix is None or filtered_songs_df is None:
        logging.error("One or more recommendation models/data are not loaded. Rejecting request.")
        raise HTTPException(status_code=503, detail="Recommendation system is still loading or failed to load. Please try again later.")

    # 1. Fuzzy Matching / Song Lookup
    match_result = find_closest_song_match(request.song_name, request.artist_name, songs_df)

    if match_result['status'] == 'not_found':
        logging.info(f"Query '{request.song_name}' by '{request.artist_name}' not found. No suggestions.")
        return RecommendationResponse(
            query_song=request.song_name,
            query_artist=request.artist_name,
            found_match=False,
            suggested_matches=[]
        )
    elif match_result['status'] == 'suggestions':
        logging.info(f"Query '{request.song_name}' by '{request.artist_name}' needs clarification. Suggestions: {match_result['suggestions']}")
        return RecommendationResponse(
            query_song=request.song_name,
            query_artist=request.artist_name,
            found_match=False,
            suggested_matches=match_result['suggestions']
        )
    
    # Use the found/corrected song and artist for recommendations
    found_song_name = match_result['song_name']
    found_artist_name = match_result['artist_name']
    logging.info(f"Matched song: '{found_song_name}' by '{found_artist_name}'. Generating recommendations.")

    # Initialize recommendation lists
    hybrid_recs_list = []
    content_recs_list = []
    collab_recs_list = []

    try:
        # Import individual recommendation systems locally within the function or globally if needed
        from backend.hybrid_recommendations import HybridRecommenderSystem
        from backend.collaborative_filtering import collaborative_recommendation
        from backend.content_filtering import get_top_k_recommendations

        # 2. Get Hybrid Recommendations
        hybrid_recommender = HybridRecommenderSystem(
            recommendation_count=request.recommendation_count,
            content_based_weight=request.content_based_weight
        )
        recommended_hybrid_df = hybrid_recommender.recommend_songs(
            song_name=found_song_name,
            artist_name=found_artist_name,
            song_metadata=songs_df, # Pass the full songs_df to Hybrid system
            unique_track_ids=track_ids, # Track IDs relevant to collaborative matrix
            user_interaction_matrix=interaction_matrix,
            tfidf_matrix=tfidf_matrix,
        )
        for _, row in recommended_hybrid_df.iterrows():
            hybrid_recs_list.append(
                SongRecommendation(
                    name=row["name"], artist=row["artist"],
                    spotify_preview_url=row.get("spotify_preview_url"),
                    year=int(row["year"]) if pd.notna(row.get("year")) else None,
                    duration_ms=int(row["duration_ms"]) if pd.notna(row.get("duration_ms")) else None,
                    album=row.get("album"), tags=row.get("tags")
                )
            )
        logging.info(f"Generated {len(hybrid_recs_list)} hybrid recommendations.")

    except ValueError as e:
        logging.warning(f"Hybrid recommendation ValueError: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during hybrid recommendation: {e}", exc_info=True)

    try:
        # 3. Get Content-Based Recommendations
        recommended_content_df = get_top_k_recommendations(
            query_name=found_song_name,
            query_artist=found_artist_name,
            raw_df=songs_df, # Pass the full songs_df
            features_matrix=tfidf_matrix,
            top_k=request.recommendation_count
        )
        for _, row in recommended_content_df.iterrows():
            content_recs_list.append(
                SongRecommendation(
                    name=row["name"], artist=row["artist"],
                    spotify_preview_url=row.get("spotify_preview_url"),
                    year=int(row["year"]) if pd.notna(row.get("year")) else None,
                    duration_ms=int(row["duration_ms"]) if pd.notna(row.get("duration_ms")) else None,
                    album=row.get("album"), tags=row.get("tags")
                )
            )
        logging.info(f"Generated {len(content_recs_list)} content-based recommendations.")

    except ValueError as e:
        logging.warning(f"Content-based recommendation ValueError: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during content-based recommendation: {e}", exc_info=True)

    try:
        # 4. Get Collaborative Recommendations
        recommended_collab_df = collaborative_recommendation(
            song_name=found_song_name,
            artist_name=found_artist_name,
            track_ids=track_ids, # Track IDs from the collaborative model
            songs_df=filtered_songs_df, # Use the filtered_songs_df for collaborative model
            interaction_matrix=interaction_matrix,
            k=request.recommendation_count
        )
        for _, row in recommended_collab_df.iterrows():
            collab_recs_list.append(
                SongRecommendation(
                    name=row["name"], artist=row["artist"],
                    spotify_preview_url=row.get("spotify_preview_url"),
                    year=int(row["year"]) if pd.notna(row.get("year")) else None,
                    duration_ms=int(row["duration_ms"]) if pd.notna(row.get("duration_ms")) else None,
                    album=row.get("album"), tags=row.get("tags")
                )
            )
        logging.info(f"Generated {len(collab_recs_list)} collaborative recommendations.")

    except ValueError as e:
        logging.warning(f"Collaborative recommendation ValueError: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during collaborative recommendation: {e}", exc_info=True)

    return RecommendationResponse(
        query_song=found_song_name,
        query_artist=found_artist_name,
        found_match=True,
        suggested_matches=None,
        hybrid_recommendations=hybrid_recs_list,
        content_based_recommendations=content_recs_list,
        collaborative_recommendations=collab_recs_list
    )
