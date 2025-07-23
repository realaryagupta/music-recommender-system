# spotify/app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import load_npz, csr_matrix
import joblib
import logging
from pathlib import Path
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the recommendation systems
from spotify.collaborative_filtering import collaborative_recommendation
from spotify.content_filtering import get_top_k_recommendations
from spotify.hybrid_recommendations import HybridRecommenderSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# File paths
PATHS = {
    "cleaned_data": PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv",
    "collab_track_ids": PROJECT_ROOT / "models" / "track_ids.npy",
    "collab_matrix": PROJECT_ROOT / "models" / "interaction_matrix.npz",
    "content_transformer": PROJECT_ROOT / "models" / "transformer.joblib",
    "content_matrix": PROJECT_ROOT / "models" / "transformed_data.npz",
    "filtered_songs": PROJECT_ROOT / "data" / "processed" / "collab_filtered_data.csv"
}

# Load data and models (with caching)
@st.cache_data
def load_data():
    """Load all necessary data and models."""
    logging.info("Loading data and models...")
    
    try:
        # Load cleaned songs data
        songs_df = pd.read_csv(PATHS["cleaned_data"])
        
        # Load collaborative filtering components
        track_ids = np.load(PATHS["collab_track_ids"], allow_pickle=True)
        interaction_matrix = load_npz(PATHS["collab_matrix"])
        
        # Load content-based filtering components
        try:
            transformer = joblib.load(PATHS["content_transformer"])
            tfidf_matrix = load_npz(PATHS["content_matrix"])
        except FileNotFoundError:
            transformer = None
            tfidf_matrix = None
            logging.warning("Content-based filtering components not found")
        
        return {
            "songs_df": songs_df,
            "track_ids": track_ids,
            "interaction_matrix": interaction_matrix,
            "transformer": transformer,
            "tfidf_matrix": tfidf_matrix,
            "filtered_songs": pd.read_csv(PATHS["filtered_songs"]) if PATHS["filtered_songs"].exists() else None
        }
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        st.error("Failed to load required data files. Please check the data paths.")
        return None

def get_song_suggestions(query: str, songs_df: pd.DataFrame) -> list:
    """Get song name suggestions based on user input."""
    if not query or len(query) < 2:
        return []
    return songs_df[songs_df["name"].str.contains(query, case=False)]["name"].unique().tolist()[:20]

def get_artist_suggestions(query: str, songs_df: pd.DataFrame, song_name: str = None) -> list:
    """Get artist name suggestions based on user input and optional song name."""
    if not query or len(query) < 2:
        return []
    
    filtered = songs_df
    if song_name:
        filtered = filtered[filtered["name"].str.lower() == song_name.lower()]
    
    return filtered[filtered["artist"].str.contains(query, case=False)]["artist"].unique().tolist()[:20]

def display_song_card(row):
    """Display a song card with information."""
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{row['name']}**")
            st.markdown(f"*{row['artist']}*")
            
            # Display metadata
            metadata_cols = st.columns(3)
            with metadata_cols[0]:
                st.caption(f"🎵 {int(row['year'])}" if pd.notna(row['year']) else "🎵 Year N/A")
            with metadata_cols[1]:
                st.caption(f"⏱️ {row['duration_ms']//1000}s" if pd.notna(row['duration_ms']) else "⏱️ Duration N/A")
            with metadata_cols[2]:
                st.caption(f"💿 {row['album']}" if pd.notna(row['album']) else "💿 Album N/A")
            
            if pd.notna(row['tags']):
                st.caption(f"🏷️ {row['tags']}")
        
        with col2:
            if pd.notna(row['spotify_preview_url']):
                st.audio(row['spotify_preview_url'], format="audio/mp3")
            else:
                st.caption("No preview available")

def main():
    st.set_page_config(
        page_title="Spotify Recommendation System",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .st-emotion-cache-1v0mbdj {
        border-radius: 10px;
    }
    .st-emotion-cache-16idsys p {
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    if data is None:
        st.stop()
    
    songs_df = data["songs_df"]
    
    # Sidebar for controls
    with st.sidebar:
        st.title("⚙️ Recommendation Settings")
        
        st.subheader("Algorithm Settings")
        content_weight = st.slider(
            "Content-based weight",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Adjust the balance between content-based and collaborative filtering"
        )
        num_recommendations = st.slider(
            "Number of recommendations",
            min_value=1,
            max_value=20,
            value=10,
            step=1
        )
        
        st.subheader("About")
        st.markdown("""
        This hybrid recommendation system combines:
        - 🎼 Content-based filtering (song features)
        - 👥 Collaborative filtering (user listening patterns)
        """)
        st.markdown("---")
        st.caption("Built with ❤️ for Spotify data")
    
    # Main content
    st.title("🎵 Spotify Hybrid Recommendation System")
    st.write("Discover songs similar to your favorites using advanced recommendation algorithms!")
    
    # Search section
    with st.container(border=False):
        st.subheader("Find Similar Songs")
        col1, col2 = st.columns(2)
        
        with col1:
            song_query = st.text_input(
                "Song name",
                key="song_query",
                placeholder="Start typing a song name...",
                help="Type at least 2 characters to see suggestions"
            )
            
            # Show song suggestions
            song_suggestions = get_song_suggestions(song_query, songs_df)
            if song_suggestions:
                selected_song = st.selectbox(
                    "Select from suggestions",
                    options=song_suggestions,
                    key="selected_song",
                    index=None,
                    placeholder="Select a song..."
                )
            else:
                selected_song = None
        
        with col2:
            if selected_song:
                artist_query = st.text_input(
                    "Artist name",
                    key="artist_query",
                    placeholder=f"Artist for '{selected_song}'...",
                    help="Type at least 2 characters to see suggestions"
                )
                
                # Show artist suggestions
                artist_suggestions = get_artist_suggestions(artist_query, songs_df, selected_song)
                if artist_suggestions:
                    selected_artist = st.selectbox(
                        "Select from artists",
                        options=artist_suggestions,
                        key="selected_artist",
                        index=None,
                        placeholder="Select an artist..."
                    )
                else:
                    selected_artist = None
            else:
                artist_query = ""
                selected_artist = None
    
    # Recommendation button
    recommend_button = st.button(
        "Get Recommendations",
        disabled=not (selected_song and selected_artist),
        type="primary",
        use_container_width=True
    )
    
    # Display recommendations
    if recommend_button and selected_song and selected_artist:
        with st.spinner("🎧 Finding the perfect recommendations..."):
            try:
                # Initialize hybrid recommender
                hybrid_recommender = HybridRecommenderSystem(
                    recommendation_count=num_recommendations,
                    content_based_weight=content_weight
                )
                
                # Get hybrid recommendations
                hybrid_recs = hybrid_recommender.recommend_songs(
                    song_name=selected_song,
                    artist_name=selected_artist,
                    song_metadata=songs_df,
                    unique_track_ids=data["track_ids"],
                    tfidf_matrix=data["tfidf_matrix"],
                    user_interaction_matrix=data["interaction_matrix"]
                )
                
                # Display results
                st.subheader(f"🎯 Recommended songs similar to '{selected_song}' by {selected_artist}")
                st.caption(f"Using hybrid recommendation (Content-based: {content_weight*100:.0f}%, Collaborative: {(1-content_weight)*100:.0f}%)")
                
                # Show as cards
                for _, row in hybrid_recs.iterrows():
                    display_song_card(row)
                
                # Also show individual method results for comparison
                with st.expander("🔍 Compare with individual recommendation methods"):
                    tab1, tab2 = st.tabs(["🎼 Content-Based", "👥 Collaborative"])
                    
                    with tab1:
                        if data["tfidf_matrix"] is not None:
                            content_recs = get_top_k_recommendations(
                                query_name=selected_song,
                                query_artist=selected_artist,
                                raw_df=songs_df,
                                features_matrix=data["tfidf_matrix"],
                                top_k=num_recommendations
                            )
                            for _, row in content_recs.iterrows():
                                display_song_card(row)
                        else:
                            st.warning("Content-based filtering components not available")
                    
                    with tab2:
                        if data["filtered_songs"] is not None:
                            collab_recs = collaborative_recommendation(
                                song_name=selected_song,
                                artist_name=selected_artist,
                                track_ids=data["track_ids"],
                                songs_df=data["filtered_songs"],
                                interaction_matrix=data["interaction_matrix"],
                                k=num_recommendations
                            )
                            for _, row in collab_recs.iterrows():
                                display_song_card(row)
                        else:
                            st.warning("Collaborative filtering data not available")
            
            except ValueError as e:
                st.error(f"❌ Error: {str(e)}")
            except Exception as e:
                st.error("❌ An unexpected error occurred while generating recommendations.")
                logging.error(f"Recommendation error: {str(e)}")

if __name__ == "__main__":
    main()