import streamlit as st
from content_filtering import content_recommendation
from scipy.sparse import load_npz
import pandas as pd
from numpy import load
import time

# Page config
st.set_page_config(
    page_title="🎵 Spotify Song Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s ease infinite;
        margin-bottom: 0.5rem;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 300;
    }
    
    /* Input containers */
    .input-container {
        background: rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Custom input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        color: white;
        font-size: 1.1rem;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4ecdc4;
        box-shadow: 0 0 20px rgba(78, 205, 196, 0.3);
    }
    
    .stSelectbox > div > div > select {
        background: rgba(255, 255, 255, 0.1);
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        color: white;
        font-size: 1.1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        border: none;
        border-radius: 25px;
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
    }
    
    /* Recommendation cards */
    .recommendation-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .recommendation-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.2);
    }
    
    .current-playing {
        background: linear-gradient(45deg, rgba(255, 107, 107, 0.2), rgba(78, 205, 196, 0.2));
        border: 2px solid rgba(78, 205, 196, 0.5);
    }
    
    .next-up {
        background: linear-gradient(45deg, rgba(69, 183, 209, 0.2), rgba(150, 206, 180, 0.2));
        border: 2px solid rgba(69, 183, 209, 0.5);
    }
    
    /* Loading animation */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .loading-spinner {
        width: 40px;
        height: 40px;
        border: 4px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top-color: #4ecdc4;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Error message styling */
    .error-message {
        background: rgba(255, 107, 107, 0.2);
        border: 2px solid rgba(255, 107, 107, 0.5);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        color: #ff6b6b;
        font-weight: 500;
    }
    
    /* Suggestion styling */
    .suggestion-container {
        background: rgba(78, 205, 196, 0.1);
        border: 2px solid rgba(78, 205, 196, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        max-height: 200px;
        overflow-y: auto;
    }
    
    .suggestion-item {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.3rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .suggestion-item:hover {
        background: rgba(78, 205, 196, 0.3);
        transform: translateX(5px);
    }
    
    .suggestion-header {
        color: #4ecdc4;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Load the data
@st.cache_data
def load_data():
    cleaned_data_path = "../data/processed/cleaned_data.csv"
    songs_data = pd.read_csv(cleaned_data_path)
    
    transformed_data_path = "../models/transformed_data.npz"
    transformed_data = load_npz(transformed_data_path)
    
    return songs_data, transformed_data

# Function to get artist suggestions
def get_artist_suggestions(partial_name, songs_data, limit=10):
    if not partial_name or len(partial_name) < 2:
        return []
    
    partial_name = partial_name.lower().strip()
    matching_artists = songs_data[songs_data['artist'].str.lower().str.contains(partial_name, na=False, regex=False)]['artist'].unique()
    
    # Sort by relevance (exact starts first, then contains)
    exact_starts = [artist for artist in matching_artists if artist.lower().startswith(partial_name)]
    contains = [artist for artist in matching_artists if not artist.lower().startswith(partial_name)]
    
    # Combine and limit results
    all_matches = exact_starts + contains
    return list(set(all_matches))[:limit]

# Function to get song suggestions for a specific artist
def get_song_suggestions(partial_song, artist_name, songs_data, limit=10):
    if not partial_song or len(partial_song) < 2:
        return []
    
    partial_song = partial_song.lower().strip()
    artist_name = artist_name.lower().strip()
    
    # Filter songs by artist first, then by song name
    artist_songs = songs_data[songs_data['artist'].str.lower() == artist_name]
    matching_songs = artist_songs[artist_songs['name'].str.lower().str.contains(partial_song, na=False, regex=False)]['name'].unique()
    
    # Sort by relevance
    exact_starts = [song for song in matching_songs if song.lower().startswith(partial_song)]
    contains = [song for song in matching_songs if not song.lower().startswith(partial_song)]
    
    all_matches = exact_starts + contains
    return list(set(all_matches))[:limit]

# Load data with caching
songs_data, transformed_data = load_data()

# Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🎵 Spotify Recommender 🎵</h1>
    <p class="subtitle">Discover your next favorite song with AI-powered recommendations</p>
</div>
""", unsafe_allow_html=True)

# Create columns for better layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    # Song input
    st.markdown("### 🎶 Song Details")
    song_name = st.text_input('🎵 Enter song name:', placeholder="e.g., Bohemian Rhapsody")
    
    # Artist input
    artist_name = st.text_input('🎤 Enter artist name:', placeholder="e.g., Queen")
    
    # Show artist suggestions if partial input is provided
    if artist_name and len(artist_name.strip()) >= 2:
        artist_suggestions = get_artist_suggestions(artist_name, songs_data)
        
        if artist_suggestions and artist_name.lower().strip() not in [artist.lower() for artist in artist_suggestions]:
            st.markdown(f"""
            <div class="suggestion-container">
                <div class="suggestion-header">🎤 Did you mean one of these artists?</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create columns for suggestions (2 per row)
            suggestion_cols = st.columns(2)
            for i, suggestion in enumerate(artist_suggestions[:8]):  # Limit to 8 suggestions
                col_idx = i % 2
                with suggestion_cols[col_idx]:
                    if st.button(f"🎤 {suggestion.title()}", key=f"artist_sugg_{i}", help=f"Click to select {suggestion}"):
                        st.session_state.selected_artist = suggestion
                        st.rerun()
    
    # Update artist name if suggestion was selected
    if 'selected_artist' in st.session_state:
        artist_name = st.session_state.selected_artist
        # Clear the selection after use
        del st.session_state.selected_artist
    
    # Show song suggestions if both artist and partial song name are provided
    if artist_name and song_name and len(song_name.strip()) >= 2:
        # Check if artist exists exactly
        if artist_name.lower().strip() in [artist.lower() for artist in songs_data['artist'].unique()]:
            song_suggestions = get_song_suggestions(song_name, artist_name, songs_data)
            
            if song_suggestions and song_name.lower().strip() not in [song.lower() for song in song_suggestions]:
                st.markdown(f"""
                <div class="suggestion-container">
                    <div class="suggestion-header">🎵 Songs by {artist_name.title()}:</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Create columns for song suggestions
                song_cols = st.columns(2)
                for i, suggestion in enumerate(song_suggestions[:6]):  # Limit to 6 songs
                    col_idx = i % 2
                    with song_cols[col_idx]:
                        if st.button(f"🎵 {suggestion.title()}", key=f"song_sugg_{i}", help=f"Click to select {suggestion}"):
                            st.session_state.selected_song = suggestion
                            st.rerun()
    
    # Update song name if suggestion was selected
    if 'selected_song' in st.session_state:
        song_name = st.session_state.selected_song
        # Clear the selection after use
        del st.session_state.selected_song
    
    # Number of recommendations
    k = st.selectbox('🔢 How many recommendations?', [5, 10, 15, 20], index=1)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process inputs
    if song_name and artist_name:
        song_name_lower = song_name.lower()
        artist_name_lower = artist_name.lower()
        
        # Search button
        if st.button('🚀 Get Recommendations'):
            # Check if song exists
            if ((songs_data["name"] == song_name_lower) & (songs_data['artist'] == artist_name_lower)).any():
                
                # Loading animation
                with st.spinner('🎵 Finding your perfect playlist...'):
                    time.sleep(1)  # Simulate processing time
                    
                    recommendations = content_recommendation(
                        song_name=song_name_lower,
                        artist_name=artist_name_lower,
                        songs_data=songs_data,
                        transformed_data=transformed_data,
                        k=k
                    )
                
                st.success(f'🎉 Found {len(recommendations)} amazing recommendations!')
                
                # Display recommendations
                for ind, recommendation in recommendations.iterrows():
                    rec_song_name = recommendation['name'].title()
                    rec_artist_name = recommendation['artist'].title()
                    
                    if ind == 0:
                        st.markdown(f"""
                        <div class="recommendation-card current-playing">
                            <h2 style="color: #4ecdc4; margin-bottom: 0.5rem;">🎵 Currently Playing</h2>
                            <h3 style="color: white; margin-bottom: 1rem;">{rec_song_name} by {rec_artist_name}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if recommendation['spotify_preview_url']:
                            st.audio(recommendation['spotify_preview_url'])
                        
                    elif ind == 1:
                        st.markdown(f"""
                        <div class="recommendation-card next-up">
                            <h3 style="color: #45b7d1; margin-bottom: 0.5rem;">⏭️ Next Up</h3>
                            <h4 style="color: white; margin-bottom: 1rem;">{ind}. {rec_song_name} by {rec_artist_name}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if recommendation['spotify_preview_url']:
                            st.audio(recommendation['spotify_preview_url'])
                            
                    else:
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4 style="color: white; margin-bottom: 1rem;">{ind}. {rec_song_name} by {rec_artist_name}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if recommendation['spotify_preview_url']:
                            st.audio(recommendation['spotify_preview_url'])
                
            else:
                st.markdown(f"""
                <div class="error-message">
                    <h3>😔 Song Not Found</h3>
                    <p>Sorry, we couldn't find "{song_name}" by "{artist_name}" in our database.</p>
                    <p>Please try another song or check the spelling!</p>
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; color: rgba(255, 255, 255, 0.6);">
    <p>🎵 Powered by AI • Made with ❤️ • Discover More Music 🎵</p>
</div>
""", unsafe_allow_html=True)