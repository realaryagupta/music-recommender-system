import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import load_npz, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random
from pathlib import Path

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="🎵 Spotify Song Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- Custom CSS ----------
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
    
    /* Additional styles for analytics */
    .analytics-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .similarity-score {
        background: linear-gradient(45deg, #ff6b6b, #ffd93d);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 500;
        font-size: 0.9rem;
        display: inline-block;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Data Loading Functions ----------
@st.cache_data
def load_filtered_songs():
    """Load the filtered songs dataset."""
    try:
        return pd.read_csv("../data/processed/collab_filtered_data.csv")
    except FileNotFoundError:
        st.error("❌ Filtered songs data not found. Please run the preprocessing pipeline first.")
        return pd.DataFrame()

@st.cache_data
def load_track_ids():
    """Load track IDs."""
    try:
        return np.load("../models/track_ids.npy", allow_pickle=True)
    except FileNotFoundError:
        st.error("❌ Track IDs not found. Please run the preprocessing pipeline first.")
        return np.array([])

@st.cache_data
def load_interaction_matrix():
    """Load the interaction matrix."""
    try:
        return load_npz("../models/interaction_matrix.npz")
    except FileNotFoundError:
        st.error("❌ Interaction matrix not found. Please run the preprocessing pipeline first.")
        return csr_matrix((0, 0))

# ---------- Recommendation Function ----------
def collaborative_recommendation(song_name, artist_name, track_ids, songs_df, interaction_matrix, k=10):
    """Generate collaborative filtering recommendations."""
    # Find the song in the dataset
    song_row = songs_df[
        (songs_df["name"].str.lower() == song_name.lower()) &
        (songs_df["artist"].str.lower() == artist_name.lower())
    ]
    
    if song_row.empty:
        raise ValueError("Song not found in dataset.")
    
    input_track_id = song_row['track_id'].values[0]
    
    # Find song index in track_ids
    song_indices = np.where(track_ids == input_track_id)[0]
    if len(song_indices) == 0:
        raise ValueError("Song not found in interaction matrix.")
    
    song_index = song_indices[0]
    
    # Get similarity scores
    input_vector = interaction_matrix[song_index]
    similarity = cosine_similarity(input_vector, interaction_matrix).ravel()
    
    # Get top recommendations (excluding the input song)
    top_indices = np.argsort(similarity)[-(k+1):][::-1]
    top_indices = top_indices[top_indices != song_index][:k]  # Remove input song
    
    top_scores = similarity[top_indices]
    top_track_ids = track_ids[top_indices]
    
    # Create recommendations DataFrame
    scores_df = pd.DataFrame({"track_id": top_track_ids, "score": top_scores})
    
    recommended_songs = (
        songs_df[songs_df["track_id"].isin(top_track_ids)]
        .merge(scores_df, on="track_id")
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    
    return recommended_songs

# ---------- Visualization Functions ----------
def create_similarity_chart(recommendations_df):
    """Create a polished similarity score chart with better formatting."""
    # Prepare data
    recommendations_df = recommendations_df.sort_values('score', ascending=True)
    max_score = recommendations_df['score'].max()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=recommendations_df['score'],
        y=[f"{row['name']} • {row['artist']}" for _, row in recommendations_df.iterrows()],
        orientation='h',
        marker=dict(
            color=recommendations_df['score'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Score",
                thickness=20,
                tickvals=[0, max_score/2, max_score],
                ticktext=["0%", f"{(max_score/2)*100:.0f}%", f"{max_score*100:.0f}%"]
            ),
            line=dict(width=0)  # Remove bar borders
        ),
        text=[f"{score*100:.1f}%" for score in recommendations_df['score']],
        textposition='outside',
        textfont=dict(
            color='white',
            size=12
        ),
        hoverinfo='text',
        hovertext=[f"<b>{row['name']}</b><br>Artist: {row['artist']}<br>Similarity: {row['score']*100:.1f}%" 
                  for _, row in recommendations_df.iterrows()],
        cliponaxis=False  # Allow text to appear outside plot area
    ))
    
    fig.update_layout(
        title={
            'text': "<b>Song Similarity Scores</b>",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18, color='white')
        },
        xaxis=dict(
            title="<b>Similarity Score</b>",
            titlefont=dict(color='white'),
            tickfont=dict(color='white'),
            gridcolor='rgba(255,255,255,0.1)',
            range=[0, max_score * 1.1],  # Add 10% padding
            tickformat=".0%",
            zerolinecolor='rgba(255,255,255,0.3)'
        ),
        yaxis=dict(
            title=None,
            tickfont=dict(color='white', size=12),
            automargin=True,  # Prevent label cutoff
            autorange=True
        ),
        height=max(400, len(recommendations_df) * 40),  # Dynamic height based on number of recommendations
        margin=dict(l=20, r=20, t=80, b=20, pad=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor='rgba(0,0,0,0.8)',
            font_size=14,
            font_color='white'
        ),
        uniformtext=dict(
            minsize=10,
            mode='show'
        )
    )
    
    # Add custom grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)')
    
    return fig


def create_artist_distribution(recommendations_df):
    """Create an enhanced artist distribution pie chart."""
    artist_counts = recommendations_df['artist'].value_counts().reset_index()
    artist_counts.columns = ['artist', 'count']
    
    # Create a color sequence based on count
    colors = px.colors.qualitative.Pastel + px.colors.qualitative.Vivid
    
    fig = go.Figure()
    
    fig.add_trace(go.Pie(
        labels=artist_counts['artist'],
        values=artist_counts['count'],
        hole=0.4,
        marker=dict(
            colors=colors,
            line=dict(color='rgba(255,255,255,0.2)', width=1)
        ),
        textinfo='percent+label',
        textposition='inside',
        textfont=dict(
            color='white',
            size=12
        ),
        hoverinfo='label+percent+value',
        hovertemplate="<b>%{label}</b><br>%{percent}<br>(%{value} songs)<extra></extra>",
        pull=[0.03 if i == 0 else 0 for i in range(len(artist_counts))]  # Slight pull on largest segment
    ))
    
    fig.update_layout(
        title={
            'text': "<b>Recommended Artists Distribution</b>",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18, color='white')
        },
        height=450,
        margin=dict(l=20, r=20, t=80, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(color='white')
        ),
        hoverlabel=dict(
            bgcolor='rgba(0,0,0,0.8)',
            font_size=14,
            font_color='white'
        ),
        uniformtext=dict(
            minsize=10,
            mode='hide'
        )
    )
    
    return fig

def create_artist_distribution(recommendations_df):
    """Create artist distribution pie chart."""
    artist_counts = recommendations_df['artist'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=artist_counts.index,
        values=artist_counts.values,
        hole=0.3,
        marker_colors=px.colors.qualitative.Set3
    )])
    
    fig.update_layout(
        title="Recommended Artists Distribution",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

# ---------- Main App ----------
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🎵 Spotify Recommender</h1>
        <p class="subtitle">Discover your next favorite song with AI-powered recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner('🎵 Loading music data...'):
        songs_df = load_filtered_songs()
        track_ids = load_track_ids()
        interaction_matrix = load_interaction_matrix()
    
    if songs_df.empty or len(track_ids) == 0 or interaction_matrix.shape[0] == 0:
        st.markdown("""
        <div class="error-message">
            <h3>😔 Data Not Available</h3>
            <p>Please ensure all required data files are available and run the preprocessing pipeline first.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Create columns for better layout
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        # Song input
        st.markdown("### 🎶 Song Details")
        song_input = st.text_input('🎵 Enter song name:', placeholder="e.g., Bohemian Rhapsody")
        
        # Artist input
        artist_input = st.text_input('🎤 Enter artist name:', placeholder="e.g., Queen")
        
        # Auto-suggestion functions
        def get_song_suggestions(query, limit=10):
            """Get song suggestions based on partial input."""
            if not query:
                return []
            query_lower = query.lower()
            suggestions = [song for song in songs_df['name'].unique() 
                         if query_lower in song.lower()]
            return sorted(suggestions)[:limit]
        
        def get_artist_suggestions(query, limit=10):
            """Get artist suggestions based on partial input."""
            if not query:
                return []
            query_lower = query.lower()
            suggestions = [artist for artist in songs_df['artist'].unique() 
                         if query_lower in artist.lower()]
            return sorted(suggestions)[:limit]
        
        # Show artist suggestions if partial input is provided
        if artist_input and len(artist_input.strip()) >= 2:
            artist_suggestions = get_artist_suggestions(artist_input)
            
            if artist_suggestions and artist_input.lower().strip() not in [a.lower() for a in artist_suggestions]:
                with st.container():
                    st.markdown("""
                    <div class="suggestion-container">
                        <div class="suggestion-header">🎤 Did you mean one of these artists?</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create columns for suggestions (2 per row)
                    cols = st.columns(2)
                    for i, suggestion in enumerate(artist_suggestions[:8]):
                        with cols[i % 2]:
                            if st.button(f"🎤 {suggestion.title()}", 
                                         key=f"artist_sugg_{i}",
                                         help=f"Select {suggestion}"):
                                st.session_state.selected_artist = suggestion
                                st.rerun()
        
        # Update artist name if suggestion was selected
        if 'selected_artist' in st.session_state:
            artist_input = st.session_state.selected_artist
            del st.session_state.selected_artist
        
        # Show song suggestions if both artist and partial song name are provided
        if artist_input and song_input and len(song_input.strip()) >= 2:
            if any(artist_input.lower() == a.lower() for a in songs_df['artist'].unique()):
                song_suggestions = get_song_suggestions(song_input)
                
                if song_suggestions and song_input.lower().strip() not in [s.lower() for s in song_suggestions]:
                    with st.container():
                        st.markdown(f"""
                        <div class="suggestion-container">
                            <div class="suggestion-header">🎵 Songs by {artist_input.title()}:</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Create columns for song suggestions
                        cols = st.columns(2)
                        for i, suggestion in enumerate(song_suggestions[:6]):
                            with cols[i % 2]:
                                if st.button(f"🎵 {suggestion.title()}",
                                           key=f"song_sugg_{i}",
                                           help=f"Select {suggestion}"):
                                    st.session_state.selected_song = suggestion
                                    st.rerun()
        
        # Update song name if suggestion was selected
        if 'selected_song' in st.session_state:
            song_input = st.session_state.selected_song
            del st.session_state.selected_song
        
        # Number of recommendations
        k = st.selectbox('🔢 How many recommendations?', [5, 10, 15, 20], index=1)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process inputs
        if song_input and artist_input:
            song_exists = not songs_df[
                (songs_df["name"].str.lower() == song_input.lower()) & 
                (songs_df["artist"].str.lower() == artist_input.lower())
            ].empty
            
            if song_exists:
                if st.button('🚀 Get Recommendations'):
                    with st.spinner('🎵 Finding your perfect playlist...'):
                        time.sleep(1)
                        recommendations = collaborative_recommendation(
                            song_input, artist_input, track_ids, 
                            songs_df, interaction_matrix, k
                        )
                    
                    st.success(f'🎉 Found {len(recommendations)} recommendations!')
                    
                    # Current playing card
                    current_card = f"""
                    <div class="recommendation-card current-playing">
                        <h2 style="color: #4ecdc4; margin-bottom: 0.5rem;">🎵 Currently Playing</h2>
                        <h3 style="color: white; margin-bottom: 1rem;">
                            {song_input.title()} by {artist_input.title()}
                        </h3>
                    </div>
                    """
                    st.markdown(current_card, unsafe_allow_html=True)
                    
                    # Display recommendations
                    for idx, (_, song) in enumerate(recommendations.head(k).iterrows(), 1):
                        similarity = song['score'] * 100
                        card_class = "next-up" if idx == 1 else ""
                        
                        card = f"""
                        <div class="recommendation-card {card_class}">
                            <h3 style="color: #45b7d1; margin-bottom: 0.5rem;">
                                {'⏭️ Next Up' if idx == 1 else f'{idx}.'}
                            </h3>
                            <h4 style="color: white; margin-bottom: 1rem;">
                                {song['name'].title()} by {song['artist'].title()}
                            </h4>
                            <div class="similarity-score">
                                {similarity:.1f}% match
                            </div>
                        </div>
                        """
                        st.markdown(card, unsafe_allow_html=True)
                    


                    # # Analytics section
                    # with st.container():
                    #     st.markdown("""
                    #     <div class="analytics-container">
                    #         <h2>📊 Recommendation Analytics</h2>
                    #     </div>
                    #     """, unsafe_allow_html=True)
                        
                    #     cols = st.columns(2)
                    #     with cols[0]:
                    #         fig1 = create_similarity_chart(recommendations.head(10))
                    #         st.plotly_chart(fig1, use_container_width=True)
                        
                    #     with cols[1]:
                    #         if len(recommendations) > 1:
                    #             fig2 = create_artist_distribution(recommendations.head(10))
                    #             st.plotly_chart(fig2, use_container_width=True)

                    # Analytics section
                    with st.container():
                        st.markdown("""
                        <div class="analytics-container">
                            <h2>📊 Recommendation Analytics</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display similarity chart first
                        fig1 = create_similarity_chart(recommendations.head(10))
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        # Display artist distribution chart below (only if we have multiple recommendations)
                        if len(recommendations) > 1:
                            fig2 = create_artist_distribution(recommendations.head(10))
                            st.plotly_chart(fig2, use_container_width=True)

                    # Download button
                    csv = recommendations.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Recommendations",
                        data=csv,
                        file_name=f"recommendations_{song_input.replace(' ', '_')}_{artist_input.replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
            else:
                error_msg = f"""
                <div class="error-message">
                    <h3>😔 Song Not Found</h3>
                    <p>Sorry, we couldn't find "{song_input}" by "{artist_input}"</p>
                    <p>Please try another song or check the spelling!</p>
                </div>
                """
                st.markdown(error_msg, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: rgba(255, 255, 255, 0.6);">
        <p>🎵 Powered by AI • Made with ❤️ • Discover More Music 🎵</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()