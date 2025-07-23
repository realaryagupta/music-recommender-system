# spotify/hybrid_recommendations.py
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


class HybridRecommenderSystem:
    
    def __init__(self,  
                 recommendation_count: int, 
                 content_based_weight: float):
        
        self.recommendation_count = recommendation_count
        self.content_based_weight = content_based_weight
        self.collaborative_weight = 1 - content_based_weight
        
        
    def __calculate_content_based_similarities(self, song_name, artist_name, song_metadata, tfidf_matrix):
        # Filter out the song from data
        target_song = song_metadata.loc[(song_metadata["name"] == song_name) & 
                                      (song_metadata["artist"] == artist_name)]
        # Get the index of song
        song_idx = target_song.index[0]
        # Generate the input vector
        song_vector = tfidf_matrix[song_idx].reshape(1, -1)
        # Calculate similarity scores
        content_similarities = cosine_similarity(song_vector, tfidf_matrix)
        return content_similarities
        
    
    def __calculate_collaborative_similarities(self, song_name, artist_name, unique_track_ids, 
                                             song_metadata, user_interaction_matrix):
        # Fetch the row from songs data
        target_song = song_metadata.loc[(song_metadata["name"] == song_name) & 
                                      (song_metadata["artist"] == artist_name)]
        # Track ID of input song
        input_track_id = target_song['track_id'].values.item()
        # Index value of track_id
        track_idx = np.where(unique_track_ids == input_track_id)[0].item()
        # Fetch the input vector
        interaction_vector = user_interaction_matrix[track_idx]
        # Get similarity scores
        collaborative_similarities = cosine_similarity(interaction_vector, user_interaction_matrix)
        return collaborative_similarities
    
    
    def __normalize_scores(self, similarity_scores):
        min_score = np.min(similarity_scores)
        max_score = np.max(similarity_scores)
        normalized_scores = (similarity_scores - min_score) / (max_score - min_score)
        return normalized_scores
    
    
    def __align_scores(self, content_scores, collaborative_scores, song_metadata, unique_track_ids):
        """Align the scores to match the same set of songs"""
        # Get all track IDs from song_metadata
        all_track_ids = song_metadata['track_id'].values
        
        # Create empty arrays for aligned scores
        aligned_content = np.zeros(len(all_track_ids))
        aligned_collab = np.zeros(len(all_track_ids))
        
        # Find indices where track IDs match between content-based and full dataset
        content_indices = np.arange(len(all_track_ids))  # Content-based uses full dataset
        
        # Find indices where track IDs match between collaborative and full dataset
        _, collab_indices, _ = np.intersect1d(all_track_ids, unique_track_ids, return_indices=True)
        
        # Fill the aligned arrays
        aligned_content[content_indices] = content_scores.ravel()[:len(content_indices)]
        aligned_collab[collab_indices] = collaborative_scores.ravel()[:len(collab_indices)]
        
        return aligned_content, aligned_collab
    
    
    def __combine_scores(self, content_scores, collaborative_scores):
        hybrid_scores = (self.content_based_weight * content_scores) + \
                       (self.collaborative_weight * collaborative_scores)
        return hybrid_scores
    
    
    def recommend_songs(self, song_name, artist_name, song_metadata, unique_track_ids, 
                       tfidf_matrix, user_interaction_matrix):
        # Calculate content-based similarities
        content_similarities = self.__calculate_content_based_similarities(
            song_name=song_name, 
            artist_name=artist_name, 
            song_metadata=song_metadata, 
            tfidf_matrix=tfidf_matrix
        )
        
        # Calculate collaborative filtering similarities
        collaborative_similarities = self.__calculate_collaborative_similarities(
            song_name=song_name, 
            artist_name=artist_name, 
            unique_track_ids=unique_track_ids, 
            song_metadata=song_metadata, 
            user_interaction_matrix=user_interaction_matrix
        )
    
        # Normalize scores
        normalized_content_scores = self.__normalize_scores(content_similarities)
        normalized_collaborative_scores = self.__normalize_scores(collaborative_similarities)
        
        # Align the scores to the same set of songs
        aligned_content, aligned_collab = self.__align_scores(
            normalized_content_scores,
            normalized_collaborative_scores,
            song_metadata,
            unique_track_ids
        )
        
        # Combine scores with weighting
        hybrid_scores = self.__combine_scores(
            content_scores=aligned_content, 
            collaborative_scores=aligned_collab
        )
        
        # Get indices of top recommendations (excluding the query song itself)
        top_indices = np.argsort(hybrid_scores)[-self.recommendation_count-1:-1][::-1]
        
        # Get recommended track IDs
        recommended_track_ids = song_metadata.iloc[top_indices]['track_id'].values
        
        # Prepare results DataFrame
        score_dataframe = pd.DataFrame({
            "track_id": recommended_track_ids,
            "score": hybrid_scores[top_indices]
        })
        
        recommended_songs = (
            song_metadata
            .loc[song_metadata["track_id"].isin(recommended_track_ids)]
            .merge(score_dataframe, on="track_id")
            .sort_values(by="score", ascending=False)
            .drop(columns=["track_id", "score"])
            .reset_index(drop=True)
        )
        
        return recommended_songs