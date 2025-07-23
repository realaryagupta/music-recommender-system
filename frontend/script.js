document.addEventListener('DOMContentLoaded', () => {
    const songNameInput = document.getElementById('songName');
    const artistNameInput = document.getElementById('artistName');
    const numRecommendationsSelect = document.getElementById('numRecommendations');
    const recommendBtn = document.getElementById('recommendBtn');
    const messageContainer = document.getElementById('messageContainer');
    const suggestionsSection = document.getElementById('suggestionsSection');
    const suggestionsList = document.getElementById('suggestionsList');
    const resultsSection = document.getElementById('resultsSection');

    const hybridRecommendationsList = document.getElementById('hybrid-recommendations');
    const contentBasedRecommendationsList = document.getElementById('content-based-recommendations');
    const collaborativeRecommendationsList = document.getElementById('collaborative-recommendations');

    // *** CRITICAL FIX: Changed from 'http://0.0.0.0:8000' to 'http://127.0.0.1:8000' ***
    const API_BASE_URL = 'http://127.0.0.1:8000'; // Make sure this matches your backend Uvicorn host/port

    // Function to validate inputs and enable/disable button
    function validateInputs() {
        const songName = songNameInput.value.trim();
        const artistName = artistNameInput.value.trim();
        if (songName && artistName) {
            recommendBtn.disabled = false;
            recommendBtn.classList.remove('disabled');
        } else {
            recommendBtn.disabled = true;
            recommendBtn.classList.add('disabled');
        }
    }

    // Add event listeners for input changes
    songNameInput.addEventListener('input', validateInputs);
    artistNameInput.addEventListener('input', validateInputs);

    // Add enter key support
    songNameInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !recommendBtn.disabled) {
            getRecommendations();
        }
    });

    artistNameInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !recommendBtn.disabled) {
            getRecommendations();
        }
    });

    function showMessage(type, message) {
        messageContainer.textContent = message;
        messageContainer.className = `message-container ${type}`;
        messageContainer.style.display = 'block';
        
        // Auto-hide success and info messages after 5 seconds
        if (type === 'success' || type === 'info') {
            setTimeout(() => {
                messageContainer.style.display = 'none';
            }, 5000);
        }
    }

    function clearMessages() {
        messageContainer.textContent = '';
        messageContainer.className = 'message-container';
        messageContainer.style.display = 'none';
    }

    function clearRecommendationsLists() {
        hybridRecommendationsList.innerHTML = '';
        contentBasedRecommendationsList.innerHTML = '';
        collaborativeRecommendationsList.innerHTML = '';
    }

    function hideSections() {
        suggestionsSection.style.display = 'none';
        resultsSection.style.display = 'none';
    }

    function showLoadingState() {
        recommendBtn.classList.add('loading');
        const btnText = recommendBtn.querySelector('.btn-text');
        const btnLoader = recommendBtn.querySelector('.btn-loader');
        const btnIcon = recommendBtn.querySelector('.btn-icon');
        
        if (btnText) btnText.style.display = 'none';
        if (btnIcon) btnIcon.style.display = 'none';
        if (btnLoader) btnLoader.style.display = 'inline-block';
    }

    function hideLoadingState() {
        recommendBtn.classList.remove('loading');
        const btnText = recommendBtn.querySelector('.btn-text');
        const btnLoader = recommendBtn.querySelector('.btn-loader');
        const btnIcon = recommendBtn.querySelector('.btn-icon');
        
        if (btnText) btnText.style.display = 'inline';
        if (btnIcon) btnIcon.style.display = 'inline';
        if (btnLoader) btnLoader.style.display = 'none';
    }

    function populateRecommendations(listElement, recommendations, type) {
        listElement.innerHTML = '';
        listElement.classList.add('recommendation-list');

        if (!recommendations || recommendations.length === 0) {
            const noRecsMessage = document.createElement('li');
            noRecsMessage.textContent = `No ${type} recommendations found.`;
            noRecsMessage.style.cssText = `
                text-align: center; 
                color: rgba(255, 255, 255, 0.7); 
                grid-column: 1 / -1;
                padding: 20px;
                font-style: italic;
            `;
            listElement.appendChild(noRecsMessage);
            return;
        }

        recommendations.forEach((song, index) => {
            const card = createSongCard(song, index + 1);
            listElement.appendChild(card);
        });
    }

    function createSongCard(song, index) {
        const card = document.createElement('li');
        card.className = 'recommendation-card';
        
        const year = song.year !== null && song.year !== undefined ? song.year : 'N/A';
        const duration = song.duration_ms !== null && song.duration_ms !== undefined ? formatDuration(song.duration_ms) : 'N/A';
        const album = song.album !== null && song.album !== undefined && song.album !== '' ? song.album : 'N/A';
        
        let audioSection = '';
        if (song.spotify_preview_url && song.spotify_preview_url.trim() !== '') {
            audioSection = `
                <div class="audio-player">
                    <audio controls preload="none">
                        <source src="${song.spotify_preview_url}" type="audio/mpeg">
                        Your browser does not support the audio element.
                    </audio>
                </div>`;
        }

        let tagsSection = '';
        if (song.tags && song.tags !== 'no_tags' && song.tags.trim() !== '') {
            const tagsArray = typeof song.tags === 'string' ? 
                song.tags.split(',').map(tag => tag.trim()).filter(tag => tag.length > 0) : [];
            if (tagsArray.length > 0) {
                tagsSection = `
                    <div class="song-tags">
                        ${tagsArray.map(tag => `<span class="tag">${tag}</span>`).join('')}
                    </div>`;
            }
        }

        card.innerHTML = `
            <div class="card-header">
                <div class="song-title">${index}. ${escapeHtml(song.name)}</div>
                <div class="song-artist">by ${escapeHtml(song.artist)}</div>
            </div>
            
            <div class="song-metadata">
                <div class="metadata-item">
                    <span class="metadata-label">Year</span>
                    <span class="metadata-value">${year}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Duration</span>
                    <span class="metadata-value">${duration}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Album</span>
                    <span class="metadata-value">${escapeHtml(album)}</span>
                </div>
            </div>

            ${audioSection}
            ${tagsSection}
        `;

        return card;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function formatDuration(durationMs) {
        if (typeof durationMs !== 'number' || isNaN(durationMs)) {
            return 'N/A';
        }
        const seconds = Math.floor(durationMs / 1000);
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    function showSuggestions(suggestions) {
        suggestionsList.innerHTML = '';
        suggestions.forEach(suggestion => {
            const li = document.createElement('li');
            li.className = 'suggestion-item';
            li.innerHTML = `
                <div class="suggestion-content">
                    <strong>${escapeHtml(suggestion.song_name)}</strong>
                    <span>by ${escapeHtml(suggestion.artist_name)}</span>
                </div>
            `;
            li.addEventListener('click', () => {
                songNameInput.value = suggestion.song_name;
                artistNameInput.value = suggestion.artist_name;
                hideSections();
                clearMessages();
                validateInputs();
                getRecommendations();
            });
            suggestionsList.appendChild(li);
        });
        suggestionsSection.style.display = 'block';
    }

    async function getRecommendations() {
        clearMessages();
        hideSections();
        showLoadingState();
        recommendBtn.disabled = true;

        const songName = songNameInput.value.trim();
        const artistName = artistNameInput.value.trim();
        const numRecommendations = parseInt(numRecommendationsSelect.value);

        if (!songName || !artistName) {
            showMessage('error', 'Please enter both song title and artist name.');
            hideLoadingState();
            validateInputs();
            return;
        }

        try {
            const requestBody = {
                song_name: songName,
                artist_name: artistName,
                recommendation_count: numRecommendations,
                content_based_weight: 0.5
            };

            console.log('Sending request:', requestBody);

            const response = await fetch(`${API_BASE_URL}/recommend`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Unknown error occurred' }));
                throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            console.log("Response data:", data);

            if (!data.found_match && data.suggested_matches && data.suggested_matches.length > 0) {
                showSuggestions(data.suggested_matches);
                showMessage('info', `Song '${data.query_song}' by '${data.query_artist}' not found. Did you mean one of these?`);
            } else if (!data.found_match) {
                showMessage('info', `Song '${data.query_song}' by '${data.query_artist}' not found and no similar matches were found.`);
            } else {
                // Match found - show recommendations
                clearRecommendationsLists();
                
                populateRecommendations(hybridRecommendationsList, data.hybrid_recommendations, 'Hybrid');
                populateRecommendations(contentBasedRecommendationsList, data.content_based_recommendations, 'Content-Based');
                populateRecommendations(collaborativeRecommendationsList, data.collaborative_recommendations, 'Collaborative');
                
                // Show results section
                resultsSection.style.display = 'block';
                
                // Switch to the tab with the most recommendations, defaulting to hybrid
                let bestTab = 'hybrid';
                let maxRecs = (data.hybrid_recommendations || []).length;
                
                if ((data.content_based_recommendations || []).length > maxRecs) {
                    bestTab = 'content-based';
                    maxRecs = data.content_based_recommendations.length;
                }
                
                if ((data.collaborative_recommendations || []).length > maxRecs) {
                    bestTab = 'collaborative';
                }
                
                openTab(bestTab);
                
                const totalRecs = (data.hybrid_recommendations || []).length + 
                                (data.content_based_recommendations || []).length + 
                                (data.collaborative_recommendations || []).length;
                
                if (totalRecs > 0) {
                    showMessage('success', `Found recommendations for '${data.query_song}' by '${data.query_artist}'!`);
                } else {
                    showMessage('info', `No recommendations available for '${data.query_song}' by '${data.artist_name}'.`);
                }
            }

        } catch (error) {
            console.error('Error fetching recommendations:', error);
            let errorMessage = 'An error occurred while fetching recommendations. Please try again.';
            
            if (error.message.includes('Failed to fetch')) {
                errorMessage = 'Unable to connect to the recommendation service. Please check if the server is running.';
            } else if (error.message) {
                errorMessage = `Error: ${error.message}`;
            }
            
            showMessage('error', errorMessage);
        } finally {
            hideLoadingState();
            validateInputs();
        }
    }

    // Function to handle tab switching
    window.openTab = function(tabName) {
        // Get all elements with class="tab-pane" and hide them
        const tabPanes = document.getElementsByClassName('tab-pane');
        for (let i = 0; i < tabPanes.length; i++) {
            tabPanes[i].classList.remove('active');
        }

        // Get all elements with class="tab-button" and remove the "active" class
        const tabButtons = document.getElementsByClassName('tab-button');
        for (let i = 0; i < tabButtons.length; i++) {
            tabButtons[i].classList.remove('active');
        }

        // Show the current tab, and add an "active" class to the button that opened the tab
        const targetTab = document.getElementById(tabName);
        if (targetTab) {
            targetTab.classList.add('active');
        }

        // Find and activate the corresponding button
        const tabButton = Array.from(tabButtons).find(btn => {
            const onclick = btn.getAttribute('onclick');
            return onclick && onclick.includes(`'${tabName}'`);
        });
        
        if (tabButton) {
            tabButton.classList.add('active');
        }
    };

    // Event listener for recommend button
    recommendBtn.addEventListener('click', getRecommendations);

    // Initialize
    validateInputs();
    
    // Add some visual feedback for form interactions
    [songNameInput, artistNameInput].forEach(input => {
        input.addEventListener('focus', function() {
            this.parentElement.classList.add('focused');
        });
        
        input.addEventListener('blur', function() {
            this.parentElement.classList.remove('focused');
        });
    });
});