document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('recommend-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = document.querySelector('.btn-text');
    const loader = document.querySelector('.loader');
    const resultsGrid = document.getElementById('results-grid');
    const statusContainer = document.getElementById('status-container');

    // Resolve API URL: relative path works on Railway (frontend served by same FastAPI app).
    // Only override for local dev when frontend is opened on a port other than 8000.
    let API_URL = '/recommend';
    if ((window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
        && window.location.port && window.location.port !== '8000') {
        API_URL = 'http://localhost:8000/recommend';
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const userId = document.getElementById('user-id').value.trim();
        const numResults = parseInt(document.getElementById('num-results').value, 10);

        if (!userId) return;

        // UI Loading State
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
        submitBtn.disabled = true;
        
        // Clear previous
        resultsGrid.innerHTML = '';
        statusContainer.innerHTML = '';
        statusContainer.classList.add('hidden');

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    user_id: userId,
                    num_candidates: 200,
                    num_results: numResults
                })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Failed to fetch recommendations');
            }

            const data = await response.json();
            renderResults(data);
        } catch (error) {
            console.error('Error:', error);
            resultsGrid.innerHTML = `<div class="error-msg"><strong>Error:</strong> ${error.message}</div>`;
        } finally {
            // Restore UI
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
            submitBtn.disabled = false;
        }
    });

    function renderResults(data) {
        // Render timing stats
        statusContainer.innerHTML = `
            <div class="status-badge">
                <span class="label">Total Time:</span>
                <span class="value">${data.total_time_ms.toFixed(1)} ms</span>
            </div>
            <div class="status-badge">
                <span class="label">Retrieval (FAISS):</span>
                <span class="value">${data.retrieval_time_ms.toFixed(1)} ms</span>
            </div>
            <div class="status-badge">
                <span class="label">Ranking (LGBM):</span>
                <span class="value">${data.ranking_time_ms.toFixed(1)} ms</span>
            </div>
        `;
        statusContainer.classList.remove('hidden');

        // Render Cards
        if (!data.recommendations || data.recommendations.length === 0) {
            resultsGrid.innerHTML = `<div class="error-msg" style="background: transparent; border: none; color: var(--text-muted)">No recommendations found.</div>`;
            return;
        }

        data.recommendations.forEach((item, index) => {
            const card = document.createElement('div');
            card.className = 'product-card';
            card.style.animationDelay = `${index * 0.05}s`; // Staggered animation

            // Format price gracefully
            const priceHtml = item.price ? `$${item.price.toFixed(2)}` : 'N/A';
            const categoryHtml = item.category && item.category !== 'Unknown' 
                ? `<span class="category-tag">${item.category}</span>` 
                : '<span class="category-tag" style="color: #6b7280;">Generic</span>';

            card.innerHTML = `
                ${categoryHtml}
                <div class="asin">${item.asin}</div>
                <h3 class="product-title" title="${item.title}">${item.title || 'Untitled Product'}</h3>
                <div class="product-footer">
                    <div class="price">${priceHtml}</div>
                    <div class="score">
                        <span>Score</span>
                        <span class="score-val">${item.score.toFixed(3)}</span>
                    </div>
                </div>
            `;

            resultsGrid.appendChild(card);
        });
    }
});
