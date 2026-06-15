import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add the project root to the python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def sample_interactions_df():
    """Create a sample interactions DataFrame."""
    np.random.seed(42)
    return pd.DataFrame({
        'user_id': np.random.randint(0, 20, 100),
        'item_id': np.random.randint(0, 50, 100),
        'rating': np.random.choice([1, 2, 3, 4, 5], 100),
        'timestamp': np.linspace(1600000000, 1610000000, 100).astype(int),
        'label': np.random.choice([0, 1], 100)
    })

@pytest.fixture
def sample_user_features_df():
    """Create sample user features DataFrame."""
    np.random.seed(42)
    df = pd.DataFrame(np.random.rand(20, 8), columns=[
        'user_avg_rating', 'user_review_count', 'user_avg_price', 
        'user_positive_ratio', 'user_top_category', 'user_top_brand', 
        'user_price_std', 'user_activity_span'
    ])
    df.index.name = 'user_id'
    return df

@pytest.fixture
def sample_item_features_df():
    """Create sample item features DataFrame."""
    np.random.seed(42)
    df = pd.DataFrame(np.random.rand(50, 7), columns=[
        'item_avg_rating', 'item_review_count', 'item_price', 
        'item_price_tier', 'item_brand_encoded', 'item_category_encoded', 
        'item_positive_ratio'
    ])
    df.index.name = 'item_id'
    return df
