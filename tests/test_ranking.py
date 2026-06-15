import pandas as pd
import numpy as np
from src.ranking.features import build_cross_features, build_ranking_feature_matrix, get_feature_names

def test_cross_features(sample_user_features_df, sample_item_features_df):
    user_feats = sample_user_features_df.iloc[0]
    item_feats = sample_item_features_df.iloc[0]
    retrieval_score = 0.85
    
    cross_feats = build_cross_features(user_feats, item_feats, retrieval_score)
    
    assert len(cross_feats) == 8
    assert 'price_match' in cross_feats
    assert 'retrieval_score' in cross_feats
    assert cross_feats['retrieval_score'] == 0.85

def test_feature_matrix_shape(sample_user_features_df, sample_item_features_df):
    # (user_id, item_id, retrieval_score, label)
    candidates = [
        (0, 0, 0.9, 1),
        (0, 1, 0.8, 0),
        (1, 0, 0.7, 1)
    ]
    
    X, y = build_ranking_feature_matrix(sample_user_features_df, sample_item_features_df, candidates)
    
    assert X.shape == (3, 23)
    assert y.shape == (3,)
    assert list(y) == [1, 0, 1]

def test_feature_names_count():
    names = get_feature_names()
    assert len(names) == 23
