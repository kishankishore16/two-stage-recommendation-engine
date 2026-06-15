import torch
import pytest
from src.retrieval.model import TwoTowerModel, infonce_loss

def test_two_tower_model_shapes():
    num_users = 100
    num_items = 100
    batch_size = 16
    embedding_dim = 64
    
    model = TwoTowerModel(num_users=num_users, num_items=num_items)
                          
    user_ids = torch.randint(0, num_users, (batch_size,))
    user_features = torch.rand(batch_size, 8)
    item_ids = torch.randint(0, num_items, (batch_size,))
    item_features = torch.rand(batch_size, 7)
    
    user_emb, item_emb = model(user_ids, user_features, item_ids, item_features)
    
    assert user_emb.shape == (batch_size, embedding_dim)
    assert item_emb.shape == (batch_size, embedding_dim)

def test_infonce_loss():
    batch_size = 16
    embedding_dim = 64
    
    user_emb = torch.rand(batch_size, embedding_dim)
    item_emb = torch.rand(batch_size, embedding_dim)
    
    loss = infonce_loss(user_emb, item_emb)
    assert loss.item() > 0

def test_embeddings_normalized():
    model = TwoTowerModel(num_users=10, num_items=10)
    user_ids = torch.randint(0, 10, (5,))
    user_features = torch.rand(5, 8)
    
    user_emb = model.encode_user(user_ids, user_features)
    norms = torch.norm(user_emb, p=2, dim=1)
    
    # Assert L2 norm is approximately 1.0
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
