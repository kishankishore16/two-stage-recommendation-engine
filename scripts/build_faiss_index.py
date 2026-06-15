import sys
import os
import pandas as pd
import torch

# Add the project root to the python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.index import build_faiss_index
from src.retrieval.model import TwoTowerModel
from src.config import get_config

if __name__ == '__main__':
    print("Building FAISS index...")
    config = get_config()
    try:
        # Load item features
        print(f"Loading item features from {config.paths.item_features_file}")
        item_features_df = pd.read_parquet(config.paths.item_features_file)
        
        # Load model
        print(f"Loading model from {config.paths.two_tower_model_file}")
        # Need to know num_users and num_items. Get them from the mappings
        user_id_map = pd.read_parquet(config.paths.user_id_map_file)
        item_id_map = pd.read_parquet(config.paths.item_id_map_file)
        num_users = len(user_id_map)
        num_items = len(item_id_map)
        
        model = TwoTowerModel(num_users=num_users, num_items=num_items)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(config.paths.two_tower_model_file, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        # Build index
        index = build_faiss_index(model, item_features_df, device)
        
        print(f"FAISS index built successfully with {index.ntotal} vectors of dimension {index.d}.")
    except Exception as e:
        print(f"FAISS index build failed: {e}")
        sys.exit(1)
