import sys
import os
import pandas as pd
import redis

# Add the project root to the python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_config

def populate_redis():
    config = get_config()
    print(f"Connecting to Redis at {config.serving.redis_host}:{config.serving.redis_port}")
    try:
        r = redis.Redis(host=config.serving.redis_host, port=config.serving.redis_port, db=config.serving.redis_db, decode_responses=True)
        r.ping()
    except Exception as e:
        print(f"Could not connect to Redis: {e}")
        return

    print("Loading user features...")
    user_features_df = pd.read_parquet(config.paths.user_features_file)
    print("Loading item features...")
    item_features_df = pd.read_parquet(config.paths.item_features_file)
    
    # We also need original user_id (string) to internal user_id (int) mapping,
    # or we key by internal user_id. Let's assume the API receives internal user_id
    # for simplicity, or we store by original user_id. Let's store by internal int id
    # as the index of these dfs is the internal int id.

    print("Populating user profiles...")
    pipe = r.pipeline()
    count = 0
    for user_id, row in user_features_df.iterrows():
        # Convert row to dict, converting float/int types to strings/floats that redis handles
        features = {k: float(v) for k, v in row.items()}
        pipe.hset(f"user:{user_id}", mapping=features)
        count += 1
        if count % 10000 == 0:
            pipe.execute()
            print(f"  Pushed {count} users")
    pipe.execute()
    print(f"Successfully populated {count} user profiles.")

    print("Populating item features...")
    pipe = r.pipeline()
    count = 0
    for item_id, row in item_features_df.iterrows():
        features = {k: float(v) for k, v in row.items()}
        pipe.hset(f"item:{item_id}", mapping=features)
        count += 1
        if count % 10000 == 0:
            pipe.execute()
            print(f"  Pushed {count} items")
    pipe.execute()
    print(f"Successfully populated {count} item profiles.")

if __name__ == '__main__':
    populate_redis()
