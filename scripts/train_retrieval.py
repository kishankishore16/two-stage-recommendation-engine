import sys
import os

# Add the project root to the python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.train import train_two_tower

if __name__ == '__main__':
    print("Starting Two-Tower retrieval model training...")
    try:
        model = train_two_tower()
        print("Training finished successfully.")
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)
