import sys
import os

# Add the project root to the python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ranking.train import train_ranker

if __name__ == '__main__':
    print("Starting LightGBM ranker training...")
    try:
        model = train_ranker()
        print("Ranker training finished successfully.")
    except Exception as e:
        print(f"Ranker training failed: {e}")
        sys.exit(1)
