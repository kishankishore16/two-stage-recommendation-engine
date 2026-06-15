import sys
import os
import time

# Add the project root to the python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.download import download_dataset
from src.data.preprocess import run_preprocessing
from src.data.features import run_feature_engineering

if __name__ == '__main__':
    print("Starting data pipeline...")
    
    try:
        start_time = time.time()
        download_dataset()
        print(f"Download completed in {time.time() - start_time:.2f} seconds.")
        
        start_time = time.time()
        run_preprocessing()
        print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds.")
        
        start_time = time.time()
        run_feature_engineering()
        print(f"Feature engineering completed in {time.time() - start_time:.2f} seconds.")
        
        print("Data pipeline finished successfully.")
    except Exception as e:
        print(f"Data pipeline failed: {e}")
        sys.exit(1)
