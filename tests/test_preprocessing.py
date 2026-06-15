import pandas as pd
from src.data.preprocess import create_implicit_labels, time_based_split

def test_create_implicit_labels():
    df = pd.DataFrame({'rating': [1, 2, 3, 4, 5, 4.5]})
    labeled_df = create_implicit_labels(df)
    assert labeled_df['label'].tolist() == [0, 0, 0, 1, 1, 1]

def test_time_based_split():
    df = pd.DataFrame({
        'timestamp': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'data': list(range(10))
    })
    train_df, test_df = time_based_split(df, train_ratio=0.8)
    assert len(train_df) == 8
    assert len(test_df) == 2
    assert train_df['timestamp'].max() <= test_df['timestamp'].min()
