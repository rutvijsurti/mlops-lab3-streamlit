from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

def load_data():
    data = load_wine()
    return data.data, data.target, data.feature_names, data.target_names

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
