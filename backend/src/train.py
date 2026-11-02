import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data import load_data, split_data

def train():
    X, y, feature_names, target_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Test accuracy: {acc:.3f}")

    os.makedirs("../model", exist_ok=True)
    joblib.dump({
        "model": model,
        "accuracy": acc,
        "features": list(feature_names),
        "classes": list(target_names),
    }, "../model/wine_model.pkl")
    print("Model saved to ../model/wine_model.pkl")

if __name__ == "__main__":
    train()
