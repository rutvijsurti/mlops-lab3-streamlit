import joblib

bundle = joblib.load("../model/wine_model.pkl")

def predict_data(X):
    model = bundle["model"]
    return model.predict(X), model.predict_proba(X)

def get_model_info():
    return bundle
