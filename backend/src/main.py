from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from predict import predict_data, get_model_info

app = FastAPI(title="Wine Classifier API")

class WineData(BaseModel):
    alcohol: float = Field(..., gt=0)
    malic_acid: float = Field(..., ge=0)
    ash: float = Field(..., ge=0)
    alcalinity_of_ash: float = Field(..., ge=0)
    magnesium: float = Field(..., ge=0)
    total_phenols: float = Field(..., ge=0)
    flavanoids: float = Field(..., ge=0)
    nonflavanoid_phenols: float = Field(..., ge=0)
    proanthocyanins: float = Field(..., ge=0)
    color_intensity: float = Field(..., ge=0)
    hue: float = Field(..., ge=0)
    od280_od315_of_diluted_wines: float = Field(..., ge=0)
    proline: float = Field(..., ge=0)

class WineResponse(BaseModel):
    class_id: int
    class_name: str
    probabilities: dict[str, float]

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.get("/model-info")
async def model_info():
    info = get_model_info()
    return {
        "model": "RandomForestClassifier",
        "test_accuracy": round(info["accuracy"], 3),
        "classes": info["classes"],
    }

@app.post("/predict", response_model=WineResponse)
async def predict_wine(wine: WineData):
    try:
        X = [list(wine.model_dump().values())]
        pred, probs = predict_data(X)
        classes = get_model_info()["classes"]
        class_id = int(pred[0])
        return WineResponse(
            class_id=class_id,
            class_name=classes[class_id],
            probabilities={c: round(float(p), 3) for c, p in zip(classes, probs[0])},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
