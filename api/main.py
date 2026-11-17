import os
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from mlflow.models import Model, get_model_info
from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Parcelle(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    Region: str
    Soil_Type: str
    Rainfall_mm: float = Field(ge=0)
    Temperature_Celsius: float
    Fertilizer_Used: bool
    Irrigation_Used: bool
    Weather_Condition: str


class PredictionInput(Parcelle):
    Crop: str


class Rendement(BaseModel):
    crop: str
    predicted_yield_t_ha: float


class Recommandation(BaseModel):
    ranking: list[Rendement]


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_uri = os.getenv("YIELD_MODEL_URI")
    if not model_uri:
        raise RuntimeError("Définis YIELD_MODEL_URI avant de lancer l’API.")

    model_manifest = Path(model_uri) / "MLmodel"
    if model_manifest.is_file():
        metadata = Model.load(str(model_manifest)).metadata
    else:
        db_path = PROJECT_ROOT / "mlflow.db"
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or (
            f"sqlite:///{db_path.as_posix()}"
        )
        mlflow.set_tracking_uri(tracking_uri)
        metadata = get_model_info(model_uri).metadata

    app.state.feature_columns = metadata["feature_columns"]
    app.state.categories = metadata["supported_categories"]
    app.state.model = mlflow.sklearn.load_model(model_uri)

    yield


app = FastAPI(title="Agritech Answers", lifespan=lifespan)


def validate_categories(values, categories):
    errors = [
        {
            "field": field,
            "value": values[field],
            "allowed_values": allowed,
        }
        for field, allowed in categories.items()
        if field in values and values[field] not in allowed
    ]
    if errors:
        raise HTTPException(status_code=422, detail=errors)


@app.post("/predict", response_model=Rendement)
def predict(payload: PredictionInput, request: Request):
    state = request.app.state
    values = payload.model_dump()
    validate_categories(values, state.categories)

    inputs = pd.DataFrame([values], columns=state.feature_columns)
    prediction = state.model.predict(inputs)[0]
    return Rendement(crop=payload.Crop, predicted_yield_t_ha=float(prediction))


@app.post("/recommend", response_model=Recommandation)
def recommend(payload: Parcelle, request: Request):
    state = request.app.state
    values = payload.model_dump()
    validate_categories(values, state.categories)

    crops = state.categories["Crop"]
    inputs = pd.DataFrame(
        [{**values, "Crop": crop} for crop in crops],
        columns=state.feature_columns,
    )
    predictions = state.model.predict(inputs)

    ranking = [
        Rendement(crop=crop, predicted_yield_t_ha=float(prediction))
        for crop, prediction in zip(crops, predictions, strict=True)
    ]
    ranking.sort(key=lambda item: (-item.predicted_yield_t_ha, item.crop))
    return Recommandation(ranking=ranking)