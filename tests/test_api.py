from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api import main


PARCELLE = {
    "Region": "West",
    "Soil_Type": "Peaty",
    "Rainfall_mm": 423.76,
    "Temperature_Celsius": 39.30,
    "Fertilizer_Used": False,
    "Irrigation_Used": True,
    "Weather_Condition": "Cloudy",
}

RENDEMENTS = {
    "Barley": 4.0,
    "Cotton": 3.0,
    "Maize": 6.0,
    "Rice": 5.0,
    "Soybean": 4.5,
    "Wheat": 5.5,
}


class ModeleTest:
    def predict(self, inputs):
        return inputs["Crop"].map(RENDEMENTS).to_numpy()


@pytest.fixture
def client(monkeypatch):
    metadata = {
        "feature_columns": [*PARCELLE, "Crop"],
        "supported_categories": {
            "Region": ["West"],
            "Soil_Type": ["Peaty"],
            "Weather_Condition": ["Cloudy"],
            "Crop": list(RENDEMENTS),
        },
    }

    monkeypatch.setenv("YIELD_MODEL_URI", "models:/test")
    monkeypatch.setattr(main.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(
        main, "get_model_info", lambda uri: SimpleNamespace(metadata=metadata)
    )
    monkeypatch.setattr(
        main.mlflow.sklearn, "load_model", lambda uri: ModeleTest()
    )

    with TestClient(main.app) as test_client:
        yield test_client


def test_predict_retourne_le_rendement_de_la_culture(client):
    response = client.post("/predict", json={**PARCELLE, "Crop": "Rice"})

    assert response.status_code == 200, response.text
    assert response.json() == {
        "crop": "Rice",
        "predicted_yield_t_ha": 5.0,
    }


def test_recommend_classe_les_six_cultures(client):
    response = client.post("/recommend", json=PARCELLE)

    assert response.status_code == 200, response.text
    assert response.json()["ranking"] == [
        {"crop": crop, "predicted_yield_t_ha": RENDEMENTS[crop]}
        for crop in ["Maize", "Wheat", "Rice", "Soybean", "Barley", "Cotton"]
    ]


@pytest.mark.parametrize(
    ("endpoint", "payload", "field"),
    [
        ("/predict", PARCELLE, "Crop"),
        ("/predict", {**PARCELLE, "Crop": "Tomato"}, "Crop"),
        ("/recommend", {**PARCELLE, "Rainfall_mm": -1}, "Rainfall_mm"),
        ("/recommend", {**PARCELLE, "Region": "Unknown"}, "Region"),
        ("/recommend", {**PARCELLE, "Crop": "Rice"}, "Crop"),
    ],
)
def test_entree_invalide(client, endpoint, payload, field):
    response = client.post(endpoint, json=payload)

    assert response.status_code == 422, response.text
    errors = response.json()["detail"]
    invalid_fields = [
        error["field"] if "field" in error else error["loc"][-1]
        for error in errors
    ]
    assert field in invalid_fields