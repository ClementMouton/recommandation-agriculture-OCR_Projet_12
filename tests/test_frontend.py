from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "frontend" / "app.py"

RANKING = [
    {"crop": "Barley", "predicted_yield_t_ha": 4.6},
    {"crop": "Rice", "predicted_yield_t_ha": 4.1},
    {"crop": "Cotton", "predicted_yield_t_ha": 4.0},
    {"crop": "Soybean", "predicted_yield_t_ha": 3.8},
    {"crop": "Wheat", "predicted_yield_t_ha": 3.6},
    {"crop": "Maize", "predicted_yield_t_ha": 3.5},
]


@pytest.mark.parametrize(
    ("mode", "endpoint"),
    [("Prédiction", "predict"), ("Recommandation", "recommend")],
)
def test_streamlit_affiche_le_resultat(monkeypatch, mode, endpoint):
    requests_sent = []
    prediction = RANKING[1]
    result = prediction if endpoint == "predict" else {"ranking": RANKING}

    def fake_post(url, *, json, timeout):
        requests_sent.append((url, json))
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: result,
        )

    monkeypatch.setenv("AGRI_API_URL", "http://api.test")
    monkeypatch.setattr(requests, "post", fake_post)

    app = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert not app.exception

    app.radio[0].set_value(mode).run()
    assert not app.exception
    assert not requests_sent

    app.button[0].click().run()
    assert not app.exception
    assert not app.error

    expected_payload = {
        "Region": "West",
        "Soil_Type": "Peaty",
        "Rainfall_mm": 423.76,
        "Temperature_Celsius": 39.30,
        "Fertilizer_Used": False,
        "Irrigation_Used": True,
        "Weather_Condition": "Cloudy",
    }

    if endpoint == "predict":
        expected_payload["Crop"] = "Rice"
        assert app.metric[0].value == "4,10 t/ha"
    else:
        assert app.dataframe[0].value["Culture"].tolist() == [
            "Orge", "Riz", "Coton", "Soja", "Blé", "Maïs",
        ]

    assert requests_sent == [
        (f"http://api.test/{endpoint}", expected_payload)
    ]