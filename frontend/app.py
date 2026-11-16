import os

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("AGRI_API_URL", "http://127.0.0.1:8000").rstrip("/")

REGIONS = {
    "Ouest": "West",
    "Est": "East",
    "Nord": "North",
    "Sud": "South",
}
SOILS = {
    "Tourbeux": "Peaty",
    "Argileux": "Clay",
    "Calcaire": "Chalky",
    "Limoneux (Loam)": "Loam",
    "Sableux": "Sandy",
    "Limoneux (Silt)": "Silt",
}
WEATHER = {
    "Nuageux": "Cloudy",
    "Pluvieux": "Rainy",
    "Ensoleillé": "Sunny",
}
CROPS = {
    "Riz": "Rice",
    "Orge": "Barley",
    "Coton": "Cotton",
    "Maïs": "Maize",
    "Soja": "Soybean",
    "Blé": "Wheat",
}
CROP_LABELS = {value: label for label, value in CROPS.items()}


def call_api(endpoint, payload):
    try:
        response = requests.post(
            f"{API_URL}/{endpoint}", json=payload, timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Impossible de joindre FastAPI à l’adresse {API_URL}.")
    except requests.exceptions.Timeout:
        st.error("Le délai de réponse de FastAPI est dépassé.")
    except requests.exceptions.HTTPError as error:
        st.error(f"L’API a renvoyé le code HTTP {error.response.status_code}.")
        if error.response.status_code == 422:
            st.write(error.response.json().get("detail", []))
    except requests.exceptions.RequestException:
        st.error("La requête a échoué ou l’API a renvoyé une réponse illisible.")
    return None


def format_yield(value):
    return f"{value:.2f}".replace(".", ",")


def show_ranking(ranking):
    table = pd.DataFrame([
        {
            "Culture": CROP_LABELS.get(row["crop"], row["crop"]),
            "Rendement estimé (t/ha)": row["predicted_yield_t_ha"],
        }
        for row in ranking
    ])

    display_table = table.copy()
    display_table.index = range(1, len(table) + 1)
    display_table.index.name = "Rang"
    display_table["Rendement estimé (t/ha)"] = (
        display_table["Rendement estimé (t/ha)"].map(format_yield)
    )

    st.subheader("Cultures classées par rendement estimé")
    st.dataframe(display_table)
    st.bar_chart(
        table,
        x="Culture",
        y="Rendement estimé (t/ha)",
        horizontal=True,
    )

    yields = table["Rendement estimé (t/ha)"]
    spread_kg_ha = (yields.max() - yields.min()) * 1_000
    spread = f"{spread_kg_ha:.2f}".replace(".", ",")
    st.caption(f"Écart entre la première et la dernière culture : {spread} kg/ha.")

    if display_table["Rendement estimé (t/ha)"].nunique() == 1:
        st.info(
            "Toutes les cultures ont le même rendement affiché à deux décimales. "
            "Les écarts estimés sont trop faibles pour guider à eux seuls "
            "un choix de plantation."
        )


st.set_page_config(page_title="Agritech Answers")
st.title("Agritech Answers")

mode = st.radio("Mode", ["Prédiction", "Recommandation"], horizontal=True)

with st.form("parcelle"):
    st.subheader("Conditions de la parcelle")
    left, right = st.columns(2)

    with left:
        region = st.selectbox("Région", list(REGIONS))
        soil = st.selectbox("Type de sol", list(SOILS))
        weather = st.selectbox("Conditions météo", list(WEATHER))

    with right:
        rainfall = st.slider("Précipitations (mm)", 100.0, 1000.0, 423.76, 0.01)
        temperature = st.slider("Température (°C)", 15.0, 40.0, 39.30, 0.01)
        fertilizer = st.checkbox("Fertilisation")
        irrigation = st.checkbox("Irrigation", value=True)

    if mode == "Prédiction":
        crop = st.selectbox("Culture", list(CROPS))

    submitted = st.form_submit_button("Calculer", type="primary")

if submitted:
    payload = {
        "Region": REGIONS[region],
        "Soil_Type": SOILS[soil],
        "Rainfall_mm": rainfall,
        "Temperature_Celsius": temperature,
        "Fertilizer_Used": fertilizer,
        "Irrigation_Used": irrigation,
        "Weather_Condition": WEATHER[weather],
    }
    endpoint = "recommend"
    if mode == "Prédiction":
        endpoint = "predict"
        payload["Crop"] = CROPS[crop]

    with st.spinner("Calcul du rendement…"):
        result = call_api(endpoint, payload)

    if result is not None:
        if mode == "Prédiction":
            value = format_yield(result["predicted_yield_t_ha"])
            st.metric(f"Rendement estimé — {crop}", f"{value} t/ha")
        else:
            show_ranking(result["ranking"])