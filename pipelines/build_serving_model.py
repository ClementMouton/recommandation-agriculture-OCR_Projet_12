from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import pandas as pd
from mlflow.artifacts import download_artifacts

from data.load_data import RAW_DATA_PATH, load_crop_yield_data
from data.preprocess import clean_data
from src.models.train_final_regression import DATA_PATH, PROJECT_ROOT
from src.models.train_final_regression import main as train_final_model


CROP_NAMES = {
    "Maize": "Maize",
    "Rice, paddy": "Rice",
    "Soybeans": "Soybean",
    "Wheat": "Wheat",
}
EXPORT_DIR = PROJECT_ROOT / "artifacts" / "serving"


def consolidate_data() -> None:
    agriculture = clean_data(load_crop_yield_data())

    with ZipFile(RAW_DATA_PATH) as outer:
        archive = BytesIO(outer.read("OC/Crop Yield Prediction Dataset.zip"))
        with ZipFile(archive) as inner, inner.open("yield_df.csv") as source:
            history = pd.read_csv(source)

    annual = history.groupby(["Area", "Item", "Year"], as_index=False).agg(
        rainfall_mm=("average_rain_fall_mm_per_year", "median"),
        temperature_c=("avg_temp", "median"),
        pesticides_tonnes=("pesticides_tonnes", "median"),
    )
    annual["Crop"] = annual["Item"].map(CROP_NAMES)

    profiles = annual.dropna(subset=["Crop"]).groupby("Crop", as_index=False).agg(
        hist_country_year_count=("Year", "size"),
        hist_rainfall_mm_median=("rainfall_mm", "median"),
        hist_temperature_c_median=("temperature_c", "median"),
        hist_pesticides_tonnes_median=("pesticides_tonnes", "median"),
    )

    consolidated = agriculture.merge(
        profiles, on="Crop", how="left", validate="many_to_one", sort=False
    )
    consolidated["has_historical_profile"] = (
        consolidated["hist_country_year_count"].notna()
    )

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=DATA_PATH.parent) as staging:
        temporary_csv = Path(staging) / DATA_PATH.name
        consolidated.to_csv(temporary_csv, index=False)
        temporary_csv.replace(DATA_PATH)

    enriched = consolidated["has_historical_profile"].sum()
    print(f"CSV consolidé : {DATA_PATH}")
    print(f"Observations : {len(consolidated):,}, dont {enriched:,} enrichies")


def main() -> None:
    consolidate_data()
    model_uri = train_final_model()

    model_id = model_uri.rsplit("/", 1)[-1]
    destination = EXPORT_DIR / model_id
    destination.mkdir(parents=True, exist_ok=True)

    model_path = Path(
        download_artifacts(artifact_uri=model_uri, dst_path=str(destination))
    ).resolve()

    if not (model_path / "MLmodel").is_file():
        raise RuntimeError(f"Fichier MLmodel introuvable dans {model_path}")

    print(f"Modèle exporté : {model_path}")


if __name__ == "__main__":
    main()