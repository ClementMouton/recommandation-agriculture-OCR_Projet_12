
from pathlib import Path

import pandas as pd

from src.data.load_data import load_crop_yield_data
from src.data.preprocess import (
    clean_data,
    prepare_classification,
    prepare_regression,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def save_datasets(
    datasets: tuple,
    target_name: str,
    output_dir: Path,
) -> None:
    """Enregistre les jeux d'entraînement et de test."""

    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = datasets

    files = {
        "X_train.csv": X_train,
        "X_test.csv": X_test,
        "y_train.csv": y_train,
        "y_test.csv": y_test,
    }

    for filename, data in files.items():
        if isinstance(data, pd.Series):
            data = data.to_frame(name=target_name)

        data.to_csv(
            output_dir / filename,
            index=False,
        )

    print(f"Données enregistrées dans : {output_dir}")


def main() -> None:
    """Exécute le pipeline de préparation des données."""

    print("1. Chargement des données...")
    df = load_crop_yield_data()
    print(f"Observations initiales : {len(df):,}")

    print("\n2. Nettoyage...")
    df = clean_data(df)
    print(f"Observations conservées : {len(df):,}")

    print("\n3. Préparation de la classification...")
    classification_data = prepare_classification(df)

    save_datasets(
        classification_data,
        target_name="Crop",
        output_dir=OUTPUT_DIR / "classification",
    )

    print("\n4. Préparation de la régression...")
    regression_data = prepare_regression(df)

    save_datasets(
        regression_data,
        target_name="Yield_tons_per_hectare",
        output_dir=OUTPUT_DIR / "regression",
    )

    print("\nPipeline terminé avec succès.")


if __name__ == "__main__":
    main()
