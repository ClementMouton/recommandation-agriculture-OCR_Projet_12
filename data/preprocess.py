
import pandas as pd

from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.20

CLASSIFICATION_TARGET = "Crop"
REGRESSION_TARGET = "Yield_tons_per_hectare"

EXCLUDED_FEATURES = ["Days_to_Harvest"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les observations avant modélisation."""

    df = df.copy()

    # Suppression des doublons exacts
    df = df.drop_duplicates()

    # Suppression des observations incomplètes
    df = df.dropna()

    # Exclusion des rendements négatifs
    df = df[df[REGRESSION_TARGET] >= 0]

    return df.reset_index(drop=True)


def prepare_classification(df: pd.DataFrame):
    """Prépare les données pour prédire le type de culture."""

    X = df.drop(
        columns=[
            CLASSIFICATION_TARGET,
            REGRESSION_TARGET,
            *EXCLUDED_FEATURES,
        ]
    )

    y = df[CLASSIFICATION_TARGET]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def prepare_regression(df: pd.DataFrame):
    """Prépare les données pour prédire le rendement."""

    X = df.drop(
        columns=[
            REGRESSION_TARGET,
            *EXCLUDED_FEATURES,
        ]
    )

    y = df[REGRESSION_TARGET]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
