
from pathlib import Path
import json

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.evaluation.regression import evaluate_regression, log_regression_analysis

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "regression"
RESULTS_DIR = PROJECT_ROOT / "reports" / "metrics"

RANDOM_STATE = 42
TRAIN_SAMPLE_SIZE = 100_000

EXPERIMENT_NAME = "agriculture_yield_regression"


# --------------------------------------------------
# Chargement des données
# --------------------------------------------------

def load_datasets():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    y_train = pd.read_csv(
        DATA_DIR / "y_train.csv"
    ).squeeze("columns")

    y_test = pd.read_csv(
        DATA_DIR / "y_test.csv"
    ).squeeze("columns")

    return X_train, X_test, y_train, y_test


# --------------------------------------------------
# Prétraitement
# --------------------------------------------------

def build_preprocessor(X):
    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ])


# --------------------------------------------------
# Modèles
# --------------------------------------------------

def get_models():
    return {
        "DummyRegressor": DummyRegressor(
            strategy="mean"
        ),
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=50,
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


# --------------------------------------------------
# Entraînement et suivi MLflow
# --------------------------------------------------

def main():
    print("Chargement des données...")

    X_train, X_test, y_train, y_test = load_datasets()

    # Même échantillon d'entraînement pour tous les modèles
    sample_indices = X_train.sample(
        n=min(TRAIN_SAMPLE_SIZE, len(X_train)),
        random_state=RANDOM_STATE,
    ).index

    X_train = X_train.loc[sample_indices]
    y_train = y_train.loc[sample_indices]

    print(f"Entraînement : {len(X_train):,} lignes")
    print(f"Évaluation : {len(X_test):,} lignes")

    # Base de suivi locale MLflow
    db_path = PROJECT_ROOT / "mlflow.db"
    mlflow.set_tracking_uri(
        f"sqlite:///{db_path.as_posix()}"
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for model_name, estimator in get_models().items():
        print(f"\nEntraînement : {model_name}")

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("model", estimator),
        ])

        with mlflow.start_run(run_name=model_name):

            mlflow.set_tag("task", "regression")
            mlflow.set_tag("target", "Yield_tons_per_hectare")

            mlflow.log_param("model_type", model_name)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))

            for param_name, param_value in (
                estimator.get_params().items()
            ):
                mlflow.log_param(
                    f"model__{param_name}",
                    str(param_value),
                )

            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_test)
            metrics = evaluate_regression(y_test, predictions)
            mlflow.log_metrics(metrics)
            log_regression_analysis(X_test["Crop"], y_test, predictions)

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                input_example=X_test.head(3),
                serialization_format="cloudpickle",
            )

            result = {
                "model": model_name,
                **metrics,
            }
            results.append(result)

            print(json.dumps(metrics, indent=2))

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("rmse")

    output_path = RESULTS_DIR / "regression_comparison.csv"
    results_df.to_csv(output_path, index=False)

    print("\nComparaison finale :")
    print(results_df.to_string(index=False))

    print(f"\nRésultats enregistrés : {output_path}")


if __name__ == "__main__":
    main()
