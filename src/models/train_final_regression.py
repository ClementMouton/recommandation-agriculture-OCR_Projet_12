import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from data.preprocess import RANDOM_STATE, REGRESSION_TARGET, TEST_SIZE
from src.evaluation.regression import evaluate_regression, log_regression_analysis
from src.models.train_regression import PROJECT_ROOT, build_preprocessor


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "crop_yield_consolidated.csv"
EXPERIMENT_NAME = "agriculture_yield_regression_final"

FEATURES = [
    "Region",
    "Soil_Type",
    "Crop",
    "Rainfall_mm",
    "Temperature_Celsius",
    "Fertilizer_Used",
    "Irrigation_Used",
    "Weather_Condition",
]
CATEGORICAL_FIELDS = ["Region", "Soil_Type", "Crop", "Weather_Condition"]


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"CSV consolidé introuvable : {DATA_PATH}. "
            "Exécute d’abord la cellule d’export du notebook de consolidation."
        )

    data = pd.read_csv(DATA_PATH, usecols=[*FEATURES, REGRESSION_TARGET])
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES],
        data[REGRESSION_TARGET],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    pipeline = make_pipeline(build_preprocessor(X_train), LinearRegression())

    model_metadata = {
        "target": REGRESSION_TARGET,
        "yield_unit": "t/ha",
        "feature_columns": FEATURES,
        "supported_categories": {
            field: sorted(X_train[field].dropna().unique().tolist())
            for field in CATEGORICAL_FIELDS
        },
    }

    db_path = PROJECT_ROOT / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="linear_regression_full_train") as run:
        mlflow.log_params({
            "model": "LinearRegression",
            "dataset": DATA_PATH.name,
            "random_state": RANDOM_STATE,
            "test_fraction": TEST_SIZE,
            "train_size": len(X_train),
            "test_size": len(X_test),
        })

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = evaluate_regression(y_test, predictions)

        mlflow.log_metrics({f"test_{name}": value for name, value in metrics.items()})
        log_regression_analysis(X_test["Crop"], y_test, predictions)
        mlflow.log_dict(model_metadata, "serving/model_metadata.json")

        example = X_train.head(5)
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="yield_model",
            input_example=example,
            signature=infer_signature(example, pipeline.predict(example)),
            metadata=model_metadata,
            serialization_format="cloudpickle",
        )

        print(f"Entraînement : {len(X_train):,} lignes")
        print(f"Test : {len(X_test):,} lignes")
        print(f"MAE : {metrics['mae']:.4f} t/ha")
        print(f"RMSE : {metrics['rmse']:.4f} t/ha")
        print(f"R² : {metrics['r2']:.4f}")
        print(f"Run MLflow : {run.info.run_id}")
        print(f"URI du modèle : {model_info.model_uri}")

    return model_info.model_uri


if __name__ == "__main__":
    main()