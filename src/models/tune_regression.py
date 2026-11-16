"""Compare trois familles de modèles sur la MAE en validation croisée."""

import json

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from src.evaluation.regression import evaluate_regression, log_regression_analysis
from src.models.train_regression import (
    PROJECT_ROOT,
    RANDOM_STATE,
    build_preprocessor,
    load_split,
    sample_training_data,
)

N_SPLITS = 5
MIN_MAE_GAIN = 0.001  # t/ha
EXPERIMENT_NAME = "agriculture_yield_regression_tuning"


def model_grids():
    # Ordre de préférence lorsque l'écart avec le meilleur score est minime.
    return [
        (LinearRegression(), {}),
        (Ridge(), {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}),
        (
            HistGradientBoostingRegressor(
                random_state=RANDOM_STATE, early_stopping=False, max_iter=100
            ),
            {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [7, 15]},
        ),
    ]


def main():
    X_train, y_train = sample_training_data(*load_split("train"))
    cv = list(KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(X_train))
    db_path = PROJECT_ROOT / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"Validation croisée à {N_SPLITS} plis sur {len(X_train):,} lignes.")
    with mlflow.start_run(run_name="comparaison_regression"):
        mlflow.log_params({
            "cv_folds": N_SPLITS,
            "random_state": RANDOM_STATE,
            "train_size": len(X_train),
            "selection_tolerance_t_ha": MIN_MAE_GAIN,
        })
        summaries = []
        searches = {}
        for estimator, grid in model_grids():
            name = type(estimator).__name__
            pipeline = Pipeline([
                ("preprocessor", build_preprocessor(X_train)),
                ("model", estimator),
            ])
            search = GridSearchCV(
                pipeline, grid, cv=cv, scoring="neg_mean_absolute_error",
                refit=False, error_score="raise",
            )
            with mlflow.start_run(run_name=name, nested=True):
                search.fit(X_train, y_train)
                results = search.cv_results_
                trials = pd.DataFrame({
                    "hyperparameters": [json.dumps(params) for params in results["params"]],
                    "cv_mae_mean": -results["mean_test_score"],
                    "cv_mae_std": results["std_test_score"],
                    "cv_rank": results["rank_test_score"],
                    **{f"fold_{i + 1}_mae": -results[f"split{i}_test_score"] for i in range(N_SPLITS)},
                })
                best = trials.iloc[search.best_index_]
                mlflow.log_params(estimator.get_params() | {
                    key.removeprefix("model__"): value for key, value in search.best_params_.items()
                })
                scores = {"cv_mae_mean": float(best["cv_mae_mean"]), "cv_mae_std": float(best["cv_mae_std"])}
                mlflow.log_metrics(scores)
                mlflow.log_table(trials, "validation/cv_trials.json")
                summaries.append({"model": name, **scores, "hyperparameters": best["hyperparameters"]})
                searches[name] = search
                print(f"{name}: MAE CV {scores['cv_mae_mean']:.6f} t/ha")

        best_mae = min(row["cv_mae_mean"] for row in summaries)
        selected = next(row for row in summaries if row["cv_mae_mean"] <= best_mae + MIN_MAE_GAIN)
        selected_name = selected["model"]
        comparison = pd.DataFrame(summaries).sort_values("cv_mae_mean")
        comparison["selected"] = comparison["model"].eq(selected_name)
        mlflow.log_table(comparison, "validation/cv_comparison.json")
        mlflow.log_param("selected_model", selected_name)
        mlflow.log_metric("selected_cv_mae", selected["cv_mae_mean"])

        search = searches[selected_name]
        model = clone(search.estimator).set_params(**search.best_params_)
        model.fit(X_train, y_train)
        X_test, y_test = load_split("test")
        predictions = model.predict(X_test)
        scores = evaluate_regression(y_test, predictions)
        mlflow.log_metrics({f"test_{key}": value for key, value in scores.items()})
        mlflow.log_param("test_size", len(X_test))
        log_regression_analysis(X_test["Crop"], y_test, predictions)
        mlflow.sklearn.log_model(
            sk_model=model, name="model", input_example=X_train.head(3),
            serialization_format="cloudpickle",
        )
        print(f"\nModèle retenu : {selected_name}")
        print(comparison.to_string(index=False))
        print("Métriques sur le jeu de test :", json.dumps(scores))


if __name__ == "__main__":
    main()
