import mlflow
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else None,
    }


def log_regression_analysis(crops, actual, predicted):
    """Enregistre les diagnostics du jeu de test dans le run actif."""
    evaluation = pd.DataFrame({
        "Crop": np.asarray(crops),
        "actual": np.asarray(actual),
        "predicted": np.asarray(predicted),
    })
    evaluation["residual"] = evaluation["actual"] - evaluation["predicted"]

    rows = []
    for crop, group in evaluation.groupby("Crop", dropna=False, observed=True):
        rows.append({
            "Crop": None if pd.isna(crop) else str(crop),
            "count": len(group),
            **evaluate_regression(group["actual"], group["predicted"]),
            "mean_residual": float(group["residual"].mean()),
        })
    by_crop = pd.DataFrame(rows).sort_values("mae")
    mlflow.log_dict({"crops": rows}, "evaluation/metrics_by_crop.json")

    sample = evaluation.sample(n=min(5_000, len(evaluation)), random_state=42)
    fig = Figure(figsize=(6, 6), layout="constrained")
    ax = fig.subplots()
    ax.scatter(sample["actual"], sample["predicted"], s=8, alpha=0.25)
    lower = sample[["actual", "predicted"]].min().min()
    upper = sample[["actual", "predicted"]].max().max()
    ax.plot([lower, upper], [lower, upper], "--", color="black")
    ax.set(
        xlabel="Rendement observé (t/ha)",
        ylabel="Rendement prédit (t/ha)",
        title=f"Rendements observés et prédits ({len(sample):,} points)",
        aspect="equal",
    )
    mlflow.log_figure(fig, "evaluation/actual_vs_predicted.png")

    fig = Figure(figsize=(8, 5), layout="constrained")
    ax = fig.subplots()
    ax.hist(evaluation["residual"], bins=60, color="#376b8b")
    ax.axvline(0, linestyle="--", color="black")
    ax.set(
        xlabel="Rendement observé moins prédit (t/ha)",
        ylabel="Nombre d’observations",
        title="Distribution des résidus",
    )
    mlflow.log_figure(fig, "evaluation/residuals.png")

    fig = Figure(figsize=(8, 5), layout="constrained")
    ax = fig.subplots()
    ax.barh(by_crop["Crop"].fillna("Culture inconnue"), by_crop["mae"])
    ax.invert_yaxis()
    ax.set(xlabel="MAE (t/ha)", title="Erreur absolue moyenne par culture")
    mlflow.log_figure(fig, "evaluation/mae_by_crop.png")
