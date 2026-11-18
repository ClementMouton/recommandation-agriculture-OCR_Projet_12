# Agritech Answers

Prototype réalisé dans le cadre du projet 12 du parcours Data Scientist Machine Learning d’OpenClassrooms.

L’application estime le rendement d’une culture en tonnes par hectare et compare les rendements prédits de six cultures pour les mêmes conditions de parcelle. L’interface Streamlit appelle une API FastAPI qui charge le modèle entraîné et son prétraitement.

[![CI Agritech API](https://github.com/ClementMouton/recommandation-agriculture-OCR_Projet_12/actions/workflows/ci.yml/badge.svg?branch=analyse-erreurs-regression)](https://github.com/ClementMouton/recommandation-agriculture-OCR_Projet_12/actions/workflows/ci.yml?query=branch%3Aanalyse-erreurs-regression)

## Reproduire la démo

**Prérequis** — Git, Python 3.11 et Docker Desktop démarré en mode conteneurs Linux. Les commandes ci-dessous sont prévues pour PowerShell.

### 1. Cloner le dépôt et reconstruire le modèle

```powershell
git clone --branch analyse-erreurs-regression https://github.com/ClementMouton/recommandation-agriculture-OCR_Projet_12.git
Set-Location .\recommandation-agriculture-OCR_Projet_12

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pipelines.build_serving_model
```

Le pipeline lit l’archive fournie dans `data/raw/`, nettoie et consolide les deux sources, entraîne la régression, l’évalue sur le jeu de test et enregistre le run dans MLflow.

| Sortie | Emplacement |
|---|---|
| Données consolidées | `data/processed/crop_yield_consolidated.csv` |
| Suivi des expériences | `mlflow.db` |
| Modèle exporté | Dossier indiqué par la ligne `Modèle exporté` dans le terminal |

### 2. Lancer l’API dans Docker

Dans le même terminal, remplace la valeur de `$modelPath` par le chemin **complet** affiché par le pipeline, vers le dossier qui contient le fichier `MLmodel`.

```powershell
$modelPath = "C:\chemin\vers\le\dossier\du\modele\exporte"
Test-Path -LiteralPath (Join-Path $modelPath "MLmodel")

docker build -f dockerfile -t agritech-api .

docker run --rm --name agritech-api `
  -p 127.0.0.1:8002:8000 `
  --mount "type=bind,source=$modelPath,target=/model,readonly" `
  -e YIELD_MODEL_URI=/model `
  agritech-api
```

`Test-Path` doit afficher `True`. Une fois l’API démarrée, laisse le terminal ouvert et ouvre [Swagger](http://127.0.0.1:8002/docs).

| Endpoint | Entrée | Résultat |
|---|---|---|
| `POST /predict` | Conditions de la parcelle et culture choisie | Rendement estimé en t/ha |
| `POST /recommend` | Conditions de la parcelle | Six cultures classées par rendement estimé décroissant |

### 3. Lancer l’interface Streamlit

Dans un second PowerShell, depuis la racine du dépôt :

```powershell
$env:AGRI_API_URL = "http://127.0.0.1:8002"
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

Ouvre [Streamlit](http://localhost:8501). Avec les valeurs par défaut du formulaire, le mode **Prédiction** estime le rendement du riz à environ **4,10 t/ha**. Le mode **Recommandation** affiche les six cultures et leurs rendements estimés. Les faibles écarts entre les cultures peuvent donner des valeurs identiques après arrondi.

## Données et choix de modélisation

Le jeu principal contient un million d’observations. Après exclusion de 231 rendements négatifs, **999 769 lignes** sont conservées. La seconde source fournit des observations historiques agrégées au niveau pays–culture–année. La jointure par culture enrichit **666 494 lignes** pour le maïs, le riz, le soja et le blé. Les observations d’orge et de coton sont conservées.

L’ACP répartit la variance presque également entre les quatre composantes. Dans le jeu principal, les précipitations, la fertilisation et l’irrigation présentent les associations les plus fortes avec le rendement parmi les variables étudiées. Le jeu historique présente des distributions différentes et ne reproduit pas la forte association pluie–rendement du jeu principal. Le [notebook de consolidation](notebooks/acp_consolidation.ipynb) détaille ces résultats et les différences de granularité.

La régression linéaire a été retenue après comparaison des modèles en validation croisée. Les profils historiques n’ont pas amélioré la MAE et restent disponibles dans le CSV consolidé. Le modèle servi a été entraîné sur **799 815 observations** et évalué sur **199 954 observations**.

| Métrique sur le jeu de test | Résultat |
|---|---:|
| MAE | 0,3983 t/ha |
| RMSE | 0,4993 t/ha |
| R² | 0,9132 |

### Portée de la recommandation

Le classement de l’application repose sur le rendement prédit. Une estimation de la rentabilité demanderait aussi les prix de vente et les coûts par culture. Sur le scénario de démo, les écarts entre cultures sont très faibles et le classement seul ne suffit pas pour guider un choix de plantation.

Une expérience complémentaire du notebook évalue le classement historique à l’échelle nationale. Sur **172 situations pays–année de 2010 à 2013**, pour quatre cultures et des pays déjà vus à l’entraînement, le modèle **pays + climat** retrouve la culture au meilleur rendement observé dans **127 cas (73,8 %)**. Le modèle **pays seul** atteint **124 cas (72,1 %)**. Cette expérience est présentée dans le notebook. L’API sert le modèle de simulation de parcelle.

## MLflow, tests et intégration continue

Pour consulter les résultats MLflow depuis la racine du dépôt :

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5002 --workers 1
```

Ouvre [MLflow](http://127.0.0.1:5002) et sélectionne l’expérience `agriculture_yield_regression_final`. Le run contient les métriques de test, les graphiques d’évaluation et les métadonnées du modèle.

Pour reproduire la comparaison des modèles avec validation croisée :

```powershell
.\.venv\Scripts\python.exe -m pipelines.prepare_data
.\.venv\Scripts\python.exe -m src.models.tune_regression
```

Les résultats de la comparaison sont enregistrés dans l’expérience `agriculture_yield_regression_tuning`.

Pour lancer les tests en local :

```powershell
.\.venv\Scripts\python.exe -m pip install pytest httpx
.\.venv\Scripts\python.exe -m pytest -q tests/
```

Le [workflow GitHub Actions](.github/workflows/ci.yml) s’exécute à chaque push et sur les pull requests vers `main`. Il lance les tests, puis construit l’image Docker si les tests passent. Les tests FastAPI utilisent un modèle simulé. Les tests Streamlit simulent les réponses de l’API et vérifient les deux modes de l’interface. Le chargement du modèle exporté et les deux endpoints ont également été vérifiés dans Docker en local.

## Organisation du dépôt

| Chemin | Rôle |
|---|---|
| `data/raw/` | Archive brute fournie avec le projet |
| `notebooks/` | Exploration, ACP, consolidation et expérience historique |
| `pipelines/build_serving_model.py` | Consolidation, entraînement final et export du modèle |
| `src/models/` | Comparaison, optimisation et entraînement |
| `api/main.py` | Endpoints de prédiction et de recommandation |
| `frontend/app.py` | Interface Streamlit |
| `tests/` | Tests automatisés de l’API et de l’interface |
| `.github/workflows/ci.yml` | Tests et build Docker sur GitHub Actions |