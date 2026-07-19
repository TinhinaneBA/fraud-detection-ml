# Fraud Detection ML — End-to-End Pipeline

> Systeme de detection de fraude financiere base sur le machine learning.
> Pipeline complet : EDA → Preprocessing → Modelisation → Interpretabilite → Deploiement.

![Python](https://img.shields.io/badge/Python-3.14-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Production-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![MLflow](https://img.shields.io/badge/MLflow-Tracked-blue)
![SHAP](https://img.shields.io/badge/SHAP-Interpretability-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 🚀 Application en ligne

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fraud-detection-ml-gwdnjvzfqok3rcwtcmx44f.streamlit.app/)

**Tester l'app :** https://fraud-detection-ml-gwdnjvzfqok3rcwtcmx44f.streamlit.app/

---

## Objectif

Detecter automatiquement les transactions bancaires frauduleuses a partir de
284 807 transactions reelles anonymisees, en gerant le defi du desequilibre
extreme des classes (0.17% de fraudes).

---

## Resultats cles

| Metrique | Valeur | Signification |
|---|---|---|
| AUC-ROC | 0.9802 | Excellente separation fraude/legitime |
| PR-AUC | 0.8644 | Performance adaptee aux classes desequilibrees |
| F1-Score | 0.8649 | Meilleur equilibre precision/rappel |
| Precision | 91.95% | Fiabilite des alertes declenchees |
| Faux Positifs | 7 | Clients legitimes bloques (vs 52 avec seuil defaut) |

---

## Pipeline ML

```
EDA → Preprocessing (RobustScaler + SMOTE)
    → Modelisation (4 modeles compares)
    → Evaluation (Threshold Tuning + SHAP)
    → Tracking (MLflow — 4 runs enregistres)
    → Deploiement (Streamlit)
```

---

## Structure du projet

```
fraud-detection-ml/
├── data/
│   ├── raw/                    <- Dataset Kaggle (non versionne)
│   └── processed/              <- Donnees apres preprocessing (non versionne)
├── notebooks/
│   ├── 01_eda.ipynb            <- Analyse exploratoire complete
│   ├── 02_preprocessing.ipynb  <- Nettoyage, SMOTE, Pipeline
│   ├── 03_modeling.ipynb       <- Entrainement et comparaison de modeles
│   ├── 04_evaluation.ipynb     <- Metriques, threshold tuning, SHAP
│   └── 05_mlflow.ipynb         <- Tracking des experiences
├── src/
│   └── data/
│       └── preprocessing.py    <- Fonctions reutilisables
├── app/
│   └── streamlit_app.py        <- Application interactive
├── models/
│   └── model_metadata.json     <- Metriques et parametres du modele
├── reports/
│   └── figures/                <- 17 graphiques exportes
├── requirements.txt
└── README.md
```

---

## Dataset

- **Source :** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Volume :** 284 807 transactions — Septembre 2013 — Europe
- **Features :** 28 composantes PCA anonymisees (V1-V28) + Time + Amount
- **Desequilibre :** 492 fraudes (0.17%) / 284 315 legitimes (99.83%)

Le dataset n'est pas versionne (trop lourd).
Telecharger `creditcard.csv` depuis Kaggle et le placer dans `data/raw/`.

---

## Decisions techniques documentees

### Choix du modele : XGBoost vs Random Forest

| Critere | Random Forest | XGBoost |
|---|---|---|
| PR-AUC | 0.8680 | 0.8644 |
| Temps d'entrainement | 37 087s (~10h) | 40s |
| Deployable en production | Non | Oui |

**Decision : XGBoost retenu.**
Difference PR-AUC negligeable (+0.36%) vs cout computationnel 927x superieur.

### Optimisation du seuil de decision

| Seuil | Fraudes detectees | Faux Positifs | F1 |
|---|---|---|---|
| 0.50 (defaut) | 84 | 52 | 0.7179 |
| **0.94 (optimal)** | **80** | **7** | **0.8649** |

**Decision : seuil 0.94.**
Reduction des faux positifs de 85% (52→7) pour seulement 4 fraudes manquees supplementaires.

---

## Stack technique

| Categorie | Outils |
|---|---|
| Manipulation | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| ML | Scikit-learn, XGBoost, LightGBM |
| Desequilibre | Imbalanced-learn (SMOTE) |
| Interpretabilite | SHAP |
| Tracking | MLflow 2.17.2 |
| Deploiement | Streamlit |

---