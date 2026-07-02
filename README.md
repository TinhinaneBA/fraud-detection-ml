#  Fraud Detection ML — End-to-End Pipeline

> Système de détection de fraude financière basé sur le machine learning.  
> Pipeline complet : EDA → Preprocessing → Modélisation → Interprétabilité → Déploiement.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-green)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red)
![MLflow](https://img.shields.io/badge/MLflow-tracking-blue)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

---

##  Objectif

Détecter automatiquement les transactions bancaires frauduleuses à partir de
284 807 transactions réelles anonymisées, en gérant le défi du **déséquilibre
extrême des classes** (0.17% de fraudes).

---

##  Structure du projet

```
fraud-detection-ml/
├── data/
│   ├── raw/                    ← Dataset Kaggle (non versionné)
│   └── processed/              ← Données après preprocessing
├── notebooks/
│   ├── 01_eda.ipynb            ← Analyse exploratoire complète
│   ├── 02_preprocessing.ipynb  ← Nettoyage, SMOTE, Pipeline
│   ├── 03_modeling.ipynb       ← Entraînement et comparaison de modèles
│   ├── 04_evaluation.ipynb     ← Métriques, courbes, SHAP
│   └── 05_mlflow.ipynb         ← Tracking des expériences
├── src/
│   ├── data/
│   │   ├── make_dataset.py     ← Chargement et validation
│   │   └── preprocessing.py   ← Pipeline de transformation
│   ├── models/
│   │   ├── train.py            ← Entraînement
│   │   ├── predict.py          ← Inférence
│   │   └── evaluate.py        ← Métriques
│   └── visualization/
│       └── plots.py            ← Fonctions de visualisation
├── app/
│   └── streamlit_app.py        ← Application interactive
├── models/                     ← Modèles sauvegardés
├── reports/
│   ├── figures/                ← Graphiques exportés
│   └── rapport_final.md        ← Synthèse métier
├── requirements.txt
└── README.md
```

---

##  Dataset

- **Source :** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Volume :** 284 807 transactions · Septembre 2013 · Europe
- **Features :** 28 composantes PCA anonymisées (V1–V28) + Time + Amount
- **Déséquilibre :** 492 fraudes (0.17%) / 284 315 légitimes (99.83%)

>  Le dataset n'est pas versionné (trop lourd).  
> Télécharger `creditcard.csv` depuis Kaggle et le placer dans `data/raw/`.

---

##  Pipeline ML

```
EDA → Feature Engineering → SMOTE → Train/Test Split
    → Logistic Regression (baseline)
    → Random Forest
    → XGBoost          ← modèle principal
    → LightGBM
    → Optimisation du seuil de décision
    → SHAP (interprétabilité globale + locale)
    → Déploiement Streamlit
```

---

##  Métriques utilisées

| Métrique | Pourquoi |
|---|---|
| **AUC-ROC** | Performance globale du modèle |
| **Precision-Recall AUC** | Adaptée aux classes déséquilibrées |
| **F1-Score** | Équilibre précision / rappel |
| **Recall (fraudes)** | Minimiser les faux négatifs |
| **Matrice de confusion** | Vision complète des erreurs |

> En détection de fraude, un **faux négatif** (fraude non détectée) coûte
> plus cher qu'un **faux positif** (transaction légitime bloquée).
> Le Recall sur la classe fraude est donc la métrique prioritaire.

---

## 🛠️ Stack technique

| Catégorie | Outils |
|---|---|
| Manipulation | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| ML | Scikit-learn, XGBoost, LightGBM |
| Déséquilibre | Imbalanced-learn (SMOTE) |
| Interprétabilité | SHAP |
| Tracking | MLflow |
| Déploiement | Streamlit |

---

##  Lancer le projet

```bash
# Cloner le repo
git clone https://github.com/TinhinaneBA/fraud-detection-ml.git
cd fraud-detection-ml

# Créer l'environnement virtuel
python -m venv venv
venv\Scripts\Activate.ps1       # Windows
# source venv/bin/activate       # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt

# Télécharger le dataset
# → https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# → Placer creditcard.csv dans data/raw/

# Lancer Jupyter
jupyter notebook

# Lancer l'application Streamlit (après modélisation)
streamlit run app/streamlit_app.py
```

---

## 👩 Auteure

**Tinhinane B.** — Étudiante M2 IWOCS · Université Le Havre Normandie  
Orientée Data Science & Machine Learning  
[GitHub](https://github.com/TinhinaneBA)

---

*Projet réalisé dans le cadre d'un portfolio Data Science professionnel — 2025*