"""
Fraud Detection - Streamlit Application
Auteure : TinhinaneBA
Modele  : XGBoost - seuil optimise 0.94
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Fraud Detection - TinhinaneBA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .fraud-alert {
        background: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .safe-alert {
        background: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgboost_fraud_detector.joblib')
    meta_path  = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_metadata.json')
    model = joblib.load(model_path)
    with open(meta_path) as f:
        metadata = json.load(f)
    return model, metadata


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


try:
    model, metadata = load_model()
    explainer = load_explainer(model)
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"Erreur de chargement du modele : {e}")


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Fraud Detection")
    st.markdown("**Auteure :** TinhinaneBA")
    st.markdown("**M2 IWOCS** - Universite Le Havre Normandie")
    st.divider()

    st.markdown("### Metriques du modele")
    if MODEL_LOADED:
        st.metric("AUC-ROC",       f"{metadata['auc_roc']:.4f}")
        st.metric("PR-AUC",        f"{metadata['pr_auc']:.4f}")
        st.metric("Recall Fraude", f"{metadata['recall_fraud']:.4f}")
        st.metric("Precision",     f"{metadata['precision_fraud']:.4f}")
        st.metric("F1-Score",      f"{metadata['f1_score']:.4f}")
        st.metric("Seuil retenu",  f"{metadata['threshold']:.2f}")
    st.divider()

    st.markdown("### Top features SHAP")
    if MODEL_LOADED:
        for i, feat in enumerate(metadata['top_shap_features'], 1):
            st.markdown(f"**{i}.** {feat}")
    st.divider()

    st.markdown("### Projet")
    st.markdown("[GitHub](https://github.com/TinhinaneBA/fraud-detection-ml)")
    st.markdown("**Dataset :** Credit Card Fraud - Kaggle")
    st.markdown("**Modele :** XGBoost - seuil 0.94")


# ── HEADER ────────────────────────────────────────────────────
st.markdown('<div class="main-header">🔍 Systeme de Detection de Fraude</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Pipeline ML end-to-end · XGBoost · SHAP · MLflow · Portfolio Data Science</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎯 Prediction en temps reel", "📊 Performance du modele", "📖 A propos du projet"])


# ══════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Saisie d'une transaction")
    st.markdown("Ajustez les parametres de la transaction pour obtenir une prediction en temps reel.")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Variables PCA (V1 - V15)")
        features = {}
        cols = st.columns(3)
        for i in range(1, 16):
            with cols[(i - 1) % 3]:
                features[f'V{i}'] = st.number_input(f'V{i}', value=0.0, min_value=-10.0, max_value=10.0, step=0.1, format="%.2f", key=f'v{i}')

    with col_right:
        st.markdown("#### Variables PCA (V16 - V28)")
        cols2 = st.columns(3)
        for i in range(16, 29):
            with cols2[(i - 16) % 3]:
                features[f'V{i}'] = st.number_input(f'V{i}', value=0.0, min_value=-10.0, max_value=10.0, step=0.1, format="%.2f", key=f'v{i}')

        st.markdown("#### Montant & Temps")
        col_a, col_b = st.columns(2)
        with col_a:
            amount = st.number_input("Montant (EUR)", value=50.0, min_value=0.0, max_value=25000.0, step=1.0)
        with col_b:
            time_val = st.number_input("Temps (s)", value=50000.0, min_value=0.0, max_value=172800.0, step=100.0)

        features['Amount_scaled'] = (amount - 22.0) / 77.0
        features['Time_scaled']   = (time_val - 84692.0) / 84975.0

    st.divider()
    predict_col, _, info_col = st.columns([1, 0.2, 2])

    with predict_col:
        predict_btn = st.button("🔍 Analyser la transaction", type="primary", use_container_width=True)

    with info_col:
        if MODEL_LOADED:
            st.info(f"Seuil de decision retenu : **{metadata['threshold']:.2f}** (optimise par F1-Score)")

    if predict_btn and MODEL_LOADED:
        input_df   = pd.DataFrame([features])[metadata['features']]
        proba      = model.predict_proba(input_df)[0][1]
        prediction = int(proba >= metadata['threshold'])

        st.divider()
        res1, res2, res3 = st.columns(3)

        with res1:
            if prediction == 1:
                st.markdown('<div class="fraud-alert"><h2>FRAUDE DETECTEE</h2><p>Transaction suspecte.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="safe-alert"><h2>TRANSACTION LEGITIME</h2><p>Transaction normale.</p></div>', unsafe_allow_html=True)

        with res2:
            st.metric("Probabilite de fraude", f"{proba*100:.2f}%", delta=f"Seuil : {metadata['threshold']*100:.0f}%", delta_color="inverse")
            fig_g, ax_g = plt.subplots(figsize=(4, 0.5))
            ax_g.barh(0, proba, color='#F44336' if prediction == 1 else '#4CAF50', height=0.4)
            ax_g.barh(0, 1 - proba, left=proba, color='#e0e0e0', height=0.4)
            ax_g.axvline(metadata['threshold'], color='black', linestyle='--', linewidth=2)
            ax_g.set_xlim([0, 1])
            ax_g.axis('off')
            st.pyplot(fig_g)
            plt.close()

        with res3:
            st.markdown("**Interpretation SHAP**")
            shap_vals = explainer.shap_values(input_df)
            shap_df = pd.DataFrame({
                'Feature': metadata['features'],
                'SHAP':    shap_vals[0]
            }).sort_values('SHAP', key=abs, ascending=False).head(8)

            fig_s, ax_s = plt.subplots(figsize=(5, 4))
            colors_s = ['#F44336' if v > 0 else '#4CAF50' for v in shap_df['SHAP']]
            ax_s.barh(shap_df['Feature'], shap_df['SHAP'], color=colors_s, edgecolor='white')
            ax_s.axvline(0, color='black', linewidth=0.8)
            ax_s.set_title('Impact SHAP - Top 8 features', fontsize=10, fontweight='bold')
            ax_s.set_xlabel('Valeur SHAP')
            plt.tight_layout()
            st.pyplot(fig_s)
            plt.close()

        with st.expander("Detail complet de la prediction"):
            d1, d2 = st.columns(2)
            with d1:
                st.markdown(f"""
| Parametre | Valeur |
|---|---|
| Probabilite fraude | `{proba:.6f}` |
| Seuil de decision | `{metadata['threshold']}` |
| Decision | `{'FRAUDE' if prediction == 1 else 'LEGITIME'}` |
| Modele | `{metadata['model']}` |
| Nombre de features | `{metadata['n_features']}` |
                """)
            with d2:
                st.markdown("**Top 5 features SHAP pour cette transaction :**")
                for _, row in shap_df.head(5).iterrows():
                    direction = "🔴 +" if row['SHAP'] > 0 else "🟢 "
                    st.markdown(f"{direction} **{row['Feature']}** : `{row['SHAP']:.4f}`")


# ══════════════════════════════════════════════════════════════
# TAB 2 — PERFORMANCE
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Performance du modele XGBoost")

    if MODEL_LOADED:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("AUC-ROC",       f"{metadata['auc_roc']:.4f}")
        m2.metric("PR-AUC",        f"{metadata['pr_auc']:.4f}")
        m3.metric("Recall Fraude", f"{metadata['recall_fraud']:.4f}")
        m4.metric("Precision",     f"{metadata['precision_fraud']:.4f}")
        m5.metric("F1-Score",      f"{metadata['f1_score']:.4f}")
        st.divider()

        st.markdown("#### Comparaison des modeles - Runs MLflow")
        df_compare = pd.DataFrame({
            'Modele':         ['Logistic Regression', 'LightGBM', 'XGBoost (seuil 0.50)', 'XGBoost PRODUCTION'],
            'PR-AUC':         [0.7235, 0.8566, 0.8644, 0.8644],
            'Recall Fraude':  [0.9184, 0.8673, 0.8571, 0.8163],
            'F1-Score':       [0.1110, 0.7143, 0.7179, 0.8649],
            'Faux Positifs':  [1434,   55,     52,     7],
            'Temps (s)':      [31.8,   128.1,  239.1,  244.9],
            'Statut':         ['Baseline', 'Candidat', 'Candidat', 'PRODUCTION']
        })

        st.dataframe(
            df_compare.style
            .highlight_max(subset=['PR-AUC', 'F1-Score'], color='#d4edda')
            .highlight_min(subset=['Faux Positifs'], color='#d4edda'),
            use_container_width=True,
            hide_index=True
        )

        st.divider()
        st.markdown("#### Visualisation des performances")
        fig_c, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors_c = ['#90CAF9', '#64B5F6', '#42A5F5', '#1565C0']

        for ax, metric, title in zip(axes,
            ['PR-AUC', 'F1-Score', 'Faux Positifs'],
            ['PR-AUC', 'F1-Score', 'Faux Positifs (moins = mieux)']):
            bars = ax.barh(df_compare['Modele'], df_compare[metric], color=colors_c, edgecolor='white')
            ax.set_title(title, fontweight='bold', fontsize=11)
            for bar, val in zip(bars, df_compare[metric]):
                ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2, f'{val}', va='center', fontsize=9)
            ax.tick_params(axis='y', labelsize=9)

        plt.suptitle('Comparaison des modeles - Fraud Detection', fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_c)
        plt.close()

        st.divider()
        st.markdown("#### Decision finale documentee")
        st.info("""
**Modele retenu : XGBoost avec seuil 0.94**

Random Forest obtient un PR-AUC marginalement superieur (+0.36%) mais necessite 10 heures
d'entrainement vs 40 secondes pour XGBoost — inexploitable en production.

Le seuil 0.94 (vs defaut 0.50) reduit les faux positifs de **85%** (52 -> 7)
au prix de 4 fraudes supplementaires manquees. Ce compromis est justifie car
bloquer 52 clients legitimes genere une friction et un cout reputationnel non justifie.
        """)


# ══════════════════════════════════════════════════════════════
# TAB 3 — A PROPOS
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### A propos du projet")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
#### Objectif
Detecter automatiquement les transactions bancaires frauduleuses
a partir de 284 807 transactions reelles anonymisees.

#### Dataset
- **Source :** Credit Card Fraud Detection - Kaggle (ULB)
- **Volume :** 284 807 transactions - Septembre 2013
- **Features :** 28 composantes PCA + Time + Amount
- **Desequilibre :** 0.17% de fraudes

#### Pipeline ML
- EDA et exploration
- Preprocessing : RobustScaler + SMOTE
- Modelisation : comparaison de 4 modeles
- Evaluation : threshold tuning + SHAP
- Tracking : MLflow
- Deploiement : Streamlit

#### Stack technique
| Categorie | Outils |
|---|---|
| ML | XGBoost, LightGBM, sklearn |
| Desequilibre | SMOTE (imbalanced-learn) |
| Interpretabilite | SHAP |
| Tracking | MLflow |
| Deploiement | Streamlit |
        """)

    with col2:
        st.markdown("""
#### Resultats cles
- **AUC-ROC :** 0.9802
- **Seuil optimise :** 0.94 (vs 0.50 par defaut)
- **Faux positifs reduits de 85%** (52 -> 7)
- **Precision :** 91.95%

#### Structure du projet
- notebooks/ : 5 notebooks Jupyter
- src/ : modules Python reutilisables
- app/ : application Streamlit
- models/ : modele + metadonnees JSON
- reports/ : figures et rapport final

#### Auteure
**Tinhinane B.**
M2 IWOCS - Universite Le Havre Normandie
Portfolio Data Science - 2025

[GitHub TinhinaneBA](https://github.com/TinhinaneBA/fraud-detection-ml)
        """)

    st.divider()
    st.markdown("""
#### Notebooks du projet
| Notebook | Contenu |
|---|---|
| 01_eda.ipynb | Analyse exploratoire |
| 02_preprocessing.ipynb | RobustScaler, SMOTE, split stratifie |
| 03_modeling.ipynb | Comparaison 4 modeles, decision documentee |
| 04_evaluation.ipynb | Threshold tuning, SHAP global et local |
| 05_mlflow.ipynb | Tracking complet des experiences |
    """)