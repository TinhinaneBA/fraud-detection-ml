
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from collections import Counter


def load_data(path: str) -> pd.DataFrame:
    """Charge le dataset brut et valide sa structure."""
    df = pd.read_csv(path)
    assert 'Class' in df.columns, "Colonne Class manquante"
    assert df.isnull().sum().sum() == 0, "Valeurs nulles détectées"
    print(f" Dataset chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise Amount et Time avec RobustScaler."""
    scaler = RobustScaler()
    df = df.copy()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled']   = scaler.fit_transform(df[['Time']])
    df = df.drop(columns=['Time', 'Amount'])
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split stratifié train/test."""
    X = df.drop(columns=['Class'])
    y = df['Class']
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


def apply_smote(X_train, y_train, random_state: int = 42):
    """Applique SMOTE sur le train set uniquement."""
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f" SMOTE : {Counter(y_train)} → {Counter(y_res)}")
    return X_res, y_res
