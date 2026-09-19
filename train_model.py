"""
Scoring de résiliation — Assurance Auto
Script d'entraînement : reconstruit le pipeline, l'évalue et sauvegarde
    models/pipeline_resiliation.pkl
    models/metadata.json
    requirements.txt   (versions réellement installées ici)

Usage :
    python train_model.py
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
RACINE = Path(__file__).resolve().parent          # chemins relatifs au script
DATA = RACINE / 'data'
MODELS = RACINE / 'models'
MODELS.mkdir(exist_ok=True)

TARGET = 'Résiliation'
RANDOM_STATE = 42

NUM_COLS = ['Âge', 'Salaire Annuel (€)', 'Prime Annuelle (€)', 'Ancienneté (mois)',
            'Coeff. Bonus-Malus', 'Nb Sinistres (3 ans)',
            'Montant Sinistres (€)', 'Score Risque (0-100)']

CAT_COLS = ['Type Contrat', 'Catégorie Prof.', 'Usage Véhicule', 'Dernier Sinistre']

# Variables volontairement exclues, et pourquoi
EXCLUES = {
    'Statut Contrat': "FUITE DE DONNÉES — révèle la cible à 100 %",
    'N° Police / Nom / Prénom': "identifiants, aucun pouvoir prédictif",
    'Code Postal / Ville': "géographie non prédictive ici, et 10 modalités à saisir",
    'Dates brutes': "information déjà portée par Ancienneté (mois)",
}


# --------------------------------------------------------------------------
# 1. Chargement
# --------------------------------------------------------------------------
def charger_donnees():
    xlsx = DATA / 'dataset_assurance_ML.xlsx'
    csv = DATA / 'dataset_assurance_ML.csv'

    df = pd.read_excel(xlsx)
    df.to_csv(csv, index=False, encoding='utf-8-sig')
    df = pd.read_csv(csv, encoding='utf-8-sig')

    print(f'Dataset          : {df.shape[0]} lignes x {df.shape[1]} colonnes')
    print(f'Valeurs manquantes : {df.isnull().sum().sum()} | doublons : {df.duplicated().sum()}')
    print(f'Taux de résiliation : {df[TARGET].mean():.1%} ({int(df[TARGET].sum())} clients)')
    return df


# --------------------------------------------------------------------------
# 2. Pipeline
# --------------------------------------------------------------------------
def construire_pipeline():
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), NUM_COLS),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_COLS),
    ])
    modele = RandomForestClassifier(
        n_estimators=300, max_depth=4, min_samples_leaf=10,
        class_weight='balanced', random_state=RANDOM_STATE)
    return Pipeline([('prep', preprocessor), ('model', modele)])


def comparer_candidats(X_train, y_train, preprocessor):
    candidats = {
        'Régression Logistique': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=4, min_samples_leaf=10,
            class_weight='balanced', random_state=RANDOM_STATE),
    }
    print('\nValidation croisée 5-fold (ROC-AUC, sur le train uniquement)')
    for nom, algo in candidats.items():
        pipe = Pipeline([('prep', preprocessor), ('model', algo)])
        s = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc')
        print(f'  {nom:24s} AUC = {s.mean():.3f} ± {s.std():.3f}')


# --------------------------------------------------------------------------
# 3. Entraînement principal
# --------------------------------------------------------------------------
def main():
    df = charger_donnees()

    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print(f'\nSplit : train {X_train.shape} | test {X_test.shape}')
    print(f'Taux de résiliation : train {y_train.mean():.2f} | test {y_test.mean():.2f}')

    pipeline = construire_pipeline()
    comparer_candidats(X_train, y_train, pipeline.named_steps['prep'])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    print('\nÉvaluation sur le test')
    print(f'  Accuracy : {accuracy_score(y_test, y_pred):.3f}')
    print(f'  F1       : {f1_score(y_test, y_pred):.3f}')
    print(f'  ROC-AUC  : {auc:.3f}')
    print(f'  Matrice de confusion :\n{confusion_matrix(y_test, y_pred)}')
    print(classification_report(y_test, y_pred, target_names=['Reste', 'Résilie']))

    # ----------------------------------------------------------------------
    # 4. Sauvegarde du pipeline
    # ----------------------------------------------------------------------
    chemin_pkl = MODELS / 'pipeline_resiliation.pkl'
    joblib.dump(pipeline, chemin_pkl)
    print(f'Modèle sauvegardé : {chemin_pkl} ({os.path.getsize(chemin_pkl) / 1024:.0f} Ko)')

    # ----------------------------------------------------------------------
    # 5. Métadonnées pour l'interface
    # ----------------------------------------------------------------------
    # Distribution des probabilités du portefeuille : permet à l'app de situer
    # un client par rapport aux autres ("plus risqué que X % du portefeuille").
    proba_portefeuille = pipeline.predict_proba(X)[:, 1]

    meta = {
        'modele': 'Random Forest',
        'auc_test': round(float(auc), 3),
        'f1_test': round(float(f1_score(y_test, y_pred)), 3),
        'taux_resiliation': round(float(y.mean()), 3),
        'num_cols': NUM_COLS,
        'cat_cols': CAT_COLS,
        'num_ranges': {c: {'min': float(X[c].min()),
                           'max': float(X[c].max()),
                           'median': float(X[c].median())} for c in NUM_COLS},
        'cat_values': {c: sorted(X[c].unique().tolist()) for c in CAT_COLS},
        'proba_portefeuille': [round(float(p), 4) for p in proba_portefeuille],
        'variables_exclues': EXCLUES,
    }

    chemin_json = MODELS / 'metadata.json'
    with open(chemin_json, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f'Métadonnées sauvegardées : {chemin_json}')

    # ----------------------------------------------------------------------
    # 6. requirements.txt figé sur les versions réellement installées
    # ----------------------------------------------------------------------
    ecrire_requirements()

    # ----------------------------------------------------------------------
    # 7. Test de bout en bout : rechargement + prédiction sur un client brut
    # ----------------------------------------------------------------------
    modele = joblib.load(chemin_pkl)
    client = pd.DataFrame([{
        'Âge': 34, 'Salaire Annuel (€)': 28000, 'Prime Annuelle (€)': 950,
        'Ancienneté (mois)': 6, 'Coeff. Bonus-Malus': 1.25, 'Nb Sinistres (3 ans)': 3,
        'Montant Sinistres (€)': 4200, 'Score Risque (0-100)': 72,
        'Type Contrat': 'Bronze', 'Catégorie Prof.': 'Entrepreneur',
        'Usage Véhicule': 'Professionnel', 'Dernier Sinistre': 'Vol',
    }])
    print(f"\nTest de rechargement — client à risque : "
          f"classe {modele.predict(client)[0]}, "
          f"probabilité {modele.predict_proba(client)[0, 1]:.1%}")
    print('\nTerminé. Lancez l\'interface avec :  streamlit run app.py')


def ecrire_requirements():
    """Fige les versions installées dans requirements.txt."""
    import importlib.metadata as md
    paquets = ['streamlit', 'scikit-learn', 'pandas', 'numpy', 'joblib', 'openpyxl']
    lignes = []
    for p in paquets:
        try:
            lignes.append(f'{p}=={md.version(p)}')
        except md.PackageNotFoundError:
            print(f'  (paquet {p} introuvable, ignoré)')
    (RACINE / 'requirements.txt').write_text('\n'.join(lignes) + '\n', encoding='utf-8')
    print('requirements.txt écrit :', ', '.join(lignes))


if __name__ == '__main__':
    main()
