# 🚗 Scoring de résiliation — Assurance Auto

Modèle de Machine Learning qui estime la probabilité qu'un client d'assurance automobile résilie son contrat, accompagné d'une interface web où un conseiller saisit un profil et obtient immédiatement son niveau de risque.

> Projet réalisé dans le cadre de la formation **Machine Learning & Data Science** — TP final Jour 3.

**🔗 Application en ligne :** https://scoring-resiliation.streamlit.app/

**📂 Dépôt :** https://github.com/bounyamine/scoring-resiliation

---

## Le problème

Le service Fidélisation perd environ 10 % de ses clients chaque année et les découvre une fois partis. L'objectif est de les détecter **avant** le départ, pour déclencher une action de rétention.

Le jeu de données couvre 500 clients décrits par 27 variables, dont 50 ont résilié (10 %).

## Le modèle

| | |
|---|---|
| Algorithme | Random Forest — 300 arbres, `max_depth=4`, `min_samples_leaf=10` |
| Déséquilibre | `class_weight='balanced'` (90 % / 10 %) |
| Prétraitement | `StandardScaler` (8 variables numériques) + `OneHotEncoder` (4 catégorielles), dans un `ColumnTransformer` |
| Validation croisée | AUC **0,801 ± 0,096** (5-fold, sur le train) |
| Test (100 clients) | Accuracy **0,870** · F1 **0,519** · **ROC-AUC 0,853** |
| Détection | **7 résiliations sur 10 détectées**, au prix de 10 fausses alertes |

Prétraitement et modèle sont réunis dans un **seul objet `Pipeline`** sérialisé en `.pkl` : l'application lui envoie des données brutes (du texte, des euros) et récupère une probabilité.

### Variables utilisées (12)

**Numériques** — Âge · Salaire Annuel (€) · Prime Annuelle (€) · Ancienneté (mois) · Coeff. Bonus-Malus · Nb Sinistres (3 ans) · Montant Sinistres (€) · Score Risque (0-100)

**Catégorielles** — Type Contrat · Catégorie Prof. · Usage Véhicule · Dernier Sinistre

### ⚠️ Fuite de données écartée

La colonne **`Statut Contrat`** (`Actif` / `Résilié` / `Suspendu`) révèle la cible à 100 %. Elle n'est renseignée qu'**après** le départ du client : un modèle l'utilisant afficherait un score parfait en test et serait inutilisable en production. Elle est exclue, comme les identifiants, le code postal et les dates brutes.

`Sexe` est également écartée : la différenciation tarifaire sur le sexe est interdite en assurance dans l'UE.

---

## Installation et lancement

```bash
# 1. Environnement
conda activate formation_ml
pip install -r requirements.txt

# 2. Entraîner le modèle (crée models/pipeline_resiliation.pkl et models/metadata.json)
python train_model.py

# 3. Lancer l'interface
streamlit run app.py
```

L'application s'ouvre sur http://localhost:8501.

> **L'étape 2 est obligatoire avant la première exécution** : le `.pkl` doit être généré avec **votre** version de scikit-learn, sinon `joblib.load` lève un avertissement de version, voire une erreur. `train_model.py` régénère aussi `requirements.txt` avec les versions réellement installées chez vous.

## Structure du projet

```
scoring_resiliation/
│
├── data/
│   └── dataset_assurance_ML.xlsx    données brutes (500 × 27)
├── models/
│   ├── pipeline_resiliation.pkl     pipeline complet sérialisé (~494 Ko)
│   └── metadata.json                bornes des curseurs, modalités des menus, métriques
├── notebooks/
│   └── tp_final.ipynb               exploration et démarche (parties A à C)
├── .streamlit/
│   └── config.toml                  thème de l'application
├── train_model.py                   entraînement + sauvegarde, en une commande
├── app.py                           interface Streamlit
├── requirements.txt                 versions figées
└── README.md
```

## L'interface

- **8 curseurs et 4 menus** dans la barre latérale, bornés par les valeurs réelles du dataset et alimentés par `metadata.json` — un ré-entraînement met l'interface à jour sans modifier `app.py`
- **Probabilité de résiliation** avec jauge et message coloré selon deux seuils
- **Curseur « Seuil d'alerte »** (0,30 à 0,70) : le conseiller règle lui-même la sensibilité selon sa capacité d'appels
- **Positionnement dans le portefeuille** : « ce client est plus risqué que X % des clients »
- **Les 8 variables les plus influentes** du modèle, en barres
- **Onglet scoring par lot** : import d'un CSV, scoring de toutes les lignes, export du fichier enrichi

### Trois profils de référence

| Profil | Probabilité |
|---|---|
| Profil médian, aucun sinistre | 12 % |
| Client Gold, 300 mois d'ancienneté, 0 sinistre | 11 % |
| Jeune conducteur Bronze, 2 sinistres, vol, malus 1,20 | 79 % |
| Tous les curseurs de risque au maximum | 83 % |

Cohérent avec l'analyse : la **sinistralité** (nb de sinistres, score de risque, bonus-malus, nature du dernier sinistre) est le moteur de la résiliation. L'âge, le salaire et la prime, eux, ne déplacent presque pas l'aiguille.
