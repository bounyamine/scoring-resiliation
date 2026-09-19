"""
Scoring de résiliation — Assurance Auto
Interface Streamlit : le conseiller saisit un profil client et obtient son risque.

Lancement local :  streamlit run app.py
Prérequis       :  python train_model.py  (crée models/pipeline_resiliation.pkl et metadata.json)
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# st.set_page_config DOIT être le premier appel Streamlit du script
st.set_page_config(page_title='Scoring Résiliation', page_icon='🚗', layout='wide')

RACINE = Path(__file__).resolve().parent          # chemins relatifs : indispensable en ligne


# --------------------------------------------------------------------------
# Chargement du modèle (une seule fois, pas à chaque interaction)
# --------------------------------------------------------------------------
@st.cache_resource
def charger_modele():
    pipeline = joblib.load(RACINE / 'models' / 'pipeline_resiliation.pkl')
    with open(RACINE / 'models' / 'metadata.json', encoding='utf-8') as f:
        meta = json.load(f)
    return pipeline, meta


try:
    pipeline, meta = charger_modele()
except FileNotFoundError:
    st.error("Modèle introuvable. Lancez d'abord `python train_model.py` "
             "puis vérifiez que `models/` est bien présent dans le dépôt.")
    st.stop()

num_cols, cat_cols = meta['num_cols'], meta['cat_cols']
rng, cats = meta['num_ranges'], meta['cat_values']
proba_portefeuille = pd.Series(meta.get('proba_portefeuille', []))

st.title('🚗 Scoring de résiliation — Assurance Auto')
st.caption(f"Modèle : {meta['modele']} · AUC test : {meta['auc_test']} · "
           f"{len(num_cols) + len(cat_cols)} variables d'entrée")


# --------------------------------------------------------------------------
# Barre latérale : profil du client
# --------------------------------------------------------------------------
st.sidebar.header('👤 Profil du client')


def curseur(col, step=1.0, fmt=None):
    """Slider borné par les min/max du dataset, positionné sur la médiane."""
    r = rng[col]
    val = st.sidebar.slider(col, min_value=r['min'], max_value=r['max'],
                            value=r['median'], step=step, format=fmt)
    return int(val) if fmt == '%d' else val


client = {}
client['Âge'] = curseur('Âge', 1.0, '%d')
client['Salaire Annuel (€)'] = curseur('Salaire Annuel (€)', 500.0, '%d')
client['Prime Annuelle (€)'] = curseur('Prime Annuelle (€)', 10.0, '%d')
client['Ancienneté (mois)'] = curseur('Ancienneté (mois)', 1.0, '%d')
client['Coeff. Bonus-Malus'] = curseur('Coeff. Bonus-Malus', 0.01, '%.2f')
client['Nb Sinistres (3 ans)'] = curseur('Nb Sinistres (3 ans)', 1.0, '%d')
client['Montant Sinistres (€)'] = curseur('Montant Sinistres (€)', 100.0, '%d')
client['Score Risque (0-100)'] = curseur('Score Risque (0-100)', 1.0, '%d')

st.sidebar.markdown('---')
for col in cat_cols:
    # Les modalités viennent du metadata : un ré-entraînement avec de nouvelles
    # catégories met l'interface à jour sans toucher à app.py.
    client[col] = st.sidebar.selectbox(col, cats[col])

st.sidebar.markdown('---')
st.sidebar.subheader('⚙️ Sensibilité')
SEUIL_RISQUE = st.sidebar.slider(
    "Seuil d'alerte", min_value=0.30, max_value=0.70, value=0.55, step=0.01,
    help="Abaisser le seuil = détecter plus de départs, au prix de plus de fausses alertes.")
SEUIL_MODERE = round(SEUIL_RISQUE - 0.15, 2)


# --------------------------------------------------------------------------
# Onglets : scoring individuel / scoring par lot
# --------------------------------------------------------------------------
onglet_client, onglet_lot = st.tabs(['🔮 Scoring individuel', '📁 Scoring par lot'])

with onglet_client:
    st.write('Renseignez le profil du client dans le panneau de gauche, '
             'puis cliquez sur **Prédire** pour estimer son risque de résiliation.')

    if st.button('🔮 Prédire', type='primary', width='stretch'):
        # L'ordre des colonnes doit être EXACTEMENT celui de l'entraînement
        df_client = pd.DataFrame([client])[num_cols + cat_cols]
        proba = float(pipeline.predict_proba(df_client)[0, 1])

        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric('Probabilité de résiliation', f'{proba:.0%}')
            if proba >= SEUIL_RISQUE:
                st.error('⚠️ Client À RISQUE — action de rétention conseillée')
            elif proba >= SEUIL_MODERE:
                st.warning('🟠 Risque modéré — à surveiller')
            else:
                st.success('✅ Client fidèle — risque faible')

            if not proba_portefeuille.empty:
                rang = (proba_portefeuille < proba).mean()
                st.caption(f'Ce client est plus risqué que **{rang:.0%}** du portefeuille.')

        with col2:
            st.write('Niveau de risque')
            st.progress(proba)
            st.write('Données envoyées au modèle :')
            # .astype(str) : une colonne mêlant nombres et texte casse l'affichage
            st.dataframe(df_client.T.astype(str).rename(columns={0: 'Valeur'}),
                         width='stretch')

        # Explication du modèle
        model = pipeline.named_steps['model']
        if hasattr(model, 'feature_importances_'):
            noms = pipeline.named_steps['prep'].get_feature_names_out()
            imp = (pd.Series(model.feature_importances_, index=noms)
                     .sort_values(ascending=False).head(8))
            imp.index = [n.split('__', 1)[1] for n in imp.index]
            st.subheader('📊 Les 8 variables les plus influentes du modèle')
            st.bar_chart(imp)
            st.caption('Importances globales du modèle — elles expliquent sur quoi il '
                       "s'appuie en général, pas la décision propre à ce client.")
    else:
        st.info('👈 Ajustez le profil dans la barre latérale, puis cliquez sur Prédire.')


with onglet_lot:
    st.write('Chargez un CSV contenant une ligne par client et les 12 colonnes du modèle. '
             'Le fichier enrichi d\'une colonne **Probabilité** est téléchargeable ensuite.')

    with st.expander('Colonnes attendues (noms exacts)'):
        st.code('\n'.join(num_cols + cat_cols))

    fichier = st.file_uploader('Fichier CSV', type=['csv'])

    if fichier is not None:
        try:
            lot = pd.read_csv(fichier, encoding='utf-8-sig')
            manquantes = [c for c in num_cols + cat_cols if c not in lot.columns]

            if manquantes:
                st.error(f'Colonnes manquantes : {manquantes}')
            else:
                lot['Probabilité'] = pipeline.predict_proba(lot[num_cols + cat_cols])[:, 1].round(3)
                lot['Alerte'] = lot['Probabilité'].apply(
                    lambda p: 'À RISQUE' if p >= SEUIL_RISQUE
                    else ('Modéré' if p >= SEUIL_MODERE else 'Fidèle'))

                n_risque = int((lot['Probabilité'] >= SEUIL_RISQUE).sum())
                c1, c2, c3 = st.columns(3)
                c1.metric('Clients scorés', len(lot))
                c2.metric('Clients à risque', n_risque)
                c3.metric('Probabilité moyenne', f"{lot['Probabilité'].mean():.0%}")

                st.dataframe(lot.sort_values('Probabilité', ascending=False),
                             width='stretch')

                st.download_button(
                    '⬇️ Télécharger le fichier scoré',
                    data=lot.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'),
                    file_name='clients_scores.csv', mime='text/csv')
        except Exception as e:
            st.error(f'Lecture impossible : {e}')


st.markdown('---')
st.caption('Formation Machine Learning & Data Science — TP Jour 3 · '
           'Modèle entraîné sur 500 clients, taux de résiliation '
           f"{meta.get('taux_resiliation', 0):.0%}.")
