import streamlit as st
import pandas as pd
from joblib import load

# CONFIGURATION DE LA PAGE

st.set_page_config(
    page_title="Pronos Foot IA — Coupe du Monde 2026",
    page_icon="⚽",
    layout="wide"
)

# EN-TÊTE

st.title("⚽ Pronos Foot IA — Coupe du Monde 2026")
st.markdown("*Pronostics générés par Intelligence Artificielle (XGBoost)*")
st.divider()

# CHARGEMENT DU MODÈLE, DE L'ENCODER ET DES DONNÉES

@st.cache_resource
def charger_modele():
    # On recharge le modèle XGBoost sauvegardé
    return load("models/modele_football.pkl")

@st.cache_resource
def charger_encoder():
    # L'encoder traduit les chiffres (0/1/2) en lettres (1/2/N)
    # XGBoost prédit en chiffres, on veut afficher en lettres
    return load("models/label_encoder.pkl")

@st.cache_data
def charger_matchs():
    return pd.read_csv("data/matchs_a_predire.csv")

@st.cache_data
def charger_historique():
    return pd.read_csv("data/matchs_entrainement.csv")

modele = charger_modele()
encoder = charger_encoder()
matchs_futurs = charger_matchs()
historique = charger_historique()

# POINTS FIFA OFFICIELS

points_fifa = {
    'France': 1877.32, 'Spain': 1876.40, 'Argentina': 1874.81,
    'England': 1825.97, 'Portugal': 1763.83, 'Brazil': 1761.16,
    'Netherlands': 1757.87, 'Morocco': 1755.87, 'Belgium': 1734.71,
    'Germany': 1730.37, 'Croatia': 1717.07, 'Italy': 1700.37,
    'Colombia': 1693.09, 'Senegal': 1688.99, 'Mexico': 1681.03,
    'United States': 1673.13, 'Uruguay': 1673.07, 'Japan': 1660.43,
    'Switzerland': 1649.40, 'Denmark': 1620.81, 'Iran': 1615.30,
    'Turkey': 1599.04, 'Ecuador': 1594.78, 'Austria': 1593.45,
    'South Korea': 1588.66, 'Nigeria': 1585.09, 'Australia': 1580.67,
    'Algeria': 1564.26, 'Egypt': 1563.24, 'Canada': 1556.48,
    'Norway': 1550.94, 'Ukraine': 1546.88, 'Panama': 1540.64,
    'Ivory Coast': 1532.98, 'Poland': 1528.00, 'Sweden': 1514.77,
    'Serbia': 1508.65, 'Paraguay': 1503.50, 'Czech Republic': 1501.38,
    'Hungary': 1500.58, 'Scotland': 1498.35, 'Tunisia': 1483.05,
    'Cameroon': 1481.24, 'DR Congo': 1478.35, 'Greece': 1475.82,
    'Slovakia': 1473.94, 'Venezuela': 1468.05, 'Uzbekistan': 1465.34,
    'Costa Rica': 1459.90, 'Mali': 1459.13, 'Peru': 1455.87,
    'Chile': 1455.28, 'Qatar': 1454.96, 'Romania': 1451.16,
    'Iraq': 1447.14, 'Slovenia': 1446.44, 'South Africa': 1429.73,
    'Saudi Arabia': 1421.43, 'Burkina Faso': 1412.49, 'Jordan': 1391.45,
    'Albania': 1388.06, 'Bosnia and Herzegovina': 1385.84,
    'Honduras': 1380.27, 'Cape Verde': 1366.13, 'Jamaica': 1358.00,
    'Georgia': 1350.18, 'Finland': 1346.41, 'Ghana': 1346.31,
    'Iceland': 1345.07, 'Bolivia': 1329.42, 'Kosovo': 1318.83,
    'Guinea': 1300.01, 'Montenegro': 1295.52, 'Curaçao': 1294.65,
    'Haiti': 1291.71, 'New Zealand': 1281.57, 'New Caledonia': 1036.95,
}

# FONCTIONS DE CALCUL DE FORME

def get_forme_attaque(equipe, historique):
    # Moyenne de buts marqués sur les 5 derniers matchs (dom. + ext.)
    matchs_dom = historique[historique['home_team'] == equipe]['home_score']
    matchs_ext = historique[historique['away_team'] == equipe]['away_score']
    tous_les_buts = pd.concat([matchs_dom, matchs_ext])
    if len(tous_les_buts) == 0:
        return 0.0
    return round(tous_les_buts.tail(5).mean(), 2)

def get_forme_defense(equipe, historique):
    # Moyenne de buts encaissés sur les 5 derniers matchs
    buts_encaisses_dom = historique[historique['home_team'] == equipe]['away_score']
    buts_encaisses_ext = historique[historique['away_team'] == equipe]['home_score']
    tous_buts_encaisses = pd.concat([buts_encaisses_dom, buts_encaisses_ext])
    if len(tous_buts_encaisses) == 0:
        return 1.0
    return round(tous_buts_encaisses.tail(5).mean(), 2)

# GÉNÉRATION DES PRÉDICTIONS

resultats = []

# Pays hôtes de la Coupe du Monde 2026 (pas de terrain neutre pour eux à domicile)
pays_hotes = ['United States', 'Canada', 'Mexico']

for _, match in matchs_futurs.iterrows():

    equipe_dom = match['home_team']
    equipe_ext = match['away_team']

    # Calcul des 6 features pour ce match
    forme_att_dom  = get_forme_attaque(equipe_dom, historique)
    forme_att_ext  = get_forme_attaque(equipe_ext, historique)
    forme_def_dom  = get_forme_defense(equipe_dom, historique)
    forme_def_ext  = get_forme_defense(equipe_ext, historique)
    fifa_dom       = points_fifa.get(equipe_dom, 1200)
    fifa_ext       = points_fifa.get(equipe_ext, 1200)

    # Pour les matchs de la Coupe du Monde 2026, on force match_neutre = 1
    # car tous les matchs se jouent en USA/Canada/Mexique sur terrain neutre
    match_neutre = 1

    # Exception : si l'équipe à domicile EST le pays hôte, ce n'est PAS neutre
    if equipe_dom in pays_hotes:
        match_neutre = 0

    # Préparation des données dans l'ordre attendu par le modèle
    donnees_match = pd.DataFrame([[
        forme_att_dom, forme_att_ext,
        forme_def_dom, forme_def_ext,
        fifa_dom, fifa_ext,
        match_neutre
    ]], columns=[
        'forme_attaque_domicile', 'forme_attaque_exterieur',
        'forme_defense_domicile', 'forme_defense_exterieur',
        'points_fifa_domicile',   'points_fifa_exterieur',
        'match_neutre'
    ])

    # XGBoost n'a pas besoin de normalisation
    probas = modele.predict_proba(donnees_match)[0]

    # On retraduit les chiffres en lettres grâce à l'encoder
    classes = encoder.inverse_transform(modele.classes_)
    probas_dict = dict(zip(classes, probas))

    # Le pronostic = celui avec la probabilité la plus haute
    pronostic = max(probas_dict, key=probas_dict.get)

    resultats.append({
        'Date'           : match['date'],
        'Domicile'       : equipe_dom,
        'Extérieur'      : equipe_ext,
        'Pronostic'      : pronostic,
        'Proba 1 (Dom.)' : f"{probas_dict.get('1', 0)*100:.0f}%",
        'Proba N (Nul)'  : f"{probas_dict.get('N', 0)*100:.0f}%",
        'Proba 2 (Ext.)' : f"{probas_dict.get('2', 0)*100:.0f}%",
    })

df_resultats = pd.DataFrame(resultats)

# AFFICHAGE

st.subheader("📅 Pronostics — 72 matchs de la Coupe du Monde 2026")
st.dataframe(df_resultats, use_container_width=True)

st.divider()
st.caption("✅ 72 matchs analysés · Modèle : XGBoost · 6 features · Données depuis 2000 · Points FIFA mai 2025")