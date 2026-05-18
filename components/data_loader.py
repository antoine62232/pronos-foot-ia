import streamlit as st
import pandas as pd
from joblib import load

# CONSTANTES — Dictionnaires de référence

# Tout le projet peut les importer depuis ce fichier

# Points FIFA officiels (source : FIFA.com)
POINTS_FIFA = {
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

# Codes ISO des pays (pour les drapeaux via flagcdn.com)
CODES_ISO = {
    'France': 'fr', 'Spain': 'es', 'Argentina': 'ar', 'England': 'gb-eng',
    'Portugal': 'pt', 'Brazil': 'br', 'Netherlands': 'nl', 'Morocco': 'ma',
    'Belgium': 'be', 'Germany': 'de', 'Croatia': 'hr', 'Italy': 'it',
    'Colombia': 'co', 'Senegal': 'sn', 'Mexico': 'mx', 'United States': 'us',
    'Uruguay': 'uy', 'Japan': 'jp', 'Switzerland': 'ch', 'Denmark': 'dk',
    'Iran': 'ir', 'Turkey': 'tr', 'Ecuador': 'ec', 'Austria': 'at',
    'South Korea': 'kr', 'Nigeria': 'ng', 'Australia': 'au', 'Algeria': 'dz',
    'Egypt': 'eg', 'Canada': 'ca', 'Norway': 'no', 'Ukraine': 'ua',
    'Panama': 'pa', 'Ivory Coast': 'ci', 'Poland': 'pl', 'Sweden': 'se',
    'Serbia': 'rs', 'Paraguay': 'py', 'Czech Republic': 'cz', 'Hungary': 'hu',
    'Scotland': 'gb-sct', 'Tunisia': 'tn', 'Cameroon': 'cm', 'DR Congo': 'cd',
    'Greece': 'gr', 'Slovakia': 'sk', 'Venezuela': 've', 'Uzbekistan': 'uz',
    'Costa Rica': 'cr', 'Mali': 'ml', 'Peru': 'pe', 'Chile': 'cl',
    'Qatar': 'qa', 'Romania': 'ro', 'Iraq': 'iq', 'Slovenia': 'si',
    'South Africa': 'za', 'Saudi Arabia': 'sa', 'Burkina Faso': 'bf',
    'Jordan': 'jo', 'Albania': 'al', 'Bosnia and Herzegovina': 'ba',
    'Honduras': 'hn', 'Cape Verde': 'cv', 'Jamaica': 'jm', 'Georgia': 'ge',
    'Finland': 'fi', 'Ghana': 'gh', 'Iceland': 'is', 'Bolivia': 'bo',
    'Kosovo': 'xk', 'Guinea': 'gn', 'Montenegro': 'me', 'Curaçao': 'cw',
    'Haiti': 'ht', 'New Zealand': 'nz', 'New Caledonia': 'nc',
}

# Pays qui hébergent la Coupe du Monde 2026
# Ils ont un avantage du terrain quand ils jouent à domicile
PAYS_HOTES_CDM_2026 = ['United States', 'Canada', 'Mexico']

# FONCTIONS DE CHARGEMENT

# @st.cache_resource = Streamlit ne charge le modèle QU'UNE FOIS
# Même si l'utilisateur clique partout, le modèle reste en mémoire
@st.cache_resource
def charger_modele():
    """Charge le modèle XGBoost depuis le fichier .pkl."""
    return load("models/modele_football.pkl")


@st.cache_resource
def charger_encoder():
    """Charge l'encoder qui traduit 0/1/2 en 1/N/2."""
    return load("models/label_encoder.pkl")


# @st.cache_data = pareil mais pour des données (DataFrames CSV)
@st.cache_data
def charger_matchs_a_predire():
    """Charge les 72 matchs de la Coupe du Monde 2026."""
    return pd.read_csv("data/matchs_a_predire.csv")


@st.cache_data
def charger_historique():
    """Charge l'historique des 25 196 matchs depuis 2000."""
    return pd.read_csv("data/matchs_entrainement.csv")


def charger_tout():
    """
    Fonction "raccourci" qui charge TOUT en une seule ligne.

    Au lieu d'écrire 4 lignes dans app.py, on en écrit une seule :
        modele, encoder, matchs, historique = charger_tout()

    Retour :
        Un tuple avec 4 éléments : (modele, encoder, matchs, historique)
    """
    return (
        charger_modele(),
        charger_encoder(),
        charger_matchs_a_predire(),
        charger_historique()
    )

# FONCTIONS UTILITAIRES SUR LES ÉQUIPES

def get_url_drapeau(equipe):
    """
    Retourne l'URL de l'image du drapeau d'une équipe.

    Paramètre :
        equipe (str) : Le nom de l'équipe (ex: 'France')

    Retour :
        str : L'URL du drapeau via flagcdn.com (résolution 80px)
    """
    code = CODES_ISO.get(equipe, 'un')  # 'un' = drapeau ONU par défaut
    return f"https://flagcdn.com/w80/{code}.png"


def get_rang_fifa(equipe):
    """
    Retourne le rang FIFA d'une équipe (1 = meilleure mondiale).

    Paramètre :
        equipe (str) : Le nom de l'équipe

    Retour :
        int ou None : Le rang FIFA (1, 2, 3...) ou None si non classée
    """
    if equipe not in POINTS_FIFA:
        return None

    # On trie les points du plus haut au plus bas
    # Le rang d'une équipe = sa position dans cette liste triée + 1
    points_tries = sorted(POINTS_FIFA.values(), reverse=True)
    return points_tries.index(POINTS_FIFA[equipe]) + 1