# ============================================================
# RÔLE    : Toute la logique de prédiction IA
# ============================================================

import pandas as pd
import streamlit as st

# On importe les constantes depuis data_loader.py
# (les majuscules indiquent que ce sont des références)
from components.data_loader import POINTS_FIFA, PAYS_HOTES_CDM_2026


# ============================================================
# FONCTIONS DE CALCUL DE FORME
# ============================================================

def get_forme_attaque(equipe, historique):
    """
    Calcule la moyenne de buts MARQUÉS par une équipe sur ses 5 derniers matchs.

    On regarde TOUS les matchs de l'équipe (à domicile ET à l'extérieur)
    et on prend la moyenne des buts marqués sur les 5 plus récents.

    Paramètres :
        equipe (str)         : Le nom de l'équipe (ex: 'France')
        historique (DataFrame): Le tableau des matchs historiques

    Retour :
        float : La moyenne de buts marqués (ex: 1.8)
                Retourne 0.0 si aucun match trouvé dans l'historique
    """
    # On récupère les buts marqués quand l'équipe joue à domicile
    matchs_dom = historique[historique['home_team'] == equipe]['home_score']

    # On récupère les buts marqués quand l'équipe joue à l'extérieur
    matchs_ext = historique[historique['away_team'] == equipe]['away_score']

    # On combine les deux séries de buts dans une seule liste
    tous_les_buts = pd.concat([matchs_dom, matchs_ext])

    # Si l'équipe n'a aucun match dans l'historique, on retourne 0
    if len(tous_les_buts) == 0:
        return 0.0

    # Sinon, on prend la moyenne des 5 derniers matchs
    return round(tous_les_buts.tail(5).mean(), 2)


def get_forme_defense(equipe, historique):
    """
    Calcule la moyenne de buts ENCAISSÉS par une équipe sur ses 5 derniers matchs.

    Logique inverse de get_forme_attaque : on regarde les buts marqués
    par l'adversaire face à cette équipe.

    Paramètres :
        equipe (str)         : Le nom de l'équipe
        historique (DataFrame): Le tableau des matchs historiques

    Retour :
        float : La moyenne de buts encaissés (plus c'est bas, meilleure est la défense)
    """
    # Buts encaissés à domicile = buts marqués par l'adversaire (away_score)
    buts_encaisses_dom = historique[historique['home_team'] == equipe]['away_score']

    # Buts encaissés à l'extérieur = buts marqués par l'équipe domicile (home_score)
    buts_encaisses_ext = historique[historique['away_team'] == equipe]['home_score']

    tous_buts_encaisses = pd.concat([buts_encaisses_dom, buts_encaisses_ext])

    if len(tous_buts_encaisses) == 0:
        return 1.0

    return round(tous_buts_encaisses.tail(5).mean(), 2)


# ============================================================
# GÉNÉRATION DES PRÉDICTIONS
# ============================================================

# @st.cache_data = on calcule UNE FOIS et on garde en mémoire
# Sans ça, on recalculerait les 72 prédictions à chaque clic utilisateur
@st.cache_data
def generer_predictions(_modele, _encoder, matchs_a_predire, historique):
    """
    Génère les prédictions pour tous les matchs à prédire.

    Pour chaque match :
        1. Calcule les 7 features (forme attaque, défense, FIFA, neutre)
        2. Envoie ces features au modèle XGBoost
        3. Récupère les probabilités pour chaque résultat (1, N, 2)
        4. Détermine le pronostic (le plus probable)

    Paramètres :
        _modele               : Le modèle XGBoost entraîné
                                (le underscore "_" devant dit à Streamlit :
                                 "ne hash pas cet argument, il est complexe")
        _encoder              : L'encoder qui traduit chiffres ↔ lettres
        matchs_a_predire (DF) : Les 72 matchs de la Coupe du Monde
        historique (DF)       : L'historique pour calculer les formes

    Retour :
        DataFrame : Un tableau avec une ligne par match prédit
                    Colonnes : date, equipe_dom, equipe_ext, pronostic,
                               proba_1, proba_N, proba_2
    """
    # On va construire une liste de dictionnaires (un par match)
    # Puis on la convertira en DataFrame à la fin
    resultats = []

    # Boucle sur chaque match à prédire
    for _, match in matchs_a_predire.iterrows():

        equipe_dom = match['home_team']
        equipe_ext = match['away_team']

        # ─── Calcul des 7 features ──────────────────────────────
        forme_att_dom = get_forme_attaque(equipe_dom, historique)
        forme_att_ext = get_forme_attaque(equipe_ext, historique)
        forme_def_dom = get_forme_defense(equipe_dom, historique)
        forme_def_ext = get_forme_defense(equipe_ext, historique)
        fifa_dom = POINTS_FIFA.get(equipe_dom, 1200)
        fifa_ext = POINTS_FIFA.get(equipe_ext, 1200)

        # Match neutre = 1 par défaut (CdM sur 3 pays)
        # Sauf si l'équipe à domicile est un pays hôte (USA/Canada/Mexique)
        match_neutre = 1
        if equipe_dom in PAYS_HOTES_CDM_2026:
            match_neutre = 0

        # ─── Préparation des données pour le modèle ─────────────
        # IMPORTANT : l'ordre des colonnes DOIT être le même que pendant l'entraînement
        donnees_match = pd.DataFrame([[
            forme_att_dom, forme_att_ext,
            forme_def_dom, forme_def_ext,
            fifa_dom, fifa_ext,
            match_neutre
        ]], columns=[
            'forme_attaque_domicile', 'forme_attaque_exterieur',
            'forme_defense_domicile', 'forme_defense_exterieur',
            'points_fifa_domicile', 'points_fifa_exterieur',
            'match_neutre'
        ])

        # ─── Prédiction ─────────────────────────────────────────
        # predict_proba() retourne les probabilités pour chaque classe
        probas = _modele.predict_proba(donnees_match)[0]

        # On retraduit les classes du modèle (0, 1, 2) en lettres ("1", "2", "N")
        classes = _encoder.inverse_transform(_modele.classes_)
        probas_dict = dict(zip(classes, probas))

        # Le pronostic = la classe avec la plus haute probabilité
        pronostic = max(probas_dict, key=probas_dict.get)

        # On stocke toutes les infos du match dans notre liste
        resultats.append({
            'date'      : match['date'],
            'equipe_dom': equipe_dom,
            'equipe_ext': equipe_ext,
            'pronostic' : pronostic,
            'proba_1'   : probas_dict.get('1', 0) * 100,
            'proba_N'   : probas_dict.get('N', 0) * 100,
            'proba_2'   : probas_dict.get('2', 0) * 100,
        })

    # Conversion finale en DataFrame pour faciliter les manipulations
    return pd.DataFrame(resultats)


def compter_pronos_haute_confiance(df_predictions, seuil=50):
    """
    Compte le nombre de pronostics avec une probabilité supérieure au seuil.

    Un prono "haute confiance" = la probabilité du résultat le plus probable
    dépasse le seuil (50% par défaut).

    Paramètres :
        df_predictions (DataFrame) : Les prédictions générées
        seuil (int)                : Le seuil en pourcentage (default: 50)

    Retour :
        int : Le nombre de pronos haute confiance
    """
    # On calcule la probabilité maximale pour chaque match
    proba_max = df_predictions[['proba_1', 'proba_N', 'proba_2']].max(axis=1)

    # On compte combien dépassent le seuil
    return int((proba_max >= seuil).sum())