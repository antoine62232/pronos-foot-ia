import pandas as pd
import streamlit as st
from components.data_loader import POINTS_FIFA, PAYS_HOTES_CDM_2026

# ====================================================================
# FONCTIONS DE CALCUL DE FORME (V2 - Adaptable 5 ou 10 matchs)
# ====================================================================

def get_forme_attaque(equipe, historique, nb_matchs=5):
    """
    Calcule la moyenne de buts MARQUÉS par une équipe sur ses X derniers matchs.
    """
    matchs_dom = historique[historique['home_team'] == equipe]['home_score']
    matchs_ext = historique[historique['away_team'] == equipe]['away_score']

    tous_les_buts = pd.concat([matchs_dom, matchs_ext])

    if len(tous_les_buts) == 0:
        return 0.0

    # On utilise 'nb_matchs' au lieu du '5' codé en dur
    return round(tous_les_buts.tail(nb_matchs).mean(), 2)


def get_forme_defense(equipe, historique, nb_matchs=5):
    """
    Calcule la moyenne de buts ENCAISSÉS par une équipe sur ses X derniers matchs.
    """
    buts_encaisses_dom = historique[historique['home_team'] == equipe]['away_score']
    buts_encaisses_ext = historique[historique['away_team'] == equipe]['home_score']

    tous_buts_encaisses = pd.concat([buts_encaisses_dom, buts_encaisses_ext])

    if len(tous_buts_encaisses) == 0:
        return 1.0

    # On utilise 'nb_matchs' au lieu du '5' codé en dur
    return round(tous_buts_encaisses.tail(nb_matchs).mean(), 2)


# ====================================================================
# GÉNÉRATION DES PRÉDICTIONS
# ====================================================================

@st.cache_data
def generer_predictions(_modele, _encoder, matchs_a_predire, historique):
    """
    Génère les prédictions pour tous les matchs à prédire en utilisant les 11 features V2.
    """
    resultats = []

    for _, match in matchs_a_predire.iterrows():

        equipe_dom = match['home_team']
        equipe_ext = match['away_team']

        # 1. Calcul des 11 features V2
        # Formes sur 5 matchs
        forme_att_dom = get_forme_attaque(equipe_dom, historique, 5)
        forme_att_ext = get_forme_attaque(equipe_ext, historique, 5)
        forme_def_dom = get_forme_defense(equipe_dom, historique, 5)
        forme_def_ext = get_forme_defense(equipe_ext, historique, 5)

        # Formes sur 10 matchs
        forme_att_dom_10 = get_forme_attaque(equipe_dom, historique, 10)
        forme_att_ext_10 = get_forme_attaque(equipe_ext, historique, 10)
        forme_def_dom_10 = get_forme_defense(equipe_dom, historique, 10)
        forme_def_ext_10 = get_forme_defense(equipe_ext, historique, 10)
        
        # FIFA et Neutre
        fifa_dom = POINTS_FIFA.get(equipe_dom, 1200)
        fifa_ext = POINTS_FIFA.get(equipe_ext, 1200)

        match_neutre = 1
        if equipe_dom in PAYS_HOTES_CDM_2026:
            match_neutre = 0

        # 2. Préparation des données pour le modèle (Même ordre que dans modele.py !)
        donnees_match = pd.DataFrame([[
            forme_att_dom, forme_att_ext,
            forme_def_dom, forme_def_ext,
            forme_att_dom_10, forme_att_ext_10,
            forme_def_dom_10, forme_def_ext_10,
            fifa_dom, fifa_ext,
            match_neutre
        ]], columns=[
            'forme_attaque_domicile', 'forme_attaque_exterieur',
            'forme_defense_domicile', 'forme_defense_exterieur',
            'forme_attaque_domicile_10', 'forme_attaque_exterieur_10',
            'forme_defense_domicile_10', 'forme_defense_exterieur_10',
            'points_fifa_domicile', 'points_fifa_exterieur',
            'match_neutre'
        ])

        # 3. Prédiction
        probas = _modele.predict_proba(donnees_match)[0]

        classes = _encoder.inverse_transform(_modele.classes_)
        probas_dict = dict(zip(classes, probas))
        pronostic = max(probas_dict, key=probas_dict.get)

        # 4. Sauvegarde
        resultats.append({
            'date'      : match['date'],
            'equipe_dom': equipe_dom,
            'equipe_ext': equipe_ext,
            'pronostic' : pronostic,
            'proba_1'   : probas_dict.get('1', 0) * 100,
            'proba_N'   : probas_dict.get('N', 0) * 100,
            'proba_2'   : probas_dict.get('2', 0) * 100,
        })

    return pd.DataFrame(resultats)


def compter_pronos_haute_confiance(df_predictions, seuil=50):
    proba_max = df_predictions[['proba_1', 'proba_N', 'proba_2']].max(axis=1)
    return int((proba_max >= seuil).sum())