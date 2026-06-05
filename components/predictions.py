import pandas as pd
import streamlit as st
from components.data_loader import PAYS_HOTES_CDM_2026

# ====================================================================
# FORME RECENTE (calculee comme a l'entrainement : matchs A DOMICILE
# pour l'equipe qui recoit, matchs A L'EXTERIEUR pour la visiteuse)
# ====================================================================

def forme_domicile(equipe, historique, nb_matchs):
    """(buts marques, buts encaisses) sur les nb derniers matchs A DOMICILE."""
    matchs = historique[historique["home_team"] == equipe].tail(nb_matchs)
    if len(matchs) == 0:
        return 0.0, 1.0
    return round(matchs["home_score"].mean(), 2), round(matchs["away_score"].mean(), 2)


def forme_exterieur(equipe, historique, nb_matchs):
    """(buts marques, buts encaisses) sur les nb derniers matchs A L'EXTERIEUR."""
    matchs = historique[historique["away_team"] == equipe].tail(nb_matchs)
    if len(matchs) == 0:
        return 0.0, 1.0
    return round(matchs["away_score"].mean(), 2), round(matchs["home_score"].mean(), 2)


# Anciennes fonctions conservees au cas ou un autre composant les importe pour
# AFFICHER une forme. Elles ne servent plus a la prediction.
def get_forme_attaque(equipe, historique, nb_matchs=5):
    tous = pd.concat([historique[historique["home_team"] == equipe]["home_score"],
                      historique[historique["away_team"] == equipe]["away_score"]])
    return round(tous.tail(nb_matchs).mean(), 2) if len(tous) else 0.0


def get_forme_defense(equipe, historique, nb_matchs=5):
    tous = pd.concat([historique[historique["home_team"] == equipe]["away_score"],
                      historique[historique["away_team"] == equipe]["home_score"]])
    return round(tous.tail(nb_matchs).mean(), 2) if len(tous) else 1.0


# ====================================================================
# GENERATION DES PREDICTIONS
# ====================================================================

@st.cache_data
def generer_predictions(_modele, _encoder, matchs_a_predire, historique):
    """Genere les predictions des matchs a venir (features : forme + Elo)."""
    # On trie pour que .tail() prenne bien les matchs LES PLUS RECENTS.
    historique = historique.sort_values("date")

    # Classement Elo courant, produit par modele.py (data/elo_actuel.csv).
    df_elo = pd.read_csv("data/elo_actuel.csv")
    elos = dict(zip(df_elo["equipe"], df_elo["elo"]))

    resultats = []
    for _, match in matchs_a_predire.iterrows():
        dom = match["home_team"]
        ext = match["away_team"]

        # Forme : domicile pour l'equipe qui recoit, exterieur pour la visiteuse
        att_dom_5, def_dom_5 = forme_domicile(dom, historique, 5)
        att_ext_5, def_ext_5 = forme_exterieur(ext, historique, 5)
        att_dom_10, def_dom_10 = forme_domicile(dom, historique, 10)
        att_ext_10, def_ext_10 = forme_exterieur(ext, historique, 10)

        # Match neutre (Coupe du Monde), sauf si un pays hote recoit
        match_neutre = 0 if dom in PAYS_HOTES_CDM_2026 else 1

        # Meme ORDRE de colonnes que dans modele.py
        donnees_match = pd.DataFrame([[
            att_dom_5, att_ext_5, def_dom_5, def_ext_5,
            att_dom_10, att_ext_10, def_dom_10, def_ext_10,
            elos.get(dom, 1500), elos.get(ext, 1500),
            match_neutre,
        ]], columns=[
            "forme_attaque_domicile", "forme_attaque_exterieur",
            "forme_defense_domicile", "forme_defense_exterieur",
            "forme_attaque_domicile_10", "forme_attaque_exterieur_10",
            "forme_defense_domicile_10", "forme_defense_exterieur_10",
            "elo_domicile", "elo_exterieur",
            "match_neutre",
        ])

        probas = _modele.predict_proba(donnees_match)[0]
        classes = _encoder.inverse_transform(_modele.classes_)
        probas_dict = dict(zip(classes, probas))
        pronostic = max(probas_dict, key=probas_dict.get)

        resultats.append({
            "date": match["date"],
            "equipe_dom": dom,
            "equipe_ext": ext,
            "group": match.get("group", ""),
            "pronostic": pronostic,
            "proba_1": probas_dict.get("1", 0) * 100,
            "proba_N": probas_dict.get("N", 0) * 100,
            "proba_2": probas_dict.get("2", 0) * 100,
        })

    return pd.DataFrame(resultats)


def compter_pronos_haute_confiance(df_predictions, seuil=50):
    proba_max = df_predictions[["proba_1", "proba_N", "proba_2"]].max(axis=1)
    return int((proba_max >= seuil).sum())