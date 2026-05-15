# Rôle : Interface web de l'application de pronostics
import streamlit as st
import pandas as pd
from joblib import load

# Configuration de la page
st.set_page_config(
    page_title="Pronos Foot IA - Coupe du Monde 2026",
    page_icon="⚽",
    layout="wide"
)

# En-tête de l'application
st.title("Pronos Foot IA - Coupe du Monde 2026")
st.markdown("*Pronostics générés par Intelligence Artificielle*")

st.divider()

# Chargement du modèle
@st.cache_resource
def charger_modele():
    return load("models/modele_football.pkl")

@st.cache_data
def charger_matchs():
    return pd.read_csv("data/matchs_a_predire.csv")

@st.cache_data
def charger_historique():
    return pd.read_csv("data/matchs_entrainement.csv")

modele = charger_modele()
matchs_futurs = charger_matchs()
historique = charger_historique()

# Calcul de la forme récente des équipes pour les matchs à prédire
def get_forme_attaque(equipe, historique):
    # On filtre les matchs où l'équipe a joué à domicile ou à l'extérieur
    matchs_dom = historique[historique['home_team'] == equipe]['home_score']
    matchs_ext = historique[historique['away_team'] == equipe]['away_score']
    tous_les_buts = pd.concat([matchs_dom, matchs_ext])

    # Si l'équipe n'a jamais joué, on retourne une forme de 0
    if len(tous_les_buts) == 0:
        return 0.0
    # Sinon, on calcule la moyenne des 5 derniers buts marqués
    return round(tous_les_buts.tail(5).mean(), 2)

# Génération des prédictions

# On crée une liste pour stocker les résultats
resultats = []

# On parcourt chaque match à prédire un par un
for _, match in matchs_futurs.iterrows():

    equipe_dom = match['home_team']
    equipe_ext = match['away_team']

    forme_dom = get_forme_attaque(equipe_dom, historique)
    forme_ext = get_forme_attaque(equipe_ext, historique)

    # On prépare les données dans un tableau pour les passer au modèle
    donnees_match = pd.DataFrame([[forme_dom, forme_ext]], 
                                 columns=['forme_attaque_domicile', 'forme_attaque_exterieur'])
    # Le modèle prédit les PROBABILITÉS pour chaque résultat (pas juste le gagnant)
    probas = modele.predict_proba(donnees_match)[0]

    # On récupère l'ordre des classes (1, 2, N) tel que le modèle les a apprises
    classes = modele.classes_

    # On crée un dictionnaire associant chaque résultat à sa probabilité
    probas_dict = dict(zip(classes, probas))

    # On détermine le pronostic = le résultat avec la plus grande probabilité
    pronostic = max(probas_dict, key=probas_dict.get)

    # On ajoute toutes ces infos à notre liste de résultats
    resultats.append({
        'Date': match['date'],
        'Domicile': equipe_dom,
        'Extérieur': equipe_ext,
        'Pronostic': pronostic,
        'Proba 1 (Dom.)' : f"{probas_dict.get('1', 0) * 100:.0f}%",
        'Proba N (Nul)' :  f"{probas_dict.get('N', 0) * 100:.0f}%",
        'Proba 2 (Ext.)' : f"{probas_dict.get('2', 0) * 100:.0f}%" 
    })
# On transforme notre liste en tableau pandas pour l'afficher proprement
df_resultats = pd.DataFrame(resultats)

# Affichage dans l'application
st.subheader("Pronostics - 72 matchs de la Coupe du Monde 2026")

st.dataframe(df_resultats, use_container_width=True)

st.divider()
st.caption(f" {len(df_resultats)} matchs analysés · Modèle : Régression Logistique · Données : depuis l'an 2000")