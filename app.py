import streamlit as st
import pandas as pd
from datetime import datetime
from streamlit_extras.metric_cards import style_metric_cards

from components.styles      import appliquer_styles, afficher_header
from components.data_loader import charger_tout
from components.predictions import generer_predictions, compter_pronos_haute_confiance
from components.match_card  import afficher_carte_match
from components.user_pronos import afficher_onglet_mes_pronos
from components.ia_explain  import afficher_onglet_ia_explique
from components.phase_eliminatoire_ui import afficher_onglet_phase_eliminatoire
from components.intro import afficher_intro
from components.resultats import afficher_onglet_resultats

# CONFIGURATION DE LA PAGE (toujours en premier)

st.set_page_config(
    page_title="Pronos Foot IA — Coupe du Monde 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Application du thème Data Sport Pro
appliquer_styles()

afficher_intro()

# CHARGEMENT DES DONNÉES (cache automatique)

# Une seule ligne pour tout charger grâce à notre fonction raccourci
modele, encoder, matchs_futurs, historique = charger_tout()

# Génération des prédictions pour les 72 matchs
df_predictions = generer_predictions(modele, encoder, matchs_futurs, historique)

# Nombre total de matchs analyses : poules + elimination directe (le bracket simule)
try:
    nb_elimination = len(pd.read_csv("data/bracket_complet.csv"))
except Exception:
    nb_elimination = 0   # securite : si le fichier manque, on compte juste les poules

nb_matchs_total = len(matchs_futurs) + nb_elimination

# CALCULS POUR LES STATS DE L'ACCUEIL

# Compte à rebours jusqu'à la Coupe du Monde (11 juin 2026)
date_cdm = datetime(2026, 6, 11)
jours_restants = max(0, (date_cdm - datetime.now()).days)

# Nombre de pronos avec une probabilité supérieure à 50%
pronos_sur = compter_pronos_haute_confiance(df_predictions)

# AFFICHAGE — Header + Stats

# Header avec titre, sous-titre et ligne dégradée
afficher_header()

# 3 cartes de stats côte à côte
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Matchs analysés",
        value=nb_matchs_total,
        help="72 matchs de poules + les matchs à élimination directe simulés à partir des qualifiés prédits."
    )

with col2:
    st.metric(label="Pronos haute confiance", value=pronos_sur)

with col3:
    st.metric(label="Coup d'envoi", value=f"J - {jours_restants}")

# Style appliqué automatiquement à toutes les st.metric ci-dessus
style_metric_cards(
    background_color="#141B2D",
    border_left_color="#A78BFA",
    border_color="#1E293B",
    border_radius_px=10
)

st.markdown("<br>", unsafe_allow_html=True)

# ONGLETS PRINCIPAUX

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Phase de groupes", "Phase Éliminatoire", "Mes pronos", "L'IA explique", "Réalité VS IA"])

# Onglet 1 : Phase de groupes (matchs regroupes par poule)
# Onglet 1 : Phase de groupes (avec filtres)
with tab1:
    st.subheader("Phase de groupes")

    # --- BARRE DE FILTRES ---
    # 1er choix : le mode d'affichage
    mode = st.radio(
        "Afficher :",
        ["Tout", "Par groupe", "Par équipe", "Par date"],
        horizontal=True,
    )

    # Par defaut on affiche tous les matchs ; on reduit selon le mode choisi.
    matchs_a_afficher = df_predictions

    # 2e choix : un menu deroulant adapte au mode
    if mode == "Par groupe":
        groupe = st.selectbox("Choisis un groupe", sorted(df_predictions["group"].unique()))
        matchs_a_afficher = df_predictions[df_predictions["group"] == groupe]

    elif mode == "Par équipe":
        # Liste de toutes les equipes (a domicile ou a l'exterieur), triee
        equipes = sorted(set(df_predictions["equipe_dom"]) | set(df_predictions["equipe_ext"]))
        equipe = st.selectbox("Choisis une équipe", equipes)
        matchs_a_afficher = df_predictions[
            (df_predictions["equipe_dom"] == equipe) | (df_predictions["equipe_ext"] == equipe)
        ]

    elif mode == "Par date":
        date = st.selectbox("Choisis une date", sorted(df_predictions["date"].unique()))
        matchs_a_afficher = df_predictions[df_predictions["date"] == date]

    st.markdown("<br>", unsafe_allow_html=True)

    # --- AFFICHAGE ---
    if mode == "Tout":
        # On garde le regroupement par poule (avec les en-tetes "Groupe X")
        for lettre in sorted(matchs_a_afficher["group"].unique()):
            st.markdown(
                f'<h3 style="color: #F8FAFC; border-left: 4px solid #A78BFA;'
                f' padding-left: 12px; margin: 28px 0 12px 0;">Groupe {lettre}</h3>',
                unsafe_allow_html=True
            )
            for _, match in matchs_a_afficher[matchs_a_afficher["group"] == lettre].iterrows():
                afficher_carte_match(match)
    else:
        # Dans les autres modes, on affiche simplement la liste filtree
        if len(matchs_a_afficher) == 0:
            st.info("Aucun match à afficher pour ce filtre.")
        for _, match in matchs_a_afficher.iterrows():
            afficher_carte_match(match)
            
# Onglet 2 : Phase Éliminatoire
with tab2:
    afficher_onglet_phase_eliminatoire(modele, encoder)

# Onglet 3 : Mes pronos
with tab3:
    afficher_onglet_mes_pronos(df_predictions)

# Onglet 4 : L'IA explique
with tab4:
    afficher_onglet_ia_explique(df_predictions)

# Onglet 5 : Résultats & Score de l'IA
with tab5:
    afficher_onglet_resultats(df_predictions)