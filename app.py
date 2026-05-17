import streamlit as st
from datetime import datetime
from streamlit_extras.metric_cards import style_metric_cards

from components.styles      import appliquer_styles, afficher_header
from components.data_loader import charger_tout
from components.predictions import generer_predictions, compter_pronos_haute_confiance
from components.match_card  import afficher_carte_match
from components.ia_explain  import afficher_onglet_ia_explique


# ============================================================
# CONFIGURATION DE LA PAGE (toujours en premier)
# ============================================================

st.set_page_config(
    page_title="Pronos Foot IA — Coupe du Monde 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Application du thème Data Sport Pro
appliquer_styles()


# ============================================================
# CHARGEMENT DES DONNÉES (cache automatique)
# ============================================================

# Une seule ligne pour tout charger grâce à notre fonction raccourci
modele, encoder, matchs_futurs, historique = charger_tout()

# Génération des prédictions pour les 72 matchs
df_predictions = generer_predictions(modele, encoder, matchs_futurs, historique)


# ============================================================
# CALCULS POUR LES STATS DE L'ACCUEIL
# ============================================================

# Compte à rebours jusqu'à la Coupe du Monde (11 juin 2026)
date_cdm = datetime(2026, 6, 11)
jours_restants = max(0, (date_cdm - datetime.now()).days)

# Nombre de pronos avec une probabilité supérieure à 50%
pronos_sur = compter_pronos_haute_confiance(df_predictions)


# ============================================================
# AFFICHAGE — Header + Stats
# ============================================================

# Header avec titre, sous-titre et ligne dégradée
afficher_header()

# 3 cartes de stats côte à côte
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Matchs analysés", value=len(matchs_futurs))

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


# ============================================================
# ONGLETS PRINCIPAUX
# ============================================================

tab1, tab2, tab3 = st.tabs(["📅 Les matchs", "🎯 Mes pronos", "🧠 L'IA explique"])

# ─── Onglet 1 : Liste des matchs ────────────────────────────
with tab1:
    st.subheader("Tous les matchs de la Coupe du Monde 2026")
    st.markdown(
        '<p style="color: #94A3B8;">Les pronostics de l\'IA pour les 72 matchs.</p>',
        unsafe_allow_html=True
    )

    # Pour chaque match, on appelle notre composant
    # C'est tout ! Une seule ligne au lieu de 60 de HTML
    for _, match in df_predictions.iterrows():
        afficher_carte_match(match)

# ─── Onglet 2 : Mes pronos (placeholder) ────────────────────
with tab2:
    st.subheader("Mes pronostics")
    st.markdown(
        '<p style="color: #94A3B8;">Saisis tes propres pronostics et compare-toi à l\'IA.</p>',
        unsafe_allow_html=True
    )
    st.info("🚧 Cet onglet sera disponible bientôt !")

# ─── Onglet 3 : L'IA explique ───────────────────────────────
with tab3:
    afficher_onglet_ia_explique(df_predictions)