import streamlit as st
from datetime import date
from streamlit_extras.metric_cards import style_metric_cards

# Importations de tes composants
from components.styles      import appliquer_styles, afficher_header, afficher_description_onglet
from components.data_loader import charger_tout
from components.predictions import generer_predictions, compter_pronos_haute_confiance
from components.match_card  import afficher_carte_match
from components.user_pronos import afficher_onglet_mes_pronos
from components.ia_explain  import afficher_onglet_ia_explique
from components.phase_eliminatoire_ui import afficher_onglet_phase_eliminatoire
from components.intro import afficher_intro
from components.resultats import afficher_onglet_resultats
from components.dates import _date_fr
from components.compteur import afficher_compteur_visites
from components.prediction_live import afficher_encart_champion_live

# Nombre total de matchs de la CdM 2026 (72 en poules + 32 en phase finale).
# Format officiel FIFA fixe : on l'affiche comme repere pour le visiteur.
NB_MATCHS_TOURNOI = 104

# Configuration
st.set_page_config(
    page_title="Prediktora — Coupe du Monde 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

appliquer_styles()
afficher_intro()

@st.cache_data
def get_predictions_cached(_mode_ia, _enc, _matchs, _hist):
    return generer_predictions(_mode_ia, _enc, _matchs, _hist)

# Chargement + predictions. Au tout premier lancement le cache est vide, d'ou
# le spinner : il rend l'attente explicite plutot que de laisser un blanc.
with st.spinner("Préparation des prédictions..."):
    modele, encoder, matchs_futurs, historique = charger_tout()
    df_predictions = get_predictions_cached(modele, encoder, matchs_futurs, historique)

# Stats
date_cdm = date(2026, 6, 11)
jours_restants = max(0, (date_cdm - date.today()).days)
pronos_sur = compter_pronos_haute_confiance(df_predictions)

# Affichage
afficher_compteur_visites()
afficher_header()

col1, col2, col3 = st.columns(3)
with col1: st.metric("Matchs analysés", NB_MATCHS_TOURNOI)
with col2: st.metric("Pronos haute confiance", pronos_sur)
with col3: st.metric("Coup d'envoi", f"J - {jours_restants}")

style_metric_cards(background_color="#141B2D", border_left_color="#A78BFA", border_color="#1E293B", border_radius_px=10)

# Badge horaire discret, aligne a droite sous les cartes
st.markdown("""
    <div style="display: flex; justify-content: flex-end; margin-bottom: 30px;">
        <span style="font-size: 12px; color: #64748B; font-style: italic;
                     padding: 4px 10px; border-radius: 6px;
                     background: rgba(255,255,255,0.03);">
            🌎 Dates basées sur le fuseau horaire de l'Amérique du Nord (UTC-5)
        </span>
    </div>
""", unsafe_allow_html=True)

# Bloc de bienvenue : presente le site en deux phrases avant la navigation
st.markdown("""
<div style="background-color: #141B2D; border: 1px solid #1E293B;
            border-radius: 10px; padding: 16px 20px; margin-bottom: 30px;">
    <p style="color: #CBD5E1; font-size: 14px; line-height: 1.6; margin: 0;">
        <span style="color: #F8FAFC; font-weight: 600;">Bienvenue sur Prediktora.</span>
        Une IA analyse des milliers de matchs internationaux pour prédire la Coupe du Monde 2026,
        du premier match de poule jusqu'au sacre final. Aucun pari, aucune intuition : juste des données.
        À toi de voir si tu es d'accord avec la machine.
    </p>
</div>
""", unsafe_allow_html=True)

# NAVIGATION
options_onglets = ["Phase de groupes", "Phase Éliminatoire", "Mes pronos VS IA", "Réalité VS IA", "L'IA explique"]
onglet_actif = st.radio("Navigation", options_onglets, horizontal=True, label_visibility="collapsed", key="nav")
st.markdown("<br>", unsafe_allow_html=True)

if onglet_actif == "Phase de groupes":
    st.subheader("Phase de groupes")
    afficher_description_onglet("Les 72 matchs de poules, avec le pronostic de l'IA et ses probabilités pour chaque rencontre.")
    # Label custom pour remplacer celui, masque, du radio natif
    st.markdown("<p style='color: #94A3B8; margin-bottom: 8px; font-size: 14px;'>Afficher :</p>", unsafe_allow_html=True)
    mode = st.radio("Afficher :", ["Tout", "Par groupe", "Par équipe", "Par date"], horizontal=True, label_visibility="collapsed")

    matchs_a_afficher = df_predictions

    if mode == "Par groupe":
        groupe = st.selectbox("Choisis un groupe", sorted(df_predictions["group"].unique()))
        matchs_a_afficher = df_predictions[df_predictions["group"] == groupe]
    elif mode == "Par équipe":
        equipes = sorted(set(df_predictions["equipe_dom"]) | set(df_predictions["equipe_ext"]))
        equipe = st.selectbox("Choisis une équipe", equipes)
        matchs_a_afficher = df_predictions[(df_predictions["equipe_dom"] == equipe) | (df_predictions["equipe_ext"] == equipe)]
    elif mode == "Par date":
        date_selectionnee = st.selectbox("Choisis une date", sorted(df_predictions["date"].unique()), format_func=_date_fr)
        matchs_a_afficher = df_predictions[df_predictions["date"] == date_selectionnee]

    st.markdown("<br>", unsafe_allow_html=True)

    if mode == "Tout":
        for lettre in sorted(matchs_a_afficher["group"].unique()):
            st.markdown(f'<h3 style="color: #F8FAFC; border-left: 4px solid #A78BFA; padding-left: 12px; margin: 28px 0 12px 0;">Groupe {lettre}</h3>', unsafe_allow_html=True)
            for _, match in matchs_a_afficher[matchs_a_afficher["group"] == lettre].iterrows():
                afficher_carte_match(match)
    else:
        if len(matchs_a_afficher) == 0:
            st.info("Aucun match à afficher.")
        for _, match in matchs_a_afficher.iterrows():
            afficher_carte_match(match)

elif onglet_actif == "Phase Éliminatoire":
    afficher_encart_champion_live()
    afficher_onglet_phase_eliminatoire(modele, encoder)
elif onglet_actif == "Mes pronos VS IA":
    afficher_onglet_mes_pronos(df_predictions)
elif onglet_actif == "Réalité VS IA":
    afficher_onglet_resultats(df_predictions)
elif onglet_actif == "L'IA explique":
    afficher_onglet_ia_explique(df_predictions)